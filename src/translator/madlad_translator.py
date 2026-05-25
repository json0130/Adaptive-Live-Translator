"""MADLAD-400-3B streaming translator (no context injection).

Uses google/madlad400-3b-mt for apples-to-apples comparison with NLLB-600M.
Format: prepend language tag (e.g., <2ko>, <2en>) to source text, then pass to model.
"""
from __future__ import annotations

import torch
from typing import AsyncIterator

from loguru import logger

from .base import TranslationChunk, Translator


class MadladTranslator(Translator):
    """MADLAD-400-3B-MT streaming translator."""

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        self.model_name = cfg["translator"]["model"]
        self.dtype = cfg["translator"].get("dtype", "float16")
        self.max_new_tokens = cfg["translator"].get("max_new_tokens", 256)
        self._model = None
        self._tokenizer = None

        # MADLAD format: <2{TARGET_LANG}> source_text
        # The tag specifies the TARGET language (where the model should translate TO).
        self.lang_to_tag = {
            "en": "<2en>",
            "ko": "<2ko>",
        }

    def _lazy_load(self) -> None:
        if self._model is not None:
            return
        logger.info(f"Loading translator: {self.model_name} (dtype={self.dtype})")

        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # On 8GB VRAM (RTX 5060), MADLAD-3B fp16 weights (~5.8 GB) leave
        # almost no room for activations/KV. We quantize with bitsandbytes
        # int8 to drop weights to ~3 GB and free 3+ GB for runtime.
        if self.dtype == "int8":
            from transformers import BitsAndBytesConfig
            quant = BitsAndBytesConfig(load_in_8bit=True)
            self._model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_name,
                quantization_config=quant,
                device_map={"": 0},
            )
            logger.info(f"Loaded {self.model_name} on cuda with bnb-int8")
        elif self.dtype == "int4":
            from transformers import BitsAndBytesConfig
            quant = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )
            self._model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_name,
                quantization_config=quant,
                device_map={"": 0},
            )
            logger.info(f"Loaded {self.model_name} on cuda with bnb-nf4")
        else:
            dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16}
            torch_dtype = dtype_map.get(self.dtype, torch.float16)
            # Load to CPU first then move to handle T5 weight-tying properly.
            self._model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_name,
                dtype=torch_dtype,
                low_cpu_mem_usage=True,
            )
            self._model = self._model.to("cuda")
            logger.info(f"Loaded {self.model_name} on cuda dtype={self.dtype}")

    def reset(self) -> None:
        """No KV cache to reset for sentence-level translation."""
        pass

    async def translate_stream(
        self,
        src_chunks: AsyncIterator[str],
        *,
        src_lang: str,
        tgt_lang: str,
        system_prompt: str,
    ) -> AsyncIterator[TranslationChunk]:
        """Translate source chunks. MADLAD has no system prompt (T5 encoder-decoder)."""
        self._lazy_load()

        # Collect all source text (eval harness sends one full text per iterate)
        src_text = ""
        async for chunk in src_chunks:
            src_text += chunk

        if not src_text.strip():
            yield TranslationChunk(text="", delta="", is_final=True)
            return

        # MADLAD tag = TARGET language. e.g. <2ko> for translating into Korean.
        tgt_tag = self.lang_to_tag.get(tgt_lang, "<2en>")
        model_input = f"{tgt_tag} {src_text}"

        try:
            # Clear CUDA cache before generation
            torch.cuda.empty_cache()

            # Tokenize and generate
            inputs = self._tokenizer(
                model_input,
                return_tensors="pt",
                max_length=512,
                truncation=True,
            ).to(self._model.device)

            # Greedy decode: num_beams=1, no sampling
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                num_beams=1,
                do_sample=False,
                temperature=1.0,  # ignored with num_beams=1
                top_p=1.0,  # ignored with num_beams=1
            )

            # Decode the output (skip input tokens)
            translated = self._tokenizer.decode(
                outputs[0],
                skip_special_tokens=True,
            )

            # Yield final translation chunk
            yield TranslationChunk(text=translated, delta=translated, is_final=True)

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.error(f"CUDA OOM: {e}")
                raise
            else:
                logger.error(f"Generation failed: {e}")
                raise
