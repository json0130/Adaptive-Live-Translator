"""NLLB-200-distilled-600M translator for baseline evaluation.

No context injection — just sentence-level NMT with language tags.
"""
from __future__ import annotations

import torch
from typing import AsyncIterator
from loguru import logger

from .base import TranslationChunk, Translator


class NllbTranslator(Translator):
    """NLLB-200-distilled-600M streaming translator."""

    # FLORES language codes for NLLB
    LANG_MAP = {
        "en": "eng_Latn",
        "ko": "kor_Hang",
    }

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        self.model_name = cfg["translator"]["model"]
        self.dtype_str = cfg["translator"].get("dtype", "float16")
        self.max_new_tokens = cfg["translator"].get("max_new_tokens", 256)
        self._model = None
        self._tokenizer = None

    def _lazy_load(self) -> None:
        if self._model is not None:
            return
        logger.info(f"Loading translator: {self.model_name}")
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        # Parse dtype
        if self.dtype_str == "float16":
            dtype = torch.float16
        elif self.dtype_str == "bfloat16":
            dtype = torch.bfloat16
        else:
            dtype = torch.float32

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_name,
            torch_dtype=dtype,
            device_map="cuda",
        )
        logger.info(f"Model loaded on cuda with dtype={self.dtype_str}")

    def reset(self) -> None:
        """No KV cache to reset for NLLB (sentence-level)."""
        pass

    async def translate_stream(
        self,
        src_chunks: AsyncIterator[str],
        *,
        src_lang: str,
        tgt_lang: str,
        system_prompt: str,
    ) -> AsyncIterator[TranslationChunk]:
        self._lazy_load()

        # Map language codes
        src_flores = self.LANG_MAP.get(src_lang, src_lang)
        tgt_flores = self.LANG_MAP.get(tgt_lang, tgt_lang)

        # Collect all source text (NLLB is sentence-level, not streaming)
        src_text = ""
        async for chunk in src_chunks:
            src_text += chunk

        # Set source language and encode
        self._tokenizer.src_lang = src_flores
        inputs = self._tokenizer(src_text, return_tensors="pt").to("cuda")

        # Generate with forced target language
        forced_bos_token_id = self._tokenizer.convert_tokens_to_ids(tgt_flores)
        with torch.no_grad():
            generated_ids = self._model.generate(
                **inputs,
                forced_bos_token_id=forced_bos_token_id,
                max_new_tokens=self.max_new_tokens,
                num_beams=1,
                do_sample=False,
            )

        # Decode
        translation = self._tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0]

        # Emit single final chunk
        yield TranslationChunk(text=translation, delta=translation, is_final=True)
