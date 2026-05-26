"""Qwen2.5-7B-Instruct translator, 4-bit (bnb-nf4) for 8GB VRAM.

Notes on quantization choice:
  - AWQ load via transformers now requires `gptqmodel`, whose build
    pulls `pypcre>=0.2.14` which has no PyPI distribution. So AWQ via
    transformers is not installable in this environment. autoawq alone
    is not enough because transformers expects gptqmodel for the
    quantization_config dispatcher.
  - bitsandbytes nf4 on the bf16 base model gives equivalent VRAM
    (~4.2 GB weights) and known-stable kernels on Blackwell sm_120.
  - We therefore load Qwen/Qwen2.5-7B-Instruct (bf16) with
    BitsAndBytesConfig(load_in_4bit=True, nf4, compute_dtype=fp16).

Context injection: system prompt contains glossary entries triggered
by case-insensitive substring match, plus top-k=3 BM25 TM neighbours,
plus the assistant-priming pattern from docs/prompt_template.md.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import AsyncIterator

import torch
from loguru import logger

from .base import TranslationChunk, Translator


_BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"


class QwenTranslator(Translator):
    """Qwen2.5-7B-Instruct in bnb-nf4 with prompt-based context injection."""

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        # We ignore cfg["translator"]["model"] for the AWQ id since AWQ
        # loading is unavailable; we always load the base instruct model
        # and quantize it at load time.
        self.requested_model = cfg["translator"]["model"]
        self.dtype = cfg["translator"].get("dtype", "int4")
        self.max_new_tokens = cfg["translator"].get("max_new_tokens", 256)
        self.context_injection = cfg["translator"].get("context_injection", True)
        self.glossary_path = cfg["translator"].get(
            "glossary_path", "data/glossaries/ml-conference-en-ko.json"
        )
        self.tm_path = cfg["translator"].get(
            "tm_path", "data/translation_memory/en-ko.jsonl"
        )
        self.tm_top_k = cfg["translator"].get("tm_top_k", 3)
        self._model = None
        self._tokenizer = None
        self._glossary: list[dict] = []
        self._tm_pairs: list[dict] = []
        self._bm25 = None

        if self.context_injection:
            self._load_glossary_and_tm()

    # ---------------------------------------------------------- context

    def _load_glossary_and_tm(self) -> None:
        gp = Path(self.glossary_path)
        if gp.exists():
            data = json.loads(gp.read_text(encoding="utf-8"))
            for e in data.get("entries", []):
                self._glossary.append({"src": e["src"], "tgt": e["tgt"]})
            logger.info(f"[Qwen ctx] glossary: {len(self._glossary)} entries")

        tp = Path(self.tm_path)
        if tp.exists():
            for line in tp.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                self._tm_pairs.append({"src": row["src"], "tgt": row["tgt"]})
            logger.info(f"[Qwen ctx] TM: {len(self._tm_pairs)} pairs")
            try:
                from rank_bm25 import BM25Okapi
                tokenised = [p["src"].lower().split() for p in self._tm_pairs]
                self._bm25 = BM25Okapi(tokenised)
            except ImportError:
                logger.warning("rank_bm25 unavailable; TM retrieval disabled.")

    def _hits_for(self, src: str) -> list[dict]:
        lo = src.lower()
        return [e for e in self._glossary if e["src"].lower() in lo]

    def _retrieve_tm(self, src: str) -> list[dict]:
        if self._bm25 is None or not self._tm_pairs:
            return []
        scores = self._bm25.get_scores(src.lower().split())
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return [self._tm_pairs[i] for i, s in ranked[: self.tm_top_k] if s > 0.0]

    def _build_system_prompt(self, src: str, src_lang: str, tgt_lang: str) -> str:
        L = {"en": "English", "ko": "Korean"}
        src_full, tgt_full = L.get(src_lang, src_lang), L.get(tgt_lang, tgt_lang)
        lines = [
            f"You are a simultaneous interpreter translating {src_full} to {tgt_full}.",
            "Translate accurately and naturally. Emit only the translation — no explanations, no preamble.",
            "Register: formal.",
        ]
        if self.context_injection and src_lang == "en":
            gloss = self._hits_for(src)
            if gloss:
                lines.append("")
                lines.append("Glossary (MUST respect these translations):")
                for g in gloss:
                    lines.append(f"  {g['src']} -> {g['tgt']}")
            tm = self._retrieve_tm(src)
            if tm:
                lines.append("")
                lines.append("Translation examples (for style reference):")
                for p in tm:
                    lines.append(f"  SRC: {p['src']}")
                    lines.append(f"  TGT: {p['tgt']}")
        return "\n".join(lines)

    # ------------------------------------------------------------ model

    def _lazy_load(self) -> None:
        if self._model is not None:
            return

        logger.info(f"Loading translator (bnb-nf4): {_BASE_MODEL}")
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        self._tokenizer = AutoTokenizer.from_pretrained(_BASE_MODEL)
        self._model = AutoModelForCausalLM.from_pretrained(
            _BASE_MODEL,
            quantization_config=bnb,
            device_map={"": 0},
            low_cpu_mem_usage=True,
        )
        logger.info(
            "Qwen2.5-7B-Instruct loaded (nf4) "
            f"| context_injection={self.context_injection}"
        )

    def reset(self) -> None:
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

        src_text = ""
        async for chunk in src_chunks:
            src_text += chunk

        # Build system prompt ourselves so we control glossary/TM logic
        # consistently across en->ko and ko->en.
        sys_prompt = self._build_system_prompt(src_text, src_lang, tgt_lang)
        user_turn = f"Translate:\n{src_text}"

        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_turn},
        ]
        text = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._tokenizer(text, return_tensors="pt").to("cuda")
        input_length = inputs["input_ids"].shape[1]

        torch.cuda.empty_cache()
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                temperature=1.0,  # ignored with do_sample=False
                eos_token_id=self._tokenizer.eos_token_id,
                pad_token_id=self._tokenizer.eos_token_id,
            )

        generated_tokens = outputs[0][input_length:]
        translation = self._tokenizer.decode(
            generated_tokens, skip_special_tokens=True
        ).strip()
        # Trim trailing chat artifacts at first newline if present
        translation = translation.split("\n")[0].strip()

        yield TranslationChunk(text=translation, delta=translation, is_final=True)
