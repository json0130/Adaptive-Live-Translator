"""NLLB-200-distilled-600M translator with OPTIONAL context injection.

Two modes controlled by cfg["translator"]["context_injection"]:
  - false (default): pure sentence-level NMT, baseline.
  - true: prepend a small structured context block to the source text
    before translation. Block contains:
      * glossary entries triggered by case-insensitive substring match
      * top-k=3 BM25 TM neighbours from data/translation_memory/en-ko.jsonl

The block goes IN the source string so it flows through NLLB's encoder.
NLLB has no system prompt; this is the lever we have.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import AsyncIterator

import torch
from loguru import logger

from .base import TranslationChunk, Translator


class NllbTranslator(Translator):
    """NLLB-200-distilled-600M, sentence-level, optional context prefix."""

    LANG_MAP = {"en": "eng_Latn", "ko": "kor_Hang"}

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        self.model_name = cfg["translator"]["model"]
        self.dtype_str = cfg["translator"].get("dtype", "float16")
        self.max_new_tokens = cfg["translator"].get("max_new_tokens", 256)
        self.context_injection = cfg["translator"].get("context_injection", False)
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

    # ------------------------------------------------------- context loading

    def _load_glossary_and_tm(self) -> None:
        gp = Path(self.glossary_path)
        if gp.exists():
            data = json.loads(gp.read_text(encoding="utf-8"))
            # data has src_lang, tgt_lang, entries[{src,tgt,dnt}]
            for e in data.get("entries", []):
                self._glossary.append({"src": e["src"], "tgt": e["tgt"]})
            logger.info(f"[ctx] glossary: {len(self._glossary)} entries")

        tp = Path(self.tm_path)
        if tp.exists():
            for line in tp.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                self._tm_pairs.append({"src": row["src"], "tgt": row["tgt"]})
            logger.info(f"[ctx] TM: {len(self._tm_pairs)} pairs")

            try:
                from rank_bm25 import BM25Okapi
                tokenised = [p["src"].lower().split() for p in self._tm_pairs]
                self._bm25 = BM25Okapi(tokenised)
            except ImportError:
                logger.warning("rank_bm25 not installed; TM retrieval skipped.")

    def _hits_for(self, src: str) -> list[dict]:
        lo = src.lower()
        return [e for e in self._glossary if e["src"].lower() in lo]

    def _retrieve_tm(self, src: str) -> list[dict]:
        if self._bm25 is None or not self._tm_pairs:
            return []
        scores = self._bm25.get_scores(src.lower().split())
        # Keep only matches with non-trivial overlap
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        top = [(i, s) for i, s in ranked[: self.tm_top_k] if s > 0.0]
        return [self._tm_pairs[i] for i, _ in top]

    def _build_context_prefix(self, src: str, src_lang: str, tgt_lang: str) -> str:
        # Only do glossary/TM in the en->ko direction since both data
        # files are en-source-keyed. For ko->en the lookup is the wrong
        # direction; we'd need a reverse index to use it usefully.
        if src_lang != "en":
            return ""
        gloss = self._hits_for(src)
        tm = self._retrieve_tm(src)
        parts: list[str] = []
        if gloss:
            parts.append(
                "GLOSS: " + "; ".join(f'{g["src"]}={g["tgt"]}' for g in gloss)
            )
        if tm:
            parts.append(
                "TM: " + " || ".join(f'{p["src"]} => {p["tgt"]}' for p in tm)
            )
        if not parts:
            return ""
        # Separator that should not appear in normal text.
        return " ".join(parts) + " ||| "

    # ------------------------------------------------------------ model

    def _lazy_load(self) -> None:
        if self._model is not None:
            return
        logger.info(f"Loading translator: {self.model_name}")
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16}
        dtype = dtype_map.get(self.dtype_str, torch.float32)

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_name, dtype=dtype, device_map="cuda",
        )
        logger.info(
            f"Model loaded on cuda dtype={self.dtype_str} "
            f"context_injection={self.context_injection}"
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

        src_flores = self.LANG_MAP.get(src_lang, src_lang)
        tgt_flores = self.LANG_MAP.get(tgt_lang, tgt_lang)

        src_text = ""
        async for chunk in src_chunks:
            src_text += chunk

        if self.context_injection:
            prefix = self._build_context_prefix(src_text, src_lang, tgt_lang)
            model_input = prefix + src_text
        else:
            model_input = src_text

        self._tokenizer.src_lang = src_flores
        inputs = self._tokenizer(
            model_input, return_tensors="pt", truncation=True, max_length=512
        ).to("cuda")

        forced_bos_token_id = self._tokenizer.convert_tokens_to_ids(tgt_flores)
        with torch.no_grad():
            generated_ids = self._model.generate(
                **inputs,
                forced_bos_token_id=forced_bos_token_id,
                max_new_tokens=self.max_new_tokens,
                num_beams=1,
                do_sample=False,
            )

        translation = self._tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0]

        yield TranslationChunk(text=translation, delta=translation, is_final=True)
