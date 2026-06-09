"""NLLB-200-distilled-600M via transformers fp16 on CUDA.

P2-1 GPU pipeline translator. Same model as Phase-1 CT2-int8, different
execution path: transformers fp16 on the GPU instead of CTranslate2 int8 on CPU.

Key choices:
  - transformers 4.27.4 compatible (MeloTTS hard-pins this version)
  - .half().to("cuda").eval() load
  - Greedy decode (num_beams=1) to match the Phase-1 baseline
  - FLORES lang codes: eng_Latn, kor_Hang

Warning notes:
  - transformers 4.27.4 may emit FutureWarning about resume_download:
    this is a huggingface_hub warning, not a model defect. Do NOT suppress it.
  - The tokenizer does NOT get fix_mistral_regex; NLLB uses SentencePiece
    which is handled correctly by the NLLB tokenizer class.
"""
from __future__ import annotations

import time
from typing import AsyncIterator, Optional

import torch
from loguru import logger
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from .base import TranslationChunk, Translator

# FLORES language codes for NLLB
FLORES_CODES = {
    "en": "eng_Latn",
    "ko": "kor_Hang",
}

MODEL_NAME = "facebook/nllb-200-distilled-600M"


class NllbFp16Translator(Translator):
    """NLLB-200-distilled-600M fp16 on CUDA via transformers.

    P2-1 GPU pipeline MT component.
    """

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        device: str = "cuda",
        max_new_tokens: int = 256,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.max_new_tokens = max_new_tokens
        self._model: Optional[AutoModelForSeq2SeqLM] = None
        self._tokenizer: Optional[AutoTokenizer] = None
        logger.info(
            f"NllbFp16Translator: model={model_name}, device={device}, "
            f"dtype=fp16, max_new_tokens={max_new_tokens}"
        )

    def load(self) -> None:
        """Load model and tokenizer eagerly."""
        if self._model is not None:
            return
        logger.info(f"Loading {self.model_name} fp16 on {self.device}...")
        # NOTE: transformers 4.27.4 compatibility: use from_pretrained + .half().to(device)
        # The newer .to(device, dtype=torch.float16) syntax also works in 4.27.4
        # but we use the explicit sequence to be safe.
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = (
            AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
            .half()
            .to(self.device)
            .eval()
        )
        logger.info(
            f"NLLB fp16 loaded on {next(self._model.parameters()).device}, "
            f"dtype={next(self._model.parameters()).dtype}"
        )

    def reset(self) -> None:
        """Stateless per-call — nothing to reset."""
        pass

    def translate(self, src_text: str, src_lang: str, tgt_lang: str) -> tuple[str, float]:
        """Translate synchronously. Returns (translated_text, mt_ms).

        Greedy decode (num_beams=1) to match the Phase-1 baseline.
        """
        if self._model is None:
            self.load()

        src_flores = FLORES_CODES.get(src_lang, src_lang)
        tgt_flores = FLORES_CODES.get(tgt_lang, tgt_lang)

        self._tokenizer.src_lang = src_flores
        inputs = self._tokenizer(src_text, return_tensors="pt").to(self.device)

        tgt_token_id = self._tokenizer.convert_tokens_to_ids(tgt_flores)

        t0 = time.perf_counter()
        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                forced_bos_token_id=tgt_token_id,
                max_new_tokens=self.max_new_tokens,
                num_beams=1,  # greedy, matches Phase-1 baseline
            )
        mt_ms = (time.perf_counter() - t0) * 1000.0

        tgt_text = self._tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return tgt_text, mt_ms

    async def translate_stream(
        self,
        src_chunks: AsyncIterator[str],
        *,
        src_lang: str,
        tgt_lang: str,
        system_prompt: str,
    ) -> AsyncIterator[TranslationChunk]:
        """Accumulate chunks, translate in one shot, yield single final chunk."""
        if self._model is None:
            self.load()

        src_text = ""
        async for chunk in src_chunks:
            src_text += chunk

        if not src_text.strip():
            return

        tgt_text, mt_ms = self.translate(src_text, src_lang, tgt_lang)
        yield TranslationChunk(text=tgt_text, delta=tgt_text, is_final=True)
