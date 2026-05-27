"""NLLB-200-distilled-600M via CTranslate2 INT8 quantization.

Key approach:
  - CTranslate2 converts the Hugging Face model to int8 quantization format
  - Runs on CPU with 8 intra-threads for 8-core laptop deployability
  - Uses greedy decoding (beam_size=1, no sampling) to match round 1 fp16 baseline
  - Multilingual SentencePiece tokenizer with FLORES lang codes
"""
from __future__ import annotations

import os
from typing import AsyncIterator

import ctranslate2
import sentencepiece as spm
from loguru import logger
from transformers import AutoTokenizer

from .base import TranslationChunk, Translator

# FLORES language codes for NLLB
FLORES_CODES = {
    "en": "eng_Latn",
    "ko": "kor_Hang",
}


class NllbCt2Translator(Translator):
    """NLLB via CTranslate2 INT8 on CPU."""

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        self.ct2_model_dir = cfg["translator"].get("ct2_model_dir", "models/nllb-600m-ct2-int8")
        self.device = cfg["translator"].get("device", "cpu")
        self.dtype = cfg["translator"].get("dtype", "int8")
        self.max_decoding_length = cfg["translator"].get("max_new_tokens", 256)

        self._ct2_translator = None
        self._tokenizer = None
        self._sp_model = None
        logger.info(f"NllbCt2Translator: model_dir={self.ct2_model_dir}, device={self.device}, dtype={self.dtype}")

    def _lazy_load(self) -> None:
        """Lazy-load model and tokenizer on first use."""
        if self._ct2_translator is not None:
            return

        logger.info(f"Loading CTranslate2 model from {self.ct2_model_dir}")

        # Load CTranslate2 translator
        self._ct2_translator = ctranslate2.Translator(
            self.ct2_model_dir,
            device=self.device,
            compute_type=self.dtype,
            intra_threads=8,
        )

        # Load tokenizer for SentencePiece
        self._tokenizer = AutoTokenizer.from_pretrained(self.ct2_model_dir, fix_mistral_regex=True)

        # Load SentencePiece model for proper token decoding
        sp_model_path = os.path.join(self.ct2_model_dir, "sentencepiece.bpe.model")
        self._sp_model = spm.SentencePieceProcessor()
        self._sp_model.Load(sp_model_path)

        logger.info("CTranslate2 translator and tokenizer loaded")

    def reset(self) -> None:
        """No KV cache to reset for CTranslate2 (stateless per batch)."""
        pass

    async def translate_stream(
        self,
        src_chunks: AsyncIterator[str],
        *,
        src_lang: str,
        tgt_lang: str,
        system_prompt: str,
    ) -> AsyncIterator[TranslationChunk]:
        """
        Accumulate source chunks into a complete sentence, translate via CTranslate2.
        Emit translation in one final chunk.
        """
        self._lazy_load()

        # Accumulate source text from chunks
        src_text = ""
        async for chunk in src_chunks:
            src_text += chunk

        if not src_text.strip():
            return

        # Get FLORES codes
        src_flores_code = FLORES_CODES.get(src_lang, src_lang)
        tgt_flores_code = FLORES_CODES.get(tgt_lang, tgt_lang)

        # Set source language on tokenizer and encode
        self._tokenizer.src_lang = src_flores_code
        src_tokens = self._tokenizer.encode(src_text, add_special_tokens=True)
        # Convert token IDs to token strings (what CTranslate2 expects)
        src_token_strings = self._tokenizer.convert_ids_to_tokens(src_tokens)

        # Create target prefix with language code (as token string)
        target_prefix_tokens = [tgt_flores_code]

        # Translate with CTranslate2
        # CTranslate2 expects List[List[str]] — strings, not IDs
        results = self._ct2_translator.translate_batch(
            [src_token_strings],
            target_prefix=[target_prefix_tokens],
            beam_size=1,  # Greedy decoding to match round 1 baseline
            max_decoding_length=self.max_decoding_length,
            return_scores=False,
        )

        # Decode the first (only) result
        # results[0].hypotheses[0] is the token list from greedy decoding (beam_size=1)
        output_token_strings = results[0].hypotheses[0]
        # Skip the target language token and decode pieces with SentencePiece
        # (SentencePiece properly handles merging of BPE subwords like ▁ and Ġ)
        if output_token_strings and output_token_strings[0] == tgt_flores_code:
            output_token_strings = output_token_strings[1:]
        tgt_text = self._sp_model.DecodePieces(output_token_strings)

        # Emit as single final chunk
        yield TranslationChunk(text=tgt_text, delta=tgt_text, is_final=True)
