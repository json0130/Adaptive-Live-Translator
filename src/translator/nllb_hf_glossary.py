"""NLLB-200-distilled-600M HF translator with optional soft logit-bias glossary.

Implements a GlossaryLogitsProcessor that raises the probability of
glossary-term token sequences at decode time (soft bias, not hard masking).

Usage:
  translator = NllbHfGlossaryTranslator(cfg, glossary_entries, use_bias=True)
  result = translator.translate(text, src_lang='eng_Latn', tgt_lang='kor_Hang')
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import AsyncIterator, Dict, List, Optional, Sequence, Tuple

import torch
from loguru import logger
from transformers import LogitsProcessor, NllbTokenizer, AutoModelForSeq2SeqLM

from .base import TranslationChunk, Translator

# FLORES language code mapping
LANG_CODE_MAP = {
    "en": "eng_Latn",
    "ko": "kor_Hang",
    "eng_Latn": "eng_Latn",
    "kor_Hang": "kor_Hang",
}


class GlossaryLogitsProcessor(LogitsProcessor):
    """Soft logit bias that boosts glossary target-term token sequences.

    State machine design:
    - self.term_sequences: list of token-id lists, one per triggered term.
    - self._matched: dict mapping seq_idx -> number of tokens matched so far.
      A seq_idx is "active" when we are in the middle of generating a term.
    - Each decode step:
        1. Look at the last generated token.
        2. For each active sequence, check if that token advanced the match:
           - If token == seq[n_matched], increment n_matched.
           - If n_matched == len(seq), term is complete -> remove from active.
           - If token != seq[n_matched], reset to 0 (mismatch kills the path).
        3. For sequences not yet active, check if last_token == seq[0]:
           if so, activate with n_matched=1 (already matched first token).
        4. Boost logits[seq[n_matched]] for all active sequences.
        5. ALSO try to start fresh sequences: boost seq[0] for all terms
           that are not currently matched (to encourage the model to begin them).
    """

    def __init__(
        self,
        tokenizer: NllbTokenizer,
        triggered_entries: List[Dict],
        bias: float = 3.0,
        tgt_lang: str = "kor_Hang",
    ) -> None:
        self.tokenizer = tokenizer
        self.bias = bias
        self.tgt_lang = tgt_lang

        # Encode each target term into token IDs (without special tokens)
        self.term_sequences: List[List[int]] = []
        # Track which terms have already been fully emitted (avoid re-biasing)
        self._completed: set = set()
        for entry in triggered_entries:
            tgt_text = entry.get("tgt", "")
            if not tgt_text:
                continue
            # Skip DNT entries (Do Not Translate)
            if entry.get("dnt", False):
                continue
            ids = tokenizer.encode(tgt_text, add_special_tokens=False)
            if ids:
                self.term_sequences.append(ids)
                logger.debug(f"  Bias target: '{tgt_text}' -> {ids}")

        # _matched[seq_idx] = number of tokens from term_sequences[seq_idx]
        # that have been generated so far in the current decode.
        self._matched: Dict[int, int] = {}

    def reset_state(self) -> None:
        self._matched = {}
        self._completed = set()

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """Called at each decode step. input_ids shape: [1, seq_len]."""
        if not self.term_sequences:
            return scores

        # Get the last generated token (the one just decided at previous step)
        last_token: Optional[int] = None
        if input_ids.shape[1] > 1:
            last_token = int(input_ids[0, -1].item())

        # --- Step 1: Update match states based on last generated token ---
        if last_token is not None:
            for seq_idx, seq in enumerate(self.term_sequences):
                if seq_idx in self._completed:
                    continue
                n = self._matched.get(seq_idx, 0)
                if n < len(seq) and last_token == seq[n]:
                    # This token advances the match
                    n += 1
                    if n == len(seq):
                        # Term fully generated
                        self._completed.add(seq_idx)
                        self._matched.pop(seq_idx, None)
                    else:
                        self._matched[seq_idx] = n
                elif n > 0:
                    # Mismatch mid-sequence: reset
                    # But check if last_token matches seq[0] (fresh start)
                    if last_token == seq[0]:
                        self._matched[seq_idx] = 1
                    else:
                        self._matched[seq_idx] = 0
                # n == 0 and no match: stays at 0, try fresh start below

        # --- Step 2: Boost next expected token for active sequences ---
        for seq_idx, seq in enumerate(self.term_sequences):
            if seq_idx in self._completed:
                continue
            n = self._matched.get(seq_idx, 0)
            if n > 0:
                # We're in the middle of this term; boost next token
                next_tok = seq[n]
                if next_tok < scores.shape[-1]:
                    scores[0, next_tok] += self.bias
            else:
                # Not started yet: boost the first token to encourage starting
                first_tok = seq[0]
                if first_tok < scores.shape[-1]:
                    scores[0, first_tok] += self.bias

        return scores


class NllbHfGlossaryTranslator:
    """Standalone NLLB HF translator with optional glossary logit bias.

    This is NOT wired through TranslationSession — it's a direct class
    used by scripts/eval_glossary.py for the experiment.
    """

    MODEL_ID = "facebook/nllb-200-distilled-600M"

    def __init__(
        self,
        glossary_entries: List[Dict],
        use_bias: bool = True,
        bias_value: float = 3.0,
        max_new_tokens: int = 256,
    ) -> None:
        self.glossary_entries = glossary_entries
        self.use_bias = use_bias
        self.bias_value = bias_value
        self.max_new_tokens = max_new_tokens

        logger.info(f"Loading NLLB tokenizer from {self.MODEL_ID}")
        # Do NOT pass fix_mistral_regex=True — that corrupts the tokenizer
        self.tokenizer = NllbTokenizer.from_pretrained(self.MODEL_ID)

        logger.info(f"Loading NLLB model from {self.MODEL_ID}")
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.MODEL_ID, torch_dtype=torch.float32
        )
        self.model.eval()
        logger.info(f"Model loaded. use_bias={use_bias}, bias={bias_value}")

    def _get_triggered_entries(self, src_text: str, src_lang: str) -> List[Dict]:
        """Find glossary entries whose source term appears in src_text."""
        # Determine direction: en->ko uses src field, ko->en is reversed
        triggered = []
        src_lower = src_text.lower()
        for entry in self.glossary_entries:
            # For en->ko: match entry['src'] in source text
            # For ko->en: match entry['tgt'] in source text (reversed direction)
            if src_lang in ("en", "eng_Latn"):
                term = entry.get("src", "")
            else:
                term = entry.get("tgt", "")
            if term and term.lower() in src_lower:
                triggered.append(entry)
        return triggered

    def _build_processor(self, triggered: List[Dict], tgt_lang: str) -> Optional[GlossaryLogitsProcessor]:
        if not triggered or not self.use_bias:
            return None
        # For ko->en direction, we need to reverse: boost the English src terms
        entries_for_bias = []
        for entry in triggered:
            if tgt_lang in ("ko", "kor_Hang"):
                # en->ko: bias towards Korean tgt
                entries_for_bias.append({"tgt": entry.get("tgt", ""), "dnt": entry.get("dnt", False)})
            else:
                # ko->en: bias towards English src
                entries_for_bias.append({"tgt": entry.get("src", ""), "dnt": entry.get("dnt", False)})
        return GlossaryLogitsProcessor(self.tokenizer, entries_for_bias, bias=self.bias_value, tgt_lang=tgt_lang)

    def translate(self, src_text: str, src_lang: str = "en", tgt_lang: str = "ko") -> Tuple[str, List[Dict]]:
        """Translate a single sentence. Returns (translated_text, triggered_entries)."""
        src_flores = LANG_CODE_MAP.get(src_lang, src_lang)
        tgt_flores = LANG_CODE_MAP.get(tgt_lang, tgt_lang)

        triggered = self._get_triggered_entries(src_text, src_lang)

        # Tokenize
        inputs = self.tokenizer(src_text, return_tensors="pt", src_lang=src_flores)
        forced_bos = self.tokenizer.convert_tokens_to_ids(tgt_flores)

        # Build logits processor if treatment mode
        processors = []
        if self.use_bias and triggered:
            proc = self._build_processor(triggered, tgt_lang)
            if proc is not None:
                proc.reset_state()
                processors.append(proc)

        with torch.no_grad():
            if processors:
                from transformers import LogitsProcessorList
                out = self.model.generate(
                    **inputs,
                    forced_bos_token_id=forced_bos,
                    max_new_tokens=self.max_new_tokens,
                    num_beams=1,
                    do_sample=False,
                    logits_processor=LogitsProcessorList(processors),
                )
            else:
                out = self.model.generate(
                    **inputs,
                    forced_bos_token_id=forced_bos,
                    max_new_tokens=self.max_new_tokens,
                    num_beams=1,
                    do_sample=False,
                )

        result = self.tokenizer.batch_decode(out, skip_special_tokens=True)[0]
        return result, triggered

    def translate_batch(
        self,
        sentences: List[str],
        src_lang: str = "en",
        tgt_lang: str = "ko",
    ) -> List[Tuple[str, List[Dict]]]:
        """Translate a list of sentences one by one (greedy, no batching for simplicity)."""
        results = []
        for i, sent in enumerate(sentences):
            hyp, triggered = self.translate(sent, src_lang=src_lang, tgt_lang=tgt_lang)
            results.append((hyp, triggered))
            if (i + 1) % 20 == 0:
                logger.info(f"  Translated {i+1}/{len(sentences)}")
        return results
