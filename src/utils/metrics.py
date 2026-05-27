"""Metrics — BLEU and StreamLAAL (non-computation-aware) evaluation helpers.

StreamLAAL (Papi et al., 2024):
  Measures average latency of each target word relative to the corresponding
  source word timestamp. Lower is better; low regime target is ≤ 2.0 s.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SegmentRecord:
    src: str
    tgt: str
    src_end_ms: int         # when ASR emitted the source text
    tgt_emit_ms: int        # when the translation was emitted
    ref_tgt: str = ""       # optional reference for BLEU


def compute_bleu(
    hypotheses: list[str],
    references: list[str],
    tokenize: str = "13a",
) -> tuple[float, str]:
    """Corpus BLEU over hypothesis/reference string lists.

    Returns (score, tokenize_used). For ko targets pass tokenize="ko-mecab"
    per the locked eval protocol. If mecab-ko-dic is unavailable, falls
    back to "char" and records that in tokenize_used so all experiments
    can be checked for matching tokenizer.
    """
    try:
        from sacrebleu.metrics import BLEU
    except ImportError:
        return -1.0, "unavailable"

    requested = tokenize
    try:
        bleu = BLEU(effective_order=True, tokenize=requested)
        result = bleu.corpus_score(hypotheses, [references])
        return result.score, requested
    except Exception:
        # mecab-ko-dic missing on this host — fall back to char (resets the
        # comparison table per the eval protocol, but at least all four
        # experiments in this round will share the same fallback).
        if requested == "ko-mecab":
            bleu = BLEU(effective_order=True, tokenize="char")
            result = bleu.corpus_score(hypotheses, [references])
            return result.score, "char(fallback-from-ko-mecab)"
        raise


def compute_streamlaal(records: list[SegmentRecord]) -> float:
    """Approximate StreamLAAL — average delay per segment in seconds."""
    if not records:
        return 0.0
    delays = [(r.tgt_emit_ms - r.src_end_ms) / 1000.0 for r in records]
    return sum(delays) / len(delays)


def print_session_summary(records: list[SegmentRecord]) -> None:
    bleu = None
    if any(r.ref_tgt for r in records):
        bleu, _ = compute_bleu(
            [r.tgt for r in records],
            [r.ref_tgt for r in records if r.ref_tgt],
        )

    laal = compute_streamlaal(records)
    print(f"Segments      : {len(records)}")
    print(f"StreamLAAL    : {laal:.2f} s")
    if bleu is not None:
        print(f"BLEU          : {bleu:.1f}")
