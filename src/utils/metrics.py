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


def compute_bleu(hypotheses: list[str], references: list[str]) -> float:
    """Corpus BLEU over hypothesis/reference string lists."""
    try:
        from sacrebleu.metrics import BLEU
        bleu = BLEU(effective_order=True)
        result = bleu.corpus_score(hypotheses, [references])
        return result.score
    except ImportError:
        return -1.0


def compute_streamlaal(records: list[SegmentRecord]) -> float:
    """Approximate StreamLAAL — average delay per segment in seconds."""
    if not records:
        return 0.0
    delays = [(r.tgt_emit_ms - r.src_end_ms) / 1000.0 for r in records]
    return sum(delays) / len(delays)


def print_session_summary(records: list[SegmentRecord]) -> None:
    bleu = compute_bleu(
        [r.tgt for r in records],
        [r.ref_tgt for r in records if r.ref_tgt],
    ) if any(r.ref_tgt for r in records) else None

    laal = compute_streamlaal(records)
    print(f"Segments      : {len(records)}")
    print(f"StreamLAAL    : {laal:.2f} s")
    if bleu is not None:
        print(f"BLEU          : {bleu:.1f}")
