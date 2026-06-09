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


_KO_MECAB_TAGGER = None


def _ko_mecab_tagger():
    """Direct MeCab wakati tagger using mecab_ko_dic's dictionary.

    We bypass sacrebleu's TokenizerKoMecab because its version coupling to
    mecab_ko / mecab_ko_dic broke mid-project (a MeloTTS install upgraded
    mecab_ko_dic; sacrebleu's tokenizer expects attributes — MECAB_ARGS,
    then DICDIR — that the new release dropped, which silently forced a
    char-BLEU fallback). Calling MeCab directly with the dictionary path is
    stable and gives the same morpheme segmentation. Returns None if MeCab
    is genuinely unavailable.
    """
    global _KO_MECAB_TAGGER
    if _KO_MECAB_TAGGER is not None:
        return _KO_MECAB_TAGGER
    try:
        import MeCab, mecab_ko_dic
        # mecab_ko_dic's API has drifted across releases. Prefer MECAB_ARGS
        # (it bundles -r <mecabrc> -d <dicdir>): a co-installed unidic — e.g.
        # pulled in by a MeloTTS install — hijacks MeCab's default dictionary,
        # so a bare "-d <dir>" is not enough; the rc file must be set too.
        # Fall back to DICDIR+mecabrc, then the older dictionary_path attr.
        if hasattr(mecab_ko_dic, "MECAB_ARGS"):
            args = mecab_ko_dic.MECAB_ARGS + " -Owakati"
        elif hasattr(mecab_ko_dic, "DICDIR"):
            args = f'-r "{mecab_ko_dic.mecabrc}" -d "{mecab_ko_dic.DICDIR}" -Owakati'
        else:
            args = f"-d {mecab_ko_dic.dictionary_path} -Owakati"
        _KO_MECAB_TAGGER = MeCab.Tagger(args)
        return _KO_MECAB_TAGGER
    except Exception:
        return None


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

    # ko-mecab: pre-tokenize with a direct MeCab tagger, then score with
    # tokenize='none' (sacrebleu's built-in ko-mecab tokenizer is broken by
    # a mecab_ko_dic version mismatch — see _ko_mecab_tagger).
    if requested == "ko-mecab":
        tagger = _ko_mecab_tagger()
        if tagger is not None:
            try:
                hyp_t = [tagger.parse(h).strip() for h in hypotheses]
                ref_t = [tagger.parse(r).strip() for r in references]
                bleu = BLEU(effective_order=True, tokenize="none")
                return bleu.corpus_score(hyp_t, [ref_t]).score, "ko-mecab"
            except Exception:
                pass
        # genuinely unavailable -> char fallback (records the fallback)
        bleu = BLEU(effective_order=True, tokenize="char")
        return bleu.corpus_score(hypotheses, [references]).score, "char(fallback-from-ko-mecab)"

    try:
        bleu = BLEU(effective_order=True, tokenize=requested)
        result = bleu.corpus_score(hypotheses, [references])
        return result.score, requested
    except Exception:
        raise


def compute_streamlaal(records: list[SegmentRecord]) -> float:
    """Approximate StreamLAAL — average delay per segment in seconds."""
    if not records:
        return 0.0
    delays = [(r.tgt_emit_ms - r.src_end_ms) / 1000.0 for r in records]
    return sum(delays) / len(delays)


def print_session_summary(records: list[SegmentRecord]) -> None:
    if any(r.ref_tgt for r in records):
        score, tok = compute_bleu(
            [r.tgt for r in records],
            [r.ref_tgt for r in records if r.ref_tgt],
        )
    else:
        score, tok = None, None

    laal = compute_streamlaal(records)
    print(f"Segments      : {len(records)}")
    print(f"StreamLAAL    : {laal:.2f} s")
    if score is not None:
        print(f"BLEU ({tok})  : {score:.1f}")
