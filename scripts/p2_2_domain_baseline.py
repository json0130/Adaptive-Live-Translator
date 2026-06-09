"""P2-2 Domain Baseline — NLLB-600M native terminology recall on ML glossary slice.

Measures NLLB's *native* (no glossary enforcement) term recall on the ML
glossary slice (146 pairs each direction). TEXT ONLY — no ASR, no TTS.

Deliverables:
  - reports/p2-2_domain_baseline/result_en_ko.json
  - reports/p2-2_domain_baseline/result_ko_en.json
  - reports/p2-2_domain_baseline/summary.md

All headline numbers must be grep-able in the committed JSON files.

Usage:
    PYTHONPATH=$PWD ./.translator/bin/python scripts/p2_2_domain_baseline.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

# ---- repo root on sys.path ----
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from loguru import logger
from src.translator.nllb_fp16_translator import NllbFp16Translator
from src.utils.metrics import compute_bleu

# ---- paths ----
GLOSSARY_PATH = REPO / "data/glossaries/ml-conference-en-ko.json"
SLICE_EN_KO = REPO / "data/eval/ml_glossary_slice_en_ko.tsv"
SLICE_KO_EN = REPO / "data/eval/ml_glossary_slice_ko_en.tsv"
REPORT_DIR = REPO / "reports/p2-2_domain_baseline"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# ---- MISS classification helpers ----
# Known valid Korean alternatives for canonical terms (for miss classification)
VALID_ALTERNATIVES_KO: dict[str, list[str]] = {
    "대형 언어 모델": ["거대 언어 모델", "언어 모델", "대규모 언어 모델"],
    "환각 현상": ["할루시네이션", "환각"],
    "어텐션 메커니즘": ["어텐션"],
    "양자화": ["퀀타이즈", "정량화"],
    "지연 시간": ["레이턴시", "대기 시간", "지연"],
    "처리량": ["처리 속도", "스루풋"],
    "벤치마크": [],
    "파인튜닝": ["미세 조정", "파인 튜닝"],
    "임베딩": [],
    "토크나이저": ["토큰화기"],
    "추론": [],
}
VALID_ALTERNATIVES_EN: dict[str, list[str]] = {
    "LLM": ["large language model", "language model"],
    "fine-tuning": ["finetuning", "fine tuning"],
    "inference": [],
    "embedding": [],
    "attention mechanism": ["attention"],
    "tokenizer": [],
    "hallucination": [],
    "benchmark": [],
    "throughput": [],
    "latency": [],
    "quantization": ["quantisation"],
}


def load_glossary(path: Path) -> list[dict]:
    """Load glossary entries from JSON."""
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["entries"]


def load_tsv(path: Path) -> list[tuple[str, str]]:
    """Load tab-separated (source, reference) pairs, no header."""
    pairs = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split("\t", 1)
        if len(parts) == 2:
            pairs.append((parts[0], parts[1]))
    return pairs


def check_term_in_output(expected_term: str, output: str, case_sensitive: bool = False) -> bool:
    """Substring check: does expected_term appear in output?"""
    if case_sensitive:
        return expected_term in output
    return expected_term.lower() in output.lower()


def classify_miss(
    entry: dict,
    output: str,
    direction: str,  # "en_ko" or "ko_en"
) -> str:
    """Classify WHY a glossary term was missed.

    Returns one of:
      "valid_alternative" — NLLB produced a different-but-valid translation
      "untranslated"      — source term appears verbatim in output (not translated)
      "wrong_or_omitted"  — genuinely wrong or absent
    """
    src_term = entry["src"]
    tgt_term = entry["tgt"]

    if direction == "en_ko":
        # Check if source English term appears verbatim in output (untranslated)
        if src_term.lower() in output.lower():
            return "untranslated"
        # Check for known valid Korean alternatives
        alts = VALID_ALTERNATIVES_KO.get(tgt_term, [])
        for alt in alts:
            if alt in output:
                return "valid_alternative"
        return "wrong_or_omitted"

    else:  # ko_en
        # Check if source Korean term appears verbatim in output
        if tgt_term.lower() in output.lower():
            return "untranslated"
        # Check for valid English alternatives
        alts = VALID_ALTERNATIVES_EN.get(src_term, [])
        for alt in alts:
            if alt.lower() in output.lower():
                return "valid_alternative"
        return "wrong_or_omitted"


def has_particle_concern(output_ko: str) -> bool:
    """Simple heuristic: flag if output looks truncated or malformed.

    Full linguistic analysis is for the reviewer; we just flag anomalies.
    """
    # Flag very short outputs (< 3 chars) or outputs with no Hangul
    if len(output_ko.strip()) < 3:
        return True
    hangul_count = sum(1 for c in output_ko if '가' <= c <= '힣' or 'ᄀ' <= c <= 'ᇿ')
    if hangul_count == 0 and len(output_ko) > 5:
        return True
    return False


def run_direction(
    translator: NllbFp16Translator,
    pairs: list[tuple[str, str]],
    entries: list[dict],
    src_lang: str,
    tgt_lang: str,
    direction: str,  # "en_ko" or "ko_en"
) -> tuple[list[dict], dict]:
    """Translate all pairs, check term recall, return (per_segment_records, aggregate_stats)."""

    logger.info(f"Running direction: {direction} — {len(pairs)} segments")

    hypotheses = []
    references = []
    per_segment = []

    # Per-term accumulators: {entry_idx: {"triggered": 0, "correct": 0}}
    per_term = [{"triggered": 0, "correct": 0} for _ in entries]

    for seg_idx, (src_text, ref_text) in enumerate(pairs):
        t0 = time.perf_counter()
        output, mt_ms = translator.translate(src_text, src_lang, tgt_lang)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        hypotheses.append(output)
        references.append(ref_text)

        # Determine triggered terms and check recall
        triggered_terms = []
        for entry_idx, entry in enumerate(entries):
            if direction == "en_ko":
                # Trigger: src English term in English source
                trigger_term = entry["src"]
                expected_output_term = entry["tgt"]
                # Use word-boundary match for trigger detection (same as glossary.py)
                import re
                pattern = r"\b" + re.escape(trigger_term.lower()) + r"\b"
                triggered = bool(re.search(pattern, src_text.lower()))
            else:  # ko_en
                # Trigger: Korean tgt term appears in Korean source
                trigger_term = entry["tgt"]
                expected_output_term = entry["src"]
                triggered = trigger_term in src_text

            if triggered:
                per_term[entry_idx]["triggered"] += 1
                if entry.get("dnt", False):
                    # DNT: check verbatim pass-through of the token in output
                    found = check_term_in_output(entry["src"], output, case_sensitive=True)
                else:
                    if direction == "en_ko":
                        # Korean target: substring check (handles particles)
                        found = check_term_in_output(expected_output_term, output, case_sensitive=True)
                    else:
                        # English target: case-insensitive
                        found = check_term_in_output(expected_output_term, output, case_sensitive=False)

                if found:
                    per_term[entry_idx]["correct"] += 1
                    miss_reason = None
                else:
                    miss_reason = classify_miss(entry, output, direction)

                triggered_terms.append({
                    "entry_src": entry["src"],
                    "entry_tgt": entry["tgt"],
                    "dnt": entry.get("dnt", False),
                    "expected_target": expected_output_term,
                    "found": found,
                    "miss_reason": miss_reason,
                })

        # Flag potential Korean grammar issues
        particle_concern = False
        if tgt_lang == "ko":
            particle_concern = has_particle_concern(output)

        record = {
            "seg_idx": seg_idx,
            "src": src_text,
            "reference": ref_text,
            "output": output,
            "mt_ms": round(elapsed_ms, 1),
            "triggered_terms": triggered_terms,
            "particle_concern": particle_concern,
        }
        per_segment.append(record)

        if (seg_idx + 1) % 25 == 0:
            logger.info(f"  [{direction}] {seg_idx+1}/{len(pairs)} done")

    # ---- Aggregate stats ----
    # BLEU
    if tgt_lang == "ko":
        bleu_score, tokenize_used = compute_bleu(hypotheses, references, tokenize="ko-mecab")
        assert tokenize_used == "ko-mecab", (
            f"ABORT: ko-mecab tokenizer not available, got fallback '{tokenize_used}'. "
            "Check MeCab installation."
        )
    else:
        bleu_score, tokenize_used = compute_bleu(hypotheses, references, tokenize="13a")
        assert tokenize_used == "13a", (
            f"ABORT: 13a tokenizer not available, got fallback '{tokenize_used}'."
        )

    # Term recall per entry
    per_term_stats = []
    total_triggered = 0
    total_correct = 0
    for entry_idx, entry in enumerate(entries):
        t = per_term[entry_idx]["triggered"]
        c = per_term[entry_idx]["correct"]
        total_triggered += t
        total_correct += c
        recall = (c / t) if t > 0 else None
        per_term_stats.append({
            "src": entry["src"],
            "tgt": entry["tgt"],
            "dnt": entry.get("dnt", False),
            "triggered": t,
            "correct": c,
            "recall": round(recall, 4) if recall is not None else None,
        })

    overall_recall = (total_correct / total_triggered) if total_triggered > 0 else 0.0

    # Miss breakdown
    miss_breakdown = {"valid_alternative": 0, "untranslated": 0, "wrong_or_omitted": 0}
    for rec in per_segment:
        for tt in rec["triggered_terms"]:
            if not tt["found"] and tt["miss_reason"] is not None:
                miss_breakdown[tt["miss_reason"]] = miss_breakdown.get(tt["miss_reason"], 0) + 1

    # Segments with particle concerns
    particle_concerns = [r["seg_idx"] for r in per_segment if r.get("particle_concern")]

    aggregate = {
        "direction": direction,
        "n_segments": len(pairs),
        "bleu": round(bleu_score, 2),
        "tokenize_used": tokenize_used,
        "total_triggered": total_triggered,
        "total_correct": total_correct,
        "overall_recall": round(overall_recall, 4),
        "miss_breakdown": miss_breakdown,
        "per_term": per_term_stats,
        "particle_concern_segments": particle_concerns,
    }

    return per_segment, aggregate


def main():
    logger.remove()
    logger.add(sys.stderr, level="INFO")

    logger.info("P2-2 Domain Baseline: loading data...")
    entries = load_glossary(GLOSSARY_PATH)
    pairs_en_ko = load_tsv(SLICE_EN_KO)
    pairs_ko_en = load_tsv(SLICE_KO_EN)

    assert len(pairs_en_ko) == 146, f"Expected 146 en->ko pairs, got {len(pairs_en_ko)}"
    assert len(pairs_ko_en) == 146, f"Expected 146 ko->en pairs, got {len(pairs_ko_en)}"
    logger.info(f"Loaded {len(entries)} glossary entries, {len(pairs_en_ko)} pairs each direction")

    # Load translator (ONCE, shared for both directions)
    logger.info("Loading NLLB-600M fp16 on CUDA...")
    translator = NllbFp16Translator(device="cuda")
    translator.load()
    logger.info("Model loaded.")

    # ---- en -> ko ----
    segs_en_ko, agg_en_ko = run_direction(
        translator, pairs_en_ko, entries,
        src_lang="en", tgt_lang="ko", direction="en_ko"
    )
    logger.info(f"en->ko BLEU={agg_en_ko['bleu']} ({agg_en_ko['tokenize_used']}), "
                f"recall={agg_en_ko['overall_recall']:.4f} "
                f"({agg_en_ko['total_correct']}/{agg_en_ko['total_triggered']})")

    # ---- ko -> en ----
    segs_ko_en, agg_ko_en = run_direction(
        translator, pairs_ko_en, entries,
        src_lang="ko", tgt_lang="en", direction="ko_en"
    )
    logger.info(f"ko->en BLEU={agg_ko_en['bleu']} ({agg_ko_en['tokenize_used']}), "
                f"recall={agg_ko_en['overall_recall']:.4f} "
                f"({agg_ko_en['total_correct']}/{agg_ko_en['total_triggered']})")

    # ---- Save full results ----
    result_en_ko = {
        "experiment": "P2-2_domain_baseline",
        "direction": "en_ko",
        "aggregate": agg_en_ko,
        "per_segment_samples": segs_en_ko,
    }
    result_ko_en = {
        "experiment": "P2-2_domain_baseline",
        "direction": "ko_en",
        "aggregate": agg_ko_en,
        "per_segment_samples": segs_ko_en,
    }

    out_en_ko = REPORT_DIR / "result_en_ko.json"
    out_ko_en = REPORT_DIR / "result_ko_en.json"
    out_en_ko.write_text(json.dumps(result_en_ko, ensure_ascii=False, indent=2), encoding="utf-8")
    out_ko_en.write_text(json.dumps(result_ko_en, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"Results saved to {out_en_ko} and {out_ko_en}")

    # ---- Write summary.md ----
    write_summary(agg_en_ko, agg_ko_en, REPORT_DIR)
    logger.info("Done. Summary written.")


def write_summary(agg_en_ko: dict, agg_ko_en: dict, report_dir: Path):
    """Write summary.md with all headline numbers grep-able in result JSON files."""

    lines = []
    lines.append("# P2-2 Domain Baseline — Summary")
    lines.append("")
    lines.append("Experiment: NLLB-200-distilled-600M fp16 on CUDA, native term recall measurement.")
    lines.append("No glossary enforcement. Greedy decode (num_beams=1). TEXT ONLY.")
    lines.append("")
    lines.append("## Headline Numbers")
    lines.append("")
    lines.append(f"| Metric | en->ko | ko->en |")
    lines.append(f"|--------|--------|--------|")
    lines.append(f"| N segments | {agg_en_ko['n_segments']} | {agg_ko_en['n_segments']} |")
    lines.append(f"| BLEU | {agg_en_ko['bleu']} | {agg_ko_en['bleu']} |")
    lines.append(f"| tokenize_used | {agg_en_ko['tokenize_used']} | {agg_ko_en['tokenize_used']} |")
    lines.append(f"| total_triggered | {agg_en_ko['total_triggered']} | {agg_ko_en['total_triggered']} |")
    lines.append(f"| total_correct | {agg_en_ko['total_correct']} | {agg_ko_en['total_correct']} |")
    lines.append(f"| overall_recall | {agg_en_ko['overall_recall']} | {agg_ko_en['overall_recall']} |")
    lines.append("")

    lines.append("## Per-Term Recall (en->ko direction)")
    lines.append("")
    lines.append("| Term (en) | Term (ko) | DNT | Triggered | Correct | Recall |")
    lines.append("|-----------|-----------|-----|-----------|---------|--------|")
    for pt in agg_en_ko["per_term"]:
        recall_str = f"{pt['recall']:.4f}" if pt["recall"] is not None else "N/A"
        lines.append(f"| {pt['src']} | {pt['tgt']} | {pt['dnt']} | {pt['triggered']} | {pt['correct']} | {recall_str} |")

    lines.append("")
    lines.append("Per-term sum check: "
                 f"triggered={sum(p['triggered'] for p in agg_en_ko['per_term'])} "
                 f"(note: one segment may trigger multiple terms), "
                 f"correct={sum(p['correct'] for p in agg_en_ko['per_term'])}")
    lines.append(f"total_triggered={agg_en_ko['total_triggered']}, total_correct={agg_en_ko['total_correct']}")
    lines.append("")

    lines.append("## Per-Term Recall (ko->en direction)")
    lines.append("")
    lines.append("| Term (en) | Term (ko) | DNT | Triggered | Correct | Recall |")
    lines.append("|-----------|-----------|-----|-----------|---------|--------|")
    for pt in agg_ko_en["per_term"]:
        recall_str = f"{pt['recall']:.4f}" if pt["recall"] is not None else "N/A"
        lines.append(f"| {pt['src']} | {pt['tgt']} | {pt['dnt']} | {pt['triggered']} | {pt['correct']} | {recall_str} |")

    lines.append("")
    lines.append(f"total_triggered={agg_ko_en['total_triggered']}, total_correct={agg_ko_en['total_correct']}")
    lines.append("")

    lines.append("## Miss Breakdown")
    lines.append("")
    lines.append("### en->ko misses")
    mb = agg_en_ko["miss_breakdown"]
    lines.append(f"- valid_alternative: {mb.get('valid_alternative', 0)}")
    lines.append(f"- untranslated: {mb.get('untranslated', 0)}")
    lines.append(f"- wrong_or_omitted: {mb.get('wrong_or_omitted', 0)}")
    total_misses_en_ko = agg_en_ko["total_triggered"] - agg_en_ko["total_correct"]
    lines.append(f"- total misses: {total_misses_en_ko}")
    lines.append(f"  (reconcile: {mb.get('valid_alternative',0)} + {mb.get('untranslated',0)} + {mb.get('wrong_or_omitted',0)} = {sum(mb.values())} vs total_misses={total_misses_en_ko})")
    lines.append("")
    lines.append("### ko->en misses")
    mb2 = agg_ko_en["miss_breakdown"]
    lines.append(f"- valid_alternative: {mb2.get('valid_alternative', 0)}")
    lines.append(f"- untranslated: {mb2.get('untranslated', 0)}")
    lines.append(f"- wrong_or_omitted: {mb2.get('wrong_or_omitted', 0)}")
    total_misses_ko_en = agg_ko_en["total_triggered"] - agg_ko_en["total_correct"]
    lines.append(f"- total misses: {total_misses_ko_en}")
    lines.append(f"  (reconcile: {mb2.get('valid_alternative',0)} + {mb2.get('untranslated',0)} + {mb2.get('wrong_or_omitted',0)} = {sum(mb2.values())} vs total_misses={total_misses_ko_en})")
    lines.append("")

    lines.append("## Reviewer Spot-Check Candidates")
    lines.append("")
    pcs_en = agg_en_ko.get("particle_concern_segments", [])
    pcs_ko = agg_ko_en.get("particle_concern_segments", [])
    lines.append(f"en->ko particle concern segments (seg_idx): {pcs_en}")
    lines.append(f"ko->en particle concern segments (seg_idx): {pcs_ko}")
    lines.append("")

    lines.append("## Reconciliation")
    lines.append("")
    lines.append("All headline numbers are grep-able in:")
    lines.append("  reports/p2-2_domain_baseline/result_en_ko.json")
    lines.append("  reports/p2-2_domain_baseline/result_ko_en.json")
    lines.append("")
    lines.append("## Reproduction")
    lines.append("")
    lines.append("```")
    lines.append("PYTHONPATH=$PWD ./.translator/bin/python scripts/p2_2_domain_baseline.py")
    lines.append("```")

    summary_path = report_dir / "summary.md"
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"Summary written to {summary_path}")


if __name__ == "__main__":
    main()
