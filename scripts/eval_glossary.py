#!/usr/bin/env python3
"""Evaluate NLLB HF translator (control vs treatment) on FLORES or glossary slice.

Usage:
  # Control (no logit bias):
  python3 scripts/eval_glossary.py \
      --testset data/eval/ml_glossary_slice_en_ko.tsv \
      --src en --tgt ko --mode control \
      --report reports/exp_nllb-logit-bias-glossary/slice_en-ko_ctrl.json

  # Treatment (logit bias=3.0):
  python3 scripts/eval_glossary.py \
      --testset data/eval/ml_glossary_slice_en_ko.tsv \
      --src en --tgt ko --mode treatment \
      --report reports/exp_nllb-logit-bias-glossary/slice_en-ko_treat.json

Reports include BLEU, per-sentence term-recall, and full hyp strings.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from loguru import logger


GLOSSARY_PATH = "data/glossaries/ml-conference-en-ko.json"
BIAS_VALUES = [3.0]  # Can extend to [2.0, 3.0, 5.0] for sweep


def load_glossary(path: str) -> List[Dict]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("entries", [])


def load_testset(path: str, limit: Optional[int] = None) -> List[Tuple[str, str]]:
    rows = []
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        src = parts[0].strip()
        ref = parts[1].strip() if len(parts) > 1 else ""
        rows.append((src, ref))
    if limit:
        rows = rows[:limit]
    return rows


def compute_bleu(hyps: List[str], refs: List[str], tgt_lang: str) -> Tuple[float, str]:
    """Compute BLEU score using sacrebleu. Korean uses ko-mecab."""
    from sacrebleu.metrics import BLEU
    tokenize = "ko-mecab" if tgt_lang == "ko" else "13a"
    try:
        bleu = BLEU(tokenize=tokenize)
        score = bleu.corpus_score(hyps, [refs])
        return round(score.score, 2), tokenize
    except Exception as e:
        logger.warning(f"ko-mecab failed ({e}), falling back to char")
        bleu = BLEU(tokenize="char")
        score = bleu.corpus_score(hyps, [refs])
        return round(score.score, 2), "char"


def check_term_recall(
    src_text: str,
    hyp_text: str,
    glossary: List[Dict],
    src_lang: str,
    tgt_lang: str,
) -> Tuple[bool, List[str]]:
    """Check if all triggered target terms appear in the hypothesis.

    Returns (all_present, list_of_missing_terms).
    For en->ko: src term triggers, check tgt (Korean) in hyp.
    For ko->en: tgt term triggers (Korean src), check src (English) in hyp.
    """
    src_lower = src_text.lower()
    hyp_lower = hyp_text.lower()

    missing = []
    triggered_any = False

    for entry in glossary:
        if entry.get("dnt", False):
            # DNT: should keep original form. We skip recall check for these.
            continue

        if src_lang in ("en", "eng_Latn"):
            match_term = entry.get("src", "").lower()
            expected_in_hyp = entry.get("tgt", "")
        else:
            # ko->en: the Korean tgt is in source, English src should be in hyp
            match_term = entry.get("tgt", "").lower()
            expected_in_hyp = entry.get("src", "")

        if match_term and match_term in src_lower:
            triggered_any = True
            if expected_in_hyp and expected_in_hyp.lower() not in hyp_lower:
                missing.append(expected_in_hyp)

    if not triggered_any:
        return True, []  # No terms triggered = vacuously True

    return len(missing) == 0, missing


def run_eval(
    testset: List[Tuple[str, str]],
    src_lang: str,
    tgt_lang: str,
    glossary: List[Dict],
    use_bias: bool,
    bias_value: float = 3.0,
) -> List[Dict]:
    """Run translation and collect per-sentence results."""
    from src.translator.nllb_hf_glossary import NllbHfGlossaryTranslator

    logger.info(f"Loading translator (use_bias={use_bias}, bias={bias_value})")
    translator = NllbHfGlossaryTranslator(
        glossary_entries=glossary,
        use_bias=use_bias,
        bias_value=bias_value,
        max_new_tokens=256,
    )

    results = []
    n = len(testset)
    t_start = time.time()

    for i, (src_text, ref_text) in enumerate(testset):
        t0 = time.perf_counter()
        hyp, triggered_entries = translator.translate(src_text, src_lang=src_lang, tgt_lang=tgt_lang)
        latency_ms = (time.perf_counter() - t0) * 1000

        all_present, missing = check_term_recall(src_text, hyp, glossary, src_lang, tgt_lang)

        triggered_terms = []
        for entry in triggered_entries:
            if not entry.get("dnt", False):
                if src_lang in ("en", "eng_Latn"):
                    triggered_terms.append(entry.get("src", ""))
                else:
                    triggered_terms.append(entry.get("tgt", ""))

        rec = {
            "src": src_text,
            "ref": ref_text,
            "hyp": hyp,
            "triggered_terms": triggered_terms,
            "all_terms_present": all_present,
            "missing_terms": missing,
            "latency_ms": round(latency_ms, 1),
        }
        results.append(rec)

        if (i + 1) % 25 == 0 or (i + 1) == n:
            elapsed = time.time() - t_start
            logger.info(f"  {i+1}/{n} | elapsed={elapsed:.1f}s | last_latency={latency_ms:.0f}ms")

    return results


def summarize(results: List[Dict], tgt_lang: str) -> Dict:
    """Compute BLEU + term-recall summary stats."""
    hyps = [r["hyp"] for r in results]
    refs = [r["ref"] for r in results]

    bleu, bleu_tok = compute_bleu(hyps, refs, tgt_lang)

    # Term recall: only sentences where at least one non-DNT term was triggered
    triggered_results = [r for r in results if r["triggered_terms"]]
    if triggered_results:
        n_correct = sum(1 for r in triggered_results if r["all_terms_present"])
        recall = round(100.0 * n_correct / len(triggered_results), 1)
        n_triggered = len(triggered_results)
    else:
        recall = None
        n_triggered = 0

    return {
        "bleu": bleu,
        "bleu_tokenize": bleu_tok,
        "term_recall_pct": recall,
        "n_sentences": len(results),
        "n_triggered": n_triggered,
        "avg_latency_ms": round(sum(r["latency_ms"] for r in results) / max(len(results), 1), 1),
    }


def main():
    parser = argparse.ArgumentParser(description="Eval NLLB HF glossary translator")
    parser.add_argument("--testset", required=True, help="TSV: src\\tref")
    parser.add_argument("--src", default="en", help="Source language (en or ko)")
    parser.add_argument("--tgt", default="ko", help="Target language (ko or en)")
    parser.add_argument(
        "--mode",
        choices=["control", "treatment"],
        required=True,
        help="control=no bias, treatment=logit bias",
    )
    parser.add_argument("--bias", type=float, default=3.0, help="Logit bias value (default 3.0)")
    parser.add_argument("--glossary", default=GLOSSARY_PATH, help="Glossary JSON path")
    parser.add_argument("--limit", type=int, default=None, help="Limit N sentences (debug)")
    parser.add_argument("--report", required=True, help="Output JSON report path")
    args = parser.parse_args()

    logger.info(f"=== eval_glossary.py | {args.testset} | {args.src}->{args.tgt} | mode={args.mode} ===")

    glossary = load_glossary(args.glossary)
    logger.info(f"Loaded {len(glossary)} glossary entries from {args.glossary}")

    testset = load_testset(args.testset, limit=args.limit)
    logger.info(f"Loaded {len(testset)} test sentences from {args.testset}")

    use_bias = args.mode == "treatment"
    results = run_eval(testset, args.src, args.tgt, glossary, use_bias=use_bias, bias_value=args.bias)

    scores = summarize(results, tgt_lang=args.tgt)

    logger.info("=== Results ===")
    for k, v in scores.items():
        logger.info(f"  {k:<30} {v}")

    report = {
        "config": {
            "testset": args.testset,
            "src": args.src,
            "tgt": args.tgt,
            "mode": args.mode,
            "bias_value": args.bias if use_bias else None,
            "glossary": args.glossary,
            "n_sentences": len(testset),
        },
        "scores": scores,
        # Full hyp outputs (required for reviewer particle spot-check)
        "all_hyps": [r["hyp"] for r in results],
        "all_results": results,  # Full per-sentence data for review
    }

    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2))
    logger.info(f"Report saved -> {args.report}")

    # Print summary line for quick reading
    print(f"\n[SUMMARY] mode={args.mode} | {args.src}->{args.tgt} | testset={Path(args.testset).name}")
    print(f"  BLEU={scores['bleu']} ({scores['bleu_tokenize']})")
    print(f"  Term-recall={scores['term_recall_pct']}% on {scores['n_triggered']} triggered sents (of {scores['n_sentences']} total)")
    print(f"  Avg latency={scores['avg_latency_ms']}ms")


if __name__ == "__main__":
    main()
