#!/usr/bin/env python3
"""Evaluate NLLB phrase-constrained decoding (control vs treatment) on FLORES or glossary slice.

Experiment: exp/glossary-phrase-constrained
Method: PhrasalConstraint (force_words_ids) with beam search (num_beams=4), cap=1 term/sent.

Usage:
  # Control (greedy, no constraint):
  python3 scripts/eval_phrase_constrained.py \
      --testset data/eval/ml_glossary_slice_en_ko.tsv \
      --src en --tgt ko --mode control \
      --report reports/exp_glossary-phrase-constrained/slice_en-ko_ctrl.json

  # Treatment (phrase constraint, beam=4, cap=1):
  python3 scripts/eval_phrase_constrained.py \
      --testset data/eval/ml_glossary_slice_en_ko.tsv \
      --src en --tgt ko --mode treatment \
      --report reports/exp_glossary-phrase-constrained/slice_en-ko_treat.json

Reports include BLEU (ko-mecab), per-sentence term-recall, and full hyp strings.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from loguru import logger

GLOSSARY_PATH = "data/glossaries/ml-conference-en-ko.json"
NUM_BEAMS = 4   # beam width for constrained beam search
CAP_TERMS = 1   # max simultaneous term constraints per sentence


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
    """Compute BLEU using the R4-5 fixed ko-mecab via src/utils/metrics.py."""
    # Add project root to path
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))
    from src.utils.metrics import compute_bleu as _compute_bleu

    tokenize = "ko-mecab" if tgt_lang in ("ko", "kor_Hang") else "13a"
    score, tok_used = _compute_bleu(hyps, refs, tokenize=tokenize)
    return round(score, 2), tok_used


def check_term_recall(
    src_text: str,
    hyp_text: str,
    glossary: List[Dict],
    src_lang: str,
    tgt_lang: str,
) -> Tuple[bool, List[str], List[str]]:
    """Check if all triggered target terms appear in the hypothesis.

    Returns (all_present, missing_terms, triggered_terms).
    """
    src_lower = src_text.lower()
    hyp_lower = hyp_text.lower()

    missing = []
    triggered_terms = []

    for entry in glossary:
        if entry.get("dnt", False):
            continue

        if src_lang in ("en", "eng_Latn"):
            match_term = entry.get("src", "").lower()
            expected_in_hyp = entry.get("tgt", "")
        else:
            match_term = entry.get("tgt", "").lower()
            expected_in_hyp = entry.get("src", "")

        if match_term and match_term in src_lower:
            triggered_terms.append(match_term)
            if expected_in_hyp and expected_in_hyp.lower() not in hyp_lower:
                missing.append(expected_in_hyp)

    return len(missing) == 0 and bool(triggered_terms), missing, triggered_terms


def run_eval(
    testset: List[Tuple[str, str]],
    src_lang: str,
    tgt_lang: str,
    glossary: List[Dict],
    use_constraint: bool,
    num_beams: int = NUM_BEAMS,
    cap_terms: int = CAP_TERMS,
) -> List[Dict]:
    """Run translation and collect per-sentence results."""
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))
    from src.translator.nllb_phrase_constrained import NllbPhraseConstrainedTranslator

    logger.info(
        f"Loading translator (use_constraint={use_constraint}, "
        f"num_beams={num_beams}, cap_terms={cap_terms})"
    )
    translator = NllbPhraseConstrainedTranslator(
        glossary_entries=glossary,
        use_constraint=use_constraint,
        num_beams=num_beams,
        max_new_tokens=128,
        cap_terms=cap_terms,
    )

    results = []
    n = len(testset)
    t_start = time.time()

    for i, (src_text, ref_text) in enumerate(testset):
        t0 = time.perf_counter()
        try:
            hyp, triggered_entries = translator.translate(
                src_text, src_lang=src_lang, tgt_lang=tgt_lang
            )
        except Exception as e:
            logger.error(f"Translation error on sentence {i}: {e}")
            hyp = ""
            triggered_entries = []
        latency_ms = (time.perf_counter() - t0) * 1000

        all_present, missing, triggered_terms = check_term_recall(
            src_text, hyp, glossary, src_lang, tgt_lang
        )

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
            logger.info(
                f"  {i+1}/{n} | elapsed={elapsed:.1f}s | last_latency={latency_ms:.0f}ms"
            )

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
        "avg_latency_ms": round(
            sum(r["latency_ms"] for r in results) / max(len(results), 1), 1
        ),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Eval NLLB phrase-constrained translator"
    )
    parser.add_argument("--testset", required=True, help="TSV: src\\tref")
    parser.add_argument("--src", default="en", help="Source language (en or ko)")
    parser.add_argument("--tgt", default="ko", help="Target language (ko or en)")
    parser.add_argument(
        "--mode",
        choices=["control", "treatment"],
        required=True,
        help="control=greedy no-constraint, treatment=phrase constrained beam search",
    )
    parser.add_argument(
        "--num-beams", type=int, default=NUM_BEAMS,
        help=f"Beam width for treatment mode (default {NUM_BEAMS})"
    )
    parser.add_argument(
        "--cap-terms", type=int, default=CAP_TERMS,
        help=f"Max terms to constrain per sentence (default {CAP_TERMS})"
    )
    parser.add_argument("--glossary", default=GLOSSARY_PATH, help="Glossary JSON path")
    parser.add_argument("--limit", type=int, default=None, help="Limit N sentences (debug)")
    parser.add_argument("--report", required=True, help="Output JSON report path")
    args = parser.parse_args()

    logger.info(
        f"=== eval_phrase_constrained.py | {args.testset} | "
        f"{args.src}->{args.tgt} | mode={args.mode} ==="
    )

    glossary = load_glossary(args.glossary)
    logger.info(f"Loaded {len(glossary)} glossary entries from {args.glossary}")

    testset = load_testset(args.testset, limit=args.limit)
    logger.info(f"Loaded {len(testset)} test sentences from {args.testset}")

    use_constraint = args.mode == "treatment"
    results = run_eval(
        testset,
        args.src,
        args.tgt,
        glossary,
        use_constraint=use_constraint,
        num_beams=args.num_beams,
        cap_terms=args.cap_terms,
    )

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
            "method": "phrase_constrained_beam_search" if use_constraint else "greedy_control",
            "num_beams": args.num_beams if use_constraint else 1,
            "cap_terms": args.cap_terms if use_constraint else None,
            "glossary": args.glossary,
            "n_sentences": len(testset),
        },
        "scores": scores,
        # Full hyp outputs (required for reviewer particle spot-check)
        "all_hyps": [r["hyp"] for r in results],
        "all_results": results,
    }

    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2))
    logger.info(f"Report saved -> {args.report}")

    # Print summary line
    print(
        f"\n[SUMMARY] mode={args.mode} | {args.src}->{args.tgt} | "
        f"testset={Path(args.testset).name}"
    )
    print(f"  BLEU={scores['bleu']} ({scores['bleu_tokenize']})")
    print(
        f"  Term-recall={scores['term_recall_pct']}% on "
        f"{scores['n_triggered']} triggered sents (of {scores['n_sentences']} total)"
    )
    print(f"  Avg latency={scores['avg_latency_ms']}ms")


if __name__ == "__main__":
    main()
