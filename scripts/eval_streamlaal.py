#!/usr/bin/env python
"""Offline evaluation: BLEU + StreamLAAL against a held-out test set.

Test set TSV format (tab-separated):
    src_text  \\t  ref_tgt_text  \\t  src_audio_path (optional)

Usage:
    python scripts/eval_streamlaal.py \\
        --testset data/eval/en-ko-dev.tsv \\
        --src en --tgt ko \\
        --config configs/default.yaml \\
        --report reports/eval_$(date +%Y%m%d).json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

import yaml
from loguru import logger


async def _run_one(
    src_text: str,
    ref_text: str,
    session,
    src_lang: str,
    tgt_lang: str,
) -> dict:
    system_prompt = session.build_system_prompt(src_text)

    async def _src_iter():
        yield src_text

    hyp = ""
    t0 = time.perf_counter()
    async for chunk in session.translator.translate_stream(
        _src_iter(),
        src_lang=src_lang,
        tgt_lang=tgt_lang,
        system_prompt=system_prompt,
    ):
        hyp = chunk.text
    latency_ms = (time.perf_counter() - t0) * 1000

    return {"src": src_text, "hyp": hyp, "ref": ref_text, "latency_ms": latency_ms}


async def evaluate(
    testset_path: Path,
    src_lang: str,
    tgt_lang: str,
    cfg: dict,
) -> list[dict]:
    from src.pipeline.session import SessionConfig, TranslationSession

    session_cfg = SessionConfig(src_lang=src_lang, tgt_lang=tgt_lang)
    session = TranslationSession(cfg, session_cfg)

    results = []
    rows = [
        line.split("\t")
        for line in testset_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]

    for i, row in enumerate(rows):
        src_text = row[0].strip()
        ref_text = row[1].strip() if len(row) > 1 else ""
        try:
            rec = await _run_one(src_text, ref_text, session, src_lang, tgt_lang)
            results.append(rec)
            if (i + 1) % 10 == 0:
                logger.info(f"  {i+1}/{len(rows)} done")
        except Exception as exc:
            logger.warning(f"Row {i} failed: {exc}")

    session.close()
    return results


def compute_scores(results: list[dict]) -> dict:
    from src.utils.metrics import compute_bleu, compute_streamlaal, SegmentRecord

    hyps = [r["hyp"] for r in results]
    refs = [r["ref"] for r in results]
    bleu = compute_bleu(hyps, refs) if any(refs) else -1.0

    records = [
        SegmentRecord(
            src=r["src"],
            tgt=r["hyp"],
            src_end_ms=0,
            tgt_emit_ms=int(r["latency_ms"]),
            ref_tgt=r["ref"],
        )
        for r in results
    ]
    laal = compute_streamlaal(records)
    avg_latency = sum(r["latency_ms"] for r in results) / max(len(results), 1)

    return {
        "bleu": round(bleu, 2),
        "stream_laal_s": round(laal, 3),
        "avg_latency_ms": round(avg_latency, 1),
        "n_segments": len(results),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--testset", required=True)
    parser.add_argument("--src", default="en")
    parser.add_argument("--tgt", default="ko")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--report", default="reports/eval.json")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    testset_path = Path(args.testset)
    if not testset_path.exists():
        logger.error(f"Test set not found: {testset_path}")
        return

    logger.info(f"Evaluating {testset_path.name} | {args.src} → {args.tgt}")
    results = asyncio.run(
        evaluate(testset_path, args.src, args.tgt, cfg)
    )
    scores = compute_scores(results)

    logger.info("── Evaluation Results ──────────────────")
    for k, v in scores.items():
        logger.info(f"  {k:<25} {v}")
    logger.info("────────────────────────────────────────")

    report = {"config": args.__dict__, "scores": scores, "samples": results[:20]}
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2))
    logger.info(f"Report saved → {args.report}")


if __name__ == "__main__":
    main()
