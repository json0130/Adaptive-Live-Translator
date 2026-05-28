#!/usr/bin/env python
"""Cloud-translation BLEU reference for the comparison table.

Calls Google Translate (via deep-translator's unofficial endpoint) on a
TSV test set, then scores with the locked eval protocol (sacrebleu
ko-mecab for ko, 13a for en). Provides a "ceiling" reference number so
we can see how far the CPU pipeline sits below best-available.

NOT a fair head-to-head experiment:
- This uses a cloud service (online, paid in production, ToS-gray
  unofficial endpoint here).
- No streaming, no chunk policy, no glossary.
- Listed in the comparison table for context only.

Usage:
    python3 scripts/eval_cloud_baseline.py \\
        --testset data/eval/flores_devtest_en_ko.tsv \\
        --src en --tgt ko \\
        --report reports/cloud_baseline/en-ko.json \\
        --limit 0     # 0 = full set

If you hit rate limits the script sleeps and retries; if blocked the
script stops cleanly and the report records the partial result.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def _translate(text: str, src: str, tgt: str, attempts: int = 5) -> str:
    from deep_translator import GoogleTranslator
    # deep-translator uses ISO codes en/ko directly
    err = None
    for n in range(attempts):
        try:
            return GoogleTranslator(source=src, target=tgt).translate(text)
        except Exception as e:
            err = e
            time.sleep(2 ** n)
    raise RuntimeError(f"cloud translate failed after {attempts}: {err}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--testset", required=True)
    ap.add_argument("--src", required=True, choices=["en", "ko"])
    ap.add_argument("--tgt", required=True, choices=["en", "ko"])
    ap.add_argument("--report", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--sleep", type=float, default=0.05,
                    help="Seconds between requests (be polite)")
    args = ap.parse_args()

    rows: list[tuple[str, str]] = []
    for line in Path(args.testset).read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.split("\t")
        rows.append((parts[0].strip(), parts[1].strip() if len(parts) > 1 else ""))
    if args.limit and args.limit > 0:
        rows = rows[: args.limit]
    print(f"[cloud] {len(rows)} sentences, {args.src} -> {args.tgt}")

    results = []
    n_done = 0
    for i, (src, ref) in enumerate(rows):
        try:
            t0 = time.perf_counter()
            hyp = _translate(src, args.src, args.tgt)
            dt = (time.perf_counter() - t0) * 1000
            results.append({"src": src, "hyp": hyp, "ref": ref, "latency_ms": round(dt, 1)})
            n_done += 1
            if (i + 1) % 50 == 0:
                print(f"[cloud]  {i+1}/{len(rows)} done")
            time.sleep(args.sleep)
        except KeyboardInterrupt:
            print("[cloud] interrupted; writing partial report")
            break
        except Exception as e:
            print(f"[cloud] row {i} failed permanently: {e}; stopping early")
            break

    # Score with the locked protocol
    from src.utils.metrics import compute_bleu
    hyps = [r["hyp"] for r in results]
    refs = [r["ref"] for r in results]
    tokenize = "ko-mecab" if args.tgt == "ko" else "13a"
    if any(refs):
        bleu, tok_used = compute_bleu(hyps, refs, tokenize=tokenize)
    else:
        bleu, tok_used = -1.0, "n/a"

    avg_latency = sum(r["latency_ms"] for r in results) / max(len(results), 1)
    scores = {
        "bleu": round(bleu, 2),
        "bleu_tokenize": tok_used,
        "avg_latency_ms": round(avg_latency, 1),
        "n_segments": n_done,
        "src": args.src,
        "tgt": args.tgt,
        "provider": "google-translate (deep-translator unofficial)",
    }
    print("[cloud] ── Results ─────────────")
    for k, v in scores.items():
        print(f"  {k:<20} {v}")
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(
        json.dumps({"scores": scores, "samples": results[:20]},
                   ensure_ascii=False, indent=2)
    )
    print(f"[cloud] report → {args.report}")


if __name__ == "__main__":
    main()
