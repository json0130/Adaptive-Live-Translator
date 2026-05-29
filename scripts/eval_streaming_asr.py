#!/usr/bin/env python3
"""Evaluate LocalAgreement-2 streaming ASR on Fleurs.

Metrics reported:
  - WER (en) / WER_mecab + CER (ko) on the FINAL transcript
  - first_emission_latency: mean and p95 (seconds from audio start to first committed word)
  - total RTFx: sum(all window decode times) / sum(audio durations)
  - peak RSS (MB)
  - N utterances

Usage:
    PYTHONPATH=. python3 scripts/eval_streaming_asr.py \\
        --manifest data/eval/fleurs_en_us_test/manifest.tsv \\
        --lang en \\
        --report reports/exp_streaming-asr-localagreement/stream_en.json

    PYTHONPATH=. python3 scripts/eval_streaming_asr.py \\
        --manifest data/eval/fleurs_ko_kr_test/manifest.tsv \\
        --lang ko \\
        --report reports/exp_streaming-asr-localagreement/stream_ko.json
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np


# ── Tokenisation & scoring helpers ─────────────────────────────────────────

def _normalize_en(text: str) -> str:
    import re
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _tokenize_ko_mecab(text: str) -> list[str]:
    try:
        from sacrebleu.tokenizers.tokenizer_ko_mecab import TokenizerKoMecab
        return TokenizerKoMecab()(text).split()
    except Exception:
        return list(text)  # char fallback


def _wer(hyps: list[str], refs: list[str]) -> float:
    try:
        import jiwer
        return jiwer.wer(refs, hyps) * 100.0
    except ImportError:
        return -1.0


def _cer(hyps: list[str], refs: list[str]) -> float:
    try:
        import jiwer
        return jiwer.cer(refs, hyps) * 100.0
    except ImportError:
        return -1.0


def _ko_wer_mecab(hyps: list[str], refs: list[str]) -> float:
    hyp_tok = [" ".join(_tokenize_ko_mecab(h)) for h in hyps]
    ref_tok = [" ".join(_tokenize_ko_mecab(r)) for r in refs]
    return _wer(hyp_tok, ref_tok)


def _peak_ram_mb() -> float:
    ru = resource.getrusage(resource.RUSAGE_SELF)
    return ru.ru_maxrss / 1024.0  # Linux: KB -> MB


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--lang", required=True, choices=["en", "ko"])
    ap.add_argument("--report", required=True)
    ap.add_argument("--limit", type=int, default=0,
                    help="Cap at N utterances (0 = all). Use 100 for timed runs.")
    ap.add_argument("--model", default="Systran/faster-whisper-medium")
    ap.add_argument("--compute-type", default="int8")
    ap.add_argument("--cpu-threads", type=int, default=8)
    ap.add_argument("--initial-chunk-s", type=float, default=1.0,
                    help="Initial audio window size in seconds (default 1.0)")
    ap.add_argument("--step-s", type=float, default=0.5,
                    help="Step size in seconds per window (default 0.5)")
    ap.add_argument("--beam-size", type=int, default=1)
    args = ap.parse_args()

    # Load manifest
    manifest_path = Path(args.manifest)
    manifest_dir = manifest_path.parent
    rows = []
    with open(manifest_path) as f:
        header = f.readline().rstrip("\n").split("\t")
        for line in f:
            parts = line.rstrip("\n").split("\t")
            row = dict(zip(header, parts))
            row["audio_path"] = str(manifest_dir / row["audio_path"])
            rows.append(row)
    if args.limit and args.limit > 0:
        rows = rows[: args.limit]
    print(f"[stream_asr] manifest: {len(rows)} utterances, lang={args.lang}")

    # Import and initialise streamer
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.asr.streaming_local_agreement import LocalAgreementASR

    streamer = LocalAgreementASR(
        model_id=args.model,
        compute_type=args.compute_type,
        cpu_threads=args.cpu_threads,
        initial_chunk_s=args.initial_chunk_s,
        step_s=args.step_s,
        beam_size=args.beam_size,
    )
    # Warm up model load before timing loop
    streamer._ensure_loaded()
    print(f"[stream_asr] model loaded: {args.model}")

    samples = []
    total_audio_s = 0.0
    total_decode_s = 0.0
    first_latencies = []

    for i, row in enumerate(rows):
        audio_path = row["audio_path"]
        ref = row["text"]

        if not Path(audio_path).exists():
            print(f"[stream_asr] SKIP missing {audio_path}")
            continue

        result = streamer.transcribe_streaming(audio_path, lang=args.lang)

        total_audio_s += result.total_audio_s
        total_decode_s += result.total_decode_s

        if result.first_emission_latency_s is not None:
            first_latencies.append(result.first_emission_latency_s)

        samples.append({
            "id": row.get("id", str(i)),
            "hyp": result.final_transcript,
            "ref": ref,
            "audio_s": round(result.total_audio_s, 3),
            "decode_s": round(result.total_decode_s, 3),
            "n_windows": result.n_windows,
            "first_emission_latency_s": round(result.first_emission_latency_s, 3)
                if result.first_emission_latency_s is not None else None,
            "n_emissions": len(result.emissions),
        })

        if (i + 1) % 10 == 0:
            cum_rtfx = total_decode_s / total_audio_s if total_audio_s > 0 else -1
            lat_mean = float(np.mean(first_latencies)) if first_latencies else -1
            print(f"[stream_asr] {i+1}/{len(rows)}  RTFx={cum_rtfx:.3f}  "
                  f"lat_mean={lat_mean:.3f}s  ram={_peak_ram_mb():.0f}MB")

    # Compute aggregate metrics
    hyps = [s["hyp"] for s in samples]
    refs = [s["ref"] for s in samples]
    rtfx = total_decode_s / total_audio_s if total_audio_s > 0 else -1.0

    if args.lang == "ko":
        metric_scores = {
            "wer_ko_mecab": round(_ko_wer_mecab(hyps, refs), 2),
            "cer": round(_cer(hyps, refs), 2),
        }
    else:
        metric_scores = {
            "wer": round(_wer([_normalize_en(h) for h in hyps],
                               [_normalize_en(r) for r in refs]), 2),
        }

    lat_arr = np.array(first_latencies) if first_latencies else np.array([float("nan")])
    scores = {
        **metric_scores,
        "rtfx_total": round(rtfx, 3),
        "first_emission_latency_mean_s": round(float(np.mean(lat_arr)), 3),
        "first_emission_latency_p95_s": round(float(np.percentile(lat_arr, 95)), 3),
        "first_emission_latency_p50_s": round(float(np.median(lat_arr)), 3),
        "total_audio_s": round(total_audio_s, 1),
        "total_decode_s": round(total_decode_s, 1),
        "ram_peak_mb": round(_peak_ram_mb(), 1),
        "n_utterances": len(samples),
        "lang": args.lang,
        "model": args.model,
        "compute_type": args.compute_type,
        "streaming_config": {
            "initial_chunk_s": args.initial_chunk_s,
            "step_s": args.step_s,
            "beam_size": args.beam_size,
            "agreement_n": 2,
        },
    }

    print("\n[stream_asr] ── Results ─────────────────────────────────────")
    for k, v in scores.items():
        print(f"  {k:<40} {v}")
    print("[stream_asr] ──────────────────────────────────────────────────")

    # Gate checks
    lat_mean = scores["first_emission_latency_mean_s"]
    rtfx_val = scores["rtfx_total"]
    ram_mb = scores["ram_peak_mb"]
    print("\n[stream_asr] GATE CHECKS:")
    print(f"  first_emission_latency_mean <= 3.5s : {'PASS' if lat_mean <= 3.5 else 'FAIL'} ({lat_mean:.3f}s)")
    print(f"  total RTFx <= 1.5               : {'PASS' if rtfx_val <= 1.5 else 'FAIL'} ({rtfx_val:.3f})")
    print(f"  peak RSS <= 2560 MB             : {'PASS' if ram_mb <= 2560 else 'FAIL'} ({ram_mb:.0f} MB)")
    if args.lang == "en":
        wer = scores.get("wer", -1)
        print(f"  WER <= 10.5 (en)               : {'PASS' if wer <= 10.5 else 'FAIL'} ({wer:.2f})")
    else:
        ko_wer = scores.get("wer_ko_mecab", -1)
        print(f"  WER_mecab <= 15.9 (ko)         : {'PASS' if ko_wer <= 15.9 else 'FAIL'} ({ko_wer:.2f})")

    out = {
        "args": vars(args),
        "scores": scores,
        "samples": samples[:20],
    }
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"\n[stream_asr] report → {args.report}")


if __name__ == "__main__":
    main()
