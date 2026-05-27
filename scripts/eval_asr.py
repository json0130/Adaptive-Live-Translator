#!/usr/bin/env python
"""ASR-only eval on Fleurs audio. Reports WER (en) / CER+WER (ko, mecab-ko).

Usage:
    python3 scripts/eval_asr.py \\
        --manifest data/eval/fleurs_en_us_test/manifest.tsv \\
        --lang en \\
        --model openai/whisper-medium \\
        --backend faster-whisper \\
        --compute-type int8 \\
        --report reports/<branch>/asr_en.json

Backends:
  - faster-whisper (CTranslate2 int8 on CPU)
  - whisper.cpp (subprocess to compiled main; not implemented here)

Output JSON:
  {scores: {wer, cer, rtfx, ram_peak_mb, n}, samples: [...]}
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import time
from pathlib import Path


def _tokenize_ko_mecab(text: str) -> list[str]:
    """Use the same Korean tokenizer sacrebleu uses for ko-mecab."""
    try:
        from sacrebleu.tokenizers.tokenizer_ko_mecab import TokenizerKoMecab
        return TokenizerKoMecab()(text).split()
    except Exception:
        # Char fallback. Reset the comparison table if you hit this.
        return list(text)


def _normalize_en(text: str) -> str:
    import re
    text = text.lower()
    text = re.sub(r"[^a-z0-9가-힣\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _wer(hyps: list[str], refs: list[str]) -> float:
    try:
        import jiwer
    except ImportError:
        return -1.0
    return jiwer.wer(refs, hyps) * 100.0


def _cer(hyps: list[str], refs: list[str]) -> float:
    try:
        import jiwer
    except ImportError:
        return -1.0
    return jiwer.cer(refs, hyps) * 100.0


def _ko_wer_mecab(hyps: list[str], refs: list[str]) -> float:
    """Token-level WER using mecab-ko tokenization."""
    hyp_tok = [" ".join(_tokenize_ko_mecab(h)) for h in hyps]
    ref_tok = [" ".join(_tokenize_ko_mecab(r)) for r in refs]
    return _wer(hyp_tok, ref_tok)


def _peak_ram_mb() -> float:
    ru = resource.getrusage(resource.RUSAGE_SELF)
    # On Linux ru_maxrss is in KB.
    return ru.ru_maxrss / 1024.0


def _run_fw(model_id: str, compute_type: str, manifest: list[dict], lang: str):
    from faster_whisper import WhisperModel
    print(f"[asr] loading faster-whisper {model_id} compute_type={compute_type}")
    model = WhisperModel(model_id, device="cpu", compute_type=compute_type)
    results: list[dict] = []
    total_audio_s = 0.0
    total_decode_s = 0.0
    for i, row in enumerate(manifest):
        audio_path = row["audio_path"]
        t0 = time.perf_counter()
        segs, info = model.transcribe(
            audio_path,
            language=lang,
            beam_size=1,
            vad_filter=False,
        )
        hyp = "".join(s.text for s in segs).strip()
        dec_s = time.perf_counter() - t0
        audio_s = float(info.duration)
        total_audio_s += audio_s
        total_decode_s += dec_s
        results.append({
            "id": row["id"],
            "hyp": hyp,
            "ref": row["text"],
            "audio_s": audio_s,
            "decode_s": dec_s,
        })
        if (i + 1) % 25 == 0:
            print(f"[asr]   {i+1}/{len(manifest)}  cum RTFx={total_decode_s/total_audio_s:.3f}")
    return results, total_audio_s, total_decode_s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--lang", required=True, choices=["en", "ko"])
    ap.add_argument("--model", required=True,
                    help="HF model id (e.g. openai/whisper-medium, openai/whisper-large-v3-turbo) "
                         "or local CT2 dir")
    ap.add_argument("--backend", default="faster-whisper", choices=["faster-whisper"])
    ap.add_argument("--compute-type", default="int8",
                    choices=["int8", "int8_float16", "int8_float32", "int16", "float32"])
    ap.add_argument("--report", required=True)
    ap.add_argument("--limit", type=int, default=0,
                    help="Optional: cap at N utterances for smoke runs")
    args = ap.parse_args()

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
    print(f"[asr] manifest: {len(rows)} utterances")

    results, total_audio_s, total_decode_s = _run_fw(
        args.model, args.compute_type, rows, args.lang
    )

    hyps = [r["hyp"] for r in results]
    refs = [r["ref"] for r in results]
    rtfx = total_decode_s / total_audio_s if total_audio_s > 0 else -1.0

    if args.lang == "ko":
        scores = {
            "wer_ko_mecab": round(_ko_wer_mecab(hyps, refs), 2),
            "cer": round(_cer(hyps, refs), 2),
        }
    else:
        scores = {
            "wer": round(_wer([_normalize_en(h) for h in hyps],
                              [_normalize_en(r) for r in refs]), 2),
        }
    scores.update({
        "rtfx": round(rtfx, 3),
        "total_audio_s": round(total_audio_s, 1),
        "total_decode_s": round(total_decode_s, 1),
        "ram_peak_mb": round(_peak_ram_mb(), 1),
        "n_utterances": len(results),
        "model": args.model,
        "compute_type": args.compute_type,
        "backend": args.backend,
        "lang": args.lang,
    })

    print("[asr] ── Results ────────────────────────────────")
    for k, v in scores.items():
        print(f"  {k:<20} {v}")
    print("[asr] ────────────────────────────────────────────")

    out = {"args": vars(args), "scores": scores, "samples": results[:20]}
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"[asr] report → {args.report}")


if __name__ == "__main__":
    main()
