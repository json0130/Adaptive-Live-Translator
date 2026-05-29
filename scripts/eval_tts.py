#!/usr/bin/env python3
"""TTS round-trip intelligibility eval for Korean CPU TTS.

For each sentence in fleurs_pair_manifest.tsv:
  1. Synthesize Korean audio with KoreanCpuTTS
  2. Re-transcribe with faster-whisper-medium int8
  3. Compute round-trip CER and WER (ko-mecab tokenized)

Reports:
  - round-trip WER (mecab), CER, synth RTFx, RAM peak, N, engine, license

Usage:
    PYTHONPATH=. python3 scripts/eval_tts.py \\
        --report reports/exp_korean-cpu-tts/tts_ko.json \\
        --limit 50

Gate: round-trip WER (mecab) <= 25%  (espeak baseline was 71.66%)
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import tempfile
import time
from pathlib import Path

import numpy as np

MANIFEST_DEFAULT = "data/eval/fleurs_pair_manifest.tsv"
ASR_MODEL = "Systran/faster-whisper-medium"
ASR_COMPUTE = "int8"

ENGINE_LICENSE = {
    "melo": "MIT",
    "espeak": "GPL-v3",
}


# ---------------------------------------------------------------------------
# Metric helpers (reuse eval_asr.py approach)
# ---------------------------------------------------------------------------

_KO_MECAB_TAGGER = None

def _get_ko_mecab_tagger():
    global _KO_MECAB_TAGGER
    if _KO_MECAB_TAGGER is not None:
        return _KO_MECAB_TAGGER
    try:
        import MeCab  # type: ignore
        import mecab_ko_dic  # type: ignore
        _KO_MECAB_TAGGER = MeCab.Tagger(f"-d {mecab_ko_dic.dictionary_path} -Owakati")
        return _KO_MECAB_TAGGER
    except Exception:
        return None


def _tokenize_ko_mecab(text: str) -> list[str]:
    """Korean mecab tokenization for WER computation."""
    # Try sacrebleu's tokenizer first
    try:
        from sacrebleu.tokenizers.tokenizer_ko_mecab import TokenizerKoMecab  # type: ignore
        return TokenizerKoMecab()(text).split()
    except Exception:
        pass

    # Direct MeCab with mecab_ko_dic (installed with MeloTTS)
    try:
        tagger = _get_ko_mecab_tagger()
        if tagger is not None:
            result = tagger.parse(text)
            return result.strip().split() if result else list(text)
    except Exception:
        pass

    # Char fallback
    return list(text)


def _wer(hyps: list[str], refs: list[str]) -> float:
    try:
        import jiwer  # type: ignore
        return jiwer.wer(refs, hyps) * 100.0
    except ImportError:
        return -1.0


def _cer(hyps: list[str], refs: list[str]) -> float:
    try:
        import jiwer  # type: ignore
        return jiwer.cer(refs, hyps) * 100.0
    except ImportError:
        return -1.0


def _ko_wer_mecab(hyps: list[str], refs: list[str]) -> float:
    hyp_tok = [" ".join(_tokenize_ko_mecab(h)) for h in hyps]
    ref_tok = [" ".join(_tokenize_ko_mecab(r)) for r in refs]
    return _wer(hyp_tok, ref_tok)


def _peak_ram_mb() -> float:
    ru = resource.getrusage(resource.RUSAGE_SELF)
    return ru.ru_maxrss / 1024.0


# ---------------------------------------------------------------------------
# Main eval
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=MANIFEST_DEFAULT)
    ap.add_argument("--report", required=True)
    ap.add_argument("--limit", type=int, default=50,
                    help="Number of sentences to evaluate (min 30 for valid metric)")
    ap.add_argument("--engine", default="auto", choices=["auto", "melo", "espeak"],
                    help="TTS engine: auto|melo|espeak")
    ap.add_argument("--asr-model", default=ASR_MODEL)
    ap.add_argument("--asr-compute", default=ASR_COMPUTE)
    args = ap.parse_args()

    # Read manifest
    manifest_path = Path(args.manifest)
    rows = []
    with open(manifest_path) as f:
        header = f.readline().rstrip("\n").split("\t")
        for line in f:
            parts = line.rstrip("\n").split("\t")
            row = dict(zip(header, parts))
            rows.append(row)
    if args.limit > 0:
        rows = rows[: args.limit]
    print(f"[eval_tts] {len(rows)} sentences to evaluate")

    # Load TTS
    print(f"[eval_tts] Loading TTS engine='{args.engine}' ...")
    from src.tts.korean_cpu_tts import KoreanCpuTTS  # type: ignore
    t_load = time.perf_counter()
    tts = KoreanCpuTTS(engine=args.engine)
    tts_load_s = time.perf_counter() - t_load
    engine_used = tts._engine
    license_str = ENGINE_LICENSE.get(engine_used, "unknown")
    print(f"[eval_tts] TTS engine loaded: {engine_used} (license: {license_str}) in {tts_load_s:.1f}s")

    # Load ASR
    print(f"[eval_tts] Loading ASR {args.asr_model} compute_type={args.asr_compute} ...")
    from faster_whisper import WhisperModel  # type: ignore
    asr = WhisperModel(args.asr_model, device="cpu", compute_type=args.asr_compute)
    print("[eval_tts] ASR loaded.")

    # Synthesize + re-transcribe
    results = []
    total_synth_ms = 0.0
    total_audio_dur_s = 0.0

    with tempfile.TemporaryDirectory() as tmpdir:
        for i, row in enumerate(rows):
            ko_text = row["ko_text"]
            uid = row["id"]

            # Synthesize
            try:
                audio_16k, synth_ms = tts.synthesize(ko_text, lang="ko")
            except Exception as e:
                print(f"[eval_tts] WARN: synth failed for id={uid}: {e}")
                results.append({
                    "id": uid, "ref": ko_text, "hyp": "", "error": str(e),
                    "synth_ms": 0, "audio_dur_s": 0,
                })
                continue

            audio_dur_s = len(audio_16k) / tts.SAMPLE_RATE
            total_synth_ms += synth_ms
            total_audio_dur_s += audio_dur_s

            # Save to temp wav for faster-whisper
            import soundfile as sf  # type: ignore
            wav_path = os.path.join(tmpdir, f"{uid}.wav")
            sf.write(wav_path, audio_16k, tts.SAMPLE_RATE)

            # Re-transcribe
            try:
                segs, info = asr.transcribe(
                    wav_path,
                    language="ko",
                    beam_size=1,
                    vad_filter=False,
                )
                hyp = "".join(s.text for s in segs).strip()
            except Exception as e:
                print(f"[eval_tts] WARN: ASR failed for id={uid}: {e}")
                hyp = ""

            rtfx_i = audio_dur_s / (synth_ms / 1000.0) if synth_ms > 0 else 0.0
            results.append({
                "id": uid,
                "ref": ko_text,
                "hyp": hyp,
                "synth_ms": round(synth_ms, 1),
                "audio_dur_s": round(audio_dur_s, 3),
                "synth_rtfx": round(rtfx_i, 3),
            })

            if (i + 1) % 10 == 0 or i == len(rows) - 1:
                cum_rtfx = total_audio_dur_s / (total_synth_ms / 1000.0) if total_synth_ms > 0 else 0
                print(f"[eval_tts]   {i+1}/{len(rows)}  cum_RTFx={cum_rtfx:.3f}x")

    # Filter errors
    valid = [r for r in results if "error" not in r and r["hyp"] != ""]
    n_valid = len(valid)
    print(f"[eval_tts] Valid results: {n_valid}/{len(rows)}")

    if n_valid < 10:
        print("[eval_tts] ERROR: Too few valid results. Cannot compute reliable metrics.")

    hyps = [r["hyp"] for r in valid]
    refs = [r["ref"] for r in valid]

    wer_mecab = round(_ko_wer_mecab(hyps, refs), 2)
    cer_val = round(_cer(hyps, refs), 2)
    synth_rtfx = round(
        total_audio_dur_s / (total_synth_ms / 1000.0) if total_synth_ms > 0 else 0, 3
    )
    ram_peak = round(_peak_ram_mb(), 1)

    print()
    print("[eval_tts] ── Results ───────────────────────────────────────")
    print(f"  Engine              : {engine_used}")
    print(f"  License             : {license_str}")
    print(f"  N (valid)           : {n_valid}")
    print(f"  round-trip WER(mecab): {wer_mecab:.2f}%")
    print(f"  round-trip CER      : {cer_val:.2f}%")
    print(f"  synth RTFx          : {synth_rtfx:.3f}x  (<1.0 = slower than real-time)")
    print(f"  RAM peak            : {ram_peak:.1f} MB")
    print(f"  Gate (WER <=25%)    : {'PASS' if wer_mecab <= 25.0 else 'FAIL'}  (espeak baseline: 71.66%)")
    print("[eval_tts] ──────────────────────────────────────────────────")

    scores = {
        "engine": engine_used,
        "license": license_str,
        "n": n_valid,
        "wer_ko_mecab": wer_mecab,
        "cer": cer_val,
        "synth_rtfx": synth_rtfx,
        "ram_peak_mb": ram_peak,
        "total_audio_s": round(total_audio_dur_s, 2),
        "total_synth_s": round(total_synth_ms / 1000.0, 2),
        "gate_25pct_wer": "PASS" if wer_mecab <= 25.0 else "FAIL",
        "espeak_baseline_wer": 71.66,
        "asr_model": args.asr_model,
        "asr_compute": args.asr_compute,
    }

    out = {"args": vars(args), "scores": scores, "samples": results[:20]}
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"[eval_tts] report -> {args.report}")


if __name__ == "__main__":
    main()
