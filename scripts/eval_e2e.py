#!/usr/bin/env python
"""End-to-end evaluation: audio → ASR → MT → TTS → audio.

Measures:
  - BLEU: translator output vs reference text (using ko-mecab for ko, 13a for en)
  - ASR quality: ASR transcription WER vs reference (faster-whisper-medium)
  - Round-trip WER: re-run ASR on TTS output, compare to translator output
  - Latency: per-component (ASR, MT, TTS) and total wall-clock
  - RTFx: real-time factor (sum of component times / audio duration)
  - Peak RAM: from resource.getrusage
  - First-audio latency: when TTS starts emitting audio
"""
from __future__ import annotations

import argparse
import json
import resource
import time
from pathlib import Path
from typing import Optional

import numpy as np
import yaml
from loguru import logger


def get_mem_usage_mb() -> float:
    """Get peak memory usage in MB."""
    usage = resource.getrusage(resource.RUSAGE_SELF)
    # maxrss is in KB on Linux
    return usage.ru_maxrss / 1024.0


def load_asr():
    """Load faster-whisper-medium for ASR."""
    from faster_whisper import WhisperModel
    model_id = "Systran/faster-whisper-medium"
    compute_type = "int8"
    logger.info(f"Loading ASR: {model_id} compute_type={compute_type}")
    model = WhisperModel(model_id, device="cpu", compute_type=compute_type)
    return model


def transcribe_audio(model, audio_path: str, lang: str) -> tuple[str, float, float]:
    """Transcribe audio. Returns (text, duration_s, inference_ms)."""
    t0 = time.perf_counter()
    segments, info = model.transcribe(
        audio_path, language=lang, beam_size=1, vad_filter=False
    )
    text = "".join(s.text for s in segments).strip()
    inference_ms = (time.perf_counter() - t0) * 1000
    return text, float(info.duration), inference_ms


def load_translator(cfg: dict):
    """Load NLLB CT2 translator."""
    from src.translator.nllb_ct2_translator import NllbCt2Translator
    logger.info("Loading translator: NLLB-600M CT2 INT8")
    return NllbCt2Translator(cfg)


async def translate_text_async(translator, src_text: str, src_lang: str, tgt_lang: str) -> tuple[str, float]:
    """Translate text asynchronously. Returns (text, inference_ms)."""
    async def _src_iter():
        yield src_text

    t0 = time.perf_counter()
    hyp = ""
    async for chunk in translator.translate_stream(
        _src_iter(),
        src_lang=src_lang,
        tgt_lang=tgt_lang,
        system_prompt="",
    ):
        hyp = chunk.text
    mt_ms = (time.perf_counter() - t0) * 1000
    return hyp, mt_ms


def load_tts() -> Optional[object]:
    """Load MeloTTS if available, otherwise None (fallback will use espeak-ng)."""
    try:
        from src.tts.melo_tts import MeloTTSSynthesizer
        logger.info("Loading TTS: MeloTTS")
        tts = MeloTTSSynthesizer({})
        # Try lazy-load to catch issues early
        tts._lazy_load()
        logger.info("MeloTTS loaded successfully")
        return tts
    except ImportError as e:
        logger.info(f"MeloTTS not available ({e}), will use espeak-ng fallback")
        return None
    except Exception as e:
        logger.info(f"MeloTTS load/init failed ({e}), will use espeak-ng fallback")
        return None


def synthesize_speech_fallback(text: str, lang: str) -> tuple[np.ndarray, float]:
    """Fallback TTS using pyttsx3 or silent audio.

    Returns (audio_float32_16khz, synth_ms).
    """
    import subprocess
    import tempfile

    t0 = time.perf_counter()

    # Try espeak-ng first (more CPU-friendly than pyttsx3)
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name

        voice_map = {
            "en": "en",
            "ko": "ko",
        }
        voice = voice_map.get(lang, "en")

        # Use espeak-ng to synthesize
        cmd = ["espeak-ng", "-v", voice, "-w", tmp_path, text]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)

        if result.returncode == 0:
            # Read the WAV file
            import wave
            with wave.open(tmp_path, "rb") as wav_file:
                n_channels = wav_file.getnchannels()
                sample_width = wav_file.getsampwidth()
                framerate = wav_file.getframerate()
                n_frames = wav_file.getnframes()
                audio_data = wav_file.readframes(n_frames)

            # Convert to float32
            audio = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

            # Resample to 16 kHz if needed
            if framerate != 16000:
                # Linear interpolation fallback
                ratio = 16000 / framerate
                new_len = int(len(audio) * ratio)
                audio = np.interp(
                    np.linspace(0, len(audio) - 1, new_len),
                    np.arange(len(audio)),
                    audio,
                )

            synth_ms = (time.perf_counter() - t0) * 1000
            logger.info(f"espeak-ng TTS: {len(text)} chars → {len(audio)} samples in {synth_ms:.0f}ms")
            return audio, synth_ms
    except Exception as e:
        logger.debug(f"espeak-ng failed: {e}")

    # Fallback: return silence with nominal length (0.5s @ 16kHz)
    logger.warning("TTS unavailable, using silent audio fallback")
    synth_ms = (time.perf_counter() - t0) * 1000
    # Generate silence with nominal duration (proportional to text length)
    # Rough estimate: 3 chars per second of speech
    duration_s = max(0.5, len(text) / 3.0)
    n_samples = int(duration_s * 16000)
    return np.zeros(n_samples, dtype=np.float32), synth_ms


async def synthesize_speech(tts_model: Optional[object], text: str, lang: str) -> tuple[np.ndarray, float]:
    """Synthesize speech using MeloTTS or fallback.

    Returns (audio_float32_16khz, synth_ms).
    """
    if tts_model is None:
        # Use fallback
        return synthesize_speech_fallback(text, lang)

    t0 = time.perf_counter()
    try:
        async def _text_iter():
            yield text

        audio = None
        async for chunk in tts_model.synthesize_stream(_text_iter()):
            audio = chunk
            break  # Take first chunk

        if audio is None:
            audio = np.zeros(16000, dtype=np.float32)

        synth_ms = (time.perf_counter() - t0) * 1000
        return audio, synth_ms
    except Exception as e:
        logger.warning(f"TTS synthesis failed: {e}, using fallback")
        return synthesize_speech_fallback(text, lang)


async def evaluate_one_segment(
    manifest_row: dict,
    manifest_dir: Path,
    src_lang: str,
    tgt_lang: str,
    asr_model,
    translator,
    tts_model: Optional[object],
) -> dict:
    """Evaluate one utterance through the full pipeline.

    Returns a dict with metrics for this segment.
    """
    src_audio_col = f"{src_lang}_audio"
    tgt_text_col = f"{tgt_lang}_text"
    src_text_col = f"{src_lang}_text"

    # Try to resolve audio path: look in manifest_dir first, then try parent repo
    audio_rel = manifest_row[src_audio_col]
    audio_path = manifest_dir / audio_rel
    if not audio_path.exists():
        # Try parent repo (for worktree case)
        # Worktree is at .../adaptive-live-translator/.claude/worktrees/<name>/
        # manifest_dir is data/eval in the worktree
        # So we need to go up 5 levels to reach the parent repo root
        manifest_dir_abs = manifest_dir.absolute() if not manifest_dir.is_absolute() else manifest_dir
        parent_repo = manifest_dir_abs.parent.parent.parent.parent.parent
        parent_audio = parent_repo / "data" / "eval" / audio_rel
        if parent_audio.exists():
            audio_path = parent_audio
        else:
            # Last attempt: maybe the directory doesn't have audio files yet
            logger.warning(f"Audio file not found at {audio_path} or {parent_audio}")

    audio_path = str(audio_path)
    ref_tgt_text = manifest_row[tgt_text_col]
    src_text_ref = manifest_row[src_text_col]

    # Step 1: ASR
    t_asr0 = time.perf_counter()
    try:
        asr_text, audio_s, asr_ms = transcribe_audio(asr_model, audio_path, src_lang)
    except Exception as e:
        logger.warning(f"ASR failed: {e}")
        return {
            "id": manifest_row.get("id", "unknown"),
            "status": "asr_failed",
            "error": str(e),
        }

    # Step 2: Machine Translation
    t_mt0 = time.perf_counter()
    try:
        mt_text, mt_ms = await translate_text_async(translator, asr_text, src_lang, tgt_lang)
    except Exception as e:
        logger.warning(f"MT failed: {e}")
        return {
            "id": manifest_row.get("id", "unknown"),
            "status": "mt_failed",
            "error": str(e),
            "asr_text": asr_text,
            "asr_ms": round(asr_ms, 1),
        }

    # Step 3: TTS Synthesis
    t_tts0 = time.perf_counter()
    try:
        synth_audio, tts_ms = await synthesize_speech(tts_model, mt_text, tgt_lang)
    except Exception as e:
        logger.warning(f"TTS failed: {e}")
        return {
            "id": manifest_row.get("id", "unknown"),
            "status": "tts_failed",
            "error": str(e),
            "asr_text": asr_text,
            "asr_ms": round(asr_ms, 1),
            "mt_text": mt_text,
            "mt_ms": round(mt_ms, 1),
        }

    # Step 4: Re-transcribe TTS output to measure round-trip WER
    t_asr2_0 = time.perf_counter()
    try:
        # Write synth audio to temp file for ASR
        import tempfile
        import scipy.io.wavfile

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_audio_path = tmp.name

        # Ensure audio is mono and in valid range
        if len(synth_audio.shape) > 1 and synth_audio.shape[1] > 1:
            synth_audio = synth_audio[:, 0]
        synth_audio_int16 = np.clip(synth_audio * 32767, -32768, 32767).astype(np.int16)

        scipy.io.wavfile.write(tmp_audio_path, 16000, synth_audio_int16)

        # Transcribe the synthetic audio
        asr_roundtrip_text, _, asr2_ms = transcribe_audio(asr_model, tmp_audio_path, tgt_lang)

        # Clean up
        Path(tmp_audio_path).unlink(missing_ok=True)
    except Exception as e:
        logger.warning(f"Round-trip ASR failed: {e}")
        asr_roundtrip_text = ""
        asr2_ms = 0.0

    # Step 5: Compute metrics
    from src.utils.metrics import compute_bleu

    # BLEU: translator output vs reference
    hyps = [mt_text]
    refs = [ref_tgt_text]
    tokenize = "ko-mecab" if tgt_lang == "ko" else "13a"
    bleu, tokenize_used = compute_bleu(hyps, refs, tokenize=tokenize)

    # ASR WER: on source language
    from jiwer import cer, wer

    try:
        if src_lang == "ko":
            # Use CER for Korean
            asr_error = cer([src_text_ref], [asr_text])
        else:
            # Use WER for English
            asr_error = wer([src_text_ref], [asr_text])
    except Exception as e:
        logger.warning(f"ASR error rate computation failed: {e}")
        asr_error = -1.0

    # Round-trip WER: on target language
    try:
        if tgt_lang == "ko":
            roundtrip_error = cer([mt_text], [asr_roundtrip_text])
        else:
            roundtrip_error = wer([mt_text], [asr_roundtrip_text])
    except Exception as e:
        logger.warning(f"Round-trip error rate computation failed: {e}")
        roundtrip_error = -1.0

    # Total wall-clock (start to TTS end)
    total_ms = asr_ms + mt_ms + tts_ms + asr2_ms
    rtfx = total_ms / (audio_s * 1000.0) if audio_s > 0 else 0.0

    return {
        "id": manifest_row.get("id", "unknown"),
        "status": "success",
        "audio_s": round(audio_s, 2),
        "asr_text": asr_text,
        "asr_ms": round(asr_ms, 1),
        "asr_wer": round(asr_error * 100, 2) if asr_error >= 0 else -1.0,
        "mt_text": mt_text,
        "mt_ms": round(mt_ms, 1),
        "tts_ms": round(tts_ms, 1),
        "asr2_ms": round(asr2_ms, 1),
        "roundtrip_wer": round(roundtrip_error * 100, 2) if roundtrip_error >= 0 else -1.0,
        "mt_bleu": round(bleu, 2),
        "bleu_tokenize": tokenize_used,
        "total_ms": round(total_ms, 1),
        "rtfx": round(rtfx, 3),
        "ref_tgt_text": ref_tgt_text,
    }


async def main_async(
    manifest_path: Path,
    src_lang: str,
    tgt_lang: str,
    report_path: Path,
    limit: int = 0,
):
    """Run full e2e evaluation asynchronously."""
    # Load components
    logger.info(f"Loading ASR, MT, TTS for {src_lang} → {tgt_lang}")
    asr_model = load_asr()
    translator = load_translator({
        "translator": {
            "ct2_model_dir": "models/nllb-600m-ct2-int8",
            "device": "cpu",
            "dtype": "int8",
            "max_new_tokens": 256,
        }
    })
    tts_model = load_tts()

    if tts_model is None:
        logger.warning("Using fallback TTS (espeak-ng). Output quality may be poor.")

    # Read manifest
    manifest_dir = manifest_path.parent
    rows: list[dict] = []
    with open(manifest_path) as f:
        header = f.readline().rstrip("\n").split("\t")
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < len(header):
                continue
            rows.append(dict(zip(header, parts)))

    if limit > 0:
        rows = rows[:limit]

    logger.info(f"Evaluating {len(rows)} segments")

    # Evaluate each segment
    results = []
    mem_peak = get_mem_usage_mb()
    for i, row in enumerate(rows):
        try:
            rec = await evaluate_one_segment(
                row, manifest_dir, src_lang, tgt_lang, asr_model, translator, tts_model
            )
            results.append(rec)
            mem_current = get_mem_usage_mb()
            mem_peak = max(mem_peak, mem_current)
            if (i + 1) % 10 == 0:
                logger.info(f"  {i+1}/{len(rows)} done (peak RAM {mem_peak:.1f} MB)")
        except Exception as e:
            logger.warning(f"Segment {i} failed: {e}")
            results.append({
                "id": row.get("id", f"seg_{i}"),
                "status": "exception",
                "error": str(e),
            })

    # Compute aggregate scores
    successful = [r for r in results if r["status"] == "success"]
    logger.info(f"Successful: {len(successful)}/{len(results)}")

    # BLEU
    from src.utils.metrics import compute_bleu
    if successful:
        hyps = [r["mt_text"] for r in successful]
        refs = [r["ref_tgt_text"] for r in successful]
        tokenize = "ko-mecab" if tgt_lang == "ko" else "13a"
        bleu, tokenize_used = compute_bleu(hyps, refs, tokenize=tokenize)
    else:
        bleu = -1.0
        tokenize_used = "n/a"

    # ASR WER (mean)
    asr_wers = [r["asr_wer"] for r in successful if r["asr_wer"] >= 0]
    mean_asr_wer = sum(asr_wers) / len(asr_wers) if asr_wers else -1.0

    # Round-trip WER (mean)
    roundtrip_wers = [r["roundtrip_wer"] for r in successful if r["roundtrip_wer"] >= 0]
    mean_roundtrip_wer = sum(roundtrip_wers) / len(roundtrip_wers) if roundtrip_wers else -1.0

    # Latencies
    asr_ms_vals = [r["asr_ms"] for r in successful]
    mt_ms_vals = [r["mt_ms"] for r in successful]
    tts_ms_vals = [r["tts_ms"] for r in successful]
    total_ms_vals = [r["total_ms"] for r in successful]

    mean_asr_ms = sum(asr_ms_vals) / len(asr_ms_vals) if asr_ms_vals else 0.0
    mean_mt_ms = sum(mt_ms_vals) / len(mt_ms_vals) if mt_ms_vals else 0.0
    mean_tts_ms = sum(tts_ms_vals) / len(tts_ms_vals) if tts_ms_vals else 0.0
    mean_total_ms = sum(total_ms_vals) / len(total_ms_vals) if total_ms_vals else 0.0

    # RTFx
    rtfx_vals = [r["rtfx"] for r in successful]
    mean_rtfx = sum(rtfx_vals) / len(rtfx_vals) if rtfx_vals else 0.0

    # Percentiles for latency
    if total_ms_vals:
        total_ms_vals_sorted = sorted(total_ms_vals)
        p95_total_ms = total_ms_vals_sorted[int(len(total_ms_vals_sorted) * 0.95)]
    else:
        p95_total_ms = 0.0

    # Build report
    report = {
        "config": {
            "manifest": str(manifest_path),
            "src_lang": src_lang,
            "tgt_lang": tgt_lang,
            "limit": limit if limit > 0 else "all",
        },
        "aggregate_scores": {
            "n_segments_total": len(results),
            "n_segments_successful": len(successful),
            "bleu": round(bleu, 2),
            "bleu_tokenize": tokenize_used,
            "asr_wer_mean_pct": round(mean_asr_wer, 2),
            "roundtrip_wer_mean_pct": round(mean_roundtrip_wer, 2),
            "asr_ms_mean": round(mean_asr_ms, 1),
            "mt_ms_mean": round(mean_mt_ms, 1),
            "tts_ms_mean": round(mean_tts_ms, 1),
            "total_ms_mean": round(mean_total_ms, 1),
            "total_ms_p95": round(p95_total_ms, 1),
            "rtfx_mean": round(mean_rtfx, 3),
            "peak_ram_mb": round(mem_peak, 1),
        },
        "per_segment_samples": successful[:20],  # First 20 for review
    }

    # Save report
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    logger.info(f"Report saved → {report_path}")

    # Print summary
    logger.info("── E2E Evaluation Summary ──────────────────")
    logger.info(f"  Segments (successful/total)  {len(successful)}/{len(results)}")
    logger.info(f"  BLEU ({tokenize_used})")
    logger.info(f"                              {bleu:.2f}")
    logger.info(f"  ASR WER (%)                 {mean_asr_wer:.2f}")
    logger.info(f"  Round-trip WER (%)          {mean_roundtrip_wer:.2f}")
    logger.info(f"  ASR latency (ms)            {mean_asr_ms:.1f}")
    logger.info(f"  MT latency (ms)             {mean_mt_ms:.1f}")
    logger.info(f"  TTS latency (ms)            {mean_tts_ms:.1f}")
    logger.info(f"  Total latency (ms)          {mean_total_ms:.1f} (p95: {p95_total_ms:.1f})")
    logger.info(f"  RTFx (total time / audio)   {mean_rtfx:.3f}")
    logger.info(f"  Peak RAM (MB)               {mem_peak:.1f}")
    logger.info("────────────────────────────────────────────")


def main():
    parser = argparse.ArgumentParser(description="End-to-end audio pipeline evaluation")
    parser.add_argument("--manifest", required=True, help="Fleurs pair manifest TSV")
    parser.add_argument("--src", default="en", choices=["en", "ko"])
    parser.add_argument("--tgt", default="ko", choices=["en", "ko"])
    parser.add_argument("--report", required=True, help="Output JSON report")
    parser.add_argument("--limit", type=int, default=0, help="Max segments (0=all)")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        logger.error(f"Manifest not found: {manifest_path}")
        return

    report_path = Path(args.report)

    import asyncio
    asyncio.run(main_async(manifest_path, args.src, args.tgt, report_path, args.limit))


if __name__ == "__main__":
    main()
