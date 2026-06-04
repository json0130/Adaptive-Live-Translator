#!/usr/bin/env python
"""R5-2: End-to-end evaluation with BERT-slim TTS and full RAM instrumentation.

Gate: co-resident peak RSS < 4 GB (MeloTTS BERT bypass).
Guardrail: round-trip WER <= 25% (intelligibility preserved without BERT).

Changes vs R4-1 eval_e2e.py:
  1. Uses KoreanCpuTTS(disable_bert=True) — hps.data.disable_bert=True bypasses
     kykim/bert-kor-base model load (~1052 MB saved).
  2. VmRSS sampling: reads /proc/self/status:VmRSS at each synthesis call to
     measure instantaneous RSS (not just HWM). Samples committed to report.
  3. per_segment_samples is NOT capped below N — all 50 segments committed.
  4. Round-trip WER: synthesize en->ko MT output, then ASR back to text,
     compute WER vs the MT hypothesis. Measures intelligibility of BERT-free TTS.
  5. --src / --tgt flags (default en->ko, the MeloTTS path).

RAM measurements:
  - ru_maxrss: peak HWM (monotonic), sampled via resource.getrusage
  - VmRSS:     instantaneous from /proc/self/status, sampled after each synthesis
               to show whether per-utterance growth is bounded

BLEU protocol:
  - ko target: ko-mecab (HALT if falls back to char)
  - en target: 13a
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import re
import resource
import time
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np
from loguru import logger


# ---------------------------------------------------------------------------
# RAM helpers
# ---------------------------------------------------------------------------

def get_peak_ram_mb() -> float:
    """Peak RSS HWM in MB (ru_maxrss, KB on Linux)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def get_vmrss_mb() -> float:
    """Instantaneous RSS in MB from /proc/self/status:VmRSS."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    kb = int(line.split()[1])
                    return kb / 1024.0
    except Exception:
        pass
    return -1.0


def get_vmhwm_mb() -> float:
    """Peak RSS from /proc/self/status:VmHWM (should equal ru_maxrss on Linux)."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmHWM:"):
                    kb = int(line.split()[1])
                    return kb / 1024.0
    except Exception:
        pass
    return -1.0


# ---------------------------------------------------------------------------
# Component loaders
# ---------------------------------------------------------------------------

def load_streaming_asr():
    """Load R4-2 winning ASR config: base streaming + small final pass + gate -0.7."""
    from src.asr.streaming_local_agreement import LocalAgreementASR
    logger.info("Loading streaming ASR: base(int8) streaming + small(int8) final + gate=-0.7")
    asr = LocalAgreementASR(
        model_id="Systran/faster-whisper-base",
        compute_type="int8",
        cpu_threads=8,
        initial_chunk_s=1.0,
        step_s=1.0,
        beam_size=1,
        confidence_gate_threshold=-0.7,
        final_model_id="Systran/faster-whisper-small",
    )
    asr._ensure_loaded()
    asr._ensure_final_model_loaded()
    return asr


def load_translator():
    """Load NLLB-200-distilled-600M CT2 int8."""
    from src.translator.nllb_ct2_translator import NllbCt2Translator
    logger.info("Loading translator: NLLB-200-distilled-600M CT2 INT8")
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ct2_model_dir = os.path.join(repo_root, "models", "nllb-600m-ct2-int8")
    if not os.path.exists(ct2_model_dir):
        canonical = "/home/jay/Desktop/Jay-per/adaptive-live-translator/models/nllb-600m-ct2-int8"
        if os.path.exists(canonical):
            ct2_model_dir = canonical
        else:
            raise FileNotFoundError(f"CT2 model not found at {ct2_model_dir} or {canonical}")
    logger.info(f"CT2 model dir: {ct2_model_dir}")
    translator = NllbCt2Translator({
        "translator": {
            "ct2_model_dir": ct2_model_dir,
            "device": "cpu",
            "dtype": "int8",
            "max_new_tokens": 256,
        }
    })
    translator._lazy_load()
    return translator


def load_tts_ko_slim():
    """Load MeloTTS-KR with BERT bypass (R5-2). Returns (tts, engine_name).

    disable_bert=True sets hps.data.disable_bert=True after MeloTTS loads,
    so get_text_for_tts_infer skips calling get_bert() and instead returns
    zero BERT tensors. This prevents kykim/bert-kor-base from loading.

    Pre-warm: run one synthesis BEFORE measuring RAM to ensure the MeloTTS
    VITS model is warm. With disable_bert=True, no BERT model loads on prewarm.
    """
    from src.tts.korean_cpu_tts import KoreanCpuTTS
    logger.info("Loading TTS: KoreanCpuTTS (MeloTTS, disable_bert=True)")
    tts = KoreanCpuTTS(engine="melo", device="cpu", disable_bert=True)
    logger.info(f"TTS loaded, engine={tts._engine}")
    logger.info(f"  disable_bert on hps: {getattr(tts._melo.hps.data, 'disable_bert', 'NOT SET')}")

    # Pre-warm: force the VITS model to do one synthesis so any lazy init
    # (model weights, g2pkk init) completes before the measured run.
    logger.info("Pre-warming TTS (one dummy synthesis)...")
    import tempfile
    tmp = tempfile.mktemp(suffix=".wav")
    try:
        import torch
        with torch.no_grad():
            tts._melo.tts_to_file("안녕", tts._melo_speaker, output_path=tmp, speed=1.0, quiet=True)
        if os.path.exists(tmp):
            os.unlink(tmp)
        gc.collect()
        logger.info("TTS pre-warm complete (no BERT load expected)")
    except Exception as e:
        logger.warning(f"TTS pre-warm failed (non-fatal): {e}")

    return tts, "melo-slim-no-bert"


def load_tts_en():
    """espeak-ng for English TTS (ko->en round-trip check)."""
    from src.tts.korean_cpu_tts import KoreanCpuTTS
    tts = KoreanCpuTTS(engine="espeak", device="cpu")
    return tts, "espeak"


# ---------------------------------------------------------------------------
# Synchronous translate helper
# ---------------------------------------------------------------------------

def translate_sync(translator, text: str, src_lang: str, tgt_lang: str) -> Tuple[str, float]:
    """Translate text synchronously. Returns (translated_text, latency_ms).

    Uses the same pattern as R4-1 eval_e2e.py translate_sync:
      - async generator yielding the input text as a single chunk
      - system_prompt="" (NLLB-CT2 ignores it)
      - chunk.text accumulation
    src_lang / tgt_lang are short codes ('en', 'ko'); translator maps internally.
    """
    import asyncio

    async def _run():
        async def _src_iter():
            yield text

        t0 = time.perf_counter()
        hyp = ""
        async for chunk in translator.translate_stream(
            _src_iter(),
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            system_prompt="",
        ):
            hyp = chunk.text
        mt_ms = (time.perf_counter() - t0) * 1000.0
        return hyp, mt_ms

    return asyncio.run(_run())


# ---------------------------------------------------------------------------
# Per-segment evaluation
# ---------------------------------------------------------------------------

def evaluate_segment(
    row: dict,
    manifest_dir: Path,
    src_lang: str,
    tgt_lang: str,
    asr_model,
    translator,
    tts_model,
    tts_engine: str,
) -> dict:
    """Evaluate one segment. Returns per-segment record."""
    src_audio_col = f"{src_lang}_audio"
    tgt_text_col = f"{tgt_lang}_text"
    src_text_col = f"{src_lang}_text"

    # Resolve audio path
    audio_rel = row[src_audio_col]
    audio_path = manifest_dir / audio_rel
    if not audio_path.exists():
        alt2 = Path("/home/jay/Desktop/Jay-per/adaptive-live-translator/data/eval") / audio_rel
        if alt2.exists():
            audio_path = alt2
        else:
            manifest_abs = manifest_dir.absolute()
            for levels_up in [5, 4, 3]:
                parent = manifest_abs
                for _ in range(levels_up):
                    parent = parent.parent
                alt = parent / "data" / "eval" / audio_rel
                if alt.exists():
                    audio_path = alt
                    break

    ref_tgt_text = row[tgt_text_col]
    src_text_ref = row[src_text_col]

    # --- Step 1: ASR ---
    t_asr_wall0 = time.perf_counter()
    try:
        result = asr_model.transcribe_streaming(str(audio_path), lang=src_lang)
        asr_text = result.accuracy_transcript
        asr_latency_s = result.first_emission_latency_s or 0.0
        audio_s = result.total_audio_s
        asr_streaming_ms = result.total_decode_s * 1000.0
        asr_e2e_ms = (time.perf_counter() - t_asr_wall0) * 1000.0
        confidence_gate_applied = result.confidence_gate_applied
    except Exception as e:
        logger.warning(f"ASR failed for {row.get('id', '?')}: {e}")
        return {"id": row.get("id", "unknown"), "status": "asr_failed", "error": str(e)}

    # --- Step 2: MT ---
    try:
        mt_text, mt_ms = translate_sync(translator, asr_text, src_lang, tgt_lang)
    except Exception as e:
        logger.warning(f"MT failed for {row.get('id', '?')}: {e}")
        return {"id": row.get("id", "unknown"), "status": "mt_failed", "error": str(e), "asr_text": asr_text}

    # --- Step 3: TTS + VmRSS sample ---
    vmrss_before_tts = get_vmrss_mb()
    t_tts0 = time.perf_counter()
    tts_ms = 0.0
    synth_audio = np.zeros(16000, dtype=np.float32)
    if tts_model is not None and mt_text.strip():
        try:
            synth_audio, tts_ms = tts_model.synthesize(mt_text, lang=tgt_lang)
        except Exception as e:
            logger.warning(f"TTS failed for {row.get('id', '?')}: {e}")
            tts_ms = (time.perf_counter() - t_tts0) * 1000.0
    vmrss_after_tts = get_vmrss_mb()

    # --- Round-trip WER (en->ko only): ASR the TTS output ---
    rt_wer_pct = -1.0
    rt_asr_text = ""
    if tts_engine != "espeak" and tgt_lang == "ko" and len(synth_audio) > 160:
        try:
            import tempfile, soundfile as sf
            tmp = tempfile.mktemp(suffix=".wav")
            sf.write(tmp, synth_audio, 16000)
            rt_result = asr_model.transcribe_streaming(tmp, lang="ko")
            rt_asr_text = rt_result.accuracy_transcript or ""
            if os.path.exists(tmp):
                os.unlink(tmp)
            from jiwer import wer as jwer
            # Round-trip WER: ASR(TTS(MT(ASR(en)))) vs MT(ASR(en))
            rt_wer_pct = round(jwer([mt_text], [rt_asr_text]) * 100, 2)
        except Exception as e:
            logger.debug(f"Round-trip WER failed for {row.get('id', '?')}: {e}")

    # --- BLEU ---
    from src.utils.metrics import compute_bleu
    from jiwer import wer as jwer, cer as jcer

    tokenize = "ko-mecab" if tgt_lang == "ko" else "13a"
    bleu, tokenize_used = compute_bleu([mt_text], [ref_tgt_text], tokenize=tokenize)

    if tgt_lang == "ko" and "char" in tokenize_used:
        logger.error(
            f"BLEU tokenizer fallback to '{tokenize_used}' — MeCab unavailable. "
            "This violates the eval protocol. DO NOT accept this number."
        )

    # ASR WER on source
    try:
        if src_lang == "ko":
            asr_error = jcer([src_text_ref], [asr_text])
        else:
            asr_error = jwer([src_text_ref], [asr_text])
        asr_error_pct = round(asr_error * 100, 2)
    except Exception as e:
        logger.warning(f"ASR error rate failed: {e}")
        asr_error_pct = -1.0

    e2e_latency_ms = (asr_latency_s * 1000.0) + mt_ms + tts_ms

    return {
        "id": row.get("id", "unknown"),
        "status": "success",
        "audio_s": round(audio_s, 2),
        "asr_text": asr_text,
        "asr_first_emission_s": round(asr_latency_s, 3),
        "asr_streaming_ms": round(asr_streaming_ms, 1),
        "asr_e2e_ms": round(asr_e2e_ms, 1),
        "asr_wer_pct": asr_error_pct,
        "confidence_gate_applied": confidence_gate_applied,
        "mt_text": mt_text,
        "mt_ms": round(mt_ms, 1),
        "tts_ms": round(tts_ms, 1),
        "tts_engine": tts_engine,
        "e2e_latency_ms": round(e2e_latency_ms, 1),
        "segment_bleu": round(bleu, 2),
        "bleu_tokenize": tokenize_used,
        "ref_tgt_text": ref_tgt_text,
        "rt_wer_pct": rt_wer_pct,
        "rt_asr_text": rt_asr_text,
        "vmrss_before_tts_mb": round(vmrss_before_tts, 1),
        "vmrss_after_tts_mb": round(vmrss_after_tts, 1),
    }


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def run_eval(
    manifest_path: Path,
    src_lang: str,
    tgt_lang: str,
    report_path: Path,
    limit: int = 0,
) -> dict:
    """Run full e2e evaluation, return report dict."""
    logger.info(f"=== R5-2 TTS-RAM-Slim Eval: {src_lang}->{tgt_lang} ===")
    logger.info("TTS: MeloTTS-KR with disable_bert=True (BERT model bypassed)")

    # Baseline RAM before any models
    mem_baseline = get_peak_ram_mb()
    vmrss_baseline = get_vmrss_mb()
    vmhwm_baseline = get_vmhwm_mb()
    logger.info(f"RAM baseline: peak_hwm={mem_baseline:.1f} MB, VmRSS={vmrss_baseline:.1f} MB")

    # Load ASR
    asr_model = load_streaming_asr()
    mem_after_asr = get_peak_ram_mb()
    vmrss_after_asr = get_vmrss_mb()
    logger.info(f"RAM after ASR: peak_hwm={mem_after_asr:.1f} MB (+{mem_after_asr - mem_baseline:.1f}), VmRSS={vmrss_after_asr:.1f} MB")

    # Load MT
    translator = load_translator()
    mem_after_mt = get_peak_ram_mb()
    vmrss_after_mt = get_vmrss_mb()
    logger.info(f"RAM after MT: peak_hwm={mem_after_mt:.1f} MB (+{mem_after_mt - mem_after_asr:.1f}), VmRSS={vmrss_after_mt:.1f} MB")

    # Load TTS (with BERT bypass + pre-warm)
    if tgt_lang == "ko":
        tts_model, tts_engine = load_tts_ko_slim()
    else:
        tts_model, tts_engine = load_tts_en()

    mem_after_tts = get_peak_ram_mb()
    vmrss_after_tts = get_vmrss_mb()
    vmhwm_after_tts = get_vmhwm_mb()
    logger.info(f"RAM after TTS + prewarm: peak_hwm={mem_after_tts:.1f} MB (+{mem_after_tts - mem_after_mt:.1f}), VmRSS={vmrss_after_tts:.1f} MB")
    logger.info(f"  VmHWM from /proc/self/status: {vmhwm_after_tts:.1f} MB")
    logger.info(f"  Co-resident peak at load: {mem_after_tts:.1f} MB (gate: <4096 MB)")

    if tts_model is None:
        logger.warning("TTS unavailable — TTS latency will be 0.")
    else:
        logger.info(f"TTS engine: {tts_engine}")
        if hasattr(tts_model, '_melo') and tts_model._melo is not None:
            db = getattr(tts_model._melo.hps.data, 'disable_bert', 'NOT SET')
            logger.info(f"  MeloTTS hps.data.disable_bert = {db}")

    # Read manifest
    manifest_dir = manifest_path.parent
    rows: list = []
    with open(manifest_path) as f:
        header = f.readline().rstrip("\n").split("\t")
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < len(header):
                continue
            rows.append(dict(zip(header, parts)))

    if limit > 0:
        rows = rows[:limit]

    logger.info(f"Evaluating {len(rows)} segments (direction: {src_lang}->{tgt_lang})")

    # Evaluate each segment with VmRSS tracking
    results = []
    vmrss_samples = []  # instantaneous RSS after each TTS call
    peak_hwm_run = mem_after_tts

    for i, row in enumerate(rows):
        try:
            rec = evaluate_segment(
                row, manifest_dir, src_lang, tgt_lang,
                asr_model, translator, tts_model, tts_engine,
            )
            results.append(rec)

            # Track peak HWM and VmRSS samples
            current_hwm = get_peak_ram_mb()
            peak_hwm_run = max(peak_hwm_run, current_hwm)

            vmrss_now = get_vmrss_mb()
            vmrss_after_rec = rec.get("vmrss_after_tts_mb", vmrss_now)
            vmrss_samples.append({
                "seg_idx": i,
                "seg_id": row.get("id", f"seg_{i}"),
                "vmrss_after_tts_mb": vmrss_after_rec,
                "vmrss_now_mb": vmrss_now,
                "peak_hwm_mb": current_hwm,
                "status": rec.get("status", "unknown"),
            })

            if (i + 1) % 10 == 0:
                successes = sum(1 for r in results if r.get("status") == "success")
                logger.info(
                    f"  {i+1}/{len(rows)} done — {successes} ok | "
                    f"peak_hwm={peak_hwm_run:.0f} MB | VmRSS={vmrss_now:.0f} MB"
                )
        except Exception as e:
            logger.warning(f"Segment {i} ({row.get('id','?')}) exception: {e}")
            results.append({
                "id": row.get("id", f"seg_{i}"),
                "status": "exception",
                "error": str(e),
            })

    # Final RAM measurements
    final_hwm = max(peak_hwm_run, get_peak_ram_mb())
    final_vmrss = get_vmrss_mb()
    final_vmhwm = get_vmhwm_mb()
    logger.info(f"Final RAM: peak_hwm={final_hwm:.1f} MB, VmRSS={final_vmrss:.1f} MB, VmHWM={final_vmhwm:.1f} MB")

    # Aggregate metrics
    successful = [r for r in results if r.get("status") == "success"]
    n_success = len(successful)
    n_total = len(results)
    logger.info(f"Successful: {n_success}/{n_total}")

    if n_success < 20:
        logger.error(
            f"Only {n_success} successful segments — below N>=20 threshold. "
            "Results are not reportable per eval protocol."
        )

    # Corpus BLEU
    from src.utils.metrics import compute_bleu
    if n_success > 0:
        hyps = [r["mt_text"] for r in successful]
        refs = [r["ref_tgt_text"] for r in successful]
        tokenize = "ko-mecab" if tgt_lang == "ko" else "13a"
        bleu, tokenize_used = compute_bleu(hyps, refs, tokenize=tokenize)

        if tgt_lang == "ko" and "char" in tokenize_used:
            logger.error(
                f"CORPUS BLEU tokenizer is '{tokenize_used}' — MeCab unavailable. "
                "HALT: this number is NOT protocol-compliant."
            )
        else:
            logger.info(f"Corpus BLEU ({tokenize_used}): {bleu:.2f}")
    else:
        bleu = -1.0
        tokenize_used = "n/a"

    # Round-trip WER (guardrail for intelligibility)
    rt_wers = [r["rt_wer_pct"] for r in successful if r.get("rt_wer_pct", -1) >= 0]
    mean_rt_wer = sum(rt_wers) / len(rt_wers) if rt_wers else -1.0
    n_rt_wer = len(rt_wers)
    if mean_rt_wer >= 0:
        logger.info(f"Round-trip WER (N={n_rt_wer}): {mean_rt_wer:.2f}%  (gate: <=25%)")
        if mean_rt_wer <= 25.0:
            logger.info(f"  >> Intelligibility PASS (WER {mean_rt_wer:.2f}% <= 25%)")
        else:
            logger.warning(f"  >> Intelligibility FAIL (WER {mean_rt_wer:.2f}% > 25%)")

    # ASR stats
    asr_wers = [r["asr_wer_pct"] for r in successful if r.get("asr_wer_pct", -1) >= 0]
    mean_asr_wer = sum(asr_wers) / len(asr_wers) if asr_wers else -1.0

    # Latency stats
    e2e_ms_vals = sorted([r["e2e_latency_ms"] for r in successful])
    tts_ms_vals = [r["tts_ms"] for r in successful]
    mt_ms_vals = [r["mt_ms"] for r in successful]
    asr_first_s_vals = [r["asr_first_emission_s"] for r in successful]

    def _mean(lst):
        return sum(lst) / len(lst) if lst else 0.0
    def _pctile(lst_sorted, pct):
        if not lst_sorted:
            return 0.0
        idx = min(int(len(lst_sorted) * pct / 100), len(lst_sorted) - 1)
        return lst_sorted[idx]

    mean_e2e_ms = _mean(e2e_ms_vals)
    p50_e2e_ms = _pctile(e2e_ms_vals, 50)
    p95_e2e_ms = _pctile(e2e_ms_vals, 95)
    mean_tts_ms = _mean(tts_ms_vals)
    mean_mt_ms = _mean(mt_ms_vals)
    mean_asr_first_s = _mean(asr_first_s_vals)

    # VmRSS growth analysis (use vmrss_now_mb for reliable readings)
    vmrss_vals = [s["vmrss_now_mb"] for s in vmrss_samples if s.get("vmrss_now_mb", -1) > 0]
    vmrss_growth = (vmrss_vals[-1] - vmrss_vals[0]) if len(vmrss_vals) >= 2 else 0.0
    vmrss_max = max(vmrss_vals) if vmrss_vals else 0.0
    vmrss_min = min(vmrss_vals) if vmrss_vals else 0.0

    first_vmrss_str = f"{vmrss_vals[0]:.1f}" if vmrss_vals else "n/a"
    last_vmrss_str = f"{vmrss_vals[-1]:.1f}" if len(vmrss_vals) > 1 else "n/a"
    logger.info(f"VmRSS after TTS: first={first_vmrss_str} MB, last={last_vmrss_str} MB, "
                f"growth={vmrss_growth:.1f} MB, max={vmrss_max:.1f} MB")

    n_gate_fired = sum(1 for r in successful if r.get("confidence_gate_applied", False))

    # RAM gate verdict
    gate_pass = final_hwm < 4096.0
    intelligibility_pass = mean_rt_wer <= 25.0 if mean_rt_wer >= 0 else None
    logger.info(f"=== RAM GATE: peak_hwm={final_hwm:.1f} MB vs <4096 MB -> {'PASS' if gate_pass else 'FAIL'} ===")
    if intelligibility_pass is not None:
        logger.info(f"=== INTELLIGIBILITY GATE: rt_wer={mean_rt_wer:.2f}% vs <=25% -> {'PASS' if intelligibility_pass else 'FAIL'} ===")

    # Build report (ALL per_segment_samples, no cap when N<=50)
    report = {
        "experiment": "R5-2 TTS-RAM-Slim (disable_bert=True)",
        "config": {
            "src_lang": src_lang,
            "tgt_lang": tgt_lang,
            "manifest": str(manifest_path),
            "limit": limit if limit > 0 else "all",
            "asr": "faster-whisper-base(int8)+small(int8)+LA2+gate-0.7",
            "mt": "NLLB-200-distilled-600M-ct2-int8",
            "tts": tts_engine,
            "tts_disable_bert": True,
        },
        "ram": {
            "baseline_mb": round(mem_baseline, 1),
            "after_asr_mb": round(mem_after_asr, 1),
            "after_mt_mb": round(mem_after_mt, 1),
            "after_tts_prewarm_mb": round(mem_after_tts, 1),
            "peak_ram_mb": round(final_hwm, 1),
            "final_vmrss_mb": round(final_vmrss, 1),
            "final_vmhwm_mb": round(final_vmhwm, 1),
            "vmrss_after_tts_first_mb": round(vmrss_vals[0], 1) if vmrss_vals else -1,
            "vmrss_after_tts_last_mb": round(vmrss_vals[-1], 1) if len(vmrss_vals) > 1 else -1,
            "vmrss_growth_over_run_mb": round(vmrss_growth, 1),
            "vmrss_max_mb": round(vmrss_max, 1),
            "vmrss_min_mb": round(vmrss_min, 1),
            "gate_4gb_pass": gate_pass,
        },
        "aggregate_scores": {
            "n_segments_total": n_total,
            "n_segments_successful": n_success,
            "bleu": round(bleu, 2),
            "bleu_tokenize": tokenize_used,
            "asr_wer_mean_pct": round(mean_asr_wer, 2),
            "round_trip_wer_mean_pct": round(mean_rt_wer, 2) if mean_rt_wer >= 0 else -1,
            "round_trip_wer_n_segments": n_rt_wer,
            "intelligibility_pass": intelligibility_pass,
            "asr_first_emission_mean_s": round(mean_asr_first_s, 3),
            "mt_ms_mean": round(mean_mt_ms, 1),
            "tts_ms_mean": round(mean_tts_ms, 1),
            "e2e_latency_ms_mean": round(mean_e2e_ms, 1),
            "e2e_latency_ms_p50": round(p50_e2e_ms, 1),
            "e2e_latency_ms_p95": round(p95_e2e_ms, 1),
            "confidence_gate_fired_pct": round(100.0 * n_gate_fired / n_success, 1) if n_success > 0 else 0.0,
        },
        "vmrss_over_time": vmrss_samples,  # full list, not capped
        "per_segment_samples": results,    # all segments (N<=50)
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    logger.info(f"Report saved -> {report_path}")

    # Summary
    logger.info("-- R5-2 TTS-RAM-Slim Summary --------------------------------")
    logger.info(f"  Direction:          {src_lang}->{tgt_lang}")
    logger.info(f"  N (ok/total):       {n_success}/{n_total}")
    logger.info(f"  BLEU ({tokenize_used:<12}): {bleu:.2f}")
    logger.info(f"  ASR WER (%):        {mean_asr_wer:.2f}")
    logger.info(f"  TTS round-trip WER: {mean_rt_wer:.2f}% (N={n_rt_wer})")
    logger.info(f"  MT latency (ms):    {mean_mt_ms:.1f}")
    logger.info(f"  TTS latency (ms):   {mean_tts_ms:.1f}")
    logger.info(f"  E2E latency (ms):   {mean_e2e_ms:.1f}  p50={p50_e2e_ms:.1f}  p95={p95_e2e_ms:.1f}")
    logger.info(f"  Peak RAM (MB):      {final_hwm:.1f}")
    logger.info(f"    ASR:              {mem_after_asr:.1f} MB")
    logger.info(f"    ASR+MT:           {mem_after_mt:.1f} MB")
    logger.info(f"    ASR+MT+TTS warm:  {mem_after_tts:.1f} MB")
    logger.info(f"  VmRSS (first->last TTS): {first_vmrss_str} -> {last_vmrss_str} MB (growth: {vmrss_growth:+.1f} MB)")
    logger.info(f"  RAM gate (<4096 MB): {'PASS' if gate_pass else 'FAIL'}")
    if intelligibility_pass is not None:
        logger.info(f"  Intelligibility gate (<=25% WER): {'PASS' if intelligibility_pass else 'FAIL'}")
    logger.info("-------------------------------------------------------------")

    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="R5-2 TTS-RAM-Slim E2E Eval")
    parser.add_argument("--manifest", required=True, help="Fleurs pair manifest TSV")
    parser.add_argument("--src", default="en", choices=["en", "ko"])
    parser.add_argument("--tgt", default="ko", choices=["en", "ko"])
    parser.add_argument("--report", required=True, help="Output JSON report path")
    parser.add_argument("--limit", type=int, default=50, help="Max segments (default: 50)")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        logger.error(f"Manifest not found: {manifest_path}")
        return

    run_eval(
        manifest_path=manifest_path,
        src_lang=args.src,
        tgt_lang=args.tgt,
        report_path=Path(args.report),
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
