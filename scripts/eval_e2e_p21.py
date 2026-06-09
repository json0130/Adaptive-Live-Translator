#!/usr/bin/env python
"""P2-1 GPU Pipeline End-to-End Evaluation.

Stack (P2-1):
  - ASR:  faster-whisper large-v3, int8_float16, CUDA, LocalAgreement-2 streaming
  - MT:   NLLB-200-distilled-600M, transformers fp16, CUDA, greedy decode
  - TTS:  MeloTTS-KR, CUDA

Eval protocol (LOCKED — identical to Phase 1):
  - Manifest: data/eval/fleurs_pair_manifest.tsv (270 rows)
  - BLEU: ko targets -> ko-mecab; en targets -> 13a
  - Latency: honest — same path produces quality output and measures latency
  - N=270 per direction (both en->ko and ko->en)

VRAM measurement (per P2-1 spec):
  - torch.cuda.max_memory_allocated() after reset: tracks PyTorch fp16 allocations
  - nvidia-smi sampled at key points: captures CTranslate2 (ASR) VRAM not seen by torch
  - Both measurements reported in the JSON

RAM measurement:
  - resource.getrusage(RUSAGE_SELF).ru_maxrss: true peak process RSS in KB (Linux)
  - Reported in MB: ru_maxrss / 1024

Warnings:
  - kykim/bert-kor-base "Some weights not used" at TTS prewarm: EXPECTED.
    BertForMaskedLM is initialized from a BertForPreTraining checkpoint;
    the seq_relationship head is unused. This is documented behavior.
    Do NOT suppress this warning.
  - transformers FutureWarning about resume_download: harmless, not suppressed.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import resource
import subprocess
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from loguru import logger


# ---------------------------------------------------------------------------
# VRAM helpers
# ---------------------------------------------------------------------------

def nvidia_smi_used_mib() -> int:
    """Query nvidia-smi for current GPU memory used (MiB)."""
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        return int(r.stdout.strip())
    except Exception as e:
        logger.warning(f"nvidia-smi query failed: {e}")
        return -1


def get_mem_mb() -> float:
    """Current peak RSS in MB (Linux: ru_maxrss in KB)."""
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return usage.ru_maxrss / 1024.0


# ---------------------------------------------------------------------------
# Component loaders
# ---------------------------------------------------------------------------

def load_asr():
    """Load faster-whisper large-v3, int8_float16, CUDA.

    P2-1 ASR config (explicitly logged per spec):
      model: large-v3
      device: cuda
      compute_type: int8_float16
    """
    from src.asr.gpu_streaming_asr import GpuLocalAgreementASR
    logger.info(
        "P2-1 ASR config: model=large-v3, device=cuda, compute_type=int8_float16, "
        "streaming=LocalAgreement-2, initial_chunk=1.0s, step=1.0s, beam=1"
    )
    asr = GpuLocalAgreementASR(
        model_id="large-v3",
        compute_type="int8_float16",
        device="cuda",
        initial_chunk_s=1.0,
        step_s=1.0,
        beam_size=1,
        confidence_gate_threshold=None,  # pure LA-2, no confidence gate
    )
    asr._ensure_loaded()
    return asr


def load_translator():
    """Load NLLB-200-distilled-600M fp16 on CUDA."""
    from src.translator.nllb_fp16_translator import NllbFp16Translator
    logger.info("Loading NLLB-200-distilled-600M fp16 on CUDA...")
    torch.cuda.reset_peak_memory_stats()
    translator = NllbFp16Translator(
        model_name="facebook/nllb-200-distilled-600M",
        device="cuda",
        max_new_tokens=256,
    )
    translator.load()
    return translator


def load_tts(device: str = "cuda", prewarm: bool = True):
    """Load MeloTTS-KR on CUDA. Returns (tts, engine_name).

    Prewarming: loads kykim/bert-kor-base by running a dummy synthesis.
    The 'Some weights not used' warning from bert-kor-base is EXPECTED
    (BertForMaskedLM from BertForPreTraining checkpoint — documented behavior).
    """
    from src.tts.korean_cpu_tts import KoreanCpuTTS
    logger.info(f"Loading MeloTTS-KR on {device}...")
    tts = KoreanCpuTTS(engine="melo", device=device)
    # Verify no silent fallback to espeak
    assert tts._engine == "melo", (
        f"BLOCKER: KoreanCpuTTS silently fell back to '{tts._engine}' — "
        "MeloTTS failed to load. This is a hard failure for P2-1."
    )
    logger.info(f"TTS loaded, engine={tts._engine}, device={device}")

    if prewarm:
        logger.info("Pre-warming TTS (triggers kykim/bert-kor-base load)...")
        import tempfile, os
        try:
            tmp = tempfile.mktemp(suffix=".wav")
            tts._melo.tts_to_file("안녕하세요", tts._melo_speaker, output_path=tmp, speed=1.0, quiet=True)
            if os.path.exists(tmp):
                os.unlink(tmp)
            logger.info("TTS pre-warm complete (bert-kor-base now loaded)")
        except Exception as e:
            logger.warning(f"TTS pre-warm failed (non-fatal): {e}")

    return tts, tts._engine


# ---------------------------------------------------------------------------
# Translation helper
# ---------------------------------------------------------------------------

def translate_sync(translator, src_text: str, src_lang: str, tgt_lang: str) -> tuple:
    """Translate text synchronously. Returns (translated_text, mt_ms)."""
    return translator.translate(src_text, src_lang, tgt_lang)


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
    """Evaluate one utterance through the full P2-1 pipeline.

    Latency is honest: the SAME path that produces quality output is timed.
    The streaming ASR produces the transcript fed to MT; MT output feeds TTS.
    first-emission latency = ASR first emission (streaming, wall-clock from audio start).
    e2e latency = ASR first-emission + MT + TTS.
    """
    src_audio_col = f"{src_lang}_audio"
    tgt_text_col = f"{tgt_lang}_text"
    src_text_col = f"{src_lang}_text"

    # Resolve audio path
    audio_rel = row[src_audio_col]
    audio_path = manifest_dir / audio_rel
    if not audio_path.exists():
        canonical = Path("/home/jay/Desktop/Jay-per/adaptive-live-translator/data/eval") / audio_rel
        if canonical.exists():
            audio_path = canonical

    if not audio_path.exists():
        return {
            "id": row.get("id", "unknown"),
            "status": "audio_not_found",
            "error": f"Audio not found: {audio_rel}",
        }

    ref_tgt_text = row[tgt_text_col]
    src_text_ref = row[src_text_col]

    # --- Step 1: Streaming ASR ---
    t_asr0 = time.perf_counter()
    try:
        result = asr_model.transcribe_streaming(str(audio_path), lang=src_lang)
        # For P2-1 (no final_model_id), accuracy_transcript == final_transcript
        # (large-v3 is accurate enough; no separate final pass needed)
        asr_text = result.accuracy_transcript
        asr_latency_s = result.first_emission_latency_s or 0.0
        audio_s = result.total_audio_s
        asr_streaming_ms = result.total_decode_s * 1000.0
        asr_e2e_ms = (time.perf_counter() - t_asr0) * 1000.0
    except Exception as e:
        logger.warning(f"ASR failed for {row.get('id','?')}: {e}")
        return {
            "id": row.get("id", "unknown"),
            "status": "asr_failed",
            "error": str(e),
        }

    # --- Step 2: Machine Translation ---
    try:
        mt_text, mt_ms = translate_sync(translator, asr_text, src_lang, tgt_lang)
    except Exception as e:
        logger.warning(f"MT failed for {row.get('id','?')}: {e}")
        return {
            "id": row.get("id", "unknown"),
            "status": "mt_failed",
            "error": str(e),
            "asr_text": asr_text,
        }

    # --- Step 3: TTS ---
    tts_ms = 0.0
    if tts_model is not None and mt_text.strip() and tgt_lang == "ko":
        try:
            t_tts0 = time.perf_counter()
            synth_audio, tts_ms = tts_model.synthesize(mt_text, lang=tgt_lang)
        except Exception as e:
            logger.warning(f"TTS failed for {row.get('id','?')}: {e}")
            tts_ms = (time.perf_counter() - t_tts0) * 1000.0
    elif tgt_lang == "en":
        # ko->en: no Korean TTS needed; TTS latency is 0 for this eval
        # (en TTS via espeak is not part of the P2-1 quality path)
        tts_ms = 0.0

    # --- BLEU ---
    from src.utils.metrics import compute_bleu
    tokenize = "ko-mecab" if tgt_lang == "ko" else "13a"
    bleu, tokenize_used = compute_bleu([mt_text], [ref_tgt_text], tokenize=tokenize)

    # Protocol compliance check
    if tgt_lang == "ko" and "char" in tokenize_used:
        logger.error(
            f"BLEU tokenizer fallback to '{tokenize_used}' for segment {row.get('id','?')} — "
            "MeCab unavailable. This violates eval protocol. DO NOT accept this number."
        )

    # --- ASR error rate ---
    try:
        from jiwer import wer, cer
        if src_lang == "ko":
            asr_error = cer([src_text_ref], [asr_text])
        else:
            asr_error = wer([src_text_ref], [asr_text])
        asr_error_pct = round(asr_error * 100, 2)
    except Exception as e:
        logger.warning(f"ASR error rate failed: {e}")
        asr_error_pct = -1.0

    # --- End-to-end latency ---
    # Honest: ASR first-emission (wall-clock from audio start) + MT + TTS
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
        "mt_text": mt_text,
        "mt_ms": round(mt_ms, 1),
        "tts_ms": round(tts_ms, 1),
        "tts_engine": tts_engine,
        "e2e_latency_ms": round(e2e_latency_ms, 1),
        "segment_bleu": round(bleu, 2),
        "bleu_tokenize": tokenize_used,
        "ref_tgt_text": ref_tgt_text,
        "src_text_ref": src_text_ref,
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
    """Run full P2-1 e2e evaluation. Returns report dict."""

    logger.info(f"=== P2-1 GPU Pipeline Eval: {src_lang}->{tgt_lang} ===")
    logger.info("ASR config: model=large-v3, device=cuda, compute_type=int8_float16")

    # VRAM baseline before any loads
    torch.cuda.reset_peak_memory_stats()
    vram_baseline_mib = nvidia_smi_used_mib()
    mem_before_load = get_mem_mb()
    logger.info(f"VRAM baseline (nvidia-smi): {vram_baseline_mib} MiB")
    logger.info(f"RAM before load: {mem_before_load:.1f} MB")

    # Load ASR
    asr_model = load_asr()
    vram_after_asr_mib = nvidia_smi_used_mib()
    mem_after_asr = get_mem_mb()
    logger.info(f"VRAM after ASR (nvidia-smi): {vram_after_asr_mib} MiB  (delta: +{vram_after_asr_mib - vram_baseline_mib} MiB)")
    logger.info(f"RAM after ASR: {mem_after_asr:.1f} MB")

    # Load MT
    translator = load_translator()
    vram_after_mt_mib = nvidia_smi_used_mib()
    mem_after_mt = get_mem_mb()
    torch_vram_after_mt_gb = torch.cuda.max_memory_allocated() / 1e9
    logger.info(f"VRAM after MT (nvidia-smi): {vram_after_mt_mib} MiB  (delta: +{vram_after_mt_mib - vram_after_asr_mib} MiB)")
    logger.info(f"torch.cuda.max_memory_allocated after MT: {torch_vram_after_mt_gb:.3f} GB")
    logger.info(f"RAM after MT: {mem_after_mt:.1f} MB")

    # Load TTS (MeloTTS on CUDA) + prewarm
    tts_model, tts_engine = load_tts(device="cuda", prewarm=True)
    vram_after_tts_mib = nvidia_smi_used_mib()
    mem_after_tts = get_mem_mb()
    torch_vram_after_tts_gb = torch.cuda.max_memory_allocated() / 1e9
    logger.info(f"VRAM after TTS+prewarm (nvidia-smi): {vram_after_tts_mib} MiB  (delta from baseline: +{vram_after_tts_mib - vram_baseline_mib} MiB)")
    logger.info(f"torch.cuda.max_memory_allocated after TTS: {torch_vram_after_tts_gb:.3f} GB")
    logger.info(f"RAM after TTS+prewarm: {mem_after_tts:.1f} MB")
    logger.info(f"Static-load VRAM peak (nvidia-smi): {vram_after_tts_mib} MiB = {vram_after_tts_mib/1024:.2f} GB")

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

    n_total_planned = len(rows)
    logger.info(f"Evaluating {n_total_planned} segments (direction: {src_lang}->{tgt_lang})")

    # Evaluate each segment
    results = []
    mem_peak_run = mem_after_tts
    vram_peak_inference_mib = vram_after_tts_mib
    torch_vram_peak_inference_gb = torch_vram_after_tts_gb

    for i, row in enumerate(rows):
        try:
            # Reset torch peak stats for this segment to measure inference VRAM
            torch.cuda.reset_peak_memory_stats()

            rec = evaluate_segment(
                row, manifest_dir, src_lang, tgt_lang,
                asr_model, translator, tts_model, tts_engine,
            )
            results.append(rec)

            # Measure inference VRAM peak for this segment
            seg_torch_vram_gb = torch.cuda.max_memory_allocated() / 1e9
            seg_nvidia_mib = nvidia_smi_used_mib()
            torch_vram_peak_inference_gb = max(torch_vram_peak_inference_gb, seg_torch_vram_gb)
            vram_peak_inference_mib = max(vram_peak_inference_mib, seg_nvidia_mib)

            mem_current = get_mem_mb()
            mem_peak_run = max(mem_peak_run, mem_current)

            if (i + 1) % 20 == 0:
                successes = sum(1 for r in results if r.get("status") == "success")
                logger.info(
                    f"  {i+1}/{n_total_planned} done — {successes} ok | "
                    f"RAM peak {mem_peak_run:.0f} MB | "
                    f"VRAM peak {vram_peak_inference_mib} MiB (nvidia) / "
                    f"{torch_vram_peak_inference_gb:.3f} GB (torch)"
                )
        except Exception as e:
            logger.warning(f"Segment {i} ({row.get('id','?')}) exception: {e}")
            results.append({
                "id": row.get("id", f"seg_{i}"),
                "status": "exception",
                "error": str(e),
            })

    # Final measurements
    final_mem_peak_mb = max(mem_peak_run, get_mem_mb())
    final_vram_mib = nvidia_smi_used_mib()
    vram_peak_inference_mib = max(vram_peak_inference_mib, final_vram_mib)
    ru_maxrss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    logger.info(f"Final peak RSS (ru_maxrss): {ru_maxrss_kb} KB = {ru_maxrss_kb/1024:.1f} MB")
    logger.info(f"Final VRAM peak during inference (nvidia-smi): {vram_peak_inference_mib} MiB")
    logger.info(f"Final torch VRAM peak during inference: {torch_vram_peak_inference_gb:.3f} GB")

    # Aggregate
    successful = [r for r in results if r.get("status") == "success"]
    n_success = len(successful)
    n_total = len(results)
    logger.info(f"Successful: {n_success}/{n_total}")

    if n_success < 20:
        logger.error(
            f"Only {n_success} successful segments — below N>=20 threshold. "
            "Results NOT reportable per eval protocol."
        )

    # Corpus BLEU
    from src.utils.metrics import compute_bleu
    if n_success > 0:
        hyps = [r["mt_text"] for r in successful]
        refs = [r["ref_tgt_text"] for r in successful]
        tokenize = "ko-mecab" if tgt_lang == "ko" else "13a"
        bleu, tokenize_used = compute_bleu(hyps, refs, tokenize=tokenize)

        if tgt_lang == "ko":
            assert tokenize_used == "ko-mecab", (
                f"PROTOCOL VIOLATION: ko-mecab tokenizer fell back to '{tokenize_used}'. "
                "This run is INVALID. Stop and investigate MeCab."
            )
            logger.info(f"BLEU tokenize_used assertion PASSED: {tokenize_used}")

        logger.info(f"Corpus BLEU ({tokenize_used}): {bleu:.2f}")
    else:
        bleu = -1.0
        tokenize_used = "n/a"

    # ASR stats
    asr_wers = [r["asr_wer_pct"] for r in successful if r.get("asr_wer_pct", -1) >= 0]
    mean_asr_wer = sum(asr_wers) / len(asr_wers) if asr_wers else -1.0

    # Latency stats
    e2e_ms_vals = sorted([r["e2e_latency_ms"] for r in successful])
    asr_first_s_vals = [r["asr_first_emission_s"] for r in successful]
    mt_ms_vals = [r["mt_ms"] for r in successful]
    tts_ms_vals = [r["tts_ms"] for r in successful]

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
    mean_asr_first_s = _mean(asr_first_s_vals)
    mean_mt_ms = _mean(mt_ms_vals)
    mean_tts_ms = _mean(tts_ms_vals)

    # Build report
    report = {
        "experiment": "P2-1 GPU Pipeline",
        "config": {
            "src_lang": src_lang,
            "tgt_lang": tgt_lang,
            "manifest": str(manifest_path),
            "limit": limit if limit > 0 else "all",
            "asr_model": "large-v3",
            "asr_device": "cuda",
            "asr_compute_type": "int8_float16",
            "asr_streaming": "LocalAgreement-2",
            "asr_initial_chunk_s": 1.0,
            "asr_step_s": 1.0,
            "asr_beam": 1,
            "asr_confidence_gate": None,
            "mt_model": "facebook/nllb-200-distilled-600M",
            "mt_device": "cuda",
            "mt_dtype": "fp16",
            "mt_beam": 1,
            "tts_engine": tts_engine,
            "tts_device": "cuda",
        },
        "aggregate_scores": {
            "n_segments_total": n_total,
            "n_segments_successful": n_success,
            "bleu": round(bleu, 2),
            "bleu_tokenize": tokenize_used,
            "asr_wer_mean_pct": round(mean_asr_wer, 2),
            "asr_first_emission_mean_s": round(mean_asr_first_s, 3),
            "mt_ms_mean": round(mean_mt_ms, 1),
            "tts_ms_mean": round(mean_tts_ms, 1),
            "e2e_latency_ms_mean": round(mean_e2e_ms, 1),
            "e2e_latency_ms_p50": round(p50_e2e_ms, 1),
            "e2e_latency_ms_p95": round(p95_e2e_ms, 1),
            "peak_ram_mb": round(final_mem_peak_mb, 1),
            "ru_maxrss_kb": ru_maxrss_kb,
            "ru_maxrss_mb": round(ru_maxrss_kb / 1024.0, 1),
            "vram_static_load_mib": vram_after_tts_mib,
            "vram_static_load_gb": round(vram_after_tts_mib / 1024.0, 3),
            "vram_peak_inference_mib": vram_peak_inference_mib,
            "vram_peak_inference_gb": round(vram_peak_inference_mib / 1024.0, 3),
            "torch_vram_peak_inference_gb": round(torch_vram_peak_inference_gb, 3),
            "vram_after_asr_mib": vram_after_asr_mib,
            "vram_after_mt_mib": vram_after_mt_mib,
            "vram_after_tts_mib": vram_after_tts_mib,
        },
        "per_segment_samples": results,  # ALL segments (N<=50 rule)
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    logger.info(f"Report saved -> {report_path}")

    # Print summary
    logger.info("-- P2-1 GPU Pipeline Eval Summary -------------------")
    logger.info(f"  Direction:          {src_lang}->{tgt_lang}")
    logger.info(f"  N (ok/total):       {n_success}/{n_total}")
    logger.info(f"  BLEU ({tokenize_used:<12}): {bleu:.2f}")
    logger.info(f"  ASR WER (%):        {mean_asr_wer:.2f}")
    logger.info(f"  ASR 1st-emit (s):   {mean_asr_first_s:.3f}")
    logger.info(f"  MT latency (ms):    {mean_mt_ms:.1f}")
    logger.info(f"  TTS latency (ms):   {mean_tts_ms:.1f}")
    logger.info(f"  E2E latency (ms):   {mean_e2e_ms:.1f}  p50={p50_e2e_ms:.1f}  p95={p95_e2e_ms:.1f}")
    logger.info(f"  Peak RAM (MB):      {final_mem_peak_mb:.1f}")
    logger.info(f"  ru_maxrss (KB/MB):  {ru_maxrss_kb} / {ru_maxrss_kb/1024:.1f}")
    logger.info(f"  VRAM static load:   {vram_after_tts_mib} MiB = {vram_after_tts_mib/1024:.2f} GB")
    logger.info(f"    ASR alone:        {vram_after_asr_mib} MiB")
    logger.info(f"    ASR+MT:           {vram_after_mt_mib} MiB")
    logger.info(f"    ASR+MT+TTS:       {vram_after_tts_mib} MiB")
    logger.info(f"  VRAM inference pk:  {vram_peak_inference_mib} MiB = {vram_peak_inference_mib/1024:.2f} GB")
    logger.info(f"  torch VRAM infer:   {torch_vram_peak_inference_gb:.3f} GB")
    logger.info("-----------------------------------------------------")

    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="P2-1 GPU Pipeline Evaluation")
    parser.add_argument("--manifest", required=True, help="Fleurs pair manifest TSV")
    parser.add_argument("--src", default="en", choices=["en", "ko"])
    parser.add_argument("--tgt", default="ko", choices=["en", "ko"])
    parser.add_argument("--report", required=True, help="Output JSON report path")
    parser.add_argument("--limit", type=int, default=0, help="Max segments (0=all)")
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
