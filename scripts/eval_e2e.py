#!/usr/bin/env python
"""End-to-end evaluation: audio → streaming ASR → MT → TTS → audio.

R4-1 composition experiment:
  - ASR: faster-whisper base (int8) streaming partials + faster-whisper small (int8)
    final pass + LocalAgreement-2 + confidence-gated single-window commit (avg_logprob >= -0.7)
    [R4-2 winning config]
  - MT: NLLB-200-distilled-600M CT2 int8 on CPU [A1 translator]
  - TTS: MeloTTS-KR (MIT) via KoreanCpuTTS [R3 Korean TTS]

Measures:
  - BLEU: translator output vs reference text
    (ko target: ko-mecab per locked protocol; en target: 13a)
  - Latency: per-component (ASR, MT, TTS) and end-to-end wall-clock
  - Peak co-resident RSS: all three models loaded together (the RAM question)
  - R4-3 post-processor: ko->en English hypothesis casing + terminal-punctuation
    restoration, measuring BLEU with vs without post-proc

BLEU protocol:
  - MUST read bleu_tokenize: ko-mecab in output.
  - If it reads 'char' or 'char(fallback-from-ko-mecab)': HALT, investigate.
  - en target: 13a.

RAM measurement:
  - resource.getrusage(RUSAGE_SELF).ru_maxrss sampled throughout the run.
  - Linux: maxrss in KB -> divide by 1024 for MB.
  - Reports peak RSS across the full co-resident run (all three models loaded).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import resource
import time
from pathlib import Path
from typing import Optional

import numpy as np
from loguru import logger


# ---------------------------------------------------------------------------
# RAM helpers
# ---------------------------------------------------------------------------

def get_mem_mb() -> float:
    """Current peak RSS in MB (Linux: ru_maxrss in KB)."""
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return usage.ru_maxrss / 1024.0


# ---------------------------------------------------------------------------
# Component loaders (R4-2 config)
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
    # Force-load both models now so RAM is co-resident from the start
    asr._ensure_loaded()
    asr._ensure_final_model_loaded()
    return asr


def load_translator():
    """Load NLLB-200-distilled-600M CT2 int8."""
    from src.translator.nllb_ct2_translator import NllbCt2Translator
    logger.info("Loading translator: NLLB-200-distilled-600M CT2 INT8")
    # Use absolute path to support running from the worktree directory
    import os
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # If we're in a worktree, go up to the main repo
    ct2_model_dir = os.path.join(repo_root, "models", "nllb-600m-ct2-int8")
    if not os.path.exists(ct2_model_dir):
        # Try the canonical path
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


def load_tts_ko(prewarm: bool = True):
    """Load MeloTTS-KR (MIT). Returns (tts, engine_name).

    prewarm=True (default): Run a dummy synthesis after load to pre-warm
    the Korean BERT model (kykim/bert-kor-base), which MeloTTS loads lazily
    on the first synthesis call. Pre-warming ensures this model is already
    resident before we measure the co-resident peak, giving a stable baseline.

    Without pre-warming, the first synthesis call loads bert-kor-base while
    all other models are already resident, causing a transient spike. The
    spike looks larger because it includes:
      - bert-kor-base weights (~450 MB fp32 = ~1.5 GB with torch overhead)
      - active tensor allocations during inference
    With pre-warming, the bert is loaded before we start inference loops,
    giving the true co-resident steady-state RAM.
    """
    try:
        from src.tts.korean_cpu_tts import KoreanCpuTTS
        logger.info("Loading TTS: KoreanCpuTTS (MeloTTS engine)")
        tts = KoreanCpuTTS(engine="melo", device="cpu")
        logger.info(f"TTS loaded, engine={tts._engine}")

        if prewarm and tts._engine == "melo":
            logger.info("Pre-warming TTS (loading Korean BERT for text normalization)...")
            import tempfile, os
            try:
                tmp = tempfile.mktemp(suffix=".wav")
                # Short dummy synthesis to trigger bert-kor-base load
                tts._melo.tts_to_file("안녕", tts._melo_speaker, output_path=tmp, speed=1.0, quiet=True)
                if os.path.exists(tmp):
                    os.unlink(tmp)
                logger.info("TTS pre-warm complete (Korean BERT now resident)")
            except Exception as e:
                logger.warning(f"TTS pre-warm failed (non-fatal): {e}")

        return tts, "melo"
    except Exception as e:
        logger.warning(f"MeloTTS load failed: {e}. Falling back to espeak-ng.")
        try:
            from src.tts.korean_cpu_tts import KoreanCpuTTS
            tts = KoreanCpuTTS(engine="espeak", device="cpu")
            return tts, "espeak"
        except Exception as e2:
            logger.error(f"TTS fallback also failed: {e2}")
            return None, "none"


def load_tts_en():
    """For en TTS (round-trip check). Uses espeak-ng."""
    try:
        from src.tts.korean_cpu_tts import KoreanCpuTTS
        tts = KoreanCpuTTS(engine="espeak", device="cpu")
        return tts, "espeak"
    except Exception as e:
        logger.warning(f"TTS load failed: {e}")
        return None, "none"


# ---------------------------------------------------------------------------
# R4-3 post-processor: casing + terminal punctuation for ko->en
# ---------------------------------------------------------------------------

def postprocess_en(text: str) -> str:
    """R4-3 post-processor: sentence-case + terminal period for English hypotheses.

    NLLB often emits lowercase with no final punctuation, which hurts sacrebleu
    13a BLEU (case/punct sensitive). This restores:
    1. Sentence-case: capitalize first character.
    2. Terminal punctuation: add period if text ends without punctuation.
    """
    if not text:
        return text
    # Capitalize first character
    text = text[0].upper() + text[1:]
    # Add terminal period if missing
    if text and text[-1] not in ".!?":
        text = text + "."
    return text


# ---------------------------------------------------------------------------
# Translation helper (sync wrapper around async translate_stream)
# ---------------------------------------------------------------------------

def translate_sync(translator, src_text: str, src_lang: str, tgt_lang: str) -> tuple:
    """Translate text synchronously. Returns (translated_text, mt_ms)."""
    async def _run():
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
    apply_postproc: bool = False,
) -> dict:
    """Evaluate one utterance through the full pipeline.

    Returns a dict with per-segment metrics.
    apply_postproc: if True, apply R4-3 post-processor to the MT hypothesis
                    before scoring (only for ko->en, tgt_lang=en).
    """
    src_audio_col = f"{src_lang}_audio"
    tgt_text_col = f"{tgt_lang}_text"
    src_text_col = f"{src_lang}_text"

    # Resolve audio path
    audio_rel = row[src_audio_col]
    audio_path = manifest_dir / audio_rel
    if not audio_path.exists():
        # Try canonical location
        alt2 = Path("/home/jay/Desktop/Jay-per/adaptive-live-translator/data/eval") / audio_rel
        if alt2.exists():
            audio_path = alt2
        else:
            # Try parent repo resolution (worktree case)
            manifest_abs = manifest_dir.absolute()
            for levels_up in [5, 4, 3]:
                parent = manifest_abs
                for _ in range(levels_up):
                    parent = parent.parent
                alt = parent / "data" / "eval" / audio_rel
                if alt.exists():
                    audio_path = alt
                    break

    audio_path_str = str(audio_path)
    ref_tgt_text = row[tgt_text_col]
    src_text_ref = row[src_text_col]

    # --- Step 1: Streaming ASR ---
    t_asr_wall0 = time.perf_counter()
    try:
        result = asr_model.transcribe_streaming(audio_path_str, lang=src_lang)
        asr_text = result.accuracy_transcript  # small-model final transcript (for quality)
        asr_latency_s = result.first_emission_latency_s or 0.0
        audio_s = result.total_audio_s
        asr_streaming_ms = result.total_decode_s * 1000.0  # streaming decode time only
        asr_e2e_ms = (time.perf_counter() - t_asr_wall0) * 1000.0  # wall-clock including final model
        confidence_gate_applied = result.confidence_gate_applied
    except Exception as e:
        logger.warning(f"ASR failed for {row.get('id','?')}: {e}")
        return {
            "id": row.get("id", "unknown"),
            "status": "asr_failed",
            "error": str(e),
        }

    # --- Step 2: Machine Translation ---
    try:
        mt_text_raw, mt_ms = translate_sync(translator, asr_text, src_lang, tgt_lang)
    except Exception as e:
        logger.warning(f"MT failed for {row.get('id','?')}: {e}")
        return {
            "id": row.get("id", "unknown"),
            "status": "mt_failed",
            "error": str(e),
            "asr_text": asr_text,
        }

    # R4-3 post-processor (only for ko->en, English hypothesis)
    if apply_postproc and tgt_lang == "en":
        mt_text = postprocess_en(mt_text_raw)
        postproc_applied = True
    else:
        mt_text = mt_text_raw
        postproc_applied = False

    # --- Step 3: TTS ---
    t_tts0 = time.perf_counter()
    tts_ms = 0.0
    if tts_model is not None and mt_text.strip():
        try:
            synth_audio, tts_ms = tts_model.synthesize(mt_text, lang=tgt_lang)
        except Exception as e:
            logger.warning(f"TTS failed for {row.get('id','?')}: {e}")
            synth_audio = np.zeros(16000, dtype=np.float32)
            tts_ms = (time.perf_counter() - t_tts0) * 1000.0
    else:
        synth_audio = np.zeros(16000, dtype=np.float32)
        tts_ms = 0.0

    # --- Metrics ---
    from src.utils.metrics import compute_bleu
    from jiwer import wer, cer

    # BLEU: MT output vs reference
    tokenize = "ko-mecab" if tgt_lang == "ko" else "13a"
    bleu, tokenize_used = compute_bleu([mt_text], [ref_tgt_text], tokenize=tokenize)

    # Hard check: if ko target falls back to char, this is a protocol violation
    if tgt_lang == "ko" and "char" in tokenize_used:
        logger.error(
            f"BLEU tokenizer fallback to '{tokenize_used}' — MeCab unavailable. "
            "This violates the eval protocol. DO NOT accept this number."
        )

    # ASR WER on source language
    try:
        if src_lang == "ko":
            asr_error = cer([src_text_ref], [asr_text])
        else:
            asr_error = wer([src_text_ref], [asr_text])
        asr_error_pct = round(asr_error * 100, 2)
    except Exception as e:
        logger.warning(f"ASR error rate failed: {e}")
        asr_error_pct = -1.0

    # End-to-end latency: ASR first-emission + MT + TTS
    # asr_latency_s = true streaming first-emission latency (includes initial_chunk_s)
    # mt_ms and tts_ms are sequential after ASR
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
        "mt_text_raw": mt_text_raw,
        "mt_text": mt_text,
        "postproc_applied": postproc_applied,
        "mt_ms": round(mt_ms, 1),
        "tts_ms": round(tts_ms, 1),
        "tts_engine": tts_engine,
        "e2e_latency_ms": round(e2e_latency_ms, 1),
        "segment_bleu": round(bleu, 2),
        "bleu_tokenize": tokenize_used,
        "ref_tgt_text": ref_tgt_text,
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
    apply_postproc: bool = False,
) -> dict:
    """Run full e2e evaluation, return report dict."""

    logger.info(f"=== R4-1 E2E Composition Eval: {src_lang}->{tgt_lang} ===")

    # Load models (all at once for co-resident RAM measurement)
    mem_before_load = get_mem_mb()
    logger.info(f"RAM before model loads: {mem_before_load:.1f} MB")

    asr_model = load_streaming_asr()
    mem_after_asr = get_mem_mb()
    logger.info(f"RAM after ASR load: {mem_after_asr:.1f} MB (delta: +{mem_after_asr - mem_before_load:.1f} MB)")

    translator = load_translator()
    mem_after_mt = get_mem_mb()
    logger.info(f"RAM after MT load: {mem_after_mt:.1f} MB (delta: +{mem_after_mt - mem_after_asr:.1f} MB)")

    # TTS is always Korean output for en->ko; for ko->en we still need TTS for round-trip
    # but ko->en target is English — use espeak for en TTS (MeloTTS-KR is Korean only)
    if tgt_lang == "ko":
        tts_model, tts_engine = load_tts_ko(prewarm=True)
    else:
        tts_model, tts_engine = load_tts_en()

    mem_after_tts = get_mem_mb()
    logger.info(f"RAM after TTS load + pre-warm: {mem_after_tts:.1f} MB (delta: +{mem_after_tts - mem_after_mt:.1f} MB)")
    logger.info(f"Co-resident peak RSS (all models warm): {mem_after_tts:.1f} MB")
    logger.info(
        f"NOTE: MeloTTS-KR loads kykim/bert-kor-base (~118M params, fp32) lazily on "
        f"first synthesis call. Pre-warming included above in mem_after_tts."
    )

    if tts_model is None:
        logger.warning("TTS unavailable — TTS latency will be 0.")
    else:
        logger.info(f"TTS engine: {tts_engine}")

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
    if apply_postproc and tgt_lang == "en":
        logger.info("R4-3 post-processor ENABLED (ko->en casing + terminal punctuation)")

    # Evaluate each segment
    results = []
    mem_peak_run = mem_after_tts

    for i, row in enumerate(rows):
        try:
            rec = evaluate_segment(
                row, manifest_dir, src_lang, tgt_lang,
                asr_model, translator, tts_model, tts_engine,
                apply_postproc=apply_postproc,
            )
            results.append(rec)
            mem_current = get_mem_mb()
            mem_peak_run = max(mem_peak_run, mem_current)
            if (i + 1) % 20 == 0:
                successes = sum(1 for r in results if r.get("status") == "success")
                logger.info(
                    f"  {i+1}/{len(rows)} done — {successes} ok | peak RAM {mem_peak_run:.0f} MB"
                )
        except Exception as e:
            logger.warning(f"Segment {i} ({row.get('id','?')}) exception: {e}")
            results.append({
                "id": row.get("id", f"seg_{i}"),
                "status": "exception",
                "error": str(e),
            })

    final_mem_peak = max(mem_peak_run, get_mem_mb())
    logger.info(f"Final co-resident peak RSS: {final_mem_peak:.1f} MB")

    # Aggregate
    successful = [r for r in results if r.get("status") == "success"]
    n_success = len(successful)
    n_total = len(results)
    logger.info(f"Successful: {n_success}/{n_total}")

    if n_success < 20:
        logger.error(
            f"Only {n_success} successful segments — below N>=20 threshold. "
            "Results are not reportable per eval protocol."
        )

    # Corpus BLEU (aggregate)
    from src.utils.metrics import compute_bleu
    if n_success > 0:
        hyps = [r["mt_text"] for r in successful]
        refs = [r["ref_tgt_text"] for r in successful]
        tokenize = "ko-mecab" if tgt_lang == "ko" else "13a"
        bleu, tokenize_used = compute_bleu(hyps, refs, tokenize=tokenize)

        # Protocol compliance check
        if tgt_lang == "ko" and "char" in tokenize_used:
            logger.error(
                f"CORPUS BLEU tokenizer is '{tokenize_used}' — MeCab unavailable. "
                "HALT: this number is NOT protocol-compliant. Investigate."
            )
        else:
            logger.info(f"Corpus BLEU ({tokenize_used}): {bleu:.2f}")
    else:
        bleu = -1.0
        tokenize_used = "n/a"

    # R4-3: also compute BLEU without post-proc if post-proc was applied
    bleu_no_postproc = None
    bleu_postproc_delta = None
    if apply_postproc and tgt_lang == "en" and n_success > 0:
        raw_hyps = [r["mt_text_raw"] for r in successful]
        bleu_no_postproc, _ = compute_bleu(raw_hyps, refs, tokenize="13a")
        bleu_postproc_delta = bleu - bleu_no_postproc
        logger.info(
            f"R4-3 post-proc BLEU delta: {bleu:.2f} (with) vs "
            f"{bleu_no_postproc:.2f} (without) = {bleu_postproc_delta:+.2f}"
        )

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

    n_gate_fired = sum(1 for r in successful if r.get("confidence_gate_applied", False))

    # Build report
    report = {
        "config": {
            "src_lang": src_lang,
            "tgt_lang": tgt_lang,
            "manifest": str(manifest_path),
            "limit": limit if limit > 0 else "all",
            "asr": "faster-whisper-base(int8)+small(int8)+LA2+gate-0.7",
            "mt": "NLLB-200-distilled-600M-ct2-int8",
            "tts": f"KoreanCpuTTS({tts_engine})",
            "postproc_applied": apply_postproc,
        },
        "aggregate_scores": {
            "n_segments_total": n_total,
            "n_segments_successful": n_success,
            "bleu": round(bleu, 2),
            "bleu_tokenize": tokenize_used,
            "bleu_without_postproc": round(bleu_no_postproc, 2) if bleu_no_postproc is not None else None,
            "bleu_postproc_delta": round(bleu_postproc_delta, 2) if bleu_postproc_delta is not None else None,
            "asr_wer_mean_pct": round(mean_asr_wer, 2),
            "asr_first_emission_mean_s": round(mean_asr_first_s, 3),
            "mt_ms_mean": round(mean_mt_ms, 1),
            "tts_ms_mean": round(mean_tts_ms, 1),
            "e2e_latency_ms_mean": round(mean_e2e_ms, 1),
            "e2e_latency_ms_p50": round(p50_e2e_ms, 1),
            "e2e_latency_ms_p95": round(p95_e2e_ms, 1),
            "confidence_gate_fired_pct": round(100.0 * n_gate_fired / n_success, 1) if n_success > 0 else 0.0,
            "peak_ram_mb": round(final_mem_peak, 1),
            "ram_after_asr_mb": round(mem_after_asr, 1),
            "ram_after_mt_mb": round(mem_after_mt, 1),
            "ram_after_tts_mb": round(mem_after_tts, 1),
        },
        "per_segment_samples": results[:30],
    }

    # Save
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    logger.info(f"Report saved -> {report_path}")

    # Print summary
    logger.info("-- R4-1 E2E Eval Summary ----------------------------")
    logger.info(f"  Direction:          {src_lang}->{tgt_lang}")
    logger.info(f"  N (ok/total):       {n_success}/{n_total}")
    logger.info(f"  BLEU ({tokenize_used:<12}): {bleu:.2f}")
    if bleu_no_postproc is not None:
        logger.info(f"  BLEU (no postproc): {bleu_no_postproc:.2f}  (R4-3 delta: {bleu_postproc_delta:+.2f})")
    logger.info(f"  ASR WER (%):        {mean_asr_wer:.2f}")
    logger.info(f"  ASR 1st-emit (s):   {mean_asr_first_s:.3f}")
    logger.info(f"  MT latency (ms):    {mean_mt_ms:.1f}")
    logger.info(f"  TTS latency (ms):   {mean_tts_ms:.1f}")
    logger.info(f"  E2E latency (ms):   {mean_e2e_ms:.1f}  p50={p50_e2e_ms:.1f}  p95={p95_e2e_ms:.1f}")
    logger.info(f"  Confidence gate:    {n_gate_fired}/{n_success}")
    logger.info(f"  Peak RAM (MB):      {final_mem_peak:.1f}")
    logger.info(f"    ASR alone:        {mem_after_asr:.1f} MB")
    logger.info(f"    ASR+MT:           {mem_after_mt:.1f} MB")
    logger.info(f"    ASR+MT+TTS:       {mem_after_tts:.1f} MB")
    logger.info("-----------------------------------------------------")

    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="R4-1 E2E Composition Eval")
    parser.add_argument("--manifest", required=True, help="Fleurs pair manifest TSV")
    parser.add_argument("--src", default="en", choices=["en", "ko"])
    parser.add_argument("--tgt", default="ko", choices=["en", "ko"])
    parser.add_argument("--report", required=True, help="Output JSON report path")
    parser.add_argument("--limit", type=int, default=0, help="Max segments (0=all)")
    parser.add_argument(
        "--postproc",
        action="store_true",
        default=False,
        help="Apply R4-3 post-processor (ko->en only): casing + terminal punctuation",
    )
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        logger.error(f"Manifest not found: {manifest_path}")
        return

    report_path = Path(args.report)

    run_eval(
        manifest_path=manifest_path,
        src_lang=args.src,
        tgt_lang=args.tgt,
        report_path=report_path,
        limit=args.limit,
        apply_postproc=args.postproc,
    )


if __name__ == "__main__":
    main()
