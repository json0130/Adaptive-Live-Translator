#!/usr/bin/env python
"""Offline evaluation: BLEU + per-segment latency on a held-out test set.

TEXT MODE (--testset some.tsv):
  TSV: src_text \t ref_tgt_text
  Feeds source TEXT directly to the translator. ASR is bypassed.
  Latency reported = translator-only wall-clock per segment.

AUDIO MODE (--audio-manifest data/eval/fleurs_pair_manifest.tsv --src en --tgt ko):
  Reads the paired Fleurs manifest. Runs: audio -> ASR -> translator -> BLEU.
  Latency reported includes ASR+translator wall-clock; this is the "real"
  end-to-end pipeline latency (no TTS yet — that's measured separately).
  Reference text for BLEU comes from the manifest's tgt-side text column.

The Korean BLEU tokenizer is locked to sacrebleu --tokenize ko-mecab per
.claude/eval-protocol.md. Falls back to char if mecab-ko-dic is missing,
and records the fallback in `bleu_tokenize_used` so reviewer can verify.
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


def _load_asr(asr_cfg: dict):
    """Lightweight ASR loader for audio-mode eval.

    Reads asr.model + asr.compute_type from the translator config and
    returns a callable(audio_path, lang) -> (text, audio_seconds).
    """
    from faster_whisper import WhisperModel
    model_id = asr_cfg.get("model", "Systran/faster-whisper-medium")
    compute_type = asr_cfg.get("compute_type", "int8")
    logger.info(f"[audio-mode] loading ASR: {model_id} compute_type={compute_type}")
    model = WhisperModel(model_id, device="cpu", compute_type=compute_type)

    def transcribe(audio_path: str, lang: str) -> tuple[str, float]:
        segs, info = model.transcribe(
            audio_path, language=lang, beam_size=1, vad_filter=False
        )
        text = "".join(s.text for s in segs).strip()
        return text, float(info.duration)

    return transcribe


async def evaluate_audio(
    manifest_path: Path,
    src_lang: str,
    tgt_lang: str,
    cfg: dict,
) -> list[dict]:
    """Audio-mode: ASR -> translator -> BLEU on the paired Fleurs manifest."""
    from src.pipeline.session import SessionConfig, TranslationSession

    session_cfg = SessionConfig(src_lang=src_lang, tgt_lang=tgt_lang)
    session = TranslationSession(cfg, session_cfg)
    transcribe = _load_asr(cfg.get("asr", {}))

    manifest_dir = manifest_path.parent
    src_audio_col = f"{src_lang}_audio"
    tgt_text_col = f"{tgt_lang}_text"
    src_text_col = f"{src_lang}_text"

    rows: list[dict] = []
    with open(manifest_path) as f:
        header = f.readline().rstrip("\n").split("\t")
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < len(header):
                continue
            rows.append(dict(zip(header, parts)))

    results: list[dict] = []
    for i, row in enumerate(rows):
        audio_path = str(manifest_dir / row[src_audio_col])
        ref_text = row[tgt_text_col]
        src_text_ref = row[src_text_col]  # reference transcript (for ASR WER)
        try:
            t_asr0 = time.perf_counter()
            asr_text, audio_s = transcribe(audio_path, src_lang)
            asr_ms = (time.perf_counter() - t_asr0) * 1000

            rec = await _run_one(asr_text, ref_text, session, src_lang, tgt_lang)
            rec["asr_ms"] = round(asr_ms, 1)
            rec["audio_s"] = round(audio_s, 2)
            rec["asr_text"] = asr_text
            rec["src_text_ref"] = src_text_ref
            rec["latency_ms"] = round(rec["latency_ms"] + asr_ms, 1)  # combined
            results.append(rec)
            if (i + 1) % 25 == 0:
                logger.info(f"  {i+1}/{len(rows)} done")
        except Exception as exc:
            logger.warning(f"Row {i} failed: {exc}")

    session.close()
    return results


def compute_scores(results: list[dict], tgt_lang: str = "en") -> dict:
    from src.utils.metrics import compute_bleu, compute_streamlaal, SegmentRecord

    hyps = [r["hyp"] for r in results]
    refs = [r["ref"] for r in results]
    # Eval protocol: ko-mecab for ko targets, 13a otherwise.
    tokenize = "ko-mecab" if tgt_lang == "ko" else "13a"
    if any(refs):
        bleu, tokenize_used = compute_bleu(hyps, refs, tokenize=tokenize)
    else:
        bleu, tokenize_used = -1.0, "n/a"

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
        "bleu_tokenize": tokenize_used,
        "stream_laal_s": round(laal, 3),
        "avg_latency_ms": round(avg_latency, 1),
        "n_segments": len(results),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--testset", default=None,
                        help="TEXT mode: TSV of src \\t ref")
    parser.add_argument("--audio-manifest", default=None,
                        help="AUDIO mode: Fleurs paired manifest (id, en_audio, ko_audio, en_text, ko_text)")
    parser.add_argument("--src", default="en")
    parser.add_argument("--tgt", default="ko")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--report", default="reports/eval.json")
    args = parser.parse_args()

    if (args.testset is None) == (args.audio_manifest is None):
        raise SystemExit("Provide exactly one of --testset or --audio-manifest.")

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.audio_manifest:
        manifest_path = Path(args.audio_manifest)
        if not manifest_path.exists():
            logger.error(f"Audio manifest not found: {manifest_path}")
            return
        logger.info(f"AUDIO-mode | {manifest_path.name} | {args.src} → {args.tgt}")
        results = asyncio.run(
            evaluate_audio(manifest_path, args.src, args.tgt, cfg)
        )
    else:
        testset_path = Path(args.testset)
        if not testset_path.exists():
            logger.error(f"Test set not found: {testset_path}")
            return
        logger.info(f"TEXT-mode | {testset_path.name} | {args.src} → {args.tgt}")
        results = asyncio.run(
            evaluate(testset_path, args.src, args.tgt, cfg)
        )
    scores = compute_scores(results, tgt_lang=args.tgt)

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
