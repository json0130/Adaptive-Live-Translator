"""Streaming loop — the main ASR → context → translator → TTS pipeline.

Data flow per chunk:
  1.  PCM audio chunk in
  2.  ASR: audio → source text delta + AlignAtt policy
  3.  Context: build prompt (glossary hits + TM + rolling context)
  4.  Translator: source delta → target delta (KV cache reuse)
  5.  (Optional) TTS: target text → audio chunk out
  6.  Emit TranslationEvent over the output queue

Latency budget (low regime):
  ASR chunk:      2.0 s
  RAG retrieval:  ~5 ms
  LLM decode:     ~80-150 ms (Qwen2.5-7B int4, RTX 4090)
  TTS synthesis:  ~150 ms (CosyVoice 2 streaming)
  ─────────────────────────
  Target wall:    ≤ 2.5 s StreamLAAL
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import AsyncIterator

import numpy as np
from loguru import logger

from .session import TranslationSession


@dataclass
class TranslationEvent:
    """Emitted per committed translation chunk."""
    src_delta: str
    tgt_delta: str
    src_cumulative: str
    tgt_cumulative: str
    latency_ms: float
    is_final: bool
    audio: np.ndarray | None = None   # PCM if TTS is enabled


async def run_streaming_loop(
    session: TranslationSession,
    audio_queue: asyncio.Queue,          # producer puts np.ndarray PCM chunks
    output_queue: asyncio.Queue,         # consumer reads TranslationEvent
    *,
    tts_enabled: bool = False,
) -> None:
    """Main async loop. Run as a background task per session.

    Args:
        session:      initialised TranslationSession
        audio_queue:  async queue of float32 PCM arrays (16 kHz mono)
        output_queue: events are put here for the WebSocket handler to forward
    """
    src_cumulative = ""
    tgt_cumulative = ""

    tts = None
    if tts_enabled:
        from ..tts.cosyvoice import CosyVoice2
        tts = CosyVoice2(session.cfg)

    async def _audio_iter() -> AsyncIterator[np.ndarray]:
        while True:
            chunk = await audio_queue.get()
            if chunk is None:          # sentinel: end of stream
                return
            yield chunk

    # ---- ASR stream
    initial_prompt = session.ctx.topic_summary or None
    asr_stream = session.asr.stream(
        _audio_iter(),
        initial_prompt=initial_prompt,
        context_prompt=None,
    )

    async for asr_chunk in asr_stream:
        if not asr_chunk.is_final or not asr_chunk.delta.strip():
            continue

        t_start = time.perf_counter()
        src_delta = asr_chunk.delta
        src_cumulative += src_delta

        # ---- Build context-aware system prompt
        system_prompt = session.build_system_prompt(src_delta)

        # ---- Translate
        tgt_delta = ""

        async def _src_iter():
            yield src_delta

        async for t_chunk in session.translator.translate_stream(
            _src_iter(),
            src_lang=session.session_cfg.src_lang,
            tgt_lang=session.session_cfg.tgt_lang,
            system_prompt=system_prompt,
        ):
            tgt_delta = t_chunk.delta

        tgt_cumulative += tgt_delta
        latency_ms = (time.perf_counter() - t_start) * 1000

        # ---- Optional TTS
        audio_out: np.ndarray | None = None
        if tts is not None:
            async def _tgt_iter():
                yield tgt_delta

            chunks = []
            async for pcm in tts.synthesize_stream(_tgt_iter()):
                chunks.append(pcm)
            if chunks:
                audio_out = np.concatenate(chunks)

        # ---- Emit
        event = TranslationEvent(
            src_delta=src_delta,
            tgt_delta=tgt_delta,
            src_cumulative=src_cumulative,
            tgt_cumulative=tgt_cumulative,
            latency_ms=latency_ms,
            is_final=True,
            audio=audio_out,
        )
        await output_queue.put(event)

        # ---- Update rolling context
        session.push_segment(src_delta, tgt_delta)

        # ---- Record metrics
        session.metrics.append({
            "src": src_delta,
            "tgt": tgt_delta,
            "latency_ms": latency_ms,
        })
        logger.debug(
            f"[{session.session_id}] "
            f"{src_delta[:40]!r} → {tgt_delta[:40]!r} | {latency_ms:.0f} ms"
        )

    # Signal downstream that this session is done
    await output_queue.put(None)
    logger.info(f"[{session.session_id}] Streaming loop finished.")
