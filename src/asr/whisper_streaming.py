"""Whisper large-v3 streaming wrapper with AlignAtt policy.

Implementation notes:
  - Uses faster-whisper for lower latency than HF transformers
  - AlignAtt policy decides READ vs WRITE based on cross-attention
  - Supports initial_prompt (domain terms) and context (rolling transcript)
"""
from __future__ import annotations

from typing import AsyncIterator

import numpy as np
from loguru import logger

from .base import ASRChunk, ASRStreamer


class WhisperStreaming(ASRStreamer):
    """Streaming ASR via Whisper large-v3 + AlignAtt.

    TODO: wire up faster-whisper. This is a scaffolding stub.
    """

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        self.model_name = cfg["asr"]["model"]
        self.chunk_seconds = cfg["asr"]["chunk_seconds"]
        self.device = cfg["asr"].get("device", "cuda")
        self._model = None
        self._buffer: list[np.ndarray] = []
        self._committed_text: str = ""

    def _lazy_load(self) -> None:
        if self._model is not None:
            return
        logger.info(f"Loading Whisper: {self.model_name}")
        # from faster_whisper import WhisperModel
        # self._model = WhisperModel(
        #     self.model_name, device=self.device,
        #     compute_type=self.cfg["asr"]["compute_type"],
        # )
        self._model = "STUB"

    def reset(self) -> None:
        self._buffer.clear()
        self._committed_text = ""

    async def stream(
        self,
        audio_iter: AsyncIterator[np.ndarray],
        *,
        initial_prompt: str | None = None,
        context_prompt: str | None = None,
    ) -> AsyncIterator[ASRChunk]:
        self._lazy_load()
        cursor_ms = 0

        async for pcm in audio_iter:
            self._buffer.append(pcm)
            samples = sum(len(b) for b in self._buffer)
            chunk_samples = int(self.chunk_seconds * 16000)
            if samples < chunk_samples:
                continue

            audio = np.concatenate(self._buffer)
            self._buffer = []

            # TODO: real transcription call
            # segments, _ = self._model.transcribe(
            #     audio,
            #     initial_prompt=initial_prompt,
            #     language=None,
            #     vad_filter=self.cfg["asr"]["vad_enabled"],
            #     beam_size=5,
            # )
            # delta = "".join(s.text for s in segments)
            delta = "[stub transcript]"

            self._committed_text += delta
            yield ASRChunk(
                text=self._committed_text,
                delta=delta,
                start_ms=cursor_ms,
                end_ms=cursor_ms + int(len(audio) / 16000 * 1000),
                is_final=True,
            )
            cursor_ms += int(len(audio) / 16000 * 1000)
