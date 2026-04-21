"""Abstract streaming ASR interface.

All ASR backends (Whisper local, Deepgram, faster-whisper) implement this.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import AsyncIterator

import numpy as np


@dataclass
class ASRChunk:
    """A partial ASR result for a chunk of audio."""

    text: str                      # cumulative stable transcript so far
    delta: str                     # new text since previous chunk
    start_ms: int                  # absolute start time of this chunk
    end_ms: int                    # absolute end time of this chunk
    is_final: bool                 # True when policy says "emit"
    language: str | None = None    # detected source language


class ASRStreamer(ABC):
    """Base class for a streaming ASR system.

    Usage:
        streamer = WhisperStreaming(cfg)
        async for chunk in streamer.stream(audio_iter):
            ...
    """

    @abstractmethod
    async def stream(
        self,
        audio_iter: AsyncIterator[np.ndarray],
        *,
        initial_prompt: str | None = None,
        context_prompt: str | None = None,
    ) -> AsyncIterator[ASRChunk]:
        """Consume PCM audio chunks and yield ASRChunks.

        Args:
            audio_iter: async iterator of float32 PCM arrays at 16 kHz
            initial_prompt: domain/meeting description injected at t=0
            context_prompt: rolling transcript from previous segments
        """
        ...

    @abstractmethod
    def reset(self) -> None:
        """Clear buffers between sessions."""
        ...
