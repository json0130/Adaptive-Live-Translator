"""Speaker identification and diarization.

Used to fire LoRA adapter swaps when the speaker changes in a multi-speaker
meeting. Falls back to a single-speaker "default" if disabled.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import AsyncIterator

import numpy as np
from loguru import logger


@dataclass
class SpeakerSegment:
    speaker_id: str
    start_ms: int
    end_ms: int
    audio: np.ndarray


class SpeakerIdentifier:
    """Wraps pyannote.audio for diarization (or a simpler x-vector approach).

    TODO: wire up pyannote.audio >= 3.3 when diarization is needed.
    """

    def __init__(self, enabled: bool = False) -> None:
        self.enabled = enabled
        self._pipeline = None

    def _lazy_load(self) -> None:
        if not self.enabled or self._pipeline is not None:
            return
        logger.info("Loading speaker diarization pipeline (pyannote.audio)...")
        # from pyannote.audio import Pipeline
        # self._pipeline = Pipeline.from_pretrained(
        #     "pyannote/speaker-diarization-3.1",
        #     use_auth_token=os.environ["HF_TOKEN"],
        # )
        self._pipeline = "STUB"

    async def stream_segments(
        self,
        audio_iter: AsyncIterator[np.ndarray],
    ) -> AsyncIterator[SpeakerSegment]:
        """Yield segments annotated with speaker_id."""
        self._lazy_load()
        cursor_ms = 0
        async for pcm in audio_iter:
            duration_ms = int(len(pcm) / 16000 * 1000)
            # TODO: real diarization call
            speaker_id = "default"
            yield SpeakerSegment(
                speaker_id=speaker_id,
                start_ms=cursor_ms,
                end_ms=cursor_ms + duration_ms,
                audio=pcm,
            )
            cursor_ms += duration_ms
