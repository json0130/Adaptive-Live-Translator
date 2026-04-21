"""Streaming TTS — CosyVoice 2 (0.5B) with optional voice cloning."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import AsyncIterator

import numpy as np
from loguru import logger


class TTSSynthesizer(ABC):
    @abstractmethod
    async def synthesize_stream(
        self,
        text_iter: AsyncIterator[str],
        *,
        speaker_audio: np.ndarray | None = None,
    ) -> AsyncIterator[np.ndarray]:
        """Consume text chunks, yield PCM audio chunks (float32, 22050 Hz)."""
        ...


class CosyVoice2(TTSSynthesizer):
    """CosyVoice 2 (0.5B) — 150ms streaming latency, 30+ languages.

    Voice cloning: pass ~10 s of reference audio via speaker_audio.
    Install: https://github.com/FunAudioLLM/CosyVoice
    """

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        self.model_name = cfg["tts"]["model"]
        self.streaming = cfg["tts"].get("streaming", True)
        self._model = None

    def _lazy_load(self) -> None:
        if self._model is not None:
            return
        logger.info(f"Loading TTS: {self.model_name}")
        # from cosyvoice.cli.cosyvoice import CosyVoice2 as _CosyVoice2
        # self._model = _CosyVoice2(self.model_name, load_jit=True, load_onnx=False)
        self._model = "STUB"

    async def synthesize_stream(
        self,
        text_iter: AsyncIterator[str],
        *,
        speaker_audio: np.ndarray | None = None,
    ) -> AsyncIterator[np.ndarray]:
        self._lazy_load()
        async for text in text_iter:
            if not text.strip():
                continue
            # TODO: real synthesis
            # if speaker_audio is not None:
            #     for chunk in self._model.inference_zero_shot(
            #         text, prompt_text="", prompt_speech_16k=speaker_audio, stream=True
            #     ):
            #         yield chunk["tts_speech"].numpy()
            # else:
            #     for chunk in self._model.inference_sft(text, stream=True):
            #         yield chunk["tts_speech"].numpy()
            samples = np.zeros(int(0.5 * 22050), dtype=np.float32)  # stub silence
            yield samples
