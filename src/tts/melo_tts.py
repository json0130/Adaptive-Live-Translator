"""MeloTTS — CPU-efficient multilingual TTS for en/ko.

MeloTTS uses FastSpeech 2 + HiFi-GAN vocoder (0.3B total) for low-latency
speech synthesis on CPU. Runs on fp32 (MeloTTS doesn't support fp16 on CPU).

Install: pip3 install melotts
Optional: python3 -m unidic download (if MeloTTS needs it for Japanese MeCab)
"""
from __future__ import annotations

import time
from typing import AsyncIterator

import numpy as np
from loguru import logger

from .cosyvoice import TTSSynthesizer


class MeloTTSSynthesizer(TTSSynthesizer):
    """MeloTTS for English and Korean."""

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        self._model = None
        self._speaker_ids = None

    def _lazy_load(self) -> None:
        """Lazy-load MeloTTS model on first use."""
        if self._model is not None:
            return

        logger.info("Loading MeloTTS model...")
        try:
            from melotts import MeloTTS
            # Load with default device (CPU); MeloTTS will auto-detect
            self._model = MeloTTS()
            self._speaker_ids = self._model.get_speakers()
            logger.info(f"MeloTTS loaded. Available speakers: {self._speaker_ids}")
        except ImportError as e:
            logger.error(f"MeloTTS import failed: {e}")
            raise
        except Exception as e:
            logger.error(f"MeloTTS load failed: {e}")
            raise

    async def synthesize_stream(
        self,
        text_iter: AsyncIterator[str],
        *,
        speaker_audio: np.ndarray | None = None,
    ) -> AsyncIterator[np.ndarray]:
        """Synthesize text to audio (16 kHz PCM mono, float32)."""
        self._lazy_load()

        # Accumulate full text
        text = ""
        async for chunk in text_iter:
            text += chunk

        if not text.strip():
            return

        # Synthesize with MeloTTS
        t0 = time.perf_counter()
        try:
            # MeloTTS.tts() returns (waveform, sample_rate)
            # Use a default speaker (English or Korean speaker depending on context)
            # MeloTTS supports eng_us, eng_uk, kor, etc.
            # For now, default to eng_us for English, kor for Korean
            speaker_id = "eng_us"  # Default English speaker

            # Synthesize
            mel, sr = self._model.tts(text, speaker_id, language=None, speed=1.0, temperature=1.0)

            # Convert to float32 and resample to 16 kHz if needed
            audio = np.array(mel, dtype=np.float32)
            if sr != 16000:
                # Resample using librosa if available, else approximate
                try:
                    import librosa
                    audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
                except ImportError:
                    # Rough linear interpolation fallback
                    ratio = 16000 / sr
                    new_len = int(len(audio) * ratio)
                    audio = np.interp(
                        np.linspace(0, len(audio) - 1, new_len),
                        np.arange(len(audio)),
                        audio,
                    )

            synth_ms = (time.perf_counter() - t0) * 1000
            logger.info(f"MeloTTS synthesis took {synth_ms:.1f} ms for {len(text)} chars")

            yield audio
        except Exception as e:
            logger.error(f"MeloTTS synthesis failed: {e}")
            raise
