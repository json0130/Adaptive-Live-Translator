"""Korean CPU TTS — MeloTTS (MIT) with espeak-ng fallback.

Exposes: synthesize(text, lang) -> (np.ndarray 16kHz mono float32, synth_ms)

Engine priority:
  1. MeloTTS from source (MIT, Korean VITS model, myshell-ai/MeloTTS)
  2. espeak-ng (GPL v3, fallback only — known-bad 71% round-trip WER)
"""
from __future__ import annotations

import subprocess
import tempfile
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

class KoreanCpuTTS:
    """Unified Korean CPU TTS.  Call synthesize() for audio.

    Args:
        engine: 'melo' | 'espeak' | 'auto' (default: auto, tries melo first)
        device: 'cpu' (only CPU supported here)
    """

    SAMPLE_RATE = 16_000  # output sample rate (after resampling)

    def __init__(self, engine: str = "auto", device: str = "cpu") -> None:
        self.device = device
        self._melo: Optional[object] = None
        self._melo_speaker: Optional[int] = None

        if engine == "melo":
            self._load_melo()
            self._engine = "melo"
        elif engine == "espeak":
            self._engine = "espeak"
        else:  # auto
            try:
                self._load_melo()
                self._engine = "melo"
            except Exception as e:
                import warnings
                warnings.warn(f"MeloTTS load failed ({e}), falling back to espeak-ng")
                self._engine = "espeak"

    def _load_melo(self) -> None:
        from melo.api import TTS as MeloTTS  # type: ignore
        self._melo = MeloTTS(language="KR", device=self.device)
        self._melo_speaker = self._melo.hps.data.spk2id["KR"]

    # ------------------------------------------------------------------
    def synthesize(self, text: str, lang: str = "ko") -> Tuple[np.ndarray, float]:
        """Synthesize text.

        Returns:
            (audio_f32_16khz, synth_ms)  — numpy float32, 16 kHz mono.
        """
        if self._engine == "melo":
            return self._synth_melo(text)
        else:
            return self._synth_espeak(text, lang)

    # ------------------------------------------------------------------
    # MeloTTS backend
    # ------------------------------------------------------------------
    def _synth_melo(self, text: str) -> Tuple[np.ndarray, float]:
        import soundfile as sf  # type: ignore

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmppath = tmp.name

        t0 = time.perf_counter()
        self._melo.tts_to_file(
            text,
            self._melo_speaker,
            output_path=tmppath,
            speed=1.0,
            quiet=True,
        )
        synth_ms = (time.perf_counter() - t0) * 1000.0

        data, sr = sf.read(tmppath)
        Path(tmppath).unlink(missing_ok=True)

        # Convert to mono float32 if needed
        if data.ndim > 1:
            data = data.mean(axis=1)
        data = data.astype(np.float32)

        # Resample to 16 kHz (MeloTTS outputs 44100 Hz)
        if sr != self.SAMPLE_RATE:
            data = _resample(data, sr, self.SAMPLE_RATE)

        return data, synth_ms

    # ------------------------------------------------------------------
    # espeak-ng fallback backend (known-bad for Korean)
    # ------------------------------------------------------------------
    def _synth_espeak(self, text: str, lang: str = "ko") -> Tuple[np.ndarray, float]:
        lang_map = {"ko": "ko", "en": "en-us"}
        espeak_lang = lang_map.get(lang, lang)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmppath = tmp.name

        t0 = time.perf_counter()
        subprocess.run(
            [
                "espeak-ng",
                "-v", espeak_lang,
                "-s", "150",
                "-w", tmppath,
                text,
            ],
            check=True,
            capture_output=True,
        )
        synth_ms = (time.perf_counter() - t0) * 1000.0

        import soundfile as sf
        data, sr = sf.read(tmppath)
        Path(tmppath).unlink(missing_ok=True)

        if data.ndim > 1:
            data = data.mean(axis=1)
        data = data.astype(np.float32)

        if sr != self.SAMPLE_RATE:
            data = _resample(data, sr, self.SAMPLE_RATE)

        return data, synth_ms


# ---------------------------------------------------------------------------
# Resampling helper
# ---------------------------------------------------------------------------

def _resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Simple polyphase resampler using scipy."""
    try:
        from scipy.signal import resample_poly  # type: ignore
        import math
        g = math.gcd(orig_sr, target_sr)
        up = target_sr // g
        down = orig_sr // g
        return resample_poly(audio, up, down).astype(np.float32)
    except ImportError:
        pass

    try:
        import resampy  # type: ignore
        return resampy.resample(audio, orig_sr, target_sr).astype(np.float32)
    except ImportError:
        pass

    # Numpy fallback (lower quality)
    duration = len(audio) / orig_sr
    n_target = int(duration * target_sr)
    indices = np.linspace(0, len(audio) - 1, n_target)
    return np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)
