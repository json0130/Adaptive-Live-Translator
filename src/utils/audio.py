"""Audio utilities — PCM chunk helpers."""
from __future__ import annotations

import io
from typing import AsyncIterator

import numpy as np


def bytes_to_pcm(raw: bytes, dtype=np.float32) -> np.ndarray:
    """Convert raw bytes to float32 PCM numpy array."""
    return np.frombuffer(raw, dtype=dtype)


def pcm_to_bytes(pcm: np.ndarray) -> bytes:
    return pcm.astype(np.float32).tobytes()


def resample(audio: np.ndarray, src_sr: int, tgt_sr: int) -> np.ndarray:
    """Simple linear resampling (use librosa.resample for quality)."""
    if src_sr == tgt_sr:
        return audio
    try:
        import librosa
        return librosa.resample(audio, orig_sr=src_sr, target_sr=tgt_sr)
    except ImportError:
        ratio = tgt_sr / src_sr
        new_len = int(len(audio) * ratio)
        return np.interp(
            np.linspace(0, len(audio) - 1, new_len),
            np.arange(len(audio)),
            audio,
        ).astype(np.float32)


async def file_to_chunk_iter(
    path: str,
    chunk_seconds: float = 2.0,
    sr: int = 16000,
) -> AsyncIterator[np.ndarray]:
    """Read a WAV/MP3 file and yield PCM chunks — useful for testing."""
    import librosa
    audio, orig_sr = librosa.load(path, sr=sr, mono=True)
    chunk_size = int(chunk_seconds * sr)
    for i in range(0, len(audio), chunk_size):
        yield audio[i : i + chunk_size]
