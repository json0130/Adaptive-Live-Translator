"""Tests for streaming ASR layer."""
import asyncio

import numpy as np
import pytest

from src.asr.base import ASRChunk, ASRStreamer
from src.asr.whisper_streaming import WhisperStreaming


MOCK_CFG = {
    "asr": {
        "model": "openai/whisper-large-v3",
        "chunk_seconds": 1.0,
        "device": "cpu",
        "compute_type": "float32",
        "vad_enabled": False,
        "policy": "align_att",
        "align_att": {"start_seconds": 1.0, "num_frames": 10, "threshold_layers": "all"},
    }
}


def make_audio_iter(n_chunks: int = 3, chunk_seconds: float = 1.0):
    """Yield n_chunks of silent float32 PCM at 16 kHz."""
    async def _iter():
        for _ in range(n_chunks):
            yield np.zeros(int(chunk_seconds * 16000), dtype=np.float32)
    return _iter()


def test_asr_chunk_fields():
    chunk = ASRChunk(
        text="hello world",
        delta="world",
        start_ms=0,
        end_ms=1000,
        is_final=True,
    )
    assert chunk.text == "hello world"
    assert chunk.delta == "world"
    assert chunk.is_final is True


def test_whisper_streaming_instantiation():
    streamer = WhisperStreaming(MOCK_CFG)
    assert streamer.chunk_seconds == 1.0
    assert streamer.device == "cpu"


def test_whisper_reset():
    streamer = WhisperStreaming(MOCK_CFG)
    streamer._committed_text = "some text"
    streamer.reset()
    assert streamer._committed_text == ""
    assert streamer._buffer == []


@pytest.mark.asyncio
async def test_whisper_stream_yields_chunks():
    streamer = WhisperStreaming(MOCK_CFG)
    audio_iter = make_audio_iter(n_chunks=3, chunk_seconds=1.0)
    chunks = []
    async for chunk in streamer.stream(audio_iter):
        chunks.append(chunk)
    # Should yield at least one chunk per full buffer
    assert len(chunks) >= 1
    for c in chunks:
        assert isinstance(c, ASRChunk)
        assert isinstance(c.text, str)
        assert isinstance(c.is_final, bool)
