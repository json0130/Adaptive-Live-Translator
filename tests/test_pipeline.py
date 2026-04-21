"""End-to-end pipeline integration tests (stub models, no GPU required)."""
import asyncio

import numpy as np
import pytest

from src.pipeline.session import SessionConfig, TranslationSession
from src.pipeline.streaming_loop import TranslationEvent, run_streaming_loop

MOCK_CFG = {
    "asr": {
        "model": "openai/whisper-large-v3",
        "chunk_seconds": 0.5,
        "device": "cpu",
        "compute_type": "float32",
        "vad_enabled": False,
        "policy": "align_att",
        "align_att": {"start_seconds": 0.5, "num_frames": 5, "threshold_layers": "all"},
    },
    "translator": {
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "device": "cpu",
        "dtype": "float32",
        "max_new_tokens": 32,
        "max_context_tokens": 512,
        "kv_cache_reuse": True,
        "policy": "local_agreement",
        "temperature": 0.0,
    },
    "context": {
        "rag": {
            "enabled": True,
            "hybrid": False,
            "dense_model": "BAAI/bge-m3",
            "top_k": 3,
            "min_similarity": 0.1,
        },
        "glossary": {
            "enabled": True,
            "injection_mode": "prompt",
            "max_entries_in_prompt": 10,
        },
        "translation_memory": {"top_k": 3, "min_similarity": 0.1},
        "rolling_context": {"prev_segments": 3},
    },
    "personalization": {
        "lora": {
            "enabled": False,
            "adapter_dir": "data/lora_adapters",
            "rank": 8,
            "alpha": 16,
        },
        "profile_dir": "/tmp/test_profiles",
    },
    "tts": {"enabled": False, "model": "", "streaming": False, "voice_cloning": False},
    "pipeline": {"latency_regime": "low", "target_streamlaal_s": 2.0, "log_metrics": False},
    "api": {"host": "0.0.0.0", "port": 8000},
}


# ---------------------------------------------------------------- Session

def test_session_creation():
    cfg = SessionConfig(src_lang="en", tgt_lang="ko", speaker_id="test_user")
    session = TranslationSession(MOCK_CFG, cfg)
    assert session.session_cfg.src_lang == "en"
    assert session.session_cfg.tgt_lang == "ko"
    assert session.ctx.src_lang == "en"
    session.close()


def test_session_push_segment():
    cfg = SessionConfig(src_lang="en", tgt_lang="ko")
    session = TranslationSession(MOCK_CFG, cfg)
    session.push_segment("Hello.", "안녕.")
    assert "Hello." in session.ctx.prev_src_segments
    assert "안녕." in session.ctx.prev_tgt_segments
    session.close()


def test_session_build_system_prompt():
    cfg = SessionConfig(src_lang="en", tgt_lang="ko", topic_summary="AI research")
    session = TranslationSession(MOCK_CFG, cfg)
    prompt = session.build_system_prompt("We trained a large language model.")
    assert isinstance(prompt, str)
    assert len(prompt) > 0
    session.close()


def test_session_rolling_context_trimmed():
    cfg = SessionConfig(src_lang="en", tgt_lang="ko")
    session = TranslationSession(MOCK_CFG, cfg)
    for i in range(10):
        session.push_segment(f"Sentence {i}.", f"문장 {i}.")
    # Should be capped at max_prev_segments (3 in test cfg)
    assert len(session.ctx.prev_src_segments) <= 3
    session.close()


# ---------------------------------------------------------------- Streaming loop

@pytest.mark.asyncio
async def test_streaming_loop_runs():
    """Smoke test: pipeline runs without errors on synthetic audio."""
    cfg = SessionConfig(src_lang="en", tgt_lang="ko")
    session = TranslationSession(MOCK_CFG, cfg)

    audio_queue: asyncio.Queue = asyncio.Queue()
    output_queue: asyncio.Queue = asyncio.Queue()

    # Feed 3 chunks of silence then sentinel
    chunk_size = int(0.5 * 16000)
    for _ in range(3):
        await audio_queue.put(np.zeros(chunk_size, dtype=np.float32))
    await audio_queue.put(None)

    await asyncio.wait_for(
        run_streaming_loop(session, audio_queue, output_queue, tts_enabled=False),
        timeout=10.0,
    )

    events = []
    while not output_queue.empty():
        item = output_queue.get_nowait()
        if item is not None:
            events.append(item)

    # Should have produced events (or at minimum, not crashed)
    for e in events:
        assert isinstance(e, TranslationEvent)
        assert isinstance(e.latency_ms, float)
        assert isinstance(e.src_delta, str)

    session.close()
