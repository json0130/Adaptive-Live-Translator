"""Shared pytest fixtures."""
import pytest


@pytest.fixture
def mock_cfg():
    return {
        "asr": {
            "model": "openai/whisper-large-v3",
            "chunk_seconds": 1.0,
            "device": "cpu",
            "compute_type": "float32",
            "vad_enabled": False,
            "policy": "align_att",
            "align_att": {"start_seconds": 1.0, "num_frames": 10, "threshold_layers": "all"},
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
            "lora": {"enabled": False, "adapter_dir": "data/lora_adapters", "rank": 8, "alpha": 16},
            "profile_dir": "/tmp/test_profiles",
        },
        "tts": {"enabled": False, "model": "", "streaming": False, "voice_cloning": False},
        "pipeline": {"latency_regime": "low", "target_streamlaal_s": 2.0, "log_metrics": False},
        "api": {"host": "0.0.0.0", "port": 8000},
    }
