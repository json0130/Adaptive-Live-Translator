"""Tests for streaming translator + policies."""
import asyncio

import pytest

from src.translator.base import TranslationChunk
from src.translator.policies import LocalAgreementPolicy, _common_prefix
from src.translator.qwen_translator import QwenTranslator

MOCK_CFG = {
    "translator": {
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "device": "cpu",
        "dtype": "float32",
        "max_new_tokens": 64,
        "max_context_tokens": 512,
        "kv_cache_reuse": True,
        "policy": "local_agreement",
        "temperature": 0.0,
    }
}


# ---------------------------------------------------------------- policies

def test_common_prefix_basic():
    assert _common_prefix("hello world", "hello there") == "hello "
    assert _common_prefix("abc", "xyz") == ""
    assert _common_prefix("same", "same") == "same"


def test_local_agreement_not_enough_history():
    policy = LocalAgreementPolicy(cfg=type("C", (), {"history_size": 2})())
    result = policy.update("hello world")
    assert result == ""   # first update — not enough history yet


def test_local_agreement_commits_common_prefix():
    policy = LocalAgreementPolicy(cfg=type("C", (), {"history_size": 2})())
    policy.update("hello world today")
    committed = policy.update("hello world tomorrow")
    assert "hello world" in committed


def test_local_agreement_reset():
    policy = LocalAgreementPolicy(cfg=type("C", (), {"history_size": 2})())
    policy.update("a b c")
    policy.update("a b d")
    policy.reset()
    assert policy._committed == ""
    assert len(policy._history) == 0


# ---------------------------------------------------------------- translator

def test_qwen_instantiation():
    t = QwenTranslator(MOCK_CFG)
    assert t.max_new_tokens == 64
    assert t.max_context_tokens == 512


def test_qwen_reset():
    t = QwenTranslator(MOCK_CFG)
    t._past_kv = "fake_kv"
    t.reset()
    assert t._past_kv is None


@pytest.mark.asyncio
async def test_qwen_translate_stream_yields_chunks():
    t = QwenTranslator(MOCK_CFG)

    async def _src_iter():
        yield "The quick brown fox."

    chunks = []
    async for chunk in t.translate_stream(
        _src_iter(),
        src_lang="en",
        tgt_lang="ko",
        system_prompt="Translate EN→KO.",
    ):
        chunks.append(chunk)

    assert len(chunks) >= 1
    for c in chunks:
        assert isinstance(c, TranslationChunk)
        assert isinstance(c.text, str)
