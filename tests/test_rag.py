"""Tests for context layer: glossary, TM, RAG, prompt builder."""
import json
import tempfile
from pathlib import Path

import pytest

from src.context.glossary import Glossary, GlossaryEntry
from src.context.prompt_builder import PromptBuilder, SessionContext
from src.context.rag import HybridRetriever, _reciprocal_rank_fusion
from src.context.translation_memory import TranslationMemory

MOCK_CFG = {
    "context": {
        "rag": {
            "enabled": True,
            "hybrid": False,            # BM25 only (no GPU needed in tests)
            "dense_model": "BAAI/bge-m3",
            "top_k": 3,
            "min_similarity": 0.1,
        },
        "glossary": {
            "enabled": True,
            "injection_mode": "prompt",
            "max_entries_in_prompt": 10,
        },
        "translation_memory": {
            "top_k": 3,
            "min_similarity": 0.1,
        },
        "rolling_context": {
            "prev_segments": 3,
        },
    }
}


# ---------------------------------------------------------------- Glossary

def test_glossary_hits_simple():
    g = Glossary.empty("en", "ko")
    g.add_entry("inference", "추론")
    g.add_entry("fine-tuning", "파인튜닝")
    hits = g.hits_for("We ran inference on the model.")
    assert len(hits) == 1
    assert hits[0].src == "inference"


def test_glossary_dnt():
    g = Glossary.empty("en", "ko")
    g.add_entry("NVIDIA", "NVIDIA", dnt=True)
    hits = g.hits_for("NVIDIA released a new GPU.")
    assert hits[0].dnt is True


def test_glossary_prompt_block():
    g = Glossary.empty()
    g.add_entry("LLM", "대형 언어 모델")
    g.add_entry("PyTorch", "PyTorch", dnt=True)
    hits = g.hits_for("We used an LLM built with PyTorch.")
    block = g.to_prompt_block(hits)
    assert "LLM" in block
    assert "PyTorch" in block
    assert "Do-Not-Translate" in block


def test_glossary_serialisation(tmp_path):
    g = Glossary(
        entries=[GlossaryEntry(src="hello", tgt="안녕")],
        src_lang="en",
        tgt_lang="ko",
    )
    path = tmp_path / "test.json"
    g.save(path)
    loaded = Glossary.from_json(path)
    assert loaded.entries[0].src == "hello"
    assert loaded.src_lang == "en"


# ---------------------------------------------------------------- Translation Memory

def test_tm_fallback_retrieve(tmp_path):
    tm_path = tmp_path / "tm.jsonl"
    entries = [
        {"src": "The model is very accurate.", "tgt": "모델은 매우 정확합니다."},
        {"src": "Training takes a long time.", "tgt": "훈련에는 오랜 시간이 걸립니다."},
    ]
    tm_path.write_text("\n".join(json.dumps(e) for e in entries), encoding="utf-8")

    tm = TranslationMemory(MOCK_CFG)
    tm.load_jsonl(tm_path)

    results = tm.retrieve("The model gave accurate results.")
    assert len(results) > 0
    assert results[0].src == "The model is very accurate."


def test_tm_prompt_block():
    from src.context.translation_memory import TMEntry
    entries = [TMEntry(src="Hello.", tgt="안녕.", score=0.9)]
    tm = TranslationMemory(MOCK_CFG)
    block = tm.to_prompt_block(entries)
    assert "Hello." in block
    assert "안녕." in block


# ---------------------------------------------------------------- RAG

def test_rrf_merge():
    list_a = [(0, 0.9), (1, 0.8), (2, 0.5)]
    list_b = [(1, 0.95), (0, 0.7), (3, 0.6)]
    merged = _reciprocal_rank_fusion(list_a, list_b)
    # Index 1 appears top in both — should rank first
    assert merged[0][0] == 1


def test_hybrid_retriever_bm25_only():
    retriever = HybridRetriever(MOCK_CFG)
    entries = [
        {"src": "attention mechanism in transformers", "tgt": "트랜스포머의 어텐션 메커니즘", "source": "glossary"},
        {"src": "gradient descent optimisation", "tgt": "경사 하강법 최적화", "source": "tm"},
        {"src": "language model pretraining", "tgt": "언어 모델 사전 학습", "source": "tm"},
    ]
    retriever.index(entries)
    results = retriever.retrieve("attention in transformers")
    assert len(results) >= 1
    assert results[0].text == "attention mechanism in transformers"


# ---------------------------------------------------------------- Prompt Builder

def test_prompt_builder_system_prompt():
    pb = PromptBuilder(MOCK_CFG)
    ctx = SessionContext(
        src_lang="en",
        tgt_lang="ko",
        topic_summary="Machine learning conference",
        speaker_name="Alice",
        register="formal",
    )
    hits = [GlossaryEntry(src="LLM", tgt="대형 언어 모델")]
    from src.context.translation_memory import TMEntry
    tm_hits = [TMEntry(src="Hello.", tgt="안녕.", score=0.9)]

    prompt = pb.build_system_prompt(ctx, hits, tm_hits)
    assert "EN → KO" in prompt or "en" in prompt.lower()
    assert "Machine learning conference" in prompt
    assert "LLM" in prompt
    assert "안녕" in prompt


def test_prompt_builder_rolling_context():
    pb = PromptBuilder(MOCK_CFG)
    ctx = SessionContext(src_lang="en", tgt_lang="ko")
    ctx.prev_src_segments = ["Good morning.", "How are you?"]
    ctx.prev_tgt_segments = ["좋은 아침입니다.", "잘 지내세요?"]

    prompt = pb.build_system_prompt(ctx, [], [])
    assert "Good morning." in prompt
    assert "좋은 아침입니다." in prompt


def test_prompt_user_turn():
    turn = PromptBuilder.build_user_turn(
        "Today we discuss neural networks.",
        tgt_so_far="오늘은 신경망에 대해",
    )
    assert "Today we discuss" in turn
    assert "오늘은 신경망에 대해" in turn
