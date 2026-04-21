"""TranslationSession — all state for one meeting / call.

One Session per WebSocket connection.
Owns: ASR, translator, RAG, speaker profile, rolling context.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from loguru import logger

from ..asr.whisper_streaming import WhisperStreaming
from ..context.glossary import Glossary
from ..context.prompt_builder import PromptBuilder, SessionContext
from ..context.rag import HybridRetriever
from ..context.translation_memory import TranslationMemory
from ..personalization.lora_loader import LoRALoader
from ..personalization.speaker_profile import SpeakerProfile, SpeakerProfileStore
from ..translator.qwen_translator import QwenTranslator


@dataclass
class SessionConfig:
    session_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    src_lang: str = "en"
    tgt_lang: str = "ko"
    speaker_id: str = "default"
    meeting_id: str = "meeting"
    topic_summary: str = ""
    register: str = "formal"
    glossary_path: str | None = None


class TranslationSession:
    """Holds all components for a single live translation session."""

    def __init__(self, cfg: dict, session_cfg: SessionConfig) -> None:
        self.cfg = cfg
        self.session_cfg = session_cfg
        self.session_id = session_cfg.session_id
        self.created_at = time.time()

        logger.info(
            f"[Session {self.session_id}] "
            f"{session_cfg.src_lang} → {session_cfg.tgt_lang} | "
            f"speaker={session_cfg.speaker_id}"
        )

        # ------ ASR
        self.asr = WhisperStreaming(cfg)

        # ------ Translator
        self.translator = QwenTranslator(cfg)

        # ------ Context
        self.glossary = self._load_glossary(session_cfg)
        self.tm = TranslationMemory(cfg)
        self.retriever = HybridRetriever(cfg)
        self.prompt_builder = PromptBuilder(cfg)

        # ------ Personalization
        profile_dir = Path(cfg.get("personalization", {}).get("profile_dir",
                           "data/speaker_profiles"))
        self.profile_store = SpeakerProfileStore(profile_dir)
        self.profile: SpeakerProfile = self.profile_store.get_or_create(
            session_cfg.speaker_id,
            default_src_lang=session_cfg.src_lang,
            default_tgt_lang=session_cfg.tgt_lang,
            register=session_cfg.register,
            topic_summary=session_cfg.topic_summary,
        )

        lora_cfg = cfg.get("personalization", {}).get("lora", {})
        self.lora_loader = LoRALoader(
            adapter_dir=lora_cfg.get("adapter_dir", "data/lora_adapters"),
            rank=lora_cfg.get("rank", 8),
            alpha=lora_cfg.get("alpha", 16),
        )
        if lora_cfg.get("enabled", False):
            self.lora_loader.attach(self.asr._model)
            self.lora_loader.load_for_speaker(session_cfg.speaker_id)

        # ------ Rolling context
        self.ctx = SessionContext(
            src_lang=session_cfg.src_lang,
            tgt_lang=session_cfg.tgt_lang,
            topic_summary=session_cfg.topic_summary or self.profile.topic_summary,
            speaker_name=self.profile.display_name or session_cfg.speaker_id,
            register=self.profile.register,
            max_prev_segments=cfg["context"]["rolling_context"]["prev_segments"],
        )

        # ------ Index retriever over glossary + TM
        self._index_retriever()

        # ------ Metrics
        self.metrics: list[dict] = []

    # ------------------------------------------------------------ setup

    def _load_glossary(self, session_cfg: SessionConfig) -> Glossary:
        if session_cfg.glossary_path:
            path = Path(session_cfg.glossary_path)
            if path.exists():
                return Glossary.from_json(path)
        # Try default path by meeting ID
        default = Path("data/glossaries") / f"{session_cfg.meeting_id}.json"
        if default.exists():
            return Glossary.from_json(default)
        return Glossary.empty(session_cfg.src_lang, session_cfg.tgt_lang)

    def _index_retriever(self) -> None:
        entries = []
        for e in self.glossary.entries:
            entries.append({"src": e.src, "tgt": e.tgt, "source": "glossary"})
        # TM entries would be added here too
        if entries:
            self.retriever.index(entries)

    # ---------------------------------------------------------- context update

    def push_segment(self, src: str, tgt: str) -> None:
        """Record a completed segment into the rolling context."""
        self.ctx.prev_src_segments.append(src)
        self.ctx.prev_tgt_segments.append(tgt)
        # Trim to max
        n = self.ctx.max_prev_segments
        if len(self.ctx.prev_src_segments) > n:
            self.ctx.prev_src_segments = self.ctx.prev_src_segments[-n:]
            self.ctx.prev_tgt_segments = self.ctx.prev_tgt_segments[-n:]

    def build_system_prompt(self, src_partial: str) -> str:
        hits = self.glossary.hits_for(src_partial)
        tm_hits = self.tm.retrieve(src_partial)
        return self.prompt_builder.build_system_prompt(self.ctx, hits, tm_hits)

    # ----------------------------------------------------------- cleanup

    def close(self) -> None:
        self.asr.reset()
        self.translator.reset()
        self.profile_store.save(self.profile)
        logger.info(f"[Session {self.session_id}] Closed. Segments: {len(self.metrics)}")
