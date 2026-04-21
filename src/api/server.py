"""FastAPI application — REST + WebSocket entrypoints.

Endpoints:
  GET  /health
  POST /sessions           — create a session, returns session_id
  GET  /sessions/{id}      — session status + metrics
  DELETE /sessions/{id}    — close session
  GET  /speakers           — list speaker profiles
  POST /speakers/{id}/glossary — add glossary entry to speaker profile
  WS   /ws/{session_id}    — bidirectional audio/text streaming
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager

import yaml
from fastapi import FastAPI, HTTPException, WebSocket
from loguru import logger
from pydantic import BaseModel

from ..pipeline.session import SessionConfig, TranslationSession
from ..personalization.speaker_profile import SpeakerProfileStore
from .ws_handler import handle_websocket

# ------------------------------------------------------------------ state

_sessions: dict[str, TranslationSession] = {}
_cfg: dict = {}
_profile_store: SpeakerProfileStore | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _cfg, _profile_store
    config_path = os.getenv("CONFIG_PATH", "configs/default.yaml")
    with open(config_path) as f:
        _cfg = yaml.safe_load(f)
    _profile_store = SpeakerProfileStore("data/speaker_profiles")
    logger.info("Adaptive Live Translator API ready.")
    yield
    for s in list(_sessions.values()):
        s.close()
    logger.info("Shutdown complete.")


app = FastAPI(
    title="Adaptive Live Translator",
    version="0.1.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------- schemas

class SessionCreateRequest(BaseModel):
    src_lang: str = "en"
    tgt_lang: str = "ko"
    speaker_id: str = "default"
    meeting_id: str = "meeting"
    topic_summary: str = ""
    register: str = "formal"
    glossary_path: str | None = None


class GlossaryAddRequest(BaseModel):
    src: str
    tgt: str
    dnt: bool = False


# ---------------------------------------------------------------- routes

@app.get("/health")
async def health():
    return {"status": "ok", "active_sessions": len(_sessions)}


@app.post("/sessions", status_code=201)
async def create_session(req: SessionCreateRequest):
    cfg = SessionCreateRequest(
        src_lang=req.src_lang,
        tgt_lang=req.tgt_lang,
        speaker_id=req.speaker_id,
        meeting_id=req.meeting_id,
        topic_summary=req.topic_summary,
        register=req.register,
        glossary_path=req.glossary_path,
    )
    session_cfg = SessionConfig(**req.model_dump())
    session = TranslationSession(_cfg, session_cfg)
    _sessions[session.session_id] = session
    return {"session_id": session.session_id}


@app.get("/sessions/{session_id}")
async def get_session(session_id: str):
    session = _sessions.get(session_id)
    if not session:
        raise HTTPException(404, "Session not found")
    return {
        "session_id": session.session_id,
        "src_lang": session.session_cfg.src_lang,
        "tgt_lang": session.session_cfg.tgt_lang,
        "speaker_id": session.session_cfg.speaker_id,
        "segments_translated": len(session.metrics),
        "avg_latency_ms": (
            sum(m["latency_ms"] for m in session.metrics) / len(session.metrics)
            if session.metrics else 0.0
        ),
    }


@app.delete("/sessions/{session_id}", status_code=204)
async def close_session(session_id: str):
    session = _sessions.pop(session_id, None)
    if not session:
        raise HTTPException(404, "Session not found")
    session.close()


@app.get("/speakers")
async def list_speakers():
    if _profile_store is None:
        return {"speakers": []}
    return {"speakers": _profile_store.list_speakers()}


@app.post("/speakers/{speaker_id}/glossary", status_code=201)
async def add_glossary_entry(speaker_id: str, req: GlossaryAddRequest):
    if _profile_store is None:
        raise HTTPException(500, "Profile store not initialised")
    profile = _profile_store.get_or_create(speaker_id)
    profile.glossary_entries.append({"src": req.src, "tgt": req.tgt, "dnt": req.dnt})
    _profile_store.save(profile)
    return {"ok": True, "total_entries": len(profile.glossary_entries)}


@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    session = _sessions.get(session_id)
    if not session:
        await websocket.close(code=4004, reason="Session not found")
        return
    await handle_websocket(websocket, session, _cfg)
