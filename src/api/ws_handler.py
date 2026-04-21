"""WebSocket handler — bridges browser/client ↔ streaming pipeline.

Protocol (binary + JSON mixed):
  Client → Server:
    - Binary frames: raw PCM float32 LE 16 kHz mono audio chunks
    - JSON text:  {"type": "config", "topic": "...", "glossary": [...]}
                  {"type": "end"}          ← signal end of audio

  Server → Client:
    - JSON text:  {"type": "partial",  "src": "...", "tgt": "...", "latency_ms": 87}
                  {"type": "final",    "src": "...", "tgt": "...", "latency_ms": 87}
                  {"type": "error",    "message": "..."}
                  {"type": "done"}
"""
from __future__ import annotations

import asyncio
import json

import numpy as np
from fastapi import WebSocket, WebSocketDisconnect
from loguru import logger

from ..pipeline.session import TranslationSession
from ..pipeline.streaming_loop import run_streaming_loop


async def handle_websocket(
    ws: WebSocket,
    session: TranslationSession,
    cfg: dict,
) -> None:
    await ws.accept()
    logger.info(f"[WS] Client connected: session={session.session_id}")

    audio_queue: asyncio.Queue = asyncio.Queue(maxsize=64)
    output_queue: asyncio.Queue = asyncio.Queue()
    tts_enabled: bool = cfg.get("tts", {}).get("enabled", False)

    # Start the pipeline loop as a background task
    loop_task = asyncio.create_task(
        run_streaming_loop(session, audio_queue, output_queue, tts_enabled=tts_enabled)
    )

    # Forward output events to client
    async def _send_events():
        while True:
            event = await output_queue.get()
            if event is None:
                await ws.send_json({"type": "done"})
                return
            payload = {
                "type": "final" if event.is_final else "partial",
                "src": event.tgt_delta,    # Note: send target to client
                "src_raw": event.src_delta,
                "tgt": event.tgt_delta,
                "src_cumulative": event.src_cumulative,
                "tgt_cumulative": event.tgt_cumulative,
                "latency_ms": round(event.latency_ms, 1),
            }
            await ws.send_json(payload)

    sender_task = asyncio.create_task(_send_events())

    try:
        while True:
            message = await ws.receive()

            if "bytes" in message and message["bytes"]:
                # Binary audio frame — convert bytes → float32 numpy array
                raw = message["bytes"]
                pcm = np.frombuffer(raw, dtype=np.float32)
                await audio_queue.put(pcm)

            elif "text" in message and message["text"]:
                data = json.loads(message["text"])
                msg_type = data.get("type", "")

                if msg_type == "end":
                    await audio_queue.put(None)  # sentinel
                    break

                elif msg_type == "config":
                    # Hot-update session context at runtime
                    if "topic" in data:
                        session.ctx.topic_summary = data["topic"]
                    if "register" in data:
                        session.ctx.register = data["register"]
                    if "glossary" in data:
                        for entry in data["glossary"]:
                            session.glossary.add_entry(
                                src=entry["src"],
                                tgt=entry["tgt"],
                                dnt=entry.get("dnt", False),
                            )
                        session._index_retriever()

    except WebSocketDisconnect:
        logger.info(f"[WS] Client disconnected: session={session.session_id}")
        await audio_queue.put(None)

    except Exception as exc:
        logger.error(f"[WS] Error in session {session.session_id}: {exc}")
        await ws.send_json({"type": "error", "message": str(exc)})
        await audio_queue.put(None)

    finally:
        await asyncio.gather(loop_task, sender_task, return_exceptions=True)
        logger.info(f"[WS] Session {session.session_id} handler closed.")
