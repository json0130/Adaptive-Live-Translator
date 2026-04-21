#!/usr/bin/env python
"""CLI WebSocket test client — stream a WAV file through the live translator.

Usage:
    # Start server first: make run
    python scripts/test_client.py \\
        --audio samples/en_tech_talk.wav \\
        --src en --tgt ko \\
        --session-id my-test
"""
from __future__ import annotations

import argparse
import asyncio
import json

import numpy as np
import websockets

SERVER = "ws://localhost:8000"


async def create_session(src: str, tgt: str, topic: str) -> str:
    import httpx
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"http://localhost:8000/sessions",
            json={"src_lang": src, "tgt_lang": tgt, "topic_summary": topic},
        )
        resp.raise_for_status()
        return resp.json()["session_id"]


async def stream_audio(
    session_id: str,
    audio_path: str,
    chunk_seconds: float = 2.0,
) -> None:
    import librosa

    audio, sr = librosa.load(audio_path, sr=16000, mono=True)
    chunk_size = int(chunk_seconds * 16000)

    uri = f"{SERVER}/ws/{session_id}"
    print(f"Connecting to {uri}")

    async with websockets.connect(uri) as ws:
        # Receive loop
        async def _receive():
            while True:
                try:
                    msg = await ws.recv()
                    data = json.loads(msg)
                    t = data.get("type", "")
                    if t == "final":
                        lat = data.get("latency_ms", 0)
                        print(f"  [{lat:.0f}ms] {data.get('src_raw','')!r}")
                        print(f"         → {data.get('tgt','')!r}")
                    elif t == "done":
                        print("Done.")
                        return
                    elif t == "error":
                        print(f"ERROR: {data['message']}")
                        return
                except Exception:
                    return

        recv_task = asyncio.create_task(_receive())

        # Send audio chunks
        for i in range(0, len(audio), chunk_size):
            chunk = audio[i : i + chunk_size].astype(np.float32)
            await ws.send(chunk.tobytes())
            await asyncio.sleep(chunk_seconds * 0.9)  # ~real-time

        # Signal end
        await ws.send(json.dumps({"type": "end"}))
        await recv_task


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True)
    parser.add_argument("--src", default="en")
    parser.add_argument("--tgt", default="ko")
    parser.add_argument("--topic", default="")
    parser.add_argument("--chunk-seconds", type=float, default=2.0)
    args = parser.parse_args()

    print(f"Creating session ({args.src} → {args.tgt})...")
    session_id = await create_session(args.src, args.tgt, args.topic)
    print(f"Session ID: {session_id}")

    await stream_audio(session_id, args.audio, args.chunk_seconds)


if __name__ == "__main__":
    asyncio.run(main())
