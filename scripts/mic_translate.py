#!/usr/bin/env python3
"""
Korean → English live translator using a USB microphone.

Usage:
  # List audio input devices to find your USB mic index:
  python scripts/mic_translate.py --list-devices

  # Run with default mic:
  python scripts/mic_translate.py

  # Run with a specific USB mic (use index from --list-devices):
  python scripts/mic_translate.py --device 2

  # Use a larger Whisper model for better accuracy (slower):
  python scripts/mic_translate.py --device 2 --model medium

Requirements (auto-installed if missing on first run):
  .translator/bin/pip install sounddevice pyttsx3
  sudo apt install espeak-ng   # for spoken TTS output on Linux
"""
from __future__ import annotations

import argparse
import queue
import sys
import threading

import numpy as np

SAMPLE_RATE = 16_000
CHUNK_SECONDS = 2.0
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_SECONDS)


def list_devices() -> None:
    import sounddevice as sd
    print("\nAvailable audio input devices:")
    print("-" * 50)
    for i, dev in enumerate(sd.query_devices()):
        if dev["max_input_channels"] > 0:
            marker = " <-- (default)" if i == sd.default.device[0] else ""
            print(f"  [{i:2d}] {dev['name']}{marker}")
    print()


def load_asr(model_name: str):
    from faster_whisper import WhisperModel
    print(f"Loading Whisper '{model_name}' for Korean ASR (first run downloads the model)...")
    # int8 works on both CPU and GPU
    model = WhisperModel(model_name, device="cpu", compute_type="int8")
    print("ASR ready.")
    return model


def load_translator():
    from transformers import MarianMTModel, MarianTokenizer
    name = "Helsinki-NLP/opus-mt-ko-en"
    print(f"Loading translation model '{name}' (first run downloads ~300 MB)...")
    tokenizer = MarianTokenizer.from_pretrained(name)
    model = MarianMTModel.from_pretrained(name)
    print("Translation ready.")
    return tokenizer, model


def load_tts():
    try:
        import pyttsx3
        engine = pyttsx3.init()
        engine.setProperty("rate", 160)
        # quick test to confirm it works
        engine.say("")
        engine.runAndWait()
        print("TTS ready (pyttsx3).")
        return engine
    except Exception as e:
        print(f"TTS unavailable ({e}).")
        print("  To enable spoken output: sudo apt install espeak-ng")
        return None


def translate_text(korean: str, tokenizer, model) -> str:
    inputs = tokenizer([korean], return_tensors="pt", padding=True, truncation=True, max_length=512)
    outputs = model.generate(**inputs, num_beams=4, max_length=512)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def speak(engine, text: str) -> None:
    if engine is None:
        return
    try:
        engine.say(text)
        engine.runAndWait()
    except Exception:
        pass


def run(device: int | None, model_name: str) -> None:
    import sounddevice as sd

    asr_model = load_asr(model_name)
    tokenizer, translator = load_translator()
    tts_engine = load_tts()

    audio_q: queue.Queue[np.ndarray] = queue.Queue()
    buffer = np.array([], dtype=np.float32)

    def audio_callback(indata: np.ndarray, frames: int, time, status) -> None:
        if status:
            print(f"[audio] {status}", file=sys.stderr)
        audio_q.put(indata[:, 0].copy())  # mono

    dev_label = f"device {device}" if device is not None else "default device"
    print(f"\nListening on {dev_label} at {SAMPLE_RATE} Hz. Speak Korean... (Ctrl+C to stop)\n")

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        device=device,
        callback=audio_callback,
        blocksize=1600,
    ):
        while True:
            chunk = audio_q.get()
            buffer = np.concatenate([buffer, chunk])

            if len(buffer) < CHUNK_SAMPLES:
                continue

            audio_chunk = buffer[:CHUNK_SAMPLES].copy()
            buffer = buffer[CHUNK_SAMPLES:]

            segments, info = asr_model.transcribe(
                audio_chunk,
                language="ko",
                beam_size=5,
                vad_filter=True,
                vad_parameters={"min_silence_duration_ms": 300},
            )
            korean = " ".join(s.text for s in segments).strip()

            if not korean:
                continue

            print(f"KO: {korean}")
            english = translate_text(korean, tokenizer, translator)
            print(f"EN: {english}\n")

            # Speak in a background thread so mic capture continues uninterrupted
            threading.Thread(target=speak, args=(tts_engine, english), daemon=True).start()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Live Korean → English translator via USB microphone"
    )
    parser.add_argument("--list-devices", action="store_true", help="List audio input devices and exit")
    parser.add_argument("--device", type=int, default=None, metavar="N", help="Input device index (from --list-devices)")
    parser.add_argument(
        "--model",
        default="small",
        choices=["tiny", "small", "medium", "large-v2", "large-v3"],
        help="Whisper model size (default: small)",
    )
    args = parser.parse_args()

    if args.list_devices:
        list_devices()
        return

    try:
        run(args.device, args.model)
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
