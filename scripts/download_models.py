#!/usr/bin/env python
"""Download all required model weights from Hugging Face.

Usage:
    python scripts/download_models.py
    python scripts/download_models.py --skip-tts
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from loguru import logger


MODELS = {
    "whisper": "openai/whisper-large-v3",
    "translator": "Qwen/Qwen2.5-7B-Instruct",
    "embeddings": "BAAI/bge-m3",
    "tts": "FunAudioLLM/CosyVoice2-0.5B",
}


def download_hf_model(repo_id: str, cache_dir: str) -> None:
    from huggingface_hub import snapshot_download
    logger.info(f"Downloading {repo_id} → {cache_dir}")
    snapshot_download(
        repo_id=repo_id,
        cache_dir=cache_dir,
        token=os.getenv("HF_TOKEN"),
        ignore_patterns=["*.msgpack", "flax_model*", "tf_model*", "rust_model*"],
    )
    logger.info(f"✓ {repo_id}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", default=os.getenv("MODEL_CACHE_DIR", "./models"))
    parser.add_argument("--skip-tts", action="store_true")
    parser.add_argument("--skip-embeddings", action="store_true")
    args = parser.parse_args()

    Path(args.cache_dir).mkdir(parents=True, exist_ok=True)

    to_download = {k: v for k, v in MODELS.items()}
    if args.skip_tts:
        to_download.pop("tts")
    if args.skip_embeddings:
        to_download.pop("embeddings")

    for name, repo_id in to_download.items():
        try:
            download_hf_model(repo_id, args.cache_dir)
        except Exception as e:
            logger.error(f"Failed to download {name} ({repo_id}): {e}")

    logger.info("All downloads complete.")


if __name__ == "__main__":
    main()
