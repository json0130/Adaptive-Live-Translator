#!/usr/bin/env python
"""Build BM25 + FAISS index over a JSONL translation memory file.

Usage:
    python scripts/build_tm_index.py \
        --tm data/translation_memory/en-ko.jsonl \
        --out data/translation_memory/en-ko.index

JSONL format (one entry per line):
    {"src": "The model achieved state-of-the-art results.", "tgt": "모델이 최첨단 결과를 달성했습니다."}
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

from loguru import logger


def load_jsonl(path: Path) -> list[tuple[str, str]]:
    entries = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        entries.append((obj["src"], obj["tgt"]))
    logger.info(f"Loaded {len(entries)} TM entries from {path}")
    return entries


def build_bm25(entries: list[tuple[str, str]], out_dir: Path) -> None:
    from rank_bm25 import BM25Okapi

    tokenised = [src.lower().split() for src, _ in entries]
    bm25 = BM25Okapi(tokenised)
    out_path = out_dir / "bm25.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(bm25, f)
    logger.info(f"BM25 index saved → {out_path}")


def build_faiss(entries: list[tuple[str, str]], out_dir: Path) -> None:
    try:
        import faiss
        import numpy as np
        from FlagEmbedding import BGEM3FlagModel
    except ImportError as e:
        logger.warning(f"Skipping FAISS index: {e}")
        return

    logger.info("Encoding entries with bge-m3 (this may take a few minutes)...")
    model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
    texts = [src for src, _ in entries]
    result = model.encode(texts, return_dense=True, batch_size=256, show_progress_bar=True)
    vecs = result["dense_vecs"].astype(np.float32)

    dim = vecs.shape[1]
    index = faiss.IndexFlatIP(dim)   # inner product = cosine after normalisation
    faiss.normalize_L2(vecs)
    index.add(vecs)

    out_path = out_dir / "faiss.index"
    faiss.write_index(index, str(out_path))
    logger.info(f"FAISS index ({index.ntotal} vectors, dim={dim}) saved → {out_path}")


def save_entries(entries: list[tuple[str, str]], out_dir: Path) -> None:
    out_path = out_dir / "entries.jsonl"
    lines = [json.dumps({"src": s, "tgt": t}, ensure_ascii=False) for s, t in entries]
    out_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"Entries saved → {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tm", required=True, help="Path to source JSONL TM file")
    parser.add_argument("--out", required=True, help="Output directory for index files")
    parser.add_argument("--skip-faiss", action="store_true", help="Skip dense FAISS index")
    args = parser.parse_args()

    tm_path = Path(args.tm)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not tm_path.exists():
        logger.error(f"TM file not found: {tm_path}")
        return

    entries = load_jsonl(tm_path)
    build_bm25(entries, out_dir)
    save_entries(entries, out_dir)

    if not args.skip_faiss:
        build_faiss(entries, out_dir)

    logger.info(f"Index build complete → {out_dir}")


if __name__ == "__main__":
    main()
