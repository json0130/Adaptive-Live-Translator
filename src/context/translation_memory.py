"""Translation Memory (TM) — retrieve similar past translations as few-shot examples.

TM format (JSONL, one entry per line):
  {"src": "The model achieved state-of-the-art results.", "tgt": "该模型取得了最先进的结果。"}
  ...

At indexing time (scripts/build_tm_index.py) this file is turned into
a FAISS index + BM25 index. At inference time we query both and merge.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from loguru import logger


@dataclass
class TMEntry:
    src: str
    tgt: str
    score: float = 0.0


class TranslationMemory:
    """Lightweight TM that falls back to in-memory exact+fuzzy search when
    pre-built indexes are not available.
    """

    def __init__(self, cfg: dict) -> None:
        self.top_k: int = cfg["context"]["translation_memory"]["top_k"]
        self.min_similarity: float = cfg["context"]["translation_memory"]["min_similarity"]
        self._entries: list[tuple[str, str]] = []
        self._bm25 = None
        self._faiss_index = None
        self._embed_model = None

    # -------------------------------------------------------------- loading

    def load_jsonl(self, path: str | Path) -> None:
        """Load raw JSONL translation memory (no index required)."""
        path = Path(path)
        if not path.exists():
            logger.warning(f"TM file not found: {path}")
            return
        self._entries = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            self._entries.append((obj["src"], obj["tgt"]))
        logger.info(f"Loaded {len(self._entries)} TM entries from {path}")

    def load_index(self, index_dir: str | Path) -> None:
        """Load pre-built BM25 + FAISS indexes."""
        index_dir = Path(index_dir)
        bm25_path = index_dir / "bm25.pkl"
        faiss_path = index_dir / "faiss.index"
        entries_path = index_dir / "entries.jsonl"

        if not bm25_path.exists():
            logger.warning(f"BM25 index not found at {bm25_path}. Run build_tm_index.py first.")
            return

        import pickle
        with open(bm25_path, "rb") as f:
            self._bm25 = pickle.load(f)

        if faiss_path.exists():
            import faiss
            self._faiss_index = faiss.read_index(str(faiss_path))

        if entries_path.exists():
            self.load_jsonl(entries_path)

        logger.info(f"Loaded TM index from {index_dir}")

    # --------------------------------------------------------------- query

    def retrieve(self, query: str) -> list[TMEntry]:
        """Return top-k TM entries most similar to query."""
        if not self._entries:
            return []

        if self._bm25 is not None:
            return self._bm25_retrieve(query)

        # Fallback: simple substring matching
        return self._fallback_retrieve(query)

    def _bm25_retrieve(self, query: str) -> list[TMEntry]:
        """BM25 retrieval using pre-built index."""
        from rank_bm25 import BM25Okapi

        tokens = query.lower().split()
        scores = self._bm25.get_scores(tokens)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[: self.top_k]
        results = []
        for idx in top_indices:
            if scores[idx] < 0.01:
                continue
            src, tgt = self._entries[idx]
            # Normalise score to [0,1] roughly
            normalised = min(scores[idx] / (max(scores) + 1e-9), 1.0)
            if normalised >= self.min_similarity:
                results.append(TMEntry(src=src, tgt=tgt, score=normalised))
        return results

    def _fallback_retrieve(self, query: str) -> list[TMEntry]:
        query_words = set(query.lower().split())
        scored = []
        for src, tgt in self._entries:
            src_words = set(src.lower().split())
            if not src_words:
                continue
            overlap = len(query_words & src_words) / len(src_words | query_words)
            if overlap >= self.min_similarity:
                scored.append(TMEntry(src=src, tgt=tgt, score=overlap))
        return sorted(scored, key=lambda e: e.score, reverse=True)[: self.top_k]

    # ------------------------------------------------------- prompt helpers

    def to_prompt_block(self, entries: list[TMEntry]) -> str:
        if not entries:
            return ""
        lines = ["Translation examples (for style/terminology reference):"]
        for e in entries:
            lines.append(f"  SRC: {e.src}")
            lines.append(f"  TGT: {e.tgt}")
        return "\n".join(lines)
