"""Hybrid retriever — BM25 (lexical) + bge-m3 (semantic) over glossary + TM.

Pipeline per chunk:
  1.  BM25 over glossary entries → top-k candidates
  2.  Dense embed query with bge-m3 → cosine similarity
  3.  Reciprocal Rank Fusion (RRF) to merge both ranked lists
  4.  Return top-k after dedup

Usage:
    retriever = HybridRetriever(cfg)
    retriever.index(entries)          # at session start
    results = retriever.retrieve(text)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from loguru import logger


@dataclass
class RetrievalResult:
    text: str           # source text of retrieved entry
    translation: str    # target translation
    score: float        # merged RRF score
    source: str         # "glossary" | "tm"


class HybridRetriever:
    def __init__(self, cfg: dict) -> None:
        rag_cfg = cfg["context"]["rag"]
        self.top_k: int = rag_cfg["top_k"]
        self.min_similarity: float = rag_cfg["min_similarity"]
        self.hybrid: bool = rag_cfg["hybrid"]
        self.dense_model_name: str = rag_cfg["dense_model"]
        self._corpus: list[dict] = []       # {"src", "tgt", "source"}
        self._bm25 = None
        self._embed_model = None
        self._embeddings: np.ndarray | None = None

    # ----------------------------------------------------------- indexing

    def index(self, entries: list[dict]) -> None:
        """Build BM25 + dense index over provided entries.

        entries: [{"src": str, "tgt": str, "source": "glossary"|"tm"}]
        """
        self._corpus = entries
        if not entries:
            return

        tokenised = [e["src"].lower().split() for e in entries]

        try:
            from rank_bm25 import BM25Okapi
            self._bm25 = BM25Okapi(tokenised)
        except ImportError:
            logger.warning("rank-bm25 not installed; BM25 retrieval disabled.")

        if self.hybrid:
            self._build_dense_index(entries)

    def _build_dense_index(self, entries: list[dict]) -> None:
        try:
            from FlagEmbedding import BGEM3FlagModel
            logger.info(f"Loading dense embedding model: {self.dense_model_name}")
            self._embed_model = BGEM3FlagModel(self.dense_model_name, use_fp16=True)
            texts = [e["src"] for e in entries]
            result = self._embed_model.encode(texts, return_dense=True, batch_size=256)
            self._embeddings = result["dense_vecs"]   # shape [N, D]
            logger.info(f"Built dense index over {len(texts)} entries.")
        except ImportError:
            logger.warning("FlagEmbedding not installed; dense retrieval disabled.")
        except Exception as exc:
            logger.warning(f"Dense index build failed: {exc}")

    # ---------------------------------------------------------- retrieval

    def retrieve(self, query: str) -> list[RetrievalResult]:
        if not self._corpus:
            return []

        bm25_ranked = self._bm25_rank(query)
        dense_ranked = self._dense_rank(query) if self._embeddings is not None else []

        if not dense_ranked:
            candidates = bm25_ranked
        else:
            candidates = _reciprocal_rank_fusion(bm25_ranked, dense_ranked, k=60)

        results = []
        seen: set[str] = set()
        for idx, score in candidates[: self.top_k]:
            entry = self._corpus[idx]
            key = entry["src"]
            if key in seen or score < self.min_similarity:
                continue
            seen.add(key)
            results.append(
                RetrievalResult(
                    text=entry["src"],
                    translation=entry["tgt"],
                    score=score,
                    source=entry.get("source", "unknown"),
                )
            )
        return results

    def _bm25_rank(self, query: str) -> list[tuple[int, float]]:
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(query.lower().split())
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        max_score = ranked[0][1] if ranked else 1.0
        return [(i, s / (max_score + 1e-9)) for i, s in ranked]

    def _dense_rank(self, query: str) -> list[tuple[int, float]]:
        if self._embed_model is None or self._embeddings is None:
            return []
        result = self._embed_model.encode([query], return_dense=True)
        qvec = result["dense_vecs"][0]
        sims = (self._embeddings @ qvec) / (
            np.linalg.norm(self._embeddings, axis=1) * np.linalg.norm(qvec) + 1e-9
        )
        ranked = sorted(enumerate(sims.tolist()), key=lambda x: x[1], reverse=True)
        return ranked


# ---------------------------------------------------------------- helpers

def _reciprocal_rank_fusion(
    list_a: list[tuple[int, float]],
    list_b: list[tuple[int, float]],
    k: int = 60,
) -> list[tuple[int, float]]:
    """Merge two ranked lists using RRF: score(d) = Σ 1/(k + rank(d))."""
    scores: dict[int, float] = {}
    for rank, (idx, _) in enumerate(list_a):
        scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
    for rank, (idx, _) in enumerate(list_b):
        scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
    max_score = max(scores.values()) if scores else 1.0
    return sorted(
        [(idx, s / max_score) for idx, s in scores.items()],
        key=lambda x: x[1],
        reverse=True,
    )
