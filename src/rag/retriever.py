"""
Hybrid retrieval: BM25 (lexical) + dense embeddings, fused with Reciprocal
Rank Fusion (RRF).

Why RRF instead of `alpha * bm25_score + (1 - alpha) * cosine_score`
(what the original script did): BM25 scores are unbounded and corpus-
dependent (a score of 8 can be "great" on one corpus and "mediocre" on
another), while cosine similarity is bounded to [-1, 1]. Adding them
together mixes two incompatible scales, so the `alpha` weight ends up
tuned to whatever your test corpus happened to produce rather than
anything principled. RRF sidesteps this entirely by fusing *ranks*, not
raw scores — this is the same technique used in Elasticsearch's and
Weaviate's hybrid search:

    RRF_score(d) = sum over retrievers of  1 / (k + rank_of_d_in_that_list)

k=60 is the constant from the original RRF paper (Cormack et al., 2009)
and is a reasonable default without tuning.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import List

import numpy as np

from .bm25 import BM25
from .chunking import Chunk
from .embeddings import Embedder, cosine_sim_matrix


@dataclass
class RetrievalResult:
    chunk: Chunk
    score: float
    bm25_rank: int | None = None
    dense_rank: int | None = None


class HybridRetriever:
    def __init__(self, chunks: List[Chunk], embedder: Embedder):
        if not chunks:
            raise ValueError("HybridRetriever requires at least one chunk")
        self.chunks = chunks
        self.texts = [c.text for c in chunks]
        self.bm25 = BM25(self.texts)
        self.embedder = embedder
        self.doc_vectors = embedder.encode(self.texts)

    def _bm25_ranked(self, query: str, top_k: int) -> List[tuple[int, float]]:
        return self.bm25.search(query, top_k=top_k)

    def _dense_ranked(self, query: str, top_k: int) -> List[tuple[int, float]]:
        q_vec = self.embedder.encode([query])[0]
        sims = cosine_sim_matrix(q_vec, self.doc_vectors)
        order = np.argsort(sims)[::-1][:top_k]
        return [(int(i), float(sims[i])) for i in order]

    def search(
        self, query: str, top_k: int = 5, candidate_pool: int = 30, rrf_k: int = 60
    ) -> List[RetrievalResult]:
        """
        Retrieve top_k chunks for `query` by fusing BM25 and dense retrieval
        ranks via RRF. `candidate_pool` controls how deep each individual
        retriever looks before fusion (deeper pool = better recall, more
        compute).
        """
        bm25_hits = self._bm25_ranked(query, candidate_pool)
        dense_hits = self._dense_ranked(query, candidate_pool)

        bm25_rank = {idx: r for r, (idx, _) in enumerate(bm25_hits)}
        dense_rank = {idx: r for r, (idx, _) in enumerate(dense_hits)}

        all_idx = set(bm25_rank) | set(dense_rank)
        fused = defaultdict(float)
        for idx in all_idx:
            if idx in bm25_rank:
                fused[idx] += 1.0 / (rrf_k + bm25_rank[idx] + 1)
            if idx in dense_rank:
                fused[idx] += 1.0 / (rrf_k + dense_rank[idx] + 1)

        ranked = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        return [
            RetrievalResult(
                chunk=self.chunks[idx],
                score=score,
                bm25_rank=bm25_rank.get(idx),
                dense_rank=dense_rank.get(idx),
            )
            for idx, score in ranked
        ]
