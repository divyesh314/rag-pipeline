"""
Reranking: a second, more expensive relevance pass over the retriever's
top candidates before they go to the LLM.

The original script's "reranker" multiplied the retrieval score by
`1 + len(doc.split()) / 100` — i.e. it rewarded longer documents,
regardless of whether they answered the query. That's not a weak
reranker, it's actively counterproductive: it will actively promote
long, unfocused chunks over short, precise ones.

A real reranker scores each (query, candidate) pair jointly with a
cross-encoder, which is slower than the bi-encoder retrieval step but far
more accurate, because it can attend across the query and document
together instead of comparing two independently-computed vectors. Only
run it on the retriever's top ~20-30 candidates, never the whole corpus.

Fallback (no sentence-transformers installed): score by query-term
overlap density within the candidate, which at least responds to actual
content instead of length.
"""

from __future__ import annotations

from typing import List, Protocol

from .bm25 import tokenize
from .retriever import RetrievalResult


class Reranker(Protocol):
    name: str

    def rerank(self, query: str, results: List[RetrievalResult]) -> List[RetrievalResult]: ...


class CrossEncoderReranker:
    """Real cross-encoder reranker. Requires: pip install sentence-transformers"""

    name = "cross-encoder (ms-marco-MiniLM-L-6-v2)"

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        from sentence_transformers import CrossEncoder  # lazy import

        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, results: List[RetrievalResult]) -> List[RetrievalResult]:
        pairs = [(query, r.chunk.text) for r in results]
        scores = self.model.predict(pairs)
        rescored = [
            RetrievalResult(chunk=r.chunk, score=float(s), bm25_rank=r.bm25_rank, dense_rank=r.dense_rank)
            for r, s in zip(results, scores)
        ]
        return sorted(rescored, key=lambda r: r.score, reverse=True)


class LexicalOverlapReranker:
    """
    Dependency-free fallback: scores candidates by the fraction of query
    terms they contain, normalized so length doesn't dominate (unlike the
    original script's length-biased "reranker").
    """

    name = "lexical-overlap (fallback, not a real cross-encoder)"

    def rerank(self, query: str, results: List[RetrievalResult]) -> List[RetrievalResult]:
        q_terms = set(tokenize(query))
        if not q_terms:
            return results

        rescored = []
        for r in results:
            doc_terms = tokenize(r.chunk.text)
            doc_term_set = set(doc_terms)
            overlap = len(q_terms & doc_term_set) / len(q_terms)
            # small length penalty so verbose-but-relevant doesn't always
            # beat concise-and-relevant
            density = overlap / (1 + 0.01 * len(doc_terms))
            rescored.append(
                RetrievalResult(
                    chunk=r.chunk, score=density, bm25_rank=r.bm25_rank, dense_rank=r.dense_rank
                )
            )
        return sorted(rescored, key=lambda r: r.score, reverse=True)


def get_reranker(prefer: str = "auto") -> Reranker:
    if prefer in ("auto", "cross-encoder"):
        try:
            return CrossEncoderReranker()
        except ImportError:
            if prefer == "cross-encoder":
                raise
    return LexicalOverlapReranker()
