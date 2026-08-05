"""
End-to-end pipeline: documents -> chunks -> hybrid retrieval -> rerank ->
grounded generation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List

from .chunking import Chunk, chunk_text
from .embeddings import get_embedder
from .generation import get_llm
from .reranker import get_reranker
from .retriever import HybridRetriever, RetrievalResult

logger = logging.getLogger("rag")


@dataclass
class RAGResponse:
    query: str
    answer: str
    sources: List[RetrievalResult] = field(default_factory=list)


class RAGPipeline:
    def __init__(
        self,
        documents: List[dict],  # [{"id": str, "text": str, "metadata": {...}}]
        chunk_size: int = 800,
        chunk_overlap: int = 150,
        embedder_pref: str = "auto",
        reranker_pref: str = "auto",
        llm_pref: str = "auto",
    ):
        self.chunks: List[Chunk] = []
        for doc in documents:
            self.chunks.extend(
                chunk_text(
                    doc["text"],
                    doc_id=doc["id"],
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    metadata=doc.get("metadata", {}),
                )
            )
        logger.info("Chunked %d documents into %d chunks", len(documents), len(self.chunks))

        texts = [c.text for c in self.chunks]
        self.embedder = get_embedder(texts, prefer=embedder_pref)
        logger.info("Embedder: %s", self.embedder.name)

        self.retriever = HybridRetriever(self.chunks, self.embedder)
        self.reranker = get_reranker(prefer=reranker_pref)
        logger.info("Reranker: %s", self.reranker.name)

        self.llm = get_llm(prefer=llm_pref)
        logger.info("LLM: %s", self.llm.name)

    def query(
        self, question: str, top_k: int = 3, candidate_pool: int = 20
    ) -> RAGResponse:
        retrieved = self.retriever.search(question, top_k=candidate_pool)
        reranked = self.reranker.rerank(question, retrieved)[:top_k]
        answer = self.llm.generate(question, reranked)
        return RAGResponse(query=question, answer=answer, sources=reranked)

    def retrieve_only(self, question: str, top_k: int = 5) -> List[RetrievalResult]:
        """Useful for evaluation: skip generation, just return ranked chunks."""
        retrieved = self.retriever.search(question, top_k=max(top_k, 20))
        return self.reranker.rerank(question, retrieved)[:top_k]
