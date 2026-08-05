"""
Embedding providers, behind one interface.

Real-world constraint this file is designed around: your dev laptop, your
CI runner, and your production box often don't have the same packages or
network access. A RAG project that hard-codes `sentence-transformers` (or
an API key) breaks the moment one of those is missing. So every provider
here implements the same `Embedder` protocol, and `get_embedder()` tries
providers in order of quality and silently falls back, but always tells you
which one it picked (check `.name`) so you're never fooled about whether
you're getting real semantic embeddings or a TF-IDF stand-in.

Providers, best to worst:
  1. SentenceTransformerEmbedder - real dense embeddings (all-MiniLM-L6-v2
     by default). Use this in production. Requires `pip install
     sentence-transformers`.
  2. TfidfEmbedder - sparse lexical vectors via scikit-learn. This is NOT
     semantic search — two sentences with zero shared vocabulary but
     identical meaning will score low. It's a legitimate lexical signal,
     just don't call it "semantic" (the original script did, which is the
     bug this file exists to fix).
"""

from __future__ import annotations

from typing import List, Protocol

import numpy as np


class Embedder(Protocol):
    name: str

    def encode(self, texts: List[str]) -> np.ndarray: ...


class SentenceTransformerEmbedder:
    """Real dense embeddings. Requires: pip install sentence-transformers"""

    name = "sentence-transformers"

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer  # lazy import

        self.model = SentenceTransformer(model_name)

    def encode(self, texts: List[str]) -> np.ndarray:
        vecs = self.model.encode(texts, normalize_embeddings=True)
        return np.asarray(vecs)


class TfidfEmbedder:
    """
    Sparse lexical fallback using scikit-learn's TfidfVectorizer.
    Honest about what it is: lexical overlap, not meaning. Fit once on a
    reference corpus; `encode()` transforms new text into that fitted space.
    """

    name = "tfidf (lexical fallback, not semantic)"

    def __init__(self, corpus: List[str]):
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.vectorizer.fit(corpus)

    def encode(self, texts: List[str]) -> np.ndarray:
        mat = self.vectorizer.transform(texts)
        return mat.toarray()


def get_embedder(corpus: List[str], prefer: str = "auto") -> Embedder:
    """
    prefer: "auto" | "sentence-transformers" | "tfidf"
    "auto" tries the real embedding model first and falls back cleanly.
    """
    if prefer in ("auto", "sentence-transformers"):
        try:
            return SentenceTransformerEmbedder()
        except ImportError:
            if prefer == "sentence-transformers":
                raise
    return TfidfEmbedder(corpus)


def cosine_sim_matrix(query_vec: np.ndarray, doc_vecs: np.ndarray) -> np.ndarray:
    q = query_vec / (np.linalg.norm(query_vec) + 1e-9)
    d_norms = np.linalg.norm(doc_vecs, axis=1, keepdims=True) + 1e-9
    d = doc_vecs / d_norms
    return d @ q
