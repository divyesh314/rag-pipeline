"""
Okapi BM25 — the actual algorithm behind "keyword search" in Elasticsearch,
Lucene, Postgres full-text search, etc.

The original version scored keyword relevance with a hand-rolled TF-IDF
cosine similarity. That's a defensible *dense* signal, but it isn't what
"keyword search" means in the field, and using the same TF-IDF vectors for
both halves of a "hybrid" search made the two retrievers redundant. BM25
adds term-frequency saturation (a term matching 10 times isn't 10x as
relevant as matching once) and document-length normalization, which is why
it beats plain TF-IDF cosine on keyword-heavy queries in practice.

Zero dependencies on purpose (no rank_bm25 package) so this runs anywhere.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import List, Sequence

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


class BM25:
    """Standard Okapi BM25 over a fixed corpus of pre-tokenized documents."""

    def __init__(self, corpus: Sequence[str], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus_tokens: List[List[str]] = [tokenize(doc) for doc in corpus]
        self.doc_lengths = [len(toks) for toks in self.corpus_tokens]
        self.avg_doc_len = (
            sum(self.doc_lengths) / len(self.doc_lengths) if self.doc_lengths else 0.0
        )
        self.n_docs = len(corpus)

        self.doc_term_counts: List[Counter] = [Counter(toks) for toks in self.corpus_tokens]

        df: Counter = Counter()
        for toks in self.corpus_tokens:
            for term in set(toks):
                df[term] += 1
        self.doc_freq = df
        self.idf = {
            term: math.log(1 + (self.n_docs - freq + 0.5) / (freq + 0.5))
            for term, freq in df.items()
        }

    def _score_doc(self, query_terms: List[str], doc_idx: int) -> float:
        score = 0.0
        term_counts = self.doc_term_counts[doc_idx]
        doc_len = self.doc_lengths[doc_idx]
        for term in query_terms:
            if term not in term_counts:
                continue
            f = term_counts[term]
            idf = self.idf.get(term, 0.0)
            denom = f + self.k1 * (1 - self.b + self.b * doc_len / (self.avg_doc_len or 1))
            score += idf * (f * (self.k1 + 1)) / (denom or 1e-9)
        return score

    def search(self, query: str, top_k: int = 10) -> List[tuple[int, float]]:
        query_terms = tokenize(query)
        scores = [self._score_doc(query_terms, i) for i in range(self.n_docs)]
        ranked = sorted(range(self.n_docs), key=lambda i: scores[i], reverse=True)
        return [(i, scores[i]) for i in ranked[:top_k] if scores[i] > 0]
