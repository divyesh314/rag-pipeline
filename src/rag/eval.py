"""
Evaluation harness for retrieval quality.

The original script measured only Recall@3 on 4 hand-written queries, which
is easy to hit 1.0 on by accident with a 7-document toy corpus — it doesn't
tell you much. This module adds the standard IR metrics side by side:

  Precision@K - of the K chunks you retrieved, what fraction were relevant?
  Recall@K    - of all relevant chunks, what fraction did you retrieve?
  MRR         - how high up was the FIRST relevant result? (matters a lot
                for RAG since the LLM weights early context more)
  nDCG@K      - like recall, but rewards relevant results appearing earlier,
                and supports graded (not just binary) relevance

Eval sets are loaded from a JSON file (data/eval_set.json), not hardcoded
in Python, so growing the eval set doesn't mean editing source code.
Expected format:

    [
      {"query": "...", "relevant_doc_ids": ["doc1::chunk_0"], "relevance": {"doc1::chunk_0": 2}}
    ]

`relevance` is optional; if omitted, every id in relevant_doc_ids gets
relevance grade 1 (binary relevance).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from .pipeline import RAGPipeline


@dataclass
class EvalItem:
    query: str
    relevant_ids: List[str]
    relevance: Dict[str, int]


def load_eval_set(path: str | Path) -> List[EvalItem]:
    raw = json.loads(Path(path).read_text())
    items = []
    for r in raw:
        relevance = r.get("relevance") or {rid: 1 for rid in r["relevant_doc_ids"]}
        items.append(
            EvalItem(query=r["query"], relevant_ids=r["relevant_doc_ids"], relevance=relevance)
        )
    return items


def _precision_at_k(retrieved_ids: List[str], relevant: set, k: int) -> float:
    top = retrieved_ids[:k]
    if not top:
        return 0.0
    return sum(1 for rid in top if rid in relevant) / len(top)


def _recall_at_k(retrieved_ids: List[str], relevant: set, k: int) -> float:
    if not relevant:
        return 0.0
    top = retrieved_ids[:k]
    return sum(1 for rid in top if rid in relevant) / len(relevant)


def _mrr(retrieved_ids: List[str], relevant: set) -> float:
    for rank, rid in enumerate(retrieved_ids, start=1):
        if rid in relevant:
            return 1.0 / rank
    return 0.0


def _ndcg_at_k(retrieved_ids: List[str], relevance: Dict[str, int], k: int) -> float:
    top = retrieved_ids[:k]
    dcg = sum(
        relevance.get(rid, 0) / math.log2(i + 2) for i, rid in enumerate(top)
    )
    ideal_gains = sorted(relevance.values(), reverse=True)[:k]
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal_gains))
    return dcg / idcg if idcg > 0 else 0.0


def evaluate(pipeline: RAGPipeline, eval_items: List[EvalItem], k: int = 5) -> dict:
    per_query = []
    for item in eval_items:
        results = pipeline.retrieve_only(item.query, top_k=k)
        retrieved_ids = [r.chunk.chunk_id for r in results]
        relevant = set(item.relevant_ids)

        per_query.append(
            {
                "query": item.query,
                "precision@k": _precision_at_k(retrieved_ids, relevant, k),
                "recall@k": _recall_at_k(retrieved_ids, relevant, k),
                "mrr": _mrr(retrieved_ids, relevant),
                "ndcg@k": _ndcg_at_k(retrieved_ids, item.relevance, k),
                "retrieved": retrieved_ids,
            }
        )

    n = len(per_query) or 1
    summary = {
        "k": k,
        "n_queries": len(per_query),
        "mean_precision@k": sum(p["precision@k"] for p in per_query) / n,
        "mean_recall@k": sum(p["recall@k"] for p in per_query) / n,
        "mean_mrr": sum(p["mrr"] for p in per_query) / n,
        "mean_ndcg@k": sum(p["ndcg@k"] for p in per_query) / n,
    }
    return {"summary": summary, "per_query": per_query}
