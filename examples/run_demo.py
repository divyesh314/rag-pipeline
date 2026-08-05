"""
End-to-end demo: build the pipeline from data/documents.json, run a query,
then run the full evaluation harness against data/eval_set.json.

Usage:
    python examples/run_demo.py
"""

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag.eval import evaluate, load_eval_set
from rag.pipeline import RAGPipeline

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

ROOT = Path(__file__).resolve().parents[1]


def main():
    documents = json.loads((ROOT / "data" / "documents.json").read_text())
    pipeline = RAGPipeline(documents)

    print("\n=== Sample query ===")
    question = "How does reranking improve on what the first-stage retriever already did?"
    response = pipeline.query(question, top_k=3)
    print(f"Q: {response.query}\n")
    print(f"A: {response.answer}\n")
    print("Sources used:")
    for i, r in enumerate(response.sources, 1):
        print(f"  [{i}] {r.chunk.chunk_id}  (fusion_score={r.score:.4f})")
        print(f"      {r.chunk.text[:120]}...")

    print("\n=== Evaluation (data/eval_set.json) ===")
    eval_items = load_eval_set(ROOT / "data" / "eval_set.json")
    report = evaluate(pipeline, eval_items, k=3)

    print(f"n_queries: {report['summary']['n_queries']}  (@k={report['summary']['k']})")
    print(f"mean Precision@k: {report['summary']['mean_precision@k']:.3f}")
    print(f"mean Recall@k:    {report['summary']['mean_recall@k']:.3f}")
    print(f"mean MRR:         {report['summary']['mean_mrr']:.3f}")
    print(f"mean nDCG@k:      {report['summary']['mean_ndcg@k']:.3f}")

    print("\nPer-query breakdown:")
    for pq in report["per_query"]:
        print(
            f"  P={pq['precision@k']:.2f} R={pq['recall@k']:.2f} "
            f"MRR={pq['mrr']:.2f} nDCG={pq['ndcg@k']:.2f}  | {pq['query']}"
        )


if __name__ == "__main__":
    main()
