# Hybrid RAG Pipeline

A Retrieval-Augmented Generation pipeline with **hybrid search (BM25 + dense
embeddings, fused with Reciprocal Rank Fusion)**, a **cross-encoder
reranking stage**, and an **evaluation harness** with real IR metrics
(Precision@K, Recall@K, MRR, nDCG@K).

Runs fully offline out of the box (zero-dependency BM25, TF-IDF fallback
embeddings, a mock LLM), and upgrades automatically to real sentence
embeddings, a real cross-encoder, and real Claude-generated answers the
moment you install the optional dependencies and set an API key. Nothing
needs to be rewritten to go from "runs on my laptop with no network" to
"production-grade."

```
documents.json
      │
      ▼
  chunking.py ──────► overlapping passage chunks
      │
      ▼
 ┌────────────────────────────┐
 │      HybridRetriever        │
 │  ┌────────┐   ┌───────────┐ │
 │  │ BM25   │   │ Dense      │ │
 │  │(lexical)│  │(embeddings)│ │
 │  └───┬────┘   └─────┬─────┘ │
 │      └── RRF fusion ┘       │
 └──────────────┬──────────────┘
                ▼
          top-N candidates
                ▼
        reranker.py (cross-encoder)
                ▼
          top-K passages
                ▼
        generation.py (Claude / mock)
                ▼
           cited answer
```

## Why this exists

I started from a common "toy RAG" pattern (small corpus, TF-IDF
similarity used for both "keyword" and "semantic" search, a reranker that
scores by document length, a hardcoded 4-query eval set) and rebuilt each
piece to match how these systems are actually built in production:

| Piece | Toy version | This version |
|---|---|---|
| Keyword search | TF-IDF cosine similarity | Real Okapi BM25 (term-frequency saturation + length normalization) |
| "Semantic" search | Same TF-IDF vectors reused — not semantic at all | Pluggable dense embedder; real sentence-transformer embeddings when available, TF-IDF fallback clearly labeled as non-semantic when not |
| Fusion | `alpha * score_a + (1-alpha) * score_b` on incomparable scales | Reciprocal Rank Fusion — the technique Elasticsearch/Weaviate hybrid search actually uses |
| Reranking | Multiplies score by document length | Real cross-encoder (`ms-marco-MiniLM-L-6-v2`) when available, lexical-overlap fallback otherwise — never rewards length |
| Chunking | None — whole sentences treated as documents | Recursive paragraph/sentence/character splitter with overlap |
| Generation | `"[LLM generation placeholder]"` string | Real Claude API call with a grounded, citation-required prompt, or an honest mock when no key is set |
| Eval | Recall@3 on 4 hardcoded queries | Precision/Recall/MRR/nDCG@K, loaded from a JSON eval set you can grow without touching code |

## Quickstart

```bash
pip install -r requirements.txt
python examples/run_demo.py
```

This runs entirely offline (TF-IDF embeddings, lexical-overlap reranker,
mock generator) and prints a sample query plus a full evaluation report
against `data/eval_set.json`.

### Upgrade to real models

```bash
pip install -r requirements.txt -r requirements-full.txt
export ANTHROPIC_API_KEY=sk-...
python examples/run_demo.py
```

The pipeline logs which embedder/reranker/LLM it's actually using
(`Embedder: sentence-transformers` vs `Embedder: tfidf (lexical fallback...)`)
so you always know whether you're getting the real thing.

### Run tests

```bash
pip install pytest
pytest tests/ -v
```

## Using it as a library

```python
import json
from src.rag.pipeline import RAGPipeline

documents = json.loads(open("data/documents.json").read())
pipeline = RAGPipeline(documents)

response = pipeline.query("How does reranking improve on the first-stage retriever?")
print(response.answer)
for source in response.sources:
    print(source.chunk.chunk_id, source.score)
```

## Extending it

- **New embedding model**: implement the `Embedder` protocol in
  `src/rag/embeddings.py` and add it to `get_embedder()`.
- **Real vector DB at scale**: swap `HybridRetriever`'s in-memory
  `doc_vectors` array for FAISS/Chroma/Pinecone once your corpus outgrows
  brute-force cosine search (this becomes necessary in the low millions of
  chunks).
- **Different LLM provider**: implement the `LLM` protocol in
  `src/rag/generation.py`.
- **Bigger eval set**: add entries to `data/eval_set.json` — no code
  changes needed. `relevance` grades are optional (defaults to binary
  relevance) and support graded relevance for nDCG.

## Honest limitations

- The in-memory retriever does brute-force cosine similarity — fine for
  thousands of chunks, not for a real production index. See "Extending
  it" above.
- Generation faithfulness (does the answer actually stick to the cited
  context) isn't automatically scored here — that typically needs an
  LLM-as-judge step, which is a reasonable next addition to `eval.py`.
- BM25 and the fallback embedder are implemented from scratch for
  zero-dependency portability, not because hand-rolling them beats
  `rank_bm25`/established libraries in production.

## Project structure

```
rag-pipeline/
├── src/rag/
│   ├── chunking.py      # document -> overlapping passages
│   ├── bm25.py           # lexical retrieval
│   ├── embeddings.py     # pluggable dense embedder
│   ├── retriever.py      # BM25 + dense fusion via RRF
│   ├── reranker.py       # pluggable cross-encoder reranking
│   ├── generation.py     # pluggable LLM answer generation
│   ├── pipeline.py       # orchestrates the above
│   └── eval.py           # Precision/Recall/MRR/nDCG harness
├── data/
│   ├── documents.json    # sample corpus
│   └── eval_set.json     # sample eval set with relevance judgments
├── examples/run_demo.py
└── tests/test_pipeline.py
```
