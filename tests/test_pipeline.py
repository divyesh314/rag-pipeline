import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest

from rag.bm25 import BM25, tokenize
from rag.chunking import chunk_text
from rag.eval import _mrr, _ndcg_at_k, _precision_at_k, _recall_at_k
from rag.pipeline import RAGPipeline


# ---- chunking ----

def test_chunk_short_text_stays_one_chunk():
    chunks = chunk_text("Hello world.", doc_id="d1", chunk_size=800, chunk_overlap=100)
    assert len(chunks) == 1
    assert chunks[0].text == "Hello world."


def test_chunk_long_text_splits_and_overlaps():
    text = ("This is a sentence about cats. " * 100).strip()
    chunks = chunk_text(text, doc_id="d1", chunk_size=200, chunk_overlap=50)
    assert len(chunks) > 1
    for c in chunks:
        assert len(c.text) <= 260  # allow small slack from overlap stitching


def test_chunk_overlap_must_be_smaller_than_size():
    with pytest.raises(ValueError):
        chunk_text("hi", doc_id="d1", chunk_size=100, chunk_overlap=100)


# ---- bm25 ----

def test_bm25_exact_keyword_match_ranks_first():
    corpus = [
        "the quick brown fox jumps over the lazy dog",
        "completely unrelated text about finance and taxes",
        "another fox related document about foxes in the wild",
    ]
    bm25 = BM25(corpus)
    results = bm25.search("fox", top_k=3)
    top_idx = results[0][0]
    assert top_idx in (0, 2)


def test_tokenize_lowercases_and_strips_punctuation():
    assert tokenize("Hello, World!") == ["hello", "world"]


# ---- eval metrics ----

def test_precision_recall_perfect_match():
    retrieved = ["a", "b", "c"]
    relevant = {"a", "b", "c"}
    assert _precision_at_k(retrieved, relevant, 3) == 1.0
    assert _recall_at_k(retrieved, relevant, 3) == 1.0


def test_mrr_first_hit_position():
    assert _mrr(["x", "y", "z"], {"z"}) == pytest.approx(1 / 3)
    assert _mrr(["z", "y", "x"], {"z"}) == 1.0
    assert _mrr(["x", "y"], {"q"}) == 0.0


def test_ndcg_rewards_earlier_relevant_hits():
    relevance = {"a": 2, "b": 1}
    ndcg_good_order = _ndcg_at_k(["a", "b", "c"], relevance, k=3)
    ndcg_bad_order = _ndcg_at_k(["c", "b", "a"], relevance, k=3)
    assert ndcg_good_order > ndcg_bad_order
    assert ndcg_good_order == pytest.approx(1.0)


# ---- pipeline integration (forces the offline fallback path) ----

@pytest.fixture(scope="module")
def pipeline():
    docs = [
        {"id": "paris", "text": "Paris is the capital of France and home to the Eiffel Tower."},
        {"id": "py", "text": "Python is a popular programming language for data science."},
    ]
    return RAGPipeline(docs, embedder_pref="tfidf", reranker_pref="lexical")


def test_pipeline_retrieves_relevant_chunk(pipeline):
    results = pipeline.retrieve_only("What is the capital of France?", top_k=1)
    assert len(results) == 1
    assert results[0].chunk.doc_id == "paris"


def test_pipeline_query_returns_answer_and_sources(pipeline):
    response = pipeline.query("Tell me about Python", top_k=1)
    assert response.answer
    assert len(response.sources) == 1
    assert response.sources[0].chunk.doc_id == "py"
