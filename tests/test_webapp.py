import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from webapp.server import app  # noqa: E402

client = TestClient(app)


def test_index_serves_ui():
    r = client.get("/")
    assert r.status_code == 200
    assert "Ask the documents" in r.text


def test_status_reports_components():
    s = client.get("/api/status").json()
    assert s["documents"] > 0 and s["chunks"] >= s["documents"]
    assert {"embedder", "reranker", "llm"} <= s.keys()


def test_topics_list_every_document():
    topics = client.get("/api/topics").json()
    ids = {t["id"] for t in topics}
    assert "doc_paris" in ids
    assert all(t["title"] for t in topics)


def test_ask_returns_answer_and_cited_sources():
    r = client.post("/api/ask", json={"question": "What is the capital of France?"})
    assert r.status_code == 200
    body = r.json()
    assert body["answer"]
    assert body["sources"][0]["doc_id"] == "doc_paris"
    assert [s["n"] for s in body["sources"]] == list(range(1, len(body["sources"]) + 1))


@pytest.mark.parametrize("payload", [{"question": ""}, {"question": "   "}, {"question": "x" * 501}])
def test_ask_rejects_bad_questions(payload):
    assert client.post("/api/ask", json=payload).status_code == 422
