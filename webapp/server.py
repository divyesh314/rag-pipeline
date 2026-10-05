"""
Web app for the RAG pipeline.

Serves a single-page UI and a small JSON API on top of the existing
`RAGPipeline`. Nothing in `src/rag` is duplicated here: search, reranking
and answer generation all run through the same code as the CLI demo, so
the app automatically uses real embeddings / cross-encoder / Claude when
those are installed and configured.

Run:
    pip install -r requirements.txt -r requirements-web.txt
    uvicorn webapp.server:app --reload
    # open http://127.0.0.1:8000

Environment:
    RAG_DOCUMENTS   path to a documents JSON file (default: data/documents.json)
    RAG_EVAL_SET    path to an eval set used for sample questions (default: data/eval_set.json)
    ANTHROPIC_API_KEY   enables real Claude answers (otherwise mock mode)
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rag.pipeline import RAGPipeline  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("rag.webapp")

DOCS_PATH = Path(os.environ.get("RAG_DOCUMENTS", ROOT / "data" / "documents.json"))
EVAL_PATH = Path(os.environ.get("RAG_EVAL_SET", ROOT / "data" / "eval_set.json"))
STATIC_DIR = Path(__file__).resolve().parent / "static"

MAX_QUESTION_CHARS = 500


# ---------------------------------------------------------------- data

def _load_json(path: Path, default):
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return default


def _title_for(doc: dict) -> str:
    """Use metadata.title if the document has one, else a readable form of its id."""
    title = (doc.get("metadata") or {}).get("title")
    if title:
        return title
    words = doc["id"].removeprefix("doc_").replace("_", " ").split()
    acronyms = {"rag", "ml", "ai", "db", "llm", "api"}
    return " ".join(w.upper() if w in acronyms else w.capitalize() for w in words)


def _sample_questions(eval_items: list) -> dict:
    """Map each document id to the first eval-set question that targets it."""
    samples: dict = {}
    for item in eval_items:
        for rid in item.get("relevant_doc_ids", []):
            doc_id = rid.split("::", 1)[0]
            samples.setdefault(doc_id, item["query"])
    return samples


DOCUMENTS: List[dict] = _load_json(DOCS_PATH, [])
if not DOCUMENTS:
    raise RuntimeError(f"No documents found at {DOCS_PATH}")
SAMPLES = _sample_questions(_load_json(EVAL_PATH, []))
DOC_INDEX = {d["id"]: d for d in DOCUMENTS}

pipeline = RAGPipeline(DOCUMENTS)


# ---------------------------------------------------------------- API models

class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=MAX_QUESTION_CHARS)
    top_k: int = Field(3, ge=1, le=10)


class Source(BaseModel):
    n: int
    doc_id: str
    chunk_id: str
    title: str
    topic: Optional[str]
    text: str
    fusion_score: float
    keyword_rank: Optional[int]
    semantic_rank: Optional[int]


class AskResponse(BaseModel):
    question: str
    answer: str
    sources: List[Source]
    llm: str
    elapsed_ms: int


# ---------------------------------------------------------------- app

app = FastAPI(title="RAG Pipeline", version="1.0")


@app.get("/api/status")
def status():
    return {
        "documents": len(DOCUMENTS),
        "chunks": len(pipeline.chunks),
        "embedder": pipeline.embedder.name,
        "reranker": pipeline.reranker.name,
        "llm": pipeline.llm.name,
        "real_answers": pipeline.llm.name == "claude",
    }


@app.get("/api/topics")
def topics():
    chunk_counts: dict = {}
    for c in pipeline.chunks:
        chunk_counts[c.doc_id] = chunk_counts.get(c.doc_id, 0) + 1
    return [
        {
            "id": d["id"],
            "title": _title_for(d),
            "topic": (d.get("metadata") or {}).get("topic"),
            "preview": d["text"][:140].rsplit(" ", 1)[0] + ("…" if len(d["text"]) > 140 else ""),
            "sample_question": SAMPLES.get(d["id"]),
            "chunks": chunk_counts.get(d["id"], 0),
        }
        for d in DOCUMENTS
    ]


@app.post("/api/ask", response_model=AskResponse)
def ask(req: AskRequest):
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=422, detail="Type a question first.")

    started = time.perf_counter()
    try:
        response = pipeline.query(question, top_k=req.top_k)
    except Exception as exc:  # surface generator/API failures as a readable error
        logger.exception("Query failed")
        raise HTTPException(status_code=502, detail=f"The answer step failed: {exc}") from exc
    elapsed = int((time.perf_counter() - started) * 1000)

    sources = []
    for n, r in enumerate(response.sources, 1):
        doc = DOC_INDEX.get(r.chunk.doc_id, {})
        sources.append(
            Source(
                n=n,
                doc_id=r.chunk.doc_id,
                chunk_id=r.chunk.chunk_id,
                title=_title_for(doc) if doc else r.chunk.doc_id,
                topic=(r.chunk.metadata or {}).get("topic"),
                text=r.chunk.text,
                fusion_score=round(float(r.score), 6),
                # retriever ranks are 0-based; show 1-based positions to people
                keyword_rank=None if r.bm25_rank is None else r.bm25_rank + 1,
                semantic_rank=None if r.dense_rank is None else r.dense_rank + 1,
            )
        )

    answer = response.answer
    if pipeline.llm.name.startswith("mock") and sources:
        # The mock generator wraps the top passage in setup instructions; the UI
        # already explains demo mode, so show just the passage with its citation.
        answer = f"{sources[0].text} [1]"

    return AskResponse(
        question=question,
        answer=answer,
        sources=sources,
        llm=pipeline.llm.name,
        elapsed_ms=elapsed,
    )


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC_DIR / "index.html")
