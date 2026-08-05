"""
Generation: turn retrieved chunks + query into a grounded answer.

The original script never actually called an LLM — it left a
"[LLM generation placeholder]" string. That's fine for a retrieval demo,
but it's not a RAG *pipeline* without the "G". This file wires up a real
call, plus a MockLLM so the pipeline still runs end-to-end (and CI/tests
still pass) without an API key or network access.

Prompting note: the context chunks are numbered and the model is
instructed to cite [1], [2], etc. and to say when the context doesn't
contain the answer, rather than to always produce a confident-sounding
answer. Ungrounded confident answers are the main failure mode of demo
RAG systems.
"""

from __future__ import annotations

import os
from typing import List, Protocol

from .retriever import RetrievalResult

SYSTEM_PROMPT = (
    "You are a precise question-answering assistant. Answer the user's "
    "question using ONLY the numbered context passages provided. Cite the "
    "passage number(s) you used like [1] or [1][3]. If the context does "
    "not contain enough information to answer, say so explicitly instead "
    "of guessing."
)


def build_prompt(query: str, results: List[RetrievalResult]) -> str:
    context_block = "\n\n".join(
        f"[{i+1}] {r.chunk.text}" for i, r in enumerate(results)
    )
    return f"Context passages:\n{context_block}\n\nQuestion: {query}\n\nAnswer:"


class LLM(Protocol):
    name: str

    def generate(self, query: str, results: List[RetrievalResult]) -> str: ...


class AnthropicLLM:
    """Real generation via the Claude API. Requires ANTHROPIC_API_KEY."""

    name = "claude"

    def __init__(self, model: str = "claude-sonnet-4-6"):
        import anthropic  # lazy import

        self.client = anthropic.Anthropic()
        self.model = model

    def generate(self, query: str, results: List[RetrievalResult]) -> str:
        prompt = build_prompt(query, results)
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=500,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(block.text for block in resp.content if block.type == "text")


class MockLLM:
    """
    Deterministic, dependency-free stand-in. Returns the top retrieved
    passage instead of a synthesized answer. This keeps the pipeline
    fully runnable offline (demos, tests, CI) without pretending to be a
    real generator.
    """

    name = "mock (no API key / offline mode)"

    def generate(self, query: str, results: List[RetrievalResult]) -> str:
        if not results:
            return "No relevant context was retrieved for this query."
        top = results[0]
        return (
            f"[MOCK LLM — no generation model configured]\n"
            f"Most relevant passage [1]: {top.chunk.text}\n"
            f"(Set ANTHROPIC_API_KEY and use AnthropicLLM for a real "
            f"synthesized, cited answer.)"
        )


def get_llm(prefer: str = "auto") -> LLM:
    if prefer in ("auto", "claude") and os.environ.get("ANTHROPIC_API_KEY"):
        try:
            return AnthropicLLM()
        except ImportError:
            if prefer == "claude":
                raise
    return MockLLM()
