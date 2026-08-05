"""
Text chunking.

Real documents (PDFs, wiki pages, support tickets) are not one sentence long.
If you embed/index whole documents, retrieval quality collapses because a
100-page PDF and a one-line FAQ answer end up competing for the same
"most relevant" slot. Chunking splits documents into overlapping, roughly
fixed-size windows so retrieval operates at the right granularity.

This is a dependency-free recursive splitter: try to split on paragraph
breaks first, then sentences, then words, only falling back to hard
character cuts if a single "sentence" is still too long. Overlap keeps
context from being severed mid-idea at chunk boundaries.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List


@dataclass
class Chunk:
    text: str
    doc_id: str
    chunk_id: str
    start_char: int
    end_char: int
    metadata: dict = field(default_factory=dict)


_PARAGRAPH_SPLIT = re.compile(r"\n\s*\n")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _split_on(pattern: "re.Pattern[str]", text: str) -> List[str]:
    parts = [p for p in pattern.split(text) if p.strip()]
    return parts if parts else [text]


def chunk_text(
    text: str,
    doc_id: str,
    chunk_size: int = 800,
    chunk_overlap: int = 150,
    metadata: dict | None = None,
) -> List[Chunk]:
    """
    Recursively split `text` into chunks of at most `chunk_size` characters,
    with `chunk_overlap` characters of overlap between consecutive chunks.

    chunk_size/chunk_overlap are in characters here to stay dependency-free;
    swap for a tokenizer (tiktoken, HF tokenizer) in production so chunk
    size lines up with your embedding model's token limit.
    """
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    units = _split_on(_PARAGRAPH_SPLIT, text)
    windows: List[str] = []
    for unit in units:
        if len(unit) <= chunk_size:
            windows.append(unit)
            continue
        # paragraph too big -> split into sentences and pack greedily
        sentences = _split_on(_SENTENCE_SPLIT, unit)
        current = ""
        for sent in sentences:
            if len(sent) > chunk_size:
                # single sentence still too long -> hard character split
                if current:
                    windows.append(current)
                    current = ""
                for i in range(0, len(sent), chunk_size):
                    windows.append(sent[i : i + chunk_size])
                continue
            if len(current) + len(sent) + 1 <= chunk_size:
                current = f"{current} {sent}".strip()
            else:
                if current:
                    windows.append(current)
                current = sent
        if current:
            windows.append(current)

    # apply char-based overlap by walking the original text for offsets,
    # and stitching a trailing slice of the previous window onto the next
    chunks: List[Chunk] = []
    cursor = 0
    prev_tail = ""
    for i, w in enumerate(windows):
        idx = text.find(w, cursor)
        if idx == -1:
            idx = cursor  # fallback if whitespace normalization shifted it
        merged = (prev_tail + " " + w).strip() if prev_tail else w
        start = max(idx - len(prev_tail), 0)
        end = idx + len(w)
        chunks.append(
            Chunk(
                text=merged,
                doc_id=doc_id,
                chunk_id=f"{doc_id}::chunk_{i}",
                start_char=start,
                end_char=end,
                metadata=dict(metadata or {}),
            )
        )
        prev_tail = w[-chunk_overlap:] if len(w) > chunk_overlap else w
        cursor = end

    return chunks
