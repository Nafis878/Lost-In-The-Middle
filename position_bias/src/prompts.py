"""Prompt construction for the Liu et al. NaturalQuestions multi-document QA data."""

from __future__ import annotations

from typing import Mapping, Sequence


QA_INSTRUCTION = (
    "Write a high-quality answer for the given question using only the provided "
    "search results (some of which might be irrelevant)."
)


def format_document(index: int, document: Mapping[str, object]) -> str:
    """Format one retrieved document in the Liu et al. prompt style."""
    title = str(document.get("title", "")).strip()
    text = str(document.get("text", "")).strip()
    if not title:
        raise ValueError(f"Document {index} is missing a title.")
    if not text:
        raise ValueError(f"Document {index} is missing text.")
    return f"Document [{index}](Title: {title}) {text}"


def build_nq_prompt(question: str, ctxs: Sequence[Mapping[str, object]]) -> str:
    """Build the NaturalQuestions multi-document prompt used in Lost in the Middle."""
    question = question.strip()
    if not question:
        raise ValueError("Question must be non-empty.")
    if not ctxs:
        raise ValueError("At least one context document is required.")

    documents = "\n".join(format_document(i + 1, doc) for i, doc in enumerate(ctxs))
    return f"{QA_INSTRUCTION}\n\n{documents}\n\nQuestion: {question}\nAnswer:"


def gold_document_indices(ctxs: Sequence[Mapping[str, object]]) -> list[int]:
    """Return zero-based positions of gold documents in an NQ example."""
    return [i for i, doc in enumerate(ctxs) if bool(doc.get("isgold"))]
