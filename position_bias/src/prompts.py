"""Prompt construction for the Liu et al. NaturalQuestions multi-document QA data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence


QA_INSTRUCTION = (
    "Write a high-quality answer for the given question using only the provided "
    "search results (some of which might be irrelevant)."
)


@dataclass(frozen=True)
class DocumentSpan:
    """Character span for one document block inside a generated prompt."""

    index: int
    title: str
    text: str
    isgold: bool
    start: int
    end: int


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
    prompt, _ = build_nq_prompt_with_spans(question, ctxs)
    return prompt


def build_nq_prompt_with_spans(
    question: str,
    ctxs: Sequence[Mapping[str, object]],
) -> tuple[str, list[DocumentSpan]]:
    """Build an NQ prompt and record character spans for each document block."""
    question = question.strip()
    if not question:
        raise ValueError("Question must be non-empty.")
    if not ctxs:
        raise ValueError("At least one context document is required.")

    parts = [QA_INSTRUCTION, "\n\n"]
    spans: list[DocumentSpan] = []
    for i, doc in enumerate(ctxs):
        if i > 0:
            parts.append("\n")
        start = sum(len(part) for part in parts)
        block = format_document(i + 1, doc)
        parts.append(block)
        end = start + len(block)
        spans.append(
            DocumentSpan(
                index=i,
                title=str(doc.get("title", "")).strip(),
                text=str(doc.get("text", "")).strip(),
                isgold=bool(doc.get("isgold")),
                start=start,
                end=end,
            )
        )

    parts.extend(["\n\nQuestion: ", question, "\nAnswer:"])
    return "".join(parts), spans


def gold_document_indices(ctxs: Sequence[Mapping[str, object]]) -> list[int]:
    """Return zero-based positions of gold documents in an NQ example."""
    return [i for i, doc in enumerate(ctxs) if bool(doc.get("isgold"))]
