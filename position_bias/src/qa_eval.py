"""Placeholder QA evaluation helpers for Phase 1 Task 2.

Only the Liu et al. normalization and best_subspan_em metric are implemented now so
local tests can lock metric parity before the full QA generation script is added.
"""

from __future__ import annotations

import string
from typing import Sequence

try:
    import regex as re
except ImportError:  # pragma: no cover - fallback is for minimal local syntax checks.
    import re  # type: ignore[no-redef]


def normalize_answer(text: str) -> str:
    """SQuAD-style answer normalization used by Liu et al."""

    def remove_articles(value: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", value)

    def white_space_fix(value: str) -> str:
        return " ".join(value.split())

    def remove_punc(value: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in value if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(text.lower())))


def best_subspan_em(prediction: str, ground_truths: Sequence[str]) -> float:
    """Return 1.0 when any normalized gold answer is contained in the prediction."""
    normalized_prediction = normalize_answer(prediction)
    for ground_truth in ground_truths:
        normalized_ground_truth = normalize_answer(ground_truth)
        if normalized_ground_truth.lower() in normalized_prediction.lower():
            return 1.0
    return 0.0
