"""Placeholder QA evaluation helpers for Phase 1 Task 2.

Only the Liu et al. normalization and best_subspan_em metric are implemented now so
local tests can lock metric parity before the full QA generation script is added.
"""

from __future__ import annotations

import string
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from jacobian import atomic_write_json, data_file_info

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


DOC_POSITIONS = {
    10: [0, 4, 9],
    20: [0, 4, 9, 14, 19],
}


def score_prediction(prediction: str, answers: Sequence[str]) -> float:
    """Score only the first prediction line, matching the Liu evaluator guardrail."""
    return best_subspan_em(prediction.split("\n")[0].strip(), answers)


def bootstrap_accuracy_ci(
    scores: Sequence[float],
    seed: int,
    n_resamples: int = 10_000,
    confidence: float = 0.95,
) -> list[float | None]:
    """Bootstrap a percentile CI for mean accuracy."""
    values = np.asarray(scores, dtype=np.float64)
    if values.size == 0:
        return [None, None]
    rng = np.random.default_rng(seed)
    sample_indices = rng.integers(0, values.size, size=(n_resamples, values.size))
    means = values[sample_indices].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    lo, hi = np.quantile(means, [alpha, 1.0 - alpha])
    return [float(lo), float(hi)]


def gold_position(example: dict[str, Any]) -> int | None:
    for i, ctx in enumerate(example.get("ctxs", [])):
        if bool(ctx.get("isgold")):
            return i
    return None


def load_generation_model(model_name: str, dtype: str, seed: int):
    """Load an inference-only causal LM for greedy QA generation."""
    import torch
    from transformers import AutoModelForCausalLM

    if dtype not in {"fp16", "fp32"}:
        raise ValueError(f"Unsupported dtype={dtype!r}.")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch_dtype = torch.float16 if dtype == "fp16" else torch.float32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    )
    model.to(device)
    model.eval()
    return model


def generate_answer(
    model: Any,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int,
) -> tuple[str, int]:
    """Greedy-decode only newly generated tokens from a base causal LM prompt."""
    import torch

    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=True,
        truncation=False,
    )
    input_ids = encoded["input_ids"].to(next(model.parameters()).device)
    attention_mask = encoded.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(input_ids.device)
    prompt_token_len = int(input_ids.shape[1])
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=pad_token_id,
        )
    new_token_ids = output_ids[0, prompt_token_len:]
    return tokenizer.decode(new_token_ids, skip_special_tokens=True).strip(), prompt_token_len


def summarize_qa_run(
    records: Sequence[dict[str, Any]],
    skipped: Sequence[dict[str, Any]],
    model_name: str,
    data_file: Path,
    out_dir: Path,
    seed: int,
    dtype: str,
    max_new_tokens: int,
    requested_n: int,
    wall_time_sec: float,
    cuda_peak_memory_bytes: int | None,
) -> dict[str, Any]:
    scores = [float(record["score"]) for record in records]
    info = data_file_info(data_file)
    accuracy = float(np.mean(scores)) if scores else None
    summary = {
        "task": "qa_eval",
        "model": model_name,
        "data_file": str(data_file.resolve()),
        "out_dir": str(out_dir.resolve()),
        "doc_count": info["doc_count"],
        "gold_position": info["gold_position"],
        "dtype": dtype,
        "max_new_tokens": int(max_new_tokens),
        "seed": int(seed),
        "requested_n_examples": int(requested_n),
        "n": len(scores),
        "n_skipped": len(skipped),
        "accuracy": accuracy,
        "bootstrap_ci_95": bootstrap_accuracy_ci(scores, seed=seed) if scores else [None, None],
        "cuda_peak_memory_bytes": cuda_peak_memory_bytes,
        "wall_time_sec": round(float(wall_time_sec), 4),
        "verdict": "pending_condition",
    }
    return summary


def update_weak_model_verdicts(summary_path: Path) -> None:
    """Mark all position summaries weak when a full doc-count condition is <10%."""
    if not summary_path.exists():
        return
    current = _read_json(summary_path)
    doc_count = current.get("doc_count")
    expected_positions = DOC_POSITIONS.get(int(doc_count)) if doc_count is not None else None
    if not expected_positions:
        return

    condition_dir = summary_path.parent.parent
    summaries: dict[int, tuple[Path, dict[str, Any]]] = {}
    for candidate in condition_dir.glob("gold_at_*/summary.json"):
        payload = _read_json(candidate)
        if payload.get("model") != current.get("model"):
            continue
        if int(payload.get("doc_count", -1)) != int(doc_count):
            continue
        position = int(payload.get("gold_position", -1))
        summaries[position] = (candidate, payload)

    if not all(position in summaries for position in expected_positions):
        return

    all_weak = all((summaries[position][1].get("accuracy") or 0.0) < 0.10 for position in expected_positions)
    verdict = "model_too_weak" if all_weak else "usable"
    for _, (path, payload) in summaries.items():
        payload["verdict"] = verdict
        atomic_write_json(path, payload)


def _read_json(path: Path) -> dict[str, Any]:
    import json

    return json.loads(path.read_text(encoding="utf-8"))
