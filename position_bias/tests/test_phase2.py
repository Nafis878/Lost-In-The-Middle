from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import numpy as np

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from correlate import fit_logistic_regression, spearmanr
from jacobian import (
    decode_and_validate_span,
    load_jsonl_records,
    token_span_from_char_span,
    tokenize_prompt_natural,
    write_jsonl_atomic,
)
from prompts import build_nq_prompt, build_nq_prompt_with_spans
from qa_eval import bootstrap_accuracy_ci


DATA_ROOT = Path(__file__).resolve().parents[2] / "lost-in-the-middle-main" / "qa_data"


def read_examples(path: Path, n: int) -> list[dict]:
    examples = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for _ in range(n):
            examples.append(json.loads(next(handle)))
    return examples


def test_prompt_with_spans_matches_plain_builder() -> None:
    example = {
        "question": "Who won?",
        "ctxs": [
            {"title": "Alpha", "text": "First document.", "isgold": False},
            {"title": "Beta", "text": "Second document.", "isgold": True},
        ],
    }
    prompt, spans = build_nq_prompt_with_spans(example["question"], example["ctxs"])
    assert prompt == build_nq_prompt(example["question"], example["ctxs"])
    assert prompt[spans[1].start : spans[1].end] == "Document [2](Title: Beta) Second document."
    assert spans[1].isgold


def test_gold_span_char_round_trip_on_real_examples() -> None:
    paths = [
        DATA_ROOT / "10_total_documents" / "nq-open-10_total_documents_gold_at_4.jsonl.gz",
        DATA_ROOT / "20_total_documents" / "nq-open-20_total_documents_gold_at_9.jsonl.gz",
    ]
    checked = 0
    for path in paths:
        for example in read_examples(path, 3):
            prompt, spans = build_nq_prompt_with_spans(example["question"], example["ctxs"])
            gold_spans = [span for span in spans if span.isgold]
            assert len(gold_spans) == 1
            gold = gold_spans[0]
            block = prompt[gold.start : gold.end]
            assert gold.text in block
            assert block.startswith(f"Document [{gold.index + 1}](Title:")
            checked += 1
    assert checked >= 5


def test_token_span_from_offsets_with_fake_tokenizer() -> None:
    prompt = "aa Document [1](Title: T) hello world zz"
    start = prompt.index("Document")
    end = prompt.index(" zz")
    offsets = [(0, 2), (3, 11), (12, 22), (23, 28), (29, 34), (35, 40)]
    assert token_span_from_char_span(offsets, start, end) == (1, 6)


def test_bootstrap_accuracy_ci() -> None:
    lo, hi = bootstrap_accuracy_ci([1, 0, 1, 1, 0], seed=0, n_resamples=500)
    assert 0.0 <= lo <= hi <= 1.0


def test_spearman_positive_and_zero_fixtures() -> None:
    positive = spearmanr([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4])
    assert positive["rho"] == 1.0
    zeroish = spearmanr([1, 2, 3, 4], [1, 2, 1, 2])
    assert abs(zeroish["rho"]) < 0.5


def test_logistic_regression_positive_signal() -> None:
    records = []
    for i in range(40):
        jac = -4.0 + i * 0.2
        records.append({"score": float(i >= 20), "jac_gold_logmean": jac, "prompt_token_len": 100 + i})
    result = fit_logistic_regression(records)
    assert result["status"] == "ok"
    assert result["coef_jac_gold_logmean"] > 0
    assert result["auc"] is not None and result["auc"] > 0.9


def test_logistic_regression_single_class_fixture() -> None:
    records = [{"score": 0.0, "jac_gold_logmean": -3.0, "prompt_token_len": 100} for _ in range(5)]
    result = fit_logistic_regression(records)
    assert result["status"] == "single_class"


def test_jsonl_round_trip_for_resume(tmp_path: Path) -> None:
    path = tmp_path / "predictions.jsonl"
    write_jsonl_atomic(path, [{"example_idx": 0}, {"example_idx": 3}])
    completed = {record["example_idx"] for record in load_jsonl_records(path)}
    assert completed == {0, 3}


def test_gold_span_token_round_trip_optional_transformers() -> None:
    try:
        from transformers import AutoTokenizer
    except ImportError:
        return

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    path = DATA_ROOT / "10_total_documents" / "nq-open-10_total_documents_gold_at_0.jsonl.gz"
    for example in read_examples(path, 5):
        prompt, spans = build_nq_prompt_with_spans(example["question"], example["ctxs"])
        gold = [span for span in spans if span.isgold][0]
        input_ids, _, offsets = tokenize_prompt_natural(tokenizer, prompt)
        token_span = token_span_from_char_span(offsets, gold.start, gold.end)
        decoded = decode_and_validate_span(tokenizer, input_ids, token_span, gold.text)
        assert gold.text.split()[0] in decoded
