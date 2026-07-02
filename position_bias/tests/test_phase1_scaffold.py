from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from jacobian import aggregate_curves, load_checkpoint, save_checkpoint
from prompts import build_nq_prompt, gold_document_indices
from qa_eval import best_subspan_em, normalize_answer


def test_prompt_matches_liu_format() -> None:
    prompt = build_nq_prompt(
        "Who won?",
        [
            {"title": "Alpha", "text": "First document.", "isgold": True},
            {"title": "Beta", "text": "Second document.", "isgold": False},
        ],
    )
    assert prompt.startswith("Write a high-quality answer")
    assert "Document [1](Title: Alpha) First document." in prompt
    assert "Document [2](Title: Beta) Second document." in prompt
    assert prompt.endswith("Question: Who won?\nAnswer:")
    assert gold_document_indices(
        [{"isgold": False}, {"isgold": True}, {"isgold": False}]
    ) == [1]


def test_metric_parity_with_liu_logic() -> None:
    assert normalize_answer("The, QUICK answer!") == "quick answer"
    assert best_subspan_em("It was Wilhelm Conrad Rontgen.", ["Wilhelm Conrad Rontgen"]) == 1.0
    assert best_subspan_em("A different answer.", ["Wilhelm Conrad Rontgen"]) == 0.0


def test_aggregate_curves_shapes() -> None:
    curves = np.array([[1.0, 1e-3, 10.0], [10.0, 1e-4, 100.0]], dtype=np.float64)
    summary = aggregate_curves(curves)
    assert set(summary) == {"x", "log10_p16", "log10_p50", "log10_p84"}
    assert summary["x"].shape == (3,)
    assert summary["log10_p50"].shape == (3,)
    assert np.all(np.isfinite(summary["log10_p50"]))


def test_checkpoint_round_trip(tmp_path: Path) -> None:
    curves = np.array([[1.0, 2.0, 3.0]], dtype=np.float64)
    records = [{"source_index": 0, "rho_final": 3.0, "rho_middle": 2.0}]
    metadata = {"model": "gpt2", "init": "random", "seq_len": 3}
    save_checkpoint(tmp_path, curves, records, metadata)

    loaded_curves, loaded_records, loaded_metadata = load_checkpoint(tmp_path)
    assert loaded_curves is not None
    assert loaded_curves.shape == (1, 3)
    assert loaded_records == records
    assert loaded_metadata["completed_n_seqs"] == 1
    assert json.loads((tmp_path / "metadata.json").read_text())["seq_len"] == 3
