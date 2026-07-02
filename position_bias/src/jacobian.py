"""Jacobian influence measurement for Phase 1."""

from __future__ import annotations

import gzip
import json
import os
import platform
import random
import sys
import time
from collections.abc import Sequence as RuntimeSequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, MutableMapping, Sequence

import numpy as np

from prompts import build_nq_prompt, gold_document_indices


DEFAULT_DATA_RELATIVE = Path(
    "lost-in-the-middle-main/qa_data/20_total_documents/"
    "nq-open-20_total_documents_gold_at_0.jsonl.gz"
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and torch when torch is available."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def dependency_versions() -> dict[str, str]:
    """Return available dependency versions without requiring heavyweight imports."""
    versions: dict[str, str] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
    }
    for package_name in ("torch", "transformers", "scipy", "matplotlib", "tqdm"):
        try:
            module = __import__(package_name)
        except ImportError:
            versions[package_name] = "not-installed"
        else:
            versions[package_name] = str(getattr(module, "__version__", "unknown"))
    return versions


def find_default_data_file(project_root: Path | None = None) -> Path:
    """Find the default Liu 20-document NQ file from common local/Colab layouts."""
    candidates: list[Path] = []
    if project_root is not None:
        root = project_root.resolve()
        candidates.extend([root.parent / DEFAULT_DATA_RELATIVE, root / DEFAULT_DATA_RELATIVE])
    cwd = Path.cwd().resolve()
    candidates.extend([cwd / DEFAULT_DATA_RELATIVE, cwd.parent / DEFAULT_DATA_RELATIVE])

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0] if candidates else DEFAULT_DATA_RELATIVE


def iter_nq_examples(path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield examples from a gzipped NQ jsonl file."""
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_tokenizer(model_name: str):
    """Load a HuggingFace tokenizer and ensure it has a pad token when possible."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_causal_lm(
    model_name: str,
    init: str,
    seed: int,
    device: str | None = None,
    gradient_checkpointing: bool = False,
):
    """Load a causal LM in fp32, either pretrained or freshly initialized."""
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM

    if init not in {"random", "pretrained"}:
        raise ValueError(f"Unsupported init={init!r}; expected 'random' or 'pretrained'.")

    set_seed(seed)
    target_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if init == "random":
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            trust_remote_code=True,
        )

    model.to(target_device, dtype=torch.float32)
    model.eval()
    if gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False
    return model


def tokenize_prompt_exactly(
    tokenizer: Any,
    prompt: str,
    seq_len: int,
) -> tuple[Any, Any | None, int] | None:
    """Tokenize a prompt and return exactly seq_len input ids, or None if too short."""
    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=True,
        truncation=False,
    )
    input_ids = encoded["input_ids"]
    token_count = int(input_ids.shape[1])
    if token_count < seq_len:
        return None
    attention_mask = encoded.get("attention_mask")
    return input_ids[:, :seq_len], attention_mask[:, :seq_len] if attention_mask is not None else None, token_count


def measure_jacobian_curve(model: Any, input_ids: Any, attention_mask: Any | None = None) -> np.ndarray:
    """Compute rho[j] = ||d sum(logits_last) / d input_embedding[j]||_2."""
    import torch

    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    model.zero_grad(set_to_none=True)
    with torch.enable_grad():
        embeddings = model.get_input_embeddings()(input_ids).detach()
        embeddings.requires_grad_(True)
        outputs = model(inputs_embeds=embeddings, attention_mask=attention_mask, use_cache=False)
        scalar = outputs.logits[0, -1, :].sum()
        scalar.backward()
        if embeddings.grad is None:
            raise RuntimeError("No gradient reached inputs_embeds; check graph construction.")
        rho = torch.linalg.vector_norm(embeddings.grad[0].float(), ord=2, dim=-1)
        curve = rho.detach().cpu().numpy().astype(np.float64, copy=False)

    del outputs, scalar, embeddings, rho
    model.zero_grad(set_to_none=True)
    return curve


def clear_cuda_cache() -> None:
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def reset_cuda_peak_memory() -> None:
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def cuda_peak_memory_bytes() -> int | None:
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    return int(torch.cuda.max_memory_allocated())


def aggregate_curves(curves: np.ndarray) -> dict[str, np.ndarray]:
    """Aggregate raw rho curves into log10 percentile summaries."""
    if curves.ndim != 2:
        raise ValueError(f"Expected curves with shape [n, L], got {curves.shape}.")
    clipped = np.clip(curves.astype(np.float64, copy=False), 1e-300, None)
    log10_curves = np.log10(clipped)
    p16, p50, p84 = np.percentile(log10_curves, [16, 50, 84], axis=0)
    x = np.arange(curves.shape[1], dtype=np.float64) / float(curves.shape[1])
    return {"x": x, "log10_p16": p16, "log10_p50": p50, "log10_p84": p84}


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(path.name + ".tmp")
    temp_path.write_text(text, encoding="utf-8")
    os.replace(temp_path, path)


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def atomic_save_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(path.name + ".tmp")
    with temp_path.open("wb") as handle:
        np.save(handle, array)
    os.replace(temp_path, path)


def atomic_save_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(path.name + ".tmp")
    with temp_path.open("wb") as handle:
        np.savez(handle, **arrays)
    os.replace(temp_path, path)


def write_records(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    lines = "".join(json.dumps(record, sort_keys=True) + "\n" for record in records)
    atomic_write_text(path, lines)


def load_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_checkpoint(out_dir: str | Path) -> tuple[np.ndarray | None, list[dict[str, Any]], dict[str, Any]]:
    out_path = Path(out_dir)
    curves_path = out_path / "curves.npy"
    records = load_records(out_path / "records.jsonl")
    curves = np.load(curves_path) if curves_path.exists() else None
    metadata_path = out_path / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}

    if curves is not None and len(records) != int(curves.shape[0]):
        keep = min(len(records), int(curves.shape[0]))
        records = records[:keep]
        curves = curves[:keep]
    return curves, records, metadata


def save_plot(out_dir: Path, curves: np.ndarray, title: str) -> None:
    import matplotlib.pyplot as plt

    summary = aggregate_curves(curves)
    p50_max = float(np.max(summary["log10_p50"]))
    floor = 1e-12
    y16 = np.maximum(10.0 ** (summary["log10_p16"] - p50_max), floor)
    y50 = np.maximum(10.0 ** (summary["log10_p50"] - p50_max), floor)
    y84 = np.maximum(10.0 ** (summary["log10_p84"] - p50_max), floor)

    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(summary["x"], y50, color="#1f77b4", linewidth=1.7, label="median")
    ax.fill_between(summary["x"], y16, y84, color="#1f77b4", alpha=0.22, label="p16-p84")
    ax.set_yscale("log")
    ax.set_xlabel("Normalized token position j / L")
    ax.set_ylabel("Influence rho, median-max normalized")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()

    path = out_dir / "curve.png"
    temp_path = path.with_name(path.name + ".tmp.png")
    fig.savefig(temp_path, dpi=180)
    plt.close(fig)
    os.replace(temp_path, path)


def save_checkpoint(
    out_dir: str | Path,
    curves: Sequence[np.ndarray] | np.ndarray,
    records: Sequence[Mapping[str, Any]],
    metadata: MutableMapping[str, Any],
    plot_title: str | None = None,
) -> None:
    """Persist raw curves, records, metadata, aggregate summaries, and optional plot."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    curve_array = np.asarray(curves, dtype=np.float64)
    if curve_array.size == 0:
        curve_array = curve_array.reshape(0, 0)
    elif curve_array.ndim == 1:
        curve_array = curve_array.reshape(1, -1)

    metadata["completed_n_seqs"] = int(curve_array.shape[0]) if curve_array.size else 0
    metadata["updated_utc"] = utc_now_iso()
    atomic_save_npy(out_path / "curves.npy", curve_array)
    write_records(out_path / "records.jsonl", records)
    atomic_write_json(out_path / "metadata.json", metadata)

    if curve_array.size:
        summary = aggregate_curves(curve_array)
        atomic_save_npz(out_path / "log10_percentiles.npz", **summary)
        if plot_title:
            save_plot(out_path, curve_array, plot_title)


def build_record(
    example: Mapping[str, Any],
    source_index: int,
    token_count: int,
    seq_len: int,
    curve: np.ndarray,
    elapsed_sec: float,
) -> dict[str, Any]:
    ctxs = example.get("ctxs", [])
    mid_index = seq_len // 2
    return {
        "source_index": int(source_index),
        "question": str(example.get("question", "")),
        "answers": list(example.get("answers", [])),
        "gold_document_indices": gold_document_indices(ctxs) if isinstance(ctxs, RuntimeSequence) else [],
        "original_token_count": int(token_count),
        "seq_len": int(seq_len),
        "rho_min": float(np.min(curve)),
        "rho_max": float(np.max(curve)),
        "rho_first": float(curve[0]),
        "rho_middle": float(curve[mid_index]),
        "rho_final": float(curve[-1]),
        "elapsed_sec": round(float(elapsed_sec), 4),
    }


def make_prompt_from_example(example: Mapping[str, Any]) -> str:
    return build_nq_prompt(str(example["question"]), example["ctxs"])
