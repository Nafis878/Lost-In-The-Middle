"""Jacobian influence measurement for Phase 1."""

from __future__ import annotations

import gzip
import json
import os
import platform
import random
import re
import sys
import time
from collections.abc import Sequence as RuntimeSequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, MutableMapping, Sequence

import numpy as np

from prompts import DocumentSpan, build_nq_prompt, build_nq_prompt_with_spans, gold_document_indices


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


def safe_model_name(model_name: str) -> str:
    return model_name.replace("/", "__").replace("\\", "__").replace(":", "_")


def data_file_info(path: str | Path) -> dict[str, int | str]:
    """Infer document count and gold position from Liu NQ file names."""
    path_obj = Path(path)
    match = re.search(r"(\d+)_total_documents_gold_at_(\d+)", path_obj.name)
    if not match:
        match = re.search(r"(\d+)_total_documents.*gold_at_(\d+)", str(path_obj))
    if not match:
        raise ValueError(f"Could not infer doc count/gold position from {path_obj}")
    return {
        "doc_count": int(match.group(1)),
        "gold_position": int(match.group(2)),
        "file_name": path_obj.name,
    }


def target_n_examples(n_examples: int, smoke: bool, smoke_n: int = 5) -> int:
    return min(n_examples, smoke_n) if smoke else n_examples


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


def model_context_limit(tokenizer: Any, model: Any | None = None) -> int | None:
    """Return a sane context limit when tokenizer/model expose one."""
    candidates: list[int] = []
    tokenizer_limit = getattr(tokenizer, "model_max_length", None)
    if isinstance(tokenizer_limit, int) and 0 < tokenizer_limit < 1_000_000:
        candidates.append(tokenizer_limit)

    config = getattr(model, "config", None)
    if config is not None:
        for attr in ("max_position_embeddings", "n_positions", "seq_length"):
            value = getattr(config, attr, None)
            if isinstance(value, int) and value > 0:
                candidates.append(value)
    return min(candidates) if candidates else None


def tokenize_prompt_natural(tokenizer: Any, prompt: str) -> tuple[Any, Any | None, list[tuple[int, int]]]:
    """Tokenize a full prompt without truncation and return offsets for span mapping."""
    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=True,
        truncation=False,
        return_offsets_mapping=True,
    )
    if "offset_mapping" not in encoded:
        raise ValueError("Tokenizer did not return offset_mapping; use a fast tokenizer.")
    offsets_raw = encoded.pop("offset_mapping")[0].tolist()
    offsets = [(int(start), int(end)) for start, end in offsets_raw]
    return encoded["input_ids"], encoded.get("attention_mask"), offsets


def token_span_from_char_span(offsets: Sequence[Sequence[int]], start: int, end: int) -> tuple[int, int]:
    """Map a character span to the token span whose offsets overlap it."""
    hits: list[int] = []
    for i, pair in enumerate(offsets):
        token_start, token_end = int(pair[0]), int(pair[1])
        if token_end <= token_start:
            continue
        if token_end > start and token_start < end:
            hits.append(i)
    if not hits:
        raise ValueError(f"No tokens overlapped character span [{start}, {end}).")
    return hits[0], hits[-1] + 1


def normalize_for_span_check(text: str) -> str:
    return " ".join(text.split())


def decode_and_validate_span(
    tokenizer: Any,
    input_ids: Any,
    token_span: tuple[int, int],
    document_text: str,
) -> str:
    """Decode a token span and assert it contains the original document text."""
    start, end = token_span
    token_ids = input_ids[0, start:end]
    if hasattr(token_ids, "tolist"):
        token_ids = token_ids.tolist()
    decoded = tokenizer.decode(token_ids, skip_special_tokens=True)
    expected = normalize_for_span_check(document_text)
    observed = normalize_for_span_check(decoded)
    if expected not in observed:
        raise ValueError(
            "Decoded gold token span does not contain the gold document text. "
            f"Span={token_span}, expected prefix={expected[:120]!r}, decoded prefix={observed[:120]!r}"
        )
    return decoded


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


def append_text_atomic(path: Path, text: str) -> None:
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    atomic_write_text(path, existing + text)


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


def write_jsonl_atomic(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    write_records(path, records)


def load_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_jsonl_records(path: str | Path) -> list[dict[str, Any]]:
    return load_records(Path(path))


def append_results_block(project_root: Path, block: str) -> None:
    append_text_atomic(project_root / "RESULTS.md", block)


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


def first_gold_span(spans: Sequence[DocumentSpan]) -> DocumentSpan:
    gold_spans = [span for span in spans if span.isgold]
    if not gold_spans:
        raise ValueError("Example has no gold document span.")
    return gold_spans[0]


def prompt_and_gold_span(example: Mapping[str, Any]) -> tuple[str, list[DocumentSpan], DocumentSpan]:
    prompt, spans = build_nq_prompt_with_spans(str(example["question"]), example["ctxs"])
    return prompt, spans, first_gold_span(spans)


def summarize_jacobian_span(curve: np.ndarray, token_span: tuple[int, int]) -> dict[str, float | list[int]]:
    start, end = token_span
    if start < 0 or end > len(curve) or start >= end:
        raise ValueError(f"Invalid token span {token_span} for curve length {len(curve)}.")
    span_curve = curve[start:end].astype(np.float64, copy=False)
    clipped = np.clip(span_curve, 1e-300, None)
    return {
        "gold_token_span": [int(start), int(end)],
        "jac_gold_mean": float(np.mean(span_curve)),
        "jac_gold_max": float(np.max(span_curve)),
        "jac_gold_logmean": float(np.mean(np.log10(clipped))),
        "jac_last_token": float(curve[-1]),
        "jac_prompt_median": float(np.median(curve)),
    }


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
