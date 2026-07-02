#!/usr/bin/env python3
"""Run scalar-probe Jacobian influence measurement."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from jacobian import (  # noqa: E402
    build_record,
    clear_cuda_cache,
    cuda_peak_memory_bytes,
    dependency_versions,
    find_default_data_file,
    iter_nq_examples,
    load_causal_lm,
    load_checkpoint,
    load_tokenizer,
    make_prompt_from_example,
    measure_jacobian_curve,
    reset_cuda_peak_memory,
    save_checkpoint,
    set_seed,
    tokenize_prompt_exactly,
    utc_now_iso,
)


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="gpt2", help="HuggingFace model id.")
    parser.add_argument("--init", choices=["random", "pretrained"], default="random")
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--n-seqs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data-file", type=Path, default=find_default_data_file(project_root))
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="Override n-seqs to at most 2.")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    return parser.parse_args()


def safe_model_name(model_name: str) -> str:
    return model_name.replace("/", "__").replace("\\", "__").replace(":", "_")


def default_out_dir(args: argparse.Namespace) -> Path:
    project_root = Path(__file__).resolve().parents[1]
    return (
        project_root
        / "results"
        / "jacobian"
        / f"{safe_model_name(args.model)}_{args.init}_L{args.seq_len}_seed{args.seed}"
    )


def append_results_md(project_root: Path, metadata: dict, records: list[dict], out_dir: Path) -> None:
    results_path = project_root / "RESULTS.md"
    if not records:
        return
    summary = np.load(out_dir / "log10_percentiles.npz")
    final_over_mid = records[-1]["rho_final"] / max(records[-1]["rho_middle"], 1e-300)
    block = f"""

### {metadata["model"]} {metadata["init"]} L={metadata["seq_len"]} seed={metadata["seed"]}

- Completed: {metadata["completed_n_seqs"]}/{metadata["target_n_seqs"]} sequences
- Data file: `{metadata["data_file"]}`
- Output: `{out_dir}`
- Plot: `{out_dir / "curve.png"}`
- Device: {metadata.get("device", "unknown")}
- CUDA peak memory bytes: {metadata.get("cuda_peak_memory_bytes")}
- Skipped short examples: {metadata.get("skipped_short_examples", 0)}
- Median log10 rho range: {float(summary["log10_p50"].min()):.4g} to {float(summary["log10_p50"].max()):.4g}
- Last recorded final/middle rho ratio: {final_over_mid:.4g}
- Updated UTC: {metadata["updated_utc"]}
"""
    with results_path.open("a", encoding="utf-8") as handle:
        handle.write(block)


def main() -> int:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    target_n_seqs = min(args.n_seqs, 2) if args.smoke else args.n_seqs
    out_dir = args.out or default_out_dir(args)

    if not args.data_file.exists():
        raise FileNotFoundError(
            f"Data file not found: {args.data_file}. Extract lost-in-the-middle-main.zip "
            "next to position_bias or pass --data-file."
        )

    set_seed(args.seed)
    existing_curves, records, previous_metadata = load_checkpoint(out_dir) if args.resume else (None, [], {})
    curve_list: list[np.ndarray] = []
    if existing_curves is not None:
        curve_list = [existing_curves[i] for i in range(existing_curves.shape[0])]
    if len(curve_list) > target_n_seqs:
        curve_list = curve_list[:target_n_seqs]
        records = records[:target_n_seqs]

    metadata: dict = {
        **previous_metadata,
        "model": args.model,
        "init": args.init,
        "seq_len": args.seq_len,
        "target_n_seqs": target_n_seqs,
        "seed": args.seed,
        "data_file": str(args.data_file.resolve()),
        "out_dir": str(out_dir.resolve()),
        "resume": bool(args.resume),
        "smoke": bool(args.smoke),
        "gradient_checkpointing": bool(args.gradient_checkpointing),
        "started_utc": previous_metadata.get("started_utc", utc_now_iso()),
        "dependency_versions": dependency_versions(),
        "skipped_short_examples": int(previous_metadata.get("skipped_short_examples", 0)),
    }

    if len(curve_list) >= target_n_seqs:
        save_checkpoint(out_dir, curve_list, records, metadata, plot_title=f"{args.model} {args.init} L={args.seq_len}")
        append_results_md(project_root, metadata, records, out_dir)
        print(f"Already complete: {len(curve_list)}/{target_n_seqs} curves in {out_dir}")
        return 0

    print(f"Loading tokenizer and {args.init} fp32 model: {args.model}")
    tokenizer = load_tokenizer(args.model)
    model = load_causal_lm(
        args.model,
        init=args.init,
        seed=args.seed,
        gradient_checkpointing=args.gradient_checkpointing,
    )
    device = str(next(model.parameters()).device)
    metadata["device"] = device

    reset_cuda_peak_memory()
    last_source_index = max((int(record["source_index"]) for record in records), default=-1)
    needed = target_n_seqs - len(curve_list)
    progress = tqdm(total=needed, desc="jacobian sequences")

    try:
        for source_index, example in enumerate(iter_nq_examples(args.data_file)):
            if source_index <= last_source_index:
                continue
            if len(curve_list) >= target_n_seqs:
                break

            prompt = make_prompt_from_example(example)
            tokenized = tokenize_prompt_exactly(tokenizer, prompt, args.seq_len)
            if tokenized is None:
                metadata["skipped_short_examples"] = int(metadata.get("skipped_short_examples", 0)) + 1
                continue

            input_ids, attention_mask, token_count = tokenized
            start = time.perf_counter()
            curve = measure_jacobian_curve(model, input_ids, attention_mask)
            elapsed = time.perf_counter() - start
            clear_cuda_cache()

            record = build_record(example, source_index, token_count, args.seq_len, curve, elapsed)
            curve_list.append(curve)
            records.append(record)
            metadata["cuda_peak_memory_bytes"] = cuda_peak_memory_bytes()
            save_checkpoint(out_dir, curve_list, records, metadata, plot_title=f"{args.model} {args.init} L={args.seq_len}")
            progress.update(1)
            progress.set_postfix(
                {
                    "source": source_index,
                    "rho_final": f"{record['rho_final']:.2e}",
                    "rho_mid": f"{record['rho_middle']:.2e}",
                }
            )
    finally:
        progress.close()

    if len(curve_list) < target_n_seqs:
        metadata["status"] = "incomplete_not_enough_long_examples"
        save_checkpoint(out_dir, curve_list, records, metadata, plot_title=f"{args.model} {args.init} L={args.seq_len}")
        print(json.dumps(metadata, indent=2))
        raise RuntimeError(f"Only collected {len(curve_list)} of {target_n_seqs} requested sequences.")

    metadata["status"] = "complete"
    metadata["cuda_peak_memory_bytes"] = cuda_peak_memory_bytes()
    save_checkpoint(out_dir, curve_list, records, metadata, plot_title=f"{args.model} {args.init} L={args.seq_len}")
    append_results_md(project_root, metadata, records, out_dir)

    summary = np.load(out_dir / "log10_percentiles.npz")
    final_over_middle = records[-1]["rho_final"] / max(records[-1]["rho_middle"], 1e-300)
    print(f"Completed {len(curve_list)}/{target_n_seqs} curves.")
    print(f"Output: {out_dir}")
    print(f"Plot: {out_dir / 'curve.png'}")
    print(f"Median log10 rho min/max: {summary['log10_p50'].min():.4g} / {summary['log10_p50'].max():.4g}")
    print(f"Last-record final/middle rho ratio: {final_over_middle:.4g}")
    print(f"CUDA peak memory bytes: {metadata.get('cuda_peak_memory_bytes')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
