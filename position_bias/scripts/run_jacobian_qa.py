#!/usr/bin/env python3
"""Run exact-prompt gold-span Jacobian measurement for QA correlation."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from jacobian import (  # noqa: E402
    append_results_block,
    atomic_save_npy,
    atomic_write_json,
    clear_cuda_cache,
    cuda_peak_memory_bytes,
    data_file_info,
    decode_and_validate_span,
    dependency_versions,
    find_default_data_file,
    iter_nq_examples,
    load_causal_lm,
    load_jsonl_records,
    load_tokenizer,
    measure_jacobian_curve,
    model_context_limit,
    prompt_and_gold_span,
    reset_cuda_peak_memory,
    safe_model_name,
    set_seed,
    summarize_jacobian_span,
    target_n_examples,
    token_span_from_char_span,
    tokenize_prompt_natural,
    utc_now_iso,
    write_jsonl_atomic,
)


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--init", choices=["random", "pretrained"], default="pretrained")
    parser.add_argument("--data-file", type=Path, default=find_default_data_file(project_root))
    parser.add_argument("--n-examples", type=int, default=100)
    parser.add_argument("--max-len", type=int, default=3072)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def default_out_dir(args: argparse.Namespace) -> Path:
    project_root = Path(__file__).resolve().parents[1]
    info = data_file_info(args.data_file)
    return (
        project_root
        / "results"
        / "jacobian_qa"
        / safe_model_name(args.model)
        / args.init
        / f"{info['doc_count']}_docs"
        / f"gold_at_{info['gold_position']}"
    )


def append_results(project_root: Path, summary: dict, out_dir: Path) -> None:
    block = f"""

### Jacobian-QA {summary["model"]} {summary["init"]} {summary["doc_count"]}-doc gold@{summary["gold_position"]}

- Output: `{out_dir}`
- Measured/skipped: {summary["n"]}/{summary["n_skipped"]}
- Median jac_gold_logmean: {summary["jac_gold_logmean_median"]}
- Full curves saved: {summary["full_curves_saved"]}
- CUDA peak memory bytes: {summary["cuda_peak_memory_bytes"]}
- Wall time sec: {summary["wall_time_sec"]}
- Updated UTC: {utc_now_iso()}
"""
    append_results_block(project_root, block)


def make_summary(
    records: list[dict],
    skipped: list[dict],
    args: argparse.Namespace,
    out_dir: Path,
    wall_time_sec: float,
) -> dict:
    info = data_file_info(args.data_file)
    logmeans = np.asarray([record["jac_gold_logmean"] for record in records], dtype=np.float64)
    return {
        "task": "jacobian_qa",
        "model": args.model,
        "init": args.init,
        "data_file": str(args.data_file.resolve()),
        "out_dir": str(out_dir.resolve()),
        "doc_count": info["doc_count"],
        "gold_position": info["gold_position"],
        "seed": args.seed,
        "requested_n_examples": target_n_examples(args.n_examples, args.smoke, smoke_n=5),
        "max_len": args.max_len,
        "n": len(records),
        "n_skipped": len(skipped),
        "jac_gold_logmean_median": float(np.median(logmeans)) if logmeans.size else None,
        "jac_gold_logmean_iqr": (
            [float(x) for x in np.quantile(logmeans, [0.25, 0.75])] if logmeans.size else [None, None]
        ),
        "full_curves_saved": len(list((out_dir / "full_curves").glob("*.npy"))) if (out_dir / "full_curves").exists() else 0,
        "cuda_peak_memory_bytes": cuda_peak_memory_bytes(),
        "wall_time_sec": round(float(wall_time_sec), 4),
        "updated_utc": utc_now_iso(),
    }


def main() -> int:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    out_dir = args.out or default_out_dir(args)
    out_dir.mkdir(parents=True, exist_ok=True)
    target_n = target_n_examples(args.n_examples, args.smoke, smoke_n=5)

    if not args.data_file.exists():
        raise FileNotFoundError(f"Data file not found: {args.data_file}")

    set_seed(args.seed)
    records_path = out_dir / "records.jsonl"
    skipped_path = out_dir / "skipped.jsonl"
    records = load_jsonl_records(records_path) if args.resume else []
    skipped = load_jsonl_records(skipped_path) if args.resume else []
    completed_or_skipped = {int(record["example_idx"]) for record in records}
    completed_or_skipped.update(int(record["example_idx"]) for record in skipped)

    metadata = {
        "task": "jacobian_qa",
        "model": args.model,
        "init": args.init,
        "data_file": str(args.data_file.resolve()),
        "out_dir": str(out_dir.resolve()),
        "target_n_examples": target_n,
        "max_len": args.max_len,
        "seed": args.seed,
        "started_utc": utc_now_iso(),
        "dependency_versions": dependency_versions(),
        "gradient_checkpointing": True,
    }
    atomic_write_json(out_dir / "metadata.json", metadata)

    print(f"Loading tokenizer and fp32 {args.init} model: {args.model}")
    tokenizer = load_tokenizer(args.model)
    model = load_causal_lm(
        args.model,
        init=args.init,
        seed=args.seed,
        gradient_checkpointing=True,
    )
    context_limit = model_context_limit(tokenizer, model)
    effective_max_len = min(args.max_len, context_limit) if context_limit is not None else args.max_len
    metadata["context_limit"] = context_limit
    metadata["effective_max_len"] = effective_max_len
    atomic_write_json(out_dir / "metadata.json", metadata)
    reset_cuda_peak_memory()

    started = time.perf_counter()
    progress = tqdm(total=target_n, initial=min(len(completed_or_skipped), target_n), desc="jacobian qa")
    try:
        for example_idx, example in enumerate(iter_nq_examples(args.data_file)):
            if example_idx >= target_n:
                break
            if example_idx in completed_or_skipped:
                continue

            prompt, _, gold_span = prompt_and_gold_span(example)
            input_ids, attention_mask, offsets = tokenize_prompt_natural(tokenizer, prompt)
            prompt_token_len = int(input_ids.shape[1])
            if prompt_token_len > effective_max_len:
                skipped.append(
                    {
                        "example_idx": example_idx,
                        "reason": "prompt_exceeds_max_len",
                        "prompt_token_len": prompt_token_len,
                        "max_len": args.max_len,
                        "context_limit": context_limit,
                    }
                )
                write_jsonl_atomic(skipped_path, skipped)
                completed_or_skipped.add(example_idx)
                progress.update(1)
                continue

            token_span = token_span_from_char_span(offsets, gold_span.start, gold_span.end)
            decode_and_validate_span(tokenizer, input_ids, token_span, gold_span.text)

            try:
                curve = measure_jacobian_curve(model, input_ids, attention_mask)
            except RuntimeError as exc:
                clear_cuda_cache()
                if "out of memory" in str(exc).lower():
                    skipped.append(
                        {
                            "example_idx": example_idx,
                            "reason": "cuda_out_of_memory",
                            "prompt_token_len": prompt_token_len,
                            "max_len": args.max_len,
                        }
                    )
                    write_jsonl_atomic(skipped_path, skipped)
                    atomic_write_json(out_dir / "metadata.json", {**metadata, "last_error": str(exc)})
                raise

            summary = summarize_jacobian_span(curve, token_span)
            record = {
                "example_idx": int(example_idx),
                "question": str(example["question"]),
                "gold_position": int(gold_span.index),
                "prompt_token_len": prompt_token_len,
                **summary,
            }
            if len(records) < 20:
                curve_path = out_dir / "full_curves" / f"example_{example_idx:06d}.npy"
                atomic_save_npy(curve_path, curve.astype(np.float32))
                record["full_curve_path"] = str(curve_path)
            records.append(record)
            write_jsonl_atomic(records_path, records)
            completed_or_skipped.add(example_idx)
            clear_cuda_cache()

            progress.update(1)
            progress.set_postfix({"tokens": prompt_token_len, "logmean": f"{record['jac_gold_logmean']:.3g}"})
    finally:
        progress.close()

    wall_time_sec = time.perf_counter() - started
    summary = make_summary(records, skipped, args, out_dir, wall_time_sec)
    atomic_write_json(out_dir / "summary.json", summary)
    append_results(project_root, summary, out_dir)

    print(f"Completed Jacobian-QA run: {summary['n']} measured, {summary['n_skipped']} skipped")
    print(f"Median jac_gold_logmean: {summary['jac_gold_logmean_median']}")
    print(f"Output: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
