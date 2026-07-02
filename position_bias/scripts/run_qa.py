#!/usr/bin/env python3
"""Run resumable NaturalQuestions QA accuracy evaluation."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from tqdm import tqdm

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from jacobian import (  # noqa: E402
    append_results_block,
    atomic_write_json,
    cuda_peak_memory_bytes,
    data_file_info,
    dependency_versions,
    find_default_data_file,
    iter_nq_examples,
    load_jsonl_records,
    load_tokenizer,
    model_context_limit,
    safe_model_name,
    set_seed,
    target_n_examples,
    utc_now_iso,
    write_jsonl_atomic,
)
from prompts import build_nq_prompt  # noqa: E402
from qa_eval import (  # noqa: E402
    generate_answer,
    gold_position,
    load_generation_model,
    score_prediction,
    summarize_qa_run,
    update_weak_model_verdicts,
)


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--data-file", type=Path, default=find_default_data_file(project_root))
    parser.add_argument("--n-examples", type=int, default=300)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dtype", choices=["fp16", "fp32"], default="fp16")
    return parser.parse_args()


def default_out_dir(args: argparse.Namespace) -> Path:
    project_root = Path(__file__).resolve().parents[1]
    info = data_file_info(args.data_file)
    return (
        project_root
        / "results"
        / "qa"
        / safe_model_name(args.model)
        / f"{info['doc_count']}_docs"
        / f"gold_at_{info['gold_position']}"
    )


def prompt_token_len(tokenizer, prompt: str) -> int:
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=True, truncation=False)
    return int(encoded["input_ids"].shape[1])


def append_results(project_root: Path, summary: dict, out_dir: Path) -> None:
    ci = summary["bootstrap_ci_95"]
    block = f"""

### QA {summary["model"]} {summary["doc_count"]}-doc gold@{summary["gold_position"]}

- Output: `{out_dir}`
- Accuracy: {summary["accuracy"]} CI95={ci}
- Scored/skipped: {summary["n"]}/{summary["n_skipped"]}
- dtype: {summary["dtype"]}; max_new_tokens: {summary["max_new_tokens"]}
- CUDA peak memory bytes: {summary["cuda_peak_memory_bytes"]}
- Wall time sec: {summary["wall_time_sec"]}
- Verdict: {summary.get("verdict")}
- Updated UTC: {utc_now_iso()}
"""
    append_results_block(project_root, block)


def main() -> int:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    out_dir = args.out or default_out_dir(args)
    out_dir.mkdir(parents=True, exist_ok=True)
    target_n = target_n_examples(args.n_examples, args.smoke, smoke_n=5)

    if not args.data_file.exists():
        raise FileNotFoundError(f"Data file not found: {args.data_file}")

    set_seed(args.seed)
    predictions_path = out_dir / "predictions.jsonl"
    skipped_path = out_dir / "skipped.jsonl"
    records = load_jsonl_records(predictions_path) if args.resume else []
    skipped = load_jsonl_records(skipped_path) if args.resume else []
    completed_or_skipped = {int(record["example_idx"]) for record in records}
    completed_or_skipped.update(int(record["example_idx"]) for record in skipped)

    started = time.perf_counter()
    print(f"Loading tokenizer and generation model: {args.model} ({args.dtype})")
    tokenizer = load_tokenizer(args.model)
    model = load_generation_model(args.model, args.dtype, args.seed)
    context_limit = model_context_limit(tokenizer, model)

    metadata = {
        "task": "qa_eval",
        "model": args.model,
        "data_file": str(args.data_file.resolve()),
        "out_dir": str(out_dir.resolve()),
        "target_n_examples": target_n,
        "seed": args.seed,
        "dtype": args.dtype,
        "max_new_tokens": args.max_new_tokens,
        "started_utc": utc_now_iso(),
        "dependency_versions": dependency_versions(),
        "context_limit": context_limit,
    }
    atomic_write_json(out_dir / "metadata.json", metadata)

    progress = tqdm(total=target_n, initial=len(completed_or_skipped), desc="qa examples")
    try:
        for example_idx, example in enumerate(iter_nq_examples(args.data_file)):
            if example_idx >= target_n:
                break
            if example_idx in completed_or_skipped:
                continue

            prompt = build_nq_prompt(str(example["question"]), example["ctxs"])
            token_len = prompt_token_len(tokenizer, prompt)
            if context_limit is not None and token_len > context_limit:
                skipped.append(
                    {
                        "example_idx": example_idx,
                        "reason": "prompt_exceeds_context",
                        "prompt_token_len": token_len,
                        "context_limit": context_limit,
                    }
                )
                write_jsonl_atomic(skipped_path, skipped)
                completed_or_skipped.add(example_idx)
                progress.update(1)
                continue

            prediction, prompt_len_from_generate = generate_answer(
                model,
                tokenizer,
                prompt,
                max_new_tokens=args.max_new_tokens,
            )
            score = score_prediction(prediction, example["answers"])
            record = {
                "example_idx": int(example_idx),
                "question": str(example["question"]),
                "gold_position": gold_position(example),
                "prompt_token_len": int(prompt_len_from_generate),
                "prediction": prediction,
                "score": float(score),
            }
            records.append(record)
            write_jsonl_atomic(predictions_path, records)
            completed_or_skipped.add(example_idx)
            progress.update(1)
            progress.set_postfix({"score": score, "tokens": prompt_len_from_generate})
    finally:
        progress.close()

    wall_time_sec = time.perf_counter() - started
    summary = summarize_qa_run(
        records=records,
        skipped=skipped,
        model_name=args.model,
        data_file=args.data_file,
        out_dir=out_dir,
        seed=args.seed,
        dtype=args.dtype,
        max_new_tokens=args.max_new_tokens,
        requested_n=target_n,
        wall_time_sec=wall_time_sec,
        cuda_peak_memory_bytes=cuda_peak_memory_bytes(),
    )
    summary["completed_source_examples"] = len(completed_or_skipped)
    summary["context_limit"] = context_limit
    atomic_write_json(out_dir / "summary.json", summary)
    update_weak_model_verdicts(out_dir / "summary.json")
    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    append_results(project_root, summary, out_dir)

    print(f"Completed QA run: {summary['n']} scored, {summary['n_skipped']} skipped")
    print(f"Accuracy: {summary['accuracy']} CI95={summary['bootstrap_ci_95']}")
    print(f"Output: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
