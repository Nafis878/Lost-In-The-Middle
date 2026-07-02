"""Correlation analysis for QA accuracy and gold-span Jacobian summaries."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from jacobian import atomic_write_json, data_file_info, load_jsonl_records, safe_model_name
from qa_eval import DOC_POSITIONS


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def rankdata(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    order = np.argsort(arr, kind="mergesort")
    ranks = np.empty(arr.size, dtype=np.float64)
    i = 0
    while i < arr.size:
        j = i + 1
        while j < arr.size and arr[order[j]] == arr[order[i]]:
            j += 1
        rank = 0.5 * (i + j - 1) + 1.0
        ranks[order[i:j]] = rank
        i = j
    return ranks


def pearsonr(x: Sequence[float], y: Sequence[float]) -> float | None:
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    if x_arr.size < 2 or y_arr.size < 2:
        return None
    x_centered = x_arr - x_arr.mean()
    y_centered = y_arr - y_arr.mean()
    denom = np.sqrt(np.sum(x_centered**2) * np.sum(y_centered**2))
    if denom == 0:
        return None
    return float(np.sum(x_centered * y_centered) / denom)


def spearmanr(x: Sequence[float], y: Sequence[float]) -> dict[str, float | int | None]:
    coef = pearsonr(rankdata(x), rankdata(y))
    pvalue = None
    try:
        from scipy import stats
    except ImportError:
        pass
    else:
        result = stats.spearmanr(x, y)
        pvalue = float(result.pvalue) if result.pvalue == result.pvalue else None
    return {"rho": coef, "pvalue": pvalue, "n": len(x)}


def roc_auc_score(y_true: Sequence[float], scores: Sequence[float]) -> float | None:
    y = np.asarray(y_true, dtype=np.float64)
    score_arr = np.asarray(scores, dtype=np.float64)
    positives = y == 1.0
    negatives = y == 0.0
    n_pos = int(np.sum(positives))
    n_neg = int(np.sum(negatives))
    if n_pos == 0 or n_neg == 0:
        return None
    ranks = rankdata(score_arr)
    sum_pos = float(np.sum(ranks[positives]))
    return (sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def sigmoid(values: np.ndarray) -> np.ndarray:
    out = np.empty_like(values, dtype=np.float64)
    positive = values >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exp_values = np.exp(values[~positive])
    out[~positive] = exp_values / (1.0 + exp_values)
    return out


def fit_logistic_regression(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {"status": "no_records"}
    y = np.asarray([float(record["score"]) for record in records], dtype=np.float64)
    jac = np.asarray([float(record["jac_gold_logmean"]) for record in records], dtype=np.float64)
    length = np.asarray([float(record["prompt_token_len"]) for record in records], dtype=np.float64)

    if np.unique(y).size < 2:
        return {
            "status": "single_class",
            "n": int(y.size),
            "coef_jac_gold_logmean": None,
            "coef_jac_gold_logmean_ci95": [None, None],
            "auc": None,
        }

    length_sd = float(np.std(length))
    length_z = np.zeros_like(length) if length_sd == 0 else (length - float(np.mean(length))) / length_sd
    x = np.column_stack([np.ones_like(jac), jac, length_z])
    beta = np.zeros(x.shape[1], dtype=np.float64)

    for _ in range(100):
        p = sigmoid(x @ beta)
        weights = np.clip(p * (1.0 - p), 1e-9, None)
        grad = x.T @ (y - p)
        hessian_pos = (x.T * weights) @ x
        try:
            step = np.linalg.solve(hessian_pos, grad)
        except np.linalg.LinAlgError:
            step = np.linalg.pinv(hessian_pos) @ grad
        beta_next = beta + step
        if np.max(np.abs(step)) < 1e-7:
            beta = beta_next
            break
        beta = beta_next

    p = sigmoid(x @ beta)
    weights = np.clip(p * (1.0 - p), 1e-9, None)
    hessian_pos = (x.T * weights) @ x
    try:
        cov = np.linalg.inv(hessian_pos)
    except np.linalg.LinAlgError:
        cov = np.linalg.pinv(hessian_pos)
    se = float(np.sqrt(max(cov[1, 1], 0.0)))
    coef = float(beta[1])
    auc = roc_auc_score(y, p)
    return {
        "status": "ok",
        "n": int(y.size),
        "coef_intercept": float(beta[0]),
        "coef_jac_gold_logmean": coef,
        "coef_jac_gold_logmean_ci95": [coef - 1.96 * se, coef + 1.96 * se],
        "coef_prompt_token_len_z": float(beta[2]),
        "auc": float(auc) if auc is not None else None,
    }


def iqr(values: Sequence[float]) -> list[float | None]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return [None, None]
    return [float(x) for x in np.quantile(arr, [0.25, 0.75])]


def position_paths(root: Path, model: str, doc_count: int, position: int, task: str, init: str = "pretrained") -> Path:
    model_dir = safe_model_name(model)
    if task == "qa":
        return root / model_dir / f"{doc_count}_docs" / f"gold_at_{position}"
    if task == "jacobian_qa":
        return root / model_dir / init / f"{doc_count}_docs" / f"gold_at_{position}"
    raise ValueError(f"Unsupported task={task!r}")


def load_position_bundle(
    qa_root: Path,
    jacobian_root: Path,
    model: str,
    doc_count: int,
    init: str = "pretrained",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    positions = DOC_POSITIONS[doc_count]
    position_table: list[dict[str, Any]] = []
    pooled_records: list[dict[str, Any]] = []

    for position in positions:
        qa_dir = position_paths(qa_root, model, doc_count, position, "qa")
        jac_dir = position_paths(jacobian_root, model, doc_count, position, "jacobian_qa", init=init)
        qa_summary = read_json(qa_dir / "summary.json")
        qa_predictions = load_jsonl_records(qa_dir / "predictions.jsonl")
        jac_records = load_jsonl_records(jac_dir / "records.jsonl")
        jac_by_idx = {int(record["example_idx"]): record for record in jac_records}
        for pred in qa_predictions:
            idx = int(pred["example_idx"])
            if idx not in jac_by_idx:
                continue
            pooled_records.append({**pred, **jac_by_idx[idx], "position": position})

        logmeans = [float(record["jac_gold_logmean"]) for record in jac_records]
        position_table.append(
            {
                "position": position,
                "accuracy": qa_summary.get("accuracy"),
                "accuracy_ci95": qa_summary.get("bootstrap_ci_95"),
                "qa_n": qa_summary.get("n"),
                "jacobian_n": len(jac_records),
                "jac_gold_logmean_median": float(np.median(logmeans)) if logmeans else None,
                "jac_gold_logmean_iqr": iqr(logmeans),
                "qa_summary": str(qa_dir / "summary.json"),
                "jacobian_records": str(jac_dir / "records.jsonl"),
            }
        )
    return position_table, pooled_records


def analyze_correlation(
    qa_root: Path,
    jacobian_root: Path,
    model: str,
    doc_count: int,
    out_dir: Path,
    init: str = "pretrained",
    random_jacobian_dir: Path | None = None,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    position_table, pooled_records = load_position_bundle(qa_root, jacobian_root, model, doc_count, init=init)
    pairs = [
        (row["jac_gold_logmean_median"], row["accuracy"])
        for row in position_table
        if row["jac_gold_logmean_median"] is not None and row["accuracy"] is not None
    ]
    xs = [float(pair[0]) for pair in pairs]
    ys = [float(pair[1]) for pair in pairs]
    spearman = spearmanr(xs, ys) if xs else {"rho": None, "pvalue": None, "n": 0}
    logistic = fit_logistic_regression(pooled_records)

    figures = make_figures(
        position_table=position_table,
        pooled_records=pooled_records,
        out_dir=out_dir,
        model=model,
        doc_count=doc_count,
        random_jacobian_dir=random_jacobian_dir,
        pretrained_jacobian_root=jacobian_root,
    )
    report = {
        "model": model,
        "doc_count": doc_count,
        "init": init,
        "position_table": position_table,
        "spearman_position_level": spearman,
        "logistic_example_level": logistic,
        "figures": figures,
    }
    atomic_write_json(out_dir / "correlation_report.json", report)
    return report


def make_figures(
    position_table: Sequence[dict[str, Any]],
    pooled_records: Sequence[dict[str, Any]],
    out_dir: Path,
    model: str,
    doc_count: int,
    random_jacobian_dir: Path | None,
    pretrained_jacobian_root: Path,
) -> dict[str, str | None]:
    import matplotlib.pyplot as plt

    figures: dict[str, str | None] = {}
    positions = [int(row["position"]) for row in position_table]
    accuracies = [row["accuracy"] for row in position_table]
    jac_medians = [row["jac_gold_logmean_median"] for row in position_table]

    overlay_path = out_dir / "accuracy_jacobian_overlay.png"
    fig, ax1 = plt.subplots(figsize=(7.5, 4.5))
    ax1.errorbar(positions, accuracies, marker="o", color="#1f77b4", label="accuracy")
    ax1.set_xlabel("Gold document position")
    ax1.set_ylabel("Accuracy", color="#1f77b4")
    ax2 = ax1.twinx()
    ax2.plot(positions, jac_medians, marker="s", color="#d62728", label="Jacobian logmean")
    ax2.set_ylabel("Median gold-span log10 Jacobian", color="#d62728")
    ax1.set_title(f"{model} {doc_count}-doc: accuracy vs gold-span Jacobian")
    fig.tight_layout()
    fig.savefig(overlay_path, dpi=180)
    plt.close(fig)
    figures["overlay"] = str(overlay_path)

    scatter_path = out_dir / "jacobian_score_scatter.png"
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    if pooled_records:
        rng = np.random.default_rng(0)
        x = np.asarray([float(record["jac_gold_logmean"]) for record in pooled_records])
        y = np.asarray([float(record["score"]) for record in pooled_records])
        colors = np.asarray([float(record["position"]) for record in pooled_records])
        jitter = rng.normal(0.0, 0.035, size=y.shape)
        points = ax.scatter(x, y + jitter, c=colors, cmap="viridis", alpha=0.75, s=24)
        fig.colorbar(points, ax=ax, label="Gold position")
    ax.set_xlabel("Gold-span log10 Jacobian mean")
    ax.set_ylabel("QA score with jitter")
    ax.set_title("Example-level Jacobian vs score")
    fig.tight_layout()
    fig.savefig(scatter_path, dpi=180)
    plt.close(fig)
    figures["scatter"] = str(scatter_path)

    figures["init_vs_pretrained"] = make_init_comparison_figure(
        out_dir=out_dir,
        random_jacobian_dir=random_jacobian_dir,
        pretrained_jacobian_root=pretrained_jacobian_root,
        model=model,
        doc_count=doc_count,
    )
    return figures


def make_init_comparison_figure(
    out_dir: Path,
    random_jacobian_dir: Path | None,
    pretrained_jacobian_root: Path,
    model: str,
    doc_count: int,
) -> str | None:
    if random_jacobian_dir is None or not random_jacobian_dir.exists():
        return None
    pretrained_candidates = list((pretrained_jacobian_root / safe_model_name(model) / "pretrained").glob(f"{doc_count}_docs/gold_at_*/full_curves/*.npy"))
    random_candidates = list(random_jacobian_dir.glob("full_curves/*.npy"))
    if not pretrained_candidates or not random_candidates:
        return None

    import matplotlib.pyplot as plt

    def median_curve(paths: Sequence[Path]) -> tuple[np.ndarray, np.ndarray]:
        arrays = [np.load(path).astype(np.float64) for path in paths]
        min_len = min(array.shape[0] for array in arrays)
        stacked = np.stack([array[:min_len] for array in arrays], axis=0)
        log_curves = np.log10(np.clip(stacked, 1e-300, None))
        median = np.median(log_curves, axis=0)
        x = np.arange(min_len, dtype=np.float64) / float(min_len)
        return x, median - float(np.max(median))

    x_pre, y_pre = median_curve(pretrained_candidates)
    x_rand, y_rand = median_curve(random_candidates)
    path = out_dir / "init_vs_pretrained_curves.png"
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(x_pre, y_pre, label="pretrained", color="#1f77b4")
    ax.plot(x_rand, y_rand, label="random init", color="#d62728")
    ax.set_xlabel("Normalized token position")
    ax.set_ylabel("Median log10 rho, max-normalized")
    ax.set_title("Stored full-curve comparison")
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return str(path)
