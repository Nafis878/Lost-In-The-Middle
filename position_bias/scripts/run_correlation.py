#!/usr/bin/env python3
"""Run Phase 2 QA/Jacobian correlation analysis."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from correlate import analyze_correlation  # noqa: E402
from jacobian import append_results_block, safe_model_name, utc_now_iso  # noqa: E402


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--doc-count", type=int, choices=[10, 20], required=True)
    parser.add_argument("--qa-root", type=Path, default=project_root / "results" / "qa")
    parser.add_argument("--jacobian-root", type=Path, default=project_root / "results" / "jacobian_qa")
    parser.add_argument("--init", choices=["pretrained", "random"], default="pretrained")
    parser.add_argument("--random-jacobian-dir", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args()


def default_out_dir(args: argparse.Namespace) -> Path:
    project_root = Path(__file__).resolve().parents[1]
    return project_root / "results" / "correlation" / safe_model_name(args.model) / f"{args.doc_count}_docs"


def append_results(project_root: Path, report: dict, out_dir: Path) -> None:
    logistic = report["logistic_example_level"]
    spearman = report["spearman_position_level"]
    block = f"""

### Correlation {report["model"]} {report["doc_count"]}-doc

- Output: `{out_dir}`
- Headline logistic jac coefficient: {logistic.get("coef_jac_gold_logmean")} CI95={logistic.get("coef_jac_gold_logmean_ci95")} AUC={logistic.get("auc")} status={logistic.get("status")}
- Position Spearman: rho={spearman.get("rho")} p={spearman.get("pvalue")} n={spearman.get("n")}
- Figures: {report.get("figures")}
- Updated UTC: {utc_now_iso()}
"""
    append_results_block(project_root, block)


def main() -> int:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    out_dir = args.out or default_out_dir(args)
    report = analyze_correlation(
        qa_root=args.qa_root,
        jacobian_root=args.jacobian_root,
        model=args.model,
        doc_count=args.doc_count,
        out_dir=out_dir,
        init=args.init,
        random_jacobian_dir=args.random_jacobian_dir,
    )
    append_results(project_root, report, out_dir)
    print(f"Wrote correlation report: {out_dir / 'correlation_report.json'}")
    print(f"Logistic: {report['logistic_example_level']}")
    print(f"Spearman: {report['spearman_position_level']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
