#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning, module="pandas")


def flatten(prefix: str, obj: Any, out: dict[str, Any]) -> None:
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = f"{prefix}.{k}" if prefix else k
            flatten(key, v, out)
    else:
        out[prefix] = obj


def load_runs(results_root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in sorted((results_root / "runs").glob("*/metrics.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        row: dict[str, Any] = {"metrics_path": str(path)}
        flatten("", data, row)
        rows.append(row)
    return pd.DataFrame(rows)


def concat_csv(results_root: Path, pattern: str, out_path: Path, key_name: str) -> None:
    chunks = []
    for path in sorted((results_root / "runs").glob(pattern)):
        run_id = path.parent.name
        df = pd.read_csv(path)
        df.insert(0, key_name, run_id)
        chunks.append(df)
    if chunks:
        pd.concat(chunks, ignore_index=True).to_csv(out_path, index=False)


def aggregate(df: pd.DataFrame, x: str, y: str) -> pd.DataFrame:
    tmp = (
        df.groupby(["network", "family", "dim", x], dropna=False)[y]
        .agg(["mean", "std", "count"])
        .reset_index()
        .sort_values(["family", "dim", x])
    )
    return tmp


def plot_grouped(df: pd.DataFrame, x: str, y: str, title: str, out_path: Path, network: str) -> None:
    sub = df[df["network"] == network].copy()
    if sub.empty or x not in sub.columns or y not in sub.columns:
        return
    agg = aggregate(sub, x, y)
    if agg.empty:
        return
    plt.figure(figsize=(8, 5))
    for (family, dim), grp in agg.groupby(["family", "dim"], dropna=False):
        plt.plot(grp[x], grp["mean"], marker="o", label=f"{family}-d{dim}")
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(title)
    plt.legend(fontsize=8)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=160)
    plt.close()


def write_report(analysis_dir: Path, df: pd.DataFrame) -> None:
    report = analysis_dir / "report.md"
    pngs = sorted(p.name for p in analysis_dir.glob("*.png"))
    lines = [
        "# KDTree Baseline 实验汇总",
        "",
        f"完成 run 数：{len(df)}",
        "",
        "## 关键导出",
        "",
        "- `runs_flat.csv`：每个 run 的扁平化指标表",
        "- `bbox_depth_all.csv`：按深度汇总的 bbox 统计",
        "- `group_size_hist_all.csv`：每个 run 的组大小直方图",
        "",
        "## 图表",
        "",
    ]
    for name in pngs:
        lines += [f"### {name}", "", f"![]({name})", ""]
    report.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", required=True)
    args = ap.parse_args()

    results_root = Path(args.results_root)
    analysis_dir = results_root / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    df = load_runs(results_root)
    if df.empty:
        (analysis_dir / "report.md").write_text("# No runs found\n", encoding="utf-8")
        return 0

    df.to_csv(analysis_dir / "runs_flat.csv", index=False)
    concat_csv(results_root, "*/bbox_by_depth.csv", analysis_dir / "bbox_depth_all.csv", "run_id")
    concat_csv(results_root, "*/group_size_hist.csv", analysis_dir / "group_size_hist_all.csv", "run_id")

    # Density network: alpha_actual as x-axis.
    plot_grouped(df, "count.alpha_actual", "timing_ms.T_count", "Density sweep: T_count vs alpha_actual", analysis_dir / "density_T_count.png", "density")
    plot_grouped(df, "count.alpha_actual", "count.P_count.mean", "Density sweep: mean P_count vs alpha_actual", analysis_dir / "density_P_count_mean.png", "density")
    plot_grouped(df, "count.alpha_actual", "search_weighted.rho.mean", "Density sweep: weighted rho vs alpha_actual", analysis_dir / "density_rho_weighted.png", "density")
    plot_grouped(df, "count.alpha_actual", "search_weighted.sigma.mean", "Density sweep: weighted sigma vs alpha_actual", analysis_dir / "density_sigma_weighted.png", "density")

    # Input-size network.
    plot_grouped(df, "N", "timing_ms.T_build", "Input-size sweep: T_build vs N", analysis_dir / "input_size_T_build.png", "input_size")
    plot_grouped(df, "N", "build.tree_height", "Input-size sweep: tree height vs N", analysis_dir / "input_size_tree_height.png", "input_size")
    plot_grouped(df, "N", "timing_ms.T_count", "Input-size sweep: T_count vs N", analysis_dir / "input_size_T_count.png", "input_size")

    # Sample-size network.
    plot_grouped(df, "sample.t", "sample.U_t", "Sample-size sweep: U_t vs t", analysis_dir / "sample_size_U_t.png", "sample_size")
    plot_grouped(df, "sample.t", "sample.G_t", "Sample-size sweep: G_t vs t", analysis_dir / "sample_size_G_t.png", "sample_size")
    plot_grouped(df, "sample.t", "sample.avg_group_size_exec", "Sample-size sweep: avg exec group size vs t", analysis_dir / "sample_size_avg_group_exec.png", "sample_size")
    plot_grouped(df, "sample.t", "sample.samples_per_sec", "Sample-size sweep: throughput vs t", analysis_dir / "sample_size_throughput.png", "sample_size")

    # Chunk network.
    plot_grouped(df, "sample.t", "sample.U_t", "Chunk sweep: U_t vs t", analysis_dir / "chunk_U_t.png", "chunk")
    plot_grouped(df, "sample.t", "sample.G_t", "Chunk sweep: G_t vs t", analysis_dir / "chunk_G_t.png", "chunk")
    plot_grouped(df, "sample.t", "sample.avg_group_size_exec", "Chunk sweep: avg exec group size vs t", analysis_dir / "chunk_avg_group_exec.png", "chunk")

    # Correctness summary table.
    correctness_cols = [
        "run_id",
        "family",
        "dim",
        "N",
        "alpha_target",
        "checks.count_sum_matches",
        "checks.decompose_mass_matches",
        "checks.sample_valid",
        "checks.empty_output_consistent",
        "checks.materialized_join_matches",
        "checks.materialized_join_size",
        "checks.pair_tv_to_uniform",
        "checks.left_tv_to_theory",
    ]
    available = [c for c in correctness_cols if c in df.columns]
    df[df["network"] == "correctness"][available].to_csv(analysis_dir / "correctness_summary.csv", index=False)

    write_report(analysis_dir, df)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
