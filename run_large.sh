#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"

exec python3 - "$ROOT" "$@" <<'PY'
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(sys.argv[1]).resolve()
CLI_ARGS = sys.argv[2:]

BENCH = REPO_ROOT / "build" / "kdtree_benchmark"
SMOKE = REPO_ROOT / "build" / "isrjs_smoke"
VERIFY_VENDOR = REPO_ROOT / "scripts" / "verify_vendor_integrity.py"
DATASET_GEN = REPO_ROOT / "python" / "generate_dataset.py"
ANALYZE = REPO_ROOT / "python" / "analyze_results.py"

PRESET = "TH"
DEFAULT_DIMS = [2, 3, 4, 5]
DEFAULT_NETWORKS = ["density", "input_size", "sample_size"]
FAMILIES = ["F0", "F1"]
DENSITY_ALPHAS = [0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0]
INPUT_NS = [100_000, 200_000, 500_000, 1_000_000, 2_000_000, 5_000_000]
SAMPLE_TS = [1_000_000, 3_000_000, 10_000_000, 30_000_000, 100_000_000, 300_000_000, 1_000_000_000]


@dataclass(frozen=True)
class DatasetKey:
    dim: int
    N: int
    alpha_target: float
    family: str
    rep: int


@dataclass(frozen=True)
class RunSpec:
    network: str
    dim: int
    N: int
    alpha_target: float
    t: int
    family: str
    rep: int = 0
    materialize_join: bool = False

    @property
    def dataset_key(self) -> DatasetKey:
        return DatasetKey(self.dim, self.N, self.alpha_target, self.family, self.rep)

    @property
    def sweep_axis(self) -> str:
        if self.network == "density":
            return "alpha_target"
        if self.network == "input_size":
            return "N"
        if self.network == "sample_size":
            return "t"
        return "unknown"

    @property
    def sweep_value(self) -> str:
        if self.network == "density":
            return f"{self.alpha_target:.12g}"
        if self.network == "input_size":
            return str(self.N)
        if self.network == "sample_size":
            return str(self.t)
        return ""


def parse_csv_list(raw: str) -> list[str]:
    return [part.strip() for part in raw.split(",") if part.strip()]


def parse_dims(raw: str) -> list[int]:
    dims = []
    for token in parse_csv_list(raw):
        try:
            dim = int(token)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"invalid dim: {token}") from exc
        if dim not in {2, 3, 4, 5}:
            raise argparse.ArgumentTypeError(f"unsupported dim: {dim}")
        dims.append(dim)
    if not dims:
        raise argparse.ArgumentTypeError("dims cannot be empty")
    return sorted(dict.fromkeys(dims))


def parse_networks(raw: str) -> list[str]:
    allowed = {"density", "input_size", "sample_size"}
    networks = []
    for token in parse_csv_list(raw):
        if token not in allowed:
            raise argparse.ArgumentTypeError(f"unsupported network: {token}")
        networks.append(token)
    if not networks:
        raise argparse.ArgumentTypeError("networks cannot be empty")
    return list(dict.fromkeys(networks))


def stable_seed(*parts: object) -> int:
    payload = "|".join(str(p) for p in parts).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "little") & ((1 << 63) - 1)


def alpha_token(alpha: float) -> str:
    return f"{alpha:.12g}".replace(".", "p")


def dataset_seed(key: DatasetKey) -> int:
    return stable_seed("data", PRESET, key.dim, key.N, alpha_token(key.alpha_target), key.family, key.rep)


def sample_seed(spec: RunSpec) -> int:
    return stable_seed(
        "sample",
        PRESET,
        spec.network,
        spec.dim,
        spec.N,
        alpha_token(spec.alpha_target),
        spec.t,
        spec.family,
        spec.rep,
    )


def run_id(spec: RunSpec) -> str:
    return (
        f"{spec.network}_d{spec.dim}_N{spec.N}_a{alpha_token(spec.alpha_target)}"
        f"_t{spec.t}_{spec.family}_rep{spec.rep}"
    )


def build_plan(dims: list[int], networks: list[str]) -> list[RunSpec]:
    specs: list[RunSpec] = []
    for dim in dims:
        for family in FAMILIES:
            if "density" in networks:
                for alpha in DENSITY_ALPHAS:
                    specs.append(RunSpec("density", dim, 1_000_000, alpha, 10_000_000, family))
            if "input_size" in networks:
                for N in INPUT_NS:
                    specs.append(RunSpec("input_size", dim, N, 100.0, 10_000_000, family))
            if "sample_size" in networks:
                for t in SAMPLE_TS:
                    specs.append(RunSpec("sample_size", dim, 1_000_000, 100.0, t, family))
    return specs


def resolve_results_root(raw: str) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def ensure_dataset(key: DatasetKey, datasets_dir: Path, force: bool) -> tuple[Path, dict]:
    seed = dataset_seed(key)
    base = f"dataset_d{key.dim}_N{key.N}_a{alpha_token(key.alpha_target)}_{key.family}_rep{key.rep}_seed{seed}"
    out_bin = datasets_dir / f"{base}.bin"
    out_meta = datasets_dir / f"{base}.json"
    cmd = [
        sys.executable,
        str(DATASET_GEN),
        "--out-bin",
        str(out_bin),
        "--out-meta",
        str(out_meta),
        "--dim",
        str(key.dim),
        "--N",
        str(key.N),
        "--alpha-target",
        str(key.alpha_target),
        "--family",
        key.family,
        "--seed",
        str(seed),
    ]
    if force:
        cmd.append("--force")
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)
    meta = json.loads(out_meta.read_text(encoding="utf-8"))
    return out_bin, meta


def write_manifest(specs: Iterable[RunSpec], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "run_id",
            "network",
            "sweep_axis",
            "sweep_value",
            "dim",
            "N",
            "alpha_target",
            "t",
            "family",
            "rep",
            "dataset_seed",
            "sample_seed",
            "materialize_join",
        ])
        for spec in specs:
            writer.writerow([
                run_id(spec),
                spec.network,
                spec.sweep_axis,
                spec.sweep_value,
                spec.dim,
                spec.N,
                spec.alpha_target,
                spec.t,
                spec.family,
                spec.rep,
                dataset_seed(spec.dataset_key),
                sample_seed(spec),
                int(spec.materialize_join),
            ])


def run_checked(cmd: list[str], *, cwd: Path) -> None:
    subprocess.run(cmd, check=True, cwd=cwd)


def run_smoke(smoke_out: Path) -> None:
    completed = subprocess.run([str(SMOKE)], check=True, cwd=REPO_ROOT, capture_output=True, text=True)
    smoke_out.parent.mkdir(parents=True, exist_ok=True)
    smoke_out.write_text(completed.stdout, encoding="utf-8")
    if completed.stdout:
        print(completed.stdout, end="")


def emit_extra_sample_reports(results_root: Path) -> None:
    try:
        import matplotlib.pyplot as plt
        import pandas as pd
    except Exception as exc:  # pragma: no cover
        print(f"[warn] skip extra sample reports: {exc}")
        return

    runs_flat = results_root / "analysis" / "runs_flat.csv"
    if not runs_flat.exists():
        return

    df = pd.read_csv(runs_flat)
    required = {"network", "dim", "family", "sample.t", "sample.samples_per_sec", "sample.ns_per_sample", "timing_ms.T_sample"}
    if not required.issubset(df.columns):
        return

    sub = df[df["network"] == "sample_size"].copy()
    if sub.empty:
        return

    summary = (
        sub[["dim", "family", "sample.t", "sample.samples_per_sec", "sample.ns_per_sample", "timing_ms.T_sample"]]
        .sort_values(["dim", "family", "sample.t"])
        .reset_index(drop=True)
    )
    summary.to_csv(results_root / "analysis" / "sample_size_perf_summary.csv", index=False)

    plt.figure(figsize=(8, 5))
    for (family, dim), grp in summary.groupby(["family", "dim"], dropna=False):
        plt.plot(grp["sample.t"], grp["sample.ns_per_sample"], marker="o", label=f"{family}-d{dim}")
    plt.xlabel("sample.t")
    plt.ylabel("sample.ns_per_sample")
    plt.title("Sample-size sweep: ns/sample vs t")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(results_root / "analysis" / "sample_size_ns_per_sample.png", dpi=160)
    plt.close()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the large experiment network from 参数网络.md over dims 2/3/4/5."
    )
    parser.add_argument("--dims", default=",".join(str(d) for d in DEFAULT_DIMS), help="Comma-separated dimensions to run, e.g. 2,3 or 4,5")
    parser.add_argument(
        "--networks",
        default=",".join(DEFAULT_NETWORKS),
        help="Comma-separated networks: density,input_size,sample_size",
    )
    parser.add_argument("--results-root", default="results/large", help="Results root, relative to repo unless absolute")
    parser.add_argument("--jobs", type=int, default=int(os.environ.get("JOBS", "2")))
    parser.add_argument("--skip-build", action="store_true", help="Skip vendor verification and make")
    parser.add_argument("--skip-smoke", action="store_true", help="Skip isrjs_smoke sanity run")
    parser.add_argument("--skip-analyze", action="store_true", help="Skip post-run CSV aggregation and plotting")
    parser.add_argument("--plan-only", action="store_true", help="Only write plan_manifest.csv and exit")
    parser.add_argument("--force-datasets", action="store_true", help="Force regenerate datasets even if cached")
    parser.add_argument("--force-runs", action="store_true", help="Re-run benchmarks even if metrics.json exists")
    args = parser.parse_args(CLI_ARGS)

    try:
        args.dims = parse_dims(args.dims)
        args.networks = parse_networks(args.networks)
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))
    if args.jobs <= 0:
        args.jobs = 2

    results_root = resolve_results_root(args.results_root)
    specs = build_plan(args.dims, args.networks)
    write_manifest(specs, results_root / "plan_manifest.csv")
    print(f"plan: {len(specs)} runs | dims={args.dims} | networks={args.networks} | results={results_root}")

    if args.plan_only:
        print(f"plan-only: wrote {results_root / 'plan_manifest.csv'}")
        return 0

    if not args.skip_build:
        run_checked([sys.executable, str(VERIFY_VENDOR)], cwd=REPO_ROOT)
        run_checked(["make", f"-j{args.jobs}"], cwd=REPO_ROOT)

    if not BENCH.exists():
        print(f"missing benchmark binary: {BENCH}", file=sys.stderr)
        return 1
    if not args.skip_smoke:
        if not SMOKE.exists():
            print(f"missing smoke binary: {SMOKE}", file=sys.stderr)
            return 1
        run_smoke(results_root / "isrjs_smoke.txt")

    datasets_dir = results_root / "datasets"
    runs_dir = results_root / "runs"
    analysis_dir = results_root / "analysis"
    datasets_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir.mkdir(parents=True, exist_ok=True)

    dataset_cache: dict[DatasetKey, tuple[Path, dict]] = {}
    total = len(specs)
    for idx, spec in enumerate(specs, start=1):
        rid = run_id(spec)
        out_dir = runs_dir / rid
        metrics_json = out_dir / "metrics.json"
        if metrics_json.exists() and not args.force_runs:
            print(f"[{idx}/{total}] skip {rid}")
            continue

        if spec.dataset_key not in dataset_cache:
            dataset_cache[spec.dataset_key] = ensure_dataset(spec.dataset_key, datasets_dir, args.force_datasets)
        dataset_path, dataset_meta = dataset_cache[spec.dataset_key]
        out_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            str(BENCH),
            "--dataset",
            str(dataset_path),
            "--out-dir",
            str(out_dir),
            "--run-id",
            rid,
            "--network",
            spec.network,
            "--preset",
            PRESET,
            "--family",
            spec.family,
            "--dim",
            str(spec.dim),
            "--N",
            str(spec.N),
            "--alpha-target",
            str(spec.alpha_target),
            "--alpha-expected",
            str(dataset_meta["alpha_expected_est"]),
            "--pair-prob",
            str(dataset_meta["pair_intersection_prob_est"]),
            "--coverage",
            str(dataset_meta["coverage"]),
            "--data-seed",
            str(dataset_seed(spec.dataset_key)),
            "--sample-seed",
            str(sample_seed(spec)),
            "--t",
            str(spec.t),
            "--data-rep",
            str(spec.rep),
            "--materialize-join",
            "1" if spec.materialize_join else "0",
        ]
        print(f"[{idx}/{total}] run {rid}")
        run_checked(cmd, cwd=REPO_ROOT)

    if not args.skip_analyze:
        run_checked([sys.executable, str(ANALYZE), "--results-root", str(results_root)], cwd=REPO_ROOT)
        emit_extra_sample_reports(results_root)
        print(f"analysis: {analysis_dir}")

    print(f"done: {results_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
PY
