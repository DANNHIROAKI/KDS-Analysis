#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
BENCH = REPO_ROOT / "build" / "kdtree_benchmark"
DATASET_GEN = REPO_ROOT / "python" / "generate_dataset.py"
ANALYZE = REPO_ROOT / "python" / "analyze_results.py"

PRESET = "TH"


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
    rep: int
    materialize_join: bool

    @property
    def dataset_key(self) -> DatasetKey:
        return DatasetKey(self.dim, self.N, self.alpha_target, self.family, self.rep)


def stable_seed(*parts: object) -> int:
    payload = "|".join(str(p) for p in parts).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "little") & ((1 << 63) - 1)


def alpha_token(alpha: float) -> str:
    text = f"{alpha:.12g}"
    return text.replace(".", "p")


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


def full_plan() -> list[RunSpec]:
    specs: list[RunSpec] = []
    for dim in [2, 3, 4, 5]:
        for family in ["F0", "F1"]:
            for alpha in [0.1, 10.0, 100.0]:
                specs.append(RunSpec("correctness", dim, 2000, alpha, 200_000, family, 0, True))
            for rep in range(3):
                for alpha in [0.1, 1.0, 10.0, 100.0]:
                    specs.append(RunSpec("density", dim, 50_000, alpha, 200_000, family, rep, False))
                for N in [10_000, 50_000, 100_000]:
                    specs.append(RunSpec("input_size", dim, N, 10.0, 200_000, family, rep, False))
                for t in [10_000, 50_000, 200_000, 1_000_000]:
                    specs.append(RunSpec("sample_size", dim, 50_000, 10.0, t, family, rep, False))
    for dim in [2, 5]:
        for family in ["F0", "F1"]:
            for rep in range(2):
                for t in [500_000, 2_000_000, 3_000_000]:
                    specs.append(RunSpec("chunk", dim, 20_000, 10.0, t, family, rep, False))
    return specs


def smoke_plan() -> list[RunSpec]:
    return [
        RunSpec("correctness", 2, 500, 1.0, 20_000, "F0", 0, True),
        RunSpec("density", 3, 2_000, 10.0, 20_000, "F1", 0, False),
        RunSpec("chunk", 2, 5_000, 10.0, 300_000, "F0", 0, False),
    ]


def ensure_dataset(key: DatasetKey, datasets_dir: Path) -> tuple[Path, dict]:
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
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)
    meta = json.loads(out_meta.read_text(encoding="utf-8"))
    return out_bin, meta


def write_manifest(specs: Iterable[RunSpec], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["run_id", "network", "dim", "N", "alpha_target", "t", "family", "rep", "dataset_seed", "sample_seed", "materialize_join"])
        for spec in specs:
            writer.writerow([
                run_id(spec),
                spec.network,
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["full", "smoke"], default="full")
    args = ap.parse_args()

    if not BENCH.exists():
        print(f"missing benchmark binary: {BENCH}", file=sys.stderr)
        return 1

    specs = full_plan() if args.mode == "full" else smoke_plan()
    mode_root = REPO_ROOT / "results" / args.mode
    datasets_dir = mode_root / "datasets"
    runs_dir = mode_root / "runs"
    analysis_dir = mode_root / "analysis"
    datasets_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir.mkdir(parents=True, exist_ok=True)
    write_manifest(specs, mode_root / "plan_manifest.csv")

    dataset_cache: dict[DatasetKey, tuple[Path, dict]] = {}
    total = len(specs)
    for idx, spec in enumerate(specs, start=1):
        rid = run_id(spec)
        out_dir = runs_dir / rid
        metrics_json = out_dir / "metrics.json"
        if metrics_json.exists():
            print(f"[{idx}/{total}] skip {rid}")
            continue

        if spec.dataset_key not in dataset_cache:
            dataset_cache[spec.dataset_key] = ensure_dataset(spec.dataset_key, datasets_dir)
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
        subprocess.run(cmd, check=True, cwd=REPO_ROOT)

    subprocess.run([sys.executable, str(ANALYZE), "--results-root", str(mode_root)], check=True, cwd=REPO_ROOT)
    print(f"done: {mode_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
