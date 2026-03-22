#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
VENDOR_ALACARTE = REPO_ROOT / "vendor" / "Alacarte"
if str(VENDOR_ALACARTE) not in sys.path:
    sys.path.insert(0, str(VENDOR_ALACARTE))

import alacarte_rectgen as ar  # noqa: E402

MAGIC = b"SJSBOX1\0"
VERSION = 1
SCALAR_TYPE_FLOAT32 = 1
PRESET = {
    "name": "TH",
    "dtype": "float32",
    "tune_samples": 200_000,
    "tune_tol_rel": 0.02,
    "tune_max_iter": 30,
    "chunk_size": 2_000_000,
    "volume_dist": "fixed",
    "volume_cv": 0.25,
}
FAMILY_TO_SIGMA = {
    "F0": 0.0,
    "F1": 0.75,
}


def write_binary_dataset(path: Path, R: np.ndarray, S: np.ndarray, dim: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = struct.pack(
        "<8sIIIIQQ",
        MAGIC,
        VERSION,
        SCALAR_TYPE_FLOAT32,
        dim,
        0,
        int(R.shape[0]),
        int(S.shape[0]),
    )
    with path.open("wb") as f:
        f.write(header)
        f.write(np.asarray(R, dtype=np.float32, order="C").tobytes(order="C"))
        f.write(np.asarray(S, dtype=np.float32, order="C").tobytes(order="C"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-bin", required=True)
    ap.add_argument("--out-meta", required=True)
    ap.add_argument("--dim", type=int, required=True)
    ap.add_argument("--N", type=int, required=True)
    ap.add_argument("--alpha-target", type=float, required=True)
    ap.add_argument("--family", choices=sorted(FAMILY_TO_SIGMA), required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    out_bin = Path(args.out_bin)
    out_meta = Path(args.out_meta)
    if out_bin.exists() and out_meta.exists() and not args.force:
        print(f"[dataset] reuse {out_bin}")
        return 0

    shape_sigma = FAMILY_TO_SIGMA[args.family]
    R, S, info = ar.make_rectangles_R_S(
        nR=args.N,
        nS=args.N,
        alpha_out=args.alpha_target,
        d=args.dim,
        universe=None,
        volume_dist=PRESET["volume_dist"],
        volume_cv=PRESET["volume_cv"],
        shape_sigma=shape_sigma,
        tune_samples=PRESET["tune_samples"],
        tune_tol_rel=PRESET["tune_tol_rel"],
        tune_max_iter=PRESET["tune_max_iter"],
        seed=args.seed,
        dtype=np.float32,
    )

    R_arr = R.as_array().astype(np.float32, copy=False)
    S_arr = S.as_array().astype(np.float32, copy=False)
    write_binary_dataset(out_bin, R_arr, S_arr, args.dim)

    meta = {
        "preset": PRESET["name"],
        "dim": args.dim,
        "nR": args.N,
        "nS": args.N,
        "family": args.family,
        "shape_sigma": shape_sigma,
        "seed": args.seed,
        "coverage": float(info["coverage"]),
        "alpha_target": float(info["alpha_target"]),
        "alpha_expected_est": float(info["alpha_expected_est"]),
        "pair_intersection_prob_est": float(info["pair_intersection_prob_est"]),
        "chunk_size": PRESET["chunk_size"],
        "volume_dist": PRESET["volume_dist"],
        "volume_cv": PRESET["volume_cv"],
        "tune_samples": PRESET["tune_samples"],
        "tune_tol_rel": PRESET["tune_tol_rel"],
        "tune_max_iter": PRESET["tune_max_iter"],
        "params": info.get("params", {}),
        "tune_history": info.get("tune_history", []),
        "dataset_path": str(out_bin),
    }
    out_meta.parent.mkdir(parents=True, exist_ok=True)
    out_meta.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[dataset] generated {out_bin}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
