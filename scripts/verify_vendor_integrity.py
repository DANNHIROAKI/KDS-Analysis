#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CHECKSUM_FILE = REPO_ROOT / "vendor" / "checksums.sha256"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    if not CHECKSUM_FILE.exists():
        print(f"missing checksum file: {CHECKSUM_FILE}", file=sys.stderr)
        return 1
    ok = True
    for line in CHECKSUM_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        expected, rel = line.split(None, 1)
        rel = rel.strip()
        path = REPO_ROOT / "vendor" / rel if not rel.endswith('.zip') else REPO_ROOT / "vendor" / rel
        if not path.exists():
            print(f"missing vendor artifact: {path}", file=sys.stderr)
            ok = False
            continue
        actual = sha256(path)
        if actual != expected:
            print(f"hash mismatch: {path}\n  expected {expected}\n  actual   {actual}", file=sys.stderr)
            ok = False
    if ok:
      print("vendor integrity verified")
      return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
