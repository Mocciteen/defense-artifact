#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Any


ROOT_DIR = Path("/path/to/high_leakage_workspace")
PYTHON_BIN = Path("/path/to/tvm_ana_workspace/.venv_ciphersteal/bin/python")
GLOW_SCRIPT = ROOT_DIR / "collect_glow_paddrtrace.py"
TVM_SCRIPT = ROOT_DIR / "collect_tvm_paddrtrace.py"
MASTER_PARENT = ROOT_DIR / "launches"

DEFAULT_GLOW_TASKS = ",".join([*(f"MC{i:02d}" for i in range(1, 18)), *(f"IC{i:02d}" for i in range(1, 18))])
DEFAULT_TVM_SPECS = ",".join([*(f"TV{i:02d}" for i in range(1, 18)), *(f"TA{i:02d}" for i in range(1, 18))])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch Glow and TVM high-order leakage collection in background.")
    parser.add_argument("--root", default=None, help="Explicit launch root directory.")
    parser.add_argument("--glow-tasks", default=DEFAULT_GLOW_TASKS)
    parser.add_argument("--tvm-specs", default=DEFAULT_TVM_SPECS)
    parser.add_argument("--glow-parallelism", type=int, default=4)
    parser.add_argument("--tvm-parallelism", type=int, default=4)
    return parser.parse_args()


def timestamp_tag() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="ascii")


def main() -> int:
    args = parse_args()
    for required in (PYTHON_BIN, GLOW_SCRIPT, TVM_SCRIPT):
        if not required.exists():
            raise FileNotFoundError(f"missing required path: {required}")

    root = Path(args.root).resolve() if args.root else (MASTER_PARENT / timestamp_tag()).resolve()
    glow_session = root / "glow"
    tvm_session = root / "tvm"
    root.mkdir(parents=True, exist_ok=True)

    glow_cmd = [
        str(PYTHON_BIN),
        str(GLOW_SCRIPT),
        "launch",
        "--session-dir",
        str(glow_session),
        "--tasks",
        args.glow_tasks,
        "--parallelism",
        str(args.glow_parallelism),
    ]
    tvm_cmd = [
        str(PYTHON_BIN),
        str(TVM_SCRIPT),
        "launch",
        "--session-dir",
        str(tvm_session),
        "--specs",
        args.tvm_specs,
        "--parallelism",
        str(args.tvm_parallelism),
    ]

    subprocess.run(glow_cmd, cwd=ROOT_DIR, check=True)
    subprocess.run(tvm_cmd, cwd=ROOT_DIR, check=True)

    glow_launch = json.loads((glow_session / "launch_summary.json").read_text(encoding="utf-8"))
    tvm_launch = json.loads((tvm_session / "launch_summary.json").read_text(encoding="utf-8"))

    summary = {
        "launch_root": str(root),
        "launched_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "glow": glow_launch,
        "tvm": tvm_launch,
        "monitor": {
            "glow_manager_log": glow_launch["manager_log"],
            "tvm_manager_log": tvm_launch["manager_log"],
            "glow_jobs": str(glow_session / "jobs.json"),
            "tvm_jobs": str(tvm_session / "jobs.json"),
        },
    }
    write_json(root / "launch_summary.json", summary)
    print(str(root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
