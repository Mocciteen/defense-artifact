#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Build a stable job manifest for incremental higher-order analysis "
            "from a live launch root. Only tasks/specs with jobs.json state=finished "
            "and returncode=0 are included."
        )
    )
    ap.add_argument("--launch-root", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--tasks", default="", help="optional comma-separated task/spec ids to include")
    return ap.parse_args()


def read_jobs(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def collect_samples(task_runs_dir: Path) -> list[str]:
    out: list[str] = []
    if not task_runs_dir.is_dir():
        return out
    for sample_dir in sorted(task_runs_dir.iterdir()):
        run_off = sample_dir / "off"
        run_on = sample_dir / "on"
        if not run_off.is_dir() or not run_on.is_dir():
            continue
        if not (run_off / "paddrtrace.bin").is_file():
            continue
        if not (run_off / "paddrtrace.ip.txt").is_file():
            continue
        if not (run_on / "paddrtrace.bin").is_file():
            continue
        if not (run_on / "paddrtrace.ip.txt").is_file():
            continue
        out.append(sample_dir.name)
    return out


def stable_ids(jobs: dict[str, Any], selected: set[str] | None) -> list[str]:
    out: list[str] = []
    for task_id, item in sorted(jobs.items()):
        if selected is not None and task_id not in selected:
            continue
        if item.get("state") != "finished":
            continue
        if int(item.get("returncode", -1)) != 0:
            continue
        out.append(task_id)
    return out


def main() -> int:
    args = parse_args()
    launch_root = Path(args.launch_root).resolve()
    selected = None
    if args.tasks.strip():
        selected = {part.strip() for part in args.tasks.split(",") if part.strip()}

    payload: list[dict[str, Any]] = []
    for domain in ("glow", "tvm"):
        session_dir = launch_root / domain
        jobs = read_jobs(session_dir / "jobs.json")
        runs_root = session_dir / "runs"
        for task_id in stable_ids(jobs, selected):
            task_runs_dir = runs_root / task_id
            for sample in collect_samples(task_runs_dir):
                payload.append(
                    {
                        "task_id": task_id,
                        "sample": sample,
                        "session_tag": launch_root.name,
                        "run_off": str(task_runs_dir / sample / "off"),
                        "run_on": str(task_runs_dir / sample / "on"),
                    }
                )

    out_json = Path(args.out_json).resolve()
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(str(out_json))
    print(f"pairs={len(payload)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
