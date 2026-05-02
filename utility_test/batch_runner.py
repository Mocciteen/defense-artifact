#!/usr/bin/env python3
"""Batch launcher for runner_eval.py.

This script only expands specs, modes, and repeats.  It does not know about
datasets or input files; those belong to the runner command supplied after `--`.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path


def die(message: str) -> None:
    raise SystemExit(f"error: {message}")


def split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def render(value: str, variables: dict[str, str]) -> str:
    try:
        return value.format(**variables)
    except KeyError as exc:
        die(f"unknown placeholder {{{exc.args[0]}}} in {value!r}")


def render_command(command: list[str], variables: dict[str, str]) -> list[str]:
    return [render(part, variables) for part in command]


def command_line(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch launcher for runner_eval.py.")
    parser.add_argument("--runner-eval", type=Path, default=Path(__file__).with_name("runner_eval.py"))
    parser.add_argument("--backend", required=True, choices=("MC", "IC", "TV", "TA"))
    parser.add_argument("--specs", required=True, help="Comma-separated spec IDs, e.g. MC01,MC03.")
    parser.add_argument("--modes", default="off,on", help="Comma-separated modes, e.g. off,on.")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--run-dir", type=Path, default=Path("runs/acc_overhead"))
    parser.add_argument("--cwd", type=Path, default=None)
    parser.add_argument("--env", nargs="*", default=[], help="Extra KEY=VALUE pairs passed to runner_eval.py.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    return parser.parse_args()


def runner_eval_invocation(args: argparse.Namespace, variables: dict[str, str], command: list[str]) -> list[str]:
    invocation = [
        sys.executable,
        str(args.runner_eval),
        "run",
        "--backend",
        args.backend,
        "--mode",
        variables["mode"],
        "--spec",
        variables["spec"],
        "--run-dir",
        str(args.run_dir),
        "--predictions",
        variables["predictions"],
        "--stdout",
        variables["stdout"],
        "--stderr",
        variables["stderr"],
        "--result",
        variables["result"],
    ]
    if args.cwd:
        invocation.extend(["--cwd", str(args.cwd)])
    if args.dry_run:
        invocation.append("--dry-run")
    if args.env:
        invocation.append("--env")
        invocation.extend(args.env)
    invocation.append("--")
    invocation.extend(command)
    return invocation


def main() -> int:
    args = parse_args()
    runner_command = args.command[1:] if args.command and args.command[0] == "--" else args.command
    if not runner_command:
        die("missing runner command after --")
    if args.repeat < 1:
        die("--repeat must be >= 1")

    failures = 0
    for spec in split_csv(args.specs):
        for mode in split_csv(args.modes):
            for repeat in range(args.repeat):
                case_dir = args.run_dir / spec / mode / f"repeat_{repeat:02d}"
                variables = {
                    "backend": args.backend,
                    "spec": spec,
                    "mode": mode,
                    "repeat": str(repeat),
                    "repeat2": f"{repeat:02d}",
                    "run_dir": str(args.run_dir),
                    "case_dir": str(case_dir),
                    "predictions": str(case_dir / "predictions.json"),
                    "stdout": str(case_dir / "stdout.txt"),
                    "stderr": str(case_dir / "stderr.txt"),
                    "result": str(case_dir / "result.json"),
                }
                rendered = render_command(runner_command, variables)
                invocation = runner_eval_invocation(args, variables, rendered)
                print(command_line(invocation), flush=True)
                completed = subprocess.run(invocation, check=False)
                if completed.returncode != 0:
                    failures += 1
                    if not args.continue_on_error:
                        return completed.returncode
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
