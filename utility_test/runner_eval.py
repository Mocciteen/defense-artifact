#!/usr/bin/env python3
"""Small runner-based accuracy and overhead evaluation helper.

The evaluation code assumes model/backend runners already exist.  It only
sets the off/on environment, invokes a runner command, records timing, and
summarizes prediction files.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shlex
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


BACKENDS = {
    "MC": "Glow model-compiler / bundle runner",
    "IC": "Glow image-classifier",
    "TV": "TVM VM runtime",
    "TA": "TVM AOT runtime",
}

THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}

GLOW_OFF_ENV: dict[str, str] = {}
GLOW_ON_ENV: dict[str, str] = {}
GLOW_ON_MAXPOOL_ENV: dict[str, str] = {}
TVM_OFF_ENV: dict[str, str] = {}
TVM_ON_ENV: dict[str, str] = {}


def die(message: str) -> None:
    raise SystemExit(f"error: {message}")


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def parse_key_values(values: list[str]) -> dict[str, str]:
    env: dict[str, str] = {}
    for item in values:
        if "=" not in item:
            die(f"expected KEY=VALUE, got {item!r}")
        key, value = item.split("=", 1)
        env[key] = value
    return env


def mode_env(backend: str, mode: str) -> dict[str, str]:
    if backend in {"MC", "IC"}:
        if mode == "off":
            return dict(GLOW_OFF_ENV)
        if mode == "on":
            return dict(GLOW_ON_ENV)
        if mode == "on_maxpool":
            return dict(GLOW_ON_MAXPOOL_ENV)
    if backend in {"TV", "TA"}:
        if mode == "off":
            return dict(TVM_OFF_ENV)
        if mode == "on":
            return dict(TVM_ON_ENV)
    die(f"unsupported backend/mode pair: {backend}/{mode}")


def command_line(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def run_command(args: argparse.Namespace) -> int:
    if args.backend not in BACKENDS:
        die(f"unknown backend {args.backend!r}")
    runner_command = args.runner_command
    if runner_command and runner_command[0] == "--":
        runner_command = runner_command[1:]
    if not runner_command:
        die("missing runner command after --")

    run_dir = args.run_dir / args.spec / args.mode
    run_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = args.stdout or run_dir / "stdout.txt"
    stderr_path = args.stderr or run_dir / "stderr.txt"
    result_path = args.result or run_dir / "result.json"

    env_update = {}
    if not args.no_thread_env:
        env_update.update(THREAD_ENV)
    env_update.update(mode_env(args.backend, args.mode))
    env_update.update(parse_key_values(args.env))

    payload = {
        "spec": args.spec,
        "backend": args.backend,
        "backend_name": BACKENDS[args.backend],
        "mode": args.mode,
        "command": runner_command,
        "command_line": command_line(runner_command),
        "cwd": str(args.cwd) if args.cwd else None,
        "env": env_update,
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
        "predictions": str(args.predictions) if args.predictions else None,
    }

    if args.dry_run:
        payload["dry_run"] = True
        write_json(result_path, payload)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    run_env = os.environ.copy()
    run_env.update(env_update)
    start = time.perf_counter()
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        completed = subprocess.run(runner_command, cwd=args.cwd, env=run_env, stdout=stdout, stderr=stderr, check=False)
    wall_time = time.perf_counter() - start

    payload.update(
        {
            "dry_run": False,
            "returncode": completed.returncode,
            "ok": completed.returncode == 0,
            "wall_time_sec": wall_time,
        }
    )
    write_json(result_path, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return completed.returncode


def parse_scalar(value: str) -> int | float | str:
    text = value.strip()
    try:
        if any(ch in text for ch in ".eE"):
            return float(text)
        return int(text)
    except ValueError:
        return text


def load_values(path: Path) -> Any:
    suffix = path.suffix.lower()
    if suffix == ".json":
        payload = read_json(path)
        if isinstance(payload, dict):
            for key in ("predictions", "labels", "logits", "scores", "values"):
                if key in payload:
                    return payload[key]
        return payload
    if suffix not in {".csv", ".tsv"}:
        die(f"unsupported file type: {path}")

    delimiter = "\t" if suffix == ".tsv" else ","
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.reader(handle, delimiter=delimiter)
        for row in reader:
            if not row:
                continue
            rows.append(parse_scalar(row[0]) if len(row) == 1 else [parse_scalar(cell) for cell in row])
    return rows


def is_vector(value: Any) -> bool:
    return isinstance(value, list | tuple)


def argmax(values: list[float]) -> int:
    return max(range(len(values)), key=lambda index: values[index])


def classes(predictions: Any) -> list[int]:
    result = []
    for item in predictions:
        if is_vector(item):
            result.append(argmax([float(value) for value in item]))
        else:
            result.append(int(item))
    return result


def check_lengths(left: list[Any], right: list[Any], name: str) -> None:
    if len(left) != len(right):
        die(f"{name}: length mismatch {len(left)} != {len(right)}")


def accuracy(predictions: Any, labels: Any) -> float:
    pred = classes(predictions)
    truth = [int(value) for value in labels]
    check_lengths(pred, truth, "accuracy")
    if not truth:
        die("accuracy requires at least one sample")
    return sum(int(a == b) for a, b in zip(pred, truth, strict=True)) / len(truth)


def sigmoid(value: float) -> float:
    if value >= 0:
        exp_value = math.exp(-value)
        return 1.0 / (1.0 + exp_value)
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


def score_matrix(scores: Any) -> list[list[float]]:
    matrix = []
    for row in scores:
        if not is_vector(row):
            die("multi-label metrics require vector rows")
        values = [float(value) for value in row]
        if values and (min(values) < 0.0 or max(values) > 1.0):
            values = [sigmoid(value) for value in values]
        matrix.append(values)
    return matrix


def label_matrix(labels: Any) -> list[list[int]]:
    return [[int(value) for value in row] for row in labels]


def binary_auc(y_true: list[int], y_score: list[float]) -> float | None:
    positives = sum(y_true)
    negatives = len(y_true) - positives
    if positives == 0 or negatives == 0:
        return None
    pairs = sorted(zip(y_score, y_true, strict=True), key=lambda item: item[0])
    rank_sum = 0.0
    rank = 1
    index = 0
    while index < len(pairs):
        end = index + 1
        while end < len(pairs) and pairs[end][0] == pairs[index][0]:
            end += 1
        avg_rank = (rank + rank + end - index - 1) / 2.0
        rank_sum += sum(avg_rank for _, label in pairs[index:end] if label)
        rank += end - index
        index = end
    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def macro_auroc(predictions: Any, labels: Any) -> float:
    scores = score_matrix(predictions)
    truth = label_matrix(labels)
    check_lengths(scores, truth, "macro_auroc")
    aucs = []
    for class_id in range(len(truth[0])):
        auc = binary_auc([row[class_id] for row in truth], [row[class_id] for row in scores])
        if auc is not None:
            aucs.append(auc)
    if not aucs:
        die("macro_auroc has no valid label column")
    return sum(aucs) / len(aucs)


def binary_label_accuracy(predictions: Any, labels: Any) -> float:
    scores = score_matrix(predictions)
    truth = label_matrix(labels)
    check_lengths(scores, truth, "binary_label_accuracy")
    total = 0
    correct = 0
    for score_row, label_row in zip(scores, truth, strict=True):
        check_lengths(score_row, label_row, "binary_label_accuracy row")
        for score, label in zip(score_row, label_row, strict=True):
            correct += int((score >= 0.5) == bool(label))
            total += 1
    if total == 0:
        die("binary_label_accuracy requires at least one label")
    return correct / total


def compute_metric(metric: str, predictions: Any, labels: Any) -> float:
    if metric in {"accuracy", "top1_accuracy"}:
        return accuracy(predictions, labels)
    if metric == "macro_auroc":
        return macro_auroc(predictions, labels)
    if metric == "binary_label_accuracy":
        return binary_label_accuracy(predictions, labels)
    die(f"unsupported metric: {metric}")


def summarize(args: argparse.Namespace) -> int:
    labels = load_values(args.labels)
    off_predictions = load_values(args.off)
    on_predictions = load_values(args.on)
    off_value = compute_metric(args.metric, off_predictions, labels)
    on_value = compute_metric(args.metric, on_predictions, labels)

    payload: dict[str, Any] = {
        "metric": args.metric,
        "off": {"metric_value": off_value, "predictions": str(args.off)},
        "on": {
            "metric_value": on_value,
            "predictions": str(args.on),
            "delta_vs_off": on_value - off_value,
        },
        "labels": str(args.labels),
    }
    if args.metric in {"accuracy", "top1_accuracy"}:
        off_classes = classes(off_predictions)
        on_classes = classes(on_predictions)
        check_lengths(off_classes, on_classes, "flip_rate_vs_off")
        payload["on"]["flip_rate_vs_off"] = sum(
            int(a != b) for a, b in zip(off_classes, on_classes, strict=True)
        ) / len(off_classes)

    if args.out:
        write_json(args.out, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def stats(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "median": None, "stdev": None, "min": None, "max": None}
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "stdev": statistics.pstdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def aggregate(args: argparse.Namespace) -> int:
    groups: dict[tuple[str, str, str], list[float]] = {}
    for path in sorted(args.input_root.rglob("*.json")):
        payload = read_json(path)
        if "wall_time_sec" not in payload:
            continue
        key = (
            str(payload.get("backend", "unknown")),
            str(payload.get("spec", payload.get("spec_id", path.parent.name))),
            str(payload.get("mode", path.parent.name)),
        )
        groups.setdefault(key, []).append(float(payload["wall_time_sec"]))

    summary = {
        f"{backend}/{spec}/{mode}": {
            "backend": backend,
            "spec": spec,
            "mode": mode,
            "wall_time_sec": stats(values),
        }
        for (backend, spec, mode), values in sorted(groups.items())
    }
    if args.out_json:
        write_json(args.out_json, summary)
    if args.out_tsv:
        args.out_tsv.parent.mkdir(parents=True, exist_ok=True)
        with args.out_tsv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, delimiter="\t")
            writer.writerow(["backend", "spec", "mode", "count", "mean", "median", "stdev"])
            for row in summary.values():
                wall = row["wall_time_sec"]
                writer.writerow([row["backend"], row["spec"], row["mode"], wall["count"], wall["mean"], wall["median"], wall["stdev"]])
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def show_env(args: argparse.Namespace) -> int:
    env = THREAD_ENV if not args.no_thread_env else {}
    env = {**env, **mode_env(args.backend, args.mode), **parse_key_values(args.env)}
    print(json.dumps(env, indent=2, sort_keys=True))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Runner-based accuracy and overhead helper.")
    subparsers = parser.add_subparsers(dest="action", required=True)

    env_parser = subparsers.add_parser("show-env")
    env_parser.add_argument("--backend", required=True, choices=sorted(BACKENDS))
    env_parser.add_argument("--mode", required=True)
    env_parser.add_argument("--env", nargs="*", default=[])
    env_parser.add_argument("--no-thread-env", action="store_true")

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--backend", required=True, choices=sorted(BACKENDS))
    run_parser.add_argument("--mode", required=True)
    run_parser.add_argument("--spec", required=True)
    run_parser.add_argument("--run-dir", type=Path, default=Path("runs/acc_overhead"))
    run_parser.add_argument("--predictions", type=Path, default=None)
    run_parser.add_argument("--stdout", type=Path, default=None)
    run_parser.add_argument("--stderr", type=Path, default=None)
    run_parser.add_argument("--result", type=Path, default=None)
    run_parser.add_argument("--cwd", type=Path, default=None)
    run_parser.add_argument("--env", nargs="*", default=[])
    run_parser.add_argument("--no-thread-env", action="store_true")
    run_parser.add_argument("--dry-run", action="store_true")
    run_parser.add_argument("runner_command", nargs=argparse.REMAINDER)

    summary_parser = subparsers.add_parser("summarize")
    summary_parser.add_argument("--metric", required=True)
    summary_parser.add_argument("--labels", type=Path, required=True)
    summary_parser.add_argument("--off", type=Path, required=True)
    summary_parser.add_argument("--on", type=Path, required=True)
    summary_parser.add_argument("--out", type=Path, default=None)

    aggregate_parser = subparsers.add_parser("aggregate")
    aggregate_parser.add_argument("--input-root", type=Path, required=True)
    aggregate_parser.add_argument("--out-json", type=Path, default=None)
    aggregate_parser.add_argument("--out-tsv", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.action == "show-env":
        return show_env(args)
    if args.action == "run":
        return run_command(args)
    if args.action == "summarize":
        return summarize(args)
    if args.action == "aggregate":
        return aggregate(args)
    die(f"unsupported command: {args.action}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
