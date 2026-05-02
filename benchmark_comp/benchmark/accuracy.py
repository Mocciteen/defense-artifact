from __future__ import annotations

import argparse
import json
import os
import pickle
import struct
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


OUTPUT_ELEMENTS = 10
OUTPUT_BYTES = OUTPUT_ELEMENTS * 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate stream-mode accuracy for benchmark groups.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset", choices=("mnist", "cifar10"), required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--groups", nargs="*", default=None)
    parser.add_argument("--allow-missing", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--input-shape", default=None, help="N,C,H,W; default inferred from variant/dataset")
    parser.add_argument("--mean", default=None, help="comma-separated channel means")
    parser.add_argument("--std", default=None, help="comma-separated channel stds")
    parser.add_argument("--resize", choices=("none", "bilinear", "bicubic"), default=None)
    parser.add_argument("--save-predictions", action="store_true")
    return parser.parse_args()


def floats(text: str | None) -> list[float] | None:
    return None if text in {None, ""} else [float(item) for item in text.split(",")]


def shape(text: str | None, default: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    return default if not text else tuple(int(item) for item in text.split(","))  # type: ignore[return-value]


def load_mnist(root: Path) -> tuple[np.ndarray, np.ndarray]:
    images = root / "t10k-images-idx3-ubyte"
    labels = root / "t10k-labels-idx1-ubyte"
    raw_i, raw_l = images.read_bytes(), labels.read_bytes()
    magic, count, rows, cols = struct.unpack(">IIII", raw_i[:16])
    if magic != 2051:
        raise RuntimeError(f"unexpected MNIST image magic: {magic}")
    magic_l, count_l = struct.unpack(">II", raw_l[:8])
    if magic_l != 2049 or count_l != count:
        raise RuntimeError("invalid MNIST label file")
    return np.frombuffer(raw_i, dtype=np.uint8, offset=16).reshape(count, rows, cols), np.frombuffer(raw_l, dtype=np.uint8, offset=8).astype(np.int64)


def load_cifar10(root: Path) -> tuple[np.ndarray, np.ndarray]:
    with (root / "test_batch").open("rb") as handle:
        payload = pickle.load(handle, encoding="bytes")
    data = np.asarray(payload[b"data"], dtype=np.uint8).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    return data, np.asarray(payload.get(b"labels", payload.get(b"fine_labels")), dtype=np.int64)


def infer_preprocess(args: argparse.Namespace, config: dict[str, Any]) -> dict[str, Any]:
    variant = str(config.get("variant", ""))
    if args.dataset == "mnist":
        default = {"shape": (1, 1, 28, 28), "mean": [0.1307], "std": [0.3081], "resize": "none"}
    elif variant == "resnet_cifar":
        default = {"shape": (1, 3, 96, 96), "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225], "resize": "bicubic"}
    else:
        default = {"shape": (1, 3, 32, 32), "mean": [0.4914, 0.4822, 0.4465], "std": [0.2023, 0.1994, 0.2010], "resize": "none"}
    default["shape"] = shape(args.input_shape, default["shape"])
    default["mean"] = floats(args.mean) or default["mean"]
    default["std"] = floats(args.std) or default["std"]
    default["resize"] = args.resize or default["resize"]
    default["input_bytes"] = int(np.prod(default["shape"])) * 4
    return default


def resize_rgb(image: np.ndarray, height: int, width: int, mode: str) -> np.ndarray:
    if image.shape[:2] == (height, width):
        return image.astype(np.float32) / np.float32(255.0)
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Pillow is required for resized CIFAR preprocessing") from exc
    resample = Image.Resampling.BICUBIC if mode == "bicubic" else Image.Resampling.BILINEAR
    return np.asarray(Image.fromarray(image, mode="RGB").resize((width, height), resample=resample), dtype=np.float32) / np.float32(255.0)


def preprocess(image: np.ndarray, spec: dict[str, Any], dataset: str) -> np.ndarray:
    _, channels, height, width = spec["shape"]
    if dataset == "mnist":
        array = image.astype(np.float32) / np.float32(255.0)
        array = ((array - np.float32(spec["mean"][0])) / np.float32(spec["std"][0]))[None, None, ...]
    else:
        array = resize_rgb(image, height, width, str(spec["resize"]))
        array = (array - np.asarray(spec["mean"], dtype=np.float32)) / np.asarray(spec["std"], dtype=np.float32)
        array = np.transpose(array, (2, 0, 1))[None, ...]
    if array.shape != tuple(spec["shape"]):
        raise RuntimeError(f"preprocessed shape {array.shape} != expected {spec['shape']}")
    return np.ascontiguousarray(array, dtype=np.float32)


def resolve_path(base: Path, value: str | None) -> str | None:
    if value is None:
        return None
    path = Path(value)
    return str(path if path.is_absolute() else base / path)


def resolve_command(base: Path, command: list[Any]) -> list[str]:
    out = []
    for part in command:
        text = str(part)
        if text.startswith("-") or "/" not in text:
            out.append(text)
        else:
            out.append(resolve_path(base, text) or text)
    return out


def groups(config: dict[str, Any], base: Path, selected: list[str] | None, allow_missing: bool) -> list[dict[str, Any]]:
    requested = set(selected or [])
    out, missing = [], []
    for raw in config.get("groups", []):
        if requested and raw.get("name") not in requested:
            continue
        if raw.get("status", "ready") != "ready":
            missing.append(str(raw.get("name")))
            continue
        group = dict(raw)
        if group.get("command"):
            command = resolve_command(base, group["command"])
        else:
            command = [resolve_path(base, str(group["executable"])), resolve_path(base, str(group["weights"])), resolve_path(base, str(group["input"]))]
        group["stream_command"] = [part for part in command if part] + ["--stream"]
        out.append(group)
    missing.extend(sorted(requested - {str(group["name"]) for group in out}))
    if missing and not allow_missing:
        raise SystemExit(f"missing or not-ready groups: {', '.join(missing)}")
    if not out:
        raise SystemExit("no groups selected")
    return out


def read_exact(stream, size: int) -> bytes:
    chunks, remaining = [], size
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            raise RuntimeError("stream ended early")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


class Stream:
    def __init__(self, group: dict[str, Any], stderr_path: Path):
        env = os.environ.copy()
        env.update({str(key): str(value) for key, value in (group.get("env") or {}).items()})
        self.proc = subprocess.Popen(group["stream_command"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=stderr_path.open("w"), env=env)

    def infer(self, tensor: np.ndarray) -> np.ndarray:
        assert self.proc.stdin is not None and self.proc.stdout is not None
        self.proc.stdin.write(tensor.tobytes(order="C"))
        self.proc.stdin.flush()
        return np.frombuffer(read_exact(self.proc.stdout, OUTPUT_BYTES), dtype=np.float32).copy()

    def close(self) -> None:
        if self.proc.stdin:
            self.proc.stdin.close()
        self.proc.wait(timeout=5)


def main() -> None:
    args = parse_args()
    if args.offset < 0 or (args.limit is not None and args.limit <= 0):
        raise SystemExit("invalid --offset/--limit")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    images, labels = load_mnist(args.dataset_root) if args.dataset == "mnist" else load_cifar10(args.dataset_root)
    spec = infer_preprocess(args, config)
    selected = np.arange(args.offset, labels.shape[0] if args.limit is None else min(labels.shape[0], args.offset + args.limit), dtype=np.int64)
    run_groups = groups(config, args.config.resolve().parent, args.groups, args.allow_missing)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    np.save(args.output_dir / "selected_indices.npy", selected)
    summary: dict[str, Any] = {"framework": config.get("framework"), "variant": config.get("variant"), "dataset": args.dataset, "num_samples": int(selected.size), "preprocess": spec, "groups": {}}
    baseline = None
    for group in run_groups:
        name = str(group["name"])
        group_dir = args.output_dir / name
        group_dir.mkdir()
        preds = np.empty(selected.size, dtype=np.int64)
        correct = 0
        session = Stream(group, group_dir / "stderr.txt")
        try:
            for local, index in enumerate(selected):
                logits = session.infer(preprocess(images[int(index)], spec, args.dataset))
                preds[local] = int(np.argmax(logits))
                correct += int(preds[local] == int(labels[int(index)]))
        finally:
            session.close()
        row = {"accuracy": correct / max(int(selected.size), 1), "correct": int(correct), "num_samples": int(selected.size), "stream_command": group["stream_command"]}
        if baseline is not None:
            flips = int(np.count_nonzero(preds != baseline))
            row["prediction_flips_vs_baseline"] = flips
            row["flip_rate_vs_baseline"] = flips / max(int(selected.size), 1)
        if args.save_predictions:
            np.save(group_dir / "predictions.npy", preds)
        if name == "undefended":
            baseline = preds.copy()
        summary["groups"][name] = row
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(args.output_dir), "groups": list(summary["groups"])}, indent=2))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
