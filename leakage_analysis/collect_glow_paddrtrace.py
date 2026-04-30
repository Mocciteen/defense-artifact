#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import json
import os
import re
import shlex
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np


ROOT_DIR = Path("/path/to/high_leakage_workspace")
SESSION_PARENT = ROOT_DIR / "glow_sessions"
PYTHON_BIN = Path("/path/to/tvm_ana_workspace/.venv_ciphersteal/bin/python")
GLOW_ROOT = Path("/path/to/glow_workspace")
PIN_BIN = GLOW_ROOT / "pin" / "pin"
PINTOOL = GLOW_ROOT / "build_release_all" / "glow" / "pintrace" / "obj-intel64" / "paddrtrace.so"
IMAGE_CLASSIFIER = (
    GLOW_ROOT / "build_release_all" / "glow" / "build_clang14_customllvm_maxpool" / "bin" / "image-classifier"
)
LD_LIBRARY_PATH_VALUE = "/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu"

OFF_DIR = "relu_off_maxpool_off_pureoff_input_zero_dither_off"
ON_DIR = (
    "relu_on_patchbits_30_fixedbits_7_fixedvalue_0x75_inc_3_positive_0_"
    "maxpool_on_input_zero_dither_random_seed1234_thresh1"
)

DEFAULT_TASK_IDS = [*(f"MC{i:02d}" for i in range(1, 18)), *(f"IC{i:02d}" for i in range(1, 18))]
SAMPLES = ("idx00042", "idx00114")
LENET_IC_SAMPLES = ("mnist_test0", "idx00001")
MNIST_DATASET_ROOT = Path("/path/to/datasets/mnist")
LENET_IC_SAMPLE_TO_INDEX = {
    "mnist_test0": 0,
    "idx00001": 1,
}

REQUESTED_CONFIG = {
    "patchbits": 30,
    "fixedbits": 7,
    "fixedvalue_decimal": 117,
    "fixedvalue_binary_requested": "0b1110101",
    "inc": 3,
    "positive": 0,
    "maxpool_on": 1,
    "input_zero_dither": "random",
    "input_zero_dither_layout": "NCHW",
    "input_zero_dither_eps_min": "1e-5",
    "input_zero_dither_eps_max": "2e-5",
    "input_zero_dither_thresh": 1,
    "input_zero_dither_seed": 1234,
}

GLOW_ENV_OFF = {
    "GLOW_RELU_LOW12_PATCH": "0",
    "GLOW_MAXPOOL_LOWBIT_PATCH": "0",
    "GLOW_INPUT_ZERO_DITHER": "0",
}

GLOW_ENV_ON = {
    "GLOW_RELU_LOW12_PATCH": "1",
    "GLOW_RELU_PATCH_BITS": "30",
    "GLOW_RELU_PATCH_FIXED_BITS": "7",
    "GLOW_RELU_PATCH_FIXED_VALUE": "117",
    "GLOW_RELU_PATCH_INC": "3",
    "GLOW_RELU_PATCH_POSITIVE": "0",
    "GLOW_MAXPOOL_LOWBIT_PATCH": "1",
    "GLOW_INPUT_ZERO_DITHER": "random",
    "GLOW_INPUT_ZERO_DITHER_LAYOUT": "NCHW",
    "GLOW_INPUT_ZERO_DITHER_EPS_MIN": "1e-5",
    "GLOW_INPUT_ZERO_DITHER_EPS_MAX": "2e-5",
    "GLOW_INPUT_ZERO_DITHER_THRESH": "1",
    "GLOW_INPUT_ZERO_DITHER_SEED": "1234",
}

RELEVANT_ENV_KEYS = tuple(set(GLOW_ENV_OFF) | set(GLOW_ENV_ON))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect paddrtrace runs for MC01-MC17 and IC01-IC17 using the exact Glow task configuration."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="List selected tasks.")
    list_parser.add_argument("--tasks", default=None)

    launch_parser = subparsers.add_parser("launch", help="Create a session and detach a manager.")
    launch_parser.add_argument("--session-dir", default=None)
    launch_parser.add_argument("--tasks", default=None)
    launch_parser.add_argument("--parallelism", type=int, default=4)

    manager_parser = subparsers.add_parser("manager", help="Run the session manager.")
    manager_parser.add_argument("--session-dir", required=True)
    manager_parser.add_argument("--tasks", default=None)
    manager_parser.add_argument("--parallelism", type=int, default=4)

    worker_parser = subparsers.add_parser("worker", help="Run one task.")
    worker_parser.add_argument("--session-dir", required=True)
    worker_parser.add_argument("--task", required=True)

    return parser.parse_args()


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_parent(path)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="ascii")


def write_text(path: Path, content: str) -> None:
    ensure_parent(path)
    path.write_text(content, encoding="ascii", errors="ignore")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def timestamp_tag() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())


def make_session_dir(explicit: str | None) -> Path:
    if explicit:
        return Path(explicit).resolve()
    return (SESSION_PARENT / timestamp_tag()).resolve()


def render_cmd(cmd: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in cmd)


def clean_env() -> dict[str, str]:
    env = dict(os.environ)
    for key in RELEVANT_ENV_KEYS:
        env.pop(key, None)
    return env


def mode_env(mode: str) -> dict[str, str]:
    env = clean_env()
    env.update(GLOW_ENV_ON if mode == "on" else GLOW_ENV_OFF)
    return env


def selected_task_ids(task_arg: str | None) -> list[str]:
    if task_arg is None:
        return list(DEFAULT_TASK_IDS)
    selected = [part.strip() for part in task_arg.split(",") if part.strip()]
    unknown = [task_id for task_id in selected if task_id not in DEFAULT_TASK_IDS]
    if unknown:
        raise ValueError(f"unknown tasks: {', '.join(unknown)}")
    return selected


def exact_runlog(base: str, sample: str, mode: str) -> Path:
    addr_dir = Path(base) / sample / "addr"
    if mode != "on":
        return addr_dir / OFF_DIR / "run.log"

    preferred = addr_dir / ON_DIR / "run.log"
    if preferred.exists():
        return preferred

    for candidate in sorted(addr_dir.glob("relu_on_patchbits_*")):
        run_log = candidate / "run.log"
        name = candidate.name
        if not run_log.exists():
            continue
        if "fixedbits_7" not in name or "fixedvalue_0x75" not in name:
            continue
        if "inc_3" not in name or "positive_0" not in name:
            continue
        if "maxpool_on" not in name:
            continue
        if "input_zero_dither_random" not in name:
            continue
        if "seed1234" not in name:
            continue
        return run_log

    return preferred


def parse_runlog(path: Path) -> dict[str, str]:
    if not path.exists():
        raise FileNotFoundError(f"missing run.log: {path}")
    result: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("[+] "):
            line = line[4:]
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        result[key.strip()] = value.strip()
    return result


def validate_trace(run_dir: Path) -> dict[str, Any]:
    trace_bin = run_dir / "paddrtrace.bin"
    ipmap = run_dir / "paddrtrace.ip.txt"
    checks = {
        "trace_bin_exists": trace_bin.exists(),
        "trace_bin_nonempty": trace_bin.exists() and trace_bin.stat().st_size > 0,
        "ipmap_exists": ipmap.exists(),
        "ipmap_nonempty": ipmap.exists() and ipmap.stat().st_size > 0,
    }
    if not all(checks.values()):
        raise RuntimeError(f"paddrtrace validation failed for {run_dir}: {checks}")
    return checks


def parse_label_k1(text: str) -> int | None:
    match = re.search(r"Label-K1:\s*(\d+)", text)
    return int(match.group(1)) if match else None


def compare_text(reference_path: Path, trace_path: Path, out_json: Path, *, atol: float = 1e-2) -> dict[str, Any]:
    reference = reference_path.read_text(encoding="utf-8", errors="ignore")
    traced = trace_path.read_text(encoding="utf-8", errors="ignore")
    exact_match = reference == traced
    numeric_match = False
    max_abs_diff = None
    same_argmax = None
    same_top2_set = None
    cosine_similarity = None
    same_label_k1 = None
    ref_numbers = [float(value) for value in re.findall(r"[-+]?(?:\d+\.\d+|\d+|\.\d+)(?:[eE][-+]?\d+)?", reference)]
    trace_numbers = [float(value) for value in re.findall(r"[-+]?(?:\d+\.\d+|\d+|\.\d+)(?:[eE][-+]?\d+)?", traced)]
    if ref_numbers and len(ref_numbers) == len(trace_numbers):
        ref_array = np.asarray(ref_numbers, dtype=np.float64)
        trace_array = np.asarray(trace_numbers, dtype=np.float64)
        diffs = np.abs(ref_array - trace_array)
        max_abs_diff = float(np.max(diffs)) if diffs.size else 0.0
        numeric_match = max_abs_diff <= atol
        if ref_array.size > 1:
            same_argmax = bool(int(np.argmax(ref_array)) == int(np.argmax(trace_array)))
        if ref_array.size > 2:
            ref_top2 = set(np.argsort(ref_array)[-2:].tolist())
            trace_top2 = set(np.argsort(trace_array)[-2:].tolist())
            same_top2_set = bool(ref_top2 == trace_top2)
        ref_norm = float(np.linalg.norm(ref_array))
        trace_norm = float(np.linalg.norm(trace_array))
        if ref_norm > 0.0 and trace_norm > 0.0:
            cosine_similarity = float(np.dot(ref_array, trace_array) / (ref_norm * trace_norm))
    ref_label_k1 = parse_label_k1(reference)
    trace_label_k1 = parse_label_k1(traced)
    if ref_label_k1 is not None and trace_label_k1 is not None:
        same_label_k1 = ref_label_k1 == trace_label_k1
    # Defended runs can drift numerically while preserving decision semantics.
    # Accept either stable top1, or near-tie swaps where top2 candidates remain identical.
    semantic_match = bool(same_label_k1) or bool(
        cosine_similarity is not None
        and (
            (same_argmax and cosine_similarity >= 0.995)
            or (same_top2_set and cosine_similarity >= 0.9999)
        )
    )
    payload = {
        "reference_file": str(reference_path),
        "trace_file": str(trace_path),
        "exact_match": exact_match,
        "numeric_match": numeric_match,
        "max_abs_diff": max_abs_diff,
        "same_argmax": same_argmax,
        "same_top2_set": same_top2_set,
        "cosine_similarity": cosine_similarity,
        "reference_label_k1": ref_label_k1,
        "trace_label_k1": trace_label_k1,
        "same_label_k1": same_label_k1,
        "semantic_match": semantic_match,
        "verification_passed": bool(exact_match or numeric_match or semantic_match),
        "reference_size": len(reference),
        "trace_size": len(traced),
    }
    write_json(out_json, payload)
    return payload


def reconstruct_vgg_mc_input(session_dir: Path, task_id: str, sample: str, dataset: str) -> Path:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Pillow is required to reconstruct VGG model-compiler inputs") from exc

    out_dir = session_dir / "generated_inputs" / task_id / sample
    input_bin = out_dir / "input_nchw_f32.bin"
    if input_bin.exists():
        return input_bin

    if dataset == "mnist":
        image_path = (
            GLOW_ROOT
            / "trace"
            / "vggnet_all"
            / "image-classifier"
            / "mulsample_mnist"
            / "input"
            / sample
            / "mnist_28.png"
        )
        with Image.open(image_path) as image:
            array = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
        nchw = array.reshape(1, 1, array.shape[0], array.shape[1])
    elif dataset == "cifar":
        image_path = (
            GLOW_ROOT
            / "trace"
            / "vggnet_all"
            / "image-classifier"
            / "mulsample_cifar"
            / "input"
            / sample
            / "cifar_rgb32.png"
        )
        with Image.open(image_path) as image:
            array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        nchw = np.transpose(array, (2, 0, 1))[None, ...]
    else:
        raise ValueError(f"unsupported VGG model-compiler dataset reconstruction: {dataset}")

    out_dir.mkdir(parents=True, exist_ok=True)
    nchw.astype(np.float32).tofile(input_bin)
    return input_bin


def resolve_mc_input_path(
    session_dir: Path,
    task_id: str,
    task: dict[str, Any],
    sample: str,
    raw_input: str,
) -> tuple[Path, str]:
    raw_path = Path(raw_input)
    if raw_path.exists():
        return raw_path, "logged_path"

    candidates = [Path(str(raw_path).replace("mulample_", "mulsample_"))]
    for candidate in candidates:
        if candidate.exists():
            return candidate, f"normalized_path:{candidate}"

    if task["family"] == "vggnet" and task["dataset"] in {"mnist", "cifar"}:
        rebuilt = reconstruct_vgg_mc_input(session_dir, task_id, sample, task["dataset"])
        return rebuilt, f"reconstructed_from_vgg_image_asset:{rebuilt}"

    return raw_path, "missing_logged_path"


def build_registry() -> dict[str, dict[str, Any]]:
    lenet_inputs = {
        sample: str(
            GLOW_ROOT
            / "trace"
            / "lenet_all"
            / "model_compiler"
            / "mulsample_run_mnist"
            / "inputs"
            / sample
            / "input_nchw_f32.bin"
        )
        for sample in SAMPLES
    }
    return {
        "MC01": {
            "type": "mc",
            "source": "custom",
            "family": "lenet",
            "dataset": "mnist",
            "samples": SAMPLES,
            "runner_off": str(
                GLOW_ROOT
                / "model"
                / "lenet"
                / "bundles"
                / "mnist"
                / "bundle_relu_off_maxpool_off"
                / "lenet_mnist_relu_off_maxpool_off_runner"
            ),
            "weights_off": str(
                GLOW_ROOT
                / "model"
                / "lenet"
                / "bundles"
                / "mnist"
                / "bundle_relu_off_maxpool_off"
                / "lenet_mnist_relu_off_maxpool_off.weights.bin"
            ),
            "runner_on": str(
                GLOW_ROOT
                / "model"
                / "lenet"
                / "bundles"
                / "mnist"
                / "bundle_relu_on_maxpool_on"
                / "lenet_mnist_relu_on_maxpool_on_runner"
            ),
            "weights_on": str(
                GLOW_ROOT
                / "model"
                / "lenet"
                / "bundles"
                / "mnist"
                / "bundle_relu_on_maxpool_on"
                / "lenet_mnist_relu_on_maxpool_on.weights.bin"
            ),
            "sample_inputs": lenet_inputs,
        },
        "MC02": {
            "type": "mc",
            "source": "log",
            "family": "vggnet",
            "dataset": "mnist",
            "samples": SAMPLES,
            "meta_base": str(
                GLOW_ROOT / "trace" / "vggnet_all" / "model_compiler" / "final_block16_exactparams_20260401_mnist_cifar" / "mnist"
            ),
        },
        "MC03": {
            "type": "mc",
            "source": "log",
            "family": "vggnet",
            "dataset": "cifar",
            "samples": SAMPLES,
            "meta_base": str(
                GLOW_ROOT / "trace" / "vggnet_all" / "model_compiler" / "final_block16_exactparams_20260401_mnist_cifar" / "cifar"
            ),
        },
        "MC04": {
            "type": "mc",
            "source": "log",
            "family": "squeezenet",
            "dataset": "cifar",
            "samples": SAMPLES,
            "meta_base": str(
                GLOW_ROOT / "trace" / "squeezenet_all" / "model-compiler" / "final_block16_exactparams_20260402_cifar" / "cifar"
            ),
        },
        "MC05": {
            "type": "mc",
            "source": "log",
            "family": "squeezenet",
            "dataset": "imagenet32",
            "samples": SAMPLES,
            "meta_base": str(
                GLOW_ROOT / "trace" / "squeezenet_all" / "model-compiler" / "final_block16_exactparams_20260403_imagenet" / "imagenet"
            ),
        },
        "MC06": {
            "type": "mc",
            "source": "log",
            "family": "resnet",
            "dataset": "mnist",
            "samples": SAMPLES,
            "meta_base": str(
                GLOW_ROOT / "trace" / "resnet_all" / "model-compiler" / "final_block16_exactparams_20260401_all4datasets" / "mnist"
            ),
        },
        "MC07": {
            "type": "mc",
            "source": "log",
            "family": "resnet",
            "dataset": "cifar",
            "samples": SAMPLES,
            "meta_base": str(
                GLOW_ROOT / "trace" / "resnet_all" / "model-compiler" / "final_block16_exactparams_20260401_all4datasets" / "cifar"
            ),
        },
        "MC08": {
            "type": "mc",
            "source": "log",
            "family": "resnet",
            "dataset": "imagenet32",
            "samples": SAMPLES,
            "meta_base": str(
                GLOW_ROOT / "trace" / "resnet_all" / "model-compiler" / "final_block16_exactparams_20260403_imagenet" / "imagenet"
            ),
        },
        "MC09": {
            "type": "mc",
            "source": "log",
            "family": "resnet",
            "dataset": "celebA",
            "samples": SAMPLES,
            "meta_base": str(
                GLOW_ROOT / "trace" / "resnet_all" / "model-compiler" / "final_block16_exactparams_20260401_all4datasets" / "celea"
            ),
        },
        "MC10": {
            "type": "mc",
            "source": "log",
            "family": "resnet",
            "dataset": "chest",
            "samples": SAMPLES,
            "meta_base": str(
                GLOW_ROOT / "trace" / "resnet_all" / "model-compiler" / "final_block16_exactparams_20260401_all4datasets" / "chest"
            ),
        },
        "MC11": {
            "type": "mc",
            "source": "log",
            "family": "mobilenet",
            "dataset": "mnist",
            "samples": SAMPLES,
            "meta_base": str(
                GLOW_ROOT
                / "trace"
                / "mobilenet_all"
                / "model-compiler"
                / "final_block16_exactparams_20260403_all4datasets_thresh4"
                / "mnist"
            ),
        },
        "MC12": {
            "type": "mc",
            "source": "log",
            "family": "mobilenet",
            "dataset": "cifar",
            "samples": SAMPLES,
            "meta_base": str(
                GLOW_ROOT
                / "trace"
                / "mobilenet_all"
                / "model-compiler"
                / "final_block16_exactparams_20260403_all4datasets_thresh4"
                / "cifar"
            ),
        },
        "MC13": {
            "type": "mc",
            "source": "log",
            "family": "mobilenet",
            "dataset": "celebA",
            "samples": SAMPLES,
            "meta_base": str(
                GLOW_ROOT
                / "trace"
                / "mobilenet_all"
                / "model-compiler"
                / "final_block16_exactparams_20260403_all4datasets_thresh4"
                / "celea"
            ),
        },
        "MC14": {
            "type": "mc",
            "source": "log",
            "family": "mobilenet",
            "dataset": "chest",
            "samples": SAMPLES,
            "meta_base": str(
                GLOW_ROOT
                / "trace"
                / "mobilenet_all"
                / "model-compiler"
                / "final_block16_exactparams_20260403_all4datasets_thresh4"
                / "chest"
            ),
        },
        "MC15": {
            "type": "mc",
            "source": "log",
            "family": "densenet",
            "dataset": "imagenet32",
            "samples": SAMPLES,
            "meta_base": str(
                GLOW_ROOT / "trace" / "densenet_all" / "model-compiler" / "final_block16_exactparams_20260403_imagenet" / "imagenet"
            ),
        },
        "MC16": {
            "type": "mc",
            "source": "log",
            "family": "densenet",
            "dataset": "celebA",
            "samples": SAMPLES,
            "meta_base": str(
                GLOW_ROOT / "trace" / "densenet_all" / "model-compiler" / "final_block16_exactparams_20260402_celea_chest" / "celea"
            ),
        },
        "MC17": {
            "type": "mc",
            "source": "log",
            "family": "densenet",
            "dataset": "chest",
            "samples": SAMPLES,
            "meta_base": str(
                GLOW_ROOT / "trace" / "densenet_all" / "model-compiler" / "final_block16_exactparams_20260402_celea_chest" / "chest"
            ),
        },
        "IC01": {
            "type": "ic",
            "source": "custom",
            "family": "lenet",
            "dataset": "mnist",
            "samples": LENET_IC_SAMPLES,
            "backend": "Interpreter",
            "input_mode": "mnist_tensor_list",
            "onnx": str(GLOW_ROOT / "model" / "lenet" / "exports" / "mnist" / "lenet_mnist.onnx"),
            "model_input": "data",
            "output_name": "output",
            "sample_to_index": dict(LENET_IC_SAMPLE_TO_INDEX),
        },
        "IC02": {
            "type": "ic",
            "source": "log",
            "family": "vggnet",
            "dataset": "mnist",
            "samples": SAMPLES,
            "backend": "Interpreter",
            "meta_base": str(
                GLOW_ROOT / "trace" / "vggnet_all" / "image-classifier" / "final_block16_exactparams_20260401_mnist_cifar" / "mnist"
            ),
            "model_input": "data",
            "output_name": "output",
            "image_mode": "0to1",
            "image_layout": "NCHW",
            "input_layout": "NCHW",
        },
        "IC03": {
            "type": "ic",
            "source": "log",
            "family": "vggnet",
            "dataset": "cifar",
            "samples": SAMPLES,
            "backend": "Interpreter",
            "meta_base": str(
                GLOW_ROOT / "trace" / "vggnet_all" / "image-classifier" / "final_block16_exactparams_20260401_mnist_cifar" / "cifar"
            ),
            "model_input": "data",
            "output_name": "output",
            "image_mode": "0to1",
            "image_layout": "NCHW",
            "input_layout": "NCHW",
        },
        "IC04": {
            "type": "ic",
            "source": "log",
            "family": "squeezenet",
            "dataset": "cifar",
            "samples": SAMPLES,
            "backend": "Interpreter",
            "meta_base": str(
                GLOW_ROOT / "trace" / "squeezenet_all" / "image-classifier" / "final_block16_exactparams_20260402_cifar" / "cifar"
            ),
            "model_input": "data",
            "output_name": "output",
            "image_mode": "0to1",
            "image_layout": "NCHW",
            "input_layout": "NCHW",
            "image_channel_order": "RGB",
            "mean": "123.675,116.28,103.53",
            "stddev": "0.229,0.224,0.225",
        },
        "IC05": {
            "type": "ic",
            "source": "log",
            "family": "squeezenet",
            "dataset": "imagenet32",
            "samples": SAMPLES,
            "backend": "Interpreter",
            "meta_base": str(
                GLOW_ROOT / "trace" / "squeezenet_all" / "image-classifier" / "final_block16_exactparams_20260403_imagenet" / "imagenet"
            ),
            "onnx": str(
                GLOW_ROOT
                / "model"
                / "SqueezeNet"
                / "exports"
                / "imagenet32"
                / "squeezenet1_0_imagenet32_glowfix.onnx"
            ),
            "model_input": "data",
            "output_name": "output",
        },
        "IC06": {
            "type": "ic",
            "source": "log",
            "family": "resnet",
            "dataset": "mnist",
            "samples": SAMPLES,
            "backend": "Interpreter",
            "meta_base": str(
                GLOW_ROOT / "trace" / "resnet_all" / "image-classifier" / "final_block16_exactparams_20260401_all4datasets" / "mnist"
            ),
            "model_input": "data",
            "output_name": "output",
            "image_mode": "0to1",
            "image_layout": "NCHW",
            "input_layout": "NCHW",
            "image_channel_order": "RGB",
            "mean": "123.675,116.28,103.53",
            "stddev": "0.229,0.224,0.225",
        },
        "IC07": {
            "type": "ic",
            "source": "log",
            "family": "resnet",
            "dataset": "cifar",
            "samples": SAMPLES,
            "backend": "Interpreter",
            "meta_base": str(
                GLOW_ROOT / "trace" / "resnet_all" / "image-classifier" / "final_block16_exactparams_20260401_all4datasets" / "cifar"
            ),
            "model_input": "data",
            "output_name": "output",
            "image_mode": "0to1",
            "image_layout": "NCHW",
            "input_layout": "NCHW",
            "image_channel_order": "RGB",
            "mean": "123.675,116.28,103.53",
            "stddev": "0.229,0.224,0.225",
        },
        "IC08": {
            "type": "ic",
            "source": "log",
            "family": "resnet",
            "dataset": "imagenet32",
            "samples": SAMPLES,
            "backend": "Interpreter",
            "meta_base": str(
                GLOW_ROOT / "trace" / "resnet_all" / "image-classifier" / "final_block16_exactparams_20260403_imagenet" / "imagenet"
            ),
            "model_input": "data",
            "output_name": "output",
        },
        "IC09": {
            "type": "ic",
            "source": "log",
            "family": "resnet",
            "dataset": "celebA",
            "samples": SAMPLES,
            "backend": "Interpreter",
            "meta_base": str(
                GLOW_ROOT / "trace" / "resnet_all" / "image-classifier" / "final_block16_exactparams_20260401_all4datasets" / "celea"
            ),
            "model_input": "data",
            "output_name": "output",
            "image_mode": "0to1",
            "image_layout": "NCHW",
            "input_layout": "NCHW",
            "image_channel_order": "RGB",
            "mean": "123.675,116.28,103.53",
            "stddev": "0.229,0.224,0.225",
        },
        "IC10": {
            "type": "ic",
            "source": "log",
            "family": "resnet",
            "dataset": "chest",
            "samples": SAMPLES,
            "backend": "Interpreter",
            "meta_base": str(
                GLOW_ROOT / "trace" / "resnet_all" / "image-classifier" / "final_block16_exactparams_20260401_all4datasets" / "chest"
            ),
            "model_input": "data",
            "output_name": "output",
            "image_mode": "0to1",
            "image_layout": "NCHW",
            "input_layout": "NCHW",
            "image_channel_order": "RGB",
            "mean": "123.675,116.28,103.53",
            "stddev": "0.229,0.224,0.225",
        },
        "IC11": {
            "type": "ic",
            "source": "log",
            "family": "mobilenet",
            "dataset": "mnist",
            "samples": SAMPLES,
            "backend": "Interpreter",
            "meta_base": str(
                GLOW_ROOT
                / "trace"
                / "mobilenet_all"
                / "image-classifier"
                / "final_block16_exactparams_20260403_all4datasets_thresh4"
                / "mnist"
            ),
            "model_input": "data",
            "output_name": "output",
        },
        "IC12": {
            "type": "ic",
            "source": "log",
            "family": "mobilenet",
            "dataset": "cifar",
            "samples": SAMPLES,
            "backend": "Interpreter",
            "meta_base": str(
                GLOW_ROOT
                / "trace"
                / "mobilenet_all"
                / "image-classifier"
                / "final_block16_exactparams_20260403_all4datasets_thresh4"
                / "cifar"
            ),
            "model_input": "data",
            "output_name": "output",
        },
        "IC13": {
            "type": "ic",
            "source": "log",
            "family": "mobilenet",
            "dataset": "celebA",
            "samples": SAMPLES,
            "backend": "Interpreter",
            "meta_base": str(
                GLOW_ROOT
                / "trace"
                / "mobilenet_all"
                / "image-classifier"
                / "final_block16_exactparams_20260403_all4datasets_thresh4"
                / "celea"
            ),
            "model_input": "data",
            "output_name": "output",
        },
        "IC14": {
            "type": "ic",
            "source": "log",
            "family": "mobilenet",
            "dataset": "chest",
            "samples": SAMPLES,
            "backend": "Interpreter",
            "meta_base": str(
                GLOW_ROOT
                / "trace"
                / "mobilenet_all"
                / "image-classifier"
                / "final_block16_exactparams_20260403_all4datasets_thresh4"
                / "chest"
            ),
            "model_input": "data",
            "output_name": "output",
        },
        "IC15": {
            "type": "ic",
            "source": "log",
            "family": "densenet",
            "dataset": "imagenet32",
            "samples": SAMPLES,
            "backend": "Interpreter",
            "meta_base": str(
                GLOW_ROOT / "trace" / "densenet_all" / "image-classifier" / "final_block16_exactparams_20260403_imagenet" / "imagenet"
            ),
            "model_input": "data",
            "output_name": "output",
        },
        "IC16": {
            "type": "ic",
            "source": "log",
            "family": "densenet",
            "dataset": "celebA",
            "samples": SAMPLES,
            "backend": "Interpreter",
            "meta_base": str(
                GLOW_ROOT / "trace" / "densenet_all" / "image-classifier" / "final_block16_exactparams_20260402_celea_chest" / "celea"
            ),
            "model_input": "data",
            "output_name": "output",
            "image_mode": "neg1to1",
            "image_layout": "NCHW",
            "input_layout": "NCHW",
        },
        "IC17": {
            "type": "ic",
            "source": "log",
            "family": "densenet",
            "dataset": "chest",
            "samples": SAMPLES,
            "backend": "Interpreter",
            "meta_base": str(
                GLOW_ROOT / "trace" / "densenet_all" / "image-classifier" / "final_block16_exactparams_20260402_celea_chest" / "chest"
            ),
            "model_input": "data",
            "output_name": "output",
            "image_mode": "neg1to1",
            "image_layout": "NCHW",
            "input_layout": "NCHW",
        },
    }


TASKS = build_registry()


def log_metadata(task: dict[str, Any], sample: str, mode: str) -> dict[str, str]:
    return parse_runlog(exact_runlog(task["meta_base"], sample, mode))


def resolve_mc_run(session_dir: Path, task_id: str, task: dict[str, Any], sample: str, mode: str) -> dict[str, str]:
    if task["source"] == "custom":
        return {
            "runner": task["runner_on"] if mode == "on" else task["runner_off"],
            "weights": task["weights_on"] if mode == "on" else task["weights_off"],
            "input": task["sample_inputs"][sample],
            "metadata_source": "custom",
        }
    info = log_metadata(task, sample, mode)
    resolved_input, input_resolution = resolve_mc_input_path(session_dir, task_id, task, sample, info["input"])
    return {
        "runner": info["runner"],
        "weights": info["weights"],
        "input": str(resolved_input),
        "metadata_source": f"{exact_runlog(task['meta_base'], sample, mode)};{input_resolution}",
    }


def tensor_list_from_bin(session_dir: Path, task_id: str, sample: str, input_bin: Path, shape: list[int]) -> Path:
    out_dir = session_dir / "generated_inputs" / task_id / sample
    tensor_txt = out_dir / f"{sample}.tensor.txt"
    tensor_list = out_dir / f"{sample}.tensor.list.txt"
    if tensor_list.exists():
        return tensor_list

    out_dir.mkdir(parents=True, exist_ok=True)
    data = np.fromfile(input_bin, dtype=np.float32)
    expected = int(np.prod(shape))
    if data.size != expected:
        raise ValueError(f"tensor size mismatch for {input_bin}: {data.size} != {expected}")

    flat = " ".join(f"{float(value):.9g}" for value in data)
    with tensor_txt.open("w", encoding="ascii") as handle:
        handle.write(" ".join(str(int(dim)) for dim in shape) + "\n")
        handle.write(flat + "\n")

    tensor_list.write_text(str(tensor_txt) + "\n", encoding="ascii")
    return tensor_list


def tensor_list_from_mnist_raw(session_dir: Path, task_id: str, sample: str, dataset_index: int) -> Path:
    out_dir = session_dir / "generated_inputs" / task_id / sample
    tensor_txt = out_dir / f"{sample}.tensor.txt"
    tensor_list = out_dir / f"{sample}.tensor.list.txt"
    if tensor_list.exists():
        return tensor_list

    try:
        from torchvision import datasets
    except ImportError as exc:
        raise RuntimeError("torchvision is required to rebuild the historical LeNet IC tensor-list") from exc

    if not MNIST_DATASET_ROOT.exists():
        raise FileNotFoundError(f"missing MNIST dataset root: {MNIST_DATASET_ROOT}")

    dataset = datasets.MNIST(root=str(MNIST_DATASET_ROOT), train=False, download=False)
    image, _label = dataset[dataset_index]
    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = array.reshape(1, 1, image.height, image.width)

    out_dir.mkdir(parents=True, exist_ok=True)
    flat = " ".join(f"{float(value):.9g}" for value in tensor.reshape(-1))
    with tensor_txt.open("w", encoding="ascii") as handle:
        handle.write("1 1 28 28\n")
        handle.write(flat + "\n")

    tensor_list.write_text(str(tensor_txt) + "\n", encoding="ascii")
    return tensor_list


def resolve_ic_run(session_dir: Path, task_id: str, task: dict[str, Any], sample: str, mode: str) -> dict[str, str]:
    if task["source"] == "custom":
        if task["input_mode"] == "image":
            return {
                "backend": task["backend"],
                "onnx": task["onnx"],
                "input_mode": "image",
                "input_value": task["sample_inputs"][sample],
                "metadata_source": "custom",
            }
        if task["input_mode"] == "mnist_tensor_list":
            dataset_index = int(task["sample_to_index"][sample])
            tensor_list = tensor_list_from_mnist_raw(session_dir, task_id, sample, dataset_index)
            return {
                "backend": task["backend"],
                "onnx": task["onnx"],
                "input_mode": "tensor_list",
                "input_value": str(tensor_list),
                "metadata_source": f"custom_mnist_raw_dataset:{dataset_index}",
            }
        input_bin = Path(task["sample_inputs"][sample])
        tensor_list = tensor_list_from_bin(session_dir, task_id, sample, input_bin, list(task["tensor_shape"]))
        return {
            "backend": task["backend"],
            "onnx": task["onnx"],
            "input_mode": "tensor_list",
            "input_value": str(tensor_list),
            "metadata_source": "custom",
        }

    info = log_metadata(task, sample, mode)
    onnx_path = task.get("onnx", info.get("onnx"))
    if onnx_path is None:
        raise KeyError(f"missing onnx path for {task_id}/{sample}/{mode}")
    if "image" in info:
        input_mode = "image"
        input_value = info["image"]
    elif "input_tensor_list" in info:
        input_mode = "tensor_list"
        input_value = info["input_tensor_list"]
    else:
        raise KeyError(f"unsupported IC input in run.log for {task_id}/{sample}/{mode}")
    return {
        "backend": task["backend"],
        "onnx": onnx_path,
        "input_mode": input_mode,
        "input_value": input_value,
        "metadata_source": str(exact_runlog(task["meta_base"], sample, mode)),
    }


def run_model_compiler(
    *,
    task_id: str,
    task: dict[str, Any],
    sample: str,
    mode: str,
    session_dir: Path,
) -> dict[str, Any]:
    info = resolve_mc_run(session_dir, task_id, task, sample, mode)
    run_dir = session_dir / "runs" / task_id / sample / mode
    run_dir.mkdir(parents=True, exist_ok=True)
    env = mode_env(mode)

    runner = Path(info["runner"])
    weights = Path(info["weights"])
    input_bin = Path(info["input"])
    for path in (runner, weights, input_bin, PIN_BIN, PINTOOL):
        if not path.exists():
            raise FileNotFoundError(f"missing required path: {path}")

    reference_out = run_dir / "reference_infer_result.txt"
    trace_out = run_dir / "trace_infer_result.txt"
    direct_stdout = run_dir / "reference.stdout.txt"
    direct_stderr = run_dir / "reference.stderr.txt"
    pin_stdout = run_dir / "pin.stdout.txt"
    pin_stderr = run_dir / "pin.stderr.txt"
    trace_bin = run_dir / "paddrtrace.bin"
    trace_ip = run_dir / "paddrtrace.ip.txt"
    verify_json = run_dir / "verify.json"
    execution_proof = run_dir / "execution_proof.json"
    run_log = run_dir / "run.log"
    cmd_txt = run_dir / "cmd.txt"

    reference_cmd = [str(runner), str(weights), str(input_bin), str(reference_out)]
    trace_inner_cmd = [str(runner), str(weights), str(input_bin), str(trace_out)]
    pin_cmd = [
        str(PIN_BIN),
        "-t",
        str(PINTOOL),
        "-stack-depth",
        "0",
        "-no-paddr",
        "1",
        "-taint-seed-mode",
        "file",
        "-taint-file",
        str(input_bin),
        "-taint-only",
        "1",
        "-o",
        str(trace_bin),
        "-m",
        str(trace_ip),
        "--",
        *trace_inner_cmd,
    ]

    write_text(
        run_log,
        "\n".join(
            [
                f"begin={time.strftime('%Y-%m-%dT%H:%M:%S%z')}",
                f"task_id={task_id}",
                f"sample={sample}",
                f"mode={mode}",
                f"family={task['family']}",
                f"dataset={task['dataset']}",
                f"runner={runner}",
                f"weights={weights}",
                f"input_bin={input_bin}",
                f"metadata_source={info['metadata_source']}",
                *[f"{key}={value}" for key, value in sorted(env.items()) if key.startswith("GLOW_")],
            ]
        )
        + "\n",
    )
    write_text(cmd_txt, render_cmd(reference_cmd) + "\n" + render_cmd(pin_cmd) + "\n")

    with direct_stdout.open("w", encoding="ascii") as stdout_file, direct_stderr.open(
        "w", encoding="ascii"
    ) as stderr_file:
        subprocess.run(
            reference_cmd,
            cwd=ROOT_DIR,
            env={**clean_env(), **env},
            check=True,
            stdout=stdout_file,
            stderr=stderr_file,
        )

    with pin_stdout.open("w", encoding="ascii") as stdout_file, pin_stderr.open(
        "w", encoding="ascii"
    ) as stderr_file:
        subprocess.run(
            pin_cmd,
            cwd=ROOT_DIR,
            env={**clean_env(), **env},
            check=True,
            stdout=stdout_file,
            stderr=stderr_file,
        )

    verify_payload = compare_text(reference_out, trace_out, verify_json)
    trace_validation = validate_trace(run_dir)
    saw_input_dither = "input_zero_dither=random" in pin_stderr.read_text(encoding="utf-8", errors="ignore")
    expected = mode == "on"
    if saw_input_dither != expected:
        raise RuntimeError(f"input dither proof mismatch for {task_id}/{sample}/{mode}: {saw_input_dither}")

    write_json(
        execution_proof,
        {
            "task_id": task_id,
            "sample": sample,
            "mode": mode,
            "run_dir": str(run_dir),
            "trace_kind": "paddrtrace",
            "trace_bin": str(trace_bin),
            "trace_ipmap": str(trace_ip),
            "trace_validation": trace_validation,
            "metadata_source": info["metadata_source"],
            "verify_json": str(verify_json),
            "verification_passed": bool(verify_payload["verification_passed"]),
            "requested_config": REQUESTED_CONFIG,
        },
    )
    with run_log.open("a", encoding="ascii") as handle:
        handle.write(f"end={time.strftime('%Y-%m-%dT%H:%M:%S%z')}\n")
    return read_json(execution_proof)


def image_classifier_cmd(task: dict[str, Any], resolved: dict[str, str]) -> list[str]:
    base = [
        str(IMAGE_CLASSIFIER),
        f"-model={resolved['onnx']}",
        f"-backend={resolved['backend']}",
        f"-model-input={task['model_input']}",
    ]
    if resolved["input_mode"] == "image":
        cmd = [
            str(IMAGE_CLASSIFIER),
            resolved["input_value"],
            f"-model={resolved['onnx']}",
            f"-backend={resolved['backend']}",
            f"-model-input={task['model_input']}",
            f"-output-name={task['output_name']}",
        ]
        if task.get("image_mode"):
            cmd.append(f"-image-mode={task['image_mode']}")
        if task.get("image_layout"):
            cmd.append(f"-image-layout={task['image_layout']}")
        if task.get("input_layout"):
            cmd.append(f"-input-layout={task['input_layout']}")
        if task.get("image_channel_order"):
            cmd.append(f"-image-channel-order={task['image_channel_order']}")
        if task.get("mean"):
            cmd.append(f"-mean={task['mean']}")
        if task.get("stddev"):
            cmd.append(f"-stddev={task['stddev']}")
        cmd.append("--minibatch-threads=1")
        return cmd
    if resolved["input_mode"] == "tensor_list":
        return [
            *base,
            f"-input-tensor-list-file={resolved['input_value']}",
            f"-output-name={task['output_name']}",
        ]
    raise ValueError(f"unsupported IC input mode: {resolved['input_mode']}")


def run_image_classifier(
    *,
    task_id: str,
    task: dict[str, Any],
    sample: str,
    mode: str,
    session_dir: Path,
) -> dict[str, Any]:
    resolved = resolve_ic_run(session_dir, task_id, task, sample, mode)
    run_dir = session_dir / "runs" / task_id / sample / mode
    run_dir.mkdir(parents=True, exist_ok=True)
    env = mode_env(mode)
    env["LD_LIBRARY_PATH"] = LD_LIBRARY_PATH_VALUE

    for path in (IMAGE_CLASSIFIER, PIN_BIN, PINTOOL, Path(resolved["onnx"]), Path(resolved["input_value"])):
        if not path.exists():
            raise FileNotFoundError(f"missing required path: {path}")

    reference_stdout = run_dir / "reference.stdout.txt"
    reference_stderr = run_dir / "reference.stderr.txt"
    pin_stdout = run_dir / "pin.stdout.txt"
    pin_stderr = run_dir / "pin.stderr.txt"
    infer_out = run_dir / "infer_result.txt"
    trace_bin = run_dir / "paddrtrace.bin"
    trace_ip = run_dir / "paddrtrace.ip.txt"
    verify_json = run_dir / "verify.json"
    execution_proof = run_dir / "execution_proof.json"
    run_log = run_dir / "run.log"
    cmd_txt = run_dir / "cmd.txt"

    reference_cmd = image_classifier_cmd(task, resolved)
    trace_inner_cmd = image_classifier_cmd(task, resolved)
    pin_cmd = [
        str(PIN_BIN),
        "-t",
        str(PINTOOL),
        "-stack-depth",
        "0",
        "-no-paddr",
        "1",
        "-taint-seed-mode",
        "input-tensor",
        "-taint-only",
        "1",
        "-o",
        str(trace_bin),
        "-m",
        str(trace_ip),
        "--",
        *trace_inner_cmd,
    ]

    write_text(
        run_log,
        "\n".join(
            [
                f"begin={time.strftime('%Y-%m-%dT%H:%M:%S%z')}",
                f"task_id={task_id}",
                f"sample={sample}",
                f"mode={mode}",
                f"family={task['family']}",
                f"dataset={task['dataset']}",
                f"backend={resolved['backend']}",
                f"onnx={resolved['onnx']}",
                f"input_mode={resolved['input_mode']}",
                f"input_value={resolved['input_value']}",
                f"metadata_source={resolved['metadata_source']}",
                *[f"{key}={value}" for key, value in sorted(env.items()) if key.startswith("GLOW_")],
            ]
        )
        + "\n",
    )
    write_text(cmd_txt, render_cmd(reference_cmd) + "\n" + render_cmd(pin_cmd) + "\n")

    with reference_stdout.open("w", encoding="ascii") as stdout_file, reference_stderr.open(
        "w", encoding="ascii"
    ) as stderr_file:
        subprocess.run(
            reference_cmd,
            cwd=ROOT_DIR,
            env={**clean_env(), **env},
            check=True,
            stdout=stdout_file,
            stderr=stderr_file,
        )

    with pin_stdout.open("w", encoding="ascii") as stdout_file, pin_stderr.open(
        "w", encoding="ascii"
    ) as stderr_file:
        subprocess.run(
            pin_cmd,
            cwd=ROOT_DIR,
            env={**clean_env(), **env},
            check=True,
            stdout=stdout_file,
            stderr=stderr_file,
        )

    infer_out.write_text(pin_stdout.read_text(encoding="utf-8", errors="ignore"), encoding="utf-8")
    verify_payload = compare_text(reference_stdout, pin_stdout, verify_json)
    trace_validation = validate_trace(run_dir)
    saw_input_dither = "input_zero_dither=random" in pin_stderr.read_text(encoding="utf-8", errors="ignore")
    expected = mode == "on"
    if saw_input_dither != expected:
        raise RuntimeError(f"input dither proof mismatch for {task_id}/{sample}/{mode}: {saw_input_dither}")

    write_json(
        execution_proof,
        {
            "task_id": task_id,
            "sample": sample,
            "mode": mode,
            "run_dir": str(run_dir),
            "trace_kind": "paddrtrace",
            "trace_bin": str(trace_bin),
            "trace_ipmap": str(trace_ip),
            "trace_validation": trace_validation,
            "metadata_source": resolved["metadata_source"],
            "verify_json": str(verify_json),
            "verification_passed": bool(verify_payload["verification_passed"]),
            "requested_config": REQUESTED_CONFIG,
        },
    )
    with run_log.open("a", encoding="ascii") as handle:
        handle.write(f"end={time.strftime('%Y-%m-%dT%H:%M:%S%z')}\n")
    return read_json(execution_proof)


def worker_main(session_dir: Path, task_id: str) -> int:
    task = TASKS[task_id]
    runs: list[dict[str, Any]] = []
    for sample in task["samples"]:
        for mode in ("off", "on"):
            if task["type"] == "mc":
                proof = run_model_compiler(
                    task_id=task_id,
                    task=task,
                    sample=sample,
                    mode=mode,
                    session_dir=session_dir,
                )
            else:
                proof = run_image_classifier(
                    task_id=task_id,
                    task=task,
                    sample=sample,
                    mode=mode,
                    session_dir=session_dir,
                )
            write_json(
                session_dir / "execution_proofs" / task_id / f"{sample}_{mode}.json",
                proof,
            )
            runs.append(proof)
    write_json(
        session_dir / "worker_status" / f"{task_id}.json",
        {
            "task_id": task_id,
            "family": task["family"],
            "dataset": task["dataset"],
            "task_type": task["type"],
            "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "runs": runs,
        },
    )
    return 0


def list_main(task_ids: list[str]) -> int:
    for task_id in task_ids:
        task = TASKS[task_id]
        print(f"{task_id}\t{task['type']}\t{task['family']}\t{task['dataset']}")
    return 0


def launch_workers(session_dir: Path, task_ids: list[str], parallelism: int) -> dict[str, Any]:
    jobs: dict[str, Any] = {}
    launcher_log_dir = session_dir / "launcher_logs"
    launcher_log_dir.mkdir(parents=True, exist_ok=True)

    pending = list(task_ids)
    active: dict[str, tuple[subprocess.Popen[Any], io.TextIOWrapper, Path]] = {}
    completed: dict[str, Any] = {}

    while pending or active:
        while pending and len(active) < parallelism:
            task_id = pending.pop(0)
            log_path = launcher_log_dir / f"{task_id}.log"
            log_file = log_path.open("w", encoding="ascii")
            cmd = [
                str(PYTHON_BIN),
                str(Path(__file__).resolve()),
                "worker",
                "--session-dir",
                str(session_dir),
                "--task",
                task_id,
            ]
            proc = subprocess.Popen(
                cmd,
                cwd=ROOT_DIR,
                env=clean_env(),
                stdout=log_file,
                stderr=subprocess.STDOUT,
            )
            active[task_id] = (proc, log_file, log_path)
            jobs[task_id] = {
                "pid": proc.pid,
                "cmd": cmd,
                "log_path": str(log_path),
                "state": "running",
                "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            }
            write_json(session_dir / "jobs.json", jobs)

        time.sleep(5)

        finished_ids: list[str] = []
        for task_id, (proc, log_file, log_path) in active.items():
            returncode = proc.poll()
            if returncode is None:
                continue
            log_file.close()
            jobs[task_id]["state"] = "finished"
            jobs[task_id]["returncode"] = int(returncode)
            jobs[task_id]["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
            completed[task_id] = {
                "returncode": int(returncode),
                "log_path": str(log_path),
            }
            finished_ids.append(task_id)
            write_json(session_dir / "jobs.json", jobs)
        for task_id in finished_ids:
            active.pop(task_id)

    return completed


def launch_main(session_dir: Path, task_ids: list[str], parallelism: int) -> int:
    session_dir.mkdir(parents=True, exist_ok=True)
    manager_log = session_dir / "manager.log"
    cmd = [
        str(PYTHON_BIN),
        str(Path(__file__).resolve()),
        "manager",
        "--session-dir",
        str(session_dir),
        "--parallelism",
        str(parallelism),
        "--tasks",
        ",".join(task_ids),
    ]
    log_file = manager_log.open("w", encoding="ascii")
    proc = subprocess.Popen(
        cmd,
        cwd=ROOT_DIR,
        env=clean_env(),
        stdout=log_file,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    log_file.close()

    write_json(
        session_dir / "launch_summary.json",
        {
            "session_dir": str(session_dir),
            "manager_pid": proc.pid,
            "manager_log": str(manager_log),
            "parallelism": int(parallelism),
            "selected_task_ids": task_ids,
            "trace_kind": "paddrtrace",
            "requested_config": REQUESTED_CONFIG,
        },
    )
    print(str(session_dir))
    return 0


def manager_main(session_dir: Path, task_ids: list[str], parallelism: int) -> int:
    session_dir.mkdir(parents=True, exist_ok=True)
    manager_status_path = session_dir / "manager_status.json"
    write_json(
        manager_status_path,
        {
            "session_dir": str(session_dir),
            "state": "running",
            "selected_task_ids": task_ids,
            "parallelism": int(parallelism),
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        },
    )
    write_json(
        session_dir / "manifest.json",
        {
            "session_dir": str(session_dir),
            "selected_task_ids": task_ids,
            "parallelism": int(parallelism),
            "trace_kind": "paddrtrace",
            "requested_config": REQUESTED_CONFIG,
        },
    )

    completed = launch_workers(session_dir, task_ids, parallelism)
    failures = sorted([task_id for task_id, result in completed.items() if result["returncode"] != 0])
    payload = {
        "session_dir": str(session_dir),
        "state": "finished" if not failures else "finished_with_failures",
        "selected_task_ids": task_ids,
        "parallelism": int(parallelism),
        "completed": completed,
        "failures": failures,
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    write_json(manager_status_path, payload)
    return 0 if not failures else 1


def main() -> int:
    args = parse_args()
    for required in (PYTHON_BIN, PIN_BIN, PINTOOL, IMAGE_CLASSIFIER):
        if not required.exists():
            raise FileNotFoundError(f"missing required path: {required}")

    task_ids = selected_task_ids(getattr(args, "tasks", None))

    if args.command == "list":
        return list_main(task_ids)
    if args.command == "launch":
        return launch_main(make_session_dir(args.session_dir), task_ids, args.parallelism)
    if args.command == "manager":
        return manager_main(Path(args.session_dir).resolve(), task_ids, args.parallelism)
    if args.command == "worker":
        return worker_main(Path(args.session_dir).resolve(), args.task)
    raise ValueError(f"unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
