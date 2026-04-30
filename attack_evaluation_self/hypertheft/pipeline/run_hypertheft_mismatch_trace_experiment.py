#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
TRACE_SUITE_PATH = ROOT / "trace_suite.json"
SETTINGS_MANIFEST_PATH = ROOT / "settings_manifest.json"
OFF_RANGE_PATH = ROOT / "off_reference_ratio_ranges.json"
GENERATED_ROOT = ROOT / "generated"
RESULTS_ROOT = ROOT / "results"
RUN_SUMMARY_CSV = ROOT / "run_summary.csv"
RUN_SUMMARY_JSON = ROOT / "run_summary.json"

BITCOUNT_TABLE = np.unpackbits(np.arange(256, dtype=np.uint8)[:, None], axis=1).sum(axis=1).astype(np.uint8)
LOW_ACC_FLIP_SUGGEST_THRESHOLD = 0.35


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def result_bucket_id(setting: dict) -> str:
    bucket = str(setting.get("result_id", setting["id"])).strip()
    if not bucket:
        raise RuntimeError(f"Invalid empty result bucket for setting: {setting.get('id')}")
    return bucket


def result_row_key(row: dict) -> tuple[str, str, str, str]:
    return (
        str(row["run_dir"]),
        str(row["checkpoint"]),
        str(row["task_name"]),
        str(row["trace_id"]),
    )


def parse_boolish(value) -> bool:
    if isinstance(value, bool):
        return bool(value)
    text = str(value).strip().lower()
    return text in {"1", "true", "t", "yes", "y"}


def parse_accuracy_row(row: dict[str, str]) -> dict:
    acc = float(row["acc"])
    acc_flip = float(row["acc_flip"]) if row.get("acc_flip", "") != "" else float(1.0 - acc)
    acc_best_polarity = (
        float(row["acc_best_polarity"])
        if row.get("acc_best_polarity", "") != ""
        else float(max(acc, acc_flip))
    )
    flip_suggested = (
        parse_boolish(row["flip_suggested"])
        if row.get("flip_suggested", "") != ""
        else bool(acc < LOW_ACC_FLIP_SUGGEST_THRESHOLD)
    )
    return {
        "setting_id": row["setting_id"],
        "dataset_key": row["dataset_key"],
        "run_dir": row["run_dir"],
        "checkpoint": row["checkpoint"],
        "task_name": row["task_name"],
        "trace_id": row["trace_id"],
        "trace_kind": row["trace_kind"],
        "ones_ratio_effective": float(row["ones_ratio_effective"]),
        "raw_ones_ratio": float(row["raw_ones_ratio"]),
        "n_eval_samples": int(row["n_eval_samples"]),
        "acc": acc,
        "acc_flip": acc_flip,
        "acc_best_polarity": acc_best_polarity,
        "flip_suggested": bool(flip_suggested),
    }


def load_accuracy_rows(csv_path: Path) -> list[dict]:
    if not csv_path.is_file():
        return []
    with csv_path.open("r", encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp)
        return [parse_accuracy_row(row) for row in reader]


def append_rows_dedup(existing_rows: list[dict], new_rows: list[dict]) -> list[dict]:
    if not existing_rows:
        return list(new_rows)
    if not new_rows:
        return list(existing_rows)
    new_keys = {result_row_key(row) for row in new_rows}
    kept = [row for row in existing_rows if result_row_key(row) not in new_keys]
    kept.extend(new_rows)
    return kept


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run HyperTheft mismatch-trace experiments with synthetic traces t00~t35."
    )
    parser.add_argument("--settings-manifest", default=str(SETTINGS_MANIFEST_PATH), type=str)
    parser.add_argument("--trace-suite", default=str(TRACE_SUITE_PATH), type=str)
    parser.add_argument("--settings", default="", type=str, help="Comma-separated setting ids")
    parser.add_argument("--trace-ids", default="", type=str, help="Comma-separated trace ids")
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--eval-batch-size", default=256, type=int)
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument(
        "--noise-weight-override",
        default=0.0,
        type=float,
        help="Override generator noise_weight for deterministic one-trace-one-model mapping.",
    )
    parser.add_argument("--recompute-off-ranges", action="store_true")
    return parser.parse_args()


def default_settings() -> dict:
    base = Path("/path/to/hypertheft_workspace")
    mnist_hyper = base / "lenet+mnist" / "hypertheft_lenet"
    mnist_trace = base / "lenet+mnist" / "trace"
    resnet_hyper = base / "resnet+cifar" / "hypertheft_resnet"
    resnet_trace = base / "resnet+cifar" / "trace"

    return {
        "settings": [
            {
                "id": "mnist_lenet_task8_9",
                "dataset_key": "mnist",
                "model_root": str(mnist_hyper),
                "run_dir": str(mnist_hyper / "runs" / "task8_9" / "conv32_off_convmax_full_task28_bs100_ep200_20260420"),
                "ckpt": str(mnist_hyper / "runs" / "task8_9" / "conv32_off_convmax_full_task28_bs100_ep200_20260420" / "ckpt" / "069.pt"),
                "mode": "off",
                "eval_trace_root": str(
                    mnist_trace
                    / "lenet_mnist32_aligned_conv32"
                    / "lenet_mnist32_task28_trace_test500_off_on_align_20260423"
                ),
                "unseen_tasks": ["task28_8vs9"],
            },
            {
                "id": "mnist_lenet_task5_9",
                "dataset_key": "mnist",
                "model_root": str(mnist_hyper),
                "run_dir": str(
                    mnist_hyper
                    / "runs"
                    / "task5_9"
                    / "conv32_off_convmax_full_task0to4_nocopy_bs100_ep200_20260423"
                ),
                "ckpt": str(
                    mnist_hyper
                    / "runs"
                    / "task5_9"
                    / "conv32_off_convmax_full_task0to4_nocopy_bs100_ep200_20260423"
                    / "ckpt"
                    / "069.pt"
                ),
                "mode": "off",
                "eval_trace_root": str(
                    mnist_trace
                    / "lenet_mnist32_aligned_conv32"
                    / "lenet_mnist32_task10_5to9_trace_test500_off_on_align_20260423"
                ),
                "unseen_tasks": [
                    "task10_5vs6",
                    "task11_5vs7",
                    "task12_5vs8",
                    "task13_5vs9",
                    "task14_6vs7",
                    "task15_6vs8",
                    "task16_6vs9",
                    "task17_7vs8",
                    "task18_7vs9",
                    "task19_8vs9",
                ],
            },
            {
                "id": "cifar_resnet",
                "dataset_key": "cifar",
                "model_root": str(resnet_hyper),
                "run_dir": str(
                    resnet_hyper
                    / "runs"
                    / "task28_8vs9"
                    / "resnet32_off_singleblock94c_task28_bs100_ep200_ngen8_20260424"
                ),
                "ckpt": str(
                    resnet_hyper
                    / "runs"
                    / "task28_8vs9"
                    / "resnet32_off_singleblock94c_task28_bs100_ep200_ngen8_20260424"
                    / "ckpt"
                    / "154.pt"
                ),
                "mode": "off",
                "eval_trace_root": str(
                    resnet_trace / "final_block16_exactparams_20260424_resnet32_cifar_singleblock94c"
                ),
                "unseen_tasks": ["task28_8vs9"],
            },
            {
                "id": "imagenet_resnet",
                "dataset_key": "imagenet",
                "model_root": str(resnet_hyper),
                "run_dir": str(
                    resnet_hyper
                    / "runs"
                    / "task28_8vs9"
                    / "resnet32_off_singleblock94c_task28_bs100_ep200_ngen8_20260424"
                ),
                "ckpt": str(
                    resnet_hyper
                    / "runs"
                    / "task28_8vs9"
                    / "resnet32_off_singleblock94c_task28_bs100_ep200_ngen8_20260424"
                    / "ckpt"
                    / "154.pt"
                ),
                "mode": "off",
                "eval_trace_root": str(
                    resnet_trace
                    / "final_block16_exactparams_20260424_resnet32_imagenet32_unseen10_singleblock94c"
                ),
                "unseen_tasks": [
                    "task00_102vs103",
                    "task01_110vs300",
                    "task02_170vs650",
                    "task03_250vs950",
                    "task04_401vs850",
                    "task05_450vs700",
                    "task06_500vs800",
                    "task07_550vs902",
                    "task08_600vs954",
                    "task09_703vs999",
                ],
            },
        ],
        "dataset_roots_for_off_range": {
            "mnist": "/path/to/hypertheft_workspace/lenet+mnist/trace/conv_mnist32_binary28_trace_train1000_test500_off_on_20260419",
            "cifar": "/path/to/hypertheft_workspace/resnet+cifar/trace/final_block16_exactparams_20260424_resnet32_cifar_singleblock94c",
            "imagenet": "/path/to/hypertheft_workspace/resnet+cifar/trace/final_block16_exactparams_20260424_resnet32_imagenet32_unseen10_singleblock94c",
        },
    }


def parse_csv_list(raw: str) -> list[str]:
    return [item.strip() for item in str(raw).split(",") if item.strip()]


def select_settings(settings: list[dict], only_ids: set[str] | None) -> list[dict]:
    if only_ids is None:
        return settings
    picked = [setting for setting in settings if setting["id"] in only_ids]
    if not picked:
        raise RuntimeError(f"No settings matched: {sorted(only_ids)}")
    return picked


def select_trace_items(suite: dict, only_ids: set[str] | None) -> list[dict]:
    items = suite.get("items", [])
    if only_ids is None:
        return list(items)
    picked = [item for item in items if str(item.get("id")) in only_ids]
    if not picked:
        raise RuntimeError(f"No trace ids matched: {sorted(only_ids)}")
    return picked


def sampled_positions(raw_bit_count: int, sample_stride: int) -> np.ndarray:
    stride = max(1, int(sample_stride))
    return np.arange(0, int(raw_bit_count), stride, dtype=np.int64)


def distribute_counts(total: int, buckets: int) -> list[int]:
    if buckets <= 0:
        return []
    base = total // buckets
    extra = total % buckets
    out = [base] * buckets
    for idx in range(extra):
        out[idx] += 1
    return out


def place_block_pattern(effective_len: int, ones_count: int, requested_blocks: int) -> tuple[np.ndarray, list[list[int]]]:
    bits = np.zeros((effective_len,), dtype=np.uint8)
    if ones_count <= 0:
        return bits, []
    if ones_count >= effective_len:
        bits[:] = 1
        return bits, [[0, effective_len]]

    actual_blocks = max(1, min(int(requested_blocks), int(ones_count)))
    block_sizes = distribute_counts(int(ones_count), actual_blocks)
    free_zeros = int(effective_len) - int(ones_count)
    gap_sizes = distribute_counts(free_zeros, actual_blocks + 1)

    intervals: list[list[int]] = []
    cursor = int(gap_sizes[0])
    for block_idx, block_size in enumerate(block_sizes):
        start = int(cursor)
        end = int(start + block_size)
        bits[start:end] = 1
        intervals.append([start, end])
        cursor = end + int(gap_sizes[block_idx + 1])
    return bits, intervals


def intervals_from_binary_bits(bits: np.ndarray) -> list[list[int]]:
    intervals: list[list[int]] = []
    in_block = False
    start = 0
    for idx, bit in enumerate(bits.tolist()):
        if bit and not in_block:
            start = int(idx)
            in_block = True
        elif (not bit) and in_block:
            intervals.append([start, int(idx)])
            in_block = False
    if in_block:
        intervals.append([start, int(bits.shape[0])])
    return intervals


def resolve_ratio_from_reference(item: dict, ratio_reference: dict, trace_id: str) -> float:
    low = float(ratio_reference["low"])
    high = float(ratio_reference["high"])
    alpha = float(item["range_alpha"])
    if not (0.0 <= alpha <= 1.0):
        raise ValueError(f"Invalid range_alpha for {trace_id}: {alpha}")
    return low + (high - low) * alpha


def materialize_effective_bits(item: dict, effective_len: int, ratio_reference: dict) -> tuple[np.ndarray, dict]:
    kind = str(item.get("kind", "")).strip()
    trace_id = str(item.get("id", "")).strip()
    if not trace_id:
        raise ValueError(f"Missing trace id: {item}")

    if kind == "constant":
        value = int(item.get("value", 0))
        bits = np.full((effective_len,), value, dtype=np.uint8)
        intervals = [] if value == 0 else [[0, effective_len]]
        return bits, {
            "id": trace_id,
            "kind": kind,
            "ones_ratio_requested": float(value),
            "ones_ratio_effective": float(bits.mean()),
            "ones_count_effective": int(bits.sum()),
            "block_count_requested": 1 if value == 1 else 0,
            "block_count_effective": 1 if value == 1 else 0,
            "intervals_effective": intervals,
        }

    if kind in {"block_sweep", "relative_block_sweep"}:
        if kind == "block_sweep":
            ones_ratio = float(item["ones_ratio"])
        else:
            ones_ratio = resolve_ratio_from_reference(item, ratio_reference, trace_id)
        requested_blocks = int(item["block_count"])
        ones_count = int(round(ones_ratio * effective_len))
        if ones_ratio > 0.0 and ones_count == 0:
            ones_count = 1
        if ones_ratio < 1.0 and ones_count >= effective_len:
            ones_count = effective_len - 1
        bits, intervals = place_block_pattern(effective_len, ones_count, requested_blocks)
        return bits, {
            "id": trace_id,
            "kind": kind,
            "ones_ratio_requested": float(ones_ratio),
            "ones_ratio_effective": float(bits.mean()),
            "ones_count_effective": int(bits.sum()),
            "block_count_requested": requested_blocks,
            "block_count_effective": len(intervals),
            "intervals_effective": intervals,
        }

    if kind == "one_minus_block_sweep_disjoint":
        zero_ratio = float(item["zero_ratio"])
        requested_blocks = int(item["block_count"])
        slot_index = int(item["slot_index"])
        slot_count = int(item["slot_count"])
        slot_start = int((slot_index * effective_len) // slot_count)
        slot_end = int(((slot_index + 1) * effective_len) // slot_count)
        slot_len = max(0, slot_end - slot_start)
        zero_count = int(round(zero_ratio * effective_len))
        if zero_ratio > 0.0 and zero_count == 0:
            zero_count = 1
        zero_count = min(zero_count, effective_len)
        if zero_count > slot_len:
            raise RuntimeError(
                f"Trace {trace_id}: zero_count {zero_count} exceeds slot_len {slot_len}"
            )

        bits = np.ones((effective_len,), dtype=np.uint8)
        zero_mask, intervals_local = place_block_pattern(slot_len, zero_count, requested_blocks)
        del zero_mask
        intervals: list[list[int]] = []
        for start_local, end_local in intervals_local:
            start = slot_start + int(start_local)
            end = slot_start + int(end_local)
            bits[start:end] = 0
            intervals.append([start, end])
        return bits, {
            "id": trace_id,
            "kind": kind,
            "ones_ratio_requested": float(1.0 - zero_ratio),
            "ones_ratio_effective": float(bits.mean()),
            "ones_count_effective": int(bits.sum()),
            "zero_ratio_requested": float(zero_ratio),
            "zero_count_effective": int(zero_count),
            "zero_slot_index": int(slot_index),
            "zero_slot_count": int(slot_count),
            "zero_slot_span_effective": [int(slot_start), int(slot_end)],
            "block_count_requested": requested_blocks,
            "block_count_effective": len(intervals),
            "intervals_effective": intervals,
        }

    if kind == "random_binary_range":
        ones_ratio = resolve_ratio_from_reference(item, ratio_reference, trace_id)
        seed = int(item.get("seed", 0))
        ones_count = int(round(ones_ratio * effective_len))
        if ones_ratio > 0.0 and ones_count == 0:
            ones_count = 1
        if ones_ratio < 1.0 and ones_count >= effective_len:
            ones_count = effective_len - 1
        bits = np.zeros((effective_len,), dtype=np.uint8)
        if ones_count > 0:
            rng = np.random.default_rng(seed)
            indices = rng.choice(effective_len, size=int(ones_count), replace=False)
            bits[np.asarray(indices, dtype=np.int64)] = 1
        intervals = intervals_from_binary_bits(bits)
        return bits, {
            "id": trace_id,
            "kind": kind,
            "seed": seed,
            "ones_ratio_requested": float(ones_ratio),
            "ones_ratio_effective": float(bits.mean()),
            "ones_count_effective": int(bits.sum()),
            "block_count_requested": None,
            "block_count_effective": len(intervals),
            "intervals_effective": intervals,
        }

    raise RuntimeError(f"Unsupported trace kind: {kind}")


def expand_to_raw_bits(effective_bits: np.ndarray, raw_bit_count: int, sample_stride: int) -> np.ndarray:
    raw = np.zeros((int(raw_bit_count),), dtype=np.uint8)
    positions = sampled_positions(raw_bit_count, sample_stride)
    if len(positions) != int(effective_bits.shape[0]):
        raise RuntimeError(
            f"Mismatch effective bits and sampled positions: {effective_bits.shape[0]} vs {len(positions)}"
        )
    raw[positions] = effective_bits
    return raw


def load_raw_bits_from_sample(sample_dir: Path) -> np.ndarray:
    singleblock = sample_dir / "singleblock_bits.txt"
    if singleblock.is_file():
        raw = singleblock.read_bytes().strip()
        values = np.frombuffer(raw, dtype=np.uint8)
        return (values - 48).astype(np.uint8, copy=False)

    conv = sorted(sample_dir.glob("conv*_bits.bin"))
    maxpool = sorted(sample_dir.glob("maxpool*_bits.bin"))
    if conv and maxpool:
        conv_arr = np.fromfile(conv[0], dtype=np.uint8)
        max_arr = np.fromfile(maxpool[0], dtype=np.uint8)
        conv_bits = np.unpackbits(conv_arr, bitorder="big").astype(np.uint8, copy=False)
        max_bits = np.unpackbits(max_arr, bitorder="big").astype(np.uint8, copy=False)
        return np.concatenate([conv_bits, max_bits], axis=0)

    raise FileNotFoundError(f"Cannot detect trace files under sample dir: {sample_dir}")


def count_ones_ratio_from_sample(sample_dir: Path) -> float:
    singleblock = sample_dir / "singleblock_bits.txt"
    if singleblock.is_file():
        raw = singleblock.read_bytes().strip()
        total = len(raw)
        if total == 0:
            raise RuntimeError(f"Empty singleblock trace: {singleblock}")
        ones = raw.count(b"1")
        return float(ones / total)

    conv = sorted(sample_dir.glob("conv*_bits.bin"))
    maxpool = sorted(sample_dir.glob("maxpool*_bits.bin"))
    if conv and maxpool:
        conv_bytes = np.fromfile(conv[0], dtype=np.uint8)
        max_bytes = np.fromfile(maxpool[0], dtype=np.uint8)
        ones = int(BITCOUNT_TABLE[conv_bytes].sum() + BITCOUNT_TABLE[max_bytes].sum())
        total = int(conv_bytes.size * 8 + max_bytes.size * 8)
        return float(ones / total)

    raise FileNotFoundError(f"Cannot detect trace files under sample dir: {sample_dir}")


def compute_off_reference_range(trace_root: Path) -> dict:
    ratios: list[float] = []
    for task_dir in sorted(path for path in trace_root.iterdir() if path.is_dir() and path.name.startswith("task")):
        manifest_path = task_dir / "task_dataset_manifest.json"
        if not manifest_path.is_file():
            continue
        manifest = read_json(manifest_path)
        for sample in manifest.get("train", []):
            ordinal = int(sample["ordinal"])
            sample_dir = task_dir / "off" / "train" / f"idx{ordinal:06d}"
            if not sample_dir.is_dir():
                continue
            ratios.append(count_ones_ratio_from_sample(sample_dir))
    if not ratios:
        raise RuntimeError(f"No off/train ratios collected from {trace_root}")

    ratios_np = np.asarray(ratios, dtype=np.float64)
    low = float(np.percentile(ratios_np, 5.0))
    high = float(np.percentile(ratios_np, 95.0))
    return {
        "low": low,
        "high": high,
        "source": "off_train_p05_p95",
        "sample_count": int(ratios_np.shape[0]),
        "mean": float(ratios_np.mean()),
        "min": float(ratios_np.min()),
        "max": float(ratios_np.max()),
    }


def ensure_off_ranges(dataset_roots: dict[str, str], recompute: bool) -> dict:
    if OFF_RANGE_PATH.is_file() and not recompute:
        payload = read_json(OFF_RANGE_PATH)
        dataset_ranges = payload.get("dataset_ranges", {})
        if all(key in dataset_ranges for key in dataset_roots):
            return dataset_ranges

    dataset_ranges = {}
    for key, root_str in dataset_roots.items():
        root = Path(root_str)
        print(f"[off-range] computing {key} from {root}")
        dataset_ranges[key] = compute_off_reference_range(root)
    write_json(
        OFF_RANGE_PATH,
        {
            "dataset_ranges": dataset_ranges,
        },
    )
    return dataset_ranges


def clear_models_package_cache() -> None:
    drop = [name for name in sys.modules if name == "models" or name.startswith("models.")]
    for name in drop:
        del sys.modules[name]


def import_models_module(model_root: Path, target_model: str):
    clear_models_package_cache()
    model_root_str = str(model_root)
    if model_root_str not in sys.path:
        sys.path.insert(0, model_root_str)
    return importlib.import_module(f"models.{target_model}")


def load_cfg(run_dir: Path) -> SimpleNamespace:
    config = read_json(run_dir / "config.json")
    return SimpleNamespace(**config["args"])


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_hypertheft(setting: dict, device: torch.device, noise_weight_override: float | None):
    run_dir = Path(setting["run_dir"])
    ckpt_path = Path(setting["ckpt"])
    model_root = Path(setting["model_root"])

    cfg = load_cfg(run_dir)
    cfg.device = str(device)

    target_model = str(getattr(cfg, "target_model"))
    models = import_models_module(model_root, target_model)
    if hasattr(models, "infer_ngen"):
        expected_ngen = int(models.infer_ngen(cfg))
        if int(getattr(cfg, "ngen")) != expected_ngen:
            raise RuntimeError(
                f"{setting['id']}: cfg.ngen={cfg.ngen} but models.{target_model}.infer_ngen={expected_ngen}"
            )

    hypertheft = models.HyperTheft(cfg)
    state = torch.load(ckpt_path, map_location="cpu")
    hypertheft.mixer.load_state_dict(state["mixer"], strict=True)
    for g_idx, gen in enumerate(hypertheft.generator.as_list()):
        gen.load_state_dict(state[f"G{g_idx}"], strict=True)
        if noise_weight_override is not None and hasattr(gen, "noise_weight"):
            gen.noise_weight = float(noise_weight_override)

    hypertheft.mixer.eval()
    for gen in hypertheft.generator.as_list():
        gen.eval()

    return cfg, hypertheft


def load_task_eval_data(
    trace_root: Path,
    task_name: str,
    input_shape: tuple[int, ...],
    split: str = "test",
) -> tuple[torch.Tensor, torch.Tensor]:
    task_dir = trace_root / task_name
    manifest_path = task_dir / "task_dataset_manifest.json"
    manifest = read_json(manifest_path)
    samples = manifest.get(split, [])
    images = []
    labels = []
    for sample in samples:
        ordinal = int(sample["ordinal"])
        label = int(sample["binary_label"])
        input_path = task_dir / "inputs" / split / f"idx{ordinal:06d}" / "input_nchw_f32.bin"
        if not input_path.is_file():
            continue
        data = np.fromfile(input_path, dtype=np.float32)
        if data.size != int(np.prod(input_shape)):
            raise RuntimeError(
                f"Bad input size for {task_name} {split} idx{ordinal:06d}: {data.size} vs {np.prod(input_shape)}"
            )
        images.append(data.reshape(input_shape))
        labels.append(label)

    if not images:
        raise RuntimeError(f"No evaluation samples loaded for task {task_name} on split '{split}'")
    image_tensor = torch.from_numpy(np.asarray(images, dtype=np.float32))
    label_tensor = torch.tensor(labels, dtype=torch.long)
    return image_tensor, label_tensor


def build_eval_data(setting: dict, cfg: SimpleNamespace) -> dict[str, dict]:
    input_shape = tuple(
        int(x)
        for x in getattr(
            cfg,
            "input_shape",
            [int(cfg.n_ch), int(cfg.image_size), int(cfg.image_size)],
        )
    )
    out = {}

    default_eval_split = str(setting.get("eval_split", "test"))
    eval_groups = [
        {
            "eval_trace_root": setting["eval_trace_root"],
            "unseen_tasks": setting["unseen_tasks"],
            "eval_split": default_eval_split,
        }
    ]
    extra_groups = setting.get("extra_eval_groups", [])
    if extra_groups:
        eval_groups.extend(extra_groups)

    for group in eval_groups:
        trace_root = Path(group["eval_trace_root"])
        eval_split = str(group.get("eval_split", default_eval_split))
        for task_name in group["unseen_tasks"]:
            if task_name in out:
                raise RuntimeError(
                    f"Duplicate task name across eval groups in setting {setting['id']}: {task_name}"
                )
            images, labels = load_task_eval_data(
                trace_root, task_name, input_shape=input_shape, split=eval_split
            )
            out[task_name] = {
                "images": images,
                "labels": labels,
            }
    return out


def make_trace_tensor(raw_bits: np.ndarray, cfg: SimpleNamespace) -> torch.Tensor:
    granularity = int(getattr(cfg, "granularity", 1))
    trace_len = int(getattr(cfg, "trace_len"))
    trace_c = int(getattr(cfg, "trace_c"))
    trace_w = int(getattr(cfg, "trace_w"))
    fold_trace = bool(int(getattr(cfg, "fold_trace")))

    bits = torch.from_numpy(raw_bits.astype(np.float32, copy=False))
    if granularity > 1:
        bits = bits[0:trace_len:granularity]
    if fold_trace:
        pad_len = trace_c * trace_w * trace_w
        if bits.numel() < pad_len:
            bits = torch.cat([bits, torch.zeros(pad_len - bits.numel(), dtype=bits.dtype)], dim=0)
        else:
            bits = bits[:pad_len]
        bits = bits.view(trace_c, trace_w, trace_w)
    else:
        if bits.numel() < trace_len:
            bits = torch.cat([bits, torch.zeros(trace_len - bits.numel(), dtype=bits.dtype)], dim=0)
        else:
            bits = bits[:trace_len]
    return bits


def evaluate_single_layers(
    model,
    single_layers: tuple[torch.Tensor, ...],
    images: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> float:
    correct = 0
    total = int(labels.numel())
    with torch.no_grad():
        for start in range(0, total, int(batch_size)):
            end = min(total, start + int(batch_size))
            batch_images = images[start:end].to(device, non_blocking=True)
            batch_labels = labels[start:end].to(device, non_blocking=True)
            logits = model.eval_f(single_layers, batch_images)
            pred = logits.argmax(-1)
            correct += int(pred.eq(batch_labels).sum().item())
    return float(correct / max(total, 1))


def generate_setting_traces(
    setting: dict,
    cfg: SimpleNamespace,
    suite_items: list[dict],
    ratio_reference: dict,
    output_dir: Path,
) -> dict[str, dict]:
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_bit_count = int(getattr(cfg, "trace_len"))
    sample_stride = int(getattr(cfg, "granularity", 1))
    effective_len = int(len(sampled_positions(raw_bit_count, sample_stride)))

    traces = {}
    for item in suite_items:
        trace_id = str(item["id"])
        effective_bits, meta = materialize_effective_bits(item, effective_len, ratio_reference=ratio_reference)
        raw_bits = expand_to_raw_bits(effective_bits, raw_bit_count, sample_stride)
        text_path = output_dir / f"{trace_id}.txt"
        text_path.write_text("".join("1" if int(x) else "0" for x in raw_bits.tolist()), encoding="ascii")

        traces[trace_id] = {
            "id": trace_id,
            "raw_bits": raw_bits,
            "tensor": make_trace_tensor(raw_bits, cfg),
            "meta": {
                **meta,
                "raw_bit_count": raw_bit_count,
                "sample_stride": sample_stride,
                "effective_len": effective_len,
                "raw_ones_count": int(raw_bits.sum()),
                "raw_ones_ratio": float(raw_bits.mean()),
                "trace_path": str(text_path),
            },
        }

    write_json(
        output_dir / "trace_manifest.json",
        {
            "setting_id": setting["id"],
            "dataset_key": setting["dataset_key"],
            "raw_bit_count": raw_bit_count,
            "sample_stride": sample_stride,
            "effective_len": effective_len,
            "ratio_reference": ratio_reference,
            "traces": {trace_id: payload["meta"] for trace_id, payload in traces.items()},
        },
    )
    return traces


def run_setting(
    setting: dict,
    suite_items: list[dict],
    ratio_reference: dict,
    device: torch.device,
    eval_batch_size: int,
    seed: int,
    noise_weight_override: float | None,
) -> list[dict]:
    cfg, hypertheft = load_hypertheft(setting, device=device, noise_weight_override=noise_weight_override)
    eval_data = build_eval_data(setting, cfg)

    setting_generated = GENERATED_ROOT / setting["id"]
    traces = generate_setting_traces(setting, cfg, suite_items, ratio_reference, output_dir=setting_generated)

    rows = []
    sorted_trace_ids = sorted(traces.keys())
    for trace_index, trace_id in enumerate(sorted_trace_ids):
        payload = traces[trace_id]
        trace_tensor = payload["tensor"].unsqueeze(0).to(device)

        # Keep each trace reproducible even when generator noise is enabled.
        set_global_seed(seed + trace_index)
        with torch.no_grad():
            codes = hypertheft.mixer(trace_tensor)
            params = hypertheft.generator(codes)
        single_layers = tuple(param[0] for param in params)

        for task_name, bundle in eval_data.items():
            acc = evaluate_single_layers(
                model=hypertheft,
                single_layers=single_layers,
                images=bundle["images"],
                labels=bundle["labels"],
                device=device,
                batch_size=eval_batch_size,
            )
            acc_flip = float(1.0 - acc)
            rows.append(
                {
                    "setting_id": setting["id"],
                    "dataset_key": setting["dataset_key"],
                    "run_dir": setting["run_dir"],
                    "checkpoint": Path(setting["ckpt"]).name,
                    "task_name": task_name,
                    "trace_id": trace_id,
                    "trace_kind": payload["meta"]["kind"],
                    "ones_ratio_effective": payload["meta"]["ones_ratio_effective"],
                    "raw_ones_ratio": payload["meta"]["raw_ones_ratio"],
                    "n_eval_samples": int(bundle["labels"].numel()),
                    "acc": float(acc),
                    "acc_flip": acc_flip,
                    "acc_best_polarity": float(max(acc, acc_flip)),
                    "flip_suggested": bool(acc < LOW_ACC_FLIP_SUGGEST_THRESHOLD),
                }
            )

    result_dir = RESULTS_ROOT / result_bucket_id(setting)
    result_dir.mkdir(parents=True, exist_ok=True)
    csv_path = result_dir / "trace_task_accuracy.csv"
    json_path = result_dir / "trace_task_accuracy.json"
    existing_rows = load_accuracy_rows(csv_path)
    output_rows = append_rows_dedup(existing_rows, rows)

    fieldnames = [
        "setting_id",
        "dataset_key",
        "run_dir",
        "checkpoint",
        "task_name",
        "trace_id",
        "trace_kind",
        "ones_ratio_effective",
        "raw_ones_ratio",
        "n_eval_samples",
        "acc",
        "acc_flip",
        "acc_best_polarity",
        "flip_suggested",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    write_json(json_path, {"rows": output_rows})

    # Expanded summary: keep per-binary-task rows (no cross-task averaging).
    summary_rows = [
        {
            "trace_id": row["trace_id"],
            "task_name": row["task_name"],
            "acc": float(row["acc"]),
            "n_eval_samples": int(row["n_eval_samples"]),
            "trace_kind": row["trace_kind"],
            "ones_ratio_effective": float(row["ones_ratio_effective"]),
            "raw_ones_ratio": float(row["raw_ones_ratio"]),
            "acc_flip": float(row["acc_flip"]),
            "acc_best_polarity": float(row["acc_best_polarity"]),
            "flip_suggested": bool(row["flip_suggested"]),
        }
        for row in output_rows
    ]
    with (result_dir / "trace_summary.csv").open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=[
                "trace_id",
                "task_name",
                "acc",
                "n_eval_samples",
                "trace_kind",
                "ones_ratio_effective",
                "raw_ones_ratio",
                "acc_flip",
                "acc_best_polarity",
                "flip_suggested",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)
    write_json(result_dir / "trace_summary.json", {"rows": summary_rows})

    # Wide table for quick per-task comparison: one row per trace, one column per binary task.
    trace_ids = list(dict.fromkeys(row["trace_id"] for row in output_rows))
    task_names = list(dict.fromkeys(row["task_name"] for row in output_rows))
    by_pair = {(row["trace_id"], row["task_name"]): float(row["acc"]) for row in output_rows}
    meta_by_trace = {
        row["trace_id"]: (
            row["trace_kind"],
            float(row["ones_ratio_effective"]),
            float(row["raw_ones_ratio"]),
        )
        for row in output_rows
    }
    pivot_rows = []
    for trace_id in trace_ids:
        trace_kind, ones_ratio_effective, raw_ones_ratio = meta_by_trace[trace_id]
        pivot = {
            "trace_id": trace_id,
            "trace_kind": trace_kind,
            "ones_ratio_effective": ones_ratio_effective,
            "raw_ones_ratio": raw_ones_ratio,
        }
        for task_name in task_names:
            pivot[task_name] = by_pair.get((trace_id, task_name), "")
        pivot_rows.append(pivot)

    pivot_fieldnames = [
        "trace_id",
        "trace_kind",
        "ones_ratio_effective",
        "raw_ones_ratio",
        *task_names,
    ]
    with (result_dir / "trace_task_pivot.csv").open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=pivot_fieldnames)
        writer.writeheader()
        writer.writerows(pivot_rows)
    write_json(
        result_dir / "trace_task_pivot.json",
        {
            "task_names": task_names,
            "rows": pivot_rows,
        },
    )

    print(f"[done] {setting['id']} -> {csv_path}")
    return rows


def main() -> None:
    args = parse_args()

    set_global_seed(int(args.seed))
    if args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    settings_manifest_path = Path(args.settings_manifest)
    if settings_manifest_path.is_file():
        manifest = read_json(settings_manifest_path)
    else:
        manifest = default_settings()
        write_json(settings_manifest_path, manifest)

    suite = read_json(Path(args.trace_suite))
    suite_items = select_trace_items(
        suite,
        only_ids=set(parse_csv_list(args.trace_ids)) if args.trace_ids else None,
    )

    settings = select_settings(
        manifest["settings"],
        only_ids=set(parse_csv_list(args.settings)) if args.settings else None,
    )

    dataset_ranges = ensure_off_ranges(
        manifest["dataset_roots_for_off_range"],
        recompute=bool(args.recompute_off_ranges),
    )

    all_rows = []
    for setting in settings:
        dataset_key = str(setting["dataset_key"])
        if dataset_key not in dataset_ranges:
            raise RuntimeError(f"Missing dataset range for {dataset_key}")
        ratio_reference = dataset_ranges[dataset_key]
        rows = run_setting(
            setting=setting,
            suite_items=suite_items,
            ratio_reference=ratio_reference,
            device=device,
            eval_batch_size=int(args.eval_batch_size),
            seed=int(args.seed),
            noise_weight_override=float(args.noise_weight_override)
            if args.noise_weight_override is not None
            else None,
        )
        all_rows.extend(rows)

    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    with RUN_SUMMARY_CSV.open("w", encoding="utf-8", newline="") as fp:
        fieldnames = [
            "setting_id",
            "dataset_key",
            "run_dir",
            "checkpoint",
            "task_name",
            "trace_id",
            "trace_kind",
            "ones_ratio_effective",
            "raw_ones_ratio",
            "n_eval_samples",
            "acc",
            "acc_flip",
            "acc_best_polarity",
            "flip_suggested",
        ]
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    write_json(
        RUN_SUMMARY_JSON,
        {
            "device": str(device),
            "settings": [setting["id"] for setting in settings],
            "trace_count": len(suite_items),
            "row_count": len(all_rows),
            "rows": all_rows,
        },
    )
    print(f"[summary] {RUN_SUMMARY_CSV}")
    print(f"[summary] {RUN_SUMMARY_JSON}")


if __name__ == "__main__":
    main()
