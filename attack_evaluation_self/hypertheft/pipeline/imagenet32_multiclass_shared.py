#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from imagenet32_binary_shared import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    INPUT_SHAPE,
    RAW_IMAGE_SIZE,
    SelectedImageNet32TrainPool,
    save_json,
)


@dataclass(frozen=True)
class MulticlassVictimSpec:
    task_index: int
    task_name: str
    classes: tuple[int, ...]
    class_names: tuple[str, ...]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_train_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.RandomCrop(RAW_IMAGE_SIZE, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def build_eval_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


class MulticlassImageNet32Dataset(Dataset):
    def __init__(
        self,
        pool: SelectedImageNet32TrainPool,
        victim: MulticlassVictimSpec,
        records: list[dict],
        transform: transforms.Compose,
    ):
        self.pool = pool
        self.victim = victim
        self.records = list(records)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        flat = self.pool.flat_image(record["label_raw"], record["class_ordinal"])
        image = flat.reshape(3, RAW_IMAGE_SIZE, RAW_IMAGE_SIZE).transpose(1, 2, 0)
        tensor = self.transform(Image.fromarray(image))
        label = int(record["label_index"])
        return tensor, label


def _make_record(
    pool: SelectedImageNet32TrainPool,
    victim: MulticlassVictimSpec,
    label_raw: int,
    label_index: int,
    class_ordinal: int,
) -> dict:
    return {
        "label_raw": int(label_raw),
        "label_index": int(label_index),
        "label_name": str(victim.class_names[label_index]),
        "class_ordinal": int(class_ordinal),
        "orig_imagenet32_train_idx": int(pool.global_index(label_raw, class_ordinal)),
    }


def _shuffle_and_ordinalize(records: list[dict], seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    indices = np.arange(len(records), dtype=np.int64)
    rng.shuffle(indices)
    ordered = []
    for ordinal, pos in enumerate(indices.tolist()):
        record = dict(records[pos])
        record["ordinal"] = int(ordinal)
        record["filtered_idx"] = int(ordinal)
        ordered.append(record)
    return ordered


def build_split_manifest(
    pool: SelectedImageNet32TrainPool,
    victim: MulticlassVictimSpec,
    *,
    victim_val_per_class: int,
    train_trace_per_class: int,
    test_per_class: int,
    seed: int,
) -> dict:
    splits: dict[str, list[dict]] = {
        "victim_train": [],
        "victim_val": [],
        "train": [],
        "test": [],
    }
    required = int(victim_val_per_class) + int(train_trace_per_class) + int(test_per_class) + 1
    rng = np.random.default_rng(seed + victim.task_index)
    for label_index, label_raw in enumerate(victim.classes):
        total = pool.count(label_raw)
        if total < required:
            raise RuntimeError(
                f"Victim {victim.task_name} label {label_raw} only has {total} samples, "
                f"required at least {required}"
            )
        class_indices = np.arange(total, dtype=np.int64)
        rng.shuffle(class_indices)
        test_slice = class_indices[:test_per_class]
        train_trace_slice = class_indices[test_per_class : test_per_class + train_trace_per_class]
        victim_val_slice = class_indices[
            test_per_class + train_trace_per_class : test_per_class + train_trace_per_class + victim_val_per_class
        ]
        victim_train_slice = class_indices[test_per_class + train_trace_per_class + victim_val_per_class :]

        for class_ordinal in victim_train_slice.tolist():
            splits["victim_train"].append(
                _make_record(pool, victim, label_raw, label_index, int(class_ordinal))
            )
        for class_ordinal in victim_val_slice.tolist():
            splits["victim_val"].append(
                _make_record(pool, victim, label_raw, label_index, int(class_ordinal))
            )
        for class_ordinal in train_trace_slice.tolist():
            splits["train"].append(
                _make_record(pool, victim, label_raw, label_index, int(class_ordinal))
            )
        for class_ordinal in test_slice.tolist():
            splits["test"].append(
                _make_record(pool, victim, label_raw, label_index, int(class_ordinal))
            )

    split_seed_offsets = {
        "victim_train": 11,
        "victim_val": 13,
        "train": 17,
        "test": 19,
    }
    for split_name, split_records in list(splits.items()):
        splits[split_name] = _shuffle_and_ordinalize(
            split_records,
            seed + victim.task_index + split_seed_offsets[split_name],
        )

    return {
        "task_index": int(victim.task_index),
        "task_name": victim.task_name,
        "classes": [int(label) for label in victim.classes],
        "class_names": [str(name) for name in victim.class_names],
        "n_class": len(victim.classes),
        "input_shape": list(INPUT_SHAPE),
        "raw_image_size": RAW_IMAGE_SIZE,
        "seed": int(seed),
        "split_counts": {name: len(records) for name, records in splits.items()},
        "splits": splits,
    }


def write_input_dataset_manifest(
    out_root: Path | str,
    *,
    data_root: Path | str,
    victim: MulticlassVictimSpec,
    split: str,
    dataset_len: int,
) -> None:
    payload = {
        "task_name": victim.task_name,
        "task_index": int(victim.task_index),
        "split": str(split),
        "classes": [int(label) for label in victim.classes],
        "class_names": [str(name) for name in victim.class_names],
        "n_class": len(victim.classes),
        "data_root": str(data_root),
        "dataset_len": int(dataset_len),
        "normalization": {
            "mean": list(IMAGENET_MEAN),
            "std": list(IMAGENET_STD),
        },
        "input_shape": list(INPUT_SHAPE),
        "raw_image_size": RAW_IMAGE_SIZE,
        "dataset": "imagenet32_train_pool",
    }
    save_json(Path(out_root) / "dataset_manifest.json", payload)


def export_input_record(
    pool: SelectedImageNet32TrainPool,
    victim: MulticlassVictimSpec,
    record: dict,
    *,
    split: str,
    out_root: Path | str,
) -> dict:
    ordinal = int(record["ordinal"])
    sample_dir = Path(out_root) / f"idx{ordinal:06d}"
    sample_dir.mkdir(parents=True, exist_ok=True)
    input_path = sample_dir / "input_nchw_f32.bin"

    flat = pool.flat_image(record["label_raw"], record["class_ordinal"])
    image = flat.reshape(3, RAW_IMAGE_SIZE, RAW_IMAGE_SIZE).astype(np.float32) / 255.0
    for channel in range(3):
        image[channel] = (image[channel] - IMAGENET_MEAN[channel]) / IMAGENET_STD[channel]

    tmp_path = input_path.with_name(f".{input_path.name}.{os.getpid()}.tmp")
    image.tofile(tmp_path)
    tmp_path.replace(input_path)

    meta = {
        "task_name": victim.task_name,
        "task_index": int(victim.task_index),
        "classes": [int(label) for label in victim.classes],
        "class_names": [str(name) for name in victim.class_names],
        "n_class": len(victim.classes),
        "split": str(split),
        "filtered_idx": ordinal,
        "label_raw": int(record["label_raw"]),
        "label_index": int(record["label_index"]),
        "label_name": str(record["label_name"]),
        "class_ordinal": int(record["class_ordinal"]),
        "orig_imagenet32_train_idx": int(record["orig_imagenet32_train_idx"]),
        "input_path": str(input_path),
        "shape": list(INPUT_SHAPE),
        "dtype": "float32",
        "normalization": {
            "mean": list(IMAGENET_MEAN),
            "std": list(IMAGENET_STD),
        },
        "dataset": "imagenet32_train_pool",
    }
    save_json(sample_dir / "sample_meta.json", meta)
    return meta


def load_victim_from_pair_manifest(
    pair_manifest: Path | str,
    *,
    task_name: str,
    task_index: int = 0,
) -> tuple[MulticlassVictimSpec, dict]:
    payload = json.loads(Path(pair_manifest).read_text(encoding="utf-8"))
    ordered: list[int] = []
    name_map: dict[int, str] = {}
    for row in payload:
        for label, name in zip(row["classes"], row["class_names"]):
            label_int = int(label)
            if label_int not in name_map:
                ordered.append(label_int)
                name_map[label_int] = str(name)
    if not ordered:
        raise RuntimeError(f"No classes found in pair manifest: {pair_manifest}")
    victim = MulticlassVictimSpec(
        task_index=int(task_index),
        task_name=str(task_name),
        classes=tuple(ordered),
        class_names=tuple(name_map[label] for label in ordered),
    )
    snapshot = {
        "source_pair_manifest": str(pair_manifest),
        "task_index": int(victim.task_index),
        "task_name": victim.task_name,
        "classes": [int(label) for label in victim.classes],
        "class_names": [str(name) for name in victim.class_names],
        "n_class": len(victim.classes),
    }
    return victim, snapshot


def build_multiclass_task_dataset_manifest(task_root: Path, split_manifest: dict) -> dict:
    payload = {
        "task_name": str(split_manifest["task_name"]),
        "task_index": int(split_manifest["task_index"]),
        "classes": [int(label) for label in split_manifest["classes"]],
        "class_names": [str(name) for name in split_manifest["class_names"]],
        "n_class": int(split_manifest["n_class"]),
        "train": [
            {
                "ordinal": int(record["ordinal"]),
                "source_index": int(record["orig_imagenet32_train_idx"]),
                "gt_label": int(record["label_raw"]),
                "label_index": int(record["label_index"]),
            }
            for record in split_manifest["splits"]["train"]
        ],
        "test": [
            {
                "ordinal": int(record["ordinal"]),
                "source_index": int(record["orig_imagenet32_train_idx"]),
                "gt_label": int(record["label_raw"]),
                "label_index": int(record["label_index"]),
            }
            for record in split_manifest["splits"]["test"]
        ],
    }
    save_json(task_root / "task_multiclass_dataset_manifest.json", payload)
    return payload
