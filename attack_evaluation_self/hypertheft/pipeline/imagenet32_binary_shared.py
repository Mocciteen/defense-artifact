#!/usr/bin/env python3
from __future__ import annotations

import json
import pickle
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

RAW_IMAGE_SIZE = 32
INPUT_SHAPE = (3, 32, 32)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
DEFAULT_PAIR_MANIFEST = Path(__file__).resolve().parent / "imagenet32_unseen10_pairs.json"


@dataclass(frozen=True)
class TaskSpec:
    task_index: int
    task_name: str
    classes: tuple[int, int]
    class_names: tuple[str, str]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_task_specs(manifest_path: Path | str = DEFAULT_PAIR_MANIFEST, include_task_names: Iterable[str] | None = None) -> list[TaskSpec]:
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    include = set(include_task_names or [])
    tasks = []
    for row in manifest:
        task = TaskSpec(
            task_index=int(row["task_index"]),
            task_name=str(row["task_name"]),
            classes=(int(row["classes"][0]), int(row["classes"][1])),
            class_names=(str(row["class_names"][0]), str(row["class_names"][1])),
        )
        if include and task.task_name not in include:
            continue
        tasks.append(task)
    tasks.sort(key=lambda task: task.task_index)
    return tasks


class SelectedImageNet32TrainPool:
    def __init__(self, data_root: Path | str, selected_labels: Iterable[int]):
        self.data_root = Path(data_root)
        self.selected_labels = sorted({int(label) for label in selected_labels})
        self.train_files = sorted(self.data_root.glob("train_data_batch_*"))
        if len(self.train_files) != 10:
            raise FileNotFoundError(
                f"Expected 10 train_data_batch_* files under {self.data_root}, found {len(self.train_files)}"
            )
        self.data_by_label: dict[int, np.ndarray] = {}
        self.global_indices_by_label: dict[int, np.ndarray] = {}
        self._loaded = False

    @staticmethod
    def _load_batch(path: Path) -> tuple[np.ndarray, np.ndarray]:
        with path.open("rb") as handle:
            payload = pickle.load(handle, encoding="latin1")
        data = np.asarray(payload["data"], dtype=np.uint8)
        labels = np.asarray(payload["labels"], dtype=np.int64)
        if data.ndim != 2 or data.shape[1] != 3072:
            raise ValueError(f"Unexpected payload shape in {path}: {data.shape}")
        return data, labels

    def load(self) -> "SelectedImageNet32TrainPool":
        if self._loaded:
            return self

        data_chunks = {label: [] for label in self.selected_labels}
        index_chunks = {label: [] for label in self.selected_labels}
        global_offset = 0
        for path in self.train_files:
            data, labels = self._load_batch(path)
            for label in self.selected_labels:
                positions = np.flatnonzero(labels == label)
                if positions.size == 0:
                    continue
                data_chunks[label].append(data[positions].copy())
                index_chunks[label].append((global_offset + positions).astype(np.int64, copy=False))
            global_offset += int(labels.shape[0])

        for label in self.selected_labels:
            if not data_chunks[label]:
                raise RuntimeError(f"Label {label} was not found under {self.data_root}")
            self.data_by_label[label] = np.concatenate(data_chunks[label], axis=0)
            self.global_indices_by_label[label] = np.concatenate(index_chunks[label], axis=0)
        self._loaded = True
        return self

    def count(self, label: int) -> int:
        return int(self.data_by_label[int(label)].shape[0])

    def flat_image(self, label: int, class_ordinal: int) -> np.ndarray:
        return self.data_by_label[int(label)][int(class_ordinal)]

    def global_index(self, label: int, class_ordinal: int) -> int:
        return int(self.global_indices_by_label[int(label)][int(class_ordinal)])


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


class BinaryImageNet32TaskDataset(Dataset):
    def __init__(
        self,
        pool: SelectedImageNet32TrainPool,
        task: TaskSpec,
        records: list[dict],
        transform: transforms.Compose,
    ):
        self.pool = pool
        self.task = task
        self.records = list(records)
        self.transform = transform
        self.cls_a, self.cls_b = task.classes

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        flat = self.pool.flat_image(record["label_raw"], record["class_ordinal"])
        image = flat.reshape(3, RAW_IMAGE_SIZE, RAW_IMAGE_SIZE).transpose(1, 2, 0)
        tensor = self.transform(Image.fromarray(image))
        label = int(record["binary_label"])
        return tensor, label


def _make_record(pool: SelectedImageNet32TrainPool, task: TaskSpec, label_raw: int, class_ordinal: int) -> dict:
    binary_label = 0 if int(label_raw) == int(task.classes[0]) else 1
    class_name = task.class_names[binary_label]
    return {
        "label_raw": int(label_raw),
        "label_binary": int(binary_label),
        "binary_label": int(binary_label),
        "label_name": str(class_name),
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
    task: TaskSpec,
    *,
    victim_val_per_class: int,
    train_trace_per_class: int,
    test_per_class: int,
    seed: int,
    victim_train_use_all: bool = False,
) -> dict:
    splits: dict[str, list[dict]] = {
        "victim_train": [],
        "victim_val": [],
        "train": [],
        "test": [],
    }
    reserved = int(victim_val_per_class) + int(train_trace_per_class) + int(test_per_class)
    required = reserved if victim_train_use_all else reserved + 1
    rng = np.random.default_rng(seed + task.task_index)
    for label in task.classes:
        total = pool.count(label)
        if total < required:
            raise RuntimeError(
                f"Task {task.task_name} label {label} only has {total} samples, required at least {required}"
            )
        class_indices = np.arange(total, dtype=np.int64)
        rng.shuffle(class_indices)
        test_slice = class_indices[:test_per_class]
        train_trace_slice = class_indices[test_per_class : test_per_class + train_trace_per_class]
        victim_val_slice = class_indices[
            test_per_class + train_trace_per_class : test_per_class + train_trace_per_class + victim_val_per_class
        ]
        if victim_train_use_all:
            victim_train_slice = class_indices
        else:
            victim_train_slice = class_indices[test_per_class + train_trace_per_class + victim_val_per_class :]

        for class_ordinal in victim_train_slice.tolist():
            splits["victim_train"].append(_make_record(pool, task, label, int(class_ordinal)))
        for class_ordinal in victim_val_slice.tolist():
            splits["victim_val"].append(_make_record(pool, task, label, int(class_ordinal)))
        for class_ordinal in train_trace_slice.tolist():
            splits["train"].append(_make_record(pool, task, label, int(class_ordinal)))
        for class_ordinal in test_slice.tolist():
            splits["test"].append(_make_record(pool, task, label, int(class_ordinal)))

    split_seed_offsets = {
        "victim_train": 11,
        "victim_val": 13,
        "train": 17,
        "test": 19,
    }
    for split_name, split_records in list(splits.items()):
        splits[split_name] = _shuffle_and_ordinalize(split_records, seed + task.task_index + split_seed_offsets[split_name])

    return {
        "task_index": int(task.task_index),
        "task_name": task.task_name,
        "classes": [int(task.classes[0]), int(task.classes[1])],
        "class_names": [task.class_names[0], task.class_names[1]],
        "input_shape": list(INPUT_SHAPE),
        "raw_image_size": RAW_IMAGE_SIZE,
        "seed": int(seed),
        "victim_train_use_all": bool(victim_train_use_all),
        "split_counts": {name: len(records) for name, records in splits.items()},
        "splits": splits,
    }


def save_json(path: Path | str, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_input_dataset_manifest(
    out_root: Path | str,
    *,
    data_root: Path | str,
    task: TaskSpec,
    split: str,
    dataset_len: int,
) -> None:
    payload = {
        "task_name": task.task_name,
        "task_index": int(task.task_index),
        "split": str(split),
        "classes": [int(task.classes[0]), int(task.classes[1])],
        "class_names": [task.class_names[0], task.class_names[1]],
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
    task: TaskSpec,
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
        "task_name": task.task_name,
        "task_index": int(task.task_index),
        "classes": [int(task.classes[0]), int(task.classes[1])],
        "class_names": [task.class_names[0], task.class_names[1]],
        "split": str(split),
        "filtered_idx": ordinal,
        "label_raw": int(record["label_raw"]),
        "label_binary": int(record["label_binary"]),
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
