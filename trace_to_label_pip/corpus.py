from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset


DATASETS = {
    "generic": {"num_classes": 0, "task": "auto"},
    "mnist": {"num_classes": 10, "task": "multiclass"},
    "cifar": {"num_classes": 10, "task": "multiclass"},
    "chest": {"num_classes": 14, "task": "multilabel"},
    "imagenet50": {"num_classes": 50, "task": "multiclass"},
}


@dataclass
class Split:
    traces: torch.Tensor
    targets: torch.Tensor
    labels: torch.Tensor | None = None


@dataclass
class Corpus:
    splits: dict[str, Split]
    trace_dim: int
    num_classes: int
    task: str


def load_split(payload: dict, name: str) -> Split | None:
    item = payload.get(name)
    if item is None:
        return None
    if "traces" not in item:
        raise KeyError(f"{name} split must contain 'traces'")
    targets = item.get("targets", item.get("target"))
    labels = item.get("gt_labels", item.get("labels"))
    if targets is None:
        if labels is None:
            raise KeyError(f"{name} split must contain 'targets' or 'labels'")
        targets, labels = labels, None
    return Split(
        traces=torch.as_tensor(item["traces"]).float(),
        targets=torch.as_tensor(targets),
        labels=torch.as_tensor(labels) if labels is not None else None,
    )


def infer_task(payload: dict, targets: torch.Tensor, dataset: str, task: str) -> str:
    if task != "auto":
        return task
    if payload.get("task") in {"multiclass", "multilabel"}:
        return payload["task"]
    if targets.ndim > 1 and targets.shape[-1] > 1:
        return "multilabel"
    default = DATASETS.get(dataset, DATASETS["generic"])["task"]
    return default if default != "auto" else "multiclass"


def infer_classes(payload: dict, targets: torch.Tensor, dataset: str, num_classes: int) -> int:
    if num_classes > 0:
        return num_classes
    if "num_classes" in payload:
        return int(payload["num_classes"])
    if targets.ndim > 1 and targets.shape[-1] > 1:
        return int(targets.shape[-1])
    default = DATASETS.get(dataset, DATASETS["generic"])["num_classes"]
    return int(default or targets.max().item() + 1)


def load_corpus(path: str | Path, args: argparse.Namespace) -> Corpus:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError("Trace-label corpus must be a torch-saved dict")
    splits = {name: split for name in ("train", "val", "test") if (split := load_split(payload, name))}
    if "test" not in splits:
        raise KeyError("Trace-label corpus must contain a test split")

    reference = splits.get("train") or splits.get("val") or splits["test"]
    task = infer_task(payload, reference.targets, args.dataset, args.task)
    num_classes = infer_classes(payload, reference.targets, args.dataset, args.num_classes)
    return Corpus(splits, int(reference.traces[0].numel()), num_classes, task)


def make_loader(split: Split, batch_size: int, shuffle: bool) -> DataLoader:
    labels = split.labels if split.labels is not None else torch.empty(split.traces.shape[0], 0)
    return DataLoader(TensorDataset(split.traces, split.targets, labels), batch_size=batch_size, shuffle=shuffle)
