from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class Split:
    inputs: torch.Tensor
    labels: torch.Tensor


@dataclass
class Task:
    name: str
    seed_traces: torch.Tensor
    train: Split
    val: Split | None = None
    test: Split | None = None
    seed_labels: torch.Tensor | None = None

    def get_split(self, name: str) -> Split:
        split = getattr(self, name, None)
        if split is None:
            raise KeyError(f"{self.name}: missing split '{name}'")
        return split


@dataclass
class Corpus:
    tasks: list[Task]
    metadata: dict[str, Any]

    @property
    def trace_shape(self) -> tuple[int, ...]:
        return tuple(self.tasks[0].seed_traces.shape[1:])

    @property
    def input_shape(self) -> tuple[int, ...]:
        return tuple(self.tasks[0].train.inputs.shape[1:])

    @property
    def num_classes(self) -> int:
        if "num_classes" in self.metadata:
            return int(self.metadata["num_classes"])
        return int(torch.cat([task.train.labels.reshape(-1) for task in self.tasks]).max().item()) + 1


def tensor(payload: dict[str, Any], *names: str) -> torch.Tensor:
    for name in names:
        value = payload.get(name)
        if value is not None:
            return value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    raise KeyError(f"missing tensor; tried {', '.join(names)}")


def maybe_tensor(payload: dict[str, Any], *names: str) -> torch.Tensor | None:
    for name in names:
        value = payload.get(name)
        if value is not None:
            return value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    return None


def split(payload: dict[str, Any] | None) -> Split | None:
    if payload is None:
        return None
    return Split(tensor(payload, "inputs", "images", "x").float(), tensor(payload, "labels", "targets", "y").long())


def task(payload: dict[str, Any], index: int) -> Task:
    seed_labels = maybe_tensor(payload, "seed_labels", "trace_labels")
    item = Task(
        name=str(payload.get("name", f"task_{index:03d}")),
        seed_traces=tensor(payload, "seed_traces", "traces").float(),
        seed_labels=seed_labels.long() if seed_labels is not None else None,
        train=split(payload.get("train"))
        or Split(tensor(payload, "train_inputs", "inputs", "images").float(), tensor(payload, "train_labels", "labels", "targets").long()),
        val=split(payload.get("val")),
        test=split(payload.get("test")),
    )
    if item.seed_labels is not None and item.seed_labels.numel() != item.seed_traces.size(0):
        raise ValueError(f"{item.name}: seed_labels length must match seed_traces")
    return item


def load_corpus(path: str | Path) -> Corpus:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError("HyperTheft corpus must be a torch-saved dictionary")
    raw_tasks = payload.get("tasks") or [{
        "name": "task_000",
        "seed_traces": payload.get("seed_traces", payload.get("traces")),
        "seed_labels": payload.get("seed_labels", payload.get("trace_labels")),
        "train": payload.get("train"),
        "val": payload.get("val"),
        "test": payload.get("test"),
    }]
    tasks = [task(item, index) for index, item in enumerate(raw_tasks)]
    if not tasks:
        raise ValueError("HyperTheft corpus contains no tasks")
    metadata = dict(payload.get("metadata", {}))
    metadata.update({key: payload[key] for key in ("dataset", "num_classes", "target") if key in payload and key not in metadata})
    return Corpus(tasks, metadata)


def make_loader(item: Split, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(TensorDataset(item.inputs.float(), item.labels.long()), batch_size=batch_size, shuffle=shuffle)


def source_labels(item: Task) -> list[int]:
    return [] if item.seed_labels is None else sorted(int(label) for label in item.seed_labels.unique().tolist())


def sample_seed_batches(item: Task, batch_size: int, n_seed: int, device: torch.device, source_label: int | None = None) -> list[torch.Tensor]:
    traces = item.seed_traces
    if source_label is not None:
        if item.seed_labels is None:
            raise ValueError(f"{item.name}: papersem mode requires seed_labels")
        traces = traces[item.seed_labels.reshape(-1).eq(int(source_label))]
        if traces.numel() == 0:
            raise ValueError(f"{item.name}: no seed traces for label {source_label}")
    return [traces[torch.randint(0, traces.size(0), (batch_size,))].to(device, non_blocking=True).float() for _ in range(n_seed)]
