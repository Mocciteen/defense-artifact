#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import os
import hashlib
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

TRACE_FILE_NAME = "singleblock_bits.txt"
DEFAULT_TRACE_FILES = (TRACE_FILE_NAME,)
HELDOUT_PREFIX = "task28_"


def _task_dirs(trace_root, include_heldout=False, include_task_names=None):
    root = Path(trace_root)
    task_dirs = sorted(
        [path for path in root.iterdir() if path.is_dir() and path.name.startswith("task")]
    )
    if include_task_names:
        include_set = set(include_task_names)
        return [path for path in task_dirs if path.name in include_set]
    if include_heldout:
        return task_dirs
    return [path for path in task_dirs if not path.name.startswith(HELDOUT_PREFIX)]


def _sample_dir(task_dir, mode, split, ordinal):
    return Path(task_dir) / mode / split / f"idx{ordinal:06d}"


def _normalize_trace_files(trace_files=None):
    if trace_files is None:
        return tuple(DEFAULT_TRACE_FILES)
    if isinstance(trace_files, str):
        values = [item.strip() for item in trace_files.split(",") if item.strip()]
    else:
        values = [str(item).strip() for item in trace_files if str(item).strip()]
    if not values:
        raise RuntimeError("trace_files resolved to an empty list")
    return tuple(values)


def _trace_paths(task_dir, mode, split, ordinal, trace_files=None):
    sample_root = _sample_dir(task_dir, mode, split, ordinal)
    return [sample_root / trace_name for trace_name in _normalize_trace_files(trace_files)]


def _trace_cache_tag(trace_files):
    joined = "|".join(_normalize_trace_files(trace_files))
    if joined == TRACE_FILE_NAME:
        return "singleblock"
    digest = hashlib.sha1(joined.encode("utf-8")).hexdigest()[:10]
    return f"multi_{digest}"


def _input_path(task_dir, split, ordinal):
    return Path(task_dir) / "inputs" / split / f"idx{ordinal:06d}" / "input_nchw_f32.bin"


def _task_trace_manifest_path(task_dir):
    return Path(task_dir) / "task_trace_manifest.json"


def _task_dataset_manifest_path(task_dir):
    return Path(task_dir) / "task_dataset_manifest.json"


def _load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json_atomic(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp_path, path)


def _orig_index_key(meta):
    for key in sorted(meta):
        if key.startswith("orig_") and key.endswith("_idx"):
            return key
    for key in ("source_index", "orig_index"):
        if key in meta:
            return key
    raise KeyError(f"Missing source index key in sample metadata: {sorted(meta)}")


def _build_split_manifest(task_dir, split):
    split_root = Path(task_dir) / "inputs" / split
    records = []
    if not split_root.exists():
        return records
    for meta_path in sorted(split_root.glob("idx*/sample_meta.json")):
        meta = _load_json(meta_path)
        records.append(
            {
                "ordinal": int(meta["filtered_idx"]),
                "source_index": int(meta[_orig_index_key(meta)]),
                "gt_label": int(meta["label_raw"]),
                "binary_label": int(meta["label_binary"]),
            }
        )
    records.sort(key=lambda item: item["ordinal"])
    return records


def ensure_task_dataset_manifest(task_dir):
    task_dir = Path(task_dir)
    task_meta = _load_json(_task_trace_manifest_path(task_dir))
    payload = {
        "task_name": task_meta["task_name"],
        "task_index": int(task_meta["task_index"]),
        "classes": [int(value) for value in task_meta["classes"]],
        "input_shape": [int(value) for value in task_meta.get("input_shape", [3, 32, 32])],
        "train": _build_split_manifest(task_dir, "train"),
        "test": _build_split_manifest(task_dir, "test"),
    }
    _write_json_atomic(_task_dataset_manifest_path(task_dir), payload)
    return payload


def _load_trace_bits(path):
    raw = Path(path).read_bytes().strip()
    if not raw:
        raise RuntimeError(f"Empty trace bit-string: {path}")
    values = np.frombuffer(raw, dtype=np.uint8)
    if not np.isin(values, (48, 49)).all():
        raise RuntimeError(f"Trace file contains non 0/1 bytes: {path}")
    return (values - 48).astype(np.float32, copy=False)


def _load_concat_trace_bits(paths):
    arrays = [_load_trace_bits(path) for path in paths]
    if len(arrays) == 1:
        return arrays[0]
    return np.concatenate(arrays, axis=0)


def inspect_trace_layout(
    trace_root,
    mode,
    trace_w_override=None,
    include_task_names=None,
    trace_files=None,
):
    trace_w = int(trace_w_override) if trace_w_override is not None else 64
    normalized_trace_files = _normalize_trace_files(trace_files)
    for task_dir in _task_dirs(trace_root, include_task_names=include_task_names):
        task_meta = _load_json(_task_trace_manifest_path(task_dir))
        for split in ("train", "test"):
            mode_root = Path(task_dir) / mode / split
            if not mode_root.exists():
                continue
            for sample_root in sorted(mode_root.glob("idx*")):
                trace_paths = [sample_root / trace_name for trace_name in normalized_trace_files]
                if not all(path.exists() for path in trace_paths):
                    continue
                component_lengths = []
                arrays = []
                for trace_path in trace_paths:
                    bits = _load_trace_bits(trace_path)
                    arrays.append(bits)
                    component_lengths.append(int(bits.shape[0]))
                bits = arrays[0] if len(arrays) == 1 else np.concatenate(arrays, axis=0)
                trace_len = int(bits.shape[0])
                trace_c = int(math.ceil(trace_len / float(trace_w * trace_w)))
                return {
                    "trace_len": trace_len,
                    "trace_c": trace_c,
                    "trace_w": trace_w,
                    "component_lengths": component_lengths,
                    "component_files": list(normalized_trace_files),
                    "pad_len": trace_c * trace_w * trace_w,
                    "input_shape": [int(value) for value in task_meta.get("input_shape", [3, 32, 32])],
                    "task_name": task_meta["task_name"],
                    "split": split,
                    "mode": mode,
                }
    raise RuntimeError(f"No completed {mode} traces found under {trace_root}")


class BinaryTaskDataset(Dataset):
    def __init__(
        self,
        task_dir,
        mode,
        split,
        fold_trace,
        trace_c,
        trace_w,
        trace_len,
        input_shape,
        granularity=1,
        require_trace=True,
        trace_files=None,
    ):
        self.task_dir = Path(task_dir)
        self.mode = mode
        self.split = split
        self.fold_trace = bool(fold_trace)
        self.trace_c = int(trace_c)
        self.trace_w = int(trace_w)
        self.trace_len = int(trace_len)
        self.input_shape = tuple(int(value) for value in input_shape)
        self.granularity = int(granularity)
        self.require_trace = bool(require_trace)
        self.trace_files = _normalize_trace_files(trace_files)
        manifest = ensure_task_dataset_manifest(self.task_dir)
        self.samples = self._filter_existing_samples(manifest[split])

    def __len__(self):
        return len(self.samples)

    def _sample_complete(self, ordinal):
        input_ok = _input_path(self.task_dir, self.split, ordinal).exists()
        if not input_ok:
            return False
        if not self.require_trace:
            return True
        return all(
            path.exists()
            for path in _trace_paths(
                self.task_dir, self.mode, self.split, ordinal, trace_files=self.trace_files
            )
        )

    def _filter_existing_samples(self, samples):
        filtered = []
        for sample in samples:
            ordinal = int(sample["ordinal"])
            if self._sample_complete(ordinal):
                filtered.append(sample)
        return filtered

    def _trace_cache_path(self, ordinal):
        sample_dir = _sample_dir(self.task_dir, self.mode, self.split, ordinal)
        fold_tag = "fold" if self.fold_trace else "flat"
        cache_name = (
            f"trace_{_trace_cache_tag(self.trace_files)}_{fold_tag}_c{self.trace_c}_w{self.trace_w}_"
            f"len{self.trace_len}_g{self.granularity}.npy"
        )
        return sample_dir / cache_name

    def _save_trace_cache(self, cache_path, trace):
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = cache_path.with_suffix(cache_path.suffix + f".tmp.{os.getpid()}")
        with open(tmp_path, "wb") as handle:
            np.save(
                handle,
                trace.detach().cpu().numpy().astype(np.float32, copy=False),
                allow_pickle=False,
            )
        os.replace(tmp_path, cache_path)

    def _load_trace(self, ordinal):
        cache_path = self._trace_cache_path(ordinal)
        if cache_path.exists():
            return torch.from_numpy(np.load(cache_path, allow_pickle=False))

        bits = torch.from_numpy(
            _load_concat_trace_bits(
                _trace_paths(
                    self.task_dir,
                    self.mode,
                    self.split,
                    ordinal,
                    trace_files=self.trace_files,
                )
            )
        )
        if self.granularity > 1:
            bits = bits[0 : self.trace_len : self.granularity]
        if self.fold_trace:
            pad_len = self.trace_c * self.trace_w * self.trace_w
            if bits.numel() < pad_len:
                bits = torch.cat([bits, torch.zeros(pad_len - bits.numel())], 0)
            else:
                bits = bits[:pad_len]
            bits = bits.view(self.trace_c, self.trace_w, self.trace_w)
        else:
            if bits.numel() < self.trace_len:
                bits = torch.cat([bits, torch.zeros(self.trace_len - bits.numel())], 0)
            else:
                bits = bits[: self.trace_len]
        self._save_trace_cache(cache_path, bits)
        return bits

    def _load_input(self, ordinal):
        image = np.fromfile(_input_path(self.task_dir, self.split, ordinal), dtype=np.float32)
        expected = int(np.prod(self.input_shape))
        if image.size != expected:
            raise RuntimeError(
                f"Bad input size for {self.task_dir.name} {self.split} idx{ordinal:06d}: "
                f"expected {expected}, got {image.size}"
            )
        return torch.from_numpy(image.reshape(self.input_shape))

    def __getitem__(self, index):
        sample = self.samples[index]
        ordinal = int(sample["ordinal"])
        trace = self._load_trace(ordinal) if self.require_trace else torch.zeros(1, dtype=torch.float32)
        image = self._load_input(ordinal)
        label = torch.tensor(int(sample["binary_label"]), dtype=torch.long)
        return trace, image, label


def load_task_loaders(
    trace_root,
    mode,
    batch_size,
    workers,
    fold_trace,
    trace_c,
    trace_w,
    trace_len,
    input_shape=(3, 32, 32),
    granularity=1,
    task_limit=None,
    include_heldout=False,
    include_task_names=None,
    trace_files=None,
):
    tasks = []
    loader_kwargs = {
        "num_workers": int(workers),
        "pin_memory": True,
    }
    for task_index, task_dir in enumerate(
        _task_dirs(
            trace_root,
            include_heldout=include_heldout,
            include_task_names=include_task_names,
        )
    ):
        if task_limit is not None and task_index >= int(task_limit):
            break
        train_dataset = BinaryTaskDataset(
            task_dir=task_dir,
            mode=mode,
            split="train",
            fold_trace=fold_trace,
            trace_c=trace_c,
            trace_w=trace_w,
            trace_len=trace_len,
            input_shape=input_shape,
            granularity=granularity,
            trace_files=trace_files,
        )
        seed_dataset = BinaryTaskDataset(
            task_dir=task_dir,
            mode=mode,
            split="test",
            fold_trace=fold_trace,
            trace_c=trace_c,
            trace_w=trace_w,
            trace_len=trace_len,
            input_shape=input_shape,
            granularity=granularity,
            trace_files=trace_files,
        )
        eval_seed_dataset = BinaryTaskDataset(
            task_dir=task_dir,
            mode=mode,
            split="train",
            fold_trace=fold_trace,
            trace_c=trace_c,
            trace_w=trace_w,
            trace_len=trace_len,
            input_shape=input_shape,
            granularity=granularity,
            trace_files=trace_files,
        )
        eval_dataset = BinaryTaskDataset(
            task_dir=task_dir,
            mode=mode,
            split="test",
            fold_trace=fold_trace,
            trace_c=trace_c,
            trace_w=trace_w,
            trace_len=trace_len,
            input_shape=input_shape,
            granularity=granularity,
            trace_files=trace_files,
        )
        if len(train_dataset) == 0 or len(seed_dataset) == 0 or len(eval_dataset) == 0:
            raise RuntimeError(
                f"Task {task_dir.name} does not have enough completed {mode} traces: "
                f"train={len(train_dataset)} seed={len(seed_dataset)} eval={len(eval_dataset)}"
            )
        tasks.append(
            {
                "task_name": task_dir.name,
                "train_loader": DataLoader(
                    train_dataset,
                    batch_size=int(batch_size),
                    shuffle=True,
                    drop_last=True,
                    **loader_kwargs,
                ),
                "seed_loader": DataLoader(
                    seed_dataset,
                    batch_size=int(batch_size),
                    shuffle=True,
                    drop_last=True,
                    **loader_kwargs,
                ),
                "eval_seed_loader": DataLoader(
                    eval_seed_dataset,
                    batch_size=int(batch_size),
                    shuffle=True,
                    drop_last=True,
                    **loader_kwargs,
                ),
                "eval_loader": DataLoader(
                    eval_dataset,
                    batch_size=int(batch_size),
                    shuffle=False,
                    drop_last=True,
                    **loader_kwargs,
                ),
            }
        )
    return tasks
