import json
import math
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


TRACE_COMPONENT_ORDER = ("conv", "maxpool", "fc_110", "fc_ef")


def _task_dirs(trace_root, include_heldout=False, include_task_names=None):
    root = Path(trace_root)
    task_dirs = sorted(
        [path for path in root.iterdir() if path.is_dir() and path.name.startswith("task")]
    )
    if include_task_names:
        include_set = set(include_task_names)
        task_dirs = [path for path in task_dirs if path.name in include_set]
    if include_heldout:
        return task_dirs
    return [path for path in task_dirs if not path.name.startswith("task28_")]


def _sample_dir(task_dir, mode, split, ordinal):
    return Path(task_dir) / mode / split / f"idx{ordinal:06d}"


def _input_dir(task_dir, split, ordinal):
    return Path(task_dir) / "inputs" / split / f"idx{ordinal:06d}" / "input_nchw_f32.bin"


def _match_component(sample_dir, component):
    matches = sorted(sample_dir.glob(f"{component}*_bits.bin"))
    if not matches:
        raise FileNotFoundError(f"Missing {component} trace in {sample_dir}")
    return matches[0]


def _unpack_trace_bits(path):
    packed = np.fromfile(path, dtype=np.uint8)
    return np.unpackbits(packed, bitorder="big").astype(np.float32)


def inspect_trace_layout(trace_root, mode, trace_w_override=None):
    task_dir = _task_dirs(trace_root)[0]
    sample_dir = _sample_dir(task_dir, mode, "train", 0)
    component_lengths = []
    component_paths = []
    for component in TRACE_COMPONENT_ORDER:
        path = _match_component(sample_dir, component)
        bits = _unpack_trace_bits(path)
        component_paths.append(path.name)
        component_lengths.append(int(bits.shape[0]))
    trace_len = int(sum(component_lengths))
    trace_w = int(trace_w_override) if trace_w_override is not None else 64
    trace_c = int(math.ceil(trace_len / float(trace_w * trace_w)))
    return {
        "trace_len": trace_len,
        "trace_c": trace_c,
        "trace_w": trace_w,
        "component_lengths": component_lengths,
        "component_files": component_paths,
        "pad_len": trace_c * trace_w * trace_w,
    }


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
        image_size=64,
        granularity=1,
    ):
        self.task_dir = Path(task_dir)
        self.mode = mode
        self.split = split
        self.fold_trace = bool(fold_trace)
        self.trace_c = trace_c
        self.trace_w = trace_w
        self.trace_len = trace_len
        self.image_size = image_size
        self.granularity = granularity
        with open(self.task_dir / "task_dataset_manifest.json", "r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        self.samples = self._filter_existing_samples(manifest[split])

    def __len__(self):
        return len(self.samples)

    def _sample_complete(self, ordinal):
        sample_dir = _sample_dir(self.task_dir, self.mode, self.split, ordinal)
        if not sample_dir.exists():
            return False
        input_path = _input_dir(self.task_dir, self.split, ordinal)
        if not input_path.exists():
            return False
        for component in TRACE_COMPONENT_ORDER:
            if not list(sample_dir.glob(f"{component}*_bits.bin")):
                return False
        return True

    def _filter_existing_samples(self, samples):
        filtered = []
        for sample in samples:
            ordinal = int(sample["ordinal"])
            if self._sample_complete(ordinal):
                filtered.append(sample)
        return filtered

    def _load_trace(self, ordinal):
        cache_path = self._trace_cache_path(ordinal)
        if cache_path.exists():
            return torch.from_numpy(np.load(cache_path, allow_pickle=False))

        sample_dir = _sample_dir(self.task_dir, self.mode, self.split, ordinal)
        chunks = []
        for component in TRACE_COMPONENT_ORDER:
            bits = _unpack_trace_bits(_match_component(sample_dir, component))
            chunks.append(bits)
        trace = np.concatenate(chunks, axis=0)
        trace = torch.from_numpy(trace)
        if self.granularity > 1:
            trace = trace[0 : self.trace_len : self.granularity]
        if self.fold_trace:
            pad_len = self.trace_c * self.trace_w * self.trace_w
            if trace.numel() < pad_len:
                trace = torch.cat([trace, torch.zeros(pad_len - trace.numel())], 0)
            else:
                trace = trace[:pad_len]
            trace = trace.view(self.trace_c, self.trace_w, self.trace_w)
        else:
            if trace.numel() < self.trace_len:
                trace = torch.cat([trace, torch.zeros(self.trace_len - trace.numel())], 0)
            else:
                trace = trace[: self.trace_len]
        self._save_trace_cache(cache_path, trace)
        return trace

    def _trace_cache_path(self, ordinal):
        sample_dir = _sample_dir(self.task_dir, self.mode, self.split, ordinal)
        fold_tag = "fold" if self.fold_trace else "flat"
        cache_name = (
            f"trace_{fold_tag}_c{self.trace_c}_w{self.trace_w}_"
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

    def _load_input(self, ordinal):
        input_path = _input_dir(self.task_dir, self.split, ordinal)
        image = np.fromfile(input_path, dtype=np.float32)
        image = image.reshape(1, self.image_size, self.image_size)
        return torch.from_numpy(image)

    def __getitem__(self, index):
        sample = self.samples[index]
        ordinal = int(sample["ordinal"])
        trace = self._load_trace(ordinal)
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
    image_size=64,
    granularity=1,
    task_limit=None,
    include_heldout=False,
    include_task_names=None,
):
    tasks = []
    loader_kwargs = {
        "num_workers": workers,
        "pin_memory": True,
    }
    for task_index, task_dir in enumerate(
        _task_dirs(
            trace_root,
            include_heldout=include_heldout,
            include_task_names=include_task_names,
        )
    ):
        if task_limit is not None and task_index >= task_limit:
            break
        train_dataset = BinaryTaskDataset(
            task_dir=task_dir,
            mode=mode,
            split="train",
            fold_trace=fold_trace,
            trace_c=trace_c,
            trace_w=trace_w,
            trace_len=trace_len,
            image_size=image_size,
            granularity=granularity,
        )
        seed_dataset = BinaryTaskDataset(
            task_dir=task_dir,
            mode=mode,
            split="test",
            fold_trace=fold_trace,
            trace_c=trace_c,
            trace_w=trace_w,
            trace_len=trace_len,
            image_size=image_size,
            granularity=granularity,
        )
        train_seed_dataset = BinaryTaskDataset(
            task_dir=task_dir,
            mode=mode,
            split="train",
            fold_trace=fold_trace,
            trace_c=trace_c,
            trace_w=trace_w,
            trace_len=trace_len,
            image_size=image_size,
            granularity=granularity,
        )
        eval_dataset = BinaryTaskDataset(
            task_dir=task_dir,
            mode=mode,
            split="test",
            fold_trace=fold_trace,
            trace_c=trace_c,
            trace_w=trace_w,
            trace_len=trace_len,
            image_size=image_size,
            granularity=granularity,
        )
        tasks.append(
            {
                "task_name": task_dir.name,
                "train_loader": DataLoader(
                    train_dataset,
                    batch_size=batch_size,
                    shuffle=True,
                    drop_last=True,
                    **loader_kwargs,
                ),
                "seed_loader": DataLoader(
                    seed_dataset,
                    batch_size=batch_size,
                    shuffle=True,
                    drop_last=True,
                    **loader_kwargs,
                ),
                "eval_seed_loader": DataLoader(
                    train_seed_dataset,
                    batch_size=batch_size,
                    shuffle=True,
                    drop_last=True,
                    **loader_kwargs,
                ),
                "eval_loader": DataLoader(
                    eval_dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    drop_last=True,
                    **loader_kwargs,
                ),
            }
        )
    return tasks
