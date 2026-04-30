#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from dataloader_trace_bridge import BinaryTaskDataset, _task_dirs


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Render task28 unified full500 representative accuracy for real_off + mismatch(28)."
        )
    )
    parser.add_argument("--run_dir", required=True, type=Path)
    parser.add_argument("--ckpt", required=True, type=Path)
    parser.add_argument("--mode", default="off", choices=["off", "on"], type=str)
    parser.add_argument("--heldout_task", default="task28_8vs9", type=str)
    parser.add_argument("--candidate_split", default="train", choices=["train", "test"], type=str)
    parser.add_argument("--probe_split", default="test", choices=["train", "test"], type=str)
    parser.add_argument("--candidate_limit", default=500, type=int)
    parser.add_argument("--probe_limit", default=500, type=int)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--candidate_batch_size", default=32, type=int)
    parser.add_argument("--image_batch_size", default=128, type=int)
    parser.add_argument("--output_dir", default=None, type=Path)
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_run_args(run_dir):
    config = json.loads((Path(run_dir) / "config.json").read_text(encoding="utf-8"))
    args = config["args"]

    class Args:
        pass

    cfg = Args()
    for key, value in args.items():
        setattr(cfg, key, value)
    return cfg


def load_checkpoint(ckpt_path, model):
    state = torch.load(ckpt_path, map_location="cpu")
    model.mixer.load_state_dict(state["mixer"], strict=True)
    for gen_idx, gen in enumerate(model.generator.as_list()):
        gen.load_state_dict(state[f"G{gen_idx}"], strict=True)


def manifest_path(task_dir: Path) -> Path:
    return task_dir / "task_dataset_manifest.json"


def task_pair_label(task_dir: Path) -> str:
    manifest = json.loads(manifest_path(task_dir).read_text(encoding="utf-8"))
    classes = manifest["classes"]
    return f"{int(classes[0])}vs{int(classes[1])}"


def sort_task_key(task_name: str) -> tuple[int, int]:
    tail = task_name.split("_", 1)[1]
    left, right = tail.split("vs")
    return int(left), int(right)


def build_dataset(cfg, task_dir: Path, mode: str, split: str):
    return BinaryTaskDataset(
        task_dir=task_dir,
        mode=mode,
        split=split,
        fold_trace=cfg.fold_trace,
        trace_c=cfg.trace_c,
        trace_w=cfg.trace_w,
        trace_len=cfg.trace_len,
        input_shape=cfg.input_shape,
        granularity=cfg.granularity,
    )


def load_probe_data(dataset: BinaryTaskDataset, limit: int):
    limit = min(int(limit), len(dataset))
    images = []
    labels = []
    for index in range(limit):
        _, image, label = dataset[index]
        images.append(image)
        labels.append(int(label.item()))
    if not images:
        raise RuntimeError("No probe samples were loaded")
    return torch.stack(images, 0), torch.tensor(labels, dtype=torch.long)


def candidate_ordinals(dataset: BinaryTaskDataset, limit: int):
    return [int(sample["ordinal"]) for sample in dataset.samples[: min(int(limit), len(dataset))]]


@torch.no_grad()
def surrogate_logits_from_layers(model, single_layers, images: torch.Tensor, device: str, batch_size: int):
    outputs = []
    for start in range(0, images.size(0), int(batch_size)):
        batch = images[start : start + int(batch_size)].to(device, non_blocking=True)
        out = model.eval_f(single_layers, batch)
        outputs.append(out.cpu())
    return torch.cat(outputs, 0).numpy()


@torch.no_grad()
def representative_behavior_for_task(
    model,
    dataset: BinaryTaskDataset,
    ordinals: list[int],
    probe_images: torch.Tensor,
    probe_labels_np: np.ndarray,
    device: str,
    candidate_batch_size: int,
    image_batch_size: int,
):
    margins = []
    accs = []
    for start in range(0, len(ordinals), int(candidate_batch_size)):
        chunk_ordinals = ordinals[start : start + int(candidate_batch_size)]
        trace_batch = torch.stack([dataset._load_trace(ordinal) for ordinal in chunk_ordinals], 0).to(
            device, non_blocking=True
        )
        codes = model.mixer(trace_batch)
        params = model.generator(codes)
        for sample_idx, ordinal in enumerate(chunk_ordinals):
            single_layers = tuple(param[sample_idx] for param in params)
            logits = surrogate_logits_from_layers(
                model,
                single_layers,
                probe_images,
                device=device,
                batch_size=image_batch_size,
            )
            margin = logits[:, 1] - logits[:, 0]
            acc = float((logits.argmax(axis=1) == probe_labels_np).mean())
            margins.append(margin)
            accs.append((ordinal, acc))
    margin_matrix = np.stack(margins, axis=0)
    center = margin_matrix.mean(axis=0, keepdims=True)
    rep_pos = int(np.argmin(np.linalg.norm(margin_matrix - center, axis=1)))
    rep_ordinal, rep_acc = accs[rep_pos]
    return {
        "rep_ordinal": rep_ordinal,
        "probe_acc": float(rep_acc),
        "margin_mean": float(margin_matrix[rep_pos].mean()),
    }


def render_acc_rank(rows, output_path: Path):
    rows = sorted(rows, key=lambda row: float(row["probe_acc"]))
    fig_h = max(7.5, 0.36 * len(rows))
    fig, ax = plt.subplots(figsize=(12.5, fig_h), dpi=180)

    real_index = None
    for idx, row in enumerate(rows):
        if row["kind"] == "real_off":
            real_index = idx
            break
    if real_index is not None:
        ax.axhspan(real_index - 0.45, real_index + 0.45, color="#dff3e4", alpha=0.8, zorder=0)

    ys = np.arange(len(rows))
    xs = np.array([100.0 * float(row["probe_acc"]) for row in rows], dtype=np.float32)
    colors = ["#2c95d3" if row["kind"] == "mismatch" else "#2aa745" for row in rows]
    sizes = [32 if row["kind"] == "mismatch" else 46 for row in rows]

    ax.scatter(xs, ys, c=colors, s=sizes, linewidths=0, zorder=3)
    ax.set_yticks(ys)
    ax.set_yticklabels([row["source_label"] for row in rows], fontsize=9)
    ax.set_xlabel("Accuracy (%)")
    ax.set_ylabel("Seed Source")
    ax.set_title(
        "task28_8vs9 unified full500 representative acc | real_off + mismatch(28)",
        fontsize=15,
        pad=12,
    )
    ax.text(
        0.0,
        1.01,
        "same full500 selection rule for mismatch and real off; same full500 single-model acc metric",
        transform=ax.transAxes,
        fontsize=9,
        ha="left",
        va="bottom",
        color="#444444",
    )
    ax.grid(axis="x", linestyle="--", linewidth=0.6, alpha=0.35)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def write_csv(path: Path, rows):
    if not rows:
        raise RuntimeError(f"No rows to save for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_meta(
    path: Path,
    args,
    cfg,
    rows,
    summary_csv: Path,
    representatives_csv: Path,
    figure_png: Path,
):
    ordered = sorted(rows, key=lambda row: float(row["probe_acc"]), reverse=True)
    lines = [
        "task28 unified full500 analysis",
        f"mode={args.mode}",
        f"device={cfg.device}",
        f"selection=representative_behavior_row on full held-out {args.probe_limit}",
        f"candidate_split={args.candidate_split}",
        f"candidate_limit={args.candidate_limit}",
        f"probe_split={args.probe_split}",
        f"probe_limit={args.probe_limit}",
        f"acc_metric=single-model accuracy on full held-out {args.probe_limit} using selected representative row",
        f"run_dir={args.run_dir}",
        f"ckpt={args.ckpt}",
        f"summary_csv={summary_csv}",
        f"representatives_csv={representatives_csv}",
        f"acc_plot={figure_png}",
        "top5=" + ", ".join(
            f"{row['source_label']}:{100.0 * float(row['probe_acc']):.2f}%"
            for row in ordered[:5]
        ),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    cfg = load_run_args(args.run_dir)
    cfg.device = args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"
    if int(cfg.n_seed) != 1:
        raise RuntimeError(
            f"This unified full500 analysis currently expects n_seed=1, got {cfg.n_seed}"
        )

    set_seed(args.seed)

    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else args.run_dir / f"task28_unified_full500_{args.mode}_realoff_plus_mismatch28"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = output_dir / "task28_unified_full500_summary.csv"
    representatives_csv = output_dir / "task28_unified_full500_representatives.csv"
    figure_png = output_dir / "task28_unified_full500_acc_rank.png"
    meta_txt = output_dir / "task28_unified_full500_meta.txt"

    import importlib

    models = importlib.import_module(f"models.{cfg.target_model}")
    model = models.HyperTheft(cfg)
    model.mixer.eval()
    for gen in model.generator.as_list():
        gen.eval()
    load_checkpoint(args.ckpt, model)

    trace_root = Path(cfg.trace_root)
    heldout_dir = trace_root / args.heldout_task
    probe_dataset = build_dataset(cfg, heldout_dir, args.mode, args.probe_split)
    probe_images, probe_labels = load_probe_data(probe_dataset, args.probe_limit)
    probe_labels_np = probe_labels.numpy()

    rows = []
    mismatch_task_dirs = [
        task_dir
        for task_dir in _task_dirs(trace_root, include_heldout=False)
        if task_dir.name != args.heldout_task
    ]
    mismatch_task_dirs = sorted(mismatch_task_dirs, key=lambda path: sort_task_key(path.name))
    all_sources = mismatch_task_dirs + [heldout_dir]

    for task_dir in all_sources:
        source_dataset = build_dataset(cfg, task_dir, args.mode, args.candidate_split)
        ordinals = candidate_ordinals(source_dataset, args.candidate_limit)
        if not ordinals:
            raise RuntimeError(f"No candidate traces found for {task_dir.name}")
        result = representative_behavior_for_task(
            model=model,
            dataset=source_dataset,
            ordinals=ordinals,
            probe_images=probe_images,
            probe_labels_np=probe_labels_np,
            device=cfg.device,
            candidate_batch_size=args.candidate_batch_size,
            image_batch_size=args.image_batch_size,
        )
        is_real = task_dir.name == args.heldout_task
        rows.append(
            {
                "source_task": task_dir.name,
                "source_pair": task_pair_label(task_dir),
                "source_label": "real_off" if is_real else task_dir.name,
                "kind": "real_off" if is_real else "mismatch",
                "rep_ordinal": int(result["rep_ordinal"]),
                "probe_acc": float(result["probe_acc"]),
                "probe_acc_percent": 100.0 * float(result["probe_acc"]),
                "margin_mean": float(result["margin_mean"]),
                "candidate_split": args.candidate_split,
                "candidate_count": len(ordinals),
                "probe_split": args.probe_split,
                "probe_count": int(min(args.probe_limit, len(probe_dataset))),
                "mode": args.mode,
            }
        )
        print(
            f"[{task_dir.name}] rep_ordinal={result['rep_ordinal']} "
            f"probe_acc={100.0 * float(result['probe_acc']):.2f}%",
            flush=True,
        )

        partial_ranked = sorted(rows, key=lambda row: float(row["probe_acc"]), reverse=True)
        partial_representatives = sorted(
            rows,
            key=lambda row: (row["kind"] == "real_off", sort_task_key(row["source_task"])),
        )
        write_csv(summary_csv, partial_ranked)
        write_csv(representatives_csv, partial_representatives)
        render_acc_rank(rows, figure_png)

    ranked_rows = sorted(rows, key=lambda row: float(row["probe_acc"]), reverse=True)
    representatives_rows = sorted(
        rows,
        key=lambda row: (row["kind"] == "real_off", sort_task_key(row["source_task"])),
    )
    write_csv(summary_csv, ranked_rows)
    write_csv(representatives_csv, representatives_rows)
    render_acc_rank(rows, figure_png)
    write_meta(meta_txt, args, cfg, rows, summary_csv, representatives_csv, figure_png)

    print(summary_csv, flush=True)
    print(representatives_csv, flush=True)
    print(figure_png, flush=True)
    print(meta_txt, flush=True)


if __name__ == "__main__":
    main()
