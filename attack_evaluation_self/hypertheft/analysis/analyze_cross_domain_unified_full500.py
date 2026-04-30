#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from dataloader_trace_bridge import BinaryTaskDataset, _task_dirs, ensure_task_dataset_manifest


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Render unified full500 representative-accuracy plots for cross-domain held-out tasks "
            "(e.g. CIFAR mismatch sources + ImageNet32 real_off)."
        )
    )
    parser.add_argument("--run_dir", required=True, type=Path)
    parser.add_argument("--ckpt", required=True, type=Path)
    parser.add_argument("--mode", default="off", choices=["off", "on"], type=str)
    parser.add_argument("--source_trace_root", required=True, type=Path)
    parser.add_argument("--heldout_trace_root", required=True, type=Path)
    parser.add_argument("--heldout_task", required=True, type=str)
    parser.add_argument("--candidate_split", default="train", choices=["train", "test"], type=str)
    parser.add_argument("--probe_split", default="test", choices=["train", "test"], type=str)
    parser.add_argument("--candidate_limit", default=500, type=int)
    parser.add_argument("--probe_limit", default=500, type=int)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--candidate_batch_size", default=32, type=int)
    parser.add_argument("--image_batch_size", default=128, type=int)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument(
        "--selection_mode",
        default="representative",
        choices=["representative", "best", "worst"],
        help=(
            "How to choose the single source trace for each task from the candidate pool: "
            "'representative' picks the trace whose probe-margin vector is closest to the candidate mean; "
            "'best' picks the trace with the highest probe accuracy; "
            "'worst' picks the trace with the lowest probe accuracy."
        ),
    )
    parser.add_argument(
        "--real_on_selection_mode",
        default=None,
        choices=["representative", "best", "worst"],
        help=(
            "Optional override used only for the held-out real_on row. "
            "If omitted, real_on uses --selection_mode like every other source."
        ),
    )
    parser.add_argument(
        "--flip_polarity",
        action="store_true",
        help="Report polarity-corrected metrics by swapping the two logits, i.e. acc -> 1-acc and margin -> -margin.",
    )
    parser.add_argument(
        "--include_real_on",
        action="store_true",
        help="Also evaluate the held-out task using its on-trace representative and append it as a real_on row.",
    )
    parser.add_argument(
        "--source_include_heldout",
        action="store_true",
        help="Include task28_* sources from source_trace_root. Default keeps the original mismatch(28) rule.",
    )
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
    manifest_file = manifest_path(task_dir)
    if not manifest_file.exists():
        ensure_task_dataset_manifest(task_dir)
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    classes = manifest["classes"]
    return f"{int(classes[0])}vs{int(classes[1])}"


def sort_task_key(task_name: str) -> tuple[int, int]:
    tail = task_name.split("_", 1)[1]
    left, right = tail.split("vs")
    return int(left), int(right)


def build_dataset(cfg, task_dir: Path, mode: str, split: str, require_trace: bool):
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
        require_trace=require_trace,
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
    selection_mode: str,
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
    distances = np.linalg.norm(margin_matrix - center, axis=1)
    acc_values = np.asarray([float(acc) for _, acc in accs], dtype=np.float32)

    if selection_mode == "representative":
        rep_pos = int(np.argmin(distances))
    elif selection_mode == "best":
        max_acc = float(acc_values.max())
        best_positions = np.flatnonzero(np.isclose(acc_values, max_acc))
        if best_positions.size == 1:
            rep_pos = int(best_positions[0])
        else:
            rep_pos = int(best_positions[np.argmin(distances[best_positions])])
    elif selection_mode == "worst":
        min_acc = float(acc_values.min())
        worst_positions = np.flatnonzero(np.isclose(acc_values, min_acc))
        if worst_positions.size == 1:
            rep_pos = int(worst_positions[0])
        else:
            rep_pos = int(worst_positions[np.argmin(distances[worst_positions])])
    else:
        raise ValueError(f"Unsupported selection_mode={selection_mode}")

    rep_ordinal, rep_acc = accs[rep_pos]
    return {
        "rep_ordinal": rep_ordinal,
        "probe_acc": float(rep_acc),
        "margin_mean": float(margin_matrix[rep_pos].mean()),
        "selection_mode": str(selection_mode),
    }


def apply_reporting_polarity(result: dict, flip_polarity: bool) -> dict:
    raw_acc = float(result["probe_acc"])
    raw_margin = float(result["margin_mean"])
    if flip_polarity:
        probe_acc = 1.0 - raw_acc
        margin_mean = -raw_margin
    else:
        probe_acc = raw_acc
        margin_mean = raw_margin
    return {
        "raw_probe_acc": raw_acc,
        "raw_probe_acc_percent": 100.0 * raw_acc,
        "raw_margin_mean": raw_margin,
        "probe_acc": probe_acc,
        "probe_acc_percent": 100.0 * probe_acc,
        "margin_mean": margin_mean,
        "polarity_flip_applied": bool(flip_polarity),
    }


def render_acc_rank(
    rows,
    output_path: Path,
    heldout_task: str,
    heldout_pair: str,
    selection_mode: str,
    real_on_selection_mode: str | None,
):
    rows = sorted(rows, key=lambda row: float(row["probe_acc"]))
    fig_h = max(7.5, 0.36 * len(rows))
    fig, ax = plt.subplots(figsize=(12.5, fig_h), dpi=180)

    real_bands = {
        "real_off": "#dff3e4",
        "real_on": "#fce7d6",
    }
    for idx, row in enumerate(rows):
        band = real_bands.get(row["kind"])
        if band is not None:
            ax.axhspan(idx - 0.45, idx + 0.45, color=band, alpha=0.8, zorder=0)

    ys = np.arange(len(rows))
    xs = np.array([100.0 * float(row["probe_acc"]) for row in rows], dtype=np.float32)
    colors = []
    sizes = []
    for row in rows:
        if row["kind"] == "mismatch":
            colors.append("#2c95d3")
            sizes.append(32)
        elif row["kind"] == "real_off":
            colors.append("#2aa745")
            sizes.append(46)
        else:
            colors.append("#d97706")
            sizes.append(46)

    ax.scatter(xs, ys, c=colors, s=sizes, linewidths=0, zorder=3)
    ax.set_yticks(ys)
    ax.set_yticklabels([row["source_label"] for row in rows], fontsize=9)
    ax.set_xlabel("Accuracy (%)")
    ax.set_ylabel("Seed Source")
    ax.set_title(
        f"{heldout_task} ({heldout_pair}) unified full500 selected acc | real_off + real_on + mismatch(28)",
        fontsize=15,
        pad=12,
    )
    selection_text = (
        "representative-of-500 selection"
        if selection_mode == "representative"
        else "best-of-500 selection"
        if selection_mode == "best"
        else "worst-of-500 selection"
    )
    if real_on_selection_mode is not None and real_on_selection_mode != selection_mode:
        selection_text += f"; real_on uses {real_on_selection_mode}-of-500"
    ax.text(
        0.0,
        1.01,
        f"{selection_text}; reported accuracy uses the configured polarity convention",
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
    heldout_pair: str,
):
    ordered = sorted(rows, key=lambda row: float(row["probe_acc"]), reverse=True)
    lines = [
        "cross-domain unified full500 analysis",
        f"heldout_task={args.heldout_task}",
        f"heldout_pair={heldout_pair}",
        f"mode={args.mode}",
        f"device={cfg.device}",
        f"flip_polarity={bool(args.flip_polarity)}",
        f"include_real_on={bool(args.include_real_on)}",
        f"selection_mode={args.selection_mode}",
        f"real_on_selection_mode={args.real_on_selection_mode}",
        (
            f"selection=representative_behavior_row on full held-out {args.probe_limit}"
            if args.selection_mode == "representative"
            else f"selection=best_probe_acc_row on full held-out {args.probe_limit}"
            if args.selection_mode == "best"
            else f"selection=worst_probe_acc_row on full held-out {args.probe_limit}"
        ),
        f"candidate_split={args.candidate_split}",
        f"candidate_limit={args.candidate_limit}",
        f"probe_split={args.probe_split}",
        f"probe_limit={args.probe_limit}",
        f"acc_metric=single-model accuracy on full held-out {args.probe_limit} using selected representative row",
        "reported_acc=raw probe_acc" + (" flipped to 1-acc" if args.flip_polarity else ""),
        f"source_trace_root={args.source_trace_root}",
        f"heldout_trace_root={args.heldout_trace_root}",
        f"run_dir={args.run_dir}",
        f"ckpt={args.ckpt}",
        f"summary_csv={summary_csv}",
        f"representatives_csv={representatives_csv}",
        f"acc_plot={figure_png}",
        "top5="
        + ", ".join(
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
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = args.output_dir / "task_unified_full500_summary.csv"
    representatives_csv = args.output_dir / "task_unified_full500_representatives.csv"
    figure_png = args.output_dir / "task_unified_full500_acc_rank.png"
    meta_txt = args.output_dir / "task_unified_full500_meta.txt"

    import importlib

    models = importlib.import_module(f"models.{cfg.target_model}")
    model = models.HyperTheft(cfg)
    model.mixer.eval()
    for gen in model.generator.as_list():
        gen.eval()
    load_checkpoint(args.ckpt, model)

    heldout_dir = Path(args.heldout_trace_root) / args.heldout_task
    if not heldout_dir.exists():
        raise FileNotFoundError(f"Missing heldout task dir: {heldout_dir}")
    heldout_pair = task_pair_label(heldout_dir)

    probe_dataset = build_dataset(
        cfg,
        heldout_dir,
        args.mode,
        args.probe_split,
        require_trace=False,
    )
    probe_images, probe_labels = load_probe_data(probe_dataset, args.probe_limit)
    probe_labels_np = probe_labels.numpy()

    mismatch_task_dirs = _task_dirs(
        args.source_trace_root,
        include_heldout=bool(args.source_include_heldout),
    )
    mismatch_task_dirs = sorted(mismatch_task_dirs, key=lambda path: sort_task_key(path.name))
    rows = []

    analysis_jobs = [(task_dir, args.mode, False) for task_dir in mismatch_task_dirs]
    analysis_jobs.append((heldout_dir, args.mode, True))
    if args.include_real_on:
        analysis_jobs.append((heldout_dir, "on", True))

    for task_dir, candidate_mode, is_real in analysis_jobs:
        effective_selection_mode = args.selection_mode
        if is_real and candidate_mode == "on" and args.real_on_selection_mode is not None:
            effective_selection_mode = args.real_on_selection_mode
        source_dataset = build_dataset(
            cfg,
            task_dir,
            candidate_mode,
            args.candidate_split,
            require_trace=True,
        )
        ordinals = candidate_ordinals(source_dataset, args.candidate_limit)
        if not ordinals:
            raise RuntimeError(f"No candidate traces found for {task_dir.name} mode={candidate_mode}")
        result = representative_behavior_for_task(
            model=model,
            dataset=source_dataset,
            ordinals=ordinals,
            probe_images=probe_images,
            probe_labels_np=probe_labels_np,
            device=cfg.device,
            candidate_batch_size=args.candidate_batch_size,
            image_batch_size=args.image_batch_size,
            selection_mode=effective_selection_mode,
        )
        reported = apply_reporting_polarity(result, bool(args.flip_polarity))
        if is_real and candidate_mode == "on":
            source_label = "real_on"
            kind = "real_on"
        elif is_real:
            source_label = "real_off"
            kind = "real_off"
        else:
            source_label = task_dir.name
            kind = "mismatch"
        rows.append(
            {
                "source_task": task_dir.name,
                "source_pair": task_pair_label(task_dir),
                "source_label": source_label,
                "kind": kind,
                "candidate_mode": candidate_mode,
                "selection_mode": effective_selection_mode,
                "rep_ordinal": int(result["rep_ordinal"]),
                "probe_acc": float(reported["probe_acc"]),
                "probe_acc_percent": float(reported["probe_acc_percent"]),
                "margin_mean": float(reported["margin_mean"]),
                "raw_probe_acc": float(reported["raw_probe_acc"]),
                "raw_probe_acc_percent": float(reported["raw_probe_acc_percent"]),
                "raw_margin_mean": float(reported["raw_margin_mean"]),
                "polarity_flip_applied": bool(reported["polarity_flip_applied"]),
                "candidate_split": args.candidate_split,
                "candidate_count": len(ordinals),
                "probe_split": args.probe_split,
                "probe_count": int(min(args.probe_limit, len(probe_dataset))),
                "mode": args.mode,
                "heldout_task": args.heldout_task,
                "heldout_pair": heldout_pair,
            }
        )
        print(
            f"[{source_label}] rep_ordinal={result['rep_ordinal']} "
            f"raw_probe_acc={100.0 * float(result['probe_acc']):.2f}% "
            f"reported_probe_acc={100.0 * float(reported['probe_acc']):.2f}% "
            f"candidate_mode={candidate_mode} selection_mode={effective_selection_mode}",
            flush=True,
        )

        partial_ranked = sorted(rows, key=lambda row: float(row["probe_acc"]), reverse=True)
        partial_representatives = sorted(
            rows,
            key=lambda row: (
                2 if row["kind"] == "real_off" else 3 if row["kind"] == "real_on" else 1,
                sort_task_key(row["source_task"]),
                row["candidate_mode"],
            ),
        )
        write_csv(summary_csv, partial_ranked)
        write_csv(representatives_csv, partial_representatives)
        render_acc_rank(
            rows,
            figure_png,
            args.heldout_task,
            heldout_pair,
            args.selection_mode,
            args.real_on_selection_mode,
        )

    ranked_rows = sorted(rows, key=lambda row: float(row["probe_acc"]), reverse=True)
    representatives_rows = sorted(
        rows,
        key=lambda row: (
            2 if row["kind"] == "real_off" else 3 if row["kind"] == "real_on" else 1,
            sort_task_key(row["source_task"]),
            row["candidate_mode"],
        ),
    )
    write_csv(summary_csv, ranked_rows)
    write_csv(representatives_csv, representatives_rows)
    render_acc_rank(
        rows,
        figure_png,
        args.heldout_task,
        heldout_pair,
        args.selection_mode,
        args.real_on_selection_mode,
    )
    write_meta(meta_txt, args, cfg, rows, summary_csv, representatives_csv, figure_png, heldout_pair)

    print(summary_csv, flush=True)
    print(representatives_csv, flush=True)
    print(figure_png, flush=True)
    print(meta_txt, flush=True)


if __name__ == "__main__":
    main()
