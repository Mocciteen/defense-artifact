import argparse
import importlib.util
import math
import sys
import types
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.decomposition import PCA


ROOT = Path("/path/to/hypertheft_workspace/lenet+mnist")
SCRIPT_DIR = ROOT / "z_analyse"
DEFAULT_ANALYSIS_ROOT = SCRIPT_DIR / "069.pt"
DEFAULT_RUN_DIR = (
    ROOT
    / "hypertheft_lenet"
    / "runs"
    / "conv32_off_convmax_full_task28_bs100_ep200_20260420"
)
DEFAULT_CKPT = DEFAULT_RUN_DIR / "ckpt" / "069.pt"
HELDOUT_LABELS = ("8", "9")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render all-task coarse direction maps for HyperTheft task transfer analysis"
    )
    parser.add_argument("--analysis-root", type=Path, default=DEFAULT_ANALYSIS_ROOT)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--probe-limit", type=int, default=64)
    parser.add_argument("--patch-size", type=int, default=4)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def load_base_module():
    if "umap" not in sys.modules:
        fake_umap = types.ModuleType("umap")

        class _UnusedUMAP:
            def __init__(self, *args, **kwargs):
                raise RuntimeError("UMAP is unavailable in this environment and should not be used here")

        fake_umap.UMAP = _UnusedUMAP
        sys.modules["umap"] = fake_umap
    module_path = SCRIPT_DIR / "analyze_hypertheft_z.py"
    spec = importlib.util.spec_from_file_location("analyze_hypertheft_z", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def pick_device(device_arg: str) -> torch.device:
    if device_arg == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def task_sort_key(task_pair: str) -> tuple[int, int]:
    if "vs" not in task_pair:
        return (999, 999)
    left, right = task_pair.split("vs")
    return (int(left), int(right))


def short_task_label(task_pair: str, is_target: bool) -> str:
    if is_target:
        return f"{task_pair}\nheld-out"
    return task_pair


def margin_from_logits(logits: np.ndarray) -> np.ndarray:
    return logits[:, 1] - logits[:, 0]


def cosine_matrix(vectors: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-8
    normalized = vectors / denom
    return normalized @ normalized.T


def load_latent_frame(analysis_root: Path) -> pd.DataFrame:
    frame_path = analysis_root / "tables" / "latent_records_embedded.csv"
    if not frame_path.exists():
        raise FileNotFoundError(f"Missing latent frame: {frame_path}")
    return pd.read_csv(frame_path, low_memory=False)


def task_summary_map(analysis_root: Path) -> dict[str, dict]:
    summary_path = analysis_root / "tables" / "source_task_mechanism_069.csv"
    if not summary_path.exists():
        return {}
    summary_df = pd.read_csv(summary_path)
    out = {}
    for _, row in summary_df.iterrows():
        out[str(row["seed_task"])] = {
            "avg_acc": float(row["avg_acc"]),
            "min_acc": float(row["min_acc"]),
            "max_acc": float(row["max_acc"]),
            "plane_mae_to_real": float(row["plane_mae_to_real"]),
            "fc1_cos": float(row["fc1_cos"]),
            "fc2_cos": float(row["fc2_cos"]),
        }
    return out


def select_task_rows(frame: pd.DataFrame) -> list[tuple[str, str, pd.DataFrame, bool]]:
    train = frame[
        (frame["domain"] == "train-domain")
        & (frame["mode"] == "off")
        & (frame["trace_condition"] == "real")
    ].copy()
    groups = []
    for task_name, group in train.groupby("task_name"):
        group = group.sort_values("ordinal").reset_index(drop=True)
        task_pair = str(group.iloc[0]["task_pair"]).replace(" ", "")
        groups.append((task_name, task_pair, group, False))

    heldout = frame[
        (frame["domain"] == "held-out")
        & (frame["mode"] == "off")
        & (frame["trace_condition"] == "real")
        & (frame["task_name"] == "task28_8vs9")
    ].sort_values("ordinal").reset_index(drop=True)
    groups = sorted(groups, key=lambda item: task_sort_key(item[1]))
    groups.append(("task28_8vs9_real", "8vs9", heldout, True))
    return groups


def representative_behavior_row(
    group: pd.DataFrame,
    probe_images: torch.Tensor,
    probe_labels: torch.Tensor,
    base,
    model,
    device: torch.device,
) -> tuple[pd.Series, np.ndarray, float]:
    used = base.latent_used_blocks(group)
    margins = []
    accs = []
    for idx in range(used.shape[0]):
        logits = base.surrogate_logits(model, used[idx], probe_images, device)
        margin = margin_from_logits(logits)
        margins.append(margin)
        accs.append(float((logits.argmax(axis=1) == probe_labels.numpy()).mean()))
    margin_matrix = np.stack(margins, axis=0)
    center = margin_matrix.mean(axis=0, keepdims=True)
    rep_idx = int(np.argmin(np.linalg.norm(margin_matrix - center, axis=1)))
    return group.iloc[rep_idx], margin_matrix[rep_idx], accs[rep_idx]


def coarse_occlusion_map(
    base,
    model,
    used_codes: np.ndarray,
    images: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
    patch_size: int,
    stride: int,
) -> tuple[np.ndarray, np.ndarray]:
    baseline_logits = base.surrogate_logits(model, used_codes, images, device)
    baseline_margin = margin_from_logits(baseline_logits)
    h = int(images.shape[-2])
    w = int(images.shape[-1])
    rows = list(range(0, h - patch_size + 1, stride))
    cols = list(range(0, w - patch_size + 1, stride))
    signed = np.zeros((len(rows), len(cols)), dtype=np.float32)
    absolute = np.zeros((len(rows), len(cols)), dtype=np.float32)
    for i, y in enumerate(rows):
        for j, x in enumerate(cols):
            occluded = images.clone()
            occluded[:, :, y : y + patch_size, x : x + patch_size] = 0.0
            logits = base.surrogate_logits(model, used_codes, occluded, device)
            margin = margin_from_logits(logits)
            delta = baseline_margin - margin
            signed[i, j] = float(delta.mean())
            absolute[i, j] = float(np.abs(delta).mean())
    return signed, absolute


def render_similarity_heatmap(
    sim_df: pd.DataFrame,
    labels: list[str],
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(15, 13), dpi=180)
    sns.heatmap(
        sim_df,
        ax=ax,
        cmap="mako",
        square=True,
        cbar_kws={"label": "cosine similarity of absolute coarse-direction maps"},
        xticklabels=labels,
        yticklabels=labels,
    )
    ax.set_title("All binary tasks in one shared coarse-direction space")
    ax.set_xlabel("task")
    ax.set_ylabel("task")
    ax.tick_params(axis="x", labelrotation=90, labelsize=8)
    ax.tick_params(axis="y", labelrotation=0, labelsize=8)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def render_gallery(
    rep_df: pd.DataFrame,
    signed_maps: np.ndarray,
    output_path: Path,
) -> None:
    order = rep_df.sort_values(
        ["is_target", "abs_cos_to_8vs9", "task_pair"],
        ascending=[False, False, True],
    ).index.to_list()
    vmax = float(np.quantile(np.abs(signed_maps), 0.98) + 1e-8)
    n = len(order)
    cols = 5
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(4.2 * cols, 3.7 * rows),
        dpi=180,
        constrained_layout=True,
    )
    axes = np.asarray(axes).reshape(-1)
    for ax in axes[n:]:
        ax.axis("off")
    for plot_idx, row_idx in enumerate(order):
        ax = axes[plot_idx]
        row = rep_df.loc[row_idx]
        im = ax.imshow(
            signed_maps[row_idx],
            cmap="coolwarm",
            vmin=-vmax,
            vmax=vmax,
            interpolation="bicubic",
        )
        ax.set_xticks([])
        ax.set_yticks([])
        title = (
            f"{short_task_label(row['task_pair'], bool(row['is_target']))}\n"
            f"sim {row['abs_cos_to_8vs9']:.3f} | rep {row['probe_acc']:.3f}"
        )
        if not pd.isna(row["avg_acc"]):
            title += f" | avg {row['avg_acc']:.3f}"
        ax.set_title(title, fontsize=9.8, pad=5)
    cbar = fig.colorbar(im, ax=axes.tolist(), shrink=0.85, pad=0.01)
    cbar.set_label(
        f"signed margin-drop after patch occlusion\nnegative = {HELDOUT_LABELS[0]}-supporting, positive = {HELDOUT_LABELS[1]}-supporting"
    )
    fig.suptitle(
        "Representative surrogate per binary task: shared coarse discriminative structure on held-out 8-vs-9 probes",
        y=0.995,
        fontsize=18,
    )
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def render_basis(
    rep_df: pd.DataFrame,
    abs_maps: np.ndarray,
    output_path: Path,
) -> pd.DataFrame:
    flat = abs_maps.reshape(abs_maps.shape[0], -1)
    n_comp = min(4, flat.shape[0], flat.shape[1])
    pca = PCA(n_components=n_comp, random_state=1)
    scores = pca.fit_transform(flat)
    comps = pca.components_.reshape(n_comp, abs_maps.shape[1], abs_maps.shape[2])
    score_df = rep_df[["task_key", "task_pair", "is_target", "abs_cos_to_8vs9"]].copy()
    for idx in range(n_comp):
        score_df[f"pc{idx + 1}"] = scores[:, idx]

    order = score_df.sort_values(
        ["is_target", "abs_cos_to_8vs9", "task_pair"],
        ascending=[False, False, True],
    )["task_key"].tolist()
    score_order = score_df.set_index("task_key").loc[order].reset_index()
    heat_data = score_order[[f"pc{idx + 1}" for idx in range(n_comp)]]

    fig = plt.figure(figsize=(16, 10), dpi=180, constrained_layout=True)
    gs = fig.add_gridspec(2, n_comp, height_ratios=[1.0, 1.8], hspace=0.28, wspace=0.12)
    vmax = float(np.quantile(np.abs(comps), 0.98) + 1e-8)
    for idx in range(n_comp):
        ax = fig.add_subplot(gs[0, idx])
        ax.imshow(comps[idx], cmap="magma", interpolation="bicubic")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(
            f"PC{idx + 1}\nvar {pca.explained_variance_ratio_[idx]:.3f}",
            fontsize=11,
        )
    ax_heat = fig.add_subplot(gs[1, :])
    sns.heatmap(
        heat_data,
        ax=ax_heat,
        cmap="vlag",
        center=0.0,
        cbar_kws={"label": "task loading"},
        yticklabels=[
            short_task_label(pair, bool(is_target))
            for pair, is_target in zip(score_order["task_pair"], score_order["is_target"], strict=True)
        ],
    )
    ax_heat.set_title("All tasks projected onto shared coarse-direction basis")
    ax_heat.set_xlabel("coarse-direction principal component")
    ax_heat.set_ylabel("task")
    ax_heat.tick_params(axis="x", labelrotation=0)
    ax_heat.tick_params(axis="y", labelrotation=0, labelsize=8)
    fig.suptitle("Shared coarse-direction basis across all binary tasks", y=0.995, fontsize=18)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return score_df


def main() -> None:
    args = parse_args()
    device = pick_device(args.device)
    analysis_root = args.analysis_root
    plots_dir = analysis_root / "plots"
    tables_dir = analysis_root / "tables"
    plots_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="talk")

    base = load_base_module()
    latent = load_latent_frame(analysis_root)
    summary_map = task_summary_map(analysis_root)
    _, model = base.load_model(device, args.run_dir, args.ckpt)
    probe_images, probe_labels = base.heldout_probe_data(latent, limit=args.probe_limit)

    task_groups = select_task_rows(latent)
    rep_rows = []
    signed_maps = []
    abs_maps = []
    for task_name, task_pair, group, is_target in task_groups:
        rep_row, rep_margin, probe_acc = representative_behavior_row(
            group,
            probe_images,
            probe_labels,
            base,
            model,
            device,
        )
        used_codes = base.used_codes_from_row(rep_row)
        signed_map, abs_map = coarse_occlusion_map(
            base,
            model,
            used_codes,
            probe_images,
            probe_labels,
            device,
            patch_size=args.patch_size,
            stride=args.stride,
        )
        summary = summary_map.get(task_name, {})
        rep_rows.append(
            {
                "task_key": task_name,
                "task_pair": task_pair,
                "is_target": is_target,
                "rep_ordinal": int(rep_row["ordinal"]),
                "probe_acc": float(probe_acc),
                "margin_mean": float(rep_margin.mean()),
                "avg_acc": summary.get("avg_acc", np.nan),
                "min_acc": summary.get("min_acc", np.nan),
                "max_acc": summary.get("max_acc", np.nan),
                "plane_mae_to_real": summary.get("plane_mae_to_real", np.nan),
                "fc1_cos": summary.get("fc1_cos", np.nan),
                "fc2_cos": summary.get("fc2_cos", np.nan),
            }
        )
        signed_maps.append(signed_map)
        abs_maps.append(abs_map)

    rep_df = pd.DataFrame(rep_rows)
    signed_maps_np = np.stack(signed_maps, axis=0).astype(np.float32)
    abs_maps_np = np.stack(abs_maps, axis=0).astype(np.float32)
    abs_flat = abs_maps_np.reshape(abs_maps_np.shape[0], -1)
    signed_flat = signed_maps_np.reshape(signed_maps_np.shape[0], -1)
    abs_cos = cosine_matrix(abs_flat)
    signed_cos = cosine_matrix(signed_flat)
    target_idx = int(rep_df.index[rep_df["is_target"]].tolist()[0])
    rep_df["abs_cos_to_8vs9"] = abs_cos[:, target_idx]
    rep_df["signed_cos_to_8vs9"] = signed_cos[:, target_idx]
    rep_df = rep_df.sort_values(["is_target", "abs_cos_to_8vs9", "task_pair"], ascending=[False, False, True]).reset_index(drop=True)

    # Reorder matrices to match the saved representative table.
    row_order = []
    for _, row in rep_df.iterrows():
        source_idx = next(i for i, item in enumerate(rep_rows) if item["task_key"] == row["task_key"])
        row_order.append(source_idx)
    abs_cos_df = pd.DataFrame(
        abs_cos[np.ix_(row_order, row_order)],
        index=[short_task_label(pair, bool(is_target)) for pair, is_target in zip(rep_df["task_pair"], rep_df["is_target"], strict=True)],
        columns=[short_task_label(pair, bool(is_target)) for pair, is_target in zip(rep_df["task_pair"], rep_df["is_target"], strict=True)],
    )
    signed_cos_df = pd.DataFrame(
        signed_cos[np.ix_(row_order, row_order)],
        index=abs_cos_df.index,
        columns=abs_cos_df.columns,
    )
    ordered_signed_maps = signed_maps_np[row_order]
    ordered_abs_maps = abs_maps_np[row_order]

    rep_df.to_csv(tables_dir / "all_task_coarse_direction_representatives_069.csv", index=False)
    abs_cos_df.to_csv(tables_dir / "all_task_coarse_direction_abs_cosine_069.csv")
    signed_cos_df.to_csv(tables_dir / "all_task_coarse_direction_signed_cosine_069.csv")
    np.save(tables_dir / "all_task_coarse_direction_signed_maps_069.npy", ordered_signed_maps)
    np.save(tables_dir / "all_task_coarse_direction_abs_maps_069.npy", ordered_abs_maps)

    render_similarity_heatmap(
        abs_cos_df,
        abs_cos_df.columns.tolist(),
        plots_dir / "all_task_coarse_direction_similarity_069.png",
    )
    render_gallery(
        rep_df,
        ordered_signed_maps,
        plots_dir / "all_task_coarse_direction_gallery_069.png",
    )
    basis_df = render_basis(
        rep_df,
        ordered_abs_maps,
        plots_dir / "all_task_coarse_direction_basis_069.png",
    )
    basis_df.to_csv(tables_dir / "all_task_coarse_direction_basis_scores_069.csv", index=False)


if __name__ == "__main__":
    main()
