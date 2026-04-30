import argparse
import importlib.util
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
LAYER_NAMES = ["conv1", "conv2", "conv3", "fc1", "fc2"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze local decoder geometry from latent codes to generated conv surrogates"
    )
    parser.add_argument("--analysis-root", type=Path, default=DEFAULT_ANALYSIS_ROOT)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--probe-limit-grad", type=int, default=128)
    parser.add_argument("--interp-steps", type=int, default=11)
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


def locate_rep_latent_row(latent: pd.DataFrame, row: pd.Series) -> pd.Series:
    if bool(row["is_target"]):
        matches = latent[
            (latent["domain"] == "held-out")
            & (latent["mode"] == "off")
            & (latent["trace_condition"] == "real")
            & (latent["task_name"] == "task28_8vs9")
            & (latent["ordinal"] == int(row["rep_ordinal"]))
        ]
    else:
        matches = latent[
            (latent["domain"] == "train-domain")
            & (latent["mode"] == "off")
            & (latent["trace_condition"] == "real")
            & (latent["task_name"] == row["task_key"])
            & (latent["ordinal"] == int(row["rep_ordinal"]))
        ]
    if matches.empty:
        raise KeyError(f"Missing representative latent row for {row['task_key']} / {row['rep_ordinal']}")
    return matches.iloc[0]


def clone_layers(layers: tuple[torch.Tensor, ...]) -> list[torch.Tensor]:
    return [tensor.detach().clone() for tensor in layers]


def eval_layers(base, model, layers: tuple[torch.Tensor, ...], images: torch.Tensor, labels: torch.Tensor, device: torch.device) -> dict:
    logits = base.surrogate_logits_from_layers(model, layers, images, device)
    return base.class_aware_metrics_from_logits(logits, labels.numpy())


def block_codes_tensor(base, row: pd.Series, device: torch.device, requires_grad: bool = False) -> torch.Tensor:
    used_codes = base.used_codes_from_row(row).astype(np.float32)
    tensor = torch.from_numpy(used_codes).unsqueeze(1).to(device)
    if requires_grad:
        tensor.requires_grad_(True)
    return tensor


def single_layers_from_code(model, code_tensor: torch.Tensor) -> tuple[torch.Tensor, ...]:
    params = model.generator(code_tensor)
    return tuple(param[0] for param in params)


def differentiable_score_sep(model, layers: tuple[torch.Tensor, ...], images8: torch.Tensor, images9: torch.Tensor, device: torch.device) -> torch.Tensor:
    out8 = model.eval_f(layers, images8.to(device))
    out9 = model.eval_f(layers, images9.to(device))
    score8 = (out8[:, 0] - out8[:, 1]).mean()
    score9 = (out9[:, 1] - out9[:, 0]).mean()
    return score8 + score9


def layer_relative_distance(src: torch.Tensor, tgt: torch.Tensor) -> float:
    num = torch.norm(src - tgt).item()
    den = torch.norm(tgt).item() + 1e-8
    return float(num / den)


def interpolation_analysis(
    base,
    model,
    focus_rows: dict[str, pd.Series],
    images: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
    steps: int,
) -> pd.DataFrame:
    target_name = "8vs9"
    target_code = base.used_codes_from_row(focus_rows[target_name]).astype(np.float32)
    rows = []
    for source_name in ["1vs4", "2vs4"]:
        source_code = base.used_codes_from_row(focus_rows[source_name]).astype(np.float32)
        for alpha in np.linspace(0.0, 1.0, steps):
            code = (1.0 - alpha) * source_code + alpha * target_code
            layers = base.generated_single_layers(model, code, device)
            metrics = eval_layers(base, model, layers, images, labels, device)
            target_layers = base.generated_single_layers(model, target_code, device)
            per_layer = {}
            for layer_idx, layer_name in enumerate(LAYER_NAMES):
                w_idx = 2 * layer_idx
                b_idx = 2 * layer_idx + 1
                per_layer[f"{layer_name}_w_rel_to_target"] = layer_relative_distance(layers[w_idx], target_layers[w_idx])
                per_layer[f"{layer_name}_b_rel_to_target"] = layer_relative_distance(layers[b_idx], target_layers[b_idx])
            rows.append(
                {
                    "source_task": source_name,
                    "target_task": target_name,
                    "alpha": float(alpha),
                    **metrics,
                    **per_layer,
                }
            )
    return pd.DataFrame(rows)


def swap_analysis(
    base,
    model,
    focus_rows: dict[str, pd.Series],
    images: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
) -> pd.DataFrame:
    target_name = "8vs9"
    target_layers = base.generated_single_layers(model, base.used_codes_from_row(focus_rows[target_name]).astype(np.float32), device)
    rows = []
    for source_name in ["1vs4", "2vs4"]:
        source_layers = base.generated_single_layers(model, base.used_codes_from_row(focus_rows[source_name]).astype(np.float32), device)
        source_metrics = eval_layers(base, model, source_layers, images, labels, device)
        rows.append(
            {
                "source_task": source_name,
                "swap_type": "baseline",
                "layer_name": "all",
                **source_metrics,
            }
        )
        target_metrics = eval_layers(base, model, target_layers, images, labels, device)
        rows.append(
            {
                "source_task": source_name,
                "swap_type": "target_all",
                "layer_name": "all",
                **target_metrics,
            }
        )
        for layer_idx, layer_name in enumerate(LAYER_NAMES):
            w_idx = 2 * layer_idx
            b_idx = 2 * layer_idx + 1
            for swap_type in ["weight_only", "bias_only", "weight_and_bias"]:
                swapped = clone_layers(source_layers)
                if swap_type in ["weight_only", "weight_and_bias"]:
                    swapped[w_idx] = target_layers[w_idx].detach().clone()
                if swap_type in ["bias_only", "weight_and_bias"]:
                    swapped[b_idx] = target_layers[b_idx].detach().clone()
                metrics = eval_layers(base, model, tuple(swapped), images, labels, device)
                rows.append(
                    {
                        "source_task": source_name,
                        "swap_type": swap_type,
                        "layer_name": layer_name,
                        "delta_acc_vs_source": metrics["acc"] - source_metrics["acc"],
                        "delta_score_sep_vs_source": metrics["score_sep"] - source_metrics["score_sep"],
                        **metrics,
                    }
                )
    return pd.DataFrame(rows)


def local_gradient_analysis(
    base,
    model,
    focus_rows: dict[str, pd.Series],
    images8: torch.Tensor,
    images9: torch.Tensor,
    device: torch.device,
) -> pd.DataFrame:
    rows = []
    for task_name, row in focus_rows.items():
        code = block_codes_tensor(base, row, device, requires_grad=True)
        layers = single_layers_from_code(model, code)
        score_sep = differentiable_score_sep(model, layers, images8, images9, device)
        grad = torch.autograd.grad(score_sep, code, retain_graph=False)[0].detach().cpu().numpy()[:, 0, :]
        for layer_idx, layer_name in enumerate(LAYER_NAMES):
            block = grad[layer_idx]
            rows.append(
                {
                    "task_pair": task_name,
                    "layer_name": layer_name,
                    "grad_l2": float(np.linalg.norm(block)),
                    "grad_l1": float(np.abs(block).sum()),
                    "grad_max_abs": float(np.abs(block).max()),
                }
            )
    return pd.DataFrame(rows)


def render_interpolation(df: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.8, 8.6), dpi=180, constrained_layout=True)
    metric_specs = [
        ("acc", "Held-out accuracy"),
        ("score_sep", "Held-out score separation"),
        ("recall8", "Recall on digit 8"),
        ("recall9", "Recall on digit 9"),
    ]
    colors = {"1vs4": "#bc6c25", "2vs4": "#33658a"}
    for ax, (metric, title) in zip(axes.ravel(), metric_specs, strict=True):
        for source_task, group in df.groupby("source_task"):
            ax.plot(group["alpha"], group[metric], marker="o", label=source_task, color=colors[source_task])
        ax.set_title(title)
        ax.set_xlabel("interpolation alpha toward 8vs9 code")
        ax.set_ylabel(metric)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0, 0].legend(frameon=False)
    fig.suptitle("Code-space interpolation toward held-out 8vs9: is the basin smooth?", y=1.01, fontsize=17)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def render_swaps(df: pd.DataFrame, output_path: Path) -> None:
    subset = df[df["swap_type"].isin(["weight_only", "bias_only", "weight_and_bias"])].copy()
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), dpi=180, constrained_layout=True)
    for ax, source_task in zip(axes, ["1vs4", "2vs4"], strict=True):
        part = subset[subset["source_task"] == source_task].copy()
        pivot = part.pivot(index="layer_name", columns="swap_type", values="delta_acc_vs_source")
        pivot = pivot.loc[LAYER_NAMES]
        sns.heatmap(
            pivot,
            ax=ax,
            cmap="vlag",
            center=0.0,
            annot=True,
            fmt=".3f",
            cbar=ax is axes[-1],
            cbar_kws={"label": "delta acc vs source"} if ax is axes[-1] else None,
        )
        ax.set_title(f"{source_task} -> swap one target layer from 8vs9")
        ax.set_xlabel("swap mode")
        ax.set_ylabel("layer")
    fig.suptitle("Which layer and which parameter type pull a source surrogate toward 8vs9?", y=1.03, fontsize=17)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def render_gradients(df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 4.8), dpi=180, constrained_layout=True)
    pivot = df.pivot(index="layer_name", columns="task_pair", values="grad_l2").loc[LAYER_NAMES]
    sns.heatmap(
        pivot,
        ax=ax,
        cmap="mako",
        annot=True,
        fmt=".3f",
        cbar_kws={"label": "|| d score_sep / d layer-code ||_2"},
    )
    ax.set_title("Local gradient of held-out task score with respect to each layer code block")
    ax.set_xlabel("representative task")
    ax.set_ylabel("layer code block")
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    sns.set_theme(style="whitegrid", context="talk")
    device = pick_device(args.device)
    base = load_base_module()
    _, model = base.load_model(device, args.run_dir, args.ckpt)

    latent = pd.read_csv(args.analysis_root / "tables" / "latent_records_embedded.csv", low_memory=False)
    coeff_df = pd.read_csv(args.analysis_root / "tables" / "class_conditional_basis_coeffs_069_k4.csv")
    focus_df = coeff_df[coeff_df["task_pair"].isin(["8vs9", "1vs4", "2vs4"])].copy()
    focus_rows = {}
    for _, row in focus_df.iterrows():
        focus_rows[str(row["task_pair"])] = locate_rep_latent_row(latent, row)

    eval_images, eval_labels = base.heldout_eval_data(latent)
    probe_images, probe_labels = base.heldout_probe_data(latent, limit=args.probe_limit_grad)
    labels_np = probe_labels.numpy()
    images8 = probe_images[labels_np == 0]
    images9 = probe_images[labels_np == 1]

    interp_df = interpolation_analysis(
        base,
        model,
        focus_rows,
        eval_images,
        eval_labels,
        device,
        args.interp_steps,
    )
    swap_df = swap_analysis(
        base,
        model,
        focus_rows,
        eval_images,
        eval_labels,
        device,
    )
    grad_df = local_gradient_analysis(
        base,
        model,
        focus_rows,
        images8,
        images9,
        device,
    )

    tables_dir = args.analysis_root / "tables"
    plots_dir = args.analysis_root / "plots"
    interp_df.to_csv(tables_dir / "decoder_local_geometry_interpolation_069.csv", index=False)
    swap_df.to_csv(tables_dir / "decoder_local_geometry_swaps_069.csv", index=False)
    grad_df.to_csv(tables_dir / "decoder_local_geometry_gradients_069.csv", index=False)

    render_interpolation(interp_df, plots_dir / "decoder_local_geometry_interpolation_069.png")
    render_swaps(swap_df, plots_dir / "decoder_local_geometry_swaps_069.png")
    render_gradients(grad_df, plots_dir / "decoder_local_geometry_gradients_069.png")


if __name__ == "__main__":
    main()
