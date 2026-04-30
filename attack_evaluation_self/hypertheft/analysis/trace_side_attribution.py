import argparse
import importlib.util
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


ROOT = Path("/path/to/hypertheft_workspace/lenet+mnist")
DEFAULT_RUN_DIR = (
    ROOT
    / "hypertheft_lenet"
    / "runs"
    / "conv32_off_convmax_full_task28_bs100_ep200_20260420"
)
DEFAULT_CKPT = DEFAULT_RUN_DIR / "ckpt" / "069.pt"
DEFAULT_OUT_ROOT = ROOT / "z_analyse" / "069.pt"
TRACE_PATCH = 8
PATCH_SAMPLE_LIMIT = 32
COMPONENT_SAMPLE_LIMIT = 64


def parse_args():
    parser = argparse.ArgumentParser(description="Trace-side attribution for cross-model HyperTheft analysis")
    parser.add_argument("--run-dir", type=str, default=str(DEFAULT_RUN_DIR))
    parser.add_argument("--ckpt", type=str, default=str(DEFAULT_CKPT))
    parser.add_argument("--output-root", type=str, default=str(DEFAULT_OUT_ROOT))
    parser.add_argument("--patch-size", type=int, default=TRACE_PATCH)
    parser.add_argument("--patch-samples", type=int, default=PATCH_SAMPLE_LIMIT)
    parser.add_argument("--component-samples", type=int, default=COMPONENT_SAMPLE_LIMIT)
    return parser.parse_args()


def load_analysis_module():
    script = ROOT / "z_analyse" / "analyze_hypertheft_z.py"
    spec = importlib.util.spec_from_file_location("azh", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def infer_component_masks(sample_dir: Path, trace_shape: tuple[int, int, int]) -> dict[str, np.ndarray]:
    conv_bits = sorted(sample_dir.glob("conv*_bits.bin"))
    maxpool_bits = sorted(sample_dir.glob("maxpool*_bits.bin"))
    if not conv_bits or not maxpool_bits:
        raise FileNotFoundError(f"Expected conv/maxpool bit files under {sample_dir}")
    conv_len = int(conv_bits[0].stat().st_size * 8)
    maxpool_len = int(maxpool_bits[0].stat().st_size * 8)
    total_capacity = int(np.prod(trace_shape))
    role = np.full(total_capacity, 2, dtype=np.int64)  # 0=conv, 1=maxpool, 2=pad
    role[:conv_len] = 0
    role[conv_len : conv_len + maxpool_len] = 1
    role = role.reshape(trace_shape)
    return {
        "conv": role == 0,
        "maxpool": role == 1,
        "pad": role == 2,
    }


@torch.no_grad()
def encode_trace_used_codes(module, model, trace_array: np.ndarray, device: torch.device) -> np.ndarray:
    tensor = torch.from_numpy(trace_array.astype(np.float32)).unsqueeze(0).to(device)
    codes = model.mixer(tensor)[: module.USED_BLOCKS].permute(1, 0, 2).detach().cpu().numpy()
    return codes[0]


def z_mean_from_used_codes(used_codes: np.ndarray) -> np.ndarray:
    return used_codes.mean(axis=0).astype(np.float32)


def latent_recovery_fraction(z_h: np.ndarray, z_m: np.ndarray, z_r: np.ndarray) -> float:
    delta_r = z_r - z_m
    denom = float(np.dot(delta_r, delta_r))
    if denom < 1e-8:
        return 0.0
    delta_h = z_h - z_m
    return float(np.dot(delta_h, delta_r) / denom)


def mean_over_patches(mask: np.ndarray, patch: int) -> np.ndarray:
    c, h, w = mask.shape
    out_h = h // patch
    out_w = w // patch
    values = np.zeros((out_h, out_w), dtype=np.float32)
    for i in range(out_h):
        for j in range(out_w):
            block = mask[:, i * patch : (i + 1) * patch, j * patch : (j + 1) * patch]
            values[i, j] = float(block.mean())
    return values


def main():
    cli = parse_args()
    module = load_analysis_module()
    out_root = Path(cli.output_root)
    plots_dir = out_root / "plots"
    tables_dir = out_root / "tables"
    cache_dir = out_root / "cache"
    plots_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, model = module.load_model(device, Path(cli.run_dir), Path(cli.ckpt))

    latent = pd.read_csv(tables_dir / "latent_records_embedded.csv", low_memory=False)
    real = latent[
        (latent["domain"] == "held-out")
        & (latent["mode"] == "off")
        & (latent["trace_condition"] == "real")
    ].sort_values("ordinal").reset_index(drop=True)
    mismatch = latent[
        (latent["domain"] == "held-out")
        & (latent["mode"] == "off")
        & (latent["trace_condition"] == "train28-mismatch")
    ].sort_values("ordinal").reset_index(drop=True)
    if real.shape[0] != mismatch.shape[0]:
        raise ValueError("Real and mismatch held-out sets must align by ordinal")

    selected = pd.read_csv(tables_dir / "causal_selected_ordinals.csv")
    component_ordinals = selected["ordinal"].astype(int).tolist()[: cli.component_samples]
    patch_ordinals = component_ordinals[: cli.patch_samples]
    ordinal_to_real_idx = {int(v): i for i, v in enumerate(real["ordinal"].tolist())}

    probe_images, probe_labels = module.heldout_probe_data(latent, limit=module.CAUSAL_PROBE_LIMIT)
    labels_np = probe_labels.numpy()

    trace_shape = (module.TRACE_C, module.TRACE_W, module.TRACE_W)
    sample_dir = Path(mismatch.iloc[0]["sample_dir"])
    masks = infer_component_masks(sample_dir, trace_shape)
    conv_density = mean_over_patches(masks["conv"].astype(np.float32), cli.patch_size)
    maxpool_density = mean_over_patches(masks["maxpool"].astype(np.float32), cli.patch_size)

    component_cache = tables_dir / "trace_component_contribution.csv"
    if component_cache.exists():
        component_df = pd.read_csv(component_cache)
    else:
        rows = []
        for ordinal in component_ordinals:
            idx = ordinal_to_real_idx[int(ordinal)]
            real_row = real.iloc[idx].to_dict()
            mismatch_row = mismatch.iloc[idx].to_dict()
            real_trace = module.load_trace_array(real_row)
            mismatch_trace = module.load_trace_array(mismatch_row)
            real_z = module.latent_mean_matrix(real.iloc[[idx]])[0]
            mismatch_z = module.latent_mean_matrix(mismatch.iloc[[idx]])[0]
            mismatch_used = module.latent_used_blocks(mismatch.iloc[[idx]])[0]
            mismatch_logits = module.surrogate_logits(model, mismatch_used, probe_images, device)
            mismatch_metrics = module.class_aware_metrics_from_logits(mismatch_logits, labels_np)
            conditions = {
                "mismatch": mismatch_trace,
                "conv_only": np.where(masks["conv"], real_trace, mismatch_trace),
                "maxpool_only": np.where(masks["maxpool"], real_trace, mismatch_trace),
                "real": real_trace,
            }
            for name, trace in conditions.items():
                used_codes = encode_trace_used_codes(module, model, trace, device)
                z_mean = z_mean_from_used_codes(used_codes)
                logits = module.surrogate_logits(model, used_codes, probe_images, device)
                metrics = module.class_aware_metrics_from_logits(logits, labels_np)
                rows.append(
                    {
                        "ordinal": int(ordinal),
                        "condition": name,
                        "latent_recovery": latent_recovery_fraction(z_mean, mismatch_z, real_z),
                        "score_sep_gain": float(metrics["score_sep"] - mismatch_metrics["score_sep"]),
                        "acc_gain": float(metrics["acc"] - mismatch_metrics["acc"]),
                        "score8_gain": float(metrics["score8"] - mismatch_metrics["score8"]),
                        "score9_gain": float(metrics["score9"] - mismatch_metrics["score9"]),
                    }
                )
        component_df = pd.DataFrame(rows)
        component_df.to_csv(component_cache, index=False)

    component_summary = (
        component_df.groupby("condition")[["latent_recovery", "score_sep_gain", "acc_gain", "score8_gain", "score9_gain"]]
        .agg(["mean", "std"])
        .reset_index()
    )
    component_summary.columns = [
        "condition" if col[0] == "condition" else f"{col[0]}_{col[1]}"
        for col in component_summary.columns.to_flat_index()
    ]
    component_summary.to_csv(tables_dir / "trace_component_contribution_summary.csv", index=False)

    patch_cache = tables_dir / f"trace_patch_correction_patch{cli.patch_size}.csv"
    if patch_cache.exists():
        patch_df = pd.read_csv(patch_cache)
    else:
        rows = []
        out_h = module.TRACE_W // cli.patch_size
        out_w = module.TRACE_W // cli.patch_size
        for ordinal in patch_ordinals:
            idx = ordinal_to_real_idx[int(ordinal)]
            real_row = real.iloc[idx].to_dict()
            mismatch_row = mismatch.iloc[idx].to_dict()
            real_trace = module.load_trace_array(real_row)
            mismatch_trace = module.load_trace_array(mismatch_row)
            real_z = module.latent_mean_matrix(real.iloc[[idx]])[0]
            mismatch_z = module.latent_mean_matrix(mismatch.iloc[[idx]])[0]
            mismatch_used = module.latent_used_blocks(mismatch.iloc[[idx]])[0]
            mismatch_logits = module.surrogate_logits(model, mismatch_used, probe_images, device)
            mismatch_metrics = module.class_aware_metrics_from_logits(mismatch_logits, labels_np)
            for i in range(out_h):
                for j in range(out_w):
                    hybrid = mismatch_trace.copy()
                    rs = slice(i * cli.patch_size, (i + 1) * cli.patch_size)
                    cs = slice(j * cli.patch_size, (j + 1) * cli.patch_size)
                    hybrid[:, rs, cs] = real_trace[:, rs, cs]
                    used_codes = encode_trace_used_codes(module, model, hybrid, device)
                    z_mean = z_mean_from_used_codes(used_codes)
                    logits = module.surrogate_logits(model, used_codes, probe_images, device)
                    metrics = module.class_aware_metrics_from_logits(logits, labels_np)
                    rows.append(
                        {
                            "ordinal": int(ordinal),
                            "patch_row": i,
                            "patch_col": j,
                            "latent_recovery": latent_recovery_fraction(z_mean, mismatch_z, real_z),
                            "score_sep_gain": float(metrics["score_sep"] - mismatch_metrics["score_sep"]),
                            "acc_gain": float(metrics["acc"] - mismatch_metrics["acc"]),
                        }
                    )
        patch_df = pd.DataFrame(rows)
        patch_df.to_csv(patch_cache, index=False)

    patch_summary = (
        patch_df.groupby(["patch_row", "patch_col"])[["latent_recovery", "score_sep_gain", "acc_gain"]]
        .mean()
        .reset_index()
    )
    patch_summary.to_csv(tables_dir / f"trace_patch_correction_summary_patch{cli.patch_size}.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.6), dpi=180)
    order = ["conv_only", "maxpool_only", "real"]
    labels = {"conv_only": "real conv bits", "maxpool_only": "real maxpool bits", "real": "full real trace"}
    colors = {"conv_only": "#3a7ca5", "maxpool_only": "#81b29a", "real": "#c1121f"}
    metrics = [
        ("latent_recovery_mean", "latent recovery"),
        ("score_sep_gain_mean", "separation gain"),
        ("acc_gain_mean", "probe accuracy gain"),
    ]
    summary_ordered = component_summary.set_index("condition").loc[order].reset_index()
    for ax, (metric, title) in zip(axes, metrics, strict=True):
        errs = summary_ordered[f"{metric.split('_mean')[0]}_std"].to_numpy(dtype=np.float32)
        vals = summary_ordered[metric].to_numpy(dtype=np.float32)
        ax.bar(
            [labels[name] for name in order],
            vals,
            yerr=errs,
            capsize=4,
            color=[colors[name] for name in order],
            alpha=0.92,
        )
        ax.axhline(0.0, color="#999999", linewidth=1.0)
        ax.set_title(title)
        ax.set_ylabel(title)
        ax.tick_params(axis="x", rotation=12)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.suptitle(
        "Trace-component contribution relative to mismatch baseline",
        y=0.985,
        fontsize=20,
    )
    fig.text(
        0.5,
        0.95,
        f"Each condition starts from mismatch and replaces only the selected trace component with the real online lenet trace, aggregated over top {len(component_ordinals)} ordinals",
        ha="center",
        va="center",
        fontsize=11.5,
        color="#444444",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(plots_dir / "trace_component_contribution.png")
    plt.close(fig)

    heat_shape = conv_density.shape
    latent_recovery_map = np.zeros(heat_shape, dtype=np.float32)
    score_sep_map = np.zeros(heat_shape, dtype=np.float32)
    acc_gain_map = np.zeros(heat_shape, dtype=np.float32)
    for _, row in patch_summary.iterrows():
        i = int(row["patch_row"])
        j = int(row["patch_col"])
        latent_recovery_map[i, j] = float(row["latent_recovery"])
        score_sep_map[i, j] = float(row["score_sep_gain"])
        acc_gain_map[i, j] = float(row["acc_gain"])

    fig, axes = plt.subplots(2, 2, figsize=(12.4, 11.2), dpi=180)
    im0 = axes[0, 0].imshow(conv_density, cmap="Blues", origin="upper")
    axes[0, 0].set_title("conv-bit density in folded trace")
    im1 = axes[0, 1].imshow(maxpool_density, cmap="Greens", origin="upper")
    axes[0, 1].set_title("maxpool-bit density in folded trace")
    im2 = axes[1, 0].imshow(latent_recovery_map, cmap="magma", origin="upper")
    axes[1, 0].set_title("patch-wise latent correction recovery")
    im3 = axes[1, 1].imshow(score_sep_map, cmap="coolwarm", origin="upper")
    axes[1, 1].set_title("patch-wise separation gain")
    for ax in axes.ravel():
        ax.set_xlabel(f"patch col (size={cli.patch_size})")
        ax.set_ylabel(f"patch row (size={cli.patch_size})")
        ax.set_xticks(range(heat_shape[1]))
        ax.set_yticks(range(heat_shape[0]))
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.colorbar(im0, ax=axes[0, 0], shrink=0.82)
    fig.colorbar(im1, ax=axes[0, 1], shrink=0.82)
    fig.colorbar(im2, ax=axes[1, 0], shrink=0.82)
    fig.colorbar(im3, ax=axes[1, 1], shrink=0.82)
    fig.suptitle(
        "Where does the real online trace inject corrective signal?",
        y=0.985,
        fontsize=20,
    )
    fig.text(
        0.5,
        0.95,
        f"Each patch experiment starts from mismatch and copies only one {cli.patch_size}x{cli.patch_size} folded-trace patch from the real online lenet trace, aggregated over top {len(patch_ordinals)} ordinals",
        ha="center",
        va="center",
        fontsize=11.5,
        color="#444444",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(plots_dir / f"trace_patch_correction_patch{cli.patch_size}.png")
    plt.close(fig)

    print(str(out_root))


if __name__ == "__main__":
    main()
