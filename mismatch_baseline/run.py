from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from trace_suite import DATASET_RATIO_RANGES, generate_traces, parse_csv, parse_shape
else:
    from .trace_suite import DATASET_RATIO_RANGES, generate_traces, parse_csv, parse_shape

ARTIFACT_ROOT = Path(__file__).resolve().parents[1]
if str(ARTIFACT_ROOT) not in sys.path:
    sys.path.insert(0, str(ARTIFACT_ROOT))


DATASETS = {
    "generic": {"num_classes": 0, "task": "auto"},
    "mnist": {"num_classes": 10, "task": "multiclass"},
    "cifar": {"num_classes": 10, "task": "multiclass"},
    "chest": {"num_classes": 14, "task": "multilabel"},
    "imagenet50": {"num_classes": 50, "task": "multiclass"},
}


@dataclass
class MismatchCorpus:
    traces: torch.Tensor
    images: torch.Tensor | None = None
    targets: torch.Tensor | None = None
    labels: torch.Tensor | None = None
    ids: list[str] | None = None
    image_shape: tuple[int, int, int] | None = None
    num_classes: int | None = None
    task: str | None = None
    dataset: str = "generic"
    protocol: dict | None = None


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def optional_tensor(item: dict, *keys: str) -> torch.Tensor | None:
    for key in keys:
        if key in item and item[key] is not None:
            return torch.as_tensor(item[key])
    return None


def parse_image_shape(raw: str) -> tuple[int, int, int] | None:
    shape = parse_shape(raw)
    if shape is None:
        return None
    if len(shape) != 3:
        raise ValueError(f"image shape must be C,H,W, got {raw!r}")
    return shape  # type: ignore[return-value]


def ids_from_payload(item: dict, count: int) -> list[str] | None:
    values = item.get("ids", item.get("trace_ids", item.get("sample_ids")))
    if values is None:
        return None
    ids = [str(value) for value in values]
    if len(ids) != count:
        raise ValueError(f"ids length {len(ids)} does not match trace count {count}")
    return ids


def load_mismatch_corpus(path: str | Path, split: str = "test") -> MismatchCorpus:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError("mismatch corpus must be a torch-saved dict")
    item = payload.get(split, payload)
    if not isinstance(item, dict) or "traces" not in item:
        raise KeyError(f"mismatch corpus must contain traces at top level or split {split!r}")

    traces = torch.as_tensor(item["traces"]).float()
    images = optional_tensor(item, "images", "image")
    targets = optional_tensor(item, "targets", "target")
    labels = optional_tensor(item, "gt_labels", "labels")
    image_shape = item.get("image_shape", payload.get("image_shape"))
    if image_shape is not None:
        image_shape = tuple(int(value) for value in image_shape)
    elif images is not None:
        image_shape = tuple(int(value) for value in images.shape[1:])

    return MismatchCorpus(
        traces=traces,
        images=images.float() if images is not None else None,
        targets=targets,
        labels=labels,
        ids=ids_from_payload(item, traces.shape[0]),
        image_shape=image_shape,
        num_classes=item.get("num_classes", payload.get("num_classes")),
        task=item.get("task", payload.get("task")),
        dataset=str(item.get("dataset", payload.get("dataset", "generic"))),
        protocol=payload.get("protocol") if isinstance(payload.get("protocol"), dict) else None,
    )


def trace_loader(corpus: MismatchCorpus, batch_size: int) -> DataLoader:
    return DataLoader(TensorDataset(corpus.traces), batch_size=batch_size, shuffle=False)


def generate(args: argparse.Namespace) -> None:
    trace_shape = parse_shape(args.trace_shape)
    traces, ids, ratios, protocol = generate_traces(
        raw_bit_count=args.raw_bit_count,
        sample_stride=args.sample_stride,
        dataset=args.dataset,
        ratio_low=args.ratio_low,
        ratio_high=args.ratio_high,
        trace_ids=parse_csv(args.trace_ids),
        trace_shape=trace_shape,
    )
    image_shape = parse_image_shape(args.image_shape)
    payload: dict[str, Any] = {
        "traces": traces,
        "ids": ids,
        "ratios": ratios,
        "dataset": args.dataset,
        "task": None if args.task == "auto" else args.task,
        "num_classes": args.num_classes or DATASETS[args.dataset]["num_classes"],
        "image_shape": image_shape,
        "protocol": protocol,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.out)
    print(str(args.out))


def summarize(args: argparse.Namespace) -> None:
    corpus = load_mismatch_corpus(args.corpus, args.split)
    traces = corpus.traces.float().flatten(1)
    ids = corpus.ids or [f"trace_{index:04d}" for index in range(traces.size(0))]
    ratios = traces.mean(dim=1)
    rows = [
        {
            "id": ids[index],
            "bit_count": int(traces.size(1)),
            "ones": int(traces[index].sum().item()),
            "ones_ratio": float(ratios[index].item()),
        }
        for index in range(traces.size(0))
    ]
    summary = {
        "count": int(ratios.numel()),
        "mean": float(ratios.mean().item()),
        "min": float(ratios.min().item()),
        "max": float(ratios.max().item()),
        "std": float(ratios.std(unbiased=False).item()) if ratios.numel() > 1 else 0.0,
        "dataset": corpus.dataset,
        "protocol": corpus.protocol,
    }
    write_csv(Path(args.output_dir) / "trace_ratios.csv", rows)
    write_json(Path(args.output_dir) / "summary.json", summary)
    print(str(args.output_dir))


def checkpoint_args(checkpoint: dict) -> dict:
    args = checkpoint.get("args", {})
    return args if isinstance(args, dict) else {}


def recovery_image_shape(corpus: MismatchCorpus, args: argparse.Namespace) -> tuple[int, int, int]:
    cli_shape = parse_image_shape(args.image_shape)
    if cli_shape is not None:
        return cli_shape
    if corpus.image_shape is not None:
        return corpus.image_shape
    raise ValueError("ciphersteal recovery needs image_shape metadata or --image-shape C,H,W")


def infer_recovery_dims(checkpoint: dict, args: argparse.Namespace) -> tuple[int, int, int]:
    saved = checkpoint_args(checkpoint)
    hidden_dim = int(args.hidden_dim or saved.get("hidden_dim", 0))
    latent_dim = int(args.latent_dim or saved.get("latent_dim", 0))
    refiner_channels = int(args.refiner_channels or saved.get("refiner_channels", 0))
    trace_state = checkpoint.get("trace_decoder", {})
    refiner_state = checkpoint.get("image_refiner") or {}
    if not hidden_dim and "net.0.weight" in trace_state:
        hidden_dim = int(trace_state["net.0.weight"].shape[0])
    if not latent_dim and "net.2.weight" in trace_state:
        latent_dim = int(trace_state["net.2.weight"].shape[0])
    if not refiner_channels and "net.0.weight" in refiner_state:
        refiner_channels = int(refiner_state["net.0.weight"].shape[0])
    if hidden_dim <= 0 or latent_dim <= 0 or (args.stage == "ti" and refiner_channels <= 0):
        raise ValueError("cannot infer recovery model dimensions; provide --hidden-dim, --latent-dim, and --refiner-channels")
    return hidden_dim, latent_dim, max(refiner_channels, 1)


def recovery_metrics(predictions: torch.Tensor, targets: torch.Tensor) -> dict[str, float]:
    mse = F.mse_loss(predictions, targets).item()
    l1 = F.l1_loss(predictions, targets).item()
    return {"mse": mse, "l1": l1, "psnr": -10.0 * math.log10(max(mse, 1e-12))}


def ciphersteal_recover(args: argparse.Namespace) -> None:
    from trace_to_image_pip.gan_prior import apply_gan_projection, load_gan_prior
    from trace_to_image_pip.recovery import AttackModels, ImageRefiner, TraceDecoder, load_checkpoint, predict, require_gan_projection

    if args.stage == "ti":
        require_gan_projection(args)
    corpus = load_mismatch_corpus(args.corpus, args.split)
    device = resolve_device(args.device)
    image_shape = recovery_image_shape(corpus, args)
    checkpoint = torch.load(args.ckpt, map_location=device, weights_only=False)
    hidden_dim, latent_dim, refiner_channels = infer_recovery_dims(checkpoint, args)
    models = AttackModels(
        trace_decoder=TraceDecoder(int(corpus.traces[0].numel()), image_shape, hidden_dim, latent_dim),
        image_refiner=ImageRefiner(image_shape[0], refiner_channels) if args.stage == "ti" else None,
    )
    models.trace_decoder.to(device).eval()
    if models.image_refiner is not None:
        models.image_refiner.to(device).eval()
    load_checkpoint(args.ckpt, models, device, require_refiner=args.stage == "ti")
    gan = load_gan_prior(args, device) if args.stage == "ti" else None

    chunks = []
    with torch.no_grad():
        for (traces,) in trace_loader(corpus, args.batch_size):
            predictions = predict(models, traces.to(device))
            if gan is not None:
                predictions, _ = apply_gan_projection(predictions, gan, args)
            chunks.append(predictions.cpu())
    recovered = torch.cat(chunks, dim=0)

    payload: dict[str, Any] = {"images": recovered, "ids": corpus.ids, "stage": args.stage, "protocol": corpus.protocol}
    if corpus.images is not None:
        payload["metrics"] = recovery_metrics(recovered, corpus.images)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_dir / "recovered.pt")
    if "metrics" in payload:
        print(" ".join(f"{key}={value:.4f}" for key, value in payload["metrics"].items()))


def parse_ints(text: str | None) -> list[int] | None:
    if text in {None, "", "auto"}:
        return None
    return [int(item) for item in str(text).split(",") if item.strip()]


def target_labels(labels: torch.Tensor, mode: str, source_label: int | None) -> torch.Tensor:
    return labels.eq(int(source_label)).long() if mode == "papersem" else labels.long()


def source_labels_for_task(task, args: argparse.Namespace) -> list[int | None]:
    if args.mode != "papersem":
        return [None]
    requested = parse_ints(args.source_labels)
    if requested:
        return requested
    if task.seed_labels is not None:
        return sorted(int(label) for label in task.seed_labels.unique().tolist())
    raise ValueError(f"{task.name}: papersem mode requires seed_labels in corpus or --source-labels")


def hyper_saved_args(checkpoint: dict) -> dict:
    saved = checkpoint.get("args", {})
    return saved if isinstance(saved, dict) else {}


def hyper_arg(args: argparse.Namespace, saved: dict, name: str, default):
    value = getattr(args, name)
    return value if value is not None else saved.get(name, default)


def build_hypertheft_model(args: argparse.Namespace, corpus, checkpoint: dict):
    from hypertheft_core.hypernetwork.model import HyperTheft

    saved = hyper_saved_args(checkpoint)
    config = checkpoint.get("model_config", {}) if isinstance(checkpoint.get("model_config"), dict) else {}
    target = args.target or saved.get("target") or config.get("target")
    if target is None:
        raise ValueError("--target is required when checkpoint metadata does not contain it")
    model = HyperTheft(
        target=target,
        input_shape=corpus.input_shape,
        trace_shape=corpus.trace_shape,
        num_classes=corpus.num_classes,
        z_dim=int(hyper_arg(args, saved, "z_dim", 64)),
        n_seed=int(hyper_arg(args, saved, "n_seed", 1)),
        hidden=int(hyper_arg(args, saved, "hidden", 512)),
        mixer=str(args.mixer or saved.get("mixer") or config.get("mixer") or "auto"),
        noise_weight=float(args.noise_weight),
    ).to(args.device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    args.mode = str(args.mode or saved.get("mode", "standard"))
    args.n_seed = model.n_seed
    return model


def adapt_trace(trace: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
    need = math.prod(shape)
    flat = trace.flatten().float()
    if flat.numel() < need:
        flat = torch.cat([flat, torch.zeros(need - flat.numel(), dtype=flat.dtype)])
    return flat[:need].view(*shape)


def binary_flip_accuracy(predictions: torch.Tensor, targets: torch.Tensor, num_classes: int) -> float | None:
    if num_classes != 2:
        return None
    flipped = 1 - predictions.long()
    return float(flipped.eq(targets.long()).float().mean().item())


def hypertheft_eval(args: argparse.Namespace) -> None:
    from hypertheft_core.hypernetwork.corpus import load_corpus, make_loader

    args.device = resolve_device(args.device)
    if args.seed is not None:
        torch.manual_seed(int(args.seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(args.seed))

    corpus = load_corpus(args.corpus)
    checkpoint = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = build_hypertheft_model(args, corpus, checkpoint)
    mismatch = load_mismatch_corpus(args.mismatch_corpus, args.mismatch_split)
    trace_shape = corpus.trace_shape
    trace_ids = mismatch.ids or [f"trace_{index:04d}" for index in range(mismatch.traces.size(0))]
    rows = []

    with torch.no_grad():
        for trace_index, trace in enumerate(mismatch.traces):
            seed_trace = adapt_trace(trace, trace_shape).unsqueeze(0).to(args.device)
            seed_batches = [seed_trace for _ in range(model.n_seed)]
            for task in corpus.tasks:
                split = task.get_split(args.split)
                for source_label in source_labels_for_task(task, args):
                    correct = total = loss_sum = 0.0
                    predictions_all, targets_all = [], []
                    for inputs, labels in make_loader(split, args.batch_size, shuffle=False):
                        inputs = inputs.to(args.device, non_blocking=True).float()
                        labels = labels.to(args.device, non_blocking=True).long()
                        targets = target_labels(labels, args.mode, source_label)
                        logits = model.classifier_logits(seed_batches, inputs).squeeze(0)
                        loss = F.cross_entropy(logits, targets)
                        predictions = logits.argmax(dim=-1)
                        loss_sum += float(loss.item()) * int(labels.numel())
                        correct += int(predictions.eq(targets).sum().item())
                        total += int(labels.numel())
                        predictions_all.append(predictions.cpu())
                        targets_all.append(targets.cpu())
                    predictions = torch.cat(predictions_all)
                    targets = torch.cat(targets_all)
                    accuracy = float(correct / max(total, 1.0))
                    flip_acc = binary_flip_accuracy(predictions, targets, corpus.num_classes)
                    row = {
                        "trace_id": trace_ids[trace_index],
                        "task": task.name,
                        "split": args.split,
                        "source_label": "" if source_label is None else int(source_label),
                        "loss": float(loss_sum / max(total, 1.0)),
                        "accuracy": accuracy,
                        "n_eval_samples": int(total),
                    }
                    if flip_acc is not None:
                        row["acc_flip"] = flip_acc
                        row["acc_best_polarity"] = max(accuracy, flip_acc)
                    rows.append(row)

    output_dir = Path(args.output_dir)
    write_csv(output_dir / "metrics.csv", rows)
    write_json(output_dir / "metrics.json", {"rows": rows, "mismatch_protocol": mismatch.protocol})
    if rows:
        mean_acc = sum(float(row["accuracy"]) for row in rows) / len(rows)
        print(f"rows={len(rows)} mean_accuracy={mean_acc:.4f}")


def add_generate_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--raw-bit-count", type=int, required=True)
    parser.add_argument("--sample-stride", type=int, default=1)
    parser.add_argument("--dataset", choices=tuple(DATASET_RATIO_RANGES), default="generic")
    parser.add_argument("--ratio-low", type=float, default=-1.0)
    parser.add_argument("--ratio-high", type=float, default=-1.0)
    parser.add_argument("--trace-ids", default="")
    parser.add_argument("--trace-shape", default="", help="optional stored trace shape, e.g. C,H,W")
    parser.add_argument("--image-shape", default="", help="optional image metadata C,H,W")
    parser.add_argument("--num-classes", type=int, default=0)
    parser.add_argument("--task", choices=("auto", "multiclass", "multilabel"), default="auto")


def add_cipher_recovery_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--corpus", required=True, help="prepared mismatch .pt corpus")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--stage", choices=("tonly", "ti"), default="tonly")
    parser.add_argument("--split", default="test")
    parser.add_argument("--image-shape", default="", help="C,H,W if not stored in corpus")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=0)
    parser.add_argument("--latent-dim", type=int, default=0)
    parser.add_argument("--refiner-channels", type=int, default=0)
    parser.add_argument("--gan-prior", default="", help="required when --stage ti")
    parser.add_argument("--gan-module", default="")
    parser.add_argument("--gan-class", default="")
    parser.add_argument("--gan-latent-dim", type=int, default=128)
    parser.add_argument("--gan-project-steps", type=int, default=0, help="must be >0 when --stage ti")
    parser.add_argument("--gan-project-lr", type=float, default=5e-2)
    parser.add_argument("--gan-blend", type=float, default=1.0)


def add_hypertheft_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--corpus", type=Path, required=True, help="prepared HyperTheft corpus with task images/labels")
    parser.add_argument("--mismatch-corpus", type=Path, required=True)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--mismatch-split", default="test")
    parser.add_argument("--target", choices=("lenet64", "resnet32"), default=None)
    parser.add_argument("--z-dim", type=int, default=None)
    parser.add_argument("--n-seed", type=int, default=None)
    parser.add_argument("--hidden", type=int, default=None)
    parser.add_argument("--mixer", choices=("auto", "mlp", "conv"), default=None)
    parser.add_argument("--mode", choices=("standard", "papersem"), default=None)
    parser.add_argument("--source-labels", default="auto")
    parser.add_argument("--noise-weight", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=1)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Shared mismatch-trace baselines for CipherSteal-style and HyperTheft-style attacks.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    gen = subparsers.add_parser("generate")
    add_generate_args(gen)
    gen.set_defaults(func=generate)

    summ = subparsers.add_parser("summarize")
    summ.add_argument("--corpus", type=Path, required=True)
    summ.add_argument("--output-dir", type=Path, required=True)
    summ.add_argument("--split", default="test")
    summ.set_defaults(func=summarize)

    recover = subparsers.add_parser("ciphersteal-recover")
    add_cipher_recovery_args(recover)
    recover.set_defaults(func=ciphersteal_recover)

    hyper = subparsers.add_parser("hypertheft-eval")
    add_hypertheft_args(hyper)
    hyper.set_defaults(func=hypertheft_eval)
    return parser


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.command == "ciphersteal-recover" and args.stage == "ti":
        if not args.gan_prior:
            parser.error("ciphersteal-recover --stage ti requires --gan-prior")
        if int(args.gan_project_steps) <= 0:
            parser.error("ciphersteal-recover --stage ti requires --gan-project-steps > 0")
        if bool(args.gan_module) != bool(args.gan_class):
            parser.error("--gan-module and --gan-class must be provided together")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    validate_args(parser, args)
    if hasattr(args, "seed") and args.seed is not None:
        torch.manual_seed(int(args.seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(args.seed))
    args.func(args)


if __name__ == "__main__":
    main()
