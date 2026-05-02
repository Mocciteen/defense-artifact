from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

try:
    from .gan_prior import GANModuleWrapper, apply_gan_projection, load_gan_prior
except ImportError:
    from gan_prior import GANModuleWrapper, apply_gan_projection, load_gan_prior


@dataclass
class SplitTensors:
    traces: torch.Tensor
    images: torch.Tensor


@dataclass
class PreparedCorpus:
    train: SplitTensors | None
    val: SplitTensors | None
    test: SplitTensors
    image_shape: tuple[int, int, int]
    trace_dim: int


@dataclass
class AttackModels:
    trace_decoder: nn.Module
    image_refiner: nn.Module | None = None


class TraceDecoder(nn.Module):
    def __init__(self, trace_dim: int, image_shape: tuple[int, int, int], hidden_dim: int, latent_dim: int):
        super().__init__()
        output_dim = math.prod(image_shape)
        self.image_shape = tuple(int(x) for x in image_shape)
        self.net = nn.Sequential(
            nn.Linear(trace_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU(inplace=True),
            nn.Linear(latent_dim, output_dim),
            nn.Tanh(),
        )

    def forward(self, traces: torch.Tensor) -> torch.Tensor:
        images = self.net(traces.view(traces.shape[0], -1).float())
        return images.view(traces.shape[0], *self.image_shape)


class ImageRefiner(nn.Module):
    def __init__(self, channels: int, hidden_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, channels, kernel_size=3, padding=1),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return torch.tanh(images + self.net(images))


def parse_image_shape(raw: str) -> tuple[int, int, int]:
    parts = [int(item.strip()) for item in str(raw).split(",") if item.strip()]
    if len(parts) != 3:
        raise ValueError(f"--image-shape must be C,H,W; got {raw!r}")
    return tuple(parts)  # type: ignore[return-value]


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _split_from_mapping(payload: dict, split: str) -> SplitTensors | None:
    item = payload.get(split)
    if item is None:
        return None
    traces = item.get("traces", item.get("trace"))
    images = item.get("images", item.get("image"))
    if traces is None or images is None:
        raise KeyError(f"Corpus split {split!r} must contain traces/trace and images/image tensors")
    return SplitTensors(torch.as_tensor(traces).float(), torch.as_tensor(images).float())


def load_prepared_corpus(path: str | Path, image_shape_arg: str = "") -> PreparedCorpus:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError("Prepared corpus must be a torch-saved dict")

    if "splits" in payload:
        traces = torch.as_tensor(payload["traces"]).float()
        images = torch.as_tensor(payload["images"]).float()
        split_map = payload["splits"]

        def from_indices(name: str) -> SplitTensors | None:
            if name not in split_map:
                return None
            idx = torch.as_tensor(split_map[name], dtype=torch.long)
            return SplitTensors(traces[idx], images[idx])

        train = from_indices("train")
        val = from_indices("val")
        test = from_indices("test")
    else:
        train = _split_from_mapping(payload, "train")
        val = _split_from_mapping(payload, "val")
        test = _split_from_mapping(payload, "test")

    if test is None:
        raise KeyError("Prepared corpus must provide a test split")
    reference = train or val or test
    image_shape = parse_image_shape(image_shape_arg) if image_shape_arg else tuple(reference.images.shape[1:])
    if len(image_shape) != 3:
        raise ValueError(f"Images must be N,C,H,W or --image-shape must be set; got {image_shape}")
    return PreparedCorpus(train, val, test, image_shape, int(reference.traces[0].numel()))


def make_loader(split: SplitTensors, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(TensorDataset(split.traces, split.images), batch_size=batch_size, shuffle=shuffle)


def build_models(corpus: PreparedCorpus, args: argparse.Namespace, *, use_refiner: bool) -> AttackModels:
    trace_decoder = TraceDecoder(corpus.trace_dim, corpus.image_shape, args.hidden_dim, args.latent_dim)
    image_refiner = ImageRefiner(corpus.image_shape[0], args.refiner_channels) if use_refiner else None
    return AttackModels(trace_decoder, image_refiner)


def move_models(models: AttackModels, device: torch.device) -> AttackModels:
    models.trace_decoder.to(device)
    if models.image_refiner is not None:
        models.image_refiner.to(device)
    return models


def require_gan_projection(args: argparse.Namespace) -> None:
    if not getattr(args, "gan_prior", ""):
        raise ValueError("TI recovery requires --gan-prior")
    if int(getattr(args, "gan_project_steps", 0)) <= 0:
        raise ValueError("TI recovery requires --gan-project-steps > 0")
    if bool(getattr(args, "gan_module", "")) != bool(getattr(args, "gan_class", "")):
        raise ValueError("--gan-module and --gan-class must be provided together")


def predict(models: AttackModels, traces: torch.Tensor) -> torch.Tensor:
    t_images = models.trace_decoder(traces)
    if models.image_refiner is None:
        return t_images
    return models.image_refiner(t_images)


def save_checkpoint(path: Path, models: AttackModels, args: argparse.Namespace, epoch: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    saved_args = {
        key: value
        for key, value in vars(args).items()
        if isinstance(value, (str, int, float, bool, type(None)))
    }
    torch.save(
        {
            "trace_decoder": models.trace_decoder.state_dict(),
            "image_refiner": models.image_refiner.state_dict() if models.image_refiner is not None else None,
            "epoch": int(epoch),
            "args": saved_args,
        },
        path,
    )


def load_checkpoint(path: str | Path, models: AttackModels, device: torch.device, *, require_refiner: bool) -> None:
    checkpoint = torch.load(Path(path), map_location=device, weights_only=False)
    models.trace_decoder.load_state_dict(checkpoint["trace_decoder"], strict=True)
    if require_refiner:
        if models.image_refiner is None or checkpoint.get("image_refiner") is None:
            raise KeyError("TI checkpoint must contain image_refiner weights")
        models.image_refiner.load_state_dict(checkpoint["image_refiner"], strict=True)


def metric_row(stage: str, split: str, predictions: torch.Tensor, targets: torch.Tensor) -> dict[str, float | str]:
    mse = F.mse_loss(predictions, targets).item()
    l1 = F.l1_loss(predictions, targets).item()
    psnr = -10.0 * math.log10(max(mse, 1e-12))
    return {"stage": stage, "split": split, "mse": mse, "l1": l1, "psnr": psnr}


def write_metrics(output_dir: Path, rows: list[dict[str, float | str]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "metrics.csv").open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=["stage", "split", "mse", "l1", "psnr"])
        writer.writeheader()
        writer.writerows(rows)


def evaluate_models(
    models: AttackModels,
    corpus: PreparedCorpus,
    args: argparse.Namespace,
    *,
    stage: str,
    gan: GANModuleWrapper | None,
    device: torch.device,
    splits: tuple[str, ...],
) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    models.trace_decoder.eval()
    if models.image_refiner is not None:
        models.image_refiner.eval()
    for split_name in splits:
        split = getattr(corpus, split_name)
        if split is None:
            continue
        predictions_all: list[torch.Tensor] = []
        targets_all: list[torch.Tensor] = []
        with torch.no_grad():
            for traces, images in make_loader(split, args.eval_batch_size, shuffle=False):
                traces = traces.to(device)
                images = images.to(device)
                predictions = predict(models, traces)
                if gan is not None and int(args.gan_project_steps) > 0:
                    predictions, _ = apply_gan_projection(predictions, gan, args)
                predictions_all.append(predictions.cpu())
                targets_all.append(images.cpu())
        rows.append(metric_row(stage, split_name, torch.cat(predictions_all), torch.cat(targets_all)))
    return rows


def train_stage(args: argparse.Namespace, *, stage: str, use_refiner: bool) -> None:
    corpus = load_prepared_corpus(args.corpus, args.image_shape)
    if corpus.train is None:
        raise KeyError("Training requires a train split in the prepared corpus")
    device = resolve_device(args.device)
    models = move_models(build_models(corpus, args, use_refiner=use_refiner), device)

    parameters = list(models.trace_decoder.parameters())
    if models.image_refiner is not None:
        parameters += list(models.image_refiner.parameters())
    optimizer = torch.optim.Adam(parameters, lr=args.lr, weight_decay=args.weight_decay)
    output_dir = Path(args.output_dir) / stage
    best_val = float("inf")

    for epoch in range(1, int(args.epochs) + 1):
        models.trace_decoder.train()
        if models.image_refiner is not None:
            models.image_refiner.train()
        for traces, images in make_loader(corpus.train, args.batch_size, shuffle=True):
            traces = traces.to(device)
            images = images.to(device)
            optimizer.zero_grad(set_to_none=True)
            t_images = models.trace_decoder(traces)
            predictions = models.image_refiner(t_images) if models.image_refiner is not None else t_images
            loss = F.mse_loss(predictions, images)
            if models.image_refiner is not None:
                loss = loss + float(args.t_loss_weight) * F.mse_loss(t_images, images)
            loss.backward()
            optimizer.step()

        rows = evaluate_models(models, corpus, args, stage=stage, gan=None, device=device, splits=("val", "test"))
        val_row = next((row for row in rows if row["split"] == "val"), rows[-1])
        if float(val_row["mse"]) <= best_val:
            best_val = float(val_row["mse"])
            save_checkpoint(output_dir / "best.pt", models, args, epoch)
        save_checkpoint(output_dir / "last.pt", models, args, epoch)
        write_metrics(output_dir, rows)


def test_stage(args: argparse.Namespace, *, stage: str, use_refiner: bool) -> None:
    if use_refiner:
        require_gan_projection(args)
    corpus = load_prepared_corpus(args.corpus, args.image_shape)
    device = resolve_device(args.device)
    models = move_models(build_models(corpus, args, use_refiner=use_refiner), device)
    load_checkpoint(args.ckpt, models, device, require_refiner=use_refiner)
    gan = load_gan_prior(args, device) if use_refiner else None
    rows = evaluate_models(models, corpus, args, stage=stage, gan=gan, device=device, splits=("test",))
    write_metrics(Path(args.output_dir) / stage, rows)


def train_tonly(args: argparse.Namespace) -> None:
    train_stage(args, stage="tonly", use_refiner=False)


def test_tonly(args: argparse.Namespace) -> None:
    test_stage(args, stage="tonly_test", use_refiner=False)


def train_ti(args: argparse.Namespace) -> None:
    train_stage(args, stage="ti", use_refiner=True)


def test_ti(args: argparse.Namespace) -> None:
    test_stage(args, stage="ti_test", use_refiner=True)
