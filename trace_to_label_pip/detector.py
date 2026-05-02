from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .corpus import Corpus, load_corpus, make_loader
except ImportError:
    from corpus import Corpus, load_corpus, make_loader


class Detector(nn.Module):
    def __init__(self, trace_dim: int, num_classes: int, hidden_dim: int, dropout: float):
        super().__init__()
        mid_dim = max(hidden_dim // 2, num_classes)
        self.net = nn.Sequential(
            nn.Linear(trace_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, mid_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(mid_dim, num_classes),
        )

    def forward(self, traces: torch.Tensor) -> torch.Tensor:
        return self.net(traces.flatten(1).float())


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def loss_fn(logits: torch.Tensor, targets: torch.Tensor, task: str) -> torch.Tensor:
    if task == "multilabel":
        return F.binary_cross_entropy_with_logits(logits, targets.float())
    return F.cross_entropy(logits, targets.long().view(-1))


def metric_row(logits: torch.Tensor, targets: torch.Tensor, labels: torch.Tensor, args: argparse.Namespace) -> dict[str, float]:
    if args.task == "multilabel":
        pred = (torch.sigmoid(logits) >= args.threshold).float()
        target = targets.float()
        row = {
            "target_acc": (pred == target).float().mean().item(),
            "sample_target_acc": (pred == target).all(dim=1).float().mean().item(),
        }
        if labels.numel():
            true = labels.float()
            row["label_acc"] = (pred == true).float().mean().item()
            row["sample_label_acc"] = (pred == true).all(dim=1).float().mean().item()
        return row

    pred = logits.argmax(dim=1)
    row = {"target_acc": (pred == targets.long().view(-1)).float().mean().item()}
    if labels.numel():
        row["label_acc"] = (pred == labels.long().view(-1)).float().mean().item()
    return row


def evaluate(model: nn.Module, corpus: Corpus, args: argparse.Namespace, split_names: tuple[str, ...]) -> list[dict]:
    model.eval()
    rows = []
    with torch.no_grad():
        for name in split_names:
            if name not in corpus.splits:
                continue
            losses, logits_all, targets_all, labels_all = [], [], [], []
            for traces, targets, labels in make_loader(corpus.splits[name], args.eval_batch_size, False):
                traces, targets, labels = traces.to(args.device), targets.to(args.device), labels.to(args.device)
                logits = model(traces)
                losses.append(loss_fn(logits, targets, args.task).item() * traces.shape[0])
                logits_all.append(logits.cpu())
                targets_all.append(targets.cpu())
                labels_all.append(labels.cpu())
            row = metric_row(torch.cat(logits_all), torch.cat(targets_all), torch.cat(labels_all), args)
            row.update({"split": name, "loss": sum(losses) / corpus.splits[name].traces.shape[0]})
            rows.append(row)
    return rows


def save_checkpoint(path: Path, model: nn.Module, corpus: Corpus, args: argparse.Namespace, epoch: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "epoch": epoch,
            "trace_dim": corpus.trace_dim,
            "num_classes": corpus.num_classes,
            "task": corpus.task,
            "hidden_dim": args.hidden_dim,
        },
        path,
    )


def save_metrics(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(rows, path)
    for row in rows:
        print(" ".join(f"{key}={value:.4f}" if isinstance(value, float) else f"{key}={value}" for key, value in row.items()))


def prepare(args: argparse.Namespace) -> Corpus:
    args.device = resolve_device(args.device)
    corpus = load_corpus(args.corpus, args)
    args.task = corpus.task
    return corpus


def build_model(corpus: Corpus, args: argparse.Namespace) -> Detector:
    return Detector(corpus.trace_dim, corpus.num_classes, args.hidden_dim, args.dropout).to(args.device)


def train(args: argparse.Namespace) -> None:
    corpus = prepare(args)
    if "train" not in corpus.splits:
        raise KeyError("Training requires a train split")
    model = build_model(corpus, args)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    output_dir = Path(args.output_dir)
    best_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        for traces, targets, _ in make_loader(corpus.splits["train"], args.batch_size, True):
            traces, targets = traces.to(args.device), targets.to(args.device)
            optimizer.zero_grad(set_to_none=True)
            loss_fn(model(traces), targets, args.task).backward()
            optimizer.step()

        rows = evaluate(model, corpus, args, ("val", "test"))
        score_row = next((row for row in rows if row["split"] == "val"), rows[-1])
        if score_row["loss"] <= best_loss:
            best_loss = score_row["loss"]
            save_checkpoint(output_dir / "best.pt", model, corpus, args, epoch)
        save_checkpoint(output_dir / "last.pt", model, corpus, args, epoch)
        save_metrics(output_dir / "metrics.pt", rows)


def eval_only(args: argparse.Namespace) -> None:
    corpus = prepare(args)
    model = build_model(corpus, args)
    checkpoint = torch.load(args.ckpt, map_location=args.device, weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    save_metrics(Path(args.output_dir) / "metrics.pt", evaluate(model, corpus, args, ("test",)))
