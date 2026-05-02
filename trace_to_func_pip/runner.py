from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from corpus import Corpus, Task, load_corpus, make_loader, sample_seed_batches, source_labels
    from model import HyperTheft
else:
    from .corpus import Corpus, Task, load_corpus, make_loader, sample_seed_batches, source_labels
    from .model import HyperTheft


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def device(name: str) -> torch.device:
    return torch.device("cuda" if name == "auto" and torch.cuda.is_available() else "cpu" if name == "auto" else name)


def ints(text: str | None) -> list[int] | None:
    return None if text in {None, "", "auto"} else [int(item) for item in str(text).split(",") if item.strip()]


def avg(values: list[float]) -> float:
    return sum(values) / max(len(values), 1)


def clean_args(args: argparse.Namespace) -> dict:
    out = {key: str(value) if isinstance(value, (Path, torch.device)) else value for key, value in vars(args).items()}
    return {key: value for key, value in out.items() if key != "func"}


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader()
        writer.writerows(rows)


def labels_for(task: Task, args: argparse.Namespace) -> list[int | None]:
    if args.mode != "papersem":
        return [None]
    labels = ints(args.source_labels) or source_labels(task)
    if not labels:
        raise ValueError(f"{task.name}: papersem mode requires seed_labels or --source-labels")
    return labels


def targets(labels: torch.Tensor, mode: str, source_label: int | None) -> torch.Tensor:
    return labels.eq(int(source_label)).long() if mode == "papersem" else labels.long()


def loss_acc(logits: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, float]:
    expanded = target.unsqueeze(0).expand(logits.size(0), -1).reshape(-1)
    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), expanded)
    return loss, float(logits.argmax(-1).eq(target.unsqueeze(0)).float().mean().item())


def finish_model_args(args: argparse.Namespace, saved: dict) -> None:
    args.target = args.target or saved.get("target")
    if args.target is None:
        raise ValueError("--target is required when checkpoint metadata does not contain it")
    args.z_dim = args.z_dim if args.z_dim is not None else int(saved.get("z_dim", 64))
    args.n_seed = args.n_seed if args.n_seed is not None else int(saved.get("n_seed", 1))
    args.hidden = args.hidden if args.hidden is not None else int(saved.get("hidden", 512))
    args.mixer = args.mixer or saved.get("mixer", "auto")
    args.mode = args.mode or saved.get("mode", "standard")


def build_model(args: argparse.Namespace, corpus: Corpus, ckpt: dict | None = None) -> HyperTheft:
    finish_model_args(args, ckpt.get("args", {}) if ckpt else {})
    model = HyperTheft(
        target=args.target,
        input_shape=corpus.input_shape,
        trace_shape=corpus.trace_shape,
        num_classes=corpus.num_classes,
        z_dim=args.z_dim,
        n_seed=args.n_seed,
        hidden=args.hidden,
        mixer=args.mixer,
        noise_weight=getattr(args, "noise_weight", 0.0),
    ).to(args.device)
    if ckpt:
        model.load_state_dict(ckpt["model"])
    return model


def run_split(model: HyperTheft, corpus: Corpus, args: argparse.Namespace, split: str) -> tuple[dict, list[dict]]:
    rows = []
    model.eval()
    with torch.no_grad():
        for task in corpus.tasks:
            for source_label in labels_for(task, args):
                losses, accs = [], []
                for inputs, labels in make_loader(task.get_split(split), args.batch_size, shuffle=False):
                    inputs = inputs.to(args.device, non_blocking=True).float()
                    labels = labels.to(args.device, non_blocking=True).long()
                    seeds = sample_seed_batches(task, inputs.size(0), args.n_seed, args.device, source_label)
                    loss, acc = loss_acc(model.classifier_logits(seeds, inputs), targets(labels, args.mode, source_label))
                    losses.append(float(loss.item()))
                    accs.append(acc)
                rows.append({"task": task.name, "split": split, "source_label": "" if source_label is None else int(source_label), "loss": avg(losses), "accuracy": avg(accs)})
    return {f"{split}_loss": avg([row["loss"] for row in rows]), f"{split}_accuracy": avg([row["accuracy"] for row in rows])}, rows


def train_epoch(model: HyperTheft, corpus: Corpus, optimizer: torch.optim.Optimizer, args: argparse.Namespace) -> dict:
    model.train()
    losses, accs = [], []
    for task in corpus.tasks:
        source_cycle = labels_for(task, args)
        for step, (inputs, labels) in enumerate(make_loader(task.train, args.batch_size, shuffle=True)):
            inputs = inputs.to(args.device, non_blocking=True).float()
            labels = labels.to(args.device, non_blocking=True).long()
            source_label = source_cycle[step % len(source_cycle)]
            seeds = sample_seed_batches(task, inputs.size(0), args.n_seed, args.device, source_label)
            loss, acc = loss_acc(model.classifier_logits(seeds, inputs), targets(labels, args.mode, source_label))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
            accs.append(acc)
    return {"train_loss": avg(losses), "train_accuracy": avg(accs)}


def save_ckpt(path: Path, model: HyperTheft, optimizer: torch.optim.Optimizer, args: argparse.Namespace, epoch: int, metrics: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "args": clean_args(args), "metrics": metrics, "model_config": model.config()}, path)


def train(args: argparse.Namespace) -> None:
    args.device = device(args.device)
    set_seed(args.seed)
    corpus = load_corpus(args.corpus)
    model = build_model(args, corpus)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    history, best = [], float("-inf")
    for epoch in range(1, args.epochs + 1):
        row = {"epoch": epoch, **train_epoch(model, corpus, optimizer, args)}
        if args.val_split:
            row.update(run_split(model, corpus, args, args.val_split)[0])
        history.append(row)
        print(" ".join(f"{key}={value}" for key, value in row.items()), flush=True)
        save_ckpt(args.output_dir / "last.pt", model, optimizer, args, epoch, row)
        metric = row.get(f"{args.val_split}_accuracy", row["train_accuracy"])
        if metric > best:
            best = metric
            save_ckpt(args.output_dir / "best.pt", model, optimizer, args, epoch, row)
    write_csv(args.output_dir / "history.csv", history)
    write_json(args.output_dir / "run_config.json", {"metadata": corpus.metadata, "model": model.config(), "args": clean_args(args)})


def eval_only(args: argparse.Namespace) -> None:
    args.device = device(args.device)
    corpus = load_corpus(args.corpus)
    model = build_model(args, corpus, torch.load(args.ckpt, map_location="cpu", weights_only=False))
    summary, rows = run_split(model, corpus, args, args.split)
    write_csv(args.output_dir / "metrics.csv", rows)
    write_json(args.output_dir / "metrics.json", {"summary": summary, "rows": rows})
    print(" ".join(f"{key}={value}" for key, value in summary.items()), flush=True)


def vote(args: argparse.Namespace) -> None:
    args.device = device(args.device)
    corpus = load_corpus(args.corpus)
    model = build_model(args, corpus, torch.load(args.ckpt, map_location="cpu", weights_only=False))
    model.eval()
    rows = []
    with torch.no_grad():
        for task in corpus.tasks:
            for source_label in labels_for(task, args):
                for vote_count in ints(args.votes) or [1]:
                    correct = covered = total = 0.0
                    for inputs, labels in make_loader(task.get_split(args.split), args.batch_size, shuffle=False):
                        inputs = inputs.to(args.device, non_blocking=True).float()
                        labels = labels.to(args.device, non_blocking=True).long()
                        counter = None
                        for _ in range(vote_count):
                            seeds = sample_seed_batches(task, inputs.size(0), args.n_seed, args.device, source_label)
                            pred = model.classifier_logits(seeds, inputs).argmax(-1)
                            if counter is None:
                                counter = torch.zeros(pred.size(0), pred.size(1), corpus.num_classes, device=args.device)
                            counter.scatter_add_(2, pred.unsqueeze(-1), torch.ones_like(pred, dtype=counter.dtype).unsqueeze(-1))
                        mask = counter.gt(int(vote_count * args.vote_threshold)).any(-1)
                        target = targets(labels, args.mode, source_label)
                        correct += counter.argmax(-1).eq(target.unsqueeze(0)).logical_and(mask).sum().item()
                        covered += mask.sum().item()
                        total += mask.numel()
                    rows.append({"task": task.name, "split": args.split, "source_label": "" if source_label is None else int(source_label), "votes": vote_count, "accuracy": correct / max(covered, 1.0), "coverage": covered / max(total, 1.0)})
    write_csv(args.output_dir / "voter_metrics.csv", rows)
    write_json(args.output_dir / "voter_metrics.json", rows)
    for row in rows:
        print(" ".join(f"{key}={value}" for key, value in row.items()), flush=True)


def add_model_args(parser: argparse.ArgumentParser, training: bool) -> None:
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--target", choices=("lenet64", "resnet32"), default="lenet64" if training else None)
    parser.add_argument("--z-dim", type=int, default=64 if training else None)
    parser.add_argument("--n-seed", type=int, default=1 if training else None)
    parser.add_argument("--hidden", type=int, default=512 if training else None)
    parser.add_argument("--mixer", choices=("auto", "mlp", "conv"), default="auto" if training else None)
    parser.add_argument("--mode", choices=("standard", "papersem"), default="standard" if training else None)
    parser.add_argument("--source-labels", default="auto")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="auto")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="HyperTheft core runner for prepared tensor corpora.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    train_parser = subparsers.add_parser("train")
    add_model_args(train_parser, True)
    train_parser.add_argument("--output-dir", type=Path, required=True)
    train_parser.add_argument("--epochs", type=int, default=80)
    train_parser.add_argument("--lr", type=float, default=1e-3)
    train_parser.add_argument("--weight-decay", type=float, default=0.0)
    train_parser.add_argument("--noise-weight", type=float, default=0.0)
    train_parser.add_argument("--seed", type=int, default=1)
    train_parser.add_argument("--val-split", default="val")
    train_parser.set_defaults(func=train)
    eval_parser = subparsers.add_parser("eval")
    add_model_args(eval_parser, False)
    eval_parser.add_argument("--ckpt", type=Path, required=True)
    eval_parser.add_argument("--output-dir", type=Path, required=True)
    eval_parser.add_argument("--split", default="test")
    eval_parser.set_defaults(func=eval_only)
    vote_parser = subparsers.add_parser("vote")
    add_model_args(vote_parser, False)
    vote_parser.add_argument("--ckpt", type=Path, required=True)
    vote_parser.add_argument("--output-dir", type=Path, required=True)
    vote_parser.add_argument("--split", default="test")
    vote_parser.add_argument("--votes", default="1,5,10")
    vote_parser.add_argument("--vote-threshold", type=float, default=0.75)
    vote_parser.set_defaults(func=vote)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
