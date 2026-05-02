from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from corpus import DATASETS
    from detector import eval_only, train
else:
    from .corpus import DATASETS
    from .detector import eval_only, train


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--corpus", required=True, help="prepared torch .pt corpus with trace targets")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset", choices=tuple(DATASETS), default="generic")
    parser.add_argument("--num-classes", type=int, default=0)
    parser.add_argument("--task", choices=("auto", "multiclass", "multilabel"), default="auto")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Trace-to-label attack for prepared tensor corpora.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train")
    add_common_args(train_parser)
    train_parser.add_argument("--epochs", type=int, default=80)
    train_parser.add_argument("--lr", type=float, default=2e-3)
    train_parser.add_argument("--weight-decay", type=float, default=0.0)
    train_parser.set_defaults(func=train)

    eval_parser = subparsers.add_parser("eval")
    add_common_args(eval_parser)
    eval_parser.add_argument("--ckpt", required=True)
    eval_parser.set_defaults(func=eval_only)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
