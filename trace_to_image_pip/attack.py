from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from recovery import test_ti, test_tonly, train_ti, train_tonly
else:
    from .recovery import test_ti, test_tonly, train_ti, train_tonly


def positive_int(raw: str) -> int:
    value = int(raw)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be > 0")
    return value


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--corpus", required=True, help="torch-saved prepared corpus with train/val/test tensors")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--image-shape", default="", help="Optional C,H,W override")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=1024)
    parser.add_argument("--latent-dim", type=int, default=256)
    parser.add_argument("--refiner-channels", type=int, default=64)


def add_gan_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--gan-prior", required=True, help="GAN generator checkpoint for TI prior projection")
    parser.add_argument("--gan-module", default="", help="Optional Python file defining the GAN class")
    parser.add_argument("--gan-class", default="", help="GAN class name inside --gan-module")
    parser.add_argument("--gan-latent-dim", type=int, default=128)
    parser.add_argument("--gan-project-steps", type=positive_int, required=True)
    parser.add_argument("--gan-project-lr", type=float, default=5e-2)
    parser.add_argument("--gan-blend", type=float, default=1.0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Core four-stage CipherSteal attack recovery.")
    subparsers = parser.add_subparsers(dest="stage", required=True)

    train_tonly_parser = subparsers.add_parser("train-tonly")
    add_common_args(train_tonly_parser)
    train_tonly_parser.add_argument("--epochs", type=int, default=80)
    train_tonly_parser.add_argument("--lr", type=float, default=2e-3)
    train_tonly_parser.add_argument("--weight-decay", type=float, default=0.0)
    train_tonly_parser.set_defaults(func=train_tonly)

    test_tonly_parser = subparsers.add_parser("test-tonly")
    add_common_args(test_tonly_parser)
    test_tonly_parser.add_argument("--ckpt", required=True)
    test_tonly_parser.set_defaults(func=test_tonly)

    train_ti_parser = subparsers.add_parser("train-ti")
    add_common_args(train_ti_parser)
    train_ti_parser.add_argument("--epochs", type=int, default=80)
    train_ti_parser.add_argument("--lr", type=float, default=2e-3)
    train_ti_parser.add_argument("--weight-decay", type=float, default=0.0)
    train_ti_parser.add_argument("--t-loss-weight", type=float, default=1.0)
    train_ti_parser.set_defaults(func=train_ti)

    test_ti_parser = subparsers.add_parser("test-ti")
    add_common_args(test_ti_parser)
    add_gan_args(test_ti_parser)
    test_ti_parser.add_argument("--ckpt", required=True)
    test_ti_parser.set_defaults(func=test_ti)
    return parser


def run_from_args(args: argparse.Namespace) -> None:
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))
    args.func(args)


def main() -> None:
    run_from_args(build_parser().parse_args())


if __name__ == "__main__":
    main()
