from __future__ import annotations

import argparse
import shlex
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compile the generic Glow bundle benchmark server.")
    parser.add_argument("--source", type=Path, default=Path(__file__).resolve().parents[1] / "runtime" / "bundle_bench_server.cpp")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compiler", default="g++")
    parser.add_argument("--std", default="c++17")
    parser.add_argument("--include-dir", type=Path, action="append", default=[])
    parser.add_argument("--extra-source", type=Path, action="append", default=[])
    parser.add_argument("--object", type=Path, action="append", default=[])
    parser.add_argument("--library", action="append", default=[])
    parser.add_argument("--define", action="append", default=[])
    parser.add_argument("--bundle-header", required=True)
    parser.add_argument("--entry", required=True)
    parser.add_argument("--constant-size", required=True)
    parser.add_argument("--mutable-size", required=True)
    parser.add_argument("--activations-size", required=True)
    parser.add_argument("--mem-align", required=True)
    parser.add_argument("--data-offset", required=True)
    parser.add_argument("--output-offset", required=True)
    parser.add_argument("--input-bytes", required=True)
    parser.add_argument("--output-elements", default="10")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def macro(name: str, value: str) -> str:
    return f"-D{name}={value}"


def main() -> None:
    args = parse_args()
    defines = [
        macro("BUNDLE_HEADER", f'"{args.bundle_header}"'),
        macro("BUNDLE_ENTRY", args.entry),
        macro("BUNDLE_CONSTANT_MEM_SIZE", args.constant_size),
        macro("BUNDLE_MUTABLE_MEM_SIZE", args.mutable_size),
        macro("BUNDLE_ACTIVATIONS_MEM_SIZE", args.activations_size),
        macro("BUNDLE_MEM_ALIGN", args.mem_align),
        macro("BUNDLE_DATA_OFFSET", args.data_offset),
        macro("BUNDLE_OUTPUT_OFFSET", args.output_offset),
        macro("BUNDLE_INPUT_BYTES", args.input_bytes),
        macro("BUNDLE_OUTPUT_ELEMENTS", args.output_elements),
    ] + [f"-D{item}" for item in args.define]
    command = [args.compiler, f"-std={args.std}", "-O3", "-DNDEBUG", *defines, "-I", str(args.source.parent)]
    for include in args.include_dir:
        command += ["-I", str(include)]
    command += [
        str(args.source),
        *[str(source) for source in args.extra_source],
        *[str(obj) for obj in args.object],
        *args.library,
        "-o",
        str(args.output),
    ]
    print(" ".join(shlex.quote(part) for part in command))
    if not args.dry_run:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        subprocess.check_call(command)


if __name__ == "__main__":
    main()
