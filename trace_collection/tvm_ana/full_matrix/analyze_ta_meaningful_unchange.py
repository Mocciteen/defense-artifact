#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


KERNEL_SO_SUFFIX = "_tvm_aot_kernels.so"
MEM_ROUTINE_TOKENS = ("memmove", "memcpy", "memset")
HELPER_TOKENS = (
    "PatchedRelu",
    "tvmgen_default",
    "fused_",
    "_compute",
    "compute_",
    "_compute_",
)


@dataclass
class FileStats:
    compare_count: int = 0
    unchanged_count: int = 0
    kept_records: int = 0
    top_owners: Counter[str] = field(default_factory=Counter)
    top_categories: Counter[str] = field(default_factory=Counter)

    @property
    def unchanged_ratio(self) -> float:
        if self.compare_count == 0:
            return 0.0
        return self.unchanged_count / self.compare_count


def is_kernel_module(module: str | None) -> bool:
    return bool(module) and module.endswith(KERNEL_SO_SUFFIX)


def is_mem_symbol(symbol: str | None) -> bool:
    if not symbol:
        return False
    sym = symbol.lower()
    return any(tok in sym for tok in MEM_ROUTINE_TOKENS)


def is_kernel_symbol(symbol: str | None) -> bool:
    if not symbol:
        return False
    return any(tok in symbol for tok in HELPER_TOKENS)


def categorize_owner(owner_module: str | None, owner_symbol: str | None) -> str:
    if is_kernel_module(owner_module):
        if owner_symbol and "PatchedRelu" in owner_symbol:
            return "helper"
        return "operator"
    if is_mem_symbol(owner_symbol):
        return "movement"
    return "other"


def parse_int_field(line: str) -> int:
    value = line.split(":", 1)[1].strip().rstrip(",")
    return int(value)


def parse_str_field(line: str) -> str:
    _, rhs = line.split(":", 1)
    rhs = rhs.strip().rstrip(",")
    if len(rhs) < 2 or rhs[0] != '"' or rhs[-1] != '"':
        return rhs
    return rhs[1:-1]


def analyze_taint_json(path: Path) -> FileStats:
    stats = FileStats()

    in_addresses = False
    section: str | None = None
    compare_count = 0
    unchanged_count = 0
    owner_module: str | None = None
    owner_symbol: str | None = None
    has_kernel_candidate = False
    cand_module: str | None = None
    cand_symbol: str | None = None

    def reset_record() -> None:
        nonlocal compare_count, unchanged_count, owner_module, owner_symbol
        nonlocal has_kernel_candidate, cand_module, cand_symbol
        compare_count = 0
        unchanged_count = 0
        owner_module = None
        owner_symbol = None
        has_kernel_candidate = False
        cand_module = None
        cand_symbol = None

    def finish_candidate() -> None:
        nonlocal has_kernel_candidate, cand_module, cand_symbol
        if is_kernel_module(cand_module) or is_kernel_symbol(cand_symbol):
            has_kernel_candidate = True
        cand_module = None
        cand_symbol = None

    def keep_record() -> bool:
        if is_kernel_module(owner_module):
            return True
        if is_mem_symbol(owner_symbol) and has_kernel_candidate:
            return True
        return False

    def finish_record() -> None:
        if not keep_record():
            return
        stats.compare_count += compare_count
        stats.unchanged_count += unchanged_count
        stats.kept_records += 1
        owner_key = owner_symbol or "<unknown>"
        stats.top_owners[owner_key] += compare_count
        stats.top_categories[categorize_owner(owner_module, owner_symbol)] += compare_count

    reset_record()

    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for raw_line in fh:
            line = raw_line.strip()

            if not in_addresses:
                if line == '"addresses": [' or line == '"addresses": ['.strip():
                    in_addresses = True
                continue

            if section == "owner":
                if line.startswith('"module":'):
                    owner_module = parse_str_field(line)
                    continue
                if line.startswith('"symbol":'):
                    owner_symbol = parse_str_field(line)
                    continue
                if line == "}" or line == "},":
                    section = None
                    continue
                continue

            if section == "candidate":
                if line.startswith('"module":'):
                    cand_module = parse_str_field(line)
                    continue
                if line.startswith('"symbol":'):
                    cand_symbol = parse_str_field(line)
                    continue
                if line == "}" or line == "},":
                    finish_candidate()
                    section = "owner_candidates"
                    continue
                continue

            if section == "owner_candidates":
                if line == "{":
                    section = "candidate"
                    continue
                if line == "]" or line == "],":
                    section = None
                    continue
                continue

            if line == "{":
                reset_record()
                continue

            if line == "]" or line == "],":
                break

            if line.startswith('"compare_count":'):
                compare_count = parse_int_field(line)
                continue

            if line.startswith('"unchanged_count":'):
                unchanged_count = parse_int_field(line)
                continue

            if line.startswith('"owner": {'):
                section = "owner"
                continue

            if line.startswith('"owner_candidates": ['):
                section = "owner_candidates"
                continue

            if (line == "}" or line == "},") and section is None:
                finish_record()
                continue

    return stats


def mean(values: Iterable[float]) -> float:
    vals = list(values)
    if not vals:
        return 0.0
    return sum(vals) / len(vals)


def load_launch_summary(session_dir: Path) -> dict:
    with (session_dir / "launch_summary.json").open("r", encoding="utf-8") as fh:
        return json.load(fh)


def case_meta(session_dir: Path, spec_id: str) -> tuple[str, str]:
    worker_path = session_dir / "worker_status" / f"{spec_id}.json"
    with worker_path.open("r", encoding="utf-8") as fh:
        worker = json.load(fh)
    return worker["base_case"], worker["runtime"]


def collect_ta_specs(session_dir: Path) -> list[str]:
    launch = load_launch_summary(session_dir)
    return [spec_id for spec_id in launch["selected_spec_ids"] if spec_id.startswith("TA")]


def sample_indices(session_dir: Path, spec_id: str) -> list[str]:
    spec_dir = session_dir / "runs" / spec_id
    return sorted(p.name for p in spec_dir.iterdir() if p.is_dir() and p.name.startswith("sample"))


def build_report(session_dir: Path, output_path: Path, selected_specs: set[str] | None = None) -> dict:
    specs = collect_ta_specs(session_dir)
    if selected_specs is not None:
        specs = [spec_id for spec_id in specs if spec_id in selected_specs]
    rows: list[dict] = []

    for spec_id in specs:
        base_case, runtime = case_meta(session_dir, spec_id)
        sample_ids = sample_indices(session_dir, spec_id)

        per_mode_ratios: dict[str, list[float]] = defaultdict(list)
        per_mode_counts: dict[str, list[int]] = defaultdict(list)
        per_mode_owners: dict[str, Counter[str]] = defaultdict(Counter)
        per_mode_categories: dict[str, Counter[str]] = defaultdict(Counter)

        for sample_id in sample_ids:
            for mode in ("off", "on"):
                taint_path = session_dir / "runs" / spec_id / sample_id / mode / "taint_block16_bits.json"
                stats = analyze_taint_json(taint_path)
                per_mode_ratios[mode].append(stats.unchanged_ratio)
                per_mode_counts[mode].append(stats.compare_count)
                per_mode_owners[mode].update(stats.top_owners)
                per_mode_categories[mode].update(stats.top_categories)

        off_ratio = mean(per_mode_ratios["off"])
        on_ratio = mean(per_mode_ratios["on"])
        drop_ratio = 0.0 if off_ratio == 0 else (off_ratio - on_ratio) / off_ratio

        rows.append(
            {
                "spec_id": spec_id,
                "base_case": base_case,
                "runtime": runtime,
                "off_ratio": off_ratio,
                "on_ratio": on_ratio,
                "drop_ratio": drop_ratio,
                "off_compares": sum(per_mode_counts["off"]),
                "on_compares": sum(per_mode_counts["on"]),
                "off_top_owners": per_mode_owners["off"].most_common(5),
                "on_top_owners": per_mode_owners["on"].most_common(5),
                "off_categories": dict(per_mode_categories["off"]),
                "on_categories": dict(per_mode_categories["on"]),
            }
        )

    lines: list[str] = []
    lines.append("# TVM TA Meaningful Unchange Analysis")
    lines.append("")
    lines.append(
        "This report keeps only meaningful points: exact owners inside the AOT kernel shared object are retained as operator/helper points; exact `memmove`/`memcpy`/`memset` owners are retained only when the same block has a kernel-side companion owner in `owner_candidates`. Runtime bookkeeping, FFI registration, loader activity, allocator paths, file I/O, summary writing, and other non-kernel exact owners are excluded."
    )
    lines.append("")
    lines.append("| TA | Base Case | Corrected off | Corrected on | Drop |")
    lines.append("|---|---|---:|---:|---:|")
    for row in rows:
        lines.append(
            f"| {row['spec_id']} | {row['base_case']} | {row['off_ratio']:.4f} | {row['on_ratio']:.4f} | {row['drop_ratio'] * 100:.1f}% |"
        )

    lines.append("")
    for row in rows:
        lines.append(f"## {row['spec_id']}")
        lines.append("")
        lines.append(
            f"`{row['spec_id']}` keeps a meaningful unchange ratio of `{row['off_ratio']:.4f} -> {row['on_ratio']:.4f}`, a relative drop of `{row['drop_ratio'] * 100:.1f}%`."
        )
        off_cats = ", ".join(
            f"{k}={v}" for k, v in sorted(row["off_categories"].items(), key=lambda item: (-item[1], item[0]))
        )
        on_cats = ", ".join(
            f"{k}={v}" for k, v in sorted(row["on_categories"].items(), key=lambda item: (-item[1], item[0]))
        )
        if off_cats:
            lines.append(f"Off kept compares by category: {off_cats}.")
        if on_cats:
            lines.append(f"On kept compares by category: {on_cats}.")
        if row["off_top_owners"]:
            off_top = ", ".join(f"`{name}` ({count})" for name, count in row["off_top_owners"])
            lines.append(f"Dominant off owners: {off_top}.")
        if row["on_top_owners"]:
            on_top = ", ".join(f"`{name}` ({count})" for name, count in row["on_top_owners"])
            lines.append(f"Dominant on owners: {on_top}.")
        lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    return {"rows": rows, "output_path": str(output_path)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session-dir", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--spec-id", action="append", dest="spec_ids")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected_specs = set(args.spec_ids) if args.spec_ids else None
    result = build_report(args.session_dir.resolve(), args.output_md.resolve(), selected_specs)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
