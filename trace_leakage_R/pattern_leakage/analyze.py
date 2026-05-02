from __future__ import annotations

import argparse
import csv
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from metrics import analyze_trace, pair_metrics
else:
    from .metrics import analyze_trace, pair_metrics


RUN_FIELDS = [
    "task_id",
    "sample",
    "mode",
    "backend",
    "group",
    "run_dir",
    "n_events_total_valid",
    "n_events_after_only_valid",
    "n_events_after_filter",
    "n_blocks_after_filter",
    "max_block_len_after_filter",
    "skipped_before_events",
    "skipped_defense_events",
    "skipped_ptr_addr_events",
    "L_nleq",
    "L_rep",
    "L_per",
    "L_motif",
    "total_kgrams",
    "defense_rule_hits",
    "defense_routine_hits",
]

PAIR_FIELDS = [
    "task_id",
    "sample",
    "backend",
    "group",
    "L_nleq_off",
    "L_nleq_on",
    "L_nleq_delta",
    "R_nleq",
    "L_rep_off",
    "L_rep_on",
    "L_rep_delta",
    "R_rep",
    "L_per_off",
    "L_per_on",
    "L_per_delta",
    "R_per",
    "L_motif_off",
    "L_motif_on",
    "L_motif_delta",
    "R_motif",
]


def read_pairs(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def run_paths(run_dir: str) -> tuple[Path, Path]:
    root = Path(run_dir)
    return root / "paddrtrace.bin", root / "paddrtrace.ip.txt"


def analyze_one_pair(row: dict[str, str], args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    run_rows: list[dict[str, Any]] = []
    metrics_by_mode: dict[str, dict[str, Any]] = {}
    for mode, key in (("off", "run_off"), ("on", "run_on")):
        trace_bin, ipmap_path = run_paths(row[key])
        metrics = analyze_trace(
            trace_bin,
            ipmap_path,
            tau_max=args.tau_max,
            kgram=args.kgram,
            bitmap_bits=args.bitmap_bits,
            topk=args.topk,
            filter_defense=not args.keep_defense_helpers,
            filter_pointers=not args.keep_pointer_values,
            rep_mode=args.rep_mode,
        )
        metrics_by_mode[mode] = metrics
        run_rows.append(
            {
                "task_id": row["task_id"],
                "sample": row["sample"],
                "mode": mode,
                "backend": row.get("backend", ""),
                "group": row.get("group", ""),
                "run_dir": row[key],
                **metrics,
            }
        )
    pair_row = {
        "task_id": row["task_id"],
        "sample": row["sample"],
        "backend": row.get("backend", ""),
        "group": row.get("group", ""),
        **pair_metrics(metrics_by_mode["off"], metrics_by_mode["on"]),
    }
    return run_rows, pair_row


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def analyze_pairs(args: argparse.Namespace) -> None:
    jobs = read_pairs(Path(args.pairs))
    if args.limit_pairs > 0:
        jobs = jobs[: args.limit_pairs]
    run_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []

    if args.jobs <= 1:
        for row in jobs:
            runs, pair = analyze_one_pair(row, args)
            run_rows.extend(runs)
            pair_rows.append(pair)
    else:
        with ProcessPoolExecutor(max_workers=args.jobs) as executor:
            futures = [executor.submit(analyze_one_pair, row, args) for row in jobs]
            for future in as_completed(futures):
                runs, pair = future.result()
                run_rows.extend(runs)
                pair_rows.append(pair)

    out_dir = Path(args.out_dir)
    write_csv(out_dir / "per_run_metrics.csv", run_rows, RUN_FIELDS)
    write_csv(out_dir / "per_pair_rpat.csv", pair_rows, PAIR_FIELDS)
    print(f"analyzed pairs={len(pair_rows)} runs={len(run_rows)} out={out_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze pattern leakage from off/on paddrtrace pairs.")
    parser.add_argument("--pairs", required=True, help="CSV with task_id,sample,run_off,run_on")
    parser.add_argument("--out-dir", default="outputs/leakage_analysis")
    parser.add_argument("--tau-max", type=int, default=64)
    parser.add_argument("--kgram", type=int, default=4)
    parser.add_argument("--bitmap-bits", type=int, default=1 << 20)
    parser.add_argument("--rep-mode", choices=("bitmap", "exact"), default="bitmap")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--topk", type=int, default=12)
    parser.add_argument("--limit-pairs", type=int, default=0)
    parser.add_argument("--keep-defense-helpers", action="store_true")
    parser.add_argument("--keep-pointer-values", action="store_true")
    return parser


def main() -> None:
    analyze_pairs(build_parser().parse_args())


if __name__ == "__main__":
    main()
