#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import json
import struct
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


HDR_V4 = struct.Struct("<8sIIQ")
REC_V4 = struct.Struct("<QIIIIQQQQQII")
MAGIC = b"PADDRTRC"
VER = 4


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Root-cause analysis for remaining negative R_pat entries.")
    ap.add_argument(
        "--analysis-dir",
        default="/path/to/high_leakage_workspace/higher_order_analysis_formal_20260422",
        help="directory containing per_run_metrics.csv and negative_after_excluding_defense_added.json",
    )
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--topk", type=int, default=20)
    return ap.parse_args()


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    import csv

    return list(csv.DictReader(path.open("r", encoding="utf-8")))


def parse_ipmap(ipmap_path: Path) -> dict[int, tuple[str, str, str]]:
    out: dict[int, tuple[str, str, str]] = {}
    for raw in ipmap_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not raw or raw.startswith("#"):
            continue
        parts = raw.split("\t")
        if not parts:
            continue
        try:
            ip = int(parts[0], 16)
        except Exception:
            continue
        routine = parts[3].strip() if len(parts) > 3 else ""
        if "+" in routine:
            routine = routine.split("+", 1)[0]
        disasm = parts[4].strip() if len(parts) > 4 else ""
        mnemonic = disasm.split(" ", 1)[0] if disasm else ""
        out[ip] = (routine or "<unknown>", mnemonic or "<unknown>", disasm or "<unknown>")
    return out


def count_events_by_key(run_dir: Path) -> tuple[int, Counter[tuple[str, str, str]]]:
    ipmap = parse_ipmap(run_dir / "paddrtrace.ip.txt")
    counts: Counter[tuple[str, str, str]] = Counter()
    n_events = 0
    with (run_dir / "paddrtrace.bin").open("rb") as handle:
        header = handle.read(HDR_V4.size)
        if len(header) != HDR_V4.size:
            raise RuntimeError(f"trace too small: {run_dir}")
        magic, ver, _reserved, _page = HDR_V4.unpack(header)
        if magic != MAGIC or ver != VER:
            raise RuntimeError(f"bad trace header: {run_dir}")
        while True:
            rec = handle.read(REC_V4.size)
            if not rec:
                break
            if len(rec) != REC_V4.size:
                raise RuntimeError(f"truncated record header: {run_dir}")
            (
                _seq,
                _tid,
                size,
                size_read,
                _block16_read,
                _vaddr,
                _paddr,
                _paddr16,
                ip,
                _instr_id,
                _flags,
                stack_depth,
            ) = REC_V4.unpack(rec)

            handle.seek(int(size), 1)
            block16 = handle.read(16)
            if len(block16) != 16:
                raise RuntimeError(f"truncated block16: {run_dir}")
            if stack_depth:
                handle.seek(int(stack_depth) * 8, 1)
            if int(size_read) != int(size) or int(size) <= 0:
                continue
            n_events += 1
            key = ipmap.get(int(ip), ("<unknown>", "<unknown>", "<unknown>"))
            counts[key] += 1
    return n_events, counts


def analyze_pair(job: dict[str, Any]) -> dict[str, Any]:
    task_id = str(job["task_id"])
    sample = str(job["sample"])
    off_dir = Path(job["off_dir"])
    on_dir = Path(job["on_dir"])
    negative_components = list(job["negative_components"])
    topk = int(job["topk"])

    off_events, off_counts = count_events_by_key(off_dir)
    on_events, on_counts = count_events_by_key(on_dir)
    delta_events = on_events - off_events

    only_on = []
    for key, on_v in on_counts.items():
        if off_counts.get(key, 0) == 0:
            only_on.append(
                {
                    "count_on": int(on_v),
                    "routine": key[0],
                    "mnemonic": key[1],
                    "disasm": key[2],
                }
            )
    only_on.sort(key=lambda x: x["count_on"], reverse=True)

    deltas = []
    for key in set(off_counts) | set(on_counts):
        off_v = int(off_counts.get(key, 0))
        on_v = int(on_counts.get(key, 0))
        dv = on_v - off_v
        if dv == 0:
            continue
        deltas.append(
            {
                "delta_on_minus_off": int(dv),
                "abs_delta": int(abs(dv)),
                "count_off": off_v,
                "count_on": on_v,
                "routine": key[0],
                "mnemonic": key[1],
                "disasm": key[2],
            }
        )
    deltas.sort(key=lambda x: x["abs_delta"], reverse=True)

    hints = []
    for comp in negative_components:
        if comp == "rep":
            hints.append(
                "rep<0: repetition density increased; inspect high positive delta store sites and on-only writer sites."
            )
        elif comp == "nleq":
            hints.append(
                "nleq<0: non-adjacent equality relations increased; often caused by added/reordered write loops."
            )
        elif comp == "per":
            hints.append(
                "per<0: periodic lag structure increased; often tied to regularized loop write sites."
            )
        elif comp == "motif":
            hints.append(
                "motif<0: ABAB/ABCABC motif density increased; usually from newly introduced patterned writes."
            )

    return {
        "task_id": task_id,
        "sample": sample,
        "off_run_dir": str(off_dir),
        "on_run_dir": str(on_dir),
        "negative_components": negative_components,
        "off_events": int(off_events),
        "on_events": int(on_events),
        "delta_events_on_minus_off": int(delta_events),
        "top_on_only_writers": only_on[:topk],
        "top_abs_delta_writers": deltas[:topk],
        "analysis_hints": hints,
    }


def main() -> int:
    args = parse_args()
    analysis_dir = Path(args.analysis_dir).resolve()
    negative_path = analysis_dir / "negative_after_excluding_defense_added.json"
    run_csv = analysis_dir / "per_run_metrics.csv"
    out_json = analysis_dir / "remaining_negative_rootcause.json"
    out_md = analysis_dir / "remaining_negative_rootcause.md"

    negative_data = json.loads(negative_path.read_text(encoding="utf-8"))
    negative_examples = negative_data["remaining_negative_examples"]
    run_rows = read_csv_rows(run_csv)
    run_idx = {(r["task_id"], r["sample"], r["mode"]): Path(r["run_dir"]) for r in run_rows}

    pair_components: dict[tuple[str, str], set[str]] = defaultdict(set)
    for item in negative_examples:
        task_id, _backend, sample, comp, _value = item
        if comp in {"nleq", "rep", "per", "motif"}:
            pair_components[(task_id, sample)].add(comp)

    jobs: list[dict[str, Any]] = []
    for (task_id, sample), comps in sorted(pair_components.items()):
        off_dir = run_idx[(task_id, sample, "off")]
        on_dir = run_idx[(task_id, sample, "on")]
        jobs.append(
            {
                "task_id": task_id,
                "sample": sample,
                "off_dir": str(off_dir),
                "on_dir": str(on_dir),
                "negative_components": sorted(comps),
                "topk": int(args.topk),
            }
        )

    results: list[dict[str, Any]] = []
    if args.jobs <= 1:
        for job in jobs:
            results.append(analyze_pair(job))
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.jobs) as ex:
            for item in ex.map(analyze_pair, jobs):
                results.append(item)

    results.sort(key=lambda x: (x["task_id"], x["sample"]))
    out_json.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    lines: list[str] = []
    lines.append("# Remaining Negative R_pat Root-Cause Report")
    lines.append("")
    lines.append(
        f"- Pairs analyzed: `{len(results)}` (all remaining negative pairs after defense-added exclusions)"
    )
    lines.append("")
    for r in results:
        lines.append(f"## {r['task_id']} / {r['sample']}")
        lines.append(f"- Negative components: `{', '.join(r['negative_components'])}`")
        lines.append(
            f"- Event count off/on: `{r['off_events']}` -> `{r['on_events']}` (delta `{r['delta_events_on_minus_off']}`)"
        )
        lines.append("- Primary hints:")
        for h in r["analysis_hints"]:
            lines.append(f"  - {h}")
        lines.append("- Top on-only writer sites:")
        for x in r["top_on_only_writers"][:8]:
            lines.append(
                f"  - +{x['count_on']} | {x['routine']} | {x['mnemonic']} | {x['disasm']}"
            )
        lines.append("- Top absolute delta writer sites (on-off):")
        for x in r["top_abs_delta_writers"][:8]:
            lines.append(
                f"  - {x['delta_on_minus_off']} | {x['routine']} | {x['mnemonic']} | {x['disasm']}"
            )
        lines.append("")
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(str(out_json))
    print(str(out_md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
