#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import math
import os
import struct
import time
from collections import Counter, defaultdict, deque
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np

HDR_V4 = struct.Struct("<8sIIQ")
REC_V4 = struct.Struct("<QIIIIQQQQQII")
MAGIC = b"PADDRTRC"
VER = 4

FORMAL_DIR_DEFAULT = Path("/path/to/high_leakage_workspace/higher_order_analysis_formal_20260422")
OUT_ROOT_DEFAULT = Path("/path/to/high_leakage_workspace")

TASK_GROUP_NAME = {
    "T01": "LeNet / MNIST",
    "T02": "VGGNet / MNIST",
    "T03": "VGGNet / CIFAR10",
    "T04": "SqueezeNet / CIFAR10",
    "T05": "SqueezeNet / ImageNet32",
    "T06": "ResNet / MNIST",
    "T07": "ResNet / CIFAR10",
    "T08": "ResNet / ImageNet32",
    "T09": "ResNet / CelebA",
    "T10": "ResNet / ChestX-ray",
}

# These are the routine names that are explicitly defense-only helpers in the current data.
DEFENSE_ROUTINE_SUBSTRINGS = [
    "patchedRelu",
    "reluResultBitsFromInputBits",
    "selectPatchedSeedBits",
    "applyInputZeroCheckerboardDitherOrExit",
    "inputzerodither",
    "tvm_relu_low12_f32",
    "tvm_relu6_low12_f32",
]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Recompute higher-order leakage metrics with event-level exclusion of "
            "defense-added helper routines and pointer/address-like written values."
        )
    )
    ap.add_argument(
        "--formal-dir",
        default=str(FORMAL_DIR_DEFAULT),
        help="existing formal analysis directory used as the source job manifest",
    )
    ap.add_argument(
        "--out-root",
        default=str(OUT_ROOT_DEFAULT),
        help="root directory used when --out-dir is omitted",
    )
    ap.add_argument(
        "--out-dir",
        default="",
        help="output directory; if omitted, a timestamped directory under --out-root is created",
    )
    ap.add_argument("--tau-max", type=int, default=64)
    ap.add_argument("--kgram", type=int, default=4)
    ap.add_argument("--bitmap-bits", type=int, default=1 << 20)
    ap.add_argument("--ams-depth", type=int, default=5)
    ap.add_argument("--ams-width", type=int, default=1 << 18)
    ap.add_argument("--jobs", type=int, default=min(64, os.cpu_count() or 1))
    ap.add_argument("--topk", type=int, default=12, help="top skipped defense routines per run in pair logs")
    ap.add_argument("--limit-pairs", type=int, default=0, help="debug only: process at most this many pairs")
    return ap.parse_args()


def splitmix64(x: int) -> int:
    x = (x + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    return x ^ (x >> 31)


def sym_from_block16(block16: bytes) -> int:
    lo = int.from_bytes(block16[:8], "little", signed=False)
    hi = int.from_bytes(block16[8:], "little", signed=False)
    return splitmix64(lo ^ splitmix64(hi))


class AMSF2:
    def __init__(self, depth: int, width: int) -> None:
        self.depth = int(depth)
        self.width = int(width)
        self.rows = [np.zeros(self.width, dtype=np.int64) for _ in range(self.depth)]
        self.seed_idx = [splitmix64(0xA5A5A5A5A5A5A5A5 + i * 17) for i in range(self.depth)]
        self.seed_sgn = [splitmix64(0x5A5A5A5A5A5A5A5A + i * 29) for i in range(self.depth)]

    def update(self, sym: int) -> None:
        for r in range(self.depth):
            h = splitmix64(sym ^ self.seed_idx[r])
            idx = h % self.width
            sgn = 1 if (splitmix64(sym ^ self.seed_sgn[r]) & 1) == 0 else -1
            self.rows[r][idx] += sgn

    def estimate_f2(self) -> float:
        vals = [float(np.dot(row, row)) for row in self.rows]
        return median(vals) if vals else 0.0


class LinearDistinct:
    def __init__(self, bitmap_bits: int, seed: int) -> None:
        self.m = int(bitmap_bits)
        self.bytes = bytearray((self.m + 7) // 8)
        self.seed = int(seed) & 0xFFFFFFFFFFFFFFFF

    def add_hash(self, value: int) -> None:
        h = splitmix64(value ^ self.seed)
        idx = h % self.m
        self.bytes[idx >> 3] |= 1 << (idx & 7)

    def estimate(self) -> float:
        ones = 0
        for b in self.bytes:
            ones += int(b).bit_count()
        zeros = self.m - ones
        if zeros <= 0:
            return float(self.m) * 10.0
        return -float(self.m) * math.log(float(zeros) / float(self.m))


def parse_task_backend(task_id: str) -> tuple[str, str]:
    prefix = task_id[:2]
    num = int(task_id[2:])
    if prefix not in {"MC", "IC", "TV", "TA"}:
        raise ValueError(f"bad task id: {task_id}")
    return f"T{num:02d}", prefix


def ratio(base: float, defended: float) -> float | None:
    if base <= 0.0:
        return None
    return (base - defended) / base


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_jsonl(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


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


def defense_rule_for_routine(routine: str) -> str | None:
    norm = routine or ""
    for needle in DEFENSE_ROUTINE_SUBSTRINGS:
        if needle in norm:
            return needle
    return None


def is_pointer_or_addr_like(block16: bytes) -> bool:
    q0, q1 = struct.unpack("<QQ", block16)
    u0, u1, u2, u3 = struct.unpack("<IIII", block16)

    def is_high_user_ptr64(x: int) -> bool:
        return 0x0000500000000000 <= x <= 0x00007FFFFFFFFFFF

    def is_low_code_ptr64(x: int) -> bool:
        return (x >> 32) == 0 and 0x00400000 <= (x & 0xFFFFFFFF) <= 0x01000000

    def is_addr32(x: int) -> bool:
        return (0x00400000 <= x <= 0x01000000) or (0x7F000000 <= x <= 0x80000000)

    return (
        is_high_user_ptr64(q0)
        or is_high_user_ptr64(q1)
        or is_low_code_ptr64(q0)
        or is_low_code_ptr64(q1)
        or is_addr32(u0)
        or is_addr32(u1)
        or is_addr32(u2)
        or is_addr32(u3)
    )


def counter_to_rows(counter: Counter[str], topk: int) -> list[dict[str, Any]]:
    return [{"key": key, "count": int(count)} for key, count in counter.most_common(topk)]


def analyze_trace_bin_filtered(
    trace_bin: Path,
    ipmap_path: Path,
    *,
    tau_max: int,
    kgram: int,
    bitmap_bits: int,
    ams_depth: int,
    ams_width: int,
    topk: int,
) -> dict[str, Any]:
    ipmap = parse_ipmap(ipmap_path)
    ams = AMSF2(depth=ams_depth, width=ams_width)
    kgram_dist = LinearDistinct(bitmap_bits=bitmap_bits, seed=0xDEADBEEF12345678)
    n_total_valid = 0
    n_kept = 0
    adj_equal = 0
    lag_equal = [0] * (tau_max + 1)
    lag_hist = deque(maxlen=tau_max)
    k_hist = deque(maxlen=max(kgram, 6))
    abab = 0
    abcabc = 0
    total_kgrams = 0
    total_m4 = 0
    total_m6 = 0
    skipped_defense = 0
    skipped_ptr_addr = 0
    defense_rule_counts: Counter[str] = Counter()
    defense_routine_counts: Counter[str] = Counter()
    prev_sym: int | None = None

    with trace_bin.open("rb") as handle:
        header = handle.read(HDR_V4.size)
        if len(header) != HDR_V4.size:
            raise RuntimeError(f"trace too small: {trace_bin}")
        magic, ver, _reserved, _page = HDR_V4.unpack(header)
        if magic != MAGIC or ver != VER:
            raise RuntimeError(f"bad trace header: {trace_bin}")
        while True:
            rec = handle.read(REC_V4.size)
            if not rec:
                break
            if len(rec) != REC_V4.size:
                raise RuntimeError(f"truncated record header: {trace_bin}")
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
            if size < 0:
                raise RuntimeError(f"bad size in record: {trace_bin}")
            handle.seek(int(size), 1)
            block16 = handle.read(16)
            if len(block16) != 16:
                raise RuntimeError(f"truncated block16: {trace_bin}")
            if stack_depth:
                handle.seek(int(stack_depth) * 8, 1)
            if int(size_read) != int(size) or int(size) <= 0:
                continue

            n_total_valid += 1
            routine, _mnemonic, _disasm = ipmap.get(int(ip), ("<unknown>", "<unknown>", "<unknown>"))
            rule = defense_rule_for_routine(routine)
            if rule is not None:
                skipped_defense += 1
                defense_rule_counts[rule] += 1
                defense_routine_counts[routine] += 1
                continue
            if is_pointer_or_addr_like(block16):
                skipped_ptr_addr += 1
                continue

            sym = sym_from_block16(block16)
            n_kept += 1
            ams.update(sym)
            if prev_sym is not None and prev_sym == sym:
                adj_equal += 1

            lag_hist.append(sym)
            for tau in range(2, min(tau_max, len(lag_hist) - 1) + 1):
                if lag_hist[-1] == lag_hist[-1 - tau]:
                    lag_equal[tau] += 1

            k_hist.append(sym)
            if len(k_hist) >= kgram:
                h = 0x9E3779B97F4A7C15
                for i in range(kgram):
                    h = splitmix64(h ^ k_hist[-kgram + i])
                kgram_dist.add_hash(h)
                total_kgrams += 1

            if len(k_hist) >= 4:
                a, b, c, d = k_hist[-4], k_hist[-3], k_hist[-2], k_hist[-1]
                if a == c and b == d and a != b:
                    abab += 1
                total_m4 += 1

            if len(k_hist) >= 6:
                a, b, c, d, e, f = (
                    k_hist[-6],
                    k_hist[-5],
                    k_hist[-4],
                    k_hist[-3],
                    k_hist[-2],
                    k_hist[-1],
                )
                if a == d and b == e and c == f and len({a, b, c}) == 3:
                    abcabc += 1
                total_m6 += 1

            prev_sym = sym

    if n_kept <= 1:
        return {
            "n_events_total_valid": int(n_total_valid),
            "n_events_after_filter": int(n_kept),
            "skipped_defense_events": int(skipped_defense),
            "skipped_ptr_addr_events": int(skipped_ptr_addr),
            "defense_rule_hits": counter_to_rows(defense_rule_counts, topk),
            "defense_routine_hits": counter_to_rows(defense_routine_counts, topk),
            "adj_equal": 0,
            "L_nleq": 0.0,
            "L_rep": 0.0,
            "L_per": 0.0,
            "L_motif": 0.0,
            "total_kgrams": 0,
            "total_m4_windows": 0,
            "total_m6_windows": 0,
            "abab_count": 0,
            "abcabc_count": 0,
        }

    f2 = ams.estimate_f2()
    eq_pairs_est = max((f2 - float(n_kept)) / 2.0, 0.0)
    nonlocal_eq_est = max(eq_pairs_est - float(adj_equal), 0.0)
    denom_nonadj = (float(n_kept) * float(n_kept - 1) / 2.0) - float(n_kept - 1)
    l_nleq = nonlocal_eq_est / denom_nonadj if denom_nonadj > 0 else 0.0
    l_nleq = max(0.0, min(1.0, l_nleq))

    if total_kgrams > 0:
        distinct_est = min(kgram_dist.estimate(), float(total_kgrams))
        l_rep = max(0.0, min(1.0, 1.0 - distinct_est / float(total_kgrams)))
    else:
        l_rep = 0.0

    l_per = 0.0
    for tau in range(2, tau_max + 1):
        denom = n_kept - tau
        if denom > 0:
            ratio_tau = float(lag_equal[tau]) / float(denom)
            if ratio_tau > l_per:
                l_per = ratio_tau
    l_per = max(0.0, min(1.0, l_per))

    m4 = float(abab) / float(total_m4) if total_m4 > 0 else 0.0
    m6 = float(abcabc) / float(total_m6) if total_m6 > 0 else 0.0
    l_motif = max(0.0, min(1.0, 0.5 * (m4 + m6)))

    return {
        "n_events_total_valid": int(n_total_valid),
        "n_events_after_filter": int(n_kept),
        "skipped_defense_events": int(skipped_defense),
        "skipped_ptr_addr_events": int(skipped_ptr_addr),
        "defense_rule_hits": counter_to_rows(defense_rule_counts, topk),
        "defense_routine_hits": counter_to_rows(defense_routine_counts, topk),
        "adj_equal": int(adj_equal),
        "L_nleq": float(l_nleq),
        "L_rep": float(l_rep),
        "L_per": float(l_per),
        "L_motif": float(l_motif),
        "total_kgrams": int(total_kgrams),
        "total_m4_windows": int(total_m4),
        "total_m6_windows": int(total_m6),
        "abab_count": int(abab),
        "abcabc_count": int(abcabc),
    }


def build_jobs_from_formal(formal_dir: Path) -> list[dict[str, str]]:
    rows = list(csv.DictReader((formal_dir / "per_run_metrics.csv").open("r", encoding="utf-8")))
    idx = {(r["task_id"], r["sample"], r["mode"]): r for r in rows}
    jobs: list[dict[str, str]] = []
    for task_id, sample in sorted({(r["task_id"], r["sample"]) for r in rows}):
        off_row = idx.get((task_id, sample, "off"))
        on_row = idx.get((task_id, sample, "on"))
        if off_row is None or on_row is None:
            continue
        jobs.append(
            {
                "task_id": task_id,
                "sample": sample,
                "session_tag": off_row.get("session_tag", ""),
                "run_off": off_row["run_dir"],
                "run_on": on_row["run_dir"],
            }
        )
    return jobs


def analyze_pair_job(job: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    started = time.time()
    task_id = str(job["task_id"])
    sample = str(job["sample"])
    session_tag = str(job.get("session_tag", ""))
    run_off = Path(job["run_off"])
    run_on = Path(job["run_on"])
    tau_max = int(job["tau_max"])
    kgram = int(job["kgram"])
    bitmap_bits = int(job["bitmap_bits"])
    ams_depth = int(job["ams_depth"])
    ams_width = int(job["ams_width"])
    topk = int(job["topk"])
    task_group, backend = parse_task_backend(task_id)

    metrics_off = analyze_trace_bin_filtered(
        run_off / "paddrtrace.bin",
        run_off / "paddrtrace.ip.txt",
        tau_max=tau_max,
        kgram=kgram,
        bitmap_bits=bitmap_bits,
        ams_depth=ams_depth,
        ams_width=ams_width,
        topk=topk,
    )
    metrics_on = analyze_trace_bin_filtered(
        run_on / "paddrtrace.bin",
        run_on / "paddrtrace.ip.txt",
        tau_max=tau_max,
        kgram=kgram,
        bitmap_bits=bitmap_bits,
        ams_depth=ams_depth,
        ams_width=ams_width,
        topk=topk,
    )

    run_off_row = {
        "task_id": task_id,
        "task_group": task_group,
        "task_group_name": TASK_GROUP_NAME[task_group],
        "backend": backend,
        "sample": sample,
        "mode": "off",
        "session_tag": session_tag,
        "run_dir": str(run_off),
        **metrics_off,
    }
    run_on_row = {
        "task_id": task_id,
        "task_group": task_group,
        "task_group_name": TASK_GROUP_NAME[task_group],
        "backend": backend,
        "sample": sample,
        "mode": "on",
        "session_tag": session_tag,
        "run_dir": str(run_on),
        **metrics_on,
    }

    pair = {
        "task_id": task_id,
        "task_group": task_group,
        "task_group_name": TASK_GROUP_NAME[task_group],
        "backend": backend,
        "sample": sample,
        "session_tag": session_tag,
        "run_off": str(run_off),
        "run_on": str(run_on),
        "L_nleq_base": metrics_off["L_nleq"],
        "L_nleq_def": metrics_on["L_nleq"],
        "L_rep_base": metrics_off["L_rep"],
        "L_rep_def": metrics_on["L_rep"],
        "L_per_base": metrics_off["L_per"],
        "L_per_def": metrics_on["L_per"],
        "L_motif_base": metrics_off["L_motif"],
        "L_motif_def": metrics_on["L_motif"],
    }
    pair["R_pat_nleq"] = ratio(pair["L_nleq_base"], pair["L_nleq_def"])
    pair["R_pat_rep"] = ratio(pair["L_rep_base"], pair["L_rep_def"])
    pair["R_pat_per"] = ratio(pair["L_per_base"], pair["L_per_def"])
    pair["R_pat_motif"] = ratio(pair["L_motif_base"], pair["L_motif_def"])
    vals = [
        v
        for v in (pair["R_pat_nleq"], pair["R_pat_rep"], pair["R_pat_per"], pair["R_pat_motif"])
        if v is not None
    ]
    pair["R_pat_agg"] = sum(vals) / len(vals) if vals else None

    pair_log = {
        "task_id": task_id,
        "task_group": task_group,
        "task_group_name": TASK_GROUP_NAME[task_group],
        "backend": backend,
        "sample": sample,
        "session_tag": session_tag,
        "elapsed_sec": round(time.time() - started, 3),
        "off": {
            "run_dir": str(run_off),
            "n_events_total_valid": metrics_off["n_events_total_valid"],
            "n_events_after_filter": metrics_off["n_events_after_filter"],
            "skipped_defense_events": metrics_off["skipped_defense_events"],
            "skipped_ptr_addr_events": metrics_off["skipped_ptr_addr_events"],
            "defense_rule_hits": metrics_off["defense_rule_hits"],
            "defense_routine_hits": metrics_off["defense_routine_hits"],
        },
        "on": {
            "run_dir": str(run_on),
            "n_events_total_valid": metrics_on["n_events_total_valid"],
            "n_events_after_filter": metrics_on["n_events_after_filter"],
            "skipped_defense_events": metrics_on["skipped_defense_events"],
            "skipped_ptr_addr_events": metrics_on["skipped_ptr_addr_events"],
            "defense_rule_hits": metrics_on["defense_rule_hits"],
            "defense_routine_hits": metrics_on["defense_routine_hits"],
        },
        "pair_metrics": {
            "R_pat_nleq": pair["R_pat_nleq"],
            "R_pat_rep": pair["R_pat_rep"],
            "R_pat_per": pair["R_pat_per"],
            "R_pat_motif": pair["R_pat_motif"],
            "R_pat_agg": pair["R_pat_agg"],
        },
    }
    return run_off_row, run_on_row, pair, pair_log


def mean(vals: list[float | None]) -> float | None:
    xs = [v for v in vals if v is not None]
    return (sum(xs) / len(xs)) if xs else None


def fmt_metric(v: float | None) -> str:
    if v is None:
        return "--"
    return f"{v:.6e}" if abs(v) < 1e-4 else f"{v:.6f}"


def fmt_pct(v: float | None) -> str:
    return "--" if v is None else f"{v * 100:.2f}\\%"


def build_output_dir(args: argparse.Namespace) -> Path:
    if args.out_dir:
        return Path(args.out_dir).resolve()
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return (Path(args.out_root).resolve() / f"higher_order_analysis_defenseop_ptraddr_{stamp}").resolve()


def build_spec(args: argparse.Namespace, formal_dir: Path, out_dir: Path, jobs: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "analysis_name": "higher_order_pattern_leakage_defenseop_ptraddr_rescan",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "formal_source_dir": str(formal_dir),
        "formal_source_manifest": str(formal_dir / "per_run_metrics.csv"),
        "output_dir": str(out_dir),
        "job_count_pairs": len(jobs),
        "job_count_runs": len(jobs) * 2,
        "symbol_source": "hash64(block16) from paddrtrace write events after event-level filtering",
        "metrics": {
            "L_nleq": "estimated non-adjacent equality-pair ratio using AMS F2 sketch; adjacent pairs excluded",
            "L_rep": f"repeated {args.kgram}-gram ratio via linear-counting distinct estimator",
            "L_per": f"max lag-equality ratio over lags 2..{args.tau_max}",
            "L_motif": "0.5*(ABAB density + ABCABC density)",
        },
        "parameters": {
            "tau_max": args.tau_max,
            "kgram": args.kgram,
            "bitmap_bits": args.bitmap_bits,
            "ams_depth": args.ams_depth,
            "ams_width": args.ams_width,
            "jobs": args.jobs,
            "topk": args.topk,
        },
        "filters": {
            "defense_added_routine_substrings": list(DEFENSE_ROUTINE_SUBSTRINGS),
            "pointer_or_addr_like_written_value": {
                "u64_high_user_ptr_range": ["0x0000500000000000", "0x00007fffffffffff"],
                "u64_low_code_ptr_low32_range": ["0x00400000", "0x01000000"],
                "u32_addr_ranges": [["0x00400000", "0x01000000"], ["0x7f000000", "0x80000000"]],
            },
        },
        "rpat_definition": "R_pat^(m)=(L_base^(m)-L_def^(m))/L_base^(m), base=off, def=on",
    }


def progress_payload(
    *,
    total: int,
    completed: int,
    started_at: float,
    current_pair: dict[str, Any] | None,
    failures: list[dict[str, Any]],
) -> dict[str, Any]:
    elapsed = time.time() - started_at
    rate = float(completed) / elapsed if elapsed > 0 and completed > 0 else None
    eta = (total - completed) / rate if rate and rate > 0 else None
    return {
        "total_pairs": int(total),
        "completed_pairs": int(completed),
        "failed_pairs": int(len(failures)),
        "started_at_epoch": started_at,
        "updated_at_epoch": time.time(),
        "elapsed_sec": round(elapsed, 3),
        "pairs_per_sec": None if rate is None else round(rate, 6),
        "eta_sec": None if eta is None else round(eta, 3),
        "last_completed": current_pair,
        "failures": failures[-10:],
    }


def main() -> int:
    args = parse_args()
    formal_dir = Path(args.formal_dir).resolve()
    out_dir = build_output_dir(args)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "pair_logs").mkdir(parents=True, exist_ok=True)

    jobs = build_jobs_from_formal(formal_dir)
    if args.limit_pairs > 0:
        jobs = jobs[: args.limit_pairs]
    for job in jobs:
        job["tau_max"] = args.tau_max
        job["kgram"] = args.kgram
        job["bitmap_bits"] = args.bitmap_bits
        job["ams_depth"] = args.ams_depth
        job["ams_width"] = args.ams_width
        job["topk"] = args.topk

    spec = build_spec(args, formal_dir, out_dir, jobs)
    write_json(out_dir / "spec.json", spec)
    write_json(out_dir / "job_manifest.json", jobs)

    spec_md = [
        "# Higher-Order Pattern Leakage Rescan",
        "",
        f"- Generated: `{spec['timestamp']}`",
        f"- Source job manifest: `{formal_dir / 'per_run_metrics.csv'}`",
        "- Scope: ciphertext trajectory only; event-level filtering before metric extraction",
        f"- Pair count: `{len(jobs)}`",
        f"- Worker count: `{args.jobs}`",
        "",
        "## Event Filters",
        "- Defense helper filter: exact substring match on the demangled routine field from `paddrtrace.ip.txt`.",
    ]
    for needle in DEFENSE_ROUTINE_SUBSTRINGS:
        spec_md.append(f"  - `{needle}`")
    spec_md += [
        "- Pointer/address-like value filter: excludes events whose written `block16` contains pointer-like or address-like integer values.",
        "",
        "## Metrics",
        "- `L_nleq`: non-adjacent equality-pair ratio.",
        "- `L_rep`: repeated k-gram ratio.",
        "- `L_per`: maximum lag-equality ratio for lags `2..tau_max`.",
        "- `L_motif`: average of ABAB density and ABCABC density.",
        "",
        "## Reduction",
        "- `R_pat^(m) = (L_base^(m)-L_def^(m))/L_base^(m)` with base=`off`, def=`on`.",
        "- `R_pat_agg`: mean of available component reductions.",
        "",
        "## Important Constraint",
        "- Filtering is based only on routine names and written-value heuristics.",
        "- No `conv/maxpool/relu_compute_` site is removed merely because it correlates with negative `R_pat`.",
    ]
    (out_dir / "SPEC.md").write_text("\n".join(spec_md) + "\n", encoding="utf-8")

    progress_log = out_dir / "progress.log"
    pair_jsonl = out_dir / "pair_results.jsonl"
    started_at = time.time()
    with progress_log.open("a", encoding="utf-8") as handle:
        handle.write(
            f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] start total_pairs={len(jobs)} jobs={args.jobs} out_dir={out_dir}\n"
        )

    per_run_rows: list[dict[str, Any]] = []
    per_pair_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    last_completed: dict[str, Any] | None = None

    def on_pair_done(run_off_row: dict[str, Any], run_on_row: dict[str, Any], pair: dict[str, Any], pair_log: dict[str, Any]) -> None:
        nonlocal last_completed
        per_run_rows.extend([run_off_row, run_on_row])
        per_pair_rows.append(pair)
        pair_path = out_dir / "pair_logs" / f"{pair['task_id']}_{pair['sample']}.json"
        write_json(pair_path, pair_log)
        append_jsonl(pair_jsonl, pair_log)
        completed = len(per_pair_rows)
        last_completed = {
            "task_id": pair["task_id"],
            "sample": pair["sample"],
            "backend": pair["backend"],
            "elapsed_sec": pair_log["elapsed_sec"],
            "off_after_filter": run_off_row["n_events_after_filter"],
            "on_after_filter": run_on_row["n_events_after_filter"],
            "off_skip_defense": run_off_row["skipped_defense_events"],
            "on_skip_defense": run_on_row["skipped_defense_events"],
            "off_skip_ptr": run_off_row["skipped_ptr_addr_events"],
            "on_skip_ptr": run_on_row["skipped_ptr_addr_events"],
        }
        payload = progress_payload(
            total=len(jobs),
            completed=completed,
            started_at=started_at,
            current_pair=last_completed,
            failures=failures,
        )
        write_json(out_dir / "progress.json", payload)
        with progress_log.open("a", encoding="utf-8") as handle:
            handle.write(
                f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] done {completed}/{len(jobs)} "
                f"{pair['task_id']}/{pair['sample']} backend={pair['backend']} "
                f"off_keep={run_off_row['n_events_after_filter']} on_keep={run_on_row['n_events_after_filter']} "
                f"off_skip_def={run_off_row['skipped_defense_events']} on_skip_def={run_on_row['skipped_defense_events']} "
                f"off_skip_ptr={run_off_row['skipped_ptr_addr_events']} on_skip_ptr={run_on_row['skipped_ptr_addr_events']} "
                f"elapsed_pair={pair_log['elapsed_sec']:.3f}s\n"
            )

    if args.jobs <= 1:
        for job in jobs:
            try:
                on_pair_done(*analyze_pair_job(job))
            except Exception as exc:
                failures.append(
                    {
                        "task_id": job["task_id"],
                        "sample": job["sample"],
                        "error": repr(exc),
                    }
                )
                write_json(
                    out_dir / "progress.json",
                    progress_payload(
                        total=len(jobs),
                        completed=len(per_pair_rows),
                        started_at=started_at,
                        current_pair=last_completed,
                        failures=failures,
                    ),
                )
                with progress_log.open("a", encoding="utf-8") as handle:
                    handle.write(
                        f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] fail {job['task_id']}/{job['sample']} error={repr(exc)}\n"
                    )
                break
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.jobs) as ex:
            future_map = {ex.submit(analyze_pair_job, job): job for job in jobs}
            for fut in concurrent.futures.as_completed(future_map):
                job = future_map[fut]
                try:
                    on_pair_done(*fut.result())
                except Exception as exc:
                    failures.append(
                        {
                            "task_id": job["task_id"],
                            "sample": job["sample"],
                            "error": repr(exc),
                        }
                    )
                    write_json(
                        out_dir / "progress.json",
                        progress_payload(
                            total=len(jobs),
                            completed=len(per_pair_rows),
                            started_at=started_at,
                            current_pair=last_completed,
                            failures=failures,
                        ),
                    )
                    with progress_log.open("a", encoding="utf-8") as handle:
                        handle.write(
                            f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] fail {job['task_id']}/{job['sample']} error={repr(exc)}\n"
                        )

    per_run_rows.sort(key=lambda x: (x["task_id"], x["sample"], x["mode"]))
    per_pair_rows.sort(key=lambda x: (x["task_id"], x["sample"]))
    write_json(out_dir / "per_run_metrics.json", per_run_rows)
    write_json(out_dir / "per_pair_rpat.json", per_pair_rows)

    run_fields = [
        "task_id",
        "task_group",
        "task_group_name",
        "backend",
        "sample",
        "mode",
        "session_tag",
        "run_dir",
        "n_events_total_valid",
        "n_events_after_filter",
        "skipped_defense_events",
        "skipped_ptr_addr_events",
        "adj_equal",
        "L_nleq",
        "L_rep",
        "L_per",
        "L_motif",
        "total_kgrams",
        "total_m4_windows",
        "total_m6_windows",
        "abab_count",
        "abcabc_count",
    ]
    with (out_dir / "per_run_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=run_fields)
        writer.writeheader()
        for row in per_run_rows:
            writer.writerow({k: row.get(k) for k in run_fields})

    pair_fields = [
        "task_id",
        "task_group",
        "task_group_name",
        "backend",
        "sample",
        "session_tag",
        "run_off",
        "run_on",
        "L_nleq_base",
        "L_nleq_def",
        "R_pat_nleq",
        "L_rep_base",
        "L_rep_def",
        "R_pat_rep",
        "L_per_base",
        "L_per_def",
        "R_pat_per",
        "L_motif_base",
        "L_motif_def",
        "R_pat_motif",
        "R_pat_agg",
    ]
    with (out_dir / "per_pair_rpat.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=pair_fields)
        writer.writeheader()
        for row in per_pair_rows:
            writer.writerow({k: row.get(k) for k in pair_fields})

    summary_rows: list[dict[str, Any]] = []
    by_group_backend: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in per_pair_rows:
        by_group_backend[(row["task_group"], row["backend"])].append(row)
    for tg in [f"T{i:02d}" for i in range(1, 11)]:
        for backend in ("MC", "IC", "TV", "TA"):
            rows = by_group_backend.get((tg, backend), [])
            item: dict[str, Any] = {
                "task_group": tg,
                "task_group_name": TASK_GROUP_NAME[tg],
                "backend": backend,
                "n_samples": len(rows),
            }
            for key in ("R_pat_nleq", "R_pat_rep", "R_pat_per", "R_pat_motif", "R_pat_agg"):
                item[key] = mean([x.get(key) for x in rows])
            summary_rows.append(item)
    write_json(out_dir / "summary_t01_t10_backend.json", summary_rows)
    with (out_dir / "summary_t01_t10_backend.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = [
            "task_group",
            "task_group_name",
            "backend",
            "n_samples",
            "R_pat_nleq",
            "R_pat_rep",
            "R_pat_per",
            "R_pat_motif",
            "R_pat_agg",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow({k: row.get(k) for k in fields})

    rows40: list[dict[str, Any]] = []
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    run_index = {(row["task_id"], row["sample"], row["mode"]): row for row in per_run_rows}
    for pair in per_pair_rows:
        by_task[pair["task_id"]].append(pair)
    for task_id in sorted(by_task):
        pairs = by_task[task_id]
        task_group = pairs[0]["task_group"]
        backend = pairs[0]["backend"]
        off_runs = [run_index[(task_id, p["sample"], "off")] for p in pairs]
        on_runs = [run_index[(task_id, p["sample"], "on")] for p in pairs]
        row = {
            "task_id": task_id,
            "task_group": task_group,
            "task_group_name": TASK_GROUP_NAME[task_group],
            "backend": backend,
            "num_samples": len(pairs),
            "L_nleq_off": mean([p["L_nleq_base"] for p in pairs]),
            "L_nleq_on": mean([p["L_nleq_def"] for p in pairs]),
            "L_rep_off": mean([p["L_rep_base"] for p in pairs]),
            "L_rep_on": mean([p["L_rep_def"] for p in pairs]),
            "L_per_off": mean([p["L_per_base"] for p in pairs]),
            "L_per_on": mean([p["L_per_def"] for p in pairs]),
            "L_motif_off": mean([p["L_motif_base"] for p in pairs]),
            "L_motif_on": mean([p["L_motif_def"] for p in pairs]),
            "off_skip_defense_mean": mean([float(r["skipped_defense_events"]) for r in off_runs]),
            "on_skip_defense_mean": mean([float(r["skipped_defense_events"]) for r in on_runs]),
            "off_skip_ptr_mean": mean([float(r["skipped_ptr_addr_events"]) for r in off_runs]),
            "on_skip_ptr_mean": mean([float(r["skipped_ptr_addr_events"]) for r in on_runs]),
        }
        for metric in ("nleq", "rep", "per", "motif"):
            off_key = f"L_{metric}_off"
            on_key = f"L_{metric}_on"
            off_v = row[off_key]
            on_v = row[on_key]
            row[f"Delta_L_{metric}_on_minus_off"] = None if off_v is None or on_v is None else (on_v - off_v)
        rows40.append(row)

    md_lines = [
        "# L Metrics Change Table (40 tasks)",
        "",
        "- Policy: event-level exclusion of defense-added helper routines by routine name, plus pointer/address-like written values.",
        "- No sample-level exclusion and no negative-driven site blacklist.",
        "- Scope: T01--T10 x MC/IC/TV/TA = 40 task entries (mean over the two matched samples).",
        "",
        "| Task | Group | Backend | L_nleq_off | L_nleq_on | Delta_nleq | L_rep_off | L_rep_on | Delta_rep | L_per_off | L_per_on | Delta_per | L_motif_off | L_motif_on | Delta_motif | off_skip_def | on_skip_def | off_skip_ptr | on_skip_ptr |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows40:
        md_lines.append(
            f"| {row['task_id']} | {row['task_group_name']} | {row['backend']} | "
            f"{fmt_metric(row['L_nleq_off'])} | {fmt_metric(row['L_nleq_on'])} | {fmt_metric(row['Delta_L_nleq_on_minus_off'])} | "
            f"{fmt_metric(row['L_rep_off'])} | {fmt_metric(row['L_rep_on'])} | {fmt_metric(row['Delta_L_rep_on_minus_off'])} | "
            f"{fmt_metric(row['L_per_off'])} | {fmt_metric(row['L_per_on'])} | {fmt_metric(row['Delta_L_per_on_minus_off'])} | "
            f"{fmt_metric(row['L_motif_off'])} | {fmt_metric(row['L_motif_on'])} | {fmt_metric(row['Delta_L_motif_on_minus_off'])} | "
            f"{fmt_metric(row['off_skip_defense_mean'])} | {fmt_metric(row['on_skip_defense_mean'])} | "
            f"{fmt_metric(row['off_skip_ptr_mean'])} | {fmt_metric(row['on_skip_ptr_mean'])} |"
        )
    l_table_name = "table_l_metrics_changes_excl_defenseop_ptraddr_40tasks.md"
    (out_dir / l_table_name).write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    (out_dir / "table_l_metrics_changes_excl_defense_added_40tasks.md").write_text(
        "\n".join(md_lines) + "\n", encoding="utf-8"
    )

    tex_lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\caption{Higher-order pattern leakage reduction ratio $R_{\\mathrm{pat}}^{\\mathrm{agg}}$ over T01--T10 across four execution chains after excluding defense-helper routine writes and pointer/address-like written values.}",
        "\\label{tab:rpat_agg_t01_t10}",
        "\\resizebox{\\columnwidth}{!}{%",
        "\\begin{tabular}{lccccc}",
        "\\toprule",
        "Group & Task group & MC & IC & TV & TA \\\\",
        "\\midrule",
    ]
    for tg in [f"T{i:02d}" for i in range(1, 11)]:
        vals = []
        for backend in ("MC", "IC", "TV", "TA"):
            rows = by_group_backend.get((tg, backend), [])
            vals.append(mean([x.get("R_pat_agg") for x in rows]))
        tex_lines.append(
            f"{tg} & {TASK_GROUP_NAME[tg]} & {fmt_pct(vals[0])} & {fmt_pct(vals[1])} & {fmt_pct(vals[2])} & {fmt_pct(vals[3])} \\\\"
        )
    tex_lines += ["\\bottomrule", "\\end{tabular}%", "}", "\\end{table}"]
    (out_dir / "table_rpat_agg_t01_t10.tex").write_text("\n".join(tex_lines) + "\n", encoding="utf-8")

    defense_rule_summary: Counter[str] = Counter()
    defense_routine_summary: Counter[str] = Counter()
    for row in per_run_rows:
        for item in row.get("defense_rule_hits", []):
            defense_rule_summary[str(item["key"])] += int(item["count"])
        for item in row.get("defense_routine_hits", []):
            defense_routine_summary[str(item["key"])] += int(item["count"])

    audit = {
        "analysis": "higher_order_pattern_leakage_defenseop_ptraddr_rescan",
        "generated_at": spec["timestamp"],
        "formal_source_dir": str(formal_dir),
        "output_dir": str(out_dir),
        "jobs_requested": args.jobs,
        "pairs_total": len(jobs),
        "pairs_completed": len(per_pair_rows),
        "pairs_failed": len(failures),
        "defense_routine_substrings": list(DEFENSE_ROUTINE_SUBSTRINGS),
        "defense_rule_hits_total": counter_to_rows(defense_rule_summary, 32),
        "defense_routine_hits_total": counter_to_rows(defense_routine_summary, 32),
        "pointer_filter": {
            "u64_high_user_ptr_range": ["0x0000500000000000", "0x00007fffffffffff"],
            "u64_low_code_ptr_low32_range": ["0x00400000", "0x01000000"],
            "u32_addr_ranges": [["0x00400000", "0x01000000"], ["0x7f000000", "0x80000000"]],
        },
        "outputs": [
            str(out_dir / "per_run_metrics.csv"),
            str(out_dir / "per_pair_rpat.csv"),
            str(out_dir / l_table_name),
            str(out_dir / "table_rpat_agg_t01_t10.tex"),
            str(out_dir / "progress.log"),
        ],
    }
    write_json(out_dir / "filter_audit.json", audit)

    final_progress = progress_payload(
        total=len(jobs),
        completed=len(per_pair_rows),
        started_at=started_at,
        current_pair=last_completed,
        failures=failures,
    )
    write_json(out_dir / "progress.json", final_progress)
    with progress_log.open("a", encoding="utf-8") as handle:
        handle.write(
            f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] finish completed={len(per_pair_rows)} failed={len(failures)}\n"
        )

    if failures:
        write_json(out_dir / "failures.json", failures)
        print(str(out_dir))
        print(f"pairs_completed={len(per_pair_rows)} failures={len(failures)}")
        return 1

    print(str(out_dir))
    print(f"pairs={len(per_pair_rows)} runs={len(per_run_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
