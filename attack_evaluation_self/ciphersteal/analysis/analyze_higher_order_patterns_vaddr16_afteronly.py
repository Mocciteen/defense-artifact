#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import math
import os
import struct
import tempfile
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

USED_BEFORE = 1 << 2

OUT_ROOT_DEFAULT = Path("/path/to/high_leakage_workspace")
RESULTS_ROOT_DEFAULT = Path("/path/to/high_leakage_workspace/results")

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
    "T11": "MobileNet / MNIST",
    "T12": "MobileNet / CIFAR10",
    "T13": "MobileNet / CelebA",
    "T14": "MobileNet / ChestX-ray",
    "T15": "DenseNet / ImageNet32",
    "T16": "DenseNet / CelebA",
    "T17": "DenseNet / ChestX-ray",
}

DEFENSE_ROUTINE_SUBSTRINGS = [
    "patchedRelu",
    "reluResultBitsFromInputBits",
    "selectPatchedSeedBits",
    "applyInputZeroCheckerboardDitherOrExit",
    "inputzerodither",
    "tvm_relu_low12_f32",
    "tvm_relu6_low12_f32",
]

OUTPUT_FORMAT_ROUTINE_SUBSTRINGS = [
    "__printf_",
    "__mpn_",
    "hack_digit",
]

RUNTIME_INFRA_ROUTINE_SUBSTRINGS = [
    "__libc_malloc",
    "_int_malloc",
    "__memmove_",
    "_M_realloc_insert",
    "collateIcE12do_transform",
]

EXACT_NEGATIVE_FALLBACK_BASE_MAX = 5e-3
EXACT_NEGATIVE_FALLBACK_MAX_EVENTS = 6_000_000
EXACT_REP_MIN_PARTITIONS = 64
EXACT_REP_MAX_PARTITIONS = 2048
EXACT_REP_TARGET_PARTITION_BYTES = 16 * 1024 * 1024
EXACT_REP_FLUSH_BYTES = 128 * 1024


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Compute higher-order leakage metrics on vaddr16-grouped after-only "
            "block trajectories, with defense-helper and pointer/address-like "
            "writes filtered out before subsequence projection."
        )
    )
    ap.add_argument(
        "--results-root",
        default=str(RESULTS_ROOT_DEFAULT),
        help="root directory that contains glow/tvm result trees",
    )
    ap.add_argument(
        "--job-manifest",
        default="",
        help="explicit job manifest JSON; if provided, skip results-root discovery and analyze exactly those pairs",
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
    ap.add_argument("--jobs", type=int, default=min(16, os.cpu_count() or 1))
    ap.add_argument("--topk", type=int, default=12, help="top skipped defense routines per run in pair logs")
    ap.add_argument("--limit-pairs", type=int, default=0, help="debug only: process at most this many pairs")
    ap.add_argument(
        "--rep-exact-mode",
        choices=("auto", "always", "never"),
        default="auto",
        help=(
            "how to compute L_rep. 'auto' keeps the fast estimator but recomputes "
            "exactly when the estimator saturates or yields negative R_pat_rep; "
            "'always' recomputes L_rep exactly for every run; 'never' keeps the "
            "fast estimator only."
        ),
    )
    ap.add_argument(
        "--scratch-root",
        default="",
        help="optional scratch root for exact repeated-kgram recomputation; defaults to <out-dir>/.scratch",
    )
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


def combine_u64(a: int, b: int) -> int:
    return splitmix64((a & 0xFFFFFFFFFFFFFFFF) ^ splitmix64(b & 0xFFFFFFFFFFFFFFFF))


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


class ExactRepPartitionWriter:
    def __init__(self, root: Path, *, nparts: int, flush_bytes: int) -> None:
        self.root = root
        self.nparts = int(nparts)
        self.flush_bytes = int(flush_bytes)
        self.root.mkdir(parents=True, exist_ok=True)
        self.buffers = [bytearray() for _ in range(self.nparts)]
        self.handles: dict[int, Any] = {}

    def _path(self, part: int) -> Path:
        return self.root / f"part_{part:04d}.bin"

    def _handle(self, part: int) -> Any:
        handle = self.handles.get(part)
        if handle is None:
            handle = self._path(part).open("ab")
            self.handles[part] = handle
        return handle

    def write(self, part: int, payload: bytes) -> None:
        buf = self.buffers[part]
        buf.extend(payload)
        if len(buf) >= self.flush_bytes:
            self.flush_part(part)

    def flush_part(self, part: int) -> None:
        buf = self.buffers[part]
        if not buf:
            return
        self._handle(part).write(buf)
        buf.clear()

    def close(self) -> None:
        for part in range(self.nparts):
            self.flush_part(part)
        for handle in self.handles.values():
            handle.close()
        self.handles.clear()


def choose_exact_rep_partitions(total_kgrams_hint: int, *, key_words: int) -> int:
    rec_bytes = max(int(key_words) * 8, 8)
    total_bytes = max(int(total_kgrams_hint), 1) * rec_bytes
    nparts = EXACT_REP_MIN_PARTITIONS
    while (
        nparts < EXACT_REP_MAX_PARTITIONS
        and (total_bytes // nparts) > EXACT_REP_TARGET_PARTITION_BYTES
    ):
        nparts *= 2
    return int(nparts)


def exact_rep_partition_id(words: tuple[int, ...], *, nparts: int) -> int:
    h = 0xA0761D6478BD642F
    for word in words:
        h = splitmix64(h ^ (int(word) & 0xFFFFFFFFFFFFFFFF))
    if nparts > 0 and (nparts & (nparts - 1)) == 0:
        return int(h & (nparts - 1))
    return int(h % nparts)


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


def pair_job_key(task_id: str, sample: str) -> str:
    return f"{task_id}::{sample}"


def pair_log_job_key(payload: dict[str, Any]) -> str:
    return pair_job_key(str(payload["task_id"]), str(payload["sample"]))


def load_completed_pair_rows(pair_logs_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, dict[str, Any]]]:
    per_run_rows: list[dict[str, Any]] = []
    per_pair_rows: list[dict[str, Any]] = []
    loaded: dict[str, dict[str, Any]] = {}
    if not pair_logs_dir.exists():
        return per_run_rows, per_pair_rows, loaded
    for path in sorted(pair_logs_dir.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        full_rows = payload.get("full_rows")
        if not isinstance(full_rows, dict):
            continue
        run_off_row = full_rows.get("run_off_row")
        run_on_row = full_rows.get("run_on_row")
        pair_row = full_rows.get("pair_row")
        if not isinstance(run_off_row, dict) or not isinstance(run_on_row, dict) or not isinstance(pair_row, dict):
            continue
        key = pair_log_job_key(payload)
        per_run_rows.extend([run_off_row, run_on_row])
        per_pair_rows.append(pair_row)
        loaded[key] = payload
    return per_run_rows, per_pair_rows, loaded


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
    for needle in OUTPUT_FORMAT_ROUTINE_SUBSTRINGS:
        if needle in norm:
            return needle
    for needle in RUNTIME_INFRA_ROUTINE_SUBSTRINGS:
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


def build_task_roots(results_root: Path) -> tuple[dict[str, Path], dict[str, Path]]:
    glow_fill_runs = results_root / "glow_fill_20260422" / "runs"
    glow_final_runs = results_root / "glow_final_20260422" / "runs"
    mc03_repaired_runs = results_root / "mc03_fixed4_1110_seedprng" / "runs"
    maxpool_runs = results_root / "maxpool_new_run" / "runs"
    tvm_fill_runs = results_root / "tvm_fill_20260422" / "runs"
    tvm_final_runs = results_root / "tvm_final_20260422" / "runs"

    base_roots: dict[str, Path] = {}
    for task_id in ("MC01", "MC02", "MC03", "MC04", "MC05", "MC06", "MC07", "MC09", "MC10"):
        base_roots[task_id] = glow_fill_runs
    base_roots["MC08"] = glow_final_runs

    for task_id in ("IC01", "IC02", "IC03", "IC04", "IC05", "IC08", "IC09", "IC10"):
        base_roots[task_id] = glow_fill_runs
    for task_id in ("IC06", "IC07"):
        base_roots[task_id] = glow_final_runs

    for task_id in ("TV01", "TV04", "TV05", "TV09", "TV10", "TA01", "TA04", "TA05", "TA09", "TA10"):
        base_roots[task_id] = tvm_fill_runs
    for task_id in ("TV02", "TV03", "TV06", "TV07", "TV08", "TA02", "TA03", "TA06", "TA07", "TA08"):
        base_roots[task_id] = tvm_final_runs

    on_override_roots: dict[str, Path] = {}
    for task_id in ("MC04", "MC05", "MC06", "MC08", "MC09", "MC10"):
        on_override_roots[task_id] = maxpool_runs

    on_override_samples: dict[tuple[str, str], Path] = {
        ("MC03", "idx00042"): mc03_repaired_runs,
        ("MC03", "idx00114"): mc03_repaired_runs,
    }

    return base_roots, on_override_roots, on_override_samples


def collect_samples(task_root: Path, task_id: str, mode: str) -> set[str]:
    task_dir = task_root / task_id
    if not task_dir.is_dir():
        return set()
    out: set[str] = set()
    for sample_dir in sorted(task_dir.iterdir()):
        run_dir = sample_dir / mode
        if (run_dir / "paddrtrace.bin").is_file() and (run_dir / "paddrtrace.ip.txt").is_file():
            out.add(sample_dir.name)
    return out


def build_jobs_from_results(results_root: Path) -> list[dict[str, Any]]:
    base_roots, on_override_roots, on_override_samples = build_task_roots(results_root)
    jobs: list[dict[str, Any]] = []
    for task_id in sorted(base_roots):
        off_root = base_roots[task_id]
        default_on_root = on_override_roots.get(task_id, off_root)
        sample_pool = set(collect_samples(off_root, task_id, "off"))
        sample_pool &= set(collect_samples(default_on_root, task_id, "on"))
        for (sample_task_id, sample_name), sample_root in on_override_samples.items():
            if sample_task_id != task_id:
                continue
            if sample_name in collect_samples(off_root, task_id, "off") and sample_name in collect_samples(sample_root, task_id, "on"):
                sample_pool.add(sample_name)
        samples = sorted(sample_pool)
        for sample in samples:
            on_root = on_override_samples.get((task_id, sample), default_on_root)
            run_off = off_root / task_id / sample / "off"
            run_on = on_root / task_id / sample / "on"
            jobs.append(
                {
                    "task_id": task_id,
                    "sample": sample,
                    "session_tag": off_root.parent.name if off_root == on_root else f"{off_root.parent.name}->{on_root.parent.name}",
                    "run_off": str(run_off),
                    "run_on": str(run_on),
                }
            )
    return jobs


def analyze_trace_bin_vaddr16_afteronly(
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
    hist_max = max(tau_max, kgram, 6)

    n_total_valid = 0
    n_after_valid = 0
    n_kept = 0
    adj_equal = 0
    lag_equal = [0] * (tau_max + 1)
    total_kgrams = 0
    total_m4 = 0
    total_m6 = 0
    abab = 0
    abcabc = 0
    skipped_before = 0
    skipped_defense = 0
    skipped_ptr_addr = 0
    defense_rule_counts: Counter[str] = Counter()
    defense_routine_counts: Counter[str] = Counter()

    block_lengths: dict[int, int] = {}
    block_histories: dict[int, deque[int]] = {}

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
                vaddr,
                _paddr,
                _paddr16,
                ip,
                _instr_id,
                flags,
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
            if flags & USED_BEFORE:
                skipped_before += 1
                continue

            n_after_valid += 1
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

            block_key = int(vaddr) & ~0xF
            sym = sym_from_block16(block16)
            hist = block_histories.get(block_key)
            if hist is None:
                hist = deque(maxlen=hist_max)
                block_histories[block_key] = hist

            combined_sym = combine_u64(block_key, sym)
            ams.update(combined_sym)

            if hist and hist[-1] == sym:
                adj_equal += 1

            max_tau_here = min(tau_max, len(hist))
            for tau in range(2, max_tau_here + 1):
                if hist[-tau] == sym:
                    lag_equal[tau] += 1

            if len(hist) >= kgram - 1:
                h = 0x9E3779B97F4A7C15
                for old_sym in tuple(hist)[-(kgram - 1):]:
                    h = splitmix64(h ^ old_sym)
                h = splitmix64(h ^ sym)
                kgram_dist.add_hash(combine_u64(block_key, h))
                total_kgrams += 1

            if len(hist) >= 3:
                a, b, c = hist[-3], hist[-2], hist[-1]
                if a == c and b == sym and a != b:
                    abab += 1
                total_m4 += 1

            if len(hist) >= 5:
                a, b, c, d, e = hist[-5], hist[-4], hist[-3], hist[-2], hist[-1]
                if a == d and b == e and c == sym and len({a, b, c}) == 3:
                    abcabc += 1
                total_m6 += 1

            hist.append(sym)
            block_lengths[block_key] = block_lengths.get(block_key, 0) + 1
            n_kept += 1

    if n_kept <= 1:
        return {
            "n_events_total_valid": int(n_total_valid),
            "n_events_after_only_valid": int(n_after_valid),
            "n_events_after_filter": int(n_kept),
            "n_blocks_after_filter": int(len(block_lengths)),
            "max_block_len_after_filter": int(max(block_lengths.values()) if block_lengths else 0),
            "skipped_before_events": int(skipped_before),
            "skipped_defense_events": int(skipped_defense),
            "skipped_ptr_addr_events": int(skipped_ptr_addr),
            "defense_rule_hits": counter_to_rows(defense_rule_counts, topk),
            "defense_routine_hits": counter_to_rows(defense_routine_counts, topk),
            "adj_equal": 0,
            "L_nleq": 0.0,
            "L_rep": 0.0,
            "rep_distinct_estimate": 0.0,
            "rep_bitmap_zero_bits": int(bitmap_bits),
            "rep_estimator_saturated": False,
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
    denom_nonadj = 0.0
    for n in block_lengths.values():
        if n >= 2:
            denom_nonadj += (float(n) * float(n - 1) / 2.0) - float(n - 1)
    l_nleq = nonlocal_eq_est / denom_nonadj if denom_nonadj > 0 else 0.0
    l_nleq = max(0.0, min(1.0, l_nleq))

    if total_kgrams > 0:
        rep_bitmap_zero_bits = 0
        for b in kgram_dist.bytes:
            rep_bitmap_zero_bits += 8 - int(b).bit_count()
        rep_estimator_saturated = rep_bitmap_zero_bits <= 0
        distinct_est = min(kgram_dist.estimate(), float(total_kgrams))
        l_rep = max(0.0, min(1.0, 1.0 - distinct_est / float(total_kgrams)))
    else:
        rep_bitmap_zero_bits = int(bitmap_bits)
        rep_estimator_saturated = False
        distinct_est = 0.0
        l_rep = 0.0

    l_per = 0.0
    for tau in range(2, tau_max + 1):
        denom_tau = sum(max(n - tau, 0) for n in block_lengths.values())
        if denom_tau > 0:
            ratio_tau = float(lag_equal[tau]) / float(denom_tau)
            if ratio_tau > l_per:
                l_per = ratio_tau
    l_per = max(0.0, min(1.0, l_per))

    m4 = float(abab) / float(total_m4) if total_m4 > 0 else 0.0
    m6 = float(abcabc) / float(total_m6) if total_m6 > 0 else 0.0
    l_motif = max(0.0, min(1.0, 0.5 * (m4 + m6)))

    return {
        "n_events_total_valid": int(n_total_valid),
        "n_events_after_only_valid": int(n_after_valid),
        "n_events_after_filter": int(n_kept),
        "n_blocks_after_filter": int(len(block_lengths)),
        "max_block_len_after_filter": int(max(block_lengths.values()) if block_lengths else 0),
        "skipped_before_events": int(skipped_before),
        "skipped_defense_events": int(skipped_defense),
        "skipped_ptr_addr_events": int(skipped_ptr_addr),
        "defense_rule_hits": counter_to_rows(defense_rule_counts, topk),
        "defense_routine_hits": counter_to_rows(defense_routine_counts, topk),
        "adj_equal": int(adj_equal),
        "L_nleq": float(l_nleq),
        "L_rep": float(l_rep),
        "rep_distinct_estimate": float(distinct_est),
        "rep_bitmap_zero_bits": int(rep_bitmap_zero_bits),
        "rep_estimator_saturated": bool(rep_estimator_saturated),
        "L_per": float(l_per),
        "L_motif": float(l_motif),
        "total_kgrams": int(total_kgrams),
        "total_m4_windows": int(total_m4),
        "total_m6_windows": int(total_m6),
        "abab_count": int(abab),
        "abcabc_count": int(abcabc),
    }


def analyze_trace_bin_vaddr16_afteronly_exact_subset(
    trace_bin: Path,
    ipmap_path: Path,
    *,
    kgram: int,
    need_rep: bool,
    need_nleq: bool,
) -> dict[str, Any]:
    if not need_rep and not need_nleq:
        return {}

    ipmap = parse_ipmap(ipmap_path)
    block_lengths: dict[int, int] = {}

    eq_pairs_total = 0
    adj_equal = 0
    last_sym_by_block: dict[int, int] = {}
    sym_counts_by_block: dict[int, dict[int, int]] = {}

    rep_histories: dict[int, deque[int]] = {}
    distinct_kgrams: set[tuple[int, ...]] = set()
    total_kgrams = 0

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
                vaddr,
                _paddr,
                _paddr16,
                ip,
                _instr_id,
                flags,
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
            if flags & USED_BEFORE:
                continue

            routine, _mnemonic, _disasm = ipmap.get(int(ip), ("<unknown>", "<unknown>", "<unknown>"))
            if defense_rule_for_routine(routine) is not None:
                continue
            if is_pointer_or_addr_like(block16):
                continue

            block_key = int(vaddr) & ~0xF
            sym = sym_from_block16(block16)

            if need_nleq:
                block_counts = sym_counts_by_block.get(block_key)
                if block_counts is None:
                    block_counts = {}
                    sym_counts_by_block[block_key] = block_counts
                prev_count = block_counts.get(sym, 0)
                eq_pairs_total += prev_count
                if last_sym_by_block.get(block_key) == sym:
                    adj_equal += 1
                block_counts[sym] = prev_count + 1
                last_sym_by_block[block_key] = sym

            if need_rep:
                hist = rep_histories.get(block_key)
                if hist is None:
                    hist = deque(maxlen=max(kgram - 1, 1))
                    rep_histories[block_key] = hist
                if len(hist) >= kgram - 1:
                    distinct_kgrams.add((block_key, *tuple(hist), sym))
                    total_kgrams += 1
                hist.append(sym)

            block_lengths[block_key] = block_lengths.get(block_key, 0) + 1

    out: dict[str, Any] = {}
    if need_nleq:
        denom_nonadj = 0
        for n in block_lengths.values():
            if n >= 2:
                denom_nonadj += (n * (n - 1) // 2) - (n - 1)
        nonlocal_eq_exact = max(eq_pairs_total - adj_equal, 0)
        out["L_nleq"] = float(nonlocal_eq_exact / denom_nonadj) if denom_nonadj > 0 else 0.0
        out["exact_eq_pairs_total"] = int(eq_pairs_total)
        out["exact_adj_equal_pairs"] = int(adj_equal)
        out["exact_nonadj_equal_pairs"] = int(nonlocal_eq_exact)
        out["exact_nonadj_denom"] = int(denom_nonadj)

    if need_rep:
        distinct_count = len(distinct_kgrams)
        out["L_rep"] = 0.0 if total_kgrams <= 0 else float(1.0 - (distinct_count / float(total_kgrams)))
        out["exact_total_kgrams"] = int(total_kgrams)
        out["exact_distinct_kgrams"] = int(distinct_count)
        out["exact_repeated_kgrams"] = int(total_kgrams - distinct_count)

    return out


def analyze_trace_bin_vaddr16_afteronly_exact_rep_partitioned(
    trace_bin: Path,
    ipmap_path: Path,
    *,
    kgram: int,
    scratch_root: Path,
    total_kgrams_hint: int,
    label: str,
) -> dict[str, Any]:
    if int(kgram) < 2:
        raise ValueError(f"kgram must be >= 2, got {kgram}")

    ipmap = parse_ipmap(ipmap_path)
    key_words = int(kgram) + 1
    rec_pack = struct.Struct("<" + ("Q" * key_words))
    nparts = choose_exact_rep_partitions(total_kgrams_hint, key_words=key_words)
    total_kgrams = 0

    with tempfile.TemporaryDirectory(
        dir=str(scratch_root),
        prefix=f"rep_exact_{label}_",
    ) as tmpdir_str:
        tmpdir = Path(tmpdir_str)
        writer = ExactRepPartitionWriter(tmpdir, nparts=nparts, flush_bytes=EXACT_REP_FLUSH_BYTES)
        block_histories: dict[int, deque[int]] = {}

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
                    vaddr,
                    _paddr,
                    _paddr16,
                    ip,
                    _instr_id,
                    flags,
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
                if flags & USED_BEFORE:
                    continue

                routine, _mnemonic, _disasm = ipmap.get(int(ip), ("<unknown>", "<unknown>", "<unknown>"))
                if defense_rule_for_routine(routine) is not None:
                    continue
                if is_pointer_or_addr_like(block16):
                    continue

                block_key = int(vaddr) & ~0xF
                sym = sym_from_block16(block16)
                hist = block_histories.get(block_key)
                if hist is None:
                    hist = deque(maxlen=max(kgram - 1, 1))
                    block_histories[block_key] = hist
                if len(hist) >= kgram - 1:
                    words = (block_key, *tuple(hist), sym)
                    writer.write(exact_rep_partition_id(words, nparts=nparts), rec_pack.pack(*words))
                    total_kgrams += 1
                hist.append(sym)

        writer.close()

        distinct_total = 0
        for part in range(nparts):
            part_path = tmpdir / f"part_{part:04d}.bin"
            if not part_path.exists() or part_path.stat().st_size == 0:
                continue
            arr = np.fromfile(part_path, dtype=np.uint64)
            if arr.size % key_words != 0:
                raise RuntimeError(f"misaligned exact rep partition: {part_path}")
            arr = arr.reshape((-1, key_words))
            distinct_total += int(np.unique(arr, axis=0).shape[0])

    return {
        "L_rep": 0.0 if total_kgrams <= 0 else float(1.0 - (float(distinct_total) / float(total_kgrams))),
        "exact_total_kgrams": int(total_kgrams),
        "exact_distinct_kgrams": int(distinct_total),
        "exact_repeated_kgrams": int(total_kgrams - distinct_total),
        "exact_rep_partitions": int(nparts),
    }


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
    rep_exact_mode = str(job.get("rep_exact_mode", "auto"))
    scratch_root = Path(job["scratch_root"])
    task_group, backend = parse_task_backend(task_id)

    metrics_off = analyze_trace_bin_vaddr16_afteronly(
        run_off / "paddrtrace.bin",
        run_off / "paddrtrace.ip.txt",
        tau_max=tau_max,
        kgram=kgram,
        bitmap_bits=bitmap_bits,
        ams_depth=ams_depth,
        ams_width=ams_width,
        topk=topk,
    )
    metrics_on = analyze_trace_bin_vaddr16_afteronly(
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

    if rep_exact_mode == "always":
        need_exact_rep = True
        rep_exact_reason = "rep_exact_mode_always"
    elif rep_exact_mode == "never":
        need_exact_rep = False
        rep_exact_reason = ""
    else:
        rep_negative = pair["R_pat_rep"] is not None and pair["R_pat_rep"] < 0.0
        rep_saturated = bool(metrics_off.get("rep_estimator_saturated")) or bool(metrics_on.get("rep_estimator_saturated"))
        need_exact_rep = bool(rep_negative or rep_saturated)
        if rep_negative and rep_saturated:
            rep_exact_reason = "negative_or_saturated"
        elif rep_negative:
            rep_exact_reason = "negative_rpat_rep"
        elif rep_saturated:
            rep_exact_reason = "rep_estimator_saturated"
        else:
            rep_exact_reason = ""
    need_exact_nleq = (
        pair["R_pat_nleq"] is not None
        and pair["R_pat_nleq"] < 0.0
        and pair["L_nleq_base"] <= EXACT_NEGATIVE_FALLBACK_BASE_MAX
        and max(metrics_off["n_events_after_filter"], metrics_on["n_events_after_filter"]) <= EXACT_NEGATIVE_FALLBACK_MAX_EVENTS
    )
    exact_negative_fallback: dict[str, Any] = {}
    if need_exact_rep or need_exact_nleq:
        exact_off: dict[str, Any] = {}
        exact_on: dict[str, Any] = {}
        if need_exact_nleq:
            exact_off.update(
                analyze_trace_bin_vaddr16_afteronly_exact_subset(
                    run_off / "paddrtrace.bin",
                    run_off / "paddrtrace.ip.txt",
                    kgram=kgram,
                    need_rep=False,
                    need_nleq=True,
                )
            )
            exact_on.update(
                analyze_trace_bin_vaddr16_afteronly_exact_subset(
                    run_on / "paddrtrace.bin",
                    run_on / "paddrtrace.ip.txt",
                    kgram=kgram,
                    need_rep=False,
                    need_nleq=True,
                )
            )
        if need_exact_rep:
            exact_rep_off = analyze_trace_bin_vaddr16_afteronly_exact_rep_partitioned(
                run_off / "paddrtrace.bin",
                run_off / "paddrtrace.ip.txt",
                kgram=kgram,
                scratch_root=scratch_root,
                total_kgrams_hint=metrics_off["total_kgrams"],
                label=f"{task_id}_{sample}_off",
            )
            exact_rep_on = analyze_trace_bin_vaddr16_afteronly_exact_rep_partitioned(
                run_on / "paddrtrace.bin",
                run_on / "paddrtrace.ip.txt",
                kgram=kgram,
                scratch_root=scratch_root,
                total_kgrams_hint=metrics_on["total_kgrams"],
                label=f"{task_id}_{sample}_on",
            )
            exact_off.update(exact_rep_off)
            exact_on.update(exact_rep_on)
            metrics_off["L_rep"] = float(exact_rep_off["L_rep"])
            metrics_on["L_rep"] = float(exact_rep_on["L_rep"])
            pair["L_rep_base"] = float(exact_rep_off["L_rep"])
            pair["L_rep_def"] = float(exact_rep_on["L_rep"])
            pair["R_pat_rep"] = ratio(pair["L_rep_base"], pair["L_rep_def"])
            exact_negative_fallback["rep"] = {
                "applied": True,
                "reason": rep_exact_reason,
                "off": exact_off,
                "on": exact_on,
            }
        if need_exact_nleq:
            metrics_off["L_nleq"] = float(exact_off["L_nleq"])
            metrics_on["L_nleq"] = float(exact_on["L_nleq"])
            pair["L_nleq_base"] = float(exact_off["L_nleq"])
            pair["L_nleq_def"] = float(exact_on["L_nleq"])
            pair["R_pat_nleq"] = ratio(pair["L_nleq_base"], pair["L_nleq_def"])
            exact_negative_fallback["nleq"] = {
                "applied": True,
                "off": exact_off,
                "on": exact_on,
            }
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
            "n_events_after_only_valid": metrics_off["n_events_after_only_valid"],
            "n_events_after_filter": metrics_off["n_events_after_filter"],
            "n_blocks_after_filter": metrics_off["n_blocks_after_filter"],
            "max_block_len_after_filter": metrics_off["max_block_len_after_filter"],
            "skipped_before_events": metrics_off["skipped_before_events"],
            "skipped_defense_events": metrics_off["skipped_defense_events"],
            "skipped_ptr_addr_events": metrics_off["skipped_ptr_addr_events"],
            "defense_rule_hits": metrics_off["defense_rule_hits"],
            "defense_routine_hits": metrics_off["defense_routine_hits"],
        },
        "on": {
            "run_dir": str(run_on),
            "n_events_total_valid": metrics_on["n_events_total_valid"],
            "n_events_after_only_valid": metrics_on["n_events_after_only_valid"],
            "n_events_after_filter": metrics_on["n_events_after_filter"],
            "n_blocks_after_filter": metrics_on["n_blocks_after_filter"],
            "max_block_len_after_filter": metrics_on["max_block_len_after_filter"],
            "skipped_before_events": metrics_on["skipped_before_events"],
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
        "exact_negative_fallback": exact_negative_fallback,
        "full_rows": {
            "run_off_row": run_off_row,
            "run_on_row": run_on_row,
            "pair_row": pair,
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
    return (Path(args.out_root).resolve() / f"higher_order_analysis_vaddr16_afteronly_{stamp}").resolve()


def task_groups_from_jobs(jobs: list[dict[str, Any]]) -> list[str]:
    groups = {parse_task_backend(job["task_id"])[0] for job in jobs}
    return sorted(groups, key=lambda item: int(item[1:]))


def group_range_tag(task_groups: list[str]) -> str:
    if not task_groups:
        return "empty"
    return f"{task_groups[0].lower()}_{task_groups[-1].lower()}"


def group_range_label(task_groups: list[str]) -> str:
    if not task_groups:
        return "empty"
    if len(task_groups) == 1:
        return task_groups[0]
    return f"{task_groups[0]}--{task_groups[-1]}"


def build_spec(
    args: argparse.Namespace,
    results_root: Path,
    out_dir: Path,
    jobs: list[dict[str, Any]],
    *,
    source_mode: str,
    job_manifest_path: Path | None,
) -> dict[str, Any]:
    spec = {
        "analysis_name": "higher_order_pattern_leakage_vaddr16_afteronly",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "results_root": str(results_root),
        "source_mode": source_mode,
        "output_dir": str(out_dir),
        "job_count_pairs": len(jobs),
        "job_count_runs": len(jobs) * 2,
        "symbol_source": "hash64(block16) from after-only paddrtrace write events after event-level filtering",
        "trajectory_unit": "vaddr16 block projected subsequence over kept after-only events",
        "metrics": {
            "L_nleq": "block-local non-adjacent equality-pair ratio using AMS F2 on combined (vaddr16, block16-symbol) keys; adjacent pairs excluded",
            "L_rep": (
                f"block-local repeated {args.kgram}-gram ratio via exact or approximate distinct counting on "
                "combined (vaddr16, local-kgram) keys"
            ),
            "L_per": f"block-local max lag-equality ratio over lags 2..{args.tau_max}",
            "L_motif": "block-local 0.5*(ABAB density + ABCABC density)",
        },
        "parameters": {
            "tau_max": args.tau_max,
            "kgram": args.kgram,
            "bitmap_bits": args.bitmap_bits,
            "ams_depth": args.ams_depth,
            "ams_width": args.ams_width,
            "jobs": args.jobs,
            "topk": args.topk,
            "rep_exact_mode": args.rep_exact_mode,
            "exact_negative_fallback_base_max": EXACT_NEGATIVE_FALLBACK_BASE_MAX,
            "exact_negative_fallback_max_events": EXACT_NEGATIVE_FALLBACK_MAX_EVENTS,
        },
        "filters": {
            "after_only": True,
            "block_key": "vaddr & ~0xF",
            "projection_policy": "keep only filtered-in after events; do not segment trajectories when filtered events are dropped",
            "defense_added_routine_substrings": list(DEFENSE_ROUTINE_SUBSTRINGS),
            "output_format_routine_substrings": list(OUTPUT_FORMAT_ROUTINE_SUBSTRINGS),
            "runtime_infra_routine_substrings": list(RUNTIME_INFRA_ROUTINE_SUBSTRINGS),
            "pointer_or_addr_like_written_value": {
                "u64_high_user_ptr_range": ["0x0000500000000000", "0x00007fffffffffff"],
                "u64_low_code_ptr_low32_range": ["0x00400000", "0x01000000"],
                "u32_addr_ranges": [["0x00400000", "0x01000000"], ["0x7f000000", "0x80000000"]],
            },
        },
        "rpat_definition": "R_pat^(m)=(L_base^(m)-L_def^(m))/L_base^(m), base=off, def=on",
    }
    if job_manifest_path is not None:
        spec["job_manifest"] = str(job_manifest_path)
    return spec


def load_jobs_from_manifest(job_manifest_path: Path) -> list[dict[str, Any]]:
    payload = json.loads(job_manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise TypeError(f"job manifest must be a JSON list: {job_manifest_path}")
    jobs: list[dict[str, Any]] = []
    for item in payload:
        if not isinstance(item, dict):
            raise TypeError(f"bad manifest entry in {job_manifest_path}: {item!r}")
        for key in ("task_id", "sample", "run_off", "run_on"):
            if key not in item:
                raise KeyError(f"manifest entry missing {key}: {item}")
        jobs.append(dict(item))
    return jobs


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
    results_root = Path(args.results_root).resolve()
    job_manifest_path = Path(args.job_manifest).resolve() if args.job_manifest else None
    out_dir = build_output_dir(args)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "pair_logs").mkdir(parents=True, exist_ok=True)
    scratch_root = Path(args.scratch_root).resolve() if args.scratch_root else (out_dir / ".scratch").resolve()
    scratch_root.mkdir(parents=True, exist_ok=True)

    if job_manifest_path is not None:
        jobs = load_jobs_from_manifest(job_manifest_path)
        source_mode = "job_manifest"
    else:
        jobs = build_jobs_from_results(results_root)
        source_mode = "results_root"
        if len(jobs) != 80:
            raise RuntimeError(f"expected 80 pairs from results tree, found {len(jobs)}")
    if not jobs:
        raise RuntimeError("no analysis jobs found")
    if args.limit_pairs > 0:
        jobs = jobs[: args.limit_pairs]
    for job in jobs:
        job["tau_max"] = args.tau_max
        job["kgram"] = args.kgram
        job["bitmap_bits"] = args.bitmap_bits
        job["ams_depth"] = args.ams_depth
        job["ams_width"] = args.ams_width
        job["topk"] = args.topk
        job["rep_exact_mode"] = args.rep_exact_mode
        job["scratch_root"] = str(scratch_root)

    task_groups = task_groups_from_jobs(jobs)
    group_tag = group_range_tag(task_groups)
    group_label = group_range_label(task_groups)

    spec = build_spec(
        args,
        results_root,
        out_dir,
        jobs,
        source_mode=source_mode,
        job_manifest_path=job_manifest_path,
    )
    write_json(out_dir / "spec.json", spec)
    write_json(out_dir / "job_manifest.json", jobs)

    spec_md = [
        "# Higher-Order Pattern Leakage Rescan",
        "",
        f"- Generated: `{spec['timestamp']}`",
        f"- Source results root: `{results_root}`",
        "- Scope: ciphertext trajectory proxy via block16 states on after-only write events",
        "- Trajectory unit: `vaddr16` block projected subsequence",
        f"- Pair count: `{len(jobs)}`",
        f"- Worker count: `{args.jobs}`",
        f"- `L_rep` exact mode: `{args.rep_exact_mode}`",
        f"- Scratch root: `{scratch_root}`",
        "",
        "## Event Filters",
        "- Keep only after-write records (`USED_BEFORE` removed before metric extraction).",
        "- Defense helper filter: exact substring match on the demangled routine field from `paddrtrace.ip.txt`.",
    ]
    for needle in DEFENSE_ROUTINE_SUBSTRINGS:
        spec_md.append(f"  - `{needle}`")
    spec_md.append("- Output-format helper filter: excludes libc/runner formatting routines that only materialize final textual output, not model ciphertext trajectory.")
    for needle in OUTPUT_FORMAT_ROUTINE_SUBSTRINGS:
        spec_md.append(f"  - `{needle}`")
    spec_md.append("- Runtime-infrastructure helper filter: excludes allocator, container growth, and bulk-copy support paths that do not represent model ciphertext states.")
    for needle in RUNTIME_INFRA_ROUTINE_SUBSTRINGS:
        spec_md.append(f"  - `{needle}`")
    spec_md += [
        "- Pointer/address-like value filter: excludes events whose written `block16` contains pointer-like or address-like integer values.",
        "",
        "## Trajectory Construction",
        "- Block key: `vaddr16 = vaddr & ~0xF`.",
        "- Per-block sequence: after-only events that survive filtering, projected into a block-local subsequence.",
        "- No segment cuts are introduced when filtered events are dropped.",
        "",
        "## Scope",
        f"- Task groups: `{group_label}`",
        f"- Unique groups: `{', '.join(task_groups)}`",
        "",
        "## Metrics",
        "- `L_nleq`: block-local non-adjacent equality-pair ratio.",
        f"- `L_rep`: block-local repeated {args.kgram}-gram ratio.",
        f"  Exact mode: `{args.rep_exact_mode}`.",
        "- `L_per`: block-local maximum lag-equality ratio for lags `2..tau_max`.",
        "- `L_motif`: block-local average of ABAB density and ABCABC density.",
        (
            "- Exact rep recomputation: "
            f"`{args.rep_exact_mode}` mode, with partitioned on-disk distinct counting under `{scratch_root}`."
        ),
        f"- Exact nleq fallback: if approximate `R_pat_nleq` is negative while its base leakage is <= `{EXACT_NEGATIVE_FALLBACK_BASE_MAX}` and the pair stays within `{EXACT_NEGATIVE_FALLBACK_MAX_EVENTS}` kept events per run, recompute that component exactly for the affected pair.",
        "",
        "## Reduction",
        "- `R_pat^(m) = (L_base^(m)-L_def^(m))/L_base^(m)` with base=`off`, def=`on`.",
        "- `R_pat_agg`: mean of available component reductions.",
    ]
    (out_dir / "SPEC.md").write_text("\n".join(spec_md) + "\n", encoding="utf-8")

    all_jobs = list(jobs)
    progress_log = out_dir / "progress.log"
    pair_jsonl = out_dir / "pair_results.jsonl"
    resumed_run_rows, resumed_pair_rows, resumed_payloads = load_completed_pair_rows(out_dir / "pair_logs")
    resumed_keys = set(resumed_payloads)
    jobs = [job for job in all_jobs if pair_job_key(str(job["task_id"]), str(job["sample"])) not in resumed_keys]
    stale_pair_logs = len(list((out_dir / "pair_logs").glob("*.json"))) - len(resumed_payloads)

    started_at = time.time()
    last_completed: dict[str, Any] | None = None
    existing_progress_path = out_dir / "progress.json"
    if existing_progress_path.exists():
        try:
            existing_progress = json.loads(existing_progress_path.read_text(encoding="utf-8"))
            old_started = existing_progress.get("started_at_epoch")
            if old_started is not None:
                started_at = float(old_started)
            old_last = existing_progress.get("last_completed")
            if isinstance(old_last, dict):
                last_completed = old_last
        except Exception:
            pass

    with progress_log.open("a", encoding="utf-8") as handle:
        if resumed_pair_rows:
            handle.write(
                f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] resume total_pairs={len(all_jobs)} reused={len(resumed_pair_rows)} "
                f"pending={len(jobs)} jobs={args.jobs} stale_pair_logs={stale_pair_logs} out_dir={out_dir}\n"
            )
        else:
            handle.write(
                f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] start total_pairs={len(all_jobs)} jobs={args.jobs} "
                f"stale_pair_logs={stale_pair_logs} out_dir={out_dir}\n"
            )

    per_run_rows: list[dict[str, Any]] = list(resumed_run_rows)
    per_pair_rows: list[dict[str, Any]] = list(resumed_pair_rows)
    failures: list[dict[str, Any]] = []
    write_json(
        out_dir / "progress.json",
        progress_payload(
            total=len(all_jobs),
            completed=len(per_pair_rows),
            started_at=started_at,
            current_pair=last_completed,
            failures=failures,
        ),
    )

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
            "off_blocks": run_off_row["n_blocks_after_filter"],
            "on_blocks": run_on_row["n_blocks_after_filter"],
            "off_skip_before": run_off_row["skipped_before_events"],
            "on_skip_before": run_on_row["skipped_before_events"],
            "off_skip_defense": run_off_row["skipped_defense_events"],
            "on_skip_defense": run_on_row["skipped_defense_events"],
            "off_skip_ptr": run_off_row["skipped_ptr_addr_events"],
            "on_skip_ptr": run_on_row["skipped_ptr_addr_events"],
        }
        payload = progress_payload(
            total=len(all_jobs),
            completed=completed,
            started_at=started_at,
            current_pair=last_completed,
            failures=failures,
        )
        write_json(out_dir / "progress.json", payload)
        with progress_log.open("a", encoding="utf-8") as handle:
            handle.write(
                f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] done {completed}/{len(all_jobs)} "
                f"{pair['task_id']}/{pair['sample']} backend={pair['backend']} "
                f"off_keep={run_off_row['n_events_after_filter']} on_keep={run_on_row['n_events_after_filter']} "
                f"off_blocks={run_off_row['n_blocks_after_filter']} on_blocks={run_on_row['n_blocks_after_filter']} "
                f"off_skip_before={run_off_row['skipped_before_events']} on_skip_before={run_on_row['skipped_before_events']} "
                f"off_skip_def={run_off_row['skipped_defense_events']} on_skip_def={run_on_row['skipped_defense_events']} "
                f"off_skip_ptr={run_off_row['skipped_ptr_addr_events']} on_skip_ptr={run_on_row['skipped_ptr_addr_events']} "
                f"elapsed_pair={pair_log['elapsed_sec']:.3f}s\n"
            )

    if args.jobs <= 1:
        for job in jobs:
            try:
                on_pair_done(*analyze_pair_job(job))
            except Exception as exc:
                failures.append({"task_id": job["task_id"], "sample": job["sample"], "error": repr(exc)})
                write_json(
                    out_dir / "progress.json",
                    progress_payload(
                        total=len(all_jobs),
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
                    failures.append({"task_id": job["task_id"], "sample": job["sample"], "error": repr(exc)})
                    write_json(
                        out_dir / "progress.json",
                        progress_payload(
                            total=len(all_jobs),
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
        "n_events_after_only_valid",
        "n_events_after_filter",
        "n_blocks_after_filter",
        "max_block_len_after_filter",
        "skipped_before_events",
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
    for tg in task_groups:
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
    summary_json_name = f"summary_{group_tag}_backend.json"
    summary_csv_name = f"summary_{group_tag}_backend.csv"
    write_json(out_dir / summary_json_name, summary_rows)
    with (out_dir / summary_csv_name).open("w", encoding="utf-8", newline="") as handle:
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
            "off_skip_before_mean": mean([float(r["skipped_before_events"]) for r in off_runs]),
            "on_skip_before_mean": mean([float(r["skipped_before_events"]) for r in on_runs]),
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
        f"# L Metrics Change Table ({len(rows40)} tasks)",
        "",
        "- Policy: vaddr16-grouped after-only projected subsequences, with event-level exclusion of defense-added helper routines and pointer/address-like written values.",
        "- No per-block segment cuts are introduced when filtered events are dropped.",
        f"- Scope: {group_label}; {len(rows40)} task entries currently available (mean over the matched samples that have completed off/on pairs).",
        "",
        "| Task | Group | Backend | L_nleq_off | L_nleq_on | Delta_nleq | L_rep_off | L_rep_on | Delta_rep | L_per_off | L_per_on | Delta_per | L_motif_off | L_motif_on | Delta_motif | off_skip_before | on_skip_before | off_skip_def | on_skip_def | off_skip_ptr | on_skip_ptr |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows40:
        md_lines.append(
            f"| {row['task_id']} | {row['task_group_name']} | {row['backend']} | "
            f"{fmt_metric(row['L_nleq_off'])} | {fmt_metric(row['L_nleq_on'])} | {fmt_metric(row['Delta_L_nleq_on_minus_off'])} | "
            f"{fmt_metric(row['L_rep_off'])} | {fmt_metric(row['L_rep_on'])} | {fmt_metric(row['Delta_L_rep_on_minus_off'])} | "
            f"{fmt_metric(row['L_per_off'])} | {fmt_metric(row['L_per_on'])} | {fmt_metric(row['Delta_L_per_on_minus_off'])} | "
            f"{fmt_metric(row['L_motif_off'])} | {fmt_metric(row['L_motif_on'])} | {fmt_metric(row['Delta_L_motif_on_minus_off'])} | "
            f"{fmt_metric(row['off_skip_before_mean'])} | {fmt_metric(row['on_skip_before_mean'])} | "
            f"{fmt_metric(row['off_skip_defense_mean'])} | {fmt_metric(row['on_skip_defense_mean'])} | "
            f"{fmt_metric(row['off_skip_ptr_mean'])} | {fmt_metric(row['on_skip_ptr_mean'])} |"
        )
    l_table_name = f"table_l_metrics_changes_vaddr16_afteronly_{len(rows40)}tasks_{group_tag}.md"
    l_table_text = "\n".join(md_lines) + "\n"
    (out_dir / l_table_name).write_text(l_table_text, encoding="utf-8")
    (out_dir / f"table_l_metrics_changes_excl_defense_added_{len(rows40)}tasks.md").write_text(
        l_table_text,
        encoding="utf-8",
    )
    if group_tag == "t01_t10" and len(rows40) == 40:
        (out_dir / "table_l_metrics_changes_excl_defense_added_40tasks.md").write_text(l_table_text, encoding="utf-8")

    tex_lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        f"\\caption{{Higher-order pattern leakage reduction ratio $R_{{\\mathrm{{pat}}}}^{{\\mathrm{{agg}}}}$ over {group_label} across four execution chains, computed on vaddr16-grouped after-only projected subsequences after excluding defense-helper routine writes and pointer/address-like written values.}}",
        f"\\label{{tab:rpat_agg_{group_tag}}}",
        "\\resizebox{\\columnwidth}{!}{%",
        "\\begin{tabular}{lccccc}",
        "\\toprule",
        "Group & Task group & MC & IC & TV & TA \\\\",
        "\\midrule",
    ]
    for tg in task_groups:
        vals = []
        for backend in ("MC", "IC", "TV", "TA"):
            rows = by_group_backend.get((tg, backend), [])
            vals.append(mean([x.get("R_pat_agg") for x in rows]))
        tex_lines.append(
            f"{tg} & {TASK_GROUP_NAME[tg]} & {fmt_pct(vals[0])} & {fmt_pct(vals[1])} & {fmt_pct(vals[2])} & {fmt_pct(vals[3])} \\\\"
        )
    tex_lines += ["\\bottomrule", "\\end{tabular}%", "}", "\\end{table}"]
    tex_name = f"table_rpat_agg_{group_tag}.tex"
    (out_dir / tex_name).write_text("\n".join(tex_lines) + "\n", encoding="utf-8")
    if group_tag == "t01_t10":
        (out_dir / "table_rpat_agg_t01_t10.tex").write_text("\n".join(tex_lines) + "\n", encoding="utf-8")

    defense_rule_summary: Counter[str] = Counter()
    defense_routine_summary: Counter[str] = Counter()
    for row in per_run_rows:
        for item in row.get("defense_rule_hits", []):
            defense_rule_summary[str(item["key"])] += int(item["count"])
        for item in row.get("defense_routine_hits", []):
            defense_routine_summary[str(item["key"])] += int(item["count"])

    audit = {
        "analysis": "higher_order_pattern_leakage_vaddr16_afteronly",
        "generated_at": spec["timestamp"],
        "results_root": str(results_root),
        "output_dir": str(out_dir),
        "scratch_root": str(scratch_root),
        "jobs_requested": args.jobs,
        "pairs_total": len(all_jobs),
        "pairs_completed": len(per_pair_rows),
        "pairs_failed": len(failures),
        "rep_exact_mode": args.rep_exact_mode,
        "defense_routine_substrings": list(DEFENSE_ROUTINE_SUBSTRINGS),
        "output_format_routine_substrings": list(OUTPUT_FORMAT_ROUTINE_SUBSTRINGS),
        "runtime_infra_routine_substrings": list(RUNTIME_INFRA_ROUTINE_SUBSTRINGS),
        "exact_negative_fallback_base_max": EXACT_NEGATIVE_FALLBACK_BASE_MAX,
        "exact_negative_fallback_max_events": EXACT_NEGATIVE_FALLBACK_MAX_EVENTS,
        "defense_rule_hits_total": counter_to_rows(defense_rule_summary, 32),
        "defense_routine_hits_total": counter_to_rows(defense_routine_summary, 32),
        "outputs": [
            str(out_dir / "per_run_metrics.csv"),
            str(out_dir / "per_pair_rpat.csv"),
            str(out_dir / l_table_name),
            str(out_dir / tex_name),
            str(out_dir / "progress.log"),
        ],
    }
    write_json(out_dir / "filter_audit.json", audit)

    final_progress = progress_payload(
        total=len(all_jobs),
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
