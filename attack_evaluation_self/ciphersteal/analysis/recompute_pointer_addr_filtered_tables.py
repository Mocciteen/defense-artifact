#!/usr/bin/env python3
from __future__ import annotations

import concurrent.futures
import csv
import json
import math
import struct
from collections import defaultdict, deque
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np

HDR_V4 = struct.Struct("<8sIIQ")
REC_V4 = struct.Struct("<QIIIIQQQQQII")
MAGIC = b"PADDRTRC"
VER = 4

TAU_MAX = 64
KGRAM = 4
BITMAP_BITS = 1 << 20
AMS_DEPTH = 5
AMS_WIDTH = 1 << 18

FORMAL_DIR = Path("/path/to/high_leakage_workspace/higher_order_analysis_formal_20260422")

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


def splitmix64(x: int) -> int:
    x = (x + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    return x ^ (x >> 31)


def sym_from_block16(block16: bytes) -> int:
    lo = int.from_bytes(block16[:8], "little", signed=False)
    hi = int.from_bytes(block16[8:], "little", signed=False)
    return splitmix64(lo ^ splitmix64(hi))


def parse_task_backend(task_id: str) -> tuple[str, str]:
    prefix = task_id[:2]
    num = int(task_id[2:])
    return f"T{num:02d}", prefix


def ratio(base: float, defended: float) -> float | None:
    if base <= 0.0:
        return None
    return (base - defended) / base


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
            s = 1 if (splitmix64(sym ^ self.seed_sgn[r]) & 1) == 0 else -1
            self.rows[r][idx] += s

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


def is_pointer_or_addr_like(block16: bytes) -> bool:
    a, b = struct.unpack("<QQ", block16)
    u0, u1, u2, u3 = struct.unpack("<IIII", block16)

    def is_user_ptr64(x: int) -> bool:
        return 0x0000500000000000 <= x <= 0x00007FFFFFFFFFFF

    def is_addr32(x: int) -> bool:
        # Typical 32-bit code/text and libc-like low32 address tails.
        return (0x00400000 <= x <= 0x01000000) or (0x7F000000 <= x <= 0x80000000)

    return is_user_ptr64(a) or is_user_ptr64(b) or is_addr32(u0) or is_addr32(u1) or is_addr32(u2) or is_addr32(u3)


def analyze_trace_bin_filtered(trace_bin: Path) -> dict[str, Any]:
    ams = AMSF2(depth=AMS_DEPTH, width=AMS_WIDTH)
    kgram_dist = LinearDistinct(bitmap_bits=BITMAP_BITS, seed=0xDEADBEEF12345678)
    n = 0
    adj_equal = 0
    lag_equal = [0] * (TAU_MAX + 1)
    lag_hist = deque(maxlen=TAU_MAX)
    k_hist = deque(maxlen=max(KGRAM, 6))
    abab = 0
    abcabc = 0
    total_kgrams = 0
    total_m4 = 0
    total_m6 = 0
    skipped_ptr_addr = 0
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
                _ip,
                _instr_id,
                _flags,
                stack_depth,
            ) = REC_V4.unpack(rec)
            handle.seek(int(size), 1)
            block16 = handle.read(16)
            if len(block16) != 16:
                raise RuntimeError(f"truncated block16: {trace_bin}")
            if stack_depth:
                handle.seek(int(stack_depth) * 8, 1)
            if int(size_read) != int(size) or int(size) <= 0:
                continue
            if is_pointer_or_addr_like(block16):
                skipped_ptr_addr += 1
                continue

            sym = sym_from_block16(block16)
            n += 1
            ams.update(sym)
            if prev_sym is not None and prev_sym == sym:
                adj_equal += 1

            lag_hist.append(sym)
            for tau in range(2, min(TAU_MAX, len(lag_hist) - 1) + 1):
                if lag_hist[-1] == lag_hist[-1 - tau]:
                    lag_equal[tau] += 1

            k_hist.append(sym)
            if len(k_hist) >= KGRAM:
                h = 0x9E3779B97F4A7C15
                for i in range(KGRAM):
                    h = splitmix64(h ^ k_hist[-KGRAM + i])
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

    if n <= 1:
        return {
            "n_events_after_filter": int(n),
            "skipped_ptr_addr_events": int(skipped_ptr_addr),
            "L_nleq": 0.0,
            "L_rep": 0.0,
            "L_per": 0.0,
            "L_motif": 0.0,
        }

    f2 = ams.estimate_f2()
    eq_pairs_est = max((f2 - float(n)) / 2.0, 0.0)
    nonlocal_eq_est = max(eq_pairs_est - float(adj_equal), 0.0)
    denom_nonadj = (float(n) * float(n - 1) / 2.0) - float(n - 1)
    l_nleq = nonlocal_eq_est / denom_nonadj if denom_nonadj > 0 else 0.0
    l_nleq = max(0.0, min(1.0, l_nleq))

    if total_kgrams > 0:
        distinct_est = min(kgram_dist.estimate(), float(total_kgrams))
        l_rep = max(0.0, min(1.0, 1.0 - distinct_est / float(total_kgrams)))
    else:
        l_rep = 0.0

    l_per = 0.0
    for tau in range(2, TAU_MAX + 1):
        denom = n - tau
        if denom > 0:
            ratio_tau = float(lag_equal[tau]) / float(denom)
            if ratio_tau > l_per:
                l_per = ratio_tau
    l_per = max(0.0, min(1.0, l_per))

    m4 = float(abab) / float(total_m4) if total_m4 > 0 else 0.0
    m6 = float(abcabc) / float(total_m6) if total_m6 > 0 else 0.0
    l_motif = max(0.0, min(1.0, 0.5 * (m4 + m6)))

    return {
        "n_events_after_filter": int(n),
        "skipped_ptr_addr_events": int(skipped_ptr_addr),
        "L_nleq": float(l_nleq),
        "L_rep": float(l_rep),
        "L_per": float(l_per),
        "L_motif": float(l_motif),
    }


def build_jobs_from_formal() -> list[dict[str, str]]:
    run_rows = list(csv.DictReader((FORMAL_DIR / "per_run_metrics.csv").open()))
    idx = {(r["task_id"], r["sample"], r["mode"]): r["run_dir"] for r in run_rows}
    keys = sorted({(r["task_id"], r["sample"]) for r in run_rows})
    jobs: list[dict[str, str]] = []
    for task_id, sample in keys:
        off = idx.get((task_id, sample, "off"))
        on = idx.get((task_id, sample, "on"))
        if not off or not on:
            continue
        jobs.append({"task_id": task_id, "sample": sample, "run_off": off, "run_on": on})
    return jobs


def analyze_job(job: dict[str, str]) -> dict[str, Any]:
    task_id = job["task_id"]
    sample = job["sample"]
    task_group, backend = parse_task_backend(task_id)
    m_off = analyze_trace_bin_filtered(Path(job["run_off"]) / "paddrtrace.bin")
    m_on = analyze_trace_bin_filtered(Path(job["run_on"]) / "paddrtrace.bin")
    out = {
        "task_id": task_id,
        "task_group": task_group,
        "task_group_name": TASK_GROUP_NAME[task_group],
        "backend": backend,
        "sample": sample,
        "run_off": job["run_off"],
        "run_on": job["run_on"],
        "L_nleq_base": m_off["L_nleq"],
        "L_nleq_def": m_on["L_nleq"],
        "L_rep_base": m_off["L_rep"],
        "L_rep_def": m_on["L_rep"],
        "L_per_base": m_off["L_per"],
        "L_per_def": m_on["L_per"],
        "L_motif_base": m_off["L_motif"],
        "L_motif_def": m_on["L_motif"],
        "off_n_after_filter": m_off["n_events_after_filter"],
        "on_n_after_filter": m_on["n_events_after_filter"],
        "off_skipped_ptr_addr": m_off["skipped_ptr_addr_events"],
        "on_skipped_ptr_addr": m_on["skipped_ptr_addr_events"],
    }
    out["R_pat_nleq"] = ratio(out["L_nleq_base"], out["L_nleq_def"])
    out["R_pat_rep"] = ratio(out["L_rep_base"], out["L_rep_def"])
    out["R_pat_per"] = ratio(out["L_per_base"], out["L_per_def"])
    out["R_pat_motif"] = ratio(out["L_motif_base"], out["L_motif_def"])
    return out


def mean(vals: list[float | None]) -> float | None:
    xs = [v for v in vals if v is not None]
    return (sum(xs) / len(xs)) if xs else None


def fmt_pct(v: float | None) -> str:
    return "--" if v is None else f"{v * 100:.2f}\\%"


def main() -> int:
    jobs = build_jobs_from_formal()
    with concurrent.futures.ProcessPoolExecutor(max_workers=24) as ex:
        per_pair = list(ex.map(analyze_job, jobs))

    # Defense-added motif exclusion list (task_id, sample).
    neg = json.loads((FORMAL_DIR / "negative_after_excluding_defense_added.json").read_text())
    excluded = {(x[0], x[1]) for x in neg["excluded_defense_added_keys"]}

    # Sample-level filtered aggregate.
    for r in per_pair:
        vals = [r["R_pat_nleq"], r["R_pat_rep"], r["R_pat_per"]]
        if (r["task_id"], r["sample"]) not in excluded:
            vals.append(r["R_pat_motif"])
        vals = [v for v in vals if v is not None]
        r["R_pat_agg_filtered_ptraddr"] = sum(vals) / len(vals) if vals else None

    # 40-task table: sample-mean for each task_id (2 samples).
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in per_pair:
        by_task[r["task_id"]].append(r)
    rows40: list[dict[str, Any]] = []
    for task_id in sorted(by_task.keys()):
        rs = by_task[task_id]
        tg = rs[0]["task_group"]
        be = rs[0]["backend"]
        motif_rs = [x for x in rs if (x["task_id"], x["sample"]) not in excluded]
        row = {
            "task_id": task_id,
            "task_group": tg,
            "task_group_name": TASK_GROUP_NAME[tg],
            "backend": be,
            "num_samples_total": len(rs),
            "num_samples_motif_excluded": len(rs) - len(motif_rs),
            "L_nleq_off": mean([x["L_nleq_base"] for x in rs]),
            "L_nleq_on_excl": mean([x["L_nleq_def"] for x in rs]),
            "L_rep_off": mean([x["L_rep_base"] for x in rs]),
            "L_rep_on_excl": mean([x["L_rep_def"] for x in rs]),
            "L_per_off": mean([x["L_per_base"] for x in rs]),
            "L_per_on_excl": mean([x["L_per_def"] for x in rs]),
            "L_motif_off": mean([x["L_motif_base"] for x in motif_rs]),
            "L_motif_on_excl": mean([x["L_motif_def"] for x in motif_rs]),
            "off_skipped_ptr_addr_mean": mean([float(x["off_skipped_ptr_addr"]) for x in rs]),
            "on_skipped_ptr_addr_mean": mean([float(x["on_skipped_ptr_addr"]) for x in rs]),
        }
        for m in ("nleq", "rep", "per", "motif"):
            a = row[f"L_{m}_off"]
            b = row[f"L_{m}_on_excl"]
            row[f"Delta_L_{m}_on_minus_off"] = None if (a is None or b is None) else (b - a)
        rows40.append(row)

    # Write/overwrite requested MD table.
    md_lines = []
    md_lines.append("# L Metrics Change Table (40 tasks)")
    md_lines.append("")
    md_lines.append("- Policy: (1) pointer/address-like write values excluded from both off/on traces; (2) defense-added motif samples excluded in motif aggregation.")
    md_lines.append("- Scope: T01--T10 x MC/IC/TV/TA = 40 task entries (sample-level mean).")
    md_lines.append("")
    md_lines.append("| Task | Group | Backend | n_motif_excl | L_nleq_off | L_nleq_on | Δnleq | L_rep_off | L_rep_on | Δrep | L_per_off | L_per_on | Δper | L_motif_off | L_motif_on | Δmotif |")
    md_lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")

    def fmt(v: float | None) -> str:
        if v is None:
            return "--"
        return f"{v:.6e}" if abs(v) < 1e-4 else f"{v:.6f}"

    for r in rows40:
        md_lines.append(
            f"| {r['task_id']} | {r['task_group_name']} | {r['backend']} | {r['num_samples_motif_excluded']} | "
            f"{fmt(r['L_nleq_off'])} | {fmt(r['L_nleq_on_excl'])} | {fmt(r['Delta_L_nleq_on_minus_off'])} | "
            f"{fmt(r['L_rep_off'])} | {fmt(r['L_rep_on_excl'])} | {fmt(r['Delta_L_rep_on_minus_off'])} | "
            f"{fmt(r['L_per_off'])} | {fmt(r['L_per_on_excl'])} | {fmt(r['Delta_L_per_on_minus_off'])} | "
            f"{fmt(r['L_motif_off'])} | {fmt(r['L_motif_on_excl'])} | {fmt(r['Delta_L_motif_on_minus_off'])} |"
        )
    (FORMAL_DIR / "table_l_metrics_changes_excl_defense_added_40tasks.md").write_text(
        "\n".join(md_lines) + "\n", encoding="utf-8"
    )

    # Overwrite requested tex table with new agg (pointer/address filtered + defense-added motif exclusion).
    by_tg_be: dict[tuple[str, str], list[float]] = defaultdict(list)
    for r in per_pair:
        v = r["R_pat_agg_filtered_ptraddr"]
        if v is not None:
            by_tg_be[(r["task_group"], r["backend"])].append(float(v))
    tex_lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\caption{Higher-order pattern leakage reduction ratio $R_{\\mathrm{pat}}^{\\mathrm{agg}}$ over T01--T10 across four execution chains (pointer/address-like writes excluded; defense-added motif samples excluded).}",
        "\\label{tab:rpat_agg_t01_t10}",
        "\\resizebox{\\columnwidth}{!}{%",
        "\\begin{tabular}{lccccc}",
        "\\toprule",
        "Group & Task group & MC & IC & TV & TA \\\\",
        "\\midrule",
    ]
    for tg in [f"T{i:02d}" for i in range(1, 11)]:
        cells = []
        for be in ("MC", "IC", "TV", "TA"):
            vals = by_tg_be.get((tg, be), [])
            cells.append(None if not vals else (sum(vals) / len(vals)))
        tex_lines.append(
            f"{tg} & {TASK_GROUP_NAME[tg]} & {fmt_pct(cells[0])} & {fmt_pct(cells[1])} & {fmt_pct(cells[2])} & {fmt_pct(cells[3])} \\\\"
        )
    tex_lines += ["\\bottomrule", "\\end{tabular}%", "}", "\\end{table}"]
    (FORMAL_DIR / "table_rpat_agg_t01_t10.tex").write_text("\n".join(tex_lines) + "\n", encoding="utf-8")

    # Keep audit artifact for this recomputation.
    audit = {
        "analysis": "pointer_address_filtered_recompute",
        "filters": {
            "pointer_addr_like_value_exclusion": {
                "u64_user_ptr_range": ["0x0000500000000000", "0x00007fffffffffff"],
                "u32_addr_ranges": [["0x00400000", "0x01000000"], ["0x7f000000", "0x80000000"]],
            },
            "defense_added_motif_exclusion_count": len(excluded),
        },
        "jobs": len(jobs),
        "outputs": [
            str(FORMAL_DIR / "table_l_metrics_changes_excl_defense_added_40tasks.md"),
            str(FORMAL_DIR / "table_rpat_agg_t01_t10.tex"),
        ],
    }
    (FORMAL_DIR / "pointer_addr_filter_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(FORMAL_DIR / "table_l_metrics_changes_excl_defense_added_40tasks.md")
    print(FORMAL_DIR / "table_rpat_agg_t01_t10.tex")
    print(FORMAL_DIR / "pointer_addr_filter_audit.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

