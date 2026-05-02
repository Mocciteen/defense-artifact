from __future__ import annotations

import math
import struct
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator


HDR_V4 = struct.Struct("<8sIIQ")
REC_V4 = struct.Struct("<QIIIIQQQQQII")
MAGIC = b"PADDRTRC"
VER = 4
PADDR16_VALID = 1 << 1
USED_BEFORE = 1 << 2

DEFENSE_ROUTINE_SUBSTRINGS = (
    "patchedRelu",
    "reluResultBitsFromInputBits",
    "selectPatchedSeedBits",
    "applyInputZeroCheckerboardDitherOrExit",
    "inputzerodither",
    "tvm_relu_low12_f32",
    "tvm_relu6_low12_f32",
)

OUTPUT_FORMAT_ROUTINE_SUBSTRINGS = ("__printf_", "__mpn_", "hack_digit")
RUNTIME_INFRA_ROUTINE_SUBSTRINGS = ("__libc_malloc", "_int_malloc", "__memmove_", "_M_realloc_insert")


@dataclass
class TraceRecord:
    block_addr: int
    ip: int
    flags: int
    block16: bytes


class LinearDistinct:
    def __init__(self, bits: int):
        if bits <= 0:
            raise ValueError("bits must be positive")
        self.bits = int(bits)
        self.bytes = bytearray((self.bits + 7) // 8)

    def add(self, value: int) -> None:
        idx = int(value) % self.bits
        self.bytes[idx >> 3] |= 1 << (idx & 7)

    def estimate(self) -> float:
        zero = 0
        for byte in self.bytes:
            zero += 8 - int(byte).bit_count()
        extra = len(self.bytes) * 8 - self.bits
        zero = max(0, zero - extra)
        if zero <= 0:
            return float(self.bits)
        return -float(self.bits) * math.log(float(zero) / float(self.bits))


def splitmix64(value: int) -> int:
    value = (int(value) + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    return (value ^ (value >> 31)) & 0xFFFFFFFFFFFFFFFF


def combine_u64(first: int, second: int) -> int:
    return splitmix64(first ^ ((second << 1) & 0xFFFFFFFFFFFFFFFF))


def sym_from_block16(block16: bytes) -> int:
    first, second = struct.unpack("<QQ", block16)
    return combine_u64(first, second)


def parse_ipmap(ipmap_path: Path) -> dict[int, tuple[str, str, str]]:
    out: dict[int, tuple[str, str, str]] = {}
    for raw in ipmap_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not raw or raw.startswith("#"):
            continue
        parts = raw.split("\t")
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
    for needle in DEFENSE_ROUTINE_SUBSTRINGS + OUTPUT_FORMAT_ROUTINE_SUBSTRINGS + RUNTIME_INFRA_ROUTINE_SUBSTRINGS:
        if needle in routine:
            return needle
    return None


def is_pointer_or_addr_like(block16: bytes) -> bool:
    q0, q1 = struct.unpack("<QQ", block16)
    u0, u1, u2, u3 = struct.unpack("<IIII", block16)

    def high_user_ptr64(value: int) -> bool:
        return 0x0000500000000000 <= value <= 0x00007FFFFFFFFFFF

    def low_code_ptr64(value: int) -> bool:
        return (value >> 32) == 0 and 0x00400000 <= (value & 0xFFFFFFFF) <= 0x01000000

    def addr32(value: int) -> bool:
        return (0x00400000 <= value <= 0x01000000) or (0x7F000000 <= value <= 0x80000000)

    return (
        high_user_ptr64(q0)
        or high_user_ptr64(q1)
        or low_code_ptr64(q0)
        or low_code_ptr64(q1)
        or addr32(u0)
        or addr32(u1)
        or addr32(u2)
        or addr32(u3)
    )


def iter_trace_records(trace_bin: Path) -> Iterator[TraceRecord]:
    with trace_bin.open("rb") as handle:
        header = handle.read(HDR_V4.size)
        if len(header) != HDR_V4.size:
            raise RuntimeError(f"trace too small: {trace_bin}")
        magic, ver, _reserved, _page = HDR_V4.unpack(header)
        if magic != MAGIC or ver != VER:
            raise RuntimeError(f"bad paddrtrace header: {trace_bin}")

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
                _compat_addr,
                _paddr,
                paddr16,
                ip,
                _instr_id,
                flags,
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
            if not (int(flags) & PADDR16_VALID):
                continue
            yield TraceRecord(int(paddr16), int(ip), int(flags), block16)


def counter_rows(counter: Counter[str], topk: int) -> list[str]:
    return [f"{key}:{count}" for key, count in counter.most_common(topk)]


def analyze_trace(
    trace_bin: Path,
    ipmap_path: Path,
    *,
    tau_max: int = 64,
    kgram: int = 4,
    bitmap_bits: int = 1 << 20,
    topk: int = 12,
    filter_defense: bool = True,
    filter_pointers: bool = True,
    rep_mode: str = "bitmap",
) -> dict[str, Any]:
    ipmap = parse_ipmap(ipmap_path)
    block_lengths: dict[int, int] = defaultdict(int)
    last_sym_by_block: dict[int, int] = {}
    lag_history_by_block: dict[int, deque[int]] = {}
    kgram_history_by_block: dict[int, deque[int]] = {}
    kgram_distinct = set() if rep_mode == "exact" else None
    kgram_bitmap = LinearDistinct(bitmap_bits)
    lag_equal = Counter()
    defense_rules = Counter()
    defense_routines = Counter()

    n_total = n_after = n_kept = 0
    skipped_before = skipped_defense = skipped_ptr = 0
    adj_equal = total_kgrams = total_m4 = total_m6 = abab = abcabc = 0

    for record in iter_trace_records(trace_bin):
        n_total += 1
        if record.flags & USED_BEFORE:
            skipped_before += 1
            continue
        n_after += 1
        routine = ipmap.get(record.ip, ("<unknown>", "<unknown>", "<unknown>"))[0]
        rule = defense_rule_for_routine(routine)
        if filter_defense and rule is not None:
            skipped_defense += 1
            defense_rules[rule] += 1
            defense_routines[routine] += 1
            continue
        if filter_pointers and is_pointer_or_addr_like(record.block16):
            skipped_ptr += 1
            continue

        block_key = record.block_addr & ~0xF
        sym = sym_from_block16(record.block16)
        n_kept += 1
        block_lengths[block_key] += 1

        if last_sym_by_block.get(block_key) == sym:
            adj_equal += 1
        last_sym_by_block[block_key] = sym

        lag_history = lag_history_by_block.setdefault(block_key, deque(maxlen=max(1, tau_max)))
        for offset, old_sym in enumerate(reversed(lag_history), start=1):
            if old_sym == sym:
                lag_equal[offset] += 1
        motif_tail = list(lag_history)[-5:]
        if len(motif_tail) >= 3:
            a, b, c = motif_tail[-3:]
            if a == sym and b != sym:
                abab += 1
            if len(motif_tail) >= 5:
                a, b, c, d, e = motif_tail[-5:]
                if a == d and b == e and c == sym:
                    abcabc += 1
        total_m4 += int(len(lag_history) >= 3)
        total_m6 += int(len(lag_history) >= 5)
        lag_history.append(sym)

        kgram_history = kgram_history_by_block.setdefault(block_key, deque(maxlen=max(1, kgram - 1)))
        if len(kgram_history) >= kgram - 1:
            words = (block_key, *tuple(kgram_history), sym)
            hashed = 0
            for word in words:
                hashed = combine_u64(hashed, int(word))
            kgram_bitmap.add(hashed)
            if kgram_distinct is not None:
                kgram_distinct.add(words)
            total_kgrams += 1
        kgram_history.append(sym)

    denom_nonadj = sum((n * (n - 1) / 2.0) - (n - 1) for n in block_lengths.values() if n >= 2)
    eq_pairs_est = max(sum(count for tau, count in lag_equal.items() if tau >= 2), 0.0)
    l_nleq = max(0.0, min(1.0, eq_pairs_est / denom_nonadj)) if denom_nonadj > 0 else 0.0

    if total_kgrams > 0:
        distinct_est = float(len(kgram_distinct)) if kgram_distinct is not None else min(kgram_bitmap.estimate(), float(total_kgrams))
        l_rep = max(0.0, min(1.0, 1.0 - distinct_est / float(total_kgrams)))
    else:
        distinct_est = 0.0
        l_rep = 0.0

    l_per = 0.0
    for tau in range(2, tau_max + 1):
        denom_tau = sum(max(n - tau, 0) for n in block_lengths.values())
        if denom_tau > 0:
            l_per = max(l_per, float(lag_equal[tau]) / float(denom_tau))

    m4 = float(abab) / float(total_m4) if total_m4 > 0 else 0.0
    m6 = float(abcabc) / float(total_m6) if total_m6 > 0 else 0.0
    l_motif = max(0.0, min(1.0, 0.5 * (m4 + m6)))

    return {
        "n_events_total_valid": int(n_total),
        "n_events_after_only_valid": int(n_after),
        "n_events_after_filter": int(n_kept),
        "n_blocks_after_filter": int(len(block_lengths)),
        "max_block_len_after_filter": int(max(block_lengths.values()) if block_lengths else 0),
        "skipped_before_events": int(skipped_before),
        "skipped_defense_events": int(skipped_defense),
        "skipped_ptr_addr_events": int(skipped_ptr),
        "defense_rule_hits": ";".join(counter_rows(defense_rules, topk)),
        "defense_routine_hits": ";".join(counter_rows(defense_routines, topk)),
        "adj_equal": int(adj_equal),
        "L_nleq": float(l_nleq),
        "L_rep": float(l_rep),
        "rep_distinct_estimate": float(distinct_est),
        "L_per": float(l_per),
        "L_motif": float(l_motif),
        "total_kgrams": int(total_kgrams),
        "total_m4_windows": int(total_m4),
        "total_m6_windows": int(total_m6),
        "abab_count": int(abab),
        "abcabc_count": int(abcabc),
    }


def ratio(base: float, defended: float) -> float | None:
    if base <= 0.0:
        return None
    return float(defended) / float(base)


def pair_metrics(metrics_off: dict[str, Any], metrics_on: dict[str, Any]) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for key in ("L_nleq", "L_rep", "L_per", "L_motif"):
        off = float(metrics_off[key])
        on = float(metrics_on[key])
        out[f"{key}_off"] = off
        out[f"{key}_on"] = on
        out[f"{key}_delta"] = on - off
        out[f"R_{key[2:]}"] = ratio(off, on)
    return out
