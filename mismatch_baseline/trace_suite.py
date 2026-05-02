from __future__ import annotations

from collections.abc import Sequence
from math import prod

import torch


DATASET_RATIO_RANGES = {
    "generic": (0.0, 1.0),
    "mnist": (0.231555, 0.542835),
    "cifar": (0.835350, 0.889512),
    "chest": (0.645394, 0.743508),
    "imagenet50": (0.514803, 0.582981),
}


def default_trace_items() -> list[dict]:
    items: list[dict] = [
        {"id": "t00_all_zero", "kind": "constant", "value": 0},
        {"id": "t01_all_one", "kind": "constant", "value": 1},
    ]
    for index, ratio_name, ratio, blocks in [
        (2, "12p5", 0.125, 1),
        (3, "12p5", 0.125, 2),
        (4, "12p5", 0.125, 4),
        (5, "12p5", 0.125, 8),
        (6, "25", 0.25, 1),
        (7, "25", 0.25, 2),
        (8, "25", 0.25, 4),
        (9, "25", 0.25, 8),
    ]:
        items.append({"id": f"t{index:02d}_ratio{ratio_name}_block{blocks}", "kind": "block_sweep", "ones_ratio": ratio, "block_count": blocks})
    for index, alpha, blocks in [
        (10, 15, 1),
        (11, 15, 4),
        (12, 35, 1),
        (13, 35, 4),
        (14, 65, 2),
        (15, 65, 8),
        (16, 85, 2),
        (17, 85, 8),
    ]:
        items.append({"id": f"t{index:02d}_offref_a{alpha:02d}_block{blocks}", "kind": "relative_block_sweep", "range_alpha": alpha / 100.0, "block_count": blocks})
    for index, zero_ratio, slot, blocks in [
        (18, 0.03, 0, 2),
        (19, 0.03, 1, 4),
        (20, 0.04, 2, 2),
        (21, 0.04, 3, 8),
        (22, 0.05, 4, 4),
        (23, 0.05, 5, 8),
        (24, 0.06, 6, 4),
        (25, 0.06, 7, 8),
    ]:
        items.append(
            {
                "id": f"t{index:02d}_onebase_z{int(zero_ratio * 100):02d}_slot{slot}_block{blocks}",
                "kind": "one_minus_block_sweep_disjoint",
                "zero_ratio": zero_ratio,
                "slot_index": slot,
                "slot_count": 8,
                "block_count": blocks,
            }
        )
    for index, alpha, seed in [
        (26, 5, 2605),
        (27, 15, 2715),
        (28, 25, 2825),
        (29, 35, 2935),
        (30, 45, 3045),
        (31, 55, 3155),
        (32, 65, 3265),
        (33, 75, 3375),
        (34, 85, 3485),
        (35, 95, 3595),
    ]:
        items.append({"id": f"t{index:02d}_offref_rand_a{alpha:02d}_seed{seed}", "kind": "random_binary_range", "range_alpha": alpha / 100.0, "seed": seed})
    return items


TRACE_ITEMS = default_trace_items()


def parse_csv(raw: str) -> list[str]:
    return [item.strip() for item in str(raw).split(",") if item.strip()]


def parse_shape(raw: str) -> tuple[int, ...] | None:
    if not raw:
        return None
    parts = tuple(int(item.strip()) for item in str(raw).replace("x", ",").split(",") if item.strip())
    if not parts or any(value <= 0 for value in parts):
        raise ValueError(f"invalid shape: {raw!r}")
    return parts


def sampled_length(raw_bit_count: int, sample_stride: int) -> int:
    if raw_bit_count <= 0:
        raise ValueError("--raw-bit-count must be positive")
    return len(range(0, int(raw_bit_count), max(1, int(sample_stride))))


def ratio_reference(dataset: str, ratio_low: float, ratio_high: float) -> dict:
    if ratio_low >= 0.0 and ratio_high >= 0.0:
        low, high, source = float(ratio_low), float(ratio_high), "cli"
    else:
        low, high = DATASET_RATIO_RANGES.get(dataset, DATASET_RATIO_RANGES["generic"])
        source = f"{dataset}_default"
    if not (0.0 <= low <= high <= 1.0):
        raise ValueError(f"invalid ratio range: low={low}, high={high}")
    return {"low": low, "high": high, "source": source}


def selected_items(trace_ids: Sequence[str] | None = None) -> list[dict]:
    by_id = {item["id"]: item for item in TRACE_ITEMS}
    ids = list(trace_ids or by_id)
    unknown = [trace_id for trace_id in ids if trace_id not in by_id]
    if unknown:
        raise ValueError(f"unknown trace ids: {unknown}")
    return [by_id[trace_id] for trace_id in ids]


def distribute_counts(total: int, buckets: int) -> list[int]:
    if total < 0:
        raise ValueError(f"total must be non-negative, got {total}")
    if buckets <= 0:
        return []
    base, extra = divmod(int(total), int(buckets))
    return [base + (1 if index < extra else 0) for index in range(buckets)]


def place_block_pattern(length: int, ones_count: int, requested_blocks: int) -> tuple[torch.Tensor, list[list[int]]]:
    bits = torch.zeros(int(length), dtype=torch.float32)
    if ones_count <= 0:
        return bits, []
    if ones_count >= length:
        bits.fill_(1.0)
        return bits, [[0, int(length)]]

    block_count = max(1, min(int(requested_blocks), int(ones_count)))
    block_sizes = distribute_counts(int(ones_count), block_count)
    gap_sizes = distribute_counts(int(length) - int(ones_count), block_count + 1)
    intervals: list[list[int]] = []
    cursor = int(gap_sizes[0])
    for index, block_size in enumerate(block_sizes):
        start = cursor
        end = start + int(block_size)
        bits[start:end] = 1.0
        intervals.append([start, end])
        cursor = end + int(gap_sizes[index + 1])
    return bits, intervals


def intervals_from_bits(bits: torch.Tensor) -> list[list[int]]:
    intervals: list[list[int]] = []
    start: int | None = None
    for index, value in enumerate(bits.bool().tolist()):
        if value and start is None:
            start = index
        elif not value and start is not None:
            intervals.append([start, index])
            start = None
    if start is not None:
        intervals.append([start, int(bits.numel())])
    return intervals


def ratio_from_reference(item: dict, reference: dict, trace_id: str) -> float:
    low, high = float(reference["low"]), float(reference["high"])
    alpha = float(item["range_alpha"])
    if not (0.0 <= alpha <= 1.0):
        raise ValueError(f"range_alpha must be in [0,1] for {trace_id}, got {alpha}")
    return low + alpha * (high - low)


def bounded_count(ratio: float, length: int, *, enforce_both_bits: bool) -> int:
    if not (0.0 <= ratio <= 1.0):
        raise ValueError(f"ratio must be in [0,1], got {ratio}")
    count = int(round(float(ratio) * int(length)))
    if enforce_both_bits:
        if ratio > 0.0 and count == 0:
            count = 1
        if ratio < 1.0 and count >= length:
            count = int(length) - 1
    return max(0, min(count, int(length)))


def materialize_effective_bits(item: dict, length: int, reference: dict) -> tuple[torch.Tensor, dict]:
    trace_id = str(item.get("id", ""))
    kind = str(item.get("kind", ""))
    if not trace_id:
        raise ValueError(f"missing trace id: {item}")

    if kind == "constant":
        value = int(item.get("value", 0))
        if value not in (0, 1):
            raise ValueError(f"unsupported constant value for {trace_id}: {value}")
        bits = torch.full((int(length),), float(value), dtype=torch.float32)
        intervals = [] if value == 0 else [[0, int(length)]]
        return bits, {
            "id": trace_id,
            "kind": kind,
            "ones_ratio_requested": float(value),
            "ones_ratio_effective": float(bits.mean().item()),
            "ones_count_effective": int(bits.sum().item()),
            "block_count_requested": 1 if value else 0,
            "block_count_effective": 1 if value else 0,
            "intervals_effective": intervals,
        }

    if kind in {"block_sweep", "relative_block_sweep"}:
        ratio = float(item["ones_ratio"]) if kind == "block_sweep" else ratio_from_reference(item, reference, trace_id)
        ones_count = bounded_count(ratio, length, enforce_both_bits=True)
        bits, intervals = place_block_pattern(length, ones_count, int(item["block_count"]))
        return bits, {
            "id": trace_id,
            "kind": kind,
            "ones_ratio_requested": float(ratio),
            "ones_ratio_effective": float(bits.mean().item()),
            "ones_count_effective": int(bits.sum().item()),
            "block_count_requested": int(item["block_count"]),
            "block_count_effective": len(intervals),
            "intervals_effective": intervals,
        }

    if kind == "one_minus_block_sweep_disjoint":
        zero_ratio = float(item["zero_ratio"])
        slot_index, slot_count = int(item["slot_index"]), int(item["slot_count"])
        if slot_count <= 0 or not (0 <= slot_index < slot_count):
            raise ValueError(f"invalid disjoint slot for {trace_id}: {slot_index}/{slot_count}")
        slot_start = int(slot_index * int(length) // slot_count)
        slot_end = int((slot_index + 1) * int(length) // slot_count)
        slot_len = max(0, slot_end - slot_start)
        zero_count = bounded_count(zero_ratio, length, enforce_both_bits=False)
        if zero_ratio > 0.0 and zero_count == 0:
            zero_count = 1
        if zero_count > slot_len:
            raise ValueError(f"{trace_id}: zero_count {zero_count} exceeds slot length {slot_len}")
        bits = torch.ones(int(length), dtype=torch.float32)
        _, local_intervals = place_block_pattern(slot_len, zero_count, int(item["block_count"]))
        intervals: list[list[int]] = []
        for start_local, end_local in local_intervals:
            start, end = slot_start + start_local, slot_start + end_local
            bits[start:end] = 0.0
            intervals.append([start, end])
        return bits, {
            "id": trace_id,
            "kind": kind,
            "ones_ratio_requested": float(1.0 - zero_ratio),
            "ones_ratio_effective": float(bits.mean().item()),
            "ones_count_effective": int(bits.sum().item()),
            "zero_ratio_requested": zero_ratio,
            "zero_count_effective": int(zero_count),
            "zero_slot_index": slot_index,
            "zero_slot_count": slot_count,
            "zero_slot_span_effective": [slot_start, slot_end],
            "block_count_requested": int(item["block_count"]),
            "block_count_effective": len(intervals),
            "intervals_effective": intervals,
        }

    if kind == "random_binary_range":
        ratio = ratio_from_reference(item, reference, trace_id)
        ones_count = bounded_count(ratio, length, enforce_both_bits=True)
        bits = torch.zeros(int(length), dtype=torch.float32)
        if ones_count:
            generator = torch.Generator()
            generator.manual_seed(int(item.get("seed", 0)))
            bits[torch.randperm(int(length), generator=generator)[:ones_count]] = 1.0
        intervals = intervals_from_bits(bits)
        return bits, {
            "id": trace_id,
            "kind": kind,
            "seed": int(item.get("seed", 0)),
            "ones_ratio_requested": float(ratio),
            "ones_ratio_effective": float(bits.mean().item()),
            "ones_count_effective": int(bits.sum().item()),
            "block_count_requested": None,
            "block_count_effective": len(intervals),
            "intervals_effective": intervals,
        }

    raise ValueError(f"unsupported trace kind for {trace_id}: {kind}")


def pad_or_trim(trace: torch.Tensor, shape: tuple[int, ...] | None) -> torch.Tensor:
    if shape is None:
        return trace.float()
    flat = trace.flatten().float()
    need = int(prod(shape))
    if flat.numel() < need:
        flat = torch.cat([flat, torch.zeros(need - flat.numel(), dtype=flat.dtype)])
    return flat[:need].view(*shape)


def generate_traces(
    *,
    raw_bit_count: int,
    sample_stride: int,
    dataset: str,
    ratio_low: float = -1.0,
    ratio_high: float = -1.0,
    trace_ids: Sequence[str] | None = None,
    trace_shape: tuple[int, ...] | None = None,
) -> tuple[torch.Tensor, list[str], torch.Tensor, dict]:
    length = sampled_length(raw_bit_count, sample_stride)
    reference = ratio_reference(dataset, ratio_low, ratio_high)
    traces, ids, metas = [], [], {}
    for item in selected_items(trace_ids):
        bits, meta = materialize_effective_bits(item, length, reference)
        trace = pad_or_trim(bits, trace_shape)
        traces.append(trace)
        ids.append(str(item["id"]))
        metas[str(item["id"])] = {**meta, "stored_shape": list(trace.shape)}
    ratios = torch.tensor([float(metas[trace_id]["ones_ratio_effective"]) for trace_id in ids], dtype=torch.float32)
    protocol = {
        "name": "mismatch_trace_suite",
        "raw_bit_count": int(raw_bit_count),
        "sample_stride": int(sample_stride),
        "effective_length": int(length),
        "ratio_reference": reference,
        "trace_shape": list(trace_shape) if trace_shape else None,
        "traces": metas,
    }
    return torch.stack(traces), ids, ratios, protocol
