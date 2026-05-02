#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator


CHUNK_SIZE = 1 << 20
BIT_TEXT_RE = re.compile(r"^[01\s]*$")


@dataclass
class BitRecord:
    source: str
    record_id: str
    unit: str
    bits: str = ""
    compare_count: int = 0
    unchanged_count: int = 0
    changed_count: int = 0
    backend: str = ""
    group: str = ""
    mode: str = ""
    sample: str = ""
    location: str = ""
    owner: str = ""

    @property
    def unchanged_ratio(self) -> float:
        if self.compare_count == 0:
            return 0.0
        return self.unchanged_count / self.compare_count

    def as_row(self, include_bits: bool = False) -> dict[str, str]:
        row = {
            "source": self.source,
            "record_id": self.record_id,
            "backend": self.backend,
            "group": self.group,
            "mode": self.mode,
            "sample": self.sample,
            "unit": self.unit,
            "location": self.location,
            "owner": self.owner,
            "compare_count": str(self.compare_count),
            "unchanged_count": str(self.unchanged_count),
            "changed_count": str(self.changed_count),
            "unchanged_ratio": f"{self.unchanged_ratio:.12g}",
        }
        if include_bits:
            row["bits"] = self.bits
        return row


BASE_COLUMNS = [
    "source",
    "record_id",
    "backend",
    "group",
    "mode",
    "sample",
    "unit",
    "location",
    "owner",
    "compare_count",
    "unchanged_count",
    "changed_count",
    "unchanged_ratio",
]


def clean_bits_text(text: str, source: str) -> str:
    if not BIT_TEXT_RE.match(text):
        bad = sorted({ch for ch in text if ch not in "01 \t\r\n"})
        raise ValueError(f"{source}: non-bit characters found: {bad[:8]}")
    return "".join(ch for ch in text if ch in "01")


def unpack_bits(data: bytes, bit_length: int | None = None) -> str:
    bits = "".join(f"{byte:08b}" for byte in data)
    if bit_length is not None:
        if bit_length < 0:
            raise ValueError("bit_length must be non-negative")
        bits = bits[:bit_length]
    return bits


def bit_counts(bits: str) -> tuple[int, int, int]:
    unchanged = bits.count("0")
    changed = bits.count("1")
    compare = len(bits)
    if unchanged + changed != compare:
        raise ValueError("bit string must contain only 0 and 1")
    return compare, unchanged, changed


def first_text(mapping: dict, keys: Iterable[str]) -> str:
    for key in keys:
        value = mapping.get(key)
        if value is not None and str(value):
            return str(value)
    return ""


def owner_label(item: dict) -> str:
    owner = item.get("owner") or {}
    if isinstance(owner, dict):
        label = first_text(owner, ["symbol", "site", "routine", "offset", "module"])
        if label:
            return label
    return first_text(item, ["symbol", "site", "routine", "offset", "module"])


def item_location(item: dict) -> str:
    return first_text(item, ["paddr_hex", "ip_hex", "location"])


def make_record_id(path: Path, unit: str, item: dict, index: int) -> str:
    location = item_location(item)
    if location:
        return f"{unit}:{location}"
    owner = owner_label(item)
    if owner:
        return f"{unit}:{owner}:{index}"
    return f"{path.name}:{unit}:{index}"


def infer_address_unit(path: Path, item: dict, forced_unit: str | None) -> str:
    if forced_unit:
        return forced_unit
    if "taint_block16" in path.name or int(item.get("size", 0) or 0) == 16:
        return "block16"
    return "address"


def metadata_value(metadata: dict[str, str], key: str) -> str:
    value = metadata.get(key, "")
    return "" if value is None else str(value)


def record_from_item(
    *,
    path: Path,
    item: dict,
    index: int,
    unit: str,
    metadata: dict[str, str],
) -> BitRecord:
    bits = str(item.get("bits") or "")
    compare, unchanged, changed = bit_counts(bits)

    compare_count = int(item.get("compare_count") or item.get("bits_len") or compare)
    unchanged_count = int(item.get("unchanged_count") or unchanged)
    changed_count = int(item.get("changed_count") or changed)

    if compare_count != compare:
        compare_count = compare
    if unchanged_count + changed_count != compare_count:
        unchanged_count = unchanged
        changed_count = changed

    return BitRecord(
        source=str(path),
        record_id=make_record_id(path, unit, item, index),
        unit=unit,
        bits=bits,
        compare_count=compare_count,
        unchanged_count=unchanged_count,
        changed_count=changed_count,
        backend=metadata_value(metadata, "backend"),
        group=metadata_value(metadata, "group"),
        mode=metadata_value(metadata, "mode"),
        sample=metadata_value(metadata, "sample"),
        location=item_location(item),
        owner=owner_label(item),
    )


def find_array_start(handle, key: str) -> str | None:
    target = f'"{key}"'
    keep = len(target) + 256
    buffer = ""

    while True:
        chunk = handle.read(CHUNK_SIZE)
        if not chunk:
            return None
        buffer += chunk
        key_pos = buffer.find(target)
        if key_pos >= 0:
            bracket_pos = buffer.find("[", key_pos + len(target))
            while bracket_pos < 0:
                chunk = handle.read(CHUNK_SIZE)
                if not chunk:
                    return None
                buffer += chunk
                bracket_pos = buffer.find("[", key_pos + len(target))
            return buffer[bracket_pos + 1 :]
        if len(buffer) > keep:
            buffer = buffer[-keep:]


def iter_array_objects(path: Path, key: str) -> Iterator[dict]:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        first = find_array_start(handle, key)
        if first is None:
            return

        def chars() -> Iterator[str]:
            yield from first
            while True:
                chunk = handle.read(CHUNK_SIZE)
                if not chunk:
                    break
                yield from chunk

        collecting = False
        in_string = False
        escaped = False
        depth = 0
        current: list[str] = []

        for ch in chars():
            if not collecting:
                if ch == "{":
                    collecting = True
                    depth = 1
                    in_string = False
                    escaped = False
                    current = ["{"]
                elif ch == "]":
                    return
                continue

            current.append(ch)
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
                continue

            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    yield json.loads("".join(current))
                    collecting = False
                    current = []


def iter_json_records(
    path: Path,
    metadata: dict[str, str],
    forced_unit: str | None = None,
) -> Iterator[BitRecord]:
    with path.open("r", encoding="utf-8", errors="replace") as header_file:
        header = header_file.read(65536)
    if '"site_bits_only": true' in header or '"sites": [' in header:
        for index, item in enumerate(iter_array_objects(path, "sites")):
            if item.get("bits"):
                yield record_from_item(
                    path=path,
                    item=item,
                    index=index,
                    unit=forced_unit or "site",
                    metadata=metadata,
                )

    for index, item in enumerate(iter_array_objects(path, "addresses")):
        if item.get("bits"):
            yield record_from_item(
                path=path,
                item=item,
                index=index,
                unit=infer_address_unit(path, item, forced_unit),
                metadata=metadata,
            )


def iter_text_record(
    path: Path,
    metadata: dict[str, str],
    unit: str,
) -> Iterator[BitRecord]:
    bits = clean_bits_text(path.read_text(encoding="utf-8", errors="replace"), str(path))
    compare, unchanged, changed = bit_counts(bits)
    yield BitRecord(
        source=str(path),
        record_id=f"{path.name}:{unit}",
        unit=unit,
        bits=bits,
        compare_count=compare,
        unchanged_count=unchanged,
        changed_count=changed,
        backend=metadata_value(metadata, "backend"),
        group=metadata_value(metadata, "group"),
        mode=metadata_value(metadata, "mode"),
        sample=metadata_value(metadata, "sample"),
        location=metadata_value(metadata, "location"),
        owner=metadata_value(metadata, "owner"),
    )


def iter_packed_record(
    path: Path,
    metadata: dict[str, str],
    unit: str,
    bit_length: int | None,
) -> Iterator[BitRecord]:
    bits = unpack_bits(path.read_bytes(), bit_length)
    compare, unchanged, changed = bit_counts(bits)
    yield BitRecord(
        source=str(path),
        record_id=f"{path.name}:{unit}",
        unit=unit,
        bits=bits,
        compare_count=compare,
        unchanged_count=unchanged,
        changed_count=changed,
        backend=metadata_value(metadata, "backend"),
        group=metadata_value(metadata, "group"),
        mode=metadata_value(metadata, "mode"),
        sample=metadata_value(metadata, "sample"),
        location=metadata_value(metadata, "location"),
        owner=metadata_value(metadata, "owner"),
    )


def resolve_format(path: Path, requested: str) -> str:
    if requested != "auto":
        return requested
    suffix = path.suffix.lower()
    if suffix == ".json":
        return "json"
    if suffix == ".bin":
        return "packed"
    return "text"


def iter_records_from_path(
    path: Path,
    *,
    file_format: str = "auto",
    metadata: dict[str, str] | None = None,
    unit: str | None = None,
    bit_length: int | None = None,
) -> Iterator[BitRecord]:
    metadata = metadata or {}
    resolved = resolve_format(path, file_format)
    if resolved == "json":
        yield from iter_json_records(path, metadata, unit)
    elif resolved == "packed":
        yield from iter_packed_record(path, metadata, unit or "bitstring", bit_length)
    elif resolved == "text":
        yield from iter_text_record(path, metadata, unit or "bitstring")
    else:
        raise ValueError(f"unsupported format: {resolved}")


def record_matches(
    record: BitRecord,
    *,
    owner_contains: str | None = None,
    record_contains: str | None = None,
    min_bits: int = 0,
) -> bool:
    if owner_contains and owner_contains not in record.owner:
        return False
    if record_contains and record_contains not in record.record_id:
        return False
    if min_bits and record.compare_count < min_bits:
        return False
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normalize adjacent-change 01 leakage records from trace outputs or bit files."
    )
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--format", choices=["auto", "json", "text", "packed"], default="auto")
    parser.add_argument("--unit", default=None, help="Override unit label, e.g. block16/address/site.")
    parser.add_argument("--backend", default="")
    parser.add_argument("--group", default="")
    parser.add_argument("--mode", default="")
    parser.add_argument("--sample", default="")
    parser.add_argument("--bit-length", type=int, default=None, help="Required only for packed files with padding bits.")
    parser.add_argument("--owner-contains", default=None)
    parser.add_argument("--record-contains", default=None)
    parser.add_argument("--min-bits", type=int, default=0)
    parser.add_argument("--include-bits", action="store_true", help="Include raw 01 strings in the TSV output.")
    return parser.parse_args()


def open_output(path: Path | None):
    if path is None:
        return sys.stdout
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.open("w", encoding="utf-8", newline="")


def main() -> int:
    args = parse_args()
    metadata = {
        "backend": args.backend,
        "group": args.group,
        "mode": args.mode,
        "sample": args.sample,
    }
    columns = BASE_COLUMNS + (["bits"] if args.include_bits else [])

    with open_output(args.output) as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for path in args.input:
            for record in iter_records_from_path(
                path,
                file_format=args.format,
                metadata=metadata,
                unit=args.unit,
                bit_length=args.bit_length,
            ):
                if record_matches(
                    record,
                    owner_contains=args.owner_contains,
                    record_contains=args.record_contains,
                    min_bits=args.min_bits,
                ):
                    writer.writerow(record.as_row(include_bits=args.include_bits))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
