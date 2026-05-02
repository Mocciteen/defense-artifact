#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

from extract import BitRecord, bit_counts, iter_records_from_path, record_matches


SUMMARY_COLUMNS = [
    "group",
    "mode",
    "backend",
    "unit",
    "record_count",
    "compare_count",
    "unchanged_count",
    "changed_count",
    "unchanged_ratio",
    "changed_ratio",
]


PAIR_COLUMNS = [
    "group",
    "backend",
    "unit",
    "baseline_mode",
    "candidate_mode",
    "baseline_unchanged_ratio",
    "candidate_unchanged_ratio",
    "absolute_drop",
    "relative_drop",
    "baseline_compare_count",
    "candidate_compare_count",
]


@dataclass
class Accumulator:
    record_count: int = 0
    compare_count: int = 0
    unchanged_count: int = 0
    changed_count: int = 0

    def add(self, record: BitRecord) -> None:
        self.record_count += 1
        self.compare_count += int(record.compare_count)
        self.unchanged_count += int(record.unchanged_count)
        self.changed_count += int(record.changed_count)

    @property
    def unchanged_ratio(self) -> float:
        if self.compare_count == 0:
            return 0.0
        return self.unchanged_count / self.compare_count

    @property
    def changed_ratio(self) -> float:
        if self.compare_count == 0:
            return 0.0
        return self.changed_count / self.compare_count


def parse_int(value: str | None, default: int | None = None) -> int | None:
    if value is None or str(value).strip() == "":
        return default
    return int(str(value), 0)


def sniff_delimiter(path: Path) -> str:
    sample = path.read_text(encoding="utf-8", errors="replace")[:4096]
    try:
        return csv.Sniffer().sniff(sample, delimiters="\t,").delimiter
    except csv.Error:
        return "\t"


def read_table(path: Path) -> Iterator[dict[str, str]]:
    delimiter = sniff_delimiter(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        for row in reader:
            yield {key: (value or "") for key, value in row.items() if key is not None}


def row_path(row: dict[str, str], base_dir: Path) -> Path:
    raw = row.get("path") or row.get("source")
    if not raw:
        raise ValueError("manifest row must contain path or source")
    path = Path(raw)
    if not path.is_absolute():
        path = base_dir / path
    return path


def row_metadata(row: dict[str, str]) -> dict[str, str]:
    return {
        "backend": row.get("backend", ""),
        "group": row.get("group", ""),
        "mode": row.get("mode", ""),
        "sample": row.get("sample", ""),
        "location": row.get("location", ""),
        "owner": row.get("owner", ""),
    }


def iter_manifest_records(path: Path) -> Iterator[BitRecord]:
    base_dir = path.parent
    for row in read_table(path):
        record_path = row_path(row, base_dir)
        file_format = row.get("format") or "auto"
        unit = row.get("unit") or None
        bit_length = parse_int(row.get("bit_length"), None)
        owner_filter = row.get("owner_contains") or None
        record_filter = row.get("record_contains") or None
        min_bits = parse_int(row.get("min_bits"), 0) or 0
        for record in iter_records_from_path(
            record_path,
            file_format=file_format,
            metadata=row_metadata(row),
            unit=unit,
            bit_length=bit_length,
        ):
            if record_matches(
                record,
                owner_contains=owner_filter,
                record_contains=record_filter,
                min_bits=min_bits,
            ):
                yield record


def record_from_tsv_row(row: dict[str, str]) -> BitRecord:
    bits = row.get("bits", "")
    if bits:
        compare, unchanged, changed = bit_counts(bits)
    else:
        compare = int(row.get("compare_count") or row.get("bits_len") or 0)
        unchanged = int(row.get("unchanged_count") or 0)
        changed = int(row.get("changed_count") or max(compare - unchanged, 0))
    return BitRecord(
        source=row.get("source", ""),
        record_id=row.get("record_id", ""),
        unit=row.get("unit", ""),
        bits=bits,
        compare_count=compare,
        unchanged_count=unchanged,
        changed_count=changed,
        backend=row.get("backend", ""),
        group=row.get("group", ""),
        mode=row.get("mode", ""),
        sample=row.get("sample", ""),
        location=row.get("location", ""),
        owner=row.get("owner", ""),
    )


def iter_records_tsv(path: Path) -> Iterator[BitRecord]:
    for row in read_table(path):
        yield record_from_tsv_row(row)


def iter_direct_records(args: argparse.Namespace) -> Iterator[BitRecord]:
    metadata = {
        "backend": args.backend,
        "group": args.group,
        "mode": args.mode,
        "sample": args.sample,
    }
    for path in args.input or []:
        yield from iter_records_from_path(
            path,
            file_format=args.format,
            metadata=metadata,
            unit=args.unit,
            bit_length=args.bit_length,
        )


def iter_all_records(args: argparse.Namespace) -> Iterator[BitRecord]:
    for path in args.manifest or []:
        yield from iter_manifest_records(path)
    for path in args.records_tsv or []:
        yield from iter_records_tsv(path)
    yield from iter_direct_records(args)


def group_fields(value: str) -> list[str]:
    fields = [part.strip() for part in value.split(",") if part.strip()]
    return fields or ["group", "mode", "backend", "unit"]


def record_key(record: BitRecord, fields: Iterable[str]) -> tuple[str, ...]:
    return tuple(str(getattr(record, field, "")) for field in fields)


def build_summary(records: Iterable[BitRecord], fields: list[str]) -> tuple[list[BitRecord], list[dict[str, str]]]:
    kept_records = list(records)
    groups: dict[tuple[str, ...], Accumulator] = {}
    for record in kept_records:
        groups.setdefault(record_key(record, fields), Accumulator()).add(record)

    rows: list[dict[str, str]] = []
    for key in sorted(groups):
        acc = groups[key]
        row = {field: value for field, value in zip(fields, key)}
        for column in SUMMARY_COLUMNS:
            row.setdefault(column, "")
        row.update(
            {
                "record_count": str(acc.record_count),
                "compare_count": str(acc.compare_count),
                "unchanged_count": str(acc.unchanged_count),
                "changed_count": str(acc.changed_count),
                "unchanged_ratio": f"{acc.unchanged_ratio:.12g}",
                "changed_ratio": f"{acc.changed_ratio:.12g}",
            }
        )
        rows.append(row)
    return kept_records, rows


def build_pair_rows(
    summary_rows: list[dict[str, str]],
    baseline_mode: str,
    candidate_mode: str,
) -> list[dict[str, str]]:
    keyed: dict[tuple[str, str, str], dict[str, dict[str, str]]] = {}
    for row in summary_rows:
        key = (row.get("group", ""), row.get("backend", ""), row.get("unit", ""))
        keyed.setdefault(key, {})[row.get("mode", "")] = row

    rows: list[dict[str, str]] = []
    for (group, backend, unit), by_mode in sorted(keyed.items()):
        baseline = by_mode.get(baseline_mode)
        candidate = by_mode.get(candidate_mode)
        if baseline is None or candidate is None:
            continue
        baseline_ratio = float(baseline.get("unchanged_ratio") or 0.0)
        candidate_ratio = float(candidate.get("unchanged_ratio") or 0.0)
        absolute_drop = baseline_ratio - candidate_ratio
        relative_drop = absolute_drop / baseline_ratio if baseline_ratio else 0.0
        rows.append(
            {
                "group": group,
                "backend": backend,
                "unit": unit,
                "baseline_mode": baseline_mode,
                "candidate_mode": candidate_mode,
                "baseline_unchanged_ratio": f"{baseline_ratio:.12g}",
                "candidate_unchanged_ratio": f"{candidate_ratio:.12g}",
                "absolute_drop": f"{absolute_drop:.12g}",
                "relative_drop": f"{relative_drop:.12g}",
                "baseline_compare_count": baseline.get("compare_count", "0"),
                "candidate_compare_count": candidate.get("compare_count", "0"),
            }
        )
    return rows


def output_columns(rows: list[dict[str, str]], preferred: list[str]) -> list[str]:
    columns = list(preferred)
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    return columns


def write_table(path: Path | None, rows: list[dict[str, str]], preferred_columns: list[str]) -> None:
    columns = output_columns(rows, preferred_columns)
    if path is None:
        handle = sys.stdout
        close = False
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        handle = path.open("w", encoding="utf-8", newline="")
        close = True
    try:
        writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    finally:
        if close:
            handle.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate adjacent-change 01 leakage records and off/on reductions."
    )
    parser.add_argument("--manifest", type=Path, action="append", default=[])
    parser.add_argument("--records-tsv", type=Path, action="append", default=[])
    parser.add_argument("--input", type=Path, action="append", default=[])
    parser.add_argument("--format", choices=["auto", "json", "text", "packed"], default="auto")
    parser.add_argument("--unit", default=None)
    parser.add_argument("--backend", default="")
    parser.add_argument("--group", default="")
    parser.add_argument("--mode", default="")
    parser.add_argument("--sample", default="")
    parser.add_argument("--bit-length", type=int, default=None)
    parser.add_argument("--owner-contains", default=None)
    parser.add_argument("--record-contains", default=None)
    parser.add_argument("--min-bits", type=int, default=0)
    parser.add_argument("--group-by", default="group,mode,backend,unit")
    parser.add_argument("--baseline-mode", default="off")
    parser.add_argument("--candidate-mode", default="on")
    parser.add_argument("--out-summary", type=Path, default=None)
    parser.add_argument("--out-pairs", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    filtered = (
        record
        for record in iter_all_records(args)
        if record_matches(
            record,
            owner_contains=args.owner_contains,
            record_contains=args.record_contains,
            min_bits=args.min_bits,
        )
    )
    _, summary_rows = build_summary(filtered, group_fields(args.group_by))
    pair_rows = build_pair_rows(summary_rows, args.baseline_mode, args.candidate_mode)

    if args.out_summary is None and args.out_pairs is None:
        write_table(None, summary_rows, SUMMARY_COLUMNS)
    else:
        if args.out_summary is not None:
            write_table(args.out_summary, summary_rows, SUMMARY_COLUMNS)
        if args.out_pairs is not None:
            write_table(args.out_pairs, pair_rows, PAIR_COLUMNS)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
