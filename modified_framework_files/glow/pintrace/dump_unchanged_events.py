#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import struct
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, Optional, TextIO


HEADER_STRUCT = struct.Struct("<8sIIQQ")
UNCHANGED_RECORD_STRUCT = struct.Struct("<QQQQQQIHH16s16s")
BLOCK_RECORD_STRUCT = struct.Struct("<QQQQQQIHHQ16s16s")
MAGIC_UNCHANGED = b"TB16UEV"
MAGIC_BLOCK = b"TB16BEV"


@dataclass(frozen=True)
class Header:
    kind: str
    version: int
    record_size: int
    block_size: int


@dataclass(frozen=True)
class SiteMeta:
    ip_hex: str
    module: str
    offset: str
    symbol: str
    disasm: str


@dataclass(frozen=True)
class Record:
    seq: int
    block_vaddr: int
    site_ip: int
    write_vaddr: int
    block_compare_index: int
    block_unchanged_index: int
    write_size: int
    overlap_offset: int
    overlap_size: int
    changed: bool
    block_before: bytes
    block_after: bytes
    overlap_after: bytes


def parse_int(value: str) -> int:
    return int(value, 0)


def hex_u64(value: int) -> str:
    return f"0x{value:016x}"


def hex_bytes(data: bytes, size: Optional[int] = None) -> str:
    payload = data if size is None else data[:size]
    return payload.hex()


def read_header(stream, path: Path) -> Header:
    raw = stream.read(HEADER_STRUCT.size)
    if len(raw) != HEADER_STRUCT.size:
        raise ValueError(f"{path} is too small to contain a valid header")

    magic, version, record_size, block_size, _reserved = HEADER_STRUCT.unpack(raw)
    if magic.startswith(MAGIC_UNCHANGED):
        kind = "unchanged"
        expected_size = UNCHANGED_RECORD_STRUCT.size
    elif magic.startswith(MAGIC_BLOCK):
        kind = "block"
        expected_size = BLOCK_RECORD_STRUCT.size
    else:
        raise ValueError(
            f"{path} has unexpected magic {magic!r}; "
            f"expected prefix {MAGIC_UNCHANGED!r} or {MAGIC_BLOCK!r}"
        )

    if record_size != expected_size:
        raise ValueError(
            f"{path} record_size={record_size}, parser expects {expected_size} "
            f"for kind={kind}"
        )

    return Header(
        kind=kind, version=version, record_size=record_size, block_size=block_size
    )


def load_header(path: Path) -> Header:
    with path.open("rb") as stream:
        return read_header(stream, path)


def iter_records(path: Path) -> Iterator[Record]:
    with path.open("rb") as stream:
        header = read_header(stream, path)
        record_struct = (
            UNCHANGED_RECORD_STRUCT
            if header.kind == "unchanged"
            else BLOCK_RECORD_STRUCT
        )

        while True:
            raw = stream.read(record_struct.size)
            if not raw:
                return
            if len(raw) != record_struct.size:
                raise ValueError(
                    f"{path} ended with a partial {header.kind} record "
                    f"({len(raw)} bytes)"
                )

            if header.kind == "unchanged":
                (
                    seq,
                    block_vaddr,
                    site_ip,
                    write_vaddr,
                    block_compare_index,
                    block_unchanged_index,
                    write_size,
                    overlap_offset,
                    overlap_size,
                    block_before,
                    overlap_after,
                ) = record_struct.unpack(raw)
                block_after = block_before
                changed = False
            else:
                (
                    seq,
                    block_vaddr,
                    site_ip,
                    write_vaddr,
                    block_compare_index,
                    block_unchanged_index,
                    write_size,
                    overlap_offset,
                    overlap_size,
                    flags,
                    block_before,
                    block_after,
                ) = record_struct.unpack(raw)
                overlap_after = block_after[overlap_offset : overlap_offset + overlap_size]
                changed = (flags & 1) != 0

            yield Record(
                seq=seq,
                block_vaddr=block_vaddr,
                site_ip=site_ip,
                write_vaddr=write_vaddr,
                block_compare_index=block_compare_index,
                block_unchanged_index=block_unchanged_index,
                write_size=write_size,
                overlap_offset=overlap_offset,
                overlap_size=overlap_size,
                changed=changed,
                block_before=block_before,
                block_after=block_after,
                overlap_after=overlap_after[:overlap_size],
            )


def load_ip_map(path: Optional[Path]) -> Dict[int, SiteMeta]:
    if path is None:
        return {}

    mapping: Dict[int, SiteMeta] = {}
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line or line.startswith("#"):
                continue
            line = line.rstrip("\n")
            parts = line.split("\t", 4)
            if len(parts) != 5:
                continue
            ip_hex, module, offset, symbol, disasm = parts
            mapping[int(ip_hex, 16)] = SiteMeta(
                ip_hex=ip_hex,
                module=module,
                offset=offset,
                symbol=symbol,
                disasm=disasm,
            )
    return mapping


def record_to_dict(record: Record, site_meta: Dict[int, SiteMeta]) -> dict:
    overlap_before = record.block_before[
        record.overlap_offset : record.overlap_offset + record.overlap_size
    ]
    payload = {
        "seq": record.seq,
        "block_vaddr": record.block_vaddr,
        "block_vaddr_hex": hex_u64(record.block_vaddr),
        "block_compare_index": record.block_compare_index,
        "block_unchanged_index": record.block_unchanged_index,
        "changed": int(record.changed),
        "site_ip": record.site_ip,
        "site_ip_hex": hex_u64(record.site_ip),
        "write_vaddr": record.write_vaddr,
        "write_vaddr_hex": hex_u64(record.write_vaddr),
        "write_size": record.write_size,
        "overlap_offset": record.overlap_offset,
        "overlap_size": record.overlap_size,
        "overlap_before_hex": hex_bytes(overlap_before),
        "overlap_after_hex": hex_bytes(record.overlap_after),
        "block_before_hex": hex_bytes(record.block_before),
        "block_after_hex": hex_bytes(record.block_after),
    }

    meta = site_meta.get(record.site_ip)
    if meta is not None:
        payload["site_module"] = meta.module
        payload["site_offset"] = meta.offset
        payload["site_symbol"] = meta.symbol
        payload["site_disasm"] = meta.disasm

    return payload


def open_output(path: Optional[Path]) -> TextIO:
    if path is None:
        return sys.stdout
    return path.open("w", encoding="utf-8", newline="")


def record_matches_filters(record: Record, args: argparse.Namespace) -> bool:
    if args.block is not None and record.block_vaddr != args.block:
        return False
    if args.k is not None and record.block_unchanged_index != args.k:
        return False
    if getattr(args, "only_unchanged", False) and record.changed:
        return False
    if getattr(args, "only_changed", False) and not record.changed:
        return False
    return True


def command_summary(args: argparse.Namespace) -> int:
    header = load_header(args.bin)
    block_counts: Counter[int] = Counter()
    total_events = 0
    changed_events = 0
    unchanged_events = 0

    for record in iter_records(args.bin):
        block_counts[record.block_vaddr] += 1
        total_events += 1
        if record.changed:
            changed_events += 1
        else:
            unchanged_events += 1

    print(f"kind={header.kind}")
    print(f"version={header.version}")
    print(f"record_size={header.record_size}")
    print(f"block_size={header.block_size}")
    print(f"events={total_events}")
    print(f"blocks={len(block_counts)}")
    print(f"changed_events={changed_events}")
    print(f"unchanged_events={unchanged_events}")

    if args.top or args.all:
        if args.sort == "addr":
            items = sorted(block_counts.items(), key=lambda item: item[0])
        else:
            items = sorted(
                block_counts.items(), key=lambda item: (-item[1], item[0])
            )
        if args.top is not None and not args.all:
            items = items[: args.top]
        print("block_vaddr_hex\tevent_count")
        for block_vaddr, count in items:
            print(f"{hex_u64(block_vaddr)}\t{count}")

    return 0


def command_dump_block(args: argparse.Namespace) -> int:
    site_meta = load_ip_map(args.ip_map)
    matched = 0

    for record in iter_records(args.bin):
        if not record_matches_filters(record, args):
            continue
        payload = record_to_dict(record, site_meta)
        if args.jsonl:
            print(json.dumps(payload, ensure_ascii=False))
        else:
            print(
                f"block={payload['block_vaddr_hex']} "
                f"compare={payload['block_compare_index']} "
                f"k0={payload['block_unchanged_index']} "
                f"changed={payload['changed']} "
                f"site={payload['site_ip_hex']} "
                f"write={payload['write_vaddr_hex']}+{payload['write_size']} "
                f"overlap={payload['overlap_offset']}:{payload['overlap_size']}"
            )
            print(
                f"  overlap_before={payload['overlap_before_hex']} "
                f"overlap_after={payload['overlap_after_hex']}"
            )
            print(
                f"  block_before={payload['block_before_hex']} "
                f"block_after={payload['block_after_hex']}"
            )
            if "site_symbol" in payload:
                print(
                    f"  symbol={payload['site_symbol']} "
                    f"module={payload['site_module']} "
                    f"offset={payload['site_offset']}"
                )
                print(f"  disasm={payload['site_disasm']}")
        matched += 1

    if matched == 0:
        message = f"no event found for block={hex_u64(args.block)}"
        if args.k is not None:
            message += f" k={args.k}"
        if args.only_unchanged:
            message += " only_unchanged=1"
        if args.only_changed:
            message += " only_changed=1"
        print(message, file=sys.stderr)
        return 1
    return 0


def command_export_map(args: argparse.Namespace) -> int:
    site_meta = load_ip_map(args.ip_map)
    output = open_output(args.output)
    close_output = output is not sys.stdout

    try:
        if args.format == "tsv":
            writer = csv.writer(output, delimiter="\t")
            writer.writerow(
                [
                    "seq",
                    "block_vaddr",
                    "block_vaddr_hex",
                    "block_compare_index",
                    "block_unchanged_index",
                    "changed",
                    "site_ip",
                    "site_ip_hex",
                    "write_vaddr",
                    "write_vaddr_hex",
                    "write_size",
                    "overlap_offset",
                    "overlap_size",
                    "overlap_before_hex",
                    "overlap_after_hex",
                    "block_before_hex",
                    "block_after_hex",
                    "site_module",
                    "site_offset",
                    "site_symbol",
                    "site_disasm",
                ]
            )
            for record in iter_records(args.bin):
                if not record_matches_filters(record, args):
                    continue
                payload = record_to_dict(record, site_meta)
                writer.writerow(
                    [
                        payload["seq"],
                        payload["block_vaddr"],
                        payload["block_vaddr_hex"],
                        payload["block_compare_index"],
                        payload["block_unchanged_index"],
                        payload["changed"],
                        payload["site_ip"],
                        payload["site_ip_hex"],
                        payload["write_vaddr"],
                        payload["write_vaddr_hex"],
                        payload["write_size"],
                        payload["overlap_offset"],
                        payload["overlap_size"],
                        payload["overlap_before_hex"],
                        payload["overlap_after_hex"],
                        payload["block_before_hex"],
                        payload["block_after_hex"],
                        payload.get("site_module", ""),
                        payload.get("site_offset", ""),
                        payload.get("site_symbol", ""),
                        payload.get("site_disasm", ""),
                    ]
                )
        else:
            for record in iter_records(args.bin):
                if not record_matches_filters(record, args):
                    continue
                payload = record_to_dict(record, site_meta)
                output.write(json.dumps(payload, ensure_ascii=False))
                output.write("\n")
    finally:
        if close_output:
            output.close()

    return 0


def add_change_filter_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--only-unchanged",
        action="store_true",
        help="only keep changed=0 events",
    )
    group.add_argument(
        "--only-changed",
        action="store_true",
        help="only keep changed=1 events",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect taintblock16trace event bins. Supports both "
            "-block-event-bin (all compares) and -unchanged-event-bin "
            "(changed=0 subset)."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    summary = subparsers.add_parser(
        "summary", help="count events and blocks present in an event bin"
    )
    summary.add_argument("bin", type=Path, help="event binary file")
    summary.add_argument("--top", type=int, default=None, help="show top N blocks")
    summary.add_argument(
        "--all", action="store_true", help="show all blocks instead of top N"
    )
    summary.add_argument(
        "--sort",
        choices=["count", "addr"],
        default="count",
        help="how to order printed block rows",
    )
    summary.set_defaults(func=command_summary)

    dump_block = subparsers.add_parser(
        "dump-block",
        help=(
            "list exact events for one 16B block. With --only-unchanged, "
            "this directly answers \"the kth 0 of this block came from which "
            "site and what value\"."
        ),
    )
    dump_block.add_argument("bin", type=Path, help="event binary file")
    dump_block.add_argument(
        "--block", required=True, type=parse_int, help="block vaddr (hex or decimal)"
    )
    dump_block.add_argument(
        "--k", type=int, default=None, help="filter by block_unchanged_index"
    )
    dump_block.add_argument(
        "--ip-map",
        type=Path,
        default=None,
        help="optional taintblock16trace -m ip map for site metadata",
    )
    dump_block.add_argument(
        "--jsonl", action="store_true", help="emit one JSON object per matching event"
    )
    add_change_filter_args(dump_block)
    dump_block.set_defaults(func=command_dump_block)

    export_map = subparsers.add_parser(
        "export-map",
        help=(
            "stream exact rows for every event. Each row carries block_vaddr, "
            "compare index, unchanged index, site, before/after bytes and change "
            "flag, so this is the precise event table."
        ),
    )
    export_map.add_argument("bin", type=Path, help="event binary file")
    export_map.add_argument(
        "--ip-map",
        type=Path,
        default=None,
        help="optional taintblock16trace -m ip map for site metadata",
    )
    export_map.add_argument(
        "--format",
        choices=["tsv", "jsonl"],
        default="tsv",
        help="output format",
    )
    export_map.add_argument(
        "--output", type=Path, default=None, help="write to file instead of stdout"
    )
    export_map.add_argument(
        "--block", type=parse_int, default=None, help="optional block filter"
    )
    export_map.add_argument(
        "--k", type=int, default=None, help="optional block_unchanged_index filter"
    )
    add_change_filter_args(export_map)
    export_map.set_defaults(func=command_export_map)

    export_zero_map = subparsers.add_parser(
        "export-zero-map",
        help="compatibility alias: same as export-map --only-unchanged",
    )
    export_zero_map.add_argument("bin", type=Path, help="event binary file")
    export_zero_map.add_argument(
        "--ip-map",
        type=Path,
        default=None,
        help="optional taintblock16trace -m ip map for site metadata",
    )
    export_zero_map.add_argument(
        "--format",
        choices=["tsv", "jsonl"],
        default="tsv",
        help="output format",
    )
    export_zero_map.add_argument(
        "--output", type=Path, default=None, help="write to file instead of stdout"
    )
    export_zero_map.add_argument(
        "--block", type=parse_int, default=None, help="optional block filter"
    )
    export_zero_map.add_argument(
        "--k", type=int, default=None, help="optional block_unchanged_index filter"
    )
    export_zero_map.set_defaults(
        func=command_export_map, only_unchanged=True, only_changed=False
    )

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
