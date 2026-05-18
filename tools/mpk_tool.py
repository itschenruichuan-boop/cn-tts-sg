#!/usr/bin/env python3
"""Extract and inspect Steins;Gate MPK archives."""

from __future__ import annotations

import argparse
import json
import shutil
import struct
from dataclasses import asdict, dataclass
from pathlib import Path


ENTRY_SIZE = 0x100
ENTRY_NAME_SIZE = 0xE4
ENTRY_TABLE_PREFIX_SIZE = 0x38
ENTRY_TABLE_ABSOLUTE_OFFSET = 0x44
COPY_CHUNK_SIZE = 1024 * 1024


@dataclass(frozen=True)
class MpkEntry:
    index: int
    offset: int
    length: int
    stored_length: int
    name: str


def _read_c_string(raw: bytes) -> str:
    return raw.split(b"\0", 1)[0].decode("utf-8")


def read_entries(path: Path) -> tuple[int, list[MpkEntry]]:
    with path.open("rb") as fh:
        magic = fh.read(4)
        if magic != b"MPK\0":
            raise ValueError(f"{path} is not an MPK archive")

        fh.read(2)
        version = struct.unpack("<H", fh.read(2))[0]
        file_count = struct.unpack("<i", fh.read(4))[0]
        fh.seek(ENTRY_TABLE_ABSOLUTE_OFFSET)

        entries: list[MpkEntry] = []
        for _ in range(file_count):
            entry_raw = fh.read(ENTRY_SIZE)
            if len(entry_raw) != ENTRY_SIZE:
                raise ValueError("Unexpected end of MPK entry table")

            index = struct.unpack_from("<i", entry_raw, 0x00)[0]
            offset = struct.unpack_from("<q", entry_raw, 0x04)[0]
            length = struct.unpack_from("<q", entry_raw, 0x0C)[0]
            stored_length = struct.unpack_from("<q", entry_raw, 0x14)[0]
            name = _read_c_string(entry_raw[0x1C : 0x1C + ENTRY_NAME_SIZE])

            if not name:
                continue
            entries.append(MpkEntry(index, offset, length, stored_length, name))

        return version, entries


def _safe_output_path(root: Path, archive_name: str) -> Path:
    output_path = root / archive_name
    resolved_root = root.resolve()
    resolved_output = output_path.resolve()
    if resolved_root not in resolved_output.parents and resolved_output != resolved_root:
        raise ValueError(f"Unsafe archive path: {archive_name}")
    return output_path


def extract_archive(path: Path, output_dir: Path, limit: int | None = None) -> None:
    version, entries = read_entries(path)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"

    selected_entries = entries[:limit] if limit else entries
    with path.open("rb") as archive:
        for entry in selected_entries:
            target = _safe_output_path(output_dir, entry.name)
            target.parent.mkdir(parents=True, exist_ok=True)
            archive.seek(entry.offset)
            remaining = entry.length
            with target.open("wb") as out:
                while remaining:
                    chunk = archive.read(min(COPY_CHUNK_SIZE, remaining))
                    if not chunk:
                        raise ValueError(f"Unexpected end of archive while extracting {entry.name}")
                    out.write(chunk)
                    remaining -= len(chunk)

    manifest = {
        "source": str(path),
        "archive_name": path.name,
        "version": version,
        "file_count": len(entries),
        "extracted_count": len(selected_entries),
        "entry_table_absolute_offset": ENTRY_TABLE_ABSOLUTE_OFFSET,
        "entry_table_prefix_size": ENTRY_TABLE_PREFIX_SIZE,
        "entry_size": ENTRY_SIZE,
        "entries": [asdict(entry) for entry in entries],
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def list_archive(path: Path, limit: int) -> None:
    version, entries = read_entries(path)
    print(f"{path.name}: version={version} file_count={len(entries)}")
    for entry in entries[:limit]:
        print(f"{entry.index:05d}  offset={entry.offset:>10}  length={entry.length:>8}  {entry.name}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect or extract Steins;Gate MPK archives.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="List files inside an MPK archive.")
    list_parser.add_argument("archive", type=Path)
    list_parser.add_argument("--limit", type=int, default=20)

    extract_parser = subparsers.add_parser("extract", help="Extract files from an MPK archive.")
    extract_parser.add_argument("archive", type=Path)
    extract_parser.add_argument("output_dir", type=Path)
    extract_parser.add_argument("--limit", type=int)

    args = parser.parse_args()
    if args.command == "list":
        list_archive(args.archive, args.limit)
    elif args.command == "extract":
        extract_archive(args.archive, args.output_dir, args.limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
