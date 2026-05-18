#!/usr/bin/env python3
"""Best-effort string table dumper for MAGES. SC3/SCX scripts."""

from __future__ import annotations

import argparse
import json
import struct
from dataclasses import asdict, dataclass
from pathlib import Path


TERMINATOR = 0xFF


@dataclass(frozen=True)
class ScxString:
    source: str
    ordinal: int
    offset: int
    text: str
    unknown_codes: list[str]


def load_charset(path: Path) -> list[str]:
    data = path.read_text(encoding="utf-8")
    return list(data.rstrip("\n"))


def load_compound_map(path: Path | None) -> dict[str, str]:
    if path is None or not path.exists():
        return {}

    mapping: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            key, value = line.split("=", 1)
        elif "\t" in line:
            key, value = line.split("\t", 1)
        else:
            continue
        mapping[key] = value
    return mapping


def decode_string(data: bytes, offset: int, charset: list[str], compound: dict[str, str]) -> tuple[str, list[str]]:
    pos = offset
    out: list[str] = []
    unknown: list[str] = []

    while pos < len(data):
        token = data[pos]
        pos += 1
        if token == TERMINATOR:
            break

        if token >= 0x80:
            if pos >= len(data):
                unknown.append(f"truncated:{token:02X}")
                break
            code = (token << 8) | data[pos]
            pos += 1
            index = code & 0x7FFF
            if index < len(charset):
                ch = charset[index]
                out.append(compound.get(ch, ch))
            else:
                unknown.append(f"0x{code:04X}")
                out.append(f"<0x{code:04X}>")
        else:
            # SC3 strings can contain inline formatting/wait/name commands.
            # For first-pass text inspection, keep visible markers and let the
            # surrounding high-bit character tokens continue decoding.
            out.append(f"<{token:02X}>")

    return "".join(out), unknown


def dump_file(path: Path, charset: list[str], compound: dict[str, str]) -> list[ScxString]:
    data = path.read_bytes()
    if data[:4] != b"SC3\0":
        raise ValueError(f"{path} is not an SC3 script")

    string_table_offset = struct.unpack_from("<I", data, 4)[0]
    return_table_offset = struct.unpack_from("<I", data, 8)[0]
    if not (0 <= string_table_offset <= return_table_offset <= len(data)):
        raise ValueError(f"{path} has unexpected table offsets")

    count = (return_table_offset - string_table_offset) // 4
    strings: list[ScxString] = []
    seen: set[int] = set()
    for ordinal in range(count):
        pointer = struct.unpack_from("<I", data, string_table_offset + ordinal * 4)[0]
        if pointer == 0 or pointer >= len(data) or pointer in seen:
            continue
        seen.add(pointer)
        text, unknown = decode_string(data, pointer, charset, compound)
        if text:
            strings.append(ScxString(path.name, ordinal, pointer, text, unknown))
    return strings


def main() -> int:
    parser = argparse.ArgumentParser(description="Dump SCX string-table text using a charset file.")
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--charset", type=Path, required=True)
    parser.add_argument("--compound-map", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    charset = load_charset(args.charset)
    compound = load_compound_map(args.compound_map)

    all_strings: list[ScxString] = []
    for input_path in args.inputs:
        paths = sorted(input_path.parent.glob(input_path.name)) if any(ch in input_path.name for ch in "*?[]") else [input_path]
        for path in paths:
            all_strings.extend(dump_file(path, charset, compound))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.suffix.lower() == ".json":
        args.out.write_text(json.dumps([asdict(s) for s in all_strings], ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        lines = ["source\tordinal\toffset\ttext\tunknown_codes"]
        for item in all_strings:
            lines.append(
                f"{item.source}\t{item.ordinal}\t{item.offset}\t"
                f"{item.text.replace(chr(9), ' ')}\t{','.join(item.unknown_codes)}"
            )
        args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Dumped {len(all_strings)} strings to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
