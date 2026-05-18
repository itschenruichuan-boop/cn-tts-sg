#!/usr/bin/env python3
"""Build an inventory for extracted Ogg Vorbis voice files."""

from __future__ import annotations

import argparse
import csv
import struct
from collections import Counter
from pathlib import Path


def read_vorbis_info(path: Path) -> tuple[int | None, int | None, float | None]:
    data = path.read_bytes()
    sample_rate: int | None = None
    channels: int | None = None
    last_granule: int | None = None
    pos = 0

    while True:
        page = data.find(b"OggS", pos)
        if page < 0 or page + 27 > len(data):
            break

        granule = struct.unpack_from("<q", data, page + 6)[0]
        if granule >= 0:
            last_granule = granule

        segment_count = data[page + 26]
        segment_table_start = page + 27
        segment_table_end = segment_table_start + segment_count
        if segment_table_end > len(data):
            break
        payload_size = sum(data[segment_table_start:segment_table_end])
        payload_start = segment_table_end
        payload_end = payload_start + payload_size
        payload = data[payload_start:payload_end]

        marker = b"\x01vorbis"
        marker_pos = payload.find(marker)
        if marker_pos >= 0 and marker_pos + 16 <= len(payload):
            channels = payload[marker_pos + 11]
            sample_rate = struct.unpack_from("<I", payload, marker_pos + 12)[0]

        pos = payload_end

    duration = None
    if sample_rate and last_granule is not None:
        duration = last_granule / sample_rate
    return sample_rate, channels, duration


def voice_prefix(path: Path) -> str:
    stem = path.stem
    if stem.startswith("_"):
        return stem
    return stem.split("_", 1)[0]


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze extracted Steins;Gate voice OGG files.")
    parser.add_argument("voice_dir", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    prefix_counts: Counter[str] = Counter()
    prefix_seconds: Counter[str] = Counter()

    for path in sorted(args.voice_dir.glob("*.OGG")):
        sample_rate, channels, duration = read_vorbis_info(path)
        prefix = voice_prefix(path)
        prefix_counts[prefix] += 1
        if duration is not None:
            prefix_seconds[prefix] += duration
        rows.append(
            {
                "name": path.name,
                "prefix": prefix,
                "bytes": path.stat().st_size,
                "sample_rate": sample_rate or "",
                "channels": channels or "",
                "duration_seconds": f"{duration:.3f}" if duration is not None else "",
            }
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else [])
        writer.writeheader()
        writer.writerows(rows)

    with args.summary.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["prefix", "count", "total_seconds", "total_minutes"])
        writer.writeheader()
        for prefix, count in prefix_counts.most_common():
            seconds = prefix_seconds[prefix]
            writer.writerow(
                {
                    "prefix": prefix,
                    "count": count,
                    "total_seconds": f"{seconds:.3f}",
                    "total_minutes": f"{seconds / 60:.2f}",
                }
            )

    print(f"Wrote {len(rows)} voice rows to {args.out}")
    print(f"Wrote {len(prefix_counts)} prefix rows to {args.summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
