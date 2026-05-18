#!/usr/bin/env python3
"""Pack final_voice OGG files back into voice.mpk."""
import json, os, struct, sys
from pathlib import Path

WORK_DIR = r"C:\opencode\SG-TTS"
MANIFEST_PATH = os.path.join(WORK_DIR, "work", "extracted", "voice", "manifest.json")
FINAL_VOICE = os.path.join(WORK_DIR, "work", "final_voice")
OUTPUT_MPK = os.path.join(WORK_DIR, "work", "final_voice.mpk")

ENTRY_SIZE = 0x100
ENTRY_NAME_SIZE = 0xE4
HEADER_SIZE = 0x44
ENTRY_TABLE_START = 0x44

with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
    manifest = json.load(f)

version = manifest["version"]
entries = manifest["entries"]
total = len(entries)

# Build entry table and calculate offsets
entry_data_start = ENTRY_TABLE_START + total * ENTRY_SIZE

packed_entries = []
current_offset = entry_data_start

for entry in entries:
    name = entry["name"]
    ogg_path = os.path.join(FINAL_VOICE, name)
    if os.path.exists(ogg_path):
        size = os.path.getsize(ogg_path)
    else:
        print(f"WARNING: {name} not found in final_voice, skipping")
        size = 0
    packed_entries.append({
        "index": entry["index"],
        "path": ogg_path if os.path.exists(ogg_path) else None,
        "name": name,
        "offset": current_offset,
        "length": size,
        "stored_length": size,
    })
    current_offset += size

total_data_size = current_offset - entry_data_start
print(f"Entries: {total}")
print(f"Entry table: {ENTRY_TABLE_START} + {total * ENTRY_SIZE} = {entry_data_start}")
print(f"Total file data: {total_data_size / 1024 / 1024:.1f} MB")
print(f"Writing to: {OUTPUT_MPK}")

with open(OUTPUT_MPK, "wb") as f:
    # Header
    f.write(b"MPK\0")
    f.write(struct.pack("<H", 0))   # unknown 2 bytes
    f.write(struct.pack("<H", version))
    f.write(struct.pack("<i", total))
    # Pad header to HEADER_SIZE
    while f.tell() < HEADER_SIZE:
        f.write(b"\0")

    # Entry table
    for entry in packed_entries:
        name_bytes = entry["name"].encode("utf-8")[:ENTRY_NAME_SIZE - 1] + b"\0"
        name_bytes = name_bytes.ljust(ENTRY_NAME_SIZE, b"\0")
        row = struct.pack("<i", entry["index"])
        row += struct.pack("<q", entry["offset"])
        row += struct.pack("<q", entry["length"])
        row += struct.pack("<q", entry["stored_length"])
        row += name_bytes
        row = row.ljust(ENTRY_SIZE, b"\0")
        f.write(row)

    # Ensure we're at the right position
    assert f.tell() == entry_data_start, f"Expected {entry_data_start}, got {f.tell()}"

    # Write file data
    written = 0
    for entry in packed_entries:
        if entry["path"]:
            with open(entry["path"], "rb") as src:
                data = src.read()
                f.write(data)
                written += len(data)

    print(f"Written: {written / 1024 / 1024:.1f} MB")

# Verify
actual_size = os.path.getsize(OUTPUT_MPK)
print(f"Output: {OUTPUT_MPK} ({actual_size / 1024 / 1024:.1f} MB)")
