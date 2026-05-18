#!/usr/bin/env python3
"""Build a searchable Chinese script corpus from extracted SC3 text dumps."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


NAME_LINE_RE = re.compile(r"^\[name\](?P<speaker>.*?)\[line\](?P<text>.*)$")
TAG_RE = re.compile(r"\[(?:/?[A-Za-z0-9_-]+)(?:\s+[^\]]*)?\]")


def clean_text(text: str) -> str:
    text = text.replace("[linebreak]", "，")
    text = text.replace("[%p]", "")
    text = re.sub(r"\[color index=\"[0-9A-Fa-f]+\"\]", "", text)
    text = TAG_RE.sub("", text)
    text = text.replace("　", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def iter_lines(txt_dir: Path):
    record_id = 0
    for path in sorted(txt_dir.glob("*.txt")):
        script_file = path.name.removesuffix(".txt")
        for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            raw = raw.strip()
            if not raw:
                continue
            speaker = ""
            text_type = "narration"
            match = NAME_LINE_RE.match(raw)
            if match:
                speaker = clean_text(match.group("speaker"))
                text = clean_text(match.group("text"))
                text_type = "dialogue"
            else:
                text = clean_text(raw)
            if not text:
                continue
            record_id += 1
            yield {
                "record_id": f"cn_{record_id:06d}",
                "script_file": script_file,
                "line_no": line_no,
                "text_type": text_type,
                "speaker": speaker,
                "text": text,
                "raw": raw,
            }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--txt-dir", type=Path, default=Path("work/extracted/cnscript/txt"))
    parser.add_argument("--out-csv", type=Path, default=Path("work/analysis/cn_corpus.csv"))
    parser.add_argument("--out-jsonl", type=Path, default=Path("work/analysis/cn_corpus.jsonl"))
    args = parser.parse_args()

    rows = list(iter_lines(args.txt_dir))
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["record_id", "script_file", "line_no", "text_type", "speaker", "text", "raw"]
    with args.out_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with args.out_jsonl.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    dialogue_count = sum(1 for row in rows if row["text_type"] == "dialogue")
    print(f"Wrote {len(rows)} rows to {args.out_csv}")
    print(f"Dialogue rows: {dialogue_count}; narration rows: {len(rows) - dialogue_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
