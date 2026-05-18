#!/usr/bin/env python3
"""Download public Hugging Face model files through hf-mirror.com.

This intentionally avoids huggingface_hub's metadata HEAD requests, because
some mirrors redirect those in a way that the official client rejects.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import sys
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen


DEFAULT_ENDPOINT = "https://hf-mirror.com"


def read_json(url: str) -> dict:
    req = Request(url, headers={"User-Agent": "sg-tts-hf-mirror-downloader"})
    with urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))


def matches_any(path: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatch(path, pattern) for pattern in patterns)


def select_files(files: list[str], includes: list[str], excludes: list[str]) -> list[str]:
    selected = []
    for file_name in files:
        if includes and not matches_any(file_name, includes):
            continue
        if excludes and matches_any(file_name, excludes):
            continue
        selected.append(file_name)
    return selected


def format_size(num: int | None) -> str:
    if num is None:
        return "unknown"
    size = float(num)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def download_file(url: str, dest: Path, retries: int = 3) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    existing = tmp.stat().st_size if tmp.exists() else 0

    headers = {"User-Agent": "sg-tts-hf-mirror-downloader"}
    if existing:
        headers["Range"] = f"bytes={existing}-"

    for attempt in range(1, retries + 1):
        try:
            req = Request(url, headers=headers)
            with urlopen(req, timeout=120) as resp, tmp.open("ab" if existing else "wb") as out:
                total_header = resp.headers.get("Content-Length")
                total = int(total_header) + existing if total_header and existing else (
                    int(total_header) if total_header else None
                )
                downloaded = existing
                last_print = time.monotonic()
                while True:
                    chunk = resp.read(1024 * 1024)
                    if not chunk:
                        break
                    out.write(chunk)
                    downloaded += len(chunk)
                    now = time.monotonic()
                    if now - last_print >= 2:
                        print(f"  {dest.name}: {format_size(downloaded)} / {format_size(total)}")
                        last_print = now
            tmp.replace(dest)
            return
        except (HTTPError, URLError, TimeoutError) as exc:
            if attempt == retries:
                raise
            print(f"  retry {attempt}/{retries - 1}: {exc}")
            time.sleep(2 * attempt)


def main() -> int:
    parser = argparse.ArgumentParser(description="Download a public Hugging Face repo via hf-mirror.com")
    parser.add_argument("repo_id", help="For example: BAAI/bge-m3")
    parser.add_argument("--local-dir", required=True, type=Path)
    parser.add_argument("--endpoint", default=os.environ.get("HF_ENDPOINT", DEFAULT_ENDPOINT))
    parser.add_argument("--include", action="append", default=[], help="Glob pattern; can be repeated")
    parser.add_argument("--exclude", action="append", default=[], help="Glob pattern; can be repeated")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    endpoint = args.endpoint.rstrip("/")
    api_url = f"{endpoint}/api/models/{args.repo_id}"
    info = read_json(api_url)
    sha = info.get("sha") or "main"
    siblings = [item["rfilename"] for item in info.get("siblings", []) if "rfilename" in item]
    files = select_files(siblings, args.include, args.exclude)

    if not files:
        print("No files selected.", file=sys.stderr)
        return 2

    print(f"{args.repo_id} @ {sha}")
    print(f"Selected {len(files)} / {len(siblings)} files")
    for file_name in files:
        print(f"  {file_name}")

    if args.dry_run:
        return 0

    args.local_dir.mkdir(parents=True, exist_ok=True)
    for index, file_name in enumerate(files, 1):
        dest = args.local_dir / file_name
        if dest.exists() and not args.force:
            print(f"[{index}/{len(files)}] skip existing {file_name}")
            continue
        encoded = "/".join(quote(part) for part in file_name.split("/"))
        url = f"{endpoint}/{args.repo_id}/resolve/{sha}/{encoded}"
        print(f"[{index}/{len(files)}] download {file_name}")
        download_file(url, dest)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
