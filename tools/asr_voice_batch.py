#!/usr/bin/env python3
"""Batch ASR on all voice files with progress display, checkpointing, and ETA."""

from __future__ import annotations

import argparse
import csv
import os
import site
import sys
import time
from pathlib import Path


def add_nvidia_dll_dirs() -> None:
    if os.name != "nt":
        return
    for root in site.getsitepackages():
        nvidia_root = Path(root) / "nvidia"
        for p in list(nvidia_root.glob("*\\bin")) + list(nvidia_root.glob("*\\lib")):
            if p.exists():
                os.add_dll_directory(str(p))
                os.environ["PATH"] = f"{p}{os.pathsep}{os.environ.get('PATH', '')}"


add_nvidia_dll_dirs()
from faster_whisper import WhisperModel


def format_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        return f"{h}h{m:02d}m"


def main() -> int:
    parser = argparse.ArgumentParser(description="Batch ASR with progress and checkpointing")
    parser.add_argument("--voice-dir", type=Path, default=Path("work/extracted/voice"))
    parser.add_argument("--out", type=Path, default=Path("work/asr/asr_full.csv"))
    parser.add_argument("--model", default="large-v3-turbo")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--compute-type", default="float16")
    parser.add_argument("--checkpoint-every", type=int, default=500)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--skip-prefix", action="append", default=[], dest="skip_prefixes")
    args = parser.parse_args()

    all_files = sorted(
        [p for p in args.voice_dir.glob("*.OGG") if not p.stem.startswith("_NOVOICE")],
        key=lambda p: (p.stem.split("_")[0], int(p.stem.split("_")[1]) if "_" in p.stem and p.stem.split("_")[1].isdigit() else 0, p.stem),
    )
    if args.skip_prefixes:
        all_files = [p for p in all_files if p.stem.split("_")[0] not in args.skip_prefixes]
    if args.limit:
        all_files = all_files[: args.limit]

    total = len(all_files)
    print(f"Files to process: {total}", flush=True)
    print(f"Loading model: {args.model} ({args.device}/{args.compute_type})...", flush=True)
    t_load = time.perf_counter()
    model = WhisperModel(args.model, device=args.device, compute_type=args.compute_type)
    print(f"Model loaded in {time.perf_counter() - t_load:.1f}s", flush=True)
    print()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    resume_from = 0
    if args.out.exists():
        with args.out.open("r", encoding="utf-8-sig") as f:
            existing = sum(1 for _ in f) - 1
        if existing > 0 and existing < total:
            resume_from = existing
            print(f"Resuming from row {resume_from} (existing output has {existing} records)", flush=True)

    fieldnames = [
        "voice_file", "prefix", "file_size", "language", "language_probability",
        "asr_japanese", "model", "device", "compute_type", "error",
    ]
    write_mode = "a" if resume_from > 0 else "w"
    fh = args.out.open(write_mode, encoding="utf-8-sig", newline="")
    writer = csv.DictWriter(fh, fieldnames=fieldnames)
    if resume_from == 0:
        writer.writeheader()

    t_start = time.perf_counter()
    total_audio_s = 0.0
    errors = 0
    empty = 0

    for idx, path in enumerate(all_files[resume_from:], start=resume_from + 1):
        file_start = time.perf_counter()
        prefix = path.stem.split("_")[0]
        row = {
            "voice_file": path.name,
            "prefix": prefix,
            "file_size": path.stat().st_size,
            "language": "",
            "language_probability": "",
            "asr_japanese": "",
            "model": args.model,
            "device": args.device,
            "compute_type": args.compute_type,
            "error": "",
        }
        try:
            segments, info = model.transcribe(
                str(path), language="ja", task="transcribe",
                beam_size=5, vad_filter=False,
            )
            row["asr_japanese"] = "".join(s.text for s in segments).strip()
            row["language"] = info.language
            row["language_probability"] = f"{info.language_probability:.4f}"
            total_audio_s += info.duration
            if not row["asr_japanese"]:
                empty += 1
        except Exception as exc:
            errors += 1
            row["error"] = repr(exc)[:200]

        writer.writerow(row)

        if idx % 10 == 0 or idx == total or errors > 0:
            fh.flush()
            elapsed = time.perf_counter() - t_start
            rt = total_audio_s / elapsed if elapsed > 0 else 0
            eta = (elapsed / idx) * (total - idx) if idx > 0 else 0
            pct = idx / total * 100
            bar_len = 20
            filled = int(bar_len * idx / total)
            bar = "=" * filled + "-" * (bar_len - filled)
            err_str = f" ERR={errors}" if errors else ""
            sys.stderr.write(
                f"\r  [{bar}] {pct:5.1f}%  {idx}/{total}  "
                f"{format_time(elapsed)}/{format_time(eta)}  {rt:.1f}x{err_str}  "
            )
            sys.stderr.flush()

    fh.close()
    t_total = time.perf_counter() - t_start
    sys.stderr.write("\n")
    sys.stderr.flush()
    print(f"\nDone in {format_time(t_total)}")
    print(f"Total audio: {total_audio_s/60:.1f}min  avg speed: {total_audio_s/t_total:.1f}x")
    print(f"Errors: {errors}  Empty: {empty}")
    print(f"Output: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
