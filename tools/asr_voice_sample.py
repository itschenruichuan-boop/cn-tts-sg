#!/usr/bin/env python3
"""Run Japanese ASR on a small set of extracted voice files."""

from __future__ import annotations

import argparse
import csv
import os
import site
from pathlib import Path


def add_nvidia_dll_dirs() -> None:
    """Make NVIDIA pip-wheel DLLs visible on Windows."""
    if os.name != "nt":
        return
    candidates: list[Path] = []
    for root in site.getsitepackages():
        nvidia_root = Path(root) / "nvidia"
        candidates.extend(nvidia_root.glob("*\\bin"))
        candidates.extend(nvidia_root.glob("*\\lib"))
    for path in candidates:
        if path.exists():
            os.add_dll_directory(str(path))
            os.environ["PATH"] = f"{path}{os.pathsep}{os.environ.get('PATH', '')}"


add_nvidia_dll_dirs()

from faster_whisper import WhisperModel


def collect_samples(voice_dir: Path, prefixes: list[str], per_prefix: int) -> list[Path]:
    samples: list[Path] = []
    for prefix in prefixes:
        samples.extend(sorted(voice_dir.glob(f"{prefix}_*.OGG"))[:per_prefix])
    return samples


def collect_largest(voice_dir: Path, count: int) -> list[Path]:
    return sorted(voice_dir.glob("*.OGG"), key=lambda path: path.stat().st_size, reverse=True)[:count]


def load_model(model_name: str, device: str, compute_type: str) -> tuple[WhisperModel, str, str]:
    if device != "auto":
        return WhisperModel(model_name, device=device, compute_type=compute_type), device, compute_type

    try:
        return WhisperModel(model_name, device="cuda", compute_type="float16"), "cuda", "float16"
    except Exception as exc:
        print(f"CUDA model load failed, falling back to CPU int8: {exc}")
        return WhisperModel(model_name, device="cpu", compute_type="int8"), "cpu", "int8"


def main() -> int:
    parser = argparse.ArgumentParser(description="Transcribe sample Steins;Gate voice OGG files.")
    parser.add_argument("--voice-dir", type=Path, default=Path("work/extracted/voice"))
    parser.add_argument("--out", type=Path, default=Path("work/asr/asr_sample.csv"))
    parser.add_argument("--model", default="large-v3-turbo")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--compute-type", default="int8")
    parser.add_argument("--prefix", action="append", dest="prefixes", default=[])
    parser.add_argument("--per-prefix", type=int, default=5)
    parser.add_argument("--top-largest", type=int, default=0)
    args = parser.parse_args()

    if args.top_largest:
        samples = collect_largest(args.voice_dir, args.top_largest)
    else:
        prefixes = args.prefixes or ["OKA", "MAY", "CRS"]
        samples = collect_samples(args.voice_dir, prefixes, args.per_prefix)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    model, device, compute_type = load_model(args.model, args.device, args.compute_type)
    rows = []
    for index, path in enumerate(samples, start=1):
        print(f"[{index}/{len(samples)}] {path.name}", flush=True)
        row = {
            "voice_file": path.name,
            "prefix": path.stem.split("_", 1)[0],
            "file_size": path.stat().st_size,
            "language": "",
            "language_probability": "",
            "asr_japanese": "",
            "model": args.model,
            "device": device,
            "compute_type": compute_type,
            "error": "",
        }
        try:
            segments, info = model.transcribe(
                str(path),
                language="ja",
                task="transcribe",
                beam_size=5,
                vad_filter=False,
            )
            row["asr_japanese"] = "".join(segment.text for segment in segments).strip()
            row["language"] = info.language
            row["language_probability"] = f"{info.language_probability:.4f}"
        except Exception as exc:
            row["error"] = repr(exc)
            print(f"  ERROR: {exc!r}", flush=True)
        rows.append(row)

    with args.out.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else [])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
