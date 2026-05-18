#!/usr/bin/env python3
"""Install local NVIDIA CUDA runtime wheels and verify DLL discovery."""

from __future__ import annotations

import argparse
import os
import site
import subprocess
import sys
from pathlib import Path


REQUIRED_PREFIXES = [
    "nvidia_cublas_cu12-",
    "nvidia_cudnn_cu12-",
    "nvidia_cuda_nvrtc_cu12-",
]


def find_wheels(wheel_dir: Path) -> list[Path]:
    wheels = sorted(wheel_dir.glob("*.whl"))
    selected: list[Path] = []
    missing: list[str] = []
    for prefix in REQUIRED_PREFIXES:
        matches = [path for path in wheels if path.name.startswith(prefix)]
        if not matches:
            missing.append(prefix)
        else:
            selected.append(matches[-1])
    if missing:
        raise SystemExit(
            "Missing wheel(s):\n"
            + "\n".join(f"  {prefix}*.whl" for prefix in missing)
            + f"\nPut them in {wheel_dir}"
        )
    return selected


def nvidia_dll_dirs() -> list[Path]:
    dirs: list[Path] = []
    for root in site.getsitepackages():
        nvidia_root = Path(root) / "nvidia"
        dirs.extend(path for path in nvidia_root.glob("*\\bin") if path.exists())
        dirs.extend(path for path in nvidia_root.glob("*\\lib") if path.exists())
    return dirs


def main() -> int:
    parser = argparse.ArgumentParser(description="Install local NVIDIA CUDA runtime wheels.")
    parser.add_argument("--wheel-dir", type=Path, default=Path("downloads/cuda-wheels"))
    args = parser.parse_args()

    wheels = find_wheels(args.wheel_dir)
    subprocess.run([sys.executable, "-m", "pip", "install", *map(str, wheels)], check=True)

    print("NVIDIA DLL directories:")
    for path in nvidia_dll_dirs():
        print(f"  {path}")

    print("Key DLLs found:")
    for dll in ["cublas64_12.dll", "cudnn_ops64_9.dll", "cudnn_cnn64_9.dll"]:
        found = []
        for path in nvidia_dll_dirs():
            found.extend(path.glob(dll))
        print(f"  {dll}: {found[0] if found else 'MISSING'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
