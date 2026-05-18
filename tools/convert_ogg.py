#!/usr/bin/env python3
"""Convert synthesized WAV → OGG (48000Hz mono Vorbis) and prepare final voice directory."""

import os, subprocess, sys, time
from pathlib import Path

WORK_DIR = r"C:\opencode\SG-TTS"
TTS_DIR = os.path.join(WORK_DIR, "work", "tts_output")
ORIG_VOICE = os.path.join(WORK_DIR, "work", "extracted", "voice")
FINAL_VOICE = os.path.join(WORK_DIR, "work", "final_voice")
LOG_FILE = os.path.join(TTS_DIR, "_completed.txt")
MATCH_CSV = os.path.join(WORK_DIR, "work", "matches", "asr_full_v5_best.csv")
SKIP_LOG = os.path.join(TTS_DIR, "_skipped.txt")

# Find ffmpeg from imageio_ffmpeg
import imageio_ffmpeg
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
print(f"ffmpeg: {FFMPEG}")


def wav_to_ogg(wav_path: str, ogg_path: str, sample_rate: int = 48000):
    """Convert WAV to OGG Vorbis at specified sample rate, mono."""
    ogg_path = os.path.abspath(ogg_path)
    os.makedirs(os.path.dirname(ogg_path), exist_ok=True)
    cmd = [
        FFMPEG, "-y", "-i", wav_path,
        "-ar", str(sample_rate), "-ac", "1",
        "-c:a", "libvorbis", "-q:a", "4",
        ogg_path,
    ]
    subprocess.run(cmd, capture_output=True, check=True)


def main():
    if not os.path.exists(LOG_FILE):
        print("No _completed.txt - nothing to convert")
        return

    with open(LOG_FILE, "r", encoding="utf-8") as f:
        completed = set(line.strip() for line in f if line.strip())

    print(f"Files to convert: {len(completed)}")

    # Build skip list: original files that were NOT synthesized
    import csv
    match_rows = list(csv.DictReader(open(MATCH_CSV, encoding="utf-8-sig")))
    all_voice_files = set(r["voice_file"] for r in match_rows)

    os.makedirs(FINAL_VOICE, exist_ok=True)
    converted = 0
    copied = 0
    errors = 0

    for vf in sorted(completed):
        wav_path = os.path.join(TTS_DIR, vf.replace(".OGG", ".wav"))
        ogg_path = os.path.join(FINAL_VOICE, vf)
        if os.path.exists(ogg_path) and os.path.getsize(ogg_path) > 100:
            converted += 1
            continue
        if not os.path.exists(wav_path):
            print(f"  MISSING WAV: {vf}")
            errors += 1
            continue
        try:
            wav_to_ogg(wav_path, ogg_path)
            converted += 1
        except Exception as e:
            print(f"  ERROR {vf}: {e}")
            errors += 1

        if converted % 200 == 0:
            print(f"  Converted: {converted}/{len(completed)}")

    # Copy original OGG for skipped files
    for vf in sorted(all_voice_files):
        ogg_path = os.path.join(FINAL_VOICE, vf)
        if os.path.exists(ogg_path):
            continue
        orig_path = os.path.join(ORIG_VOICE, vf)
        if os.path.exists(orig_path):
            import shutil
            shutil.copy2(orig_path, ogg_path)
            copied += 1

    print(f"\nDone! Converted: {converted}, Copied(originals): {copied}, Errors: {errors}")
    print(f"Final voice directory: {FINAL_VOICE}")
    print(f"Total OGG files: {len(list(Path(FINAL_VOICE).glob('*.OGG')))}")


if __name__ == "__main__":
    main()
