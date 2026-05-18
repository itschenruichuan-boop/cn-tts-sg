#!/usr/bin/env python3
"""Batch TTS synthesis with checkpointing, progress, and smart skipping.

Usage:
  cd C:\opencode\SG-TTS\index-tts
  uv run python C:\opencode\SG-TTS\tools\tts_batch.py
"""

import csv, io, json, os, re, sys, time
from pathlib import Path

os.chdir(r"C:\opencode\SG-TTS\index-tts")
sys.path.insert(0, ".")
from indextts.infer_v2 import IndexTTS2

# --- Config ---
MATCH_CSV = r"C:\opencode\SG-TTS\work\matches\asr_full_v5_best.csv"
VOICE_DIR = r"C:\opencode\SG-TTS\work\extracted\voice"
OUT_DIR = r"C:\opencode\SG-TTS\work\tts_output"
SPEAKER_MAP = r"C:\opencode\SG-TTS\config\voice_speaker_map.json"
LOG_FILE = r"C:\opencode\SG-TTS\work\tts_output\_completed.txt"

# Skip: prologue, low confidence, tiny prefixes
PROLOGUE_BOUNDS = {"OKA": 26}
SKIP_PREFIXES = {
    "STJ", "VFA", "MIX", "VFB", "VFC", "MCA", "CJO", "KYJ", "RNC",
    "KAN", "STA", "STU", "VFD", "RNB", "RND", "BOY", "FAN", "INA",
    "INB", "INC", "KNA", "KNB", "KNC", "VFE", "VFF", "MAS",
}
MIN_SCORE = 0.55

# Pinyin dictionary for proper nouns (avoids word segmentation issues)
PINYIN_MAP = {
    "秋叶原": "QIU1 YE4 YUAN2",
    "秋葉原": "QIU1 YE4 YUAN2",
    "秋叶": "QIU1 YE4",
    "秋葉": "QIU1 YE4",
}

# Traditional to simplified Chinese (common S;G script variants)
T2S_MAP = {
    "脫": "脱", "論": "论", "麼": "么", "瞭": "了", "麵": "面",
    "裡": "里", "後": "后", "禦": "御", "穀": "谷", "髮": "发",
    "乾": "干", "鬆": "松", "繫": "系", "捨": "舍", "採": "采",
    "週": "周", "準": "准", "遊": "游", "徵": "征", "衝": "冲",
}

# Characters to strip (not in IndexTTS2 BPE vocab, or produce bad results)
STRIP_CHARS = "〜♪『』・⋯⋯「」『』––—"


def clean_text(text: str) -> str:
    # Strip unsupported chars
    for ch in STRIP_CHARS:
        text = text.replace(ch, "")
    # Traditional → Simplified
    for trad, simp in T2S_MAP.items():
        text = text.replace(trad, simp)
    # Pinyin for proper nouns (must be after t2s since some nouns are in trad)
    for word, pinyin in PINYIN_MAP.items():
        text = text.replace(word, pinyin)
    return text.strip()


def parse_voice_number(stem: str) -> tuple[str, int, int]:
    m = re.match(r"^([A-Za-z_]+)_(\d+)(?:_(\d+))?$", stem)
    if not m:
        return (stem, 0, 0)
    return (m.group(1), int(m.group(2)), int(m.group(3) or 0))


def is_prologue(vf: str) -> bool:
    stem = Path(vf).stem
    prefix, num, _ = parse_voice_number(stem)
    return num <= PROLOGUE_BOUNDS.get(prefix, -1)


def format_time(s: float) -> str:
    if s < 60:
        return f"{s:.0f}s"
    if s < 3600:
        return f"{s / 60:.1f}m"
    return f"{s / 3600:.1f}h"


def main():
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

    match_rows = list(csv.DictReader(open(MATCH_CSV, encoding="utf-8-sig")))
    completed = set()
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            completed = set(line.strip() for line in f if line.strip())

    # Filter: skip what we decided to skip
    todo = []
    skipped = {"prologue": 0, "low_score": 0, "tiny_prefix": 0, "completed": 0, "no_ref": 0}
    for r in match_rows:
        vf = r["voice_file"]
        if vf in completed:
            skipped["completed"] += 1
            continue
        prefix = r["prefix"]
        score = float(r["combined_score"])
        script_file = r.get("script_file", "ZZZ.SCX")

        if is_prologue(vf):
            skipped["prologue"] += 1
            continue
        if score < MIN_SCORE:
            skipped["low_score"] += 1
            continue
        if prefix in SKIP_PREFIXES:
            skipped["tiny_prefix"] += 1
            continue

        ref_path = os.path.join(VOICE_DIR, vf)
        if not os.path.exists(ref_path):
            skipped["no_ref"] += 1
            continue

        todo.append((vf, ref_path, clean_text(r["matched_chinese"]), prefix, score, script_file))

    # Sort by chapter order so SG01 processes first
    todo.sort(key=lambda x: x[5])  # x[5] = script_file

    total = len(todo)
    print(f"Total match rows: {len(match_rows)}")
    print(f"Skipped: {skipped}")
    print(f"To synthesize: {total}")
    if total == 0:
        print("All done!")
        return
    print()

    # Load TTS model once
    print("Loading IndexTTS2 model...", flush=True)
    tts = IndexTTS2(cfg_path="checkpoints/config.yaml", model_dir="checkpoints",
                    use_fp16=True, use_cuda_kernel=False)
    print("Model loaded.\n", flush=True)

    os.makedirs(OUT_DIR, exist_ok=True)
    log_fh = open(LOG_FILE, "a", encoding="utf-8")

    t_start = time.perf_counter()
    done_count = 0
    errors = 0

    for idx, (vf, ref_path, text, prefix, score, script_file) in enumerate(todo, 1):
        out_path = os.path.join(OUT_DIR, vf.replace(".OGG", ".wav"))
        if os.path.exists(out_path) and os.path.getsize(out_path) > 1000:
            done_count += 1
            completed.add(vf)
            log_fh.write(vf + "\n")
            log_fh.flush()
            continue

        t0 = time.perf_counter()
        try:
            tts.infer(spk_audio_prompt=ref_path, text=text, emo_audio_prompt=ref_path,
                      emo_alpha=1.0, output_path=out_path, verbose=False)
            t_elapsed = time.perf_counter() - t0
            done_count += 1
            errors_this = 0
        except Exception as exc:
            t_elapsed = time.perf_counter() - t0
            errors += 1
            errors_this = 1
            print(f"  [{idx}/{total}] {vf} ERROR: {exc!r}", flush=True)

        completed.add(vf)
        log_fh.write(vf + "\n")
        log_fh.flush()

        if idx % 50 == 0 or idx == total or errors_this:
            elapsed_all = time.perf_counter() - t_start
            avg_time = elapsed_all / done_count if done_count > 0 else 0
            eta = avg_time * (total - idx)
            bar_len = 30
            filled = int(bar_len * idx / total)
            bar = "=" * filled + "-" * (bar_len - filled)
            pct = idx / total * 100
            err_str = f" ERR={errors}" if errors else ""
            sys.stderr.write(
                f"\r  [{bar}] {pct:5.1f}%  {idx}/{total}  "
                f"{format_time(elapsed_all)}/{format_time(eta)}  "
                f"{t_elapsed:.1f}s/file{err_str}  "
            )
            sys.stderr.flush()

        if idx % 500 == 0:
            print(f"\n  Checkpoint at {idx}/{total}, {format_time(time.perf_counter() - t_start)} elapsed", flush=True)

    log_fh.close()
    sys.stderr.write("\n")
    sys.stderr.flush()
    t_total = time.perf_counter() - t_start
    print(f"\nDone in {format_time(t_total)}")
    print(f"Files: {done_count}  Errors: {errors}")
    print(f"Output: {OUT_DIR}")


if __name__ == "__main__":
    main()
