#!/usr/bin/env python3
"""Match JA ASR → CN script with position-aware scoring + quality checks.

v3: Adds prologue boundary, role mismatch detection, sequence outlier flags,
    and a review CSV for manual inspection of suspicious matches.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer


_VOICE_NUM_RE = re.compile(r"^([A-Za-z_]+)_(\d+)(?:_(\d+))?$")


def parse_voice_number(stem: str) -> tuple[str, int, int]:
    m = _VOICE_NUM_RE.match(stem)
    if not m:
        return (stem, 0, 0)
    return (m.group(1), int(m.group(2)), int(m.group(3) or 0))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_speaker_map(path: Path) -> dict[str, list[str]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return {str(prefix): [str(s) for s in speakers] for prefix, speakers in data.items()}


def build_voice_positions(asr_rows: list[dict[str, str]]) -> dict[str, float]:
    prefix_groups: dict[str, list[tuple[int, int, str]]] = defaultdict(list)
    for row in asr_rows:
        vf = row.get("voice_file", "")
        stem = Path(vf).stem
        prefix, num, sub = parse_voice_number(stem)
        prefix_groups[prefix].append((num, sub, vf))
    positions: dict[str, float] = {}
    for prefix, items in prefix_groups.items():
        items.sort(key=lambda x: (x[0], x[1]))
        total = len(items)
        for idx, (_, _, vf) in enumerate(items):
            positions[vf] = idx / max(total - 1, 1)
    return positions


def build_corpus_positions(
    corpus_rows: list[dict[str, str]],
    text_type: str,
) -> tuple[dict[str, list[tuple[int, int, float]]], dict[int, int]]:
    script_files = sorted(set(row.get("script_file", "") for row in corpus_rows))
    script_order = {sf: i for i, sf in enumerate(script_files)}
    global_order: dict[int, int] = {}
    for i in range(len(corpus_rows)):
        sf = corpus_rows[i].get("script_file", "")
        ln = int(corpus_rows[i].get("line_no", 0))
        global_order[i] = script_order.get(sf, 0) * 100000 + ln

    speaker_lines: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for i in range(len(corpus_rows)):
        if text_type == "dialogue" and corpus_rows[i].get("text_type") != "dialogue":
            continue
        if text_type == "narration" and corpus_rows[i].get("text_type") != "narration":
            continue
        speaker = corpus_rows[i].get("speaker", "")
        speaker_lines[speaker].append((i, global_order[i]))

    speaker_positions: dict[str, list[tuple[int, int, float]]] = {}
    for speaker, lines in speaker_lines.items():
        lines.sort(key=lambda x: x[1])
        total = len(lines)
        fracs: list[tuple[int, int, float]] = []
        for idx, (corpus_idx, gpos) in enumerate(lines):
            frac = idx / max(total - 1, 1)
            fracs.append((corpus_idx, gpos, frac))
        speaker_positions[speaker] = fracs
    return speaker_positions, global_order


def build_speaker_frac_lookup(
    speaker_positions: dict[str, list[tuple[int, int, float]]],
) -> dict[int, float]:
    lookup: dict[int, float] = {}
    for _, lines in speaker_positions.items():
        for corpus_idx, _, frac in lines:
            lookup[corpus_idx] = frac
    return lookup


def encode_texts(
    model: SentenceTransformer,
    texts: list[str],
    batch_size: int,
    cache_path: Path | None = None,
    rebuild: bool = False,
) -> np.ndarray:
    if cache_path and cache_path.exists() and not rebuild:
        data = np.load(cache_path, allow_pickle=False)
        embeddings = data["embeddings"]
        if embeddings.shape[0] == len(texts):
            print(f"Loaded cached embeddings: {cache_path}", flush=True)
            return embeddings
        print("Embedding cache row count mismatch; rebuilding.", flush=True)
    embeddings = model.encode(
        texts, batch_size=batch_size, convert_to_numpy=True,
        normalize_embeddings=True, show_progress_bar=True,
    ).astype(np.float32, copy=False)
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, embeddings=embeddings)
        print(f"Saved embeddings: {cache_path}", flush=True)
    return embeddings


def is_prologue(voice_file: str, prologue_bounds: dict[str, int]) -> bool:
    stem = Path(voice_file).stem
    prefix, num, _ = parse_voice_number(stem)
    max_num = prologue_bounds.get(prefix)
    return max_num is not None and num <= max_num


def detect_flags(
    row: dict[str, object],
    speaker_map: dict[str, list[str]],
    corpus_speakers: np.ndarray,
    prev_match_gpos: dict[str, int],
    prev_match_ci: dict[str, int],
    global_order: dict[int, int],
) -> list[str]:
    flags: list[str] = []
    score = float(row.get("combined_score", 0))
    if score < 0.60:
        flags.append("LOW_CONFIDENCE")

    prefix = str(row.get("prefix", ""))
    matched_speaker = str(row.get("speaker", ""))
    allowed = speaker_map.get(prefix, [])
    if allowed and matched_speaker and matched_speaker not in allowed:
        flags.append("ROLE_MISMATCH")

    if prefix in prev_match_gpos:
        ci = int(row.get("_corpus_idx", -1))
        curr_gpos = global_order.get(ci, 0)
        prev_gpos = prev_match_gpos[prefix]
        if curr_gpos > 0 and prev_gpos > 0:
            if curr_gpos < prev_gpos:
                flags.append("SEQ_BACKWARD")
            elif curr_gpos - prev_gpos > 500000:
                flags.append("SEQ_LARGE_JUMP")

    prev_match_gpos[prefix] = global_order.get(
        int(row.get("_corpus_idx", -1)), 0
    )

    return flags


def main() -> int:
    parser = argparse.ArgumentParser(description="Match JA ASR → CN script v3 (with quality flags)")
    parser.add_argument("--asr-csv", type=Path, required=True)
    parser.add_argument("--corpus-csv", type=Path, default=Path("work/analysis/cn_corpus.csv"))
    parser.add_argument("--model-dir", type=Path, default=Path("models/bge-m3"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--out-review", type=Path)
    parser.add_argument("--cache", type=Path, default=Path("work/index/cn_corpus_bge_m3_torch.npz"))
    parser.add_argument("--query-column", default="asr_japanese")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--text-type", choices=["all", "dialogue", "narration"], default="all")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--speaker-map", type=Path, default=Path("config/voice_speaker_map.json"))
    parser.add_argument("--speaker-mode", choices=["filter", "boost"], default="filter")
    parser.add_argument("--speaker-bonus", type=float, default=0.08)
    parser.add_argument("--position-weight", type=float, default=0.15)
    parser.add_argument("--script-glob", default="SG*.SCX")
    parser.add_argument("--no-script-filter", action="store_true")
    parser.add_argument("--prologue-bounds", default="OKA:26")
    parser.add_argument("--min-score", type=float, default=0.55)
    parser.add_argument("--rebuild-index", action="store_true")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but torch.cuda.is_available() is False")

    prologue_bounds: dict[str, int] = {}
    for item in args.prologue_bounds.split(","):
        item = item.strip()
        if ":" in item:
            prefix, max_n = item.split(":", 1)
            prologue_bounds[prefix.strip()] = int(max_n.strip())

    print(f"torch: {torch.__version__}", flush=True)
    print(f"cuda available: {torch.cuda.is_available()}", flush=True)
    if torch.cuda.is_available():
        print(f"gpu: {torch.cuda.get_device_name(0)}", flush=True)

    asr_rows = [row for row in read_csv(args.asr_csv) if row.get(args.query_column, "").strip()]
    corpus_rows = read_csv(args.corpus_csv)

    if args.text_type == "dialogue":
        corpus_rows = [row for row in corpus_rows if row.get("text_type") == "dialogue"]
    elif args.text_type == "narration":
        corpus_rows = [row for row in corpus_rows if row.get("text_type") == "narration"]

    if not args.no_script_filter and args.script_glob:
        import fnmatch as _fnmatch
        corpus_rows = [
            row for row in corpus_rows
            if _fnmatch.fnmatch(row.get("script_file", ""), args.script_glob)
        ]
        print(f"Script filter '{args.script_glob}': kept {len(corpus_rows)} rows", flush=True)

    if not asr_rows:
        raise SystemExit(f"No ASR rows found in {args.asr_csv}")
    if not corpus_rows:
        raise SystemExit(f"No corpus rows found")

    speaker_map = load_speaker_map(args.speaker_map)
    corpus_speakers = np.array([row.get("speaker", "") for row in corpus_rows])

    voice_positions = build_voice_positions(asr_rows)
    speaker_positions, global_order = build_corpus_positions(corpus_rows, args.text_type)
    speaker_frac_lookup = build_speaker_frac_lookup(speaker_positions)

    print(f"ASR rows: {len(asr_rows)}", flush=True)
    print(f"Corpus rows: {len(corpus_rows)}", flush=True)
    print(f"Prologue bounds: {prologue_bounds}", flush=True)
    print(f"Position weight: {args.position_weight}", flush=True)
    print(f"Loading model: {args.model_dir}", flush=True)

    model = SentenceTransformer(str(args.model_dir), device=args.device)
    corpus_embeddings = encode_texts(
        model, [row["text"] for row in corpus_rows],
        args.batch_size, args.cache, args.rebuild_index,
    )
    query_embeddings = encode_texts(
        model, [row[args.query_column] for row in asr_rows],
        args.batch_size,
    )

    scores = query_embeddings @ corpus_embeddings.T
    out_rows: list[dict[str, object]] = []
    prev_match_gpos: dict[str, int] = {}
    prev_match_ci: dict[str, int] = {}

    for query_idx, asr_row in enumerate(asr_rows):
        prefix = asr_row.get("prefix", "")
        voice_file = asr_row.get("voice_file", "")
        allowed_speakers = speaker_map.get(prefix, [])

        score_row = scores[query_idx].copy()
        speaker_scope = "all"

        allowed_set = set(allowed_speakers)
        is_narration = np.array([r.get("text_type") == "narration" for r in corpus_rows])
        if allowed_set:
            speaker_mask = np.isin(corpus_speakers, list(allowed_set))
            if args.speaker_mode == "filter":
                mask = speaker_mask | is_narration
                if mask.any():
                    score_row = np.where(mask, score_row, -np.inf)
                speaker_scope = "|".join(allowed_speakers)
            else:
                mask = speaker_mask | is_narration
                score_row = score_row + mask.astype(np.float32) * args.speaker_bonus
                speaker_scope = "|".join(allowed_speakers)

        voice_frac = voice_positions.get(voice_file, 0.5)
        position_scores = np.zeros(len(corpus_rows), dtype=np.float32)
        for corpus_idx in range(len(corpus_rows)):
            sp_frac = speaker_frac_lookup.get(corpus_idx, 0.5)
            position_scores[corpus_idx] = 1.0 - abs(voice_frac - sp_frac)

        combined = score_row + args.position_weight * position_scores
        top_k_mask = np.argsort(-combined)
        valid_mask = ~np.isneginf(combined[top_k_mask])
        top_k_indices = top_k_mask[valid_mask][: args.top_k]

        prologue = is_prologue(voice_file, prologue_bounds)

        for rank, corpus_idx in enumerate(top_k_indices, 1):
            corpus_row = corpus_rows[int(corpus_idx)]
            row = {
                "voice_file": voice_file,
                "prefix": prefix,
                "voice_frac": f"{voice_frac:.4f}",
                "speaker_scope": speaker_scope,
                "asr_japanese": asr_row.get(args.query_column, ""),
                "rank": rank,
                "combined_score": f"{float(combined[corpus_idx]):.6f}",
                "cosine_score": f"{float(scores[query_idx, corpus_idx]):.6f}",
                "position_score": f"{float(position_scores[corpus_idx]):.4f}",
                "record_id": corpus_row["record_id"],
                "script_file": corpus_row["script_file"],
                "line_no": corpus_row["line_no"],
                "text_type": corpus_row.get("text_type", ""),
                "speaker": corpus_row.get("speaker", ""),
                "matched_chinese": corpus_row["text"],
                "_corpus_idx": int(corpus_idx),
                "prologue": "yes" if prologue else "",
                "flags": "",
            }
            out_rows.append(row)

    best_rows: list[dict[str, object]] = []
    review_rows: list[dict[str, object]] = []
    seen_best: set[str] = set()

    for row in out_rows:
        vf = row["voice_file"]
        if int(row["rank"]) == 1 and vf not in seen_best:
            seen_best.add(vf)
            flags = detect_flags(row, speaker_map, corpus_speakers,
                                 prev_match_gpos, prev_match_ci, global_order)
            combined_s = float(row["combined_score"])
            if row["prologue"] == "yes":
                flags.append("PROLOGUE")

            if combined_s < args.min_score and "LOW_CONFIDENCE" not in flags:
                flags.append("LOW_CONFIDENCE")

            flag_str = "|".join(flags) if flags else ""
            row["flags"] = flag_str

            best_rows.append({
                "voice_file": vf,
                "prefix": row["prefix"],
                "voice_frac": row["voice_frac"],
                "speaker": row["speaker"],
                "script_file": row["script_file"],
                "line_no": row["line_no"],
                "combined_score": row["combined_score"],
                "cosine_score": row["cosine_score"],
                "position_score": row["position_score"],
                "matched_chinese": row["matched_chinese"],
                "asr_japanese": row["asr_japanese"],
                "text_type": row["text_type"],
                "flags": flag_str,
                "prologue": row["prologue"],
            })
            if flags:
                review_rows.append(best_rows[-1])

    best_fieldnames = [
        "voice_file", "prefix", "voice_frac", "speaker", "script_file", "line_no",
        "combined_score", "cosine_score", "position_score", "text_type",
        "matched_chinese", "asr_japanese", "flags", "prologue",
    ]

    all_fieldnames = [
        "voice_file", "prefix", "voice_frac", "speaker_scope",
        "asr_japanese", "rank", "combined_score", "cosine_score",
        "position_score", "record_id", "script_file", "line_no",
        "text_type", "speaker", "matched_chinese", "flags", "prologue",
    ]

    write_csv(args.out, out_rows, all_fieldnames)
    print(f"Wrote {len(out_rows)} candidates to {args.out}", flush=True)

    best_path = Path(str(args.out).replace(".csv", "_best.csv"))
    write_csv(best_path, best_rows, best_fieldnames)
    print(f"Wrote {len(best_rows)} best-match rows to {best_path}", flush=True)

    review_path = args.out_review or Path(str(args.out).replace(".csv", "_review.csv"))
    if review_rows:
        write_csv(review_path, review_rows, best_fieldnames)
        print(f"Wrote {len(review_rows)} flagged rows to {review_path}", flush=True)
    else:
        print("No flagged matches to review.", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
