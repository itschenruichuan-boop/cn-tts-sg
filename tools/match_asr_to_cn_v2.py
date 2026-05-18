#!/usr/bin/env python3
"""Match Japanese ASR lines to Chinese script lines with BGE-M3 + position-aware scoring.

Enhancements over v1:
- No translation layer needed (direct JA→ZH cross-lingual via BGE-M3)
- Voice file sequence position bias (voice number order ≈ script line order)
- Speaker role filtering via voice_speaker_map.json
- Sequential consistency check (nearby voice lines → nearby script lines)
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


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_speaker_map(path: Path) -> dict[str, list[str]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return {str(prefix): [str(s) for s in speakers] for prefix, speakers in data.items()}


_VOICE_NUM_RE = re.compile(r"^([A-Za-z_]+)_(\d+)(?:_(\d+))?$")


def parse_voice_number(stem: str) -> tuple[str, int, int]:
    """Parse voice file stem into (prefix, number, sub_number).
    e.g. CRS_0031 → ('CRS', 31, 0), ANA_0007_2 → ('ANA', 7, 2)
    """
    m = _VOICE_NUM_RE.match(stem)
    if not m:
        return (stem, 0, 0)
    return (m.group(1), int(m.group(2)), int(m.group(3) or 0))


def build_voice_positions(asr_rows: list[dict[str, str]]) -> dict[str, dict[str, float]]:
    """Compute relative position (0..1) of each voice file within its prefix group.
    Returns {voice_file: position_frac}
    """
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


def build_corpus_speaker_positions(
    corpus_rows: list[dict[str, str]],
    text_type: str,
) -> tuple[dict[str, list[tuple[int, int, float]]], dict[int, int]]:
    """Build speaker-indexed corpus positions and a global script order.
    Returns:
      speaker_positions: {speaker: [(corpus_idx, global_pos, speaker_frac), ...]}
      global_order: {corpus_idx: global_pos}
    """
    script_files = sorted(set(row.get("script_file", "") for row in corpus_rows))
    script_order = {sf: i for i, sf in enumerate(script_files)}

    global_order: dict[int, int] = {}
    valid_indices = set(range(len(corpus_rows)))
    for i in valid_indices:
        sf = corpus_rows[i].get("script_file", "")
        ln = int(corpus_rows[i].get("line_no", 0))
        global_order[i] = script_order.get(sf, 0) * 100000 + ln

    speaker_lines: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for i in valid_indices:
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
    """Flatten speaker positions into {corpus_idx: speaker_frac} for fast lookup."""
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
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype(np.float32, copy=False)

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, embeddings=embeddings)
        print(f"Saved embeddings: {cache_path}", flush=True)
    return embeddings


def sequential_smooth(
    voice_order: list[tuple[str, int, int, float, str]],
    top1_matches: dict[str, tuple[int, float, float]],
    global_order: dict[int, int],
) -> dict[str, tuple[int, float, float]]:
    """Per-prefix monotonicity check: nearby voice files of same speaker
    should map to script lines in increasing order."""
    from collections import defaultdict

    by_prefix: dict[str, list[tuple[str, int, int, float]]] = defaultdict(list)
    for vf, num, sub, frac, prefix in voice_order:
        by_prefix[prefix].append((vf, num, sub, frac))

    smoothed: dict[str, tuple[int, float, float]] = dict(top1_matches)

    for prefix, files in by_prefix.items():
        if len(files) < 3:
            continue
        files_sorted = sorted(files, key=lambda x: (x[1], x[2]))
        for i in range(1, len(files_sorted) - 1):
            vf = files_sorted[i][0]
            prev_vf = files_sorted[i - 1][0]
            next_vf = files_sorted[i + 1][0]
            if prev_vf not in top1_matches or vf not in top1_matches or next_vf not in top1_matches:
                continue
            prev_ci, _, _ = top1_matches[prev_vf]
            curr_ci, _, _ = top1_matches[vf]
            next_ci, _, _ = top1_matches[next_vf]
            if prev_ci is None or curr_ci is None or next_ci is None:
                continue
            prev_gpos = global_order.get(prev_ci, 0)
            curr_gpos = global_order.get(curr_ci, 0)
            next_gpos = global_order.get(next_ci, 0)
            if prev_gpos > curr_gpos or curr_gpos > next_gpos:
                smoothed[vf] = (None, -1.0, 0.0)

    return smoothed


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Match JA ASR → CN script with position-aware scoring (no translation needed)"
    )
    parser.add_argument("--asr-csv", type=Path, default=Path("work/asr/asr_sample_cuda.csv"))
    parser.add_argument("--corpus-csv", type=Path, default=Path("work/analysis/cn_corpus.csv"))
    parser.add_argument("--model-dir", type=Path, default=Path("models/bge-m3"))
    parser.add_argument("--out", type=Path, default=Path("work/matches/asr_cn_matches_v2.csv"))
    parser.add_argument("--out-best", type=Path, default=Path("work/matches/asr_cn_best_v2.csv"))
    parser.add_argument("--cache", type=Path, default=Path("work/index/cn_corpus_bge_m3_torch.npz"))
    parser.add_argument("--query-column", default="asr_japanese")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--text-type", choices=["all", "dialogue", "narration"], default="all")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--speaker-map", type=Path, default=Path("config/voice_speaker_map.json"))
    parser.add_argument("--speaker-mode", choices=["filter", "boost"], default="filter")
    parser.add_argument("--speaker-bonus", type=float, default=0.08)
    parser.add_argument(
        "--script-glob",
        default="SG*.SCX",
        help="Glob pattern to filter corpus script_file names (default: SG*.SCX for main story)",
    )
    parser.add_argument("--no-script-filter", action="store_true")
    parser.add_argument("--position-weight", type=float, default=0.15)
    parser.add_argument("--min-score", type=float, default=0.55)
    parser.add_argument("--score-gap", type=float, default=0.05)
    parser.add_argument("--rebuild-index", action="store_true")
    parser.add_argument("--no-sequential-smooth", action="store_true")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but torch.cuda.is_available() is False")

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
        raise SystemExit(f"No corpus rows found in {args.corpus_csv}")

    speaker_map = load_speaker_map(args.speaker_map)
    corpus_speakers = np.array([row.get("speaker", "") for row in corpus_rows])

    voice_positions = build_voice_positions(asr_rows)
    speaker_positions, global_order = build_corpus_speaker_positions(corpus_rows, args.text_type)
    speaker_frac_lookup = build_speaker_frac_lookup(speaker_positions)

    prefix_speakers: dict[str, set[str]] = {}
    for prefix, speakers in speaker_map.items():
        prefix_speakers[prefix] = set(speakers)

    print(f"ASR rows: {len(asr_rows)}", flush=True)
    print(f"Corpus rows (dialogue): {len(corpus_rows)}", flush=True)
    print(f"Voice prefixes: {len(voice_positions)}", flush=True)
    print(f"Speaker positions: {sum(len(v) for v in speaker_positions.values())}", flush=True)
    print(f"Position weight: {args.position_weight}", flush=True)
    print(f"Loading model: {args.model_dir}", flush=True)

    model = SentenceTransformer(str(args.model_dir), device=args.device)

    corpus_texts = [row["text"] for row in corpus_rows]
    corpus_embeddings = encode_texts(
        model, corpus_texts, args.batch_size, args.cache, args.rebuild_index
    )

    query_texts = [row[args.query_column] for row in asr_rows]
    query_embeddings = encode_texts(model, query_texts, args.batch_size)

    scores = query_embeddings @ corpus_embeddings.T
    out_rows: list[dict[str, object]] = []
    best_rows: list[dict[str, object]] = []

    for query_idx, asr_row in enumerate(asr_rows):
        prefix = asr_row.get("prefix", "")
        voice_file = asr_row.get("voice_file", "")
        allowed_speakers = speaker_map.get(prefix, [])

        score_row = scores[query_idx].copy()
        speaker_scope = "all"

        allowed_set = set(allowed_speakers)
        if allowed_set:
            speaker_mask = np.isin(corpus_speakers, list(allowed_set))
            is_narration = np.array([r.get("text_type") == "narration" for r in corpus_rows])
            if args.speaker_mode == "filter":
                mask = speaker_mask | is_narration
                if mask.any():
                    score_row = np.where(mask, score_row, -np.inf)
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

        for rank, corpus_idx in enumerate(top_k_indices, 1):
            corpus_row = corpus_rows[int(corpus_idx)]
            out_rows.append({
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
                "speaker": corpus_row["speaker"],
                "matched_chinese": corpus_row["text"],
            })

    record_id_to_corpus_idx = {r["record_id"]: i for i, r in enumerate(corpus_rows)}

    voice_order: list[tuple[str, int, int, float, str]] = []
    for row in asr_rows:
        vf = row.get("voice_file", "")
        stem = Path(vf).stem
        _, num, sub = parse_voice_number(stem)
        frac = voice_positions.get(vf, 0.0)
        prefix = row.get("prefix", "")
        voice_order.append((vf, num, sub, frac, prefix))

    top1_matches: dict[str, tuple[int, float, float]] = {}
    for row in out_rows:
        vf = row["voice_file"]
        if vf not in top1_matches and int(row["rank"]) == 1:
            ci = record_id_to_corpus_idx.get(row["record_id"])
            if ci is None:
                continue
            combined_s = float(row["combined_score"])
            cosine_s = float(row["cosine_score"])
            top1_matches[vf] = (ci, combined_s, cosine_s)

    if not args.no_sequential_smooth:
        top1_matches = sequential_smooth(voice_order, top1_matches, global_order)

    best_fieldnames = [
        "voice_file", "prefix", "voice_frac", "speaker", "script_file", "line_no",
        "combined_score", "cosine_score", "position_score",
        "matched_chinese", "asr_japanese", "speaker_scope",
    ]
    for row in out_rows:
        vf = row["voice_file"]
        if int(row["rank"]) == 1:
            if vf in top1_matches and top1_matches[vf][0] is None:
                continue
            combined_s = float(row["combined_score"])
            cosine_s = float(row["cosine_score"])
            if combined_s < args.min_score:
                continue
            best_rows.append({
                "voice_file": vf,
                "prefix": row["prefix"],
                "voice_frac": row["voice_frac"],
                "speaker": row["speaker"],
                "script_file": row["script_file"],
                "line_no": row["line_no"],
                "combined_score": f"{combined_s:.6f}",
                "cosine_score": f"{cosine_s:.6f}",
                "position_score": row["position_score"],
                "matched_chinese": row["matched_chinese"],
                "asr_japanese": row["asr_japanese"],
                "speaker_scope": row["speaker_scope"],
            })

    fieldnames = [
        "voice_file", "prefix", "voice_frac", "speaker_scope",
        "asr_japanese", "rank", "combined_score", "cosine_score",
        "position_score", "record_id", "script_file", "line_no",
        "speaker", "matched_chinese",
    ]
    write_csv(args.out, out_rows, fieldnames)
    write_csv(args.out_best, best_rows, best_fieldnames)

    print(f"Wrote {len(out_rows)} candidate rows to {args.out}", flush=True)
    print(f"Wrote {len(best_rows)} best-match rows to {args.out_best}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
