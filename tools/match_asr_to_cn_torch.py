#!/usr/bin/env python3
"""Match Japanese ASR lines to Chinese script lines with BGE-M3 on CUDA."""

from __future__ import annotations

import argparse
import csv
import json
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


def load_speaker_map(path: Path | None) -> dict[str, list[str]]:
    if not path:
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return {str(prefix): [str(speaker) for speaker in speakers] for prefix, speakers in data.items()}


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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asr-csv", type=Path, default=Path("work/asr/asr_sample_cuda.csv"))
    parser.add_argument("--corpus-csv", type=Path, default=Path("work/analysis/cn_corpus.csv"))
    parser.add_argument("--model-dir", type=Path, default=Path("models/bge-m3"))
    parser.add_argument("--out", type=Path, default=Path("work/matches/asr_cn_matches_torch.csv"))
    parser.add_argument("--cache", type=Path, default=Path("work/index/cn_corpus_bge_m3_torch.npz"))
    parser.add_argument("--query-column", default="asr_japanese")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--text-type", choices=["all", "dialogue", "narration"], default="all")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--speaker-map", type=Path, default=Path("config/voice_speaker_map.json"))
    parser.add_argument(
        "--speaker-mode",
        choices=["none", "filter", "boost"],
        default="filter",
        help="filter: search matching speaker rows first; boost: search all rows but add speaker bonus",
    )
    parser.add_argument("--speaker-bonus", type=float, default=0.08)
    parser.add_argument("--fallback-top-k", type=int, default=5)
    parser.add_argument("--rebuild-index", action="store_true")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but torch.cuda.is_available() is False")

    print(f"torch: {torch.__version__}", flush=True)
    print(f"cuda available: {torch.cuda.is_available()}", flush=True)
    if torch.cuda.is_available():
        print(f"gpu: {torch.cuda.get_device_name(0)}", flush=True)

    asr_rows = [row for row in read_csv(args.asr_csv) if row.get(args.query_column, "").strip()]
    corpus_rows = read_csv(args.corpus_csv)
    if args.text_type != "all":
        corpus_rows = [row for row in corpus_rows if row.get("text_type") == args.text_type]

    if not asr_rows:
        raise SystemExit(f"No ASR rows found in {args.asr_csv}")
    if not corpus_rows:
        raise SystemExit(f"No corpus rows found in {args.corpus_csv}")

    speaker_map = load_speaker_map(args.speaker_map) if args.speaker_mode != "none" else {}
    corpus_speakers = np.array([row.get("speaker", "") for row in corpus_rows])
    print(f"ASR rows: {len(asr_rows)}", flush=True)
    print(f"Corpus rows: {len(corpus_rows)}", flush=True)
    print(f"Loading model: {args.model_dir}", flush=True)
    model = SentenceTransformer(str(args.model_dir), device=args.device)

    corpus_embeddings = encode_texts(
        model,
        [row["text"] for row in corpus_rows],
        batch_size=args.batch_size,
        cache_path=args.cache,
        rebuild=args.rebuild_index,
    )
    query_embeddings = encode_texts(
        model,
        [row[args.query_column] for row in asr_rows],
        batch_size=args.batch_size,
    )

    scores = query_embeddings @ corpus_embeddings.T
    out_rows: list[dict[str, object]] = []
    for query_idx, asr_row in enumerate(asr_rows):
        prefix = asr_row.get("prefix", "")
        allowed_speakers = speaker_map.get(prefix, [])
        score_row = scores[query_idx].copy()
        speaker_scope = "all"
        if allowed_speakers and args.speaker_mode == "filter":
            mask = np.isin(corpus_speakers, allowed_speakers)
            if mask.any():
                score_row = np.where(mask, score_row, -np.inf)
                speaker_scope = "|".join(allowed_speakers)
        elif allowed_speakers and args.speaker_mode == "boost":
            mask = np.isin(corpus_speakers, allowed_speakers)
            score_row = score_row + mask.astype(np.float32) * args.speaker_bonus
            speaker_scope = "|".join(allowed_speakers)

        best = np.argsort(-score_row)[: args.top_k]
        if np.isneginf(score_row[best]).all():
            best = np.argsort(-scores[query_idx])[: args.fallback_top_k]
            speaker_scope = "fallback_all"
        for rank, corpus_idx in enumerate(best, 1):
            corpus_row = corpus_rows[int(corpus_idx)]
            out_rows.append(
                {
                    "voice_file": asr_row.get("voice_file", ""),
                    "prefix": prefix,
                    "speaker_scope": speaker_scope,
                    "asr_japanese": asr_row.get("asr_japanese", ""),
                    "query_text": asr_row.get(args.query_column, ""),
                    "rank": rank,
                    "score": f"{float(score_row[corpus_idx]):.6f}",
                    "raw_score": f"{float(scores[query_idx, corpus_idx]):.6f}",
                    "record_id": corpus_row["record_id"],
                    "script_file": corpus_row["script_file"],
                    "line_no": corpus_row["line_no"],
                    "text_type": corpus_row["text_type"],
                    "speaker": corpus_row["speaker"],
                    "matched_chinese": corpus_row["text"],
                }
            )

    fieldnames = [
        "voice_file",
        "prefix",
        "speaker_scope",
        "asr_japanese",
        "query_text",
        "rank",
        "score",
        "raw_score",
        "record_id",
        "script_file",
        "line_no",
        "text_type",
        "speaker",
        "matched_chinese",
    ]
    write_csv(args.out, out_rows, fieldnames)
    print(f"Wrote {len(out_rows)} candidate rows to {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
