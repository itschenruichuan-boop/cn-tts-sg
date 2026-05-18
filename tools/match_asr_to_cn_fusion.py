#!/usr/bin/env python3
"""Role-aware fused matching with Japanese ASR and NLLB Chinese query scores."""

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


def load_speaker_map(path: Path) -> dict[str, list[str]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return {str(prefix): [str(speaker) for speaker in speakers] for prefix, speakers in data.items()}


def encode_texts(model: SentenceTransformer, texts: list[str], batch_size: int) -> np.ndarray:
    return model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype(np.float32, copy=False)


def load_corpus_embeddings(cache_path: Path, expected_rows: int) -> np.ndarray:
    data = np.load(cache_path, allow_pickle=False)
    embeddings = data["embeddings"]
    if embeddings.shape[0] != expected_rows:
        raise SystemExit(
            f"Corpus embedding row count mismatch: {embeddings.shape[0]} != {expected_rows}. Rebuild the cache."
        )
    return embeddings


def is_short_query(text: str) -> bool:
    compact = "".join(ch for ch in text if not ch.isspace())
    return len(compact) <= 8


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asr-csv", type=Path, default=Path("work/asr/asr_sample_nllb.csv"))
    parser.add_argument("--corpus-csv", type=Path, default=Path("work/analysis/cn_corpus.csv"))
    parser.add_argument("--corpus-cache", type=Path, default=Path("work/index/cn_dialogue_bge_m3_torch.npz"))
    parser.add_argument("--model-dir", type=Path, default=Path("models/bge-m3"))
    parser.add_argument("--speaker-map", type=Path, default=Path("config/voice_speaker_map.json"))
    parser.add_argument("--out", type=Path, default=Path("work/matches/asr_cn_matches_fusion.csv"))
    parser.add_argument("--ja-column", default="asr_japanese")
    parser.add_argument("--zh-column", default="asr_chinese_nllb")
    parser.add_argument("--text-type", choices=["dialogue", "all"], default="dialogue")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--short-ja-weight", type=float, default=0.8)
    parser.add_argument("--long-ja-weight", type=float, default=0.35)
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but torch.cuda.is_available() is False")

    asr_rows = [row for row in read_csv(args.asr_csv) if row.get(args.ja_column, "").strip()]
    corpus_rows = read_csv(args.corpus_csv)
    if args.text_type == "dialogue":
        corpus_rows = [row for row in corpus_rows if row.get("text_type") == "dialogue"]
    speaker_map = load_speaker_map(args.speaker_map)
    corpus_speakers = np.array([row.get("speaker", "") for row in corpus_rows])

    print(f"ASR rows: {len(asr_rows)}", flush=True)
    print(f"Corpus rows: {len(corpus_rows)}", flush=True)
    corpus_embeddings = load_corpus_embeddings(args.corpus_cache, len(corpus_rows))

    print(f"Loading model: {args.model_dir}", flush=True)
    model = SentenceTransformer(str(args.model_dir), device=args.device)
    ja_embeddings = encode_texts(model, [row[args.ja_column] for row in asr_rows], args.batch_size)
    zh_embeddings = encode_texts(model, [row.get(args.zh_column, "") or row[args.ja_column] for row in asr_rows], args.batch_size)

    ja_scores = ja_embeddings @ corpus_embeddings.T
    zh_scores = zh_embeddings @ corpus_embeddings.T

    out_rows: list[dict[str, object]] = []
    for query_idx, asr_row in enumerate(asr_rows):
        prefix = asr_row.get("prefix", "")
        speakers = speaker_map.get(prefix, [])
        mask = np.isin(corpus_speakers, speakers) if speakers else np.ones(len(corpus_rows), dtype=bool)
        ja_text = asr_row[args.ja_column]
        zh_text = asr_row.get(args.zh_column, "")
        ja_weight = args.short_ja_weight if is_short_query(ja_text) else args.long_ja_weight
        zh_weight = 1.0 - ja_weight
        fused = ja_scores[query_idx] * ja_weight + zh_scores[query_idx] * zh_weight
        fused = np.where(mask, fused, -np.inf)
        best = np.argsort(-fused)[: args.top_k]

        for rank, corpus_idx in enumerate(best, 1):
            corpus_row = corpus_rows[int(corpus_idx)]
            out_rows.append(
                {
                    "voice_file": asr_row.get("voice_file", ""),
                    "prefix": prefix,
                    "speaker_scope": "|".join(speakers) if speakers else "all",
                    "asr_japanese": ja_text,
                    "asr_chinese_nllb": zh_text,
                    "rank": rank,
                    "fused_score": f"{float(fused[corpus_idx]):.6f}",
                    "ja_score": f"{float(ja_scores[query_idx, corpus_idx]):.6f}",
                    "zh_score": f"{float(zh_scores[query_idx, corpus_idx]):.6f}",
                    "ja_weight": f"{ja_weight:.2f}",
                    "record_id": corpus_row["record_id"],
                    "script_file": corpus_row["script_file"],
                    "line_no": corpus_row["line_no"],
                    "speaker": corpus_row["speaker"],
                    "matched_chinese": corpus_row["text"],
                }
            )

    fieldnames = [
        "voice_file",
        "prefix",
        "speaker_scope",
        "asr_japanese",
        "asr_chinese_nllb",
        "rank",
        "fused_score",
        "ja_score",
        "zh_score",
        "ja_weight",
        "record_id",
        "script_file",
        "line_no",
        "speaker",
        "matched_chinese",
    ]
    write_csv(args.out, out_rows, fieldnames)
    print(f"Wrote {len(out_rows)} candidate rows to {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
