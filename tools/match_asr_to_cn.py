#!/usr/bin/env python3
"""Match Japanese ASR lines to Chinese script lines with BGE-M3 ONNX embeddings."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


class BgeM3Onnx:
    def __init__(self, model_dir: Path, providers: list[str], max_length: int):
        print(f"Loading tokenizer from {model_dir / 'tokenizer.json'}", flush=True)
        self.tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        self.tokenizer.enable_truncation(max_length=max_length)
        self.tokenizer.enable_padding(length=max_length, pad_id=1, pad_token="<pad>")
        print(f"Loading ONNX model from {model_dir / 'onnx' / 'model.onnx'}", flush=True)
        self.session = ort.InferenceSession(
            str(model_dir / "onnx" / "model.onnx"),
            providers=providers,
        )
        print("ONNX model loaded.", flush=True)

    def encode(self, texts: list[str], batch_size: int) -> np.ndarray:
        chunks: list[np.ndarray] = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            encoded = self.tokenizer.encode_batch(batch)
            input_ids = np.array([item.ids for item in encoded], dtype=np.int64)
            attention_mask = np.array([item.attention_mask for item in encoded], dtype=np.int64)
            outputs = self.session.run(
                ["sentence_embedding"],
                {"input_ids": input_ids, "attention_mask": attention_mask},
            )[0]
            outputs = outputs.astype(np.float32, copy=False)
            outputs /= np.maximum(np.linalg.norm(outputs, axis=1, keepdims=True), 1e-12)
            chunks.append(outputs)
            print(f"encoded {min(start + batch_size, len(texts))}/{len(texts)}", flush=True)
        return np.vstack(chunks)


def load_or_build_corpus_embeddings(
    encoder: BgeM3Onnx,
    corpus_rows: list[dict[str, str]],
    cache_path: Path,
    batch_size: int,
    rebuild: bool,
) -> np.ndarray:
    if cache_path.exists() and not rebuild:
        data = np.load(cache_path, allow_pickle=False)
        embeddings = data["embeddings"]
        if embeddings.shape[0] == len(corpus_rows):
            print(f"Loaded cached corpus embeddings: {cache_path}")
            return embeddings
            print("Corpus cache row count mismatch; rebuilding.", flush=True)

    embeddings = encoder.encode([row["text"] for row in corpus_rows], batch_size=batch_size)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, embeddings=embeddings)
    print(f"Saved corpus embeddings: {cache_path}", flush=True)
    return embeddings


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asr-csv", type=Path, default=Path("work/asr/asr_sample_cuda.csv"))
    parser.add_argument("--corpus-csv", type=Path, default=Path("work/analysis/cn_corpus.csv"))
    parser.add_argument("--model-dir", type=Path, default=Path("models/bge-m3"))
    parser.add_argument("--out", type=Path, default=Path("work/matches/asr_cn_matches.csv"))
    parser.add_argument("--cache", type=Path, default=Path("work/index/cn_corpus_bge_m3_onnx.npz"))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--text-type", choices=["all", "dialogue", "narration"], default="all")
    parser.add_argument("--rebuild-index", action="store_true")
    args = parser.parse_args()

    asr_rows = [row for row in read_csv(args.asr_csv) if row.get("asr_japanese", "").strip()]
    corpus_rows = read_csv(args.corpus_csv)
    if args.text_type != "all":
        corpus_rows = [row for row in corpus_rows if row.get("text_type") == args.text_type]

    if not asr_rows:
        raise SystemExit(f"No ASR rows found in {args.asr_csv}")
    if not corpus_rows:
        raise SystemExit(f"No corpus rows found in {args.corpus_csv}")

    print(f"ASR rows: {len(asr_rows)}", flush=True)
    print(f"Corpus rows: {len(corpus_rows)}", flush=True)
    providers = ["CPUExecutionProvider"]
    encoder = BgeM3Onnx(args.model_dir, providers=providers, max_length=args.max_length)
    corpus_embeddings = load_or_build_corpus_embeddings(
        encoder, corpus_rows, args.cache, args.batch_size, args.rebuild_index
    )
    query_embeddings = encoder.encode([row["asr_japanese"] for row in asr_rows], batch_size=args.batch_size)

    scores = query_embeddings @ corpus_embeddings.T
    out_rows: list[dict[str, object]] = []
    for query_idx, asr_row in enumerate(asr_rows):
        best = np.argsort(-scores[query_idx])[: args.top_k]
        for rank, corpus_idx in enumerate(best, 1):
            corpus_row = corpus_rows[int(corpus_idx)]
            out_rows.append(
                {
                    "voice_file": asr_row.get("voice_file", ""),
                    "prefix": asr_row.get("prefix", ""),
                    "asr_japanese": asr_row.get("asr_japanese", ""),
                    "rank": rank,
                    "score": f"{float(scores[query_idx, corpus_idx]):.6f}",
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
        "asr_japanese",
        "rank",
        "score",
        "record_id",
        "script_file",
        "line_no",
        "text_type",
        "speaker",
        "matched_chinese",
    ]
    write_csv(args.out, out_rows, fieldnames)
    print(f"Wrote {len(out_rows)} candidate rows to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
