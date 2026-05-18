#!/usr/bin/env python3
"""Translate Japanese ASR text to Simplified Chinese with local NLLB."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asr-csv", type=Path, default=Path("work/asr/asr_sample_cuda.csv"))
    parser.add_argument("--out", type=Path, default=Path("work/asr/asr_sample_nllb.csv"))
    parser.add_argument("--model-dir", type=Path, default=Path("models/nllb-200-distilled-1.3B"))
    parser.add_argument("--source-column", default="asr_japanese")
    parser.add_argument("--target-column", default="asr_chinese_nllb")
    parser.add_argument("--src-lang", default="jpn_Jpan")
    parser.add_argument("--tgt-lang", default="zho_Hans")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but torch.cuda.is_available() is False")

    rows = read_csv(args.asr_csv)
    texts = [row.get(args.source_column, "").strip() for row in rows]
    if not any(texts):
        raise SystemExit(f"No source text found in column {args.source_column}")

    print(f"torch: {torch.__version__}", flush=True)
    print(f"cuda available: {torch.cuda.is_available()}", flush=True)
    if torch.cuda.is_available():
        print(f"gpu: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"Loading tokenizer/model: {args.model_dir}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(str(args.model_dir), local_files_only=True)
    tokenizer.src_lang = args.src_lang
    dtype = torch.float16 if args.device == "cuda" else torch.float32
    model = AutoModelForSeq2SeqLM.from_pretrained(
        str(args.model_dir),
        local_files_only=True,
        torch_dtype=dtype,
    ).to(args.device)
    model.eval()
    forced_bos_token_id = tokenizer.convert_tokens_to_ids(args.tgt_lang)

    translations: list[str] = []
    for start in range(0, len(texts), args.batch_size):
        batch_texts = texts[start : start + args.batch_size]
        nonempty = [text if text else "。" for text in batch_texts]
        encoded = tokenizer(nonempty, return_tensors="pt", padding=True, truncation=True).to(args.device)
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                forced_bos_token_id=forced_bos_token_id,
                max_new_tokens=args.max_new_tokens,
            )
        decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
        for source, target in zip(batch_texts, decoded):
            translations.append(target.strip() if source else "")
        print(f"translated {min(start + args.batch_size, len(texts))}/{len(texts)}", flush=True)

    out_rows: list[dict[str, str]] = []
    for row, translation in zip(rows, translations):
        updated = dict(row)
        updated[args.target_column] = translation
        out_rows.append(updated)

    fieldnames = list(rows[0].keys())
    if args.target_column not in fieldnames:
        fieldnames.append(args.target_column)
    write_csv(args.out, out_rows, fieldnames)
    print(f"Wrote translated ASR rows to {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
