#!/usr/bin/env python3
"""Translate Japanese ASR text with a local Sakura/GalTransl GGUF model."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

from llama_cpp import Llama


SYSTEM_PROMPT = (
    "你是一个视觉小说翻译模型，可以通顺地使用给定的术语表以指定的风格将日文翻译成简体中文，"
    "并联系上下文正确使用人称代词，注意不要混淆使役态和被动态的主语和宾语，"
    "不要擅自添加原文中没有的特殊符号，也不要擅自增加或减少换行。"
)

GLOSSARY = """\
岡部->冈部 #男主姓氏
倫太郎->伦太郎 #男主名字
オカリン->冈伦 #真由理对冈部伦太郎的昵称
鳳凰院凶真->凤凰院凶真 #冈部的中二名
ホウオウイン・キョウマ->凤凰院凶真 #冈部的中二名
鳳凰院->凤凰院
凶真->凶真
牧瀬紅莉栖->牧濑红莉栖 #女主
紅莉栖->红莉栖
クリスティーナ->克里斯蒂娜
椎名まゆり->椎名真由理
まゆり->真由理
まゆしぃ->真由氏
橋田至->桥田至
ダル->桶子
阿万音鈴羽->阿万音铃羽
鈴羽->铃羽
漆原るか->漆原琉华
るか->琉华
フェイリス->菲莉丝
桐生萌郁->桐生萌郁
萌郁->萌郁
ラボ->LAB
未来ガジェット研究所->未来装置研究所
電話レンジ->电话微波炉
シュタインズゲート->Steins Gate
機関->机关
セレン->SERN
"""


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_user_prompt(text: str, glossary: str) -> str:
    return (
        "参考以下术语表（可为空，格式为src->dst #备注）：\n"
        f"{glossary.strip()}\n\n"
        "根据以上术语表的对应关系和备注，结合历史剧情和上下文，将下面的文本从日文翻译成简体中文：\n"
        f"{text.strip()}"
    )


def clean_output(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^翻译[:：]\s*", "", text)
    text = text.strip(" \n\r\t")
    return text


def translate_one(llm: Llama, text: str, max_tokens: int, temperature: float, top_p: float) -> str:
    if not text.strip():
        return ""
    result = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(text, GLOSSARY)},
        ],
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )
    return clean_output(result["choices"][0]["message"]["content"])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asr-csv", type=Path, default=Path("work/asr/asr_largest50_cuda.csv"))
    parser.add_argument("--out", type=Path, default=Path("work/asr/asr_largest50_sakura.csv"))
    parser.add_argument("--model", type=Path, default=Path("models/GalTransl-v4-4B-2601/Galtransl-v4-4B-2601.gguf"))
    parser.add_argument("--source-column", default="asr_japanese")
    parser.add_argument("--target-column", default="asr_chinese_sakura")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--n-ctx", type=int, default=4096)
    parser.add_argument("--n-gpu-layers", type=int, default=0)
    parser.add_argument("--n-threads", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--top-p", type=float, default=0.8)
    args = parser.parse_args()

    rows = read_csv(args.asr_csv)
    if args.limit:
        rows = rows[: args.limit]

    print(f"Loading GGUF model: {args.model}", flush=True)
    llm = Llama(
        model_path=str(args.model),
        n_ctx=args.n_ctx,
        n_gpu_layers=args.n_gpu_layers,
        n_threads=args.n_threads,
        verbose=False,
    )

    out_rows: list[dict[str, str]] = []
    for index, row in enumerate(rows, 1):
        source = row.get(args.source_column, "")
        print(f"[{index}/{len(rows)}] {row.get('voice_file', '')}", flush=True)
        updated = dict(row)
        try:
            updated[args.target_column] = translate_one(
                llm,
                source,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            )
            print(f"  {updated[args.target_column]}", flush=True)
        except Exception as exc:
            updated[args.target_column] = ""
            updated["sakura_error"] = repr(exc)
            print(f"  ERROR: {exc!r}", flush=True)
        out_rows.append(updated)

    fieldnames = list(out_rows[0].keys()) if out_rows else []
    if args.target_column not in fieldnames:
        fieldnames.append(args.target_column)
    if any("sakura_error" in row for row in out_rows) and "sakura_error" not in fieldnames:
        fieldnames.append("sakura_error")
    write_csv(args.out, out_rows, fieldnames)
    print(f"Wrote {len(out_rows)} rows to {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
