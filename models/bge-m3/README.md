# BGE-M3 模型

多语言嵌入模型，用于日文 ASR → 中文台词跨语言匹配。

## 下载

```powershell
python tools/download_hf_mirror.py BAAI/bge-m3 --local-dir models/bge-m3
```

或从 HuggingFace 直接下载：
https://huggingface.co/BAAI/bge-m3

需要文件：`pytorch_model.bin`, `tokenizer.json`, `config.json`, `modules.json` 等。
