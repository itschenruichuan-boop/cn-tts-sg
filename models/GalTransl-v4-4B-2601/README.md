# GalTransl-v4-4B（可选）

基于 Sakura LLM 的 Galgame 专用日→中翻译模型。

## 用途

在台词匹配流水线中**可选**使用，用于将 Whisper ASR 的日文结果翻译为中文，
再与中文台词库做匹配（提高匹配精度）。如果不使用此模型，BGE-M3 也可以
直接做日→中跨语言匹配。

## 下载

```powershell
python tools/download_hf_mirror.py SakuraLLM/GalTransl-v4-4B-2601 --local-dir models/GalTransl-v4-4B-2601 --include "*.gguf"
```

模型地址：https://huggingface.co/SakuraLLM/GalTransl-v4-4B-2601

需要文件：`Galtransl-v4-4B-2601.gguf` (Q6K 量化，需 ~6GB 显存)

## 使用

需要安装 `llama-cpp-python`：
```powershell
pip install llama-cpp-python
```

当前流水线默认**不启用**翻译步骤，直接使用 BGE-M3 跨语言匹配。
