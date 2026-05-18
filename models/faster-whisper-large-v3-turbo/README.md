# Faster-Whisper Large-v3-turbo

日语语音识别模型（ASR），将 OGG 音频转为日文文本。

## 下载

`faster-whisper` 库首次运行时会自动从 HuggingFace 下载模型到缓存目录。
也可以手动下载后放到本目录：

```powershell
python tools/download_hf_mirror.py Systran/faster-whisper-large-v3-turbo --local-dir models/faster-whisper-large-v3-turbo
```

模型地址：https://huggingface.co/Systran/faster-whisper-large-v3-turbo

需要文件：`model.bin`, `config.json`, `tokenizer.json` 等。
