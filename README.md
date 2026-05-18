# SG-TTS：命运石之门 / Steins;Gate 中文语音替换

用 ASR + 跨语言匹配 + Zero-Shot TTS 将日文配音替换为中文合成语音。

## 前置准备（clone 后手动完成）

以下文件被 `.gitignore` 排除，需自行准备：

### 1. 游戏原始文件 → `raw/sg/`
```
raw/sg/
├── voice.mpk               # 游戏原始语音包
├── script.mpk              # 游戏原始脚本包
└── cn_txt/                 # 中文台词文本（从 SCX 导出）
```
- 用 `tools/mpk_tool.py extract` 解包 MPK
- 中文台词 `.txt` 需用 MagesTools 或 sc3tools 从 SCX 导出

### 2. AI 模型 → `models/`
```
models/
├── bge-m3/                         # BAAI/bge-m3 (SentenceTransformer)
├── faster-whisper-large-v3-turbo/   # Whisper 语音识别
├── GalTransl-v4-4B-2601/           # （可选）Galgame 日→中翻译
└── nllb-200-distilled-1.3B/        # （可选）NLLB 翻译
```
下载命令：
```powershell
python tools/download_hf_mirror.py BAAI/bge-m3 --local-dir models/bge-m3
# faster-whisper 会自动下载到 models/
```

### 3. IndexTTS2 引擎 → `index-tts/`
```powershell
git clone https://github.com/index-tts/index-tts.git
cd index-tts
uv sync --extra webui
uv run modelscope download --model IndexTeam/IndexTTS-2 --local_dir checkpoints
```

### 4. Python 虚拟环境
```powershell
python -m venv .venv-asr        # faster-whisper + CUDA wheels
python -m venv .venv-match      # sentence-transformers + torch
```

## 项目结构

```
SG-TTS/
├── tools/                  # 流水线脚本
├── config/                 # 角色映射配置
├── models/                 # AI 模型（git ignore）
├── index-tts/              # IndexTTS2 引擎
│
├── raw/sg/                 # 原始游戏解包文件
│   ├── voice/              #   14512 条 OGG 日文语音
│   ├── scripts/            #   SCX 脚本文件
│   └── cn_txt/             #   中文台词文本
│
├── output/sg/              # 生成产物
│   ├── analysis/           #   语音清单、语料库 CSV
│   ├── asr/                #   Whisper ASR 转写结果
│   ├── matches/            #   匹配结果（best + review）
│   ├── index/              #   BGE-M3 嵌入索引
│   ├── tts_wav/            #   IndexTTS2 合成 WAV
│   └── final/              #   最终 OGG + voice.mpk
│
└── output/sg0/             # （未来）命运石之门0
```

## 流水线

### 1. 提取游戏资源
```powershell
python tools/mpk_tool.py extract raw/sg/voice.mpk raw/sg/voice
python tools/mpk_tool.py extract raw/sg/script.mpk raw/sg/scripts
```

### 2. 构建中文语料库
```powershell
python tools/build_cn_corpus.py
```

### 3. 全量 ASR（日文语音 → 日文文本）
```powershell
# 需要 .venv-asr 环境（faster-whisper）
python tools/asr_voice_batch.py
```

### 4. 台词匹配（日文 ASR → 中文台词）
```powershell
# 需要 .venv-match 环境（sentence-transformers）
python tools/match_asr_to_cn_v3.py
```

### 5. TTS 合成（中文台词 → 中文语音）
```powershell
# 需要 IndexTTS2（cd index-tts && uv sync --extra webui）
cd index-tts
uv run python ../tools/tts_batch.py
```

### 6. 转换 + 打包
```powershell
python tools/convert_ogg.py
python tools/pack_mpk.py
```

## 环境

| 环境 | 用途 | 安装 |
|------|------|------|
| `.venv-asr` | Whisper ASR | `pip install faster-whisper nvidia-cublas-cu12` |
| `.venv-match` | BGE-M3 匹配 | `pip install sentence-transformers torch numpy` |
| `index-tts/.venv` | IndexTTS2 TTS | `cd index-tts && uv sync --extra webui` |

## 适配新游戏（如 Steins;Gate 0）

1. 创建角色映射 `config/sg0_voice_speaker_map.json`
2. 修改 `tools/paths.py` → `GAME = "sg0"`
3. 准备原始文件放入 `raw/sg0/`（MPK 解包 + 中文台词 txt）
4. 按流水线 1→6 步执行
5. 每次跑完更新 MPK：`python tools/convert_ogg.py && python tools/pack_mpk.py`
