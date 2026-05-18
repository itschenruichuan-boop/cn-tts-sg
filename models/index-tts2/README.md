# IndexTTS2

零样本语音合成模型，用于克隆日文声优音色合成中文语音。

## 安装

IndexTTS2 包含完整 Python 项目（推理代码 + 模型权重），安装在项目根目录 `index-tts/` 下：

```powershell
git clone https://github.com/index-tts/index-tts.git
cd index-tts
uv sync --extra webui
uv run modelscope download --model IndexTeam/IndexTTS-2 --local_dir checkpoints
```

模型权重位于 `index-tts/checkpoints/`（约 3.3GB），代码位于 `index-tts/indextts/`。

## 模型信息

- 模型：IndexTTS-2（Bilibili）
- 大小：~3.3GB（GPT + s2mel + BigVGAN）
- 显存：FP16 模式下 ~8GB
- 速度：RTX 5080 上 ~1.7x RTF
- 论文：https://arxiv.org/abs/2506.21619

## 为什么不在 models/ 下

IndexTTS2 是一个完整的 Python 项目，不仅包含模型权重，还包含推理代码、
WebUI、预处理管线等。其代码需要和 checkpoints 在同一目录下运行。
因此整个项目安装在 `index-tts/`，而非 `models/` 下。
