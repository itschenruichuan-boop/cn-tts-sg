"""Shared paths for SG-TTS pipeline. Change GAME to 'sg0' for Steins;Gate 0."""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GAME = "sg"

RAW = os.path.join(ROOT, "raw", GAME)
RAW_VOICE = os.path.join(RAW, "voice")
RAW_SCRIPTS = os.path.join(RAW, "scripts")
RAW_CN_TXT = os.path.join(RAW, "cn_txt")

OUT = os.path.join(ROOT, "output", GAME)
OUT_ANALYSIS = os.path.join(OUT, "analysis")
OUT_ASR = os.path.join(OUT, "asr")
OUT_MATCHES = os.path.join(OUT, "matches")
OUT_INDEX = os.path.join(OUT, "index")
OUT_TTS_WAV = os.path.join(OUT, "tts_wav")
OUT_FINAL = os.path.join(OUT, "final")

CONFIG = os.path.join(ROOT, "config")
SPEAKER_MAP = os.path.join(CONFIG, "sg_voice_speaker_map.json")
MODELS = os.path.join(ROOT, "models")
BGE_M3 = os.path.join(MODELS, "bge-m3")
