"""Fixed settings for every run. Changing any of these changes the cache keys or the item sample."""

import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA = os.path.join(ROOT, "data")
RAW = os.path.join(DATA, "raw")
ITEMS = os.path.join(DATA, "items.jsonl")
RESULTS = os.path.join(ROOT, "results")
REPORTS = os.path.join(ROOT, "reports")
FIGURES = os.path.join(REPORTS, "figures")
CACHE_DB = os.path.join(ROOT, "cache", "llm.sqlite")

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")

# One family per row; each level is an Ollama library tag. Ollama's untagged `llama3.1:8b`, `qwen3:8b` and
# `gemma3:12b` are the q4_K_M builds (identical manifest digests), so those names stand in for q4_K_M.
# Two families have the full ladder; qwen3 and Gemma 3 are published at three levels, and Gemma 3 also
# as a quantization-aware-trained 4-bit build (QAT), which is compared with its ordinary q4_K_M.
FAMILIES = {
    "llama3.1-8b": {
        "fp16": "llama3.1:8b-instruct-fp16", "q8_0": "llama3.1:8b-instruct-q8_0",
        "q6_K": "llama3.1:8b-instruct-q6_K", "q5_K_M": "llama3.1:8b-instruct-q5_K_M",
        "q4_K_M": "llama3.1:8b", "q3_K_M": "llama3.1:8b-instruct-q3_K_M", "q2_K": "llama3.1:8b-instruct-q2_K",
    },
    "qwen2.5-7b": {
        "fp16": "qwen2.5:7b-instruct-fp16", "q8_0": "qwen2.5:7b-instruct-q8_0",
        "q6_K": "qwen2.5:7b-instruct-q6_K", "q5_K_M": "qwen2.5:7b-instruct-q5_K_M",
        "q4_K_M": "qwen2.5:7b-instruct-q4_K_M", "q3_K_M": "qwen2.5:7b-instruct-q3_K_M",
        "q2_K": "qwen2.5:7b-instruct-q2_K",
    },
    "qwen3-8b": {"fp16": "qwen3:8b-fp16", "q8_0": "qwen3:8b-q8_0", "q4_K_M": "qwen3:8b"},
    "gemma3-12b": {"fp16": "gemma3:12b-it-fp16", "q8_0": "gemma3:12b-it-q8_0", "q4_K_M": "gemma3:12b",
                   "qat": "gemma3:12b-it-qat"},
}
LEVELS = ["fp16", "q8_0", "q6_K", "q5_K_M", "q4_K_M", "qat", "q3_K_M", "q2_K"]
# Nominal bits per weight of each llama.cpp format, for the x axis (K-quants mix block types, so approximate).
BITS = {"fp16": 16, "q8_0": 8.5, "q6_K": 6.56, "q5_K_M": 5.69, "q4_K_M": 4.85, "qat": 4.5, "q3_K_M": 3.91,
        "q2_K": 3.35}
REFERENCE = "fp16"
# The A/A repeat: GSM8K and MMLU sent a second time at q4_K_M, the noise floor against which a change at
# another level is read. The pilot also repeated fp16 (0 of 80 answers changed); the main run skips it to
# save about 10 hours of fp16 calls.
AA_LEVELS = ["q4_K_M"]


def tag(family: str, level: str) -> str:
    return FAMILIES[family][level]


def all_runs() -> list[tuple[str, str]]:
    return [(f, lv) for f, lv_map in FAMILIES.items() for lv in LEVELS if lv in lv_map]


SEED = 7
BASE_OPTIONS = {"temperature": 0, "seed": SEED, "num_ctx": 4096}
NUM_PREDICT = {"gsm8k": 1024, "mmlu": 768, "drift": 300}

# Pinned Hugging Face sources (the same files as prompt-brittleness and format-tax): (repo, revision, path, sha256).
SOURCES = {
    "gsm8k_test": ("openai/gsm8k", "740312add88f781978c0658806c59bc2815b9866",
                   "main/test-00000-of-00001.parquet",
                   "ee7b8da9e381df27b9e3f7758a159ab2bdaa4dbaa910546cbbc47e0cb44e4f59"),
    "e2e_test": ("tuetschek/e2e_nlg_cleaned", "6f77a74f6ce3095770924452f3f1b9631f837535",
                 "default/test/0000.parquet",
                 "2ffd18a7c46d32dd34cebadf38413cd45dd614e81c25a6e53084a55363f64053"),
}
MMLU_REPO = "cais/mmlu"
MMLU_REVISION = "c30699e8356da336a370243923dbaf21066bb9fe"
MMLU_SHA256 = {
    "high_school_biology": "73600cc5c8032ffd20115887ec4f2c868d75c403c70ac1d115295e72cb8d80e3",
    "high_school_geography": "d083f0a37e9802ff70af8c0ccee95dc6836d80db74f45327519756327076a2c1",
    "high_school_macroeconomics": "a0f2d6a33352452442a13ffb947ab04dcf16513e87bee7c52e8957c2bc761a4c",
    "high_school_psychology": "9df4a744eb6faa12bb359df1f808c0461c055f240b998a9ec9d97b1c4e49a958",
}
MMLU_SUBJECTS = list(MMLU_SHA256)
for _s in MMLU_SUBJECTS:
    SOURCES[f"mmlu_{_s}"] = (MMLU_REPO, MMLU_REVISION, f"{_s}/test-00000-of-00001.parquet", MMLU_SHA256[_s])

# The seeded draws of prompt-brittleness, format-tax and groupthink: 160 GSM8K and 160 MMLU items (the
# same 320 as groupthink) and format-tax's 150 E2E records, each with 10 pilot items drawn first.
N_PER_TASK = 160
N_PILOT = 10
N_DRIFT = 150
N_DRIFT_PILOT = 10
TASKS = ["gsm8k", "mmlu", "drift"]

BOOT = 2000


def options(task: str) -> dict:
    return {**BASE_OPTIONS, "num_predict": NUM_PREDICT[task]}
