"""Configuration constants for the Ollama audit module.

Configuration constants for the Ollama audit module.
"""

from __future__ import annotations

BASE_URL = "http://localhost:11434"
REGISTRY = "https://registry.ollama.ai"
CSV = "ollama_test_results.csv"
LOG = "ollama_test_run.log"
STATE = "ollama_test_state.json"

DISK_MIN_PCT = 20.0
DISK_TARGET_PULL_PCT = 35.0
DISK_TARGET_TEST_PCT = 20.0
SAMPLE_SEC = 0.5
SWAP_WARN_OUT_MB = 500
GPU_LOW_PCT = 99.0
OLLAMA_START_TIMEOUT_S = 20

TEST_PROMPT_COLD = "What is 2 + 2? Use the provided tool to calculate."
TEST_PROMPT_WARM = "What is 17 + 25? Use the provided tool to calculate."
TEST_TOOL = {
    "type": "function",
    "function": {
        "name": "add_numbers",
        "description": "Add two numbers together.",
        "parameters": {
            "type": "object",
            "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
            "required": ["a", "b"],
        },
    },
}

DEFAULT_CANDIDATES: list[str] = []

BASE_COLS = [
    "timestamp",
    "run_id",
    "model",
    "status",
    "failure_reason",
    "tps",
    "duration_seconds",
    "tool_called",
    "warm_ok",
    "warm_status",
    "pulled",
    "response_preview",
    "tool_call_details",
    "capabilities",
    "score",
]
MODEL_COLS = ["model_size_gb", "param_size", "quantization", "family"]
TIMING_COLS = [
    "load_duration_ms",
    "prompt_eval_count",
    "prompt_eval_duration_ms",
    "eval_count",
    "eval_duration_ms",
    "decode_tps",
    "prefill_tps",
    "total_duration_ms",
]
STREAM_COLS = [
    "ttft_cold_ms",
    "ttft_warm_ms",
    "itl_cold_p50_ms",
    "itl_cold_p95_ms",
    "itl_warm_p50_ms",
    "itl_warm_p95_ms",
    "cold_chunks",
    "warm_chunks",
]
RESOURCE_COLS = [
    "swap_start_mb",
    "swap_peak_mb",
    "swap_end_mb",
    "swap_out_mb",
    "swap_in_mb",
    "wired_peak_mb",
    "compressed_peak_mb",
    "mem_avail_min_mb",
    "ps_size_gb",
    "ps_vram_gb",
    "ps_gpu_pct",
    "ps_ctx",
    "cpu_peak_pct",
    "load_avg_peak",
    "mem_samples",
    "ollama_rss_peak_mb",
    "ollama_cpu_peak_pct",
    "baseline_swap_used_mb",
]

ALL_REQUIRED_COLS = BASE_COLS + MODEL_COLS + TIMING_COLS + STREAM_COLS + RESOURCE_COLS
