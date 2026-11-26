"""Ollama model audit module for LLMhubAudit.

Pulls, tests, and prunes Ollama models that support tool calling.
Measures cold/warm latency, resource usage (RSS, CPU, swap), and stores
results in a CSV. Implements resumption, interruption handling, and a
composite score.

This package is a refactored, library-friendly version of the standalone
script at ``scripts/ollama/test_ollama_models.py``. That script remains
untouched and does not import from this package.
"""

from .cli import main
from .config import (
    ALL_REQUIRED_COLS,
    BASE_COLS,
    BASE_URL,
    CSV,
    DEFAULT_CANDIDATES,
    DISK_MIN_PCT,
    DISK_TARGET_PULL_PCT,
    DISK_TARGET_TEST_PCT,
    GPU_LOW_PCT,
    LOG,
    MODEL_COLS,
    OLLAMA_START_TIMEOUT_S,
    REGISTRY,
    RESOURCE_COLS,
    SAMPLE_SEC,
    STATE,
    STREAM_COLS,
    SWAP_WARN_OUT_MB,
    TEST_PROMPT_COLD,
    TEST_PROMPT_WARM,
    TEST_TOOL,
    TIMING_COLS,
)
from .discovery import (
    _param_size_b,
    _repo_has_tools,
    _scrape_links,
    fetch_registry_models,
    load_candidates,
)
from .metrics import Sampler, _extract_timing, _percentile, compute_score
from .reporting import print_plan, print_summary
from .runner import (
    _normalize_tool_call,
    _parse_tool_calls_from_content,
    _stream_chat,
    _validate_tool_result,
    run_test,
)
from .state import (
    clear_state,
    csv_fields,
    delete,
    ensure_ollama,
    fetch_registry_size,
    free_pct,
    history,
    installed_models_and_sizes,
    model_info,
    ollama_alive,
    pull,
    read_state,
    resolve_interrupted,
    unload,
    write_row,
    write_state,
)

__all__ = [
    "main",
    "BASE_URL",
    "REGISTRY",
    "CSV",
    "LOG",
    "STATE",
    "DISK_MIN_PCT",
    "DISK_TARGET_PULL_PCT",
    "DISK_TARGET_TEST_PCT",
    "SAMPLE_SEC",
    "SWAP_WARN_OUT_MB",
    "GPU_LOW_PCT",
    "OLLAMA_START_TIMEOUT_S",
    "TEST_PROMPT_COLD",
    "TEST_PROMPT_WARM",
    "TEST_TOOL",
    "DEFAULT_CANDIDATES",
    "ALL_REQUIRED_COLS",
    "BASE_COLS",
    "MODEL_COLS",
    "TIMING_COLS",
    "STREAM_COLS",
    "RESOURCE_COLS",
    "fetch_registry_models",
    "load_candidates",
    "_param_size_b",
    "_repo_has_tools",
    "_scrape_links",
    "run_test",
    "_stream_chat",
    "_validate_tool_result",
    "_normalize_tool_call",
    "_parse_tool_calls_from_content",
    "compute_score",
    "_extract_timing",
    "_percentile",
    "csv_fields",
    "history",
    "write_row",
    "read_state",
    "write_state",
    "clear_state",
    "resolve_interrupted",
    "installed_models_and_sizes",
    "model_info",
    "fetch_registry_size",
    "unload",
    "delete",
    "pull",
    "ensure_ollama",
    "ollama_alive",
    "free_pct",
    "Sampler",
    "print_summary",
    "print_plan",
]
