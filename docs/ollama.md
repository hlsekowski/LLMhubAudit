# Ollama Audit Module

The `llmhubaudit.ollama` package pulls, tests, and prunes Ollama models that
support tool calling. It measures cold/warm latency, resource usage (RSS, CPU,
swap), and stores results in a CSV. It implements resumption, interruption
handling, and a composite score for ranking usable models.

## Installation

The core `llmhubaudit` package has no runtime dependencies. The Ollama module
requires `ollama`, `requests`, and `psutil`. Install them with the `ollama`
extra:

```bash
pip install -e ".[ollama]"
```

## Usage

### As a library

```python
from llmhubaudit.ollama import run_test, compute_score

result = run_test("llama3:8b", baseline_swap_used_mb=100.0)
print(result["status"], result["score"])
```

### As a CLI

```bash
python -m llmhubaudit.ollama.cli <model> [options]
```

Or, if installed with a console entry point:

```bash
llmhubaudit-ollama <model> [options]
```

### CLI arguments

| Argument | Description |
|---|---|
| `model` | Single model name to test (e.g. `gemma4:12b`). Required unless `--from-registry` is given. |
| `--force-retest MODEL` | Force retest of a model (can be repeated). |
| `--force-retest-all` | Force retest of every model already in the CSV. |
| `--interrupted-action` | How to resolve an interrupted test from a previous run: `skip`, `retest`, or `delete` (default: `skip`). |
| `--from-registry` | Read candidate models from the Ollama registry catalog instead of testing a single model. |
| `--registry-limit` | Max number of candidate models to fetch from the registry (default: 200). |
| `--min-params` | Minimum parameter size in billions when using `--from-registry` (e.g. `7`). |
| `--max-params` | Maximum parameter size in billions when using `--from-registry` (e.g. `12`). |

## Output

Results are appended to `ollama_test_results.csv` (in the current working
directory). Each row includes:

- **Identity**: `timestamp`, `run_id`, `model`, `status`, `failure_reason`
- **Performance**: `tps`, `duration_seconds`, `score`
- **Tool calling**: `tool_called`, `warm_ok`, `warm_status`, `response_preview`, `tool_call_details`
- **Pull info**: `pulled`, `capabilities`
- **Model info**: `model_size_gb`, `param_size`, `quantization`, `family`
- **Timing**: `ttft_cold_ms`, `ttft_warm_ms`, `itl_cold_p50_ms`, `itl_cold_p95_ms`, `itl_warm_p50_ms`, `itl_warm_p95_ms`, `cold_chunks`, `warm_chunks`, plus `load_duration_ms`, `prompt_eval_count`, `prompt_eval_duration_ms`, `eval_count`, `eval_duration_ms`, `decode_tps`, `prefill_tps`, `total_duration_ms`
- **Resources**: `swap_start_mb`, `swap_peak_mb`, `swap_end_mb`, `swap_out_mb`, `swap_in_mb`, `wired_peak_mb`, `compressed_peak_mb`, `mem_avail_min_mb`, `ps_size_gb`, `ps_vram_gb`, `ps_gpu_pct`, `ps_ctx`, `cpu_peak_pct`, `load_avg_peak`, `mem_samples`, `ollama_rss_peak_mb`, `ollama_cpu_peak_pct`, `baseline_swap_used_mb`

An in-progress marker is kept in `ollama_test_state.json` and cleared when a
test completes.

## Status codes

| Code | Meaning |
|---|---|
| `SUCCESS` | Cold and warm passes both called `add_numbers` correctly. |
| `PARTIAL_COLD_OK` | Cold passed but warm failed. |
| `FAILED_STREAM_ERROR` | Streaming chat errored. |
| `FAILED_NO_TOOL_CALL` | Model replied with text but no tool call. |
| `FAILED_EMPTY_RESPONSE` | Empty response. |
| `FAILED_WRONG_TOOL` | Called a tool other than `add_numbers`. |
| `FAILED_MISSING_ARGS` | Tool call missing `a` or `b`. |
| `FAILED_INVALID_ARGS` | Tool call args were not numbers. |
| `CRASH_SERVER_DOWN` | Ollama server went down during the test. |
| `CRASH_EXCEPTION` | Unexpected exception during the test. |
| `INTERRUPTED` | Test was interrupted in a previous run. |
| `SKIPPED_NO_TOOLS` | Model does not advertise the `tools` capability. |
| `FAILED_PULL` | Model pull failed. |
| `FAILED_DISK_SPACE` | Insufficient disk space. |

## Composite score

The `score` column ranks `SUCCESS` models by usability:

```
score = decode_tps × swap_factor × gpu_factor × warm_factor
```

- `swap_factor` penalizes heavy swap-out (floored at 0.1 for 2 GB swap).
- `gpu_factor` reflects VRAM residency (0.5–1.0).
- `warm_factor` penalizes slow warm TTFT (0.5–1.0).

Non-`SUCCESS` rows score `0.0`.

## Resumption & interruptions

- On startup, models already present in the CSV are skipped unless
  `--force-retest` is used.
- If a previous run was interrupted, the state file records the in-progress
  model. On the next run you are prompted (via `--interrupted-action`) to
  `skip`, `retest`, or `delete` it.
- Disk space is guarded: models are deleted (oldest first) to keep free space
  above 20 %, and pulls are aborted if free space would drop below that
  threshold.

## Module layout

```
src/llmhubaudit/ollama/
├── __init__.py   # public exports
├── cli.py        # CLI entry point + test plan orchestration
├── config.py     # constants (URLs, thresholds, prompts, CSV schema)
├── discovery.py  # model discovery + parameter-size filtering
├── metrics.py    # resource sampling, timing, percentile, composite score
├── reporting.py  # test plan + final summary printing
├── runner.py     # streaming chat, tool-call parsing/validation, test execution
└── state.py      # CSV/history/row writing, JSON state, disk checks, Ollama process mgmt
```

## Testing

Run the unit tests:

```bash
pytest tests/
```

The tests cover tool-call classification (including the `_normalize_tool_call`
shapes from Issue #2 A9), CSV schema migration and history logic, disk-space
guard calculations, parameter-size filtering, and interruption handling.
