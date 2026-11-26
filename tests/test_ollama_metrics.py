"""Unit tests for the Ollama metrics module (percentile, timing, score)."""

from __future__ import annotations

from llmhubaudit.ollama.metrics import (
    _extract_timing,
    _percentile,
    compute_score,
)


class TestPercentile:
    def test_empty(self):
        assert _percentile([], 50) == 0.0

    def test_single(self):
        assert _percentile([10.0], 50) == 10.0

    def test_p50(self):
        assert _percentile([1, 2, 3, 4, 5], 50) == 3.0

    def test_p95(self):
        assert _percentile([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 95) == 10.0


class TestExtractTiming:
    def test_basic(self):
        meta = {
            "load_duration": 1_000_000_000,
            "prompt_eval_count": 10,
            "prompt_eval_duration": 500_000_000,
            "eval_count": 20,
            "eval_duration": 1_000_000_000,
            "total_duration": 2_000_000_000,
        }
        t = _extract_timing(meta, 2.0)
        assert t["load_duration_ms"] == 1000.0
        assert t["prompt_eval_count"] == 10
        assert t["prompt_eval_duration_ms"] == 500.0
        assert t["eval_count"] == 20
        assert t["eval_duration_ms"] == 1000.0
        assert t["total_duration_ms"] == 2000.0
        assert t["prefill_tps"] == 20.0
        assert t["decode_tps"] == 20.0

    def test_missing_values(self):
        t = _extract_timing({}, 0.0)
        assert t["load_duration_ms"] == 0.0
        assert t["prompt_eval_count"] == ""
        assert t["prefill_tps"] == ""
        assert t["decode_tps"] == ""


class TestComputeScore:
    def test_non_success_returns_zero(self):
        assert compute_score({"status": "FAILED_PULL"}) == 0.0
        assert compute_score({"status": "PARTIAL_COLD_OK"}) == 0.0
        assert compute_score({"status": "CRASH_SERVER_DOWN"}) == 0.0

    def test_success_basic(self):
        r = {
            "status": "SUCCESS",
            "timing": {"decode_tps": 50.0},
            "resources": {"swap_out_mb": 0, "ps_gpu_pct": 100},
            "ttft_warm_ms": 100,
        }
        score = compute_score(r)
        assert score > 0
        assert isinstance(score, float)

    def test_swap_penalty(self):
        r_low_swap = {
            "status": "SUCCESS",
            "timing": {"decode_tps": 50.0},
            "resources": {"swap_out_mb": 0, "ps_gpu_pct": 100},
            "ttft_warm_ms": 100,
        }
        r_high_swap = {
            "status": "SUCCESS",
            "timing": {"decode_tps": 50.0},
            "resources": {"swap_out_mb": 1000, "ps_gpu_pct": 100},
            "ttft_warm_ms": 100,
        }
        assert compute_score(r_high_swap) < compute_score(r_low_swap)

    def test_unknown_gpu_neutral(self):
        r = {
            "status": "SUCCESS",
            "timing": {"decode_tps": 50.0},
            "resources": {"swap_out_mb": 0, "ps_gpu_pct": ""},
            "ttft_warm_ms": 100,
        }
        score = compute_score(r)
        assert score > 0
