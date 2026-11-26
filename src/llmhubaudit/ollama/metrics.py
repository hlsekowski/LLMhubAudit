"""Metrics collection for the Ollama audit module.

Handles system resource sampling (swap, memory, CPU, Ollama process RSS),
timing extraction, percentile computation, and the composite score.
"""

from __future__ import annotations

import os
import threading
from typing import Any, cast

import psutil  # type: ignore[import-untyped]
import requests  # type: ignore[import-untyped]

from .config import SAMPLE_SEC


def _battery_pct() -> str | float:
    try:
        b = psutil.sensors_battery()
        return round(b.percent, 1) if b else ""
    except Exception:
        return ""


def _find_ollama_pids() -> list[int]:
    pids: list[int] = []
    for p in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            info = p.info or {}
            name = (info.get("name") or "").lower()
            cmdline = " ".join(info.get("cmdline") or []).lower()
            if "ollama" in name or "ollama" in cmdline:
                pid = info.get("pid")
                if isinstance(pid, int):
                    pids.append(pid)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return pids


def _percentile(values: list[float], p: float) -> float:
    """Return the p-th percentile of ``values`` (0.0 if empty)."""
    if not values:
        return 0.0
    s = sorted(values)
    k = int(round((len(s) - 1) * p / 100))
    return round(s[k], 1)


def _extract_timing(meta: dict[str, Any], duration_s: float) -> dict[str, Any]:
    """Extract timing metrics from an Ollama stream chunk metadata dict."""

    def _ms(ns: Any) -> float:
        return round((ns or 0) / 1e6, 1)

    load = _ms(meta.get("load_duration", 0))
    pe_count: Any = meta.get("prompt_eval_count", "") or ""
    pe_dur = _ms(meta.get("prompt_eval_duration", 0))
    ev_count: Any = meta.get("eval_count", "") or ""
    ev_dur = _ms(meta.get("eval_duration", 0))
    total = _ms(meta.get("total_duration", 0))

    prefill_tps: Any = ""
    if pe_count and pe_dur:
        try:
            prefill_tps = round(float(pe_count) / (float(pe_dur) / 1000), 2)
        except (TypeError, ValueError, ZeroDivisionError):
            pass

    decode_tps: Any = ""
    if ev_count and ev_dur:
        try:
            decode_tps = round(float(ev_count) / (float(ev_dur) / 1000), 2)
        except (TypeError, ValueError, ZeroDivisionError):
            pass

    return {
        "load_duration_ms": load,
        "prompt_eval_count": pe_count,
        "prompt_eval_duration_ms": pe_dur,
        "eval_count": ev_count,
        "eval_duration_ms": ev_dur,
        "prefill_tps": prefill_tps,
        "decode_tps": decode_tps,
        "total_duration_ms": total,
    }


def _ps_entry(model: str) -> dict[str, Any] | None:
    from .config import BASE_URL

    try:
        r = requests.get(f"{BASE_URL}/api/ps", timeout=5)
        r.raise_for_status()
    except requests.RequestException:
        return None
    base = model.split(":")[0]
    for m in r.json().get("models", []):
        name = m.get("name") or m.get("model") or ""
        if name == model or name.split(":")[0] == base:
            return cast(dict[str, Any], m)
    return None


def _ps_fields(entry: dict[str, Any] | None) -> dict[str, Any]:
    if not entry:
        return {
            "ps_size_gb": "",
            "ps_vram_gb": "",
            "ps_gpu_pct": "",
            "ps_ctx": "",
        }
    size = entry.get("size") or 0
    vram = entry.get("size_vram")
    gpu = round(100.0 * vram / size, 1) if (size and vram is not None) else ""
    return {
        "ps_size_gb": round(size / 1e9, 2) if size else "",
        "ps_vram_gb": round(vram / 1e9, 2) if vram is not None else "",
        "ps_gpu_pct": gpu,
        "ps_ctx": entry.get("context_length", ""),
    }


class Sampler(threading.Thread):
    """Polls system + ollama-process state every SAMPLE_SEC while the main
    thread blocks. Tracks ollama process PIDs, refreshing every ~2s to catch
    per-load runners.
    """

    def __init__(self, model: str) -> None:
        super().__init__(daemon=True)
        self.model = model
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self.samples: list[dict[str, Any]] = []
        self.ps: dict[str, Any] | None = None
        self._ollama_pids = _find_ollama_pids()
        self._pid_refresh_counter = 0

    def _ollama_rss_cpu(self) -> tuple[int, float]:
        rss, cpu = 0, 0.0
        for pid in self._ollama_pids:
            try:
                p = psutil.Process(pid)
                with p.oneshot():
                    rss += p.memory_info().rss
                    cpu += p.cpu_percent(interval=None)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        self._pid_refresh_counter += 1
        if self._pid_refresh_counter % 4 == 0:
            for pid in _find_ollama_pids():
                if pid not in self._ollama_pids:
                    self._ollama_pids.append(pid)
        return rss, cpu

    def _snap(self) -> dict[str, Any]:
        sw = psutil.swap_memory()
        vm = psutil.virtual_memory()
        try:
            load = os.getloadavg()[0]
        except Exception:
            load = 0.0
        rss, cpu = self._ollama_rss_cpu()
        return {
            "swap_used": sw.used,
            "swap_in": sw.sin,
            "swap_out": sw.sout,
            "avail": vm.available,
            "wired": getattr(vm, "wired", 0) or 0,
            "compressed": getattr(vm, "compressed", 0) or 0,
            "cpu": psutil.cpu_percent(interval=None),
            "load": load,
            "ollama_rss": rss,
            "ollama_cpu": cpu,
            "battery": _battery_pct(),
        }

    def run(self) -> None:
        with self._lock:
            self.samples.append(self._snap())
        while not self._stop.wait(SAMPLE_SEC):
            with self._lock:
                self.samples.append(self._snap())
            if self.ps is None:
                e = _ps_entry(self.model)
                if e:
                    with self._lock:
                        self.ps = e
        with self._lock:
            self.samples.append(self._snap())
        if self.ps is None:
            e = _ps_entry(self.model)
            if e:
                with self._lock:
                    self.ps = e

    def stop(self) -> None:
        self._stop.set()

    def summary(self, baseline_swap_used_mb: str | float = "") -> dict[str, Any]:
        with self._lock:
            s = list(self.samples)
            ps = self.ps
        if not s:
            return {}
        first, last = s[0], s[-1]
        out: dict[str, Any] = {
            "swap_start_mb": round(first["swap_used"] / 1e6, 1),
            "swap_peak_mb": round(max(x["swap_used"] for x in s) / 1e6, 1),
            "swap_end_mb": round(last["swap_used"] / 1e6, 1),
            "swap_out_mb": round((last["swap_out"] - first["swap_out"]) / 1e6, 1),
            "swap_in_mb": round((last["swap_in"] - first["swap_in"]) / 1e6, 1),
            "wired_peak_mb": round(max(x["wired"] for x in s) / 1e6, 1),
            "compressed_peak_mb": round(max(x["compressed"] for x in s) / 1e6, 1),
            "mem_avail_min_mb": round(min(x["avail"] for x in s) / 1e6, 1),
            "mem_samples": len(s),
            "cpu_peak_pct": round(max(x["cpu"] for x in s), 1),
            "load_avg_peak": round(max(x["load"] for x in s), 2),
            "ollama_rss_peak_mb": round(max(x["ollama_rss"] for x in s) / 1e6, 1),
            "ollama_cpu_peak_pct": round(max(x["ollama_cpu"] for x in s) / 1e6, 1),
            "baseline_swap_used_mb": baseline_swap_used_mb,
        }
        out.update(_ps_fields(ps))
        return out


def compute_score(r: dict[str, Any]) -> float:
    """Compute a composite usability score for a result row.

    Returns 0.0 for anything except SUCCESS. Otherwise:
    decode_speed × swap_factor × gpu_factor × warm_factor.
    """
    if r.get("status") != "SUCCESS":
        return 0.0

    timing = r.get("timing") or {}
    decode: Any = timing.get("decode_tps")
    if decode in ("", None):
        decode = r.get("tps", 0) or 0
    try:
        decode = float(decode)
    except (TypeError, ValueError):
        decode = 0.0

    res = r.get("resources") or {}
    swap_out: Any = res.get("swap_out_mb", 0) or 0
    try:
        swap_out = float(swap_out)
    except (TypeError, ValueError):
        swap_out = 0.0
    swap_factor = max(0.1, 1.0 - swap_out / 2000)  # 2 GB swap → 0.1 floor

    gpu_val: Any = res.get("ps_gpu_pct")
    try:
        gpu_factor = 0.5 + 0.5 * (float(gpu_val) / 100.0)
    except (TypeError, ValueError):
        gpu_factor = 0.7  # unknown → neutral

    warm_ttft: Any = r.get("ttft_warm_ms", 0) or 0
    try:
        warm_ttft = float(warm_ttft)
    except (TypeError, ValueError):
        warm_ttft = 0.0
    warm_factor = max(0.5, 1.0 - warm_ttft / 4000.0) if warm_ttft > 0 else 0.7

    return round(float(decode * swap_factor * gpu_factor * warm_factor), 2)
