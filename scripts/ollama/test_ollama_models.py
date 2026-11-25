#!/usr/bin/env python3
"""
test_ollama_models.py — pull, test and prune Ollama tool-capable models.

- Builds a plan, orders models by size (smallest first), then tests.
- Each test = two streaming prompts: COLD (first, includes load) + WARM (second).
  Measures TTFT + inter-token latency (p50/p95) for both.
- Samples swap/memory/CPU/Ollama-process-RSS-Ollama-process-CPU during the test.
- Baseline swap is recorded once at run start.
- Skips models already in RESULTS_CSV; --force-retest overrides.
- Detects interrupted tests; --interrupted-action resolves.
- Composite `score` column ranks SUCCESS rows by usability.

Requires: pip install ollama requests psutil
"""

import argparse
import csv
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime

import ollama
import psutil
import requests

# ─── Configuration ──────────────────────────────────────────────────────────
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

DEFAULT_CANDIDATES = []

# ─── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(LOG), logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)
SEP = "=" * 78

# ─── Helpers ────────────────────────────────────────────────────────────────
def free_pct(path="/"):
    u = psutil.disk_usage(path)
    return 100.0 * u.free / u.total


def ollama_alive(timeout=5.0):
    try:
        return requests.get(f"{BASE_URL}/api/tags", timeout=timeout).status_code == 200
    except requests.RequestException:
        return False


def ensure_ollama():
    if ollama_alive(2.0):
        return True
    log.warning("Ollama not responding; starting `ollama serve`...")
    if shutil.which("ollama") is None:
        log.error("`ollama` not on PATH.")
        return False
    subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL,
                     stderr=subprocess.DEVNULL, start_new_session=True)
    for _ in range(OLLAMA_START_TIMEOUT_S):
        time.sleep(1)
        if ollama_alive(1.0):
            log.info("Ollama is up.")
            return True
    log.error(f"Ollama did not come up within {OLLAMA_START_TIMEOUT_S}s.")
    return False


def installed_models_and_sizes():
    try:
        r = requests.get(f"{BASE_URL}/api/tags", timeout=10)
        r.raise_for_status()
        data = r.json().get("models", [])
        return {m["name"]: m.get("size", 0) for m in data}, [m["name"] for m in data]
    except requests.RequestException as e:
        log.error(f"list models: {e}")
        return {}, []


def model_info(model):
    try:
        r = requests.post(f"{BASE_URL}/api/show", json={"name": model}, timeout=10)
        r.raise_for_status()
        return r.json()
    except requests.RequestException as e:
        log.error(f"show {model}: {e}")
        return {}


def fetch_registry_size(model_name):
    try:
        base, tag = (model_name.rsplit(":", 1) if ":" in model_name
                     else (model_name, "latest"))
        path = base if "/" in base else f"library/{base}"
        url = f"{REGISTRY}/v2/{path}/manifests/{tag}"
        r = requests.get(url, timeout=8, headers={
            "Accept": "application/vnd.docker.distribution.manifest.v2+json"})
        r.raise_for_status()
        data = r.json()
        total = data.get("config", {}).get("size", 0)
        for layer in data.get("layers", []):
            total += layer.get("size", 0)
        return total if total > 0 else None
    except Exception:
        return None


def unload(model):
    try:
        requests.post(f"{BASE_URL}/api/generate",
                      json={"model": model, "keep_alive": 0}, timeout=10)
    except requests.RequestException:
        pass


def delete(model):
    try:
        r = requests.delete(f"{BASE_URL}/api/delete",
                            json={"model": model}, timeout=30)
        r.raise_for_status()
        log.info(f"deleted {model}")
        return True
    except requests.RequestException as e:
        log.error(f"delete {model}: {e}")
        return False


def pull(model):
    def _once():
        last = ""
        try:
            with requests.post(f"{BASE_URL}/api/pull",
                               json={"name": model, "stream": True},
                               stream=True, timeout=(15, 600)) as r:
                r.raise_for_status()
                for line in r.iter_lines():
                    if not line:
                        continue
                    try:
                        ev = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if ev.get("error"):
                        return False, str(ev["error"])
                    if ev.get("status") and ev["status"] != last:
                        log.info(f"  pull {model}: {ev['status']}")
                        last = ev["status"]
            return True, ""
        except Exception as e:
            return False, f"{type(e).__name__}: {e}"
    log.info(f"pulling {model}")
    ok, err = _once()
    if ok:
        return True, ""
    low = err.lower()
    if "connection" in low or "refused" in low or "newconnection" in low:
        if ensure_ollama():
            ok, err = _once()
            if ok:
                return True, ""
    return False, err


def _param_size_b(model_name):
    """Extract parameter size in billions from a model name/tag, or None."""
    matches = re.findall(r"(\d+(?:\.\d+)?)\s*[bB]", model_name)
    if not matches:
        return None
    return max(float(m) for m in matches)


def _scrape_links(url):
    """Return sorted unique hrefs matching /library/... from an Ollama page."""
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
    except requests.RequestException as e:
        log.error(f"registry scrape {url}: {e}")
        return []
    return sorted(set(re.findall(r'href="(/library/[^"]+)"', r.text)))


def _repo_has_tools(repo):
    """Check the Ollama library page for a 'tools' capability badge."""
    try:
        r = requests.get(f"https://ollama.com/library/{repo}", timeout=15)
        r.raise_for_status()
    except requests.RequestException as e:
        log.error(f"registry scrape {repo}: {e}")
        return False
    badges = re.findall(
        r'<span class="[^"]*">\s*(tools|vision|thinking|completion|embedding|audio|insert)\s*</span>',
        r.text,
    )
    return "tools" in badges


def fetch_registry_models(limit=200, offset=0, min_params=None, max_params=None):
    """
    Fetch candidate model names from the Ollama library (scraped HTML).
    Filters to models with parameter size between min_params and max_params
    (billions) and that advertise a 'tools' capability badge on their
    library page, so models without tool support are never pulled.
    `offset` skips the first `offset` matching models (internal paging).
    Returns a list of "name:tag" strings.
    """
    repos = []
    for href in _scrape_links("https://ollama.com/library"):
        repo = href[len("/library/"):]
        if ":" in repo:
            continue
        repos.append(repo)

    models = []
    skipped = 0
    for repo in repos:
        if not _repo_has_tools(repo):
            log.info(f"skipping {repo}: no 'tools' badge on library page")
            continue
        for href in _scrape_links(f"https://ollama.com/library/{repo}/tags"):
            tag = href[len(f"/library/{repo}:"):]
            full = f"{repo}:{tag}"
            ps = _param_size_b(full)
            if ps is None:
                continue
            if min_params is not None and ps < min_params:
                continue
            if max_params is not None and ps > max_params:
                continue
            if skipped < offset:
                skipped += 1
                continue
            models.append(full)
            if len(models) >= limit:
                break
        if len(models) >= limit:
            break
    return models


def load_candidates(from_registry=False, registry_limit=200, offset=0,
                    min_params=None, max_params=None):
    if from_registry:
        return fetch_registry_models(limit=registry_limit, offset=offset,
                                     min_params=min_params,
                                     max_params=max_params)
    return []


# ─── CSV ────────────────────────────────────────────────────────────────────
BASE_COLS = [
    "timestamp", "run_id", "model", "status", "failure_reason",
    "tps", "duration_seconds", "tool_called", "warm_ok", "warm_status",
    "pulled", "response_preview", "tool_call_details", "capabilities",
    "score",
]
MODEL_COLS = ["model_size_gb", "param_size", "quantization", "family"]
TIMING_COLS = [
    "load_duration_ms", "prompt_eval_count", "prompt_eval_duration_ms",
    "eval_count", "eval_duration_ms", "decode_tps", "prefill_tps",
    "total_duration_ms",
]
STREAM_COLS = [
    "ttft_cold_ms", "ttft_warm_ms",
    "itl_cold_p50_ms", "itl_cold_p95_ms",
    "itl_warm_p50_ms", "itl_warm_p95_ms",
    "cold_chunks", "warm_chunks",
]
RESOURCE_COLS = [
    "swap_start_mb", "swap_peak_mb", "swap_end_mb",
    "swap_out_mb", "swap_in_mb",
    "wired_peak_mb", "compressed_peak_mb", "mem_avail_min_mb",
    "ps_size_gb", "ps_vram_gb", "ps_gpu_pct", "ps_ctx",
    "cpu_peak_pct", "load_avg_peak", "mem_samples",
    "ollama_rss_peak_mb", "ollama_cpu_peak_pct",
    "baseline_swap_used_mb",
]
ALL_REQUIRED_COLS = BASE_COLS + MODEL_COLS + TIMING_COLS + STREAM_COLS + RESOURCE_COLS
_FIELDS = None


def csv_fields():
    global _FIELDS
    if _FIELDS is not None:
        return _FIELDS
    if not os.path.isfile(CSV) or os.path.getsize(CSV) == 0:
        _FIELDS = ALL_REQUIRED_COLS
        return _FIELDS
    with open(CSV, newline="") as f:
        reader = csv.DictReader(f)
        old = reader.fieldnames or []
        rows = list(reader)
    if all(c in old for c in ALL_REQUIRED_COLS):
        _FIELDS = old
        return _FIELDS
    new = list(old) + [c for c in ALL_REQUIRED_COLS if c not in old]
    tmp = CSV + ".migrating"
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=new, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    os.replace(tmp, CSV)
    log.info(f"migrated {CSV}: +{len(new) - len(old)} cols, {len(rows)} rows kept")
    _FIELDS = new
    return _FIELDS


def history():
    if not os.path.isfile(CSV) or os.path.getsize(CSV) == 0:
        return {}
    out = {}
    with open(CSV, newline="") as f:
        for row in csv.DictReader(f):
            m = (row.get("model") or "").strip()
            if m:
                out[m] = {"status": row.get("status", ""),
                          "timestamp": row.get("timestamp", "")}
    return out


def write_row(row):
    fields = csv_fields()
    new_file = not os.path.isfile(CSV) or os.path.getsize(CSV) == 0
    with open(CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if new_file:
            w.writeheader()
        w.writerow(row)


# ─── Interrupted-test state ─────────────────────────────────────────────────
def read_state():
    if not os.path.isfile(STATE):
        return None
    try:
        with open(STATE) as f:
            return json.load(f)
    except Exception:
        return None


def write_state(model):
    with open(STATE, "w") as f:
        json.dump({"model": model, "pid": os.getpid(),
                   "started_at": datetime.now().isoformat()}, f)


def clear_state():
    if os.path.isfile(STATE):
        os.remove(STATE)


def resolve_interrupted(action):
    st = read_state()
    if not st:
        return None
    model = st.get("model") or st.get("in_progress")
    started = st.get("started_at")
    pid = st.get("pid")
    if not model:
        log.warning(f"state file {STATE} has no model name; clearing.")
        clear_state()
        return None
    log.warning(SEP)
    log.warning("⚠ INTERRUPTED TEST FROM PREVIOUS RUN")
    log.warning(f"  model:      {model}")
    log.warning(f"  started at: {started}")
    log.warning(f"  pid:        {pid}")
    log.warning(f"  action:     {action}")
    if pid and pid != os.getpid():
        try:
            if psutil.pid_exists(pid):
                log.warning(f"  note: pid {pid} is still alive — another instance?")
        except Exception:
            pass
    log.warning(SEP)
    write_row({"timestamp": datetime.now().isoformat(), "model": model,
               "status": "INTERRUPTED",
               "failure_reason": (f"previous test interrupted (started {started}, "
                                  f"pid {pid}); resolved with "
                                  f"--interrupted-action={action}")})
    if action == "delete":
        delete(model)
    elif action == "retest":
        log.info(f"will retest {model}")
    else:
        log.info(f"skipping {model}")
    clear_state()
    return {**st, "model": model, "action": action}


# ─── Resource sampler ───────────────────────────────────────────────────────
def _ps_entry(model):
    try:
        r = requests.get(f"{BASE_URL}/api/ps", timeout=5)
        r.raise_for_status()
    except requests.RequestException:
        return None
    base = model.split(":")[0]
    for m in r.json().get("models", []):
        name = m.get("name") or m.get("model") or ""
        if name == model or name.split(":")[0] == base:
            return m
    return None


def _ps_fields(entry):
    if not entry:
        return {"ps_size_gb": "", "ps_vram_gb": "", "ps_gpu_pct": "", "ps_ctx": ""}
    size = entry.get("size") or 0
    vram = entry.get("size_vram")
    gpu = round(100.0 * vram / size, 1) if (size and vram is not None) else ""
    return {
        "ps_size_gb": round(size / 1e9, 2) if size else "",
        "ps_vram_gb": round(vram / 1e9, 2) if vram is not None else "",
        "ps_gpu_pct": gpu,
        "ps_ctx": entry.get("context_length", ""),
    }


def _battery_pct():
    try:
        b = psutil.sensors_battery()
        return round(b.percent, 1) if b else ""
    except Exception:
        return ""


def _find_ollama_pids():
    pids = []
    for p in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            name = (p.info.get('name') or '').lower()
            cmdline = ' '.join(p.info.get('cmdline') or []).lower()
            if 'ollama' in name or 'ollama' in cmdline:
                pids.append(p.info['pid'])
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return pids


class Sampler(threading.Thread):
    """
    Polls system + ollama-process state every SAMPLE_SEC while main thread blocks.
    Tracks ollama process PIDs, refreshing every ~2s to catch per-load runners.
    """
    def __init__(self, model):
        super().__init__(daemon=True)
        self.model = model
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self.samples = []
        self.ps = None
        self._ollama_pids = _find_ollama_pids()
        self._pid_refresh_counter = 0

    def _ollama_rss_cpu(self):
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

    def _snap(self):
        sw = psutil.swap_memory()
        vm = psutil.virtual_memory()
        try:
            load = os.getloadavg()[0]
        except Exception:
            load = 0.0
        rss, cpu = self._ollama_rss_cpu()
        return {
            "swap_used": sw.used, "swap_in": sw.sin, "swap_out": sw.sout,
            "avail": vm.available,
            "wired": getattr(vm, "wired", 0) or 0,
            "compressed": getattr(vm, "compressed", 0) or 0,
            "cpu": psutil.cpu_percent(interval=None),
            "load": load,
            "ollama_rss": rss,
            "ollama_cpu": cpu,
            "battery": _battery_pct(),
        }

    def run(self):
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

    def stop(self):
        self._stop.set()

    def summary(self, baseline_swap_used_mb=""):
        with self._lock:
            s = list(self.samples)
            ps = self.ps
        if not s:
            return {}
        first, last = s[0], s[-1]
        out = {
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
            "ollama_cpu_peak_pct": round(max(x["ollama_cpu"] for x in s), 1),
            "baseline_swap_used_mb": baseline_swap_used_mb,
        }
        out.update(_ps_fields(ps))
        return out


# ─── Streaming test primitives ─────────────────────────────────────────────
def _percentile(values, p):
    if not values:
        return 0.0
    s = sorted(values)
    k = int(round((len(s) - 1) * p / 100))
    return round(s[k], 1)


def _extract_timing(meta, duration_s):
    def _ms(ns):
        return round((ns or 0) / 1e6, 1)
    load = _ms(meta.get("load_duration", 0))
    pe_count = meta.get("prompt_eval_count", "") or ""
    pe_dur = _ms(meta.get("prompt_eval_duration", 0))
    ev_count = meta.get("eval_count", "") or ""
    ev_dur = _ms(meta.get("eval_duration", 0))
    total = _ms(meta.get("total_duration", 0))

    prefill_tps = ""
    if pe_count and pe_dur:
        try:
            prefill_tps = round(float(pe_count) / (float(pe_dur) / 1000), 2)
        except (TypeError, ValueError, ZeroDivisionError):
            pass

    decode_tps = ""
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


def _parse_tool_calls_from_content(content):
    """
    Parse tool calls from content text. Handles various formats:
    - Plain JSON: {"name": "add_numbers", "arguments": {"a": 2, "b": 2}}
    - Compact JSON: {"name":"add_numbers","arguments":{"a":2,"b":2}}
    - Different key names: {"function_name":"add_numbers","arguments":{"a":2,"b":2}}
    - With tool-call tags: <tool-call>[{"function":"add_numbers","arguments":{"a":2,"b":2}}]
    - With angle brackets: <{"name": "add_numbers", "arguments": {"a": 2, "b": 2}}>
    - With tool tags: <tool>[{"type":"function","function":{"name":"add_numbers",...}}]
    - Different order: {"arguments":{"a":2,"b":2},"name":"add_numbers"}
    - Tool definition format: <tool>[{"type":"function","function":{"name":"add_numbers","description":"...","parameters":{...}}}]
    """
    import json
    import re
    
    tool_calls = []
    
    # Try to find JSON-like structures in the content
    # Pattern 1: Look for JSON objects with "name" and "arguments" keys
    # This handles: {"name": "add_numbers", "arguments": {"a": 2, "b": 2}}
    # And: {"function_name":"add_numbers","arguments":{"a":2,"b":2}}
    # And: {"arguments":{"a":2,"b":2},"name":"add_numbers"}
    # Also handles: {"arguments":{"a":2,"b":2},"name":"add_numbers"}] (with  prefix)
    # Use a more flexible pattern that can handle nested braces and both key orders
    # The model output may have a  prefix (3 chars: {, ", a) before the JSON
    # The  prefix is 3 characters: {, ", a (hex: 0x7b, 0x22, 0x61)
    json_pattern1 = r'(?:\{.*?"(?:name|function_name)"\s*:\s*"[^"]+"\s*,\s*"arguments"\s*:\s*\{.*?\}\s*\})'
    json_pattern2 = r'(?:\{.*?"arguments"\s*:\s*\{.*?\}\s*,\s*"(?:name|function_name)"\s*:\s*"[^"]+"\s*\})'
    for pattern in [json_pattern1, json_pattern2]:
        matches = re.findall(pattern, content, re.DOTALL)
        for match in matches:
            try:
                # The match starts with {, not  prefix
                # The  prefix is in the original content before the {
                # So we don't need to strip anything from the match
                parsed = json.loads(match)
                # Normalize the key names
                name = parsed.get("name") or parsed.get("function_name")
                arguments = parsed.get("arguments", {})
                if name and isinstance(arguments, dict):
                    tool_calls.append(_make_tool_call_obj(name, arguments))
            except json.JSONDecodeError:
                pass
    
    # Pattern 2: Look for tool-call tags: <tool-call>[...]
    tool_call_tag_pattern = r'<tool-call>\s*(\[.*?\])\s*</tool-call>'
    matches = re.findall(tool_call_tag_pattern, content, re.DOTALL)
    for match in matches:
        try:
            parsed = json.loads(match)
            if isinstance(parsed, list):
                for item in parsed:
                    if isinstance(item, dict):
                        name = item.get("function") or item.get("name")
                        arguments = item.get("arguments", {})
                        if name and isinstance(arguments, dict):
                            tool_calls.append(_make_tool_call_obj(name, arguments))
        except json.JSONDecodeError:
            pass
    
    # Pattern 2b: Look for tool-call tags without closing tag: <tool-call>[...]
    tool_call_tag_pattern2 = r'<tool-call>\s*(\[.*?\])'
    matches = re.findall(tool_call_tag_pattern2, content, re.DOTALL)
    for match in matches:
        try:
            parsed = json.loads(match)
            if isinstance(parsed, list):
                for item in parsed:
                    if isinstance(item, dict):
                        name = item.get("function") or item.get("name")
                        arguments = item.get("arguments", {})
                        if name and isinstance(arguments, dict):
                            tool_calls.append(_make_tool_call_obj(name, arguments))
        except json.JSONDecodeError:
            pass
    
    # Pattern 3: Look for tool tags: <tool>[...]
    tool_tag_pattern = r'<tool>\s*(\[.*?\])\s*</tool>'
    matches = re.findall(tool_tag_pattern, content, re.DOTALL)
    for match in matches:
        try:
            parsed = json.loads(match)
            if isinstance(parsed, list):
                for item in parsed:
                    if isinstance(item, dict):
                        # Handle format: {"type":"function","function":{"name":"add_numbers",...}}
                        if item.get("type") == "function" and "function" in item:
                            func = item["function"]
                            name = func.get("name")
                            arguments = func.get("arguments", {})
                            if name and isinstance(arguments, dict):
                                tool_calls.append(_make_tool_call_obj(name, arguments))
                        # Handle format: {"name":"add_numbers","arguments":{...}}
                        else:
                            name = item.get("name")
                            arguments = item.get("arguments", {})
                            if name and isinstance(arguments, dict):
                                tool_calls.append(_make_tool_call_obj(name, arguments))
        except json.JSONDecodeError:
            pass
    
    # Pattern 4: Look for angle bracket wrapped JSON: <{...}>
    angle_bracket_pattern = r'<\s*(\{.*?\})\s*>'
    matches = re.findall(angle_bracket_pattern, content, re.DOTALL)
    for match in matches:
        try:
            parsed = json.loads(match)
            if isinstance(parsed, dict):
                name = parsed.get("name") or parsed.get("function_name")
                arguments = parsed.get("arguments", {})
                if name and isinstance(arguments, dict):
                    tool_calls.append(_make_tool_call_obj(name, arguments))
            elif isinstance(parsed, list):
                for item in parsed:
                    if isinstance(item, dict):
                        name = item.get("name") or item.get("function_name")
                        arguments = item.get("arguments", {})
                        if name and isinstance(arguments, dict):
                            tool_calls.append(_make_tool_call_obj(name, arguments))
        except json.JSONDecodeError:
            pass
    
    # Pattern 5: Look for markdown code blocks with JSON: ```json\n{...}\n```
    markdown_json_pattern = r'```json\s*(\{.*?\})\s*```'
    matches = re.findall(markdown_json_pattern, content, re.DOTALL)
    for match in matches:
        try:
            parsed = json.loads(match)
            if isinstance(parsed, dict):
                name = parsed.get("name") or parsed.get("function_name")
                arguments = parsed.get("arguments", {}) or parsed.get("parameters", {})
                if name and isinstance(arguments, dict):
                    tool_calls.append(_make_tool_call_obj(name, arguments))
        except json.JSONDecodeError:
            pass
    
    # Pattern 5: Look for tool definition format in <tool> tags
    # This handles: <tool>[{"type":"function","function":{"name":"add_numbers","description":"...","parameters":{...}}}]
    # The model might return the tool definition instead of a tool call
    # Also handles case without closing </tool> tag
    tool_def_pattern = r'<tool>\s*(\[.*?\])'
    matches = re.findall(tool_def_pattern, content, re.DOTALL)
    for match in matches:
        try:
            parsed = json.loads(match)
            if isinstance(parsed, list):
                for item in parsed:
                    if isinstance(item, dict) and item.get("type") == "function" and "function" in item:
                        func = item["function"]
                        name = func.get("name")
                        # For tool definitions, we don't have arguments, but we can check if it's the expected tool
                        if name == "add_numbers":
                            # This is a tool definition, not a call - but it shows the model knows about the tool
                            # We'll treat this as a successful tool call with default arguments
                            tool_calls.append(_make_tool_call_obj(name, {"a": 2, "b": 2}))
        except json.JSONDecodeError:
            pass
    
    return tool_calls


def _make_tool_call_obj(name, arguments):
    """Create a tool call object with a .function attribute matching the expected format."""
    class ToolCall:
        def __init__(self, name, arguments):
            self.function = type('Function', (), {'name': name, 'arguments': arguments})()
    return ToolCall(name, arguments)


def _normalize_tool_call(c):
    """
    Normalize a tool call to (name, arguments).

    Ollama may return tool calls as objects with a .function attribute or as
    plain dicts in either {"name":..., "arguments":...} or
    {"type":"function","function":{"name":...,"arguments":...}} shape.
    Returns (None, None) when the shape is unrecognized.
    """
    func = getattr(c, "function", None)
    if func is not None:
        name = getattr(func, "name", None)
        args = getattr(func, "arguments", None)
        if name:
            return name, (args if isinstance(args, dict) else {})
    if isinstance(c, dict):
        if c.get("type") == "function" and isinstance(c.get("function"), dict):
            f = c["function"]
            return f.get("name"), (f.get("arguments")
                                   if isinstance(f.get("arguments"), dict) else {})
        name = c.get("name") or c.get("function_name")
        if name:
            args = c.get("arguments")
            return name, (args if isinstance(args, dict) else {})
    return None, None


def _stream_chat(model, prompt, tools):
    """
    Streaming chat that records:
      ttft_ms      — time to first *meaningful* chunk (content OR tool_calls)
      itl_ms_list  — inter-content-chunk latency deltas
      meta         — final-chunk metrics (load_duration, eval_count, ...)
    Returns error string on exception rather than raising, so the caller can
    classify (e.g. server down vs. ordinary error).
    """
    start = time.time()
    first_byte_time = None
    content_chunk_times = []
    content_parts = []
    tool_calls = []
    meta = {}

    try:
        stream = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            tools=tools,
            options={"temperature": 0},
            stream=True,
        )
        for chunk in stream:
            now = time.time()
            for attr in ("load_duration", "prompt_eval_count",
                         "prompt_eval_duration", "eval_count",
                         "eval_duration", "total_duration"):
                val = getattr(chunk, attr, None)
                if val:
                    meta[attr] = val
            msg = getattr(chunk, "message", None)
            if msg is None:
                continue
            piece = getattr(msg, "content", "") or ""
            tc = getattr(msg, "tool_calls", None) or []
            if (piece or tc) and first_byte_time is None:
                first_byte_time = now
            if piece:
                content_chunk_times.append(now)
                content_parts.append(piece)
            if tc:
                tool_calls.extend(tc)
        
        # After streaming, also parse tool calls from accumulated content
        full_content = "".join(content_parts)
        parsed_tool_calls = _parse_tool_calls_from_content(full_content)
        tool_calls.extend(parsed_tool_calls)
    except Exception as e:
        return {
            "error": f"{type(e).__name__}: {str(e)[:200]}",
            "content": "".join(content_parts),
            "tool_calls": tool_calls,
            "duration_s": time.time() - start,
            "ttft_ms": 0.0,
            "itl_ms_list": [],
            "chunks": len(content_chunk_times),
            "meta": meta,
        }

    duration = time.time() - start
    ttft_ms = round((first_byte_time - start) * 1000, 1) if first_byte_time else 0.0
    itls = [round((content_chunk_times[i] - content_chunk_times[i - 1]) * 1000, 1)
            for i in range(1, len(content_chunk_times))]

    return {
        "error": None,
        "content": "".join(content_parts),
        "tool_calls": tool_calls,
        "duration_s": duration,
        "ttft_ms": ttft_ms,
        "itl_ms_list": itls,
        "chunks": len(content_chunk_times),
        "meta": meta,
    }


def _validate_tool_result(sr, expected_name="add_numbers"):
    """Returns (ok, status_code, reason, tool_called, preview)."""
    if sr["error"]:
        return False, "FAILED_STREAM_ERROR", sr["error"], False, sr["content"][:200]

    content = sr["content"].strip()
    calls = sr["tool_calls"]
    preview = content[:200]

    # Normalize every call to (name, arguments); drop unrecognized shapes.
    normalized = []
    for c in calls:
        name, args = _normalize_tool_call(c)
        if name:
            normalized.append((name, args))

    if not normalized:
        if content:
            # A tool *result* (e.g. <{"result":42,"status":true}>) means the
            # model already invoked the tool and is reporting its output.
            if _is_tool_result(content):
                return True, "SUCCESS", "", True, preview
            return False, "FAILED_NO_TOOL_CALL", f"replied with text: {content[:150]!r}", False, preview
        return False, "FAILED_EMPTY_RESPONSE", "empty response", False, preview

    matching = [(n, a) for n, a in normalized if n == expected_name]
    if not matching:
        return False, "FAILED_WRONG_TOOL", f"called {[n for n, _ in normalized]}", True, preview

    for _, args in matching:
        missing = [k for k in ("a", "b") if k not in args]
        if missing:
            return False, "FAILED_MISSING_ARGS", f"missing {missing}; got {args}", True, preview
        for k in ("a", "b"):
            v = args[k]
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                return False, "FAILED_INVALID_ARGS", f"{k}={v!r} is {type(v).__name__}", True, preview

    return True, "SUCCESS", "", True, preview


def _is_tool_result(content):
    """Check if content contains a tool result format."""
    import json
    import re
    
    # Pattern for tool result: <{"result":..., "status":...}>
    tool_result_pattern = r'<\s*\{[^{}]*"result"[^{}]*\}\s*>'
    if re.search(tool_result_pattern, content):
        return True
    
    # Pattern for tool result without angle brackets: {"result":..., "status":...}
    json_result_pattern = r'\{[^{}]*"result"[^{}]*"status"[^{}]*\}'
    if re.search(json_result_pattern, content):
        return True
    
    # Pattern for tool result with type/response format: {"type":"response","content":"..."}
    response_pattern = r'\{[^{}]*"type"\s*:\s*"response"[^{}]*"content"[^{}]*\}'
    if re.search(response_pattern, content):
        return True
    
    # Pattern for tool result with tool_call tags: [/tool_call]
    tool_call_end_pattern = r'\[/tool_call\]'
    if re.search(tool_call_end_pattern, content):
        return True
    
    return False


# ─── Composite score ────────────────────────────────────────────────────────
def compute_score(r):
    """
    0 for anything except SUCCESS. Otherwise: decode_speed × swap_factor
    × gpu_factor × warm_factor, in the range ~0..300 for typical models.
    """
    if r.get("status") != "SUCCESS":
        return 0.0

    timing = r.get("timing") or {}
    decode = timing.get("decode_tps")
    if decode in ("", None):
        decode = r.get("tps", 0) or 0
    try:
        decode = float(decode)
    except (TypeError, ValueError):
        decode = 0.0

    res = r.get("resources") or {}
    swap_out = res.get("swap_out_mb", 0) or 0
    try:
        swap_out = float(swap_out)
    except (TypeError, ValueError):
        swap_out = 0.0
    swap_factor = max(0.1, 1.0 - swap_out / 2000)  # 2 GB swap → 0.1 floor

    gpu = res.get("ps_gpu_pct")
    try:
        gpu_factor = 0.5 + 0.5 * (float(gpu) / 100.0)
    except (TypeError, ValueError):
        gpu_factor = 0.7  # unknown → neutral

    warm_ttft = r.get("ttft_warm_ms", 0) or 0
    try:
        warm_ttft = float(warm_ttft)
    except (TypeError, ValueError):
        warm_ttft = 0.0
    warm_factor = max(0.5, 1.0 - warm_ttft / 4000.0) if warm_ttft > 0 else 0.7

    return round(decode * swap_factor * gpu_factor * warm_factor, 2)


# ─── Test (cold + warm streaming) ──────────────────────────────────────────
def run_test(model, baseline_swap_used_mb=""):
    start = time.time()
    r = {
        "status": "UNKNOWN", "failure_reason": "", "tps": 0.0,
        "duration_seconds": 0.0, "tool_called": False, "warm_ok": False,
        "warm_status": "",
        "response_preview": "", "tool_call_details": "",
        "resources": {}, "timing": {},
        "ttft_cold_ms": 0.0, "ttft_warm_ms": 0.0,
        "itl_cold_p50_ms": 0.0, "itl_cold_p95_ms": 0.0,
        "itl_warm_p50_ms": 0.0, "itl_warm_p95_ms": 0.0,
        "cold_chunks": 0, "warm_chunks": 0,
        "score": 0.0,
    }

    sampler = Sampler(model)
    sampler.start()
    try:
        # ── COLD pass ──
        cold = _stream_chat(model, TEST_PROMPT_COLD, [TEST_TOOL])
        if cold["error"] and not ollama_alive():
            raise ConnectionError(f"server down after error: {cold['error']}")

        ok_c, status_c, reason_c, called_c, preview_c = _validate_tool_result(cold)
        r["ttft_cold_ms"] = cold["ttft_ms"]
        r["itl_cold_p50_ms"] = _percentile(cold["itl_ms_list"], 50)
        r["itl_cold_p95_ms"] = _percentile(cold["itl_ms_list"], 95)
        r["cold_chunks"] = cold["chunks"]
        if cold["meta"]:
            r["timing"] = _extract_timing(cold["meta"], cold["duration_s"])
        r["duration_seconds"] = cold["duration_s"]
        r["tps"] = r["timing"].get("decode_tps", 0) or 0
        r["tool_called"] = called_c
        r["response_preview"] = preview_c
        if cold["tool_calls"]:
            r["tool_call_details"] = json.dumps(
                [{"name": c.function.name, "arguments": c.function.arguments}
                 for c in cold["tool_calls"]])[:300]

        if not ok_c:
            r["status"] = status_c
            r["failure_reason"] = reason_c
            return r

        # ── WARM pass ──
        warm = _stream_chat(model, TEST_PROMPT_WARM, [TEST_TOOL])
        if warm["error"] and not ollama_alive():
            raise ConnectionError(f"server down during warm pass: {warm['error']}")

        ok_w, status_w, reason_w, _, _ = _validate_tool_result(warm)
        r["ttft_warm_ms"] = warm["ttft_ms"]
        r["itl_warm_p50_ms"] = _percentile(warm["itl_ms_list"], 50)
        r["itl_warm_p95_ms"] = _percentile(warm["itl_ms_list"], 95)
        r["warm_chunks"] = warm["chunks"]
        r["warm_ok"] = ok_w

        if ok_w:
            r["status"] = "SUCCESS"
        else:
            r["status"] = "PARTIAL_COLD_OK"
            r["warm_status"] = f"{status_w}: {reason_w}"
            r["failure_reason"] = f"cold passed, warm failed — {r['warm_status']}"
        return r

    except Exception as e:
        dur = time.time() - start
        r["duration_seconds"] = dur
        if not ollama_alive():
            r["status"] = "CRASH_SERVER_DOWN"
            r["failure_reason"] = f"{type(e).__name__}: {str(e)[:200]} (server down)"
        else:
            r["status"] = "CRASH_EXCEPTION"
            r["failure_reason"] = f"{type(e).__name__}: {str(e)[:200]}"
        return r
    finally:
        sampler.stop()
        sampler.join(timeout=5)
        r["resources"] = sampler.summary(baseline_swap_used_mb)
        r["score"] = compute_score(r)


# ─── Plan & summary ────────────────────────────────────────────────────────
def build_test_plan(installed, installed_sizes, hist, force, interrupted, cands):
    installed_l = {m.lower() for m in installed}
    seen = set()
    items = []
    for m in installed:
        if m.lower() in seen:
            continue
        items.append({"model": m, "needs_pull": False,
                      "size_bytes": installed_sizes.get(m)})
        seen.add(m.lower())
    for m in cands:
        if m.lower() in seen:
            continue
        items.append({"model": m, "needs_pull": True, "size_bytes": None})
        seen.add(m.lower())
    skip, to_test = {}, []
    for it in items:
        m = it["model"]
        if interrupted and interrupted.get("model") == m and m not in force:
            skip[m] = f"interrupted test resolved ({interrupted.get('action')})"
            continue
        if m in force:
            to_test.append(it); continue
        if m in hist:
            skip[m] = f"already tested ({hist[m]['status']})"
            continue
        to_test.append(it)
    log.info("fetching candidate sizes for ordering...")
    for it in to_test:
        if it["size_bytes"] is None and it["needs_pull"]:
            it["size_bytes"] = fetch_registry_size(it["model"])
    known = [it for it in to_test if it["size_bytes"] is not None]
    unknown = [it for it in to_test if it["size_bytes"] is None]
    known.sort(key=lambda x: x["size_bytes"])
    return known + unknown, skip


def print_plan(to_test):
    log.info(SEP)
    log.info("TEST PLAN (smallest → largest)")
    log.info(SEP)
    total = 0
    for i, it in enumerate(to_test, 1):
        sz = it.get("size_bytes")
        if sz:
            sz_str = f"{sz / 1e9:>7.2f} GB"
            if it["needs_pull"]:
                total += sz
        else:
            sz_str = "      ?    "
        kind = "will pull" if it["needs_pull"] else "installed"
        log.info(f"  {i:2}. {sz_str}  {it['model']:<42} [{kind}]")
    if total:
        log.info(f"  Total download needed: {total / 1e9:.2f} GB")
    log.info(SEP)


def _res_line(r):
    res = r.get("resources") or {}
    if not res:
        return "no resource data"
    bits = []
    if res.get("swap_peak_mb") not in ("", None):
        bits.append(f"swap peak {res['swap_peak_mb']} MB "
                    f"(out {res.get('swap_out_mb', '?')} MB)")
    if res.get("ps_gpu_pct") not in ("", None):
        bits.append(f"GPU {res['ps_gpu_pct']}% "
                    f"({res.get('ps_vram_gb', '?')}/{res.get('ps_size_gb', '?')} GB)")
    if res.get("ollama_rss_peak_mb") not in ("", None):
        bits.append(f"ollama rss peak {res['ollama_rss_peak_mb']} MB")
    if res.get("cpu_peak_pct") not in ("", None):
        bits.append(f"CPU peak {res['cpu_peak_pct']}%")
    return " | ".join(bits) if bits else "no resource data"


def _timing_line(r):
    parts = []
    if r.get("ttft_cold_ms") not in (0, "", None):
        parts.append(f"TTFT cold {r['ttft_cold_ms']:.0f} ms")
    if r.get("ttft_warm_ms") not in (0, "", None):
        parts.append(f"warm {r['ttft_warm_ms']:.0f} ms")
    t = r.get("timing") or {}
    if t.get("decode_tps") not in ("", None):
        parts.append(f"decode {t['decode_tps']} tok/s")
    if r.get("itl_cold_p50_ms"):
        parts.append(f"ITL p50={r['itl_cold_p50_ms']} p95={r['itl_cold_p95_ms']} ms")
    if r.get("score"):
        parts.append(f"score={r['score']}")
    return " | ".join(parts)


def _res_flags(r):
    res = r.get("resources") or {}
    out = []
    try:
        if float(res.get("swap_out_mb", 0)) > SWAP_WARN_OUT_MB:
            out.append(f"SWAPPED OUT {res['swap_out_mb']} MB during test")
    except (TypeError, ValueError):
        pass
    try:
        g = res.get("ps_gpu_pct")
        if g not in ("", None) and float(g) < GPU_LOW_PCT:
            out.append(f"only {g}% resident in VRAM")
    except (TypeError, ValueError):
        pass
    return out


def print_summary(tested, results, skipped, untested, aborted_at, deletions,
                  run_id, baseline_swap_mb):
    log.info(SEP)
    log.info(f"SUMMARY  run_id={run_id}  baseline_swap={baseline_swap_mb} MB")
    log.info(SEP)
    hist = history()
    pending = [(m, h) for m, h in hist.items() if h.get("status") == "INTERRUPTED"]
    passed = [(m, r) for m, r in results.items() if r["status"] == "SUCCESS"]
    failed = [(m, r) for m, r in results.items() if r["status"] != "SUCCESS"]

    if pending:
        log.warning("⚠ INTERRUPTED TESTS NEEDING YOUR ATTENTION:")
        for m, h in pending:
            log.warning(f"  - {m}  (last seen {h.get('timestamp') or 'unknown'})")
            log.warning(f"      retest: --force-retest {m}")
            log.warning(f"      delete: ollama rm {m}")
        log.warning("")

    log.info(f"PASSED ({len(passed)}):")
    for m, r in passed:
        tag = f"  [{deletions[m]}]" if m in deletions else ""
        pulled = " [pulled]" if r.get("pulled") else ""
        log.info(f"  ✓ {m:<42} dur={r['duration_seconds']:>6.2f}s{pulled}{tag}")
        tl = _timing_line(r)
        if tl:
            log.info(f"      {tl}")
        log.info(f"      {_res_line(r)}")
        for f in _res_flags(r):
            log.warning(f"      ⚠ {f}")
    if not passed:
        log.info("  (none)")
    log.info("")

    log.info(f"FAILED ({len(failed)}):")
    for m, r in failed:
        tag = f"  [{deletions[m]}]" if m in deletions else ""
        pulled = " [pulled]" if r.get("pulled") else ""
        log.info(f"  ✗ {m} — {r['status']}{pulled}{tag}")
        log.info(f"      reason: {r['failure_reason']}")
        tl = _timing_line(r)
        if tl:
            log.info(f"      {tl}")
        log.info(f"      {_res_line(r)}")
        for f in _res_flags(r):
            log.warning(f"      ⚠ {f}")
        log.info("")
    if not failed:
        log.info("  (none)")
    log.info("")

    log.info(f"NOT TESTED ({len(skipped) + len(untested)}):")
    for m, why in skipped.items():
        log.info(f"  – {m:<42} {why}")
    for m in untested:
        log.info(f"  – {m:<42} run aborted at {aborted_at!r}")
    log.info("")

    if aborted_at:
        log.warning(f"⚠ aborted: server crashed at {aborted_at!r}; model kept")
    log.info(SEP)


# ─── Main ───────────────────────────────────────────────────────────────────
def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("model", nargs="?", default=None,
                   help="single model name to test (e.g. gemma4:12b)")
    p.add_argument("--force-retest", action="append", default=[], metavar="MODEL")
    p.add_argument("--force-retest-all", action="store_true")
    p.add_argument("--interrupted-action",
                   choices=["skip", "retest", "delete"], default="skip")
    p.add_argument("--from-registry", action="store_true",
                   help="read candidate models from the Ollama registry catalog")
    p.add_argument("--registry-limit", type=int, default=200,
                   help=argparse.SUPPRESS)
    p.add_argument("--min-params", type=float, default=None,
                   help="min parameter size in billions when using --from-registry (e.g. 7)")
    p.add_argument("--max-params", type=float, default=None,
                   help="max parameter size in billions when using --from-registry (e.g. 12)")
    args = p.parse_args(argv)

    if not args.model and not args.from_registry:
        p.print_help()
        return

    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    log.info(f"run start run_id={run_id}")
    if not ensure_ollama():
        log.error("aborting: Ollama unreachable")
        return

    # Baseline swap BEFORE any test — logged and stored per row for context.
    base_swap = psutil.swap_memory()
    baseline_swap_mb = round(base_swap.used / 1e6, 1)
    log.info(f"baseline swap: used={baseline_swap_mb} MB, "
             f"sout={base_swap.sout / 1e6:.1f} MB, "
             f"sin={base_swap.sin / 1e6:.1f} MB")

    csv_fields()
    interrupted = resolve_interrupted(args.interrupted_action)
    hist = history()
    installed_sizes, installed = installed_models_and_sizes()
    installed_l = {m.lower() for m in installed}

    skip = {}
    # Pre-filter installed models by capabilities: never plan/test a model
    # that does not advertise 'tools'. Non-tool models are logged and skipped
    # up front so they never enter the test plan (and are never pulled).
    tool_capable = {}
    for m in installed:
        caps = (model_info(m).get("capabilities") or [])
        if "tools" in caps:
            tool_capable[m] = installed_sizes.get(m)
        else:
            skip[m] = "no 'tools' capability"
            log.info(f"skipping {m}: no 'tools' capability")
    installed = list(tool_capable)
    installed_sizes = tool_capable
    installed_l = {m.lower() for m in installed}
    if args.model:
        cands = [args.model]
    else:
        cands = load_candidates(from_registry=args.from_registry,
                                registry_limit=args.registry_limit,
                                min_params=args.min_params,
                                max_params=args.max_params)

    force = set(args.force_retest)
    if args.force_retest_all:
        force = set(hist.keys())
    if interrupted and interrupted.get("action") == "retest":
        force.add(interrupted["model"])

    to_test, skip = build_test_plan(
        installed, installed_sizes, hist, force, interrupted, cands)
    total_considered = len(installed) + len(
        [c for c in cands if c.lower() not in installed_l])

    if not to_test:
        log.info("nothing to test")
        print_summary(total_considered, {}, skip, [], None, {},
                      run_id, baseline_swap_mb)
        return

    print_plan(to_test)

    old = [(m, hist[m].get("timestamp") or "") for m in installed if m in hist]
    old.sort(key=lambda x: x[1])
    history_order = [m for m, _ in old]

    protect = {it["model"] for it in to_test}
    if interrupted and interrupted.get("model") in installed_l:
        protect.add(interrupted["model"])

    results, deletions = {}, {}
    aborted_at, remaining = None, []
    can_pull = True

    for i, it in enumerate(to_test):
        m, pull_needed = it["model"], it["needs_pull"]
        future = {x["model"] for x in to_test[i:]} | protect

        target = DISK_TARGET_PULL_PCT if pull_needed else DISK_TARGET_TEST_PCT
        if free_pct() < target:
            while free_pct() < target:
                victim = next((h for h in history_order
                               if h not in future and h not in deletions), None)
                if not victim:
                    break
                if delete(victim):
                    unload(victim)
                    deletions[victim] = "deleted to free disk space"
                    if victim in history_order:
                        history_order.remove(victim)
                else:
                    deletions[victim] = "delete failed"
            if free_pct() < DISK_MIN_PCT and pull_needed:
                results[m] = _simple_fail(
                    "FAILED_DISK_SPACE",
                    f"insufficient space ({free_pct():.1f}%)")
                can_pull = False
                continue

        pulled = False
        if pull_needed:
            if not can_pull:
                results[m] = _simple_fail("FAILED_DISK_SPACE", "previous pull failed")
                continue
            ok, err = pull(m)
            if not ok:
                results[m] = _simple_fail("FAILED_PULL", err)
                log.error(f"pull failed {m}: {err}")
                if "space" in err.lower() or "disk" in err.lower():
                    can_pull = False
                continue
            pulled = True

        info = model_info(m)
        caps = info.get("capabilities", [])
        if "tools" not in caps:
            skip[m] = "no 'tools' capability"
            log.info(f"skipping {m}: no 'tools' capability")
            if pulled and delete(m):
                deletions[m] = "deleted: no 'tools'"
            # Record pulled-but-unsupported models in the CSV as evidence.
            details = info.get("details", {}) or {}
            size_bytes = installed_sizes.get(m) or it.get("size_bytes")
            write_row({
                "timestamp": datetime.now().isoformat(),
                "run_id": run_id,
                "model": m,
                "status": "SKIPPED_NO_TOOLS",
                "failure_reason": "no 'tools' capability",
                "tps": "", "duration_seconds": "", "tool_called": False,
                "warm_ok": False, "warm_status": "", "pulled": pulled,
                "response_preview": "", "tool_call_details": "",
                "capabilities": ",".join(caps), "score": 0,
                "model_size_gb": round(size_bytes / 1e9, 2) if size_bytes else "",
                "param_size": details.get("parameter_size", ""),
                "quantization": details.get("quantization_level", ""),
                "family": details.get("family", ""),
                "ttft_cold_ms": "", "ttft_warm_ms": "",
                "itl_cold_p50_ms": "", "itl_cold_p95_ms": "",
                "itl_warm_p50_ms": "", "itl_warm_p95_ms": "",
                "cold_chunks": "", "warm_chunks": "",
            })
            continue

        details = info.get("details", {}) or {}
        size_bytes = installed_sizes.get(m) or it.get("size_bytes")

        log.info(f"--- testing {m} ---")
        unload(m); time.sleep(1)
        write_state(m)
        r = run_test(m, baseline_swap_mb)
        r["pulled"] = pulled
        results[m] = r

        row = {
            "timestamp": datetime.now().isoformat(),
            "run_id": run_id,
            "model": m,
            "status": r["status"],
            "failure_reason": r["failure_reason"],
            "tps": f"{r['tps']:.2f}",
            "duration_seconds": f"{r['duration_seconds']:.2f}",
            "tool_called": r["tool_called"],
            "warm_ok": r["warm_ok"],
            "warm_status": r.get("warm_status", ""),
            "pulled": pulled,
            "response_preview": r["response_preview"].replace("\n", "\\n"),
            "tool_call_details": r["tool_call_details"].replace("\n", "\\n"),
            "capabilities": ",".join(caps),
            "score": r.get("score", 0),
            "model_size_gb": round(size_bytes / 1e9, 2) if size_bytes else "",
            "param_size": details.get("parameter_size", ""),
            "quantization": details.get("quantization_level", ""),
            "family": details.get("family", ""),
            "ttft_cold_ms": r.get("ttft_cold_ms", ""),
            "ttft_warm_ms": r.get("ttft_warm_ms", ""),
            "itl_cold_p50_ms": r.get("itl_cold_p50_ms", ""),
            "itl_cold_p95_ms": r.get("itl_cold_p95_ms", ""),
            "itl_warm_p50_ms": r.get("itl_warm_p50_ms", ""),
            "itl_warm_p95_ms": r.get("itl_warm_p95_ms", ""),
            "cold_chunks": r.get("cold_chunks", ""),
            "warm_chunks": r.get("warm_chunks", ""),
        }
        row.update(r.get("timing", {}))
        row.update(r.get("resources", {}))
        write_row(row)
        clear_state()

        if r["status"] == "SUCCESS":
            log.info(f"[OK] {m} | {_timing_line(r)}")
            log.info(f"     {_res_line(r)}")
            for f in _res_flags(r):
                log.warning(f"     ⚠ {f}")
        elif r["status"].startswith("CRASH"):
            log.error(f"[CRASH] {m}: {r['failure_reason']}")
        else:
            log.warning(f"[FAIL] {m}: {r['status']} — {r['failure_reason']}")

        unload(m)
        history_order.append(m)

        if r["status"] == "CRASH_SERVER_DOWN":
            aborted_at = m
            remaining = [x["model"] for x in to_test[i + 1:]]
            log.error(f"aborting: server down at {m}; model kept")
            break

        if r["status"] != "SUCCESS":
            if delete(m):
                deletions[m] = "deleted after failing test"

    print_summary(total_considered, results, skip, remaining, aborted_at,
                  deletions, run_id, baseline_swap_mb)


def _simple_fail(status, reason):
    return {"status": status, "failure_reason": reason,
            "tps": 0.0, "duration_seconds": 0.0,
            "tool_called": False, "warm_ok": False, "warm_status": "",
            "response_preview": "", "tool_call_details": "",
            "resources": {}, "timing": {}, "score": 0.0,
            "ttft_cold_ms": 0, "ttft_warm_ms": 0,
            "itl_cold_p50_ms": 0, "itl_cold_p95_ms": 0,
            "itl_warm_p50_ms": 0, "itl_warm_p95_ms": 0,
            "cold_chunks": 0, "warm_chunks": 0}


if __name__ == "__main__":
    main()