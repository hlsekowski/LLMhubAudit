"""State management for the Ollama audit module.

Handles CSV results (schema migration, history, row writing), JSON
in-progress state files, disk-space checks, and Ollama process
management (alive check, pull, delete, unload, model info).
"""

from __future__ import annotations

import csv
import json
import logging
import os
import shutil
import subprocess
import time
from datetime import datetime
from typing import Any, cast

import psutil  # type: ignore[import-untyped]
import requests  # type: ignore[import-untyped]

from .config import (
    ALL_REQUIRED_COLS,
    BASE_URL,
    CSV,
    LOG,
    OLLAMA_START_TIMEOUT_S,
    REGISTRY,
    STATE,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(LOG), logging.StreamHandler(__import__("sys").stdout)],
)
log = logging.getLogger(__name__)

_FIELDS: list[str] | None = None


def free_pct(path: str = "/") -> float:
    """Return the percentage of free disk space for ``path``."""
    u = psutil.disk_usage(path)
    return 100.0 * float(u.free) / float(u.total)


def ollama_alive(timeout: float = 5.0) -> bool:
    """Return True if the Ollama server is responding."""
    try:
        return bool(requests.get(f"{BASE_URL}/api/tags", timeout=timeout).status_code == 200)
    except requests.RequestException:
        return False


def ensure_ollama() -> bool:
    """Ensure Ollama is running, starting ``ollama serve`` if needed."""
    if ollama_alive(2.0):
        return True
    log.warning("Ollama not responding; starting `ollama serve`...")
    if shutil.which("ollama") is None:
        log.error("`ollama` not on PATH.")
        return False
    subprocess.Popen(
        ["ollama", "serve"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    for _ in range(OLLAMA_START_TIMEOUT_S):
        time.sleep(1)
        if ollama_alive(1.0):
            log.info("Ollama is up.")
            return True
    log.error(f"Ollama did not come up within {OLLAMA_START_TIMEOUT_S}s.")
    return False


def installed_models_and_sizes() -> tuple[dict[str, int], list[str]]:
    """Return (name -> size_bytes, [names]) of installed Ollama models."""
    try:
        r = requests.get(f"{BASE_URL}/api/tags", timeout=10)
        r.raise_for_status()
        data = r.json().get("models", [])
        return {m["name"]: m.get("size", 0) for m in data}, [m["name"] for m in data]
    except requests.RequestException as e:
        log.error(f"list models: {e}")
        return {}, []


def model_info(model: str) -> dict[str, Any]:
    """Return the ``/api/show`` JSON for a model, or {} on error."""
    try:
        r = requests.post(f"{BASE_URL}/api/show", json={"name": model}, timeout=10)
        r.raise_for_status()
        return cast(dict[str, Any], r.json())
    except requests.RequestException as e:
        log.error(f"show {model}: {e}")
        return {}


def fetch_registry_size(model_name: str) -> int | None:
    """Return the total manifest size in bytes for a registry model, or None."""
    try:
        base, tag = model_name.rsplit(":", 1) if ":" in model_name else (model_name, "latest")
        path = base if "/" in base else f"library/{base}"
        url = f"{REGISTRY}/v2/{path}/manifests/{tag}"
        r = requests.get(
            url,
            timeout=8,
            headers={"Accept": "application/vnd.docker.distribution.manifest.v2+json"},
        )
        r.raise_for_status()
        data = r.json()
        total = data.get("config", {}).get("size", 0)
        for layer in data.get("layers", []):
            total += layer.get("size", 0)
        return total if total > 0 else None
    except Exception:
        return None


def unload(model: str) -> None:
    """Unload a model from Ollama (keep_alive=0)."""
    try:
        requests.post(
            f"{BASE_URL}/api/generate",
            json={"model": model, "keep_alive": 0},
            timeout=10,
        )
    except requests.RequestException:
        pass


def delete(model: str) -> bool:
    """Delete a model from Ollama. Returns True on success."""
    try:
        r = requests.delete(f"{BASE_URL}/api/delete", json={"model": model}, timeout=30)
        r.raise_for_status()
        log.info(f"deleted {model}")
        return True
    except requests.RequestException as e:
        log.error(f"delete {model}: {e}")
        return False


def pull(model: str) -> tuple[bool, str]:
    """Pull a model from the Ollama registry.

    Returns ``(ok, error_message)``. On a connection error, attempts to
    start Ollama and retries once.
    """

    def _once() -> tuple[bool, str]:
        last = ""
        try:
            with requests.post(
                f"{BASE_URL}/api/pull",
                json={"name": model, "stream": True},
                stream=True,
                timeout=(15, 600),
            ) as r:
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


# ─── CSV results ──────────────────────────────────────────────────────────


def csv_fields() -> list[str]:
    """Return the CSV field list, migrating older schemas as needed."""
    global _FIELDS
    if _FIELDS is not None:
        return _FIELDS
    if not os.path.isfile(CSV) or os.path.getsize(CSV) == 0:
        _FIELDS = list(ALL_REQUIRED_COLS)
        return _FIELDS
    with open(CSV, newline="") as f:
        reader = csv.DictReader(f)
        old = reader.fieldnames or []
        rows = list(reader)
    if all(c in old for c in ALL_REQUIRED_COLS):
        _FIELDS = list(old)
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
    return new


def history() -> dict[str, dict[str, str]]:
    """Return {model: {status, timestamp}} from the results CSV."""
    if not os.path.isfile(CSV) or os.path.getsize(CSV) == 0:
        return {}
    out: dict[str, dict[str, str]] = {}
    with open(CSV, newline="") as f:
        for row in csv.DictReader(f):
            m = (row.get("model") or "").strip()
            if m:
                out[m] = {
                    "status": row.get("status", ""),
                    "timestamp": row.get("timestamp", ""),
                }
    return out


def write_row(row: dict[str, Any]) -> None:
    """Append a row to the results CSV, writing the header if new."""
    fields = csv_fields()
    new_file = not os.path.isfile(CSV) or os.path.getsize(CSV) == 0
    with open(CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if new_file:
            w.writeheader()
        w.writerow(row)


# ─── Interrupted-test state ───────────────────────────────────────────────


def read_state() -> dict[str, Any] | None:
    """Read the in-progress state JSON file, or None if absent/malformed."""
    if not os.path.isfile(STATE):
        return None
    try:
        with open(STATE) as f:
            return cast(dict[str, Any], json.load(f))
    except Exception:
        return None


def write_state(model: str) -> None:
    """Write the in-progress state JSON file for ``model``."""
    with open(STATE, "w") as f:
        json.dump(
            {
                "model": model,
                "pid": os.getpid(),
                "started_at": datetime.now().isoformat(),
            },
            f,
        )


def clear_state() -> None:
    """Remove the in-progress state file if it exists."""
    if os.path.isfile(STATE):
        os.remove(STATE)


def resolve_interrupted(action: str) -> dict[str, Any] | None:
    """Resolve an interrupted test from a previous run.

    Writes an ``INTERRUPTED`` row to the CSV, optionally deletes the model,
    and clears the state file. Returns the resolved state dict or None.
    """
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
    SEP = "=" * 78
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
            log.debug(f"pid check failed for {pid}")
    log.warning(SEP)
    row = {
        "timestamp": datetime.now().isoformat(),
        "model": model,
        "status": "INTERRUPTED",
        "failure_reason": (
            f"previous test interrupted (started {started}, "
            f"pid {pid}); resolved with --interrupted-action={action}"
        ),
    }
    write_row(row)
    if action == "delete":
        delete(model)
    elif action == "retest":
        log.info(f"will retest {model}")
    else:
        log.info(f"skipping {model}")
    clear_state()
    return {**st, "model": model, "action": action}
