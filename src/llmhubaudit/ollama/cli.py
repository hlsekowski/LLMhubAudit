"""CLI entry point for the Ollama audit module.

Mirrors the arguments and behaviour of the standalone
``scripts/ollama/test_ollama_models.py`` script, but uses the modular
components in this package so the logic is importable and testable.
"""

from __future__ import annotations

import argparse
import logging
import time
from datetime import datetime
from typing import Any, Mapping

import psutil  # type: ignore[import-untyped]

from .config import (
    DISK_MIN_PCT,
    DISK_TARGET_PULL_PCT,
    DISK_TARGET_TEST_PCT,
)
from .discovery import load_candidates
from .reporting import (
    _res_flags,
    _res_line,
    _timing_line,
    print_plan,
    print_summary,
)
from .runner import run_test
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
    pull,
    resolve_interrupted,
    unload,
    write_row,
    write_state,
)

log = logging.getLogger(__name__)
SEP = "=" * 78


def _simple_fail(status: str, reason: str) -> dict[str, Any]:
    return {
        "status": status,
        "failure_reason": reason,
        "tps": 0.0,
        "duration_seconds": 0.0,
        "tool_called": False,
        "warm_ok": False,
        "warm_status": "",
        "response_preview": "",
        "tool_call_details": "",
        "resources": {},
        "timing": {},
        "score": 0.0,
        "ttft_cold_ms": 0,
        "ttft_warm_ms": 0,
        "itl_cold_p50_ms": 0,
        "itl_cold_p95_ms": 0,
        "itl_warm_p50_ms": 0,
        "itl_warm_p95_ms": 0,
        "cold_chunks": 0,
        "warm_chunks": 0,
    }


def build_test_plan(
    installed: list[str],
    installed_sizes: Mapping[str, int | None],
    hist: dict[str, dict[str, str]],
    force: set[str],
    interrupted: dict[str, Any] | None,
    cands: list[str],
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Build the ordered test plan from installed models and candidates."""
    seen: set[str] = set()
    items: list[dict[str, Any]] = []
    for m in installed:
        if m.lower() in seen:
            continue
        items.append({"model": m, "needs_pull": False, "size_bytes": installed_sizes.get(m)})
        seen.add(m.lower())
    for m in cands:
        if m.lower() in seen:
            continue
        items.append({"model": m, "needs_pull": True, "size_bytes": None})
        seen.add(m.lower())
    skip: dict[str, str] = {}
    to_test: list[dict[str, Any]] = []
    for it in items:
        m = it["model"]
        if interrupted and interrupted.get("model") == m and m not in force:
            skip[m] = f"interrupted test resolved ({interrupted.get('action')})"
            continue
        if m in force:
            to_test.append(it)
            continue
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
    known.sort(key=lambda x: x["size_bytes"] or 0)
    return known + unknown, skip


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "model", nargs="?", default=None, help="single model name to test (e.g. gemma4:12b)"
    )
    p.add_argument("--force-retest", action="append", default=[], metavar="MODEL")
    p.add_argument("--force-retest-all", action="store_true")
    p.add_argument("--interrupted-action", choices=["skip", "retest", "delete"], default="skip")
    p.add_argument(
        "--from-registry",
        action="store_true",
        help="read candidate models from the Ollama registry catalog",
    )
    p.add_argument("--registry-limit", type=int, default=200, help=argparse.SUPPRESS)
    p.add_argument(
        "--min-params",
        type=float,
        default=None,
        help="min parameter size in billions when using --from-registry (e.g. 7)",
    )
    p.add_argument(
        "--max-params",
        type=float,
        default=None,
        help="max parameter size in billions when using --from-registry (e.g. 12)",
    )
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
    log.info(
        f"baseline swap: used={baseline_swap_mb} MB, "
        f"sout={base_swap.sout / 1e6:.1f} MB, "
        f"sin={base_swap.sin / 1e6:.1f} MB"
    )

    csv_fields()
    interrupted = resolve_interrupted(args.interrupted_action)
    hist = history()
    raw_sizes, installed = installed_models_and_sizes()
    installed_l = {m.lower() for m in installed}

    skip: dict[str, str] = {}
    # Pre-filter installed models by capabilities: never plan/test a model
    # that does not advertise 'tools'. Non-tool models are logged and skipped
    # up front so they never enter the test plan (and are never pulled).
    tool_capable: dict[str, int | None] = {}
    for m in installed:
        caps = model_info(m).get("capabilities") or []
        if "tools" in caps:
            tool_capable[m] = raw_sizes.get(m)
        else:
            skip[m] = "no 'tools' capability"
            log.info(f"skipping {m}: no 'tools' capability")
    installed = list(tool_capable)
    installed_sizes: dict[str, int | None] = {k: v for k, v in tool_capable.items()}
    installed_l = {m.lower() for m in installed}
    if args.model:
        cands: list[str] = [args.model]
    else:
        cands = load_candidates(
            from_registry=args.from_registry,
            registry_limit=args.registry_limit,
            min_params=args.min_params,
            max_params=args.max_params,
        )

    force: set[str] = set(args.force_retest)
    if args.force_retest_all:
        force = set(hist.keys())
    if interrupted and interrupted.get("action") == "retest":
        force.add(interrupted["model"])

    to_test, skip = build_test_plan(installed, installed_sizes, hist, force, interrupted, cands)
    total_considered = len(installed) + len([c for c in cands if c.lower() not in installed_l])

    if not to_test:
        log.info("nothing to test")
        print_summary(total_considered, {}, skip, [], None, {}, run_id, baseline_swap_mb)
        return

    print_plan(to_test)

    old = [(m, hist[m].get("timestamp") or "") for m in installed if m in hist]
    old.sort(key=lambda x: x[1])
    history_order = [m for m, _ in old]

    protect = {it["model"] for it in to_test}
    if interrupted and interrupted.get("model") in installed_l:
        protect.add(interrupted["model"])

    results: dict[str, dict[str, Any]] = {}
    deletions: dict[str, str] = {}
    aborted_at: str | None = None
    remaining: list[str] = []
    can_pull = True

    for i, it in enumerate(to_test):
        m, pull_needed = it["model"], it["needs_pull"]
        future = {x["model"] for x in to_test[i:]} | protect

        target = DISK_TARGET_PULL_PCT if pull_needed else DISK_TARGET_TEST_PCT
        if free_pct() < target:
            while free_pct() < target:
                victim = next(
                    (h for h in history_order if h not in future and h not in deletions), None
                )
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
                    "FAILED_DISK_SPACE", f"insufficient space ({free_pct():.1f}%)"
                )
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
            skip_row = {
                "timestamp": datetime.now().isoformat(),
                "run_id": run_id,
                "model": m,
                "status": "SKIPPED_NO_TOOLS",
                "failure_reason": "no 'tools' capability",
                "tps": "",
                "duration_seconds": "",
                "tool_called": False,
                "warm_ok": False,
                "warm_status": "",
                "pulled": pulled,
                "response_preview": "",
                "tool_call_details": "",
                "capabilities": ",".join(caps),
                "score": 0,
                "model_size_gb": round(size_bytes / 1e9, 2) if size_bytes else "",
                "param_size": details.get("parameter_size", ""),
                "quantization": details.get("quantization_level", ""),
                "family": details.get("family", ""),
                "ttft_cold_ms": "",
                "ttft_warm_ms": "",
                "itl_cold_p50_ms": "",
                "itl_cold_p95_ms": "",
                "itl_warm_p50_ms": "",
                "itl_warm_p95_ms": "",
                "cold_chunks": "",
                "warm_chunks": "",
            }
            write_row(skip_row)
            continue

        details = info.get("details", {}) or {}
        size_bytes = installed_sizes.get(m) or it.get("size_bytes")

        log.info(f"--- testing {m} ---")
        unload(m)
        time.sleep(1)
        write_state(m)
        r = run_test(m, baseline_swap_mb)
        r["pulled"] = pulled
        results[m] = r

        row: dict[str, Any] = {
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
            remaining = [x["model"] for x in to_test[i + 1 :]]
            log.error(f"aborting: server down at {m}; model kept")
            break

        if r["status"] != "SUCCESS":
            if delete(m):
                deletions[m] = "deleted after failing test"

    print_summary(
        total_considered, results, skip, remaining, aborted_at, deletions, run_id, baseline_swap_mb
    )


if __name__ == "__main__":
    main()
