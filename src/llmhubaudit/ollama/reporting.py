"""Reporting for the Ollama audit module.

Handles printing the test plan and final summary, separating passed,
failed, and not-tested models.
"""

from __future__ import annotations

from typing import Any

from .config import GPU_LOW_PCT, SWAP_WARN_OUT_MB


def _res_line(r: dict[str, Any]) -> str:
    res = r.get("resources") or {}
    if not res:
        return "no resource data"
    bits: list[str] = []
    if res.get("swap_peak_mb") not in ("", None):
        bits.append(f"swap peak {res['swap_peak_mb']} MB (out {res.get('swap_out_mb', '?')} MB)")
    if res.get("ps_gpu_pct") not in ("", None):
        bits.append(
            f"GPU {res['ps_gpu_pct']}% "
            f"({res.get('ps_vram_gb', '?')}/{res.get('ps_size_gb', '?')} GB)"
        )
    if res.get("ollama_rss_peak_mb") not in ("", None):
        bits.append(f"ollama rss peak {res['ollama_rss_peak_mb']} MB")
    if res.get("cpu_peak_pct") not in ("", None):
        bits.append(f"CPU peak {res['cpu_peak_pct']}%")
    return " | ".join(bits) if bits else "no resource data"


def _timing_line(r: dict[str, Any]) -> str:
    parts: list[str] = []
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


def _res_flags(r: dict[str, Any]) -> list[str]:
    res = r.get("resources") or {}
    out: list[str] = []
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


def print_plan(to_test: list[dict[str, Any]]) -> None:
    """Print the test plan (smallest → largest)."""
    SEP = "=" * 78
    import logging

    log = logging.getLogger(__name__)
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


def print_summary(
    tested: int,
    results: dict[str, dict[str, Any]],
    skipped: dict[str, str],
    untested: list[str],
    aborted_at: str | None,
    deletions: dict[str, str],
    run_id: str,
    baseline_swap_mb: str | float,
) -> None:
    """Print the final summary separating passed, failed, and not-tested."""
    from .state import history

    SEP = "=" * 78
    import logging

    log = logging.getLogger(__name__)
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
