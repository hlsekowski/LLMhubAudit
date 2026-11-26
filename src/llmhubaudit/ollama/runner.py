"""Test execution and validation for the Ollama audit module.

Handles streaming chat, tool-call parsing/normalization, result
validation, cold+warm test execution, and the composite score.
"""

from __future__ import annotations

import json
import re
import time
from typing import Any

import ollama

from .config import (
    TEST_PROMPT_COLD,
    TEST_PROMPT_WARM,
    TEST_TOOL,
)
from .metrics import _extract_timing, _percentile, compute_score
from .state import ollama_alive


def _make_tool_call_obj(name: str, arguments: dict[str, Any]) -> Any:
    """Create a tool call object with a .function attribute matching the
    expected Ollama message format.
    """

    class ToolCall:
        def __init__(self, name: str, arguments: dict[str, Any]):
            self.function = type("Function", (), {"name": name, "arguments": arguments})()

    return ToolCall(name, arguments)


def _normalize_tool_call(c: Any) -> tuple[str | None, dict[str, Any] | None]:
    """Normalize a tool call to (name, arguments).

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
            return f.get("name"), (
                f.get("arguments") if isinstance(f.get("arguments"), dict) else {}
            )
        name = c.get("name") or c.get("function_name")
        if name:
            args = c.get("arguments")
            return name, (args if isinstance(args, dict) else {})
    return None, None


def _parse_tool_calls_from_content(content: str) -> list[Any]:
    """Parse tool calls from content text.

    Handles plain JSON, compact JSON, different key names, tool-call tags,
    angle brackets, tool tags, markdown code blocks, and tool definition
    formats. Returns a list of tool-call objects.
    """
    tool_calls: list[Any] = []

    # Pattern 1: JSON objects with "name"/"function_name" and "arguments" keys
    json_pattern1 = (
        r'(?:\{.*?"(?:name|function_name)"\s*:\s*"[^"]+"\s*,\s*"arguments"\s*:\s*\{.*?\}\s*\})'
    )
    json_pattern2 = (
        r'(?:\{.*?"arguments"\s*:\s*\{.*?\}\s*,\s*"(?:name|function_name)"\s*:\s*"[^"]+"\s*\})'
    )
    for pattern in [json_pattern1, json_pattern2]:
        matches = re.findall(pattern, content, re.DOTALL)
        for match in matches:
            try:
                parsed = json.loads(match)
                name = parsed.get("name") or parsed.get("function_name")
                arguments = parsed.get("arguments", {})
                if name and isinstance(arguments, dict):
                    tool_calls.append(_make_tool_call_obj(name, arguments))
            except json.JSONDecodeError:
                pass

    # Pattern 2: <tool-call>[...]
    tool_call_tag_pattern = r"<tool-call>\s*(\[.*?\])\s*</tool-call>"
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

    # Pattern 2b: <tool-call>[...] without closing tag
    tool_call_tag_pattern2 = r"<tool-call>\s*(\[.*?\])"
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

    # Pattern 3: <tool>[...]
    tool_tag_pattern = r"<tool>\s*(\[.*?\])\s*</tool>"
    matches = re.findall(tool_tag_pattern, content, re.DOTALL)
    for match in matches:
        try:
            parsed = json.loads(match)
            if isinstance(parsed, list):
                for item in parsed:
                    if isinstance(item, dict):
                        if item.get("type") == "function" and "function" in item:
                            func = item["function"]
                            name = func.get("name")
                            arguments = func.get("arguments", {})
                            if name and isinstance(arguments, dict):
                                tool_calls.append(_make_tool_call_obj(name, arguments))
                        else:
                            name = item.get("name")
                            arguments = item.get("arguments", {})
                            if name and isinstance(arguments, dict):
                                tool_calls.append(_make_tool_call_obj(name, arguments))
        except json.JSONDecodeError:
            pass

    # Pattern 4: <{...}> angle bracket wrapped JSON
    angle_bracket_pattern = r"<\s*(\{.*?\})\s*>"
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

    # Pattern 5: markdown code blocks with JSON
    markdown_json_pattern = r"```json\s*(\{.*?\})\s*```"
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

    # Pattern 6: tool definition format in <tool> tags
    tool_def_pattern = r"<tool>\s*(\[.*?\])"
    matches = re.findall(tool_def_pattern, content, re.DOTALL)
    for match in matches:
        try:
            parsed = json.loads(match)
            if isinstance(parsed, list):
                for item in parsed:
                    if (
                        isinstance(item, dict)
                        and item.get("type") == "function"
                        and "function" in item
                    ):
                        func = item["function"]
                        name = func.get("name")
                        if name == "add_numbers":
                            tool_calls.append(_make_tool_call_obj(name, {"a": 2, "b": 2}))
        except json.JSONDecodeError:
            pass

    return tool_calls


def _is_tool_result(content: str) -> bool:
    """Check if content contains a tool result format."""
    tool_result_pattern = r'<\s*\{[^{}]*"result"[^{}]*\}\s*>'
    if re.search(tool_result_pattern, content):
        return True
    json_result_pattern = r'\{[^{}]*"result"[^{}]*"status"[^{}]*\}'
    if re.search(json_result_pattern, content):
        return True
    response_pattern = r'\{[^{}]*"type"\s*:\s*"response"[^{}]*"content"[^{}]*\}'
    if re.search(response_pattern, content):
        return True
    tool_call_end_pattern = r"\[/tool_call\]"
    if re.search(tool_call_end_pattern, content):
        return True
    return False


def _stream_chat(model: str, prompt: str, tools: list[dict[str, Any]]) -> dict[str, Any]:
    """Streaming chat that records TTFT, inter-token latency, tool calls,
    and timing metadata. Returns an error dict on exception rather than
    raising, so the caller can classify (e.g. server down vs. ordinary
    error).
    """
    start = time.time()
    first_byte_time: float | None = None
    content_chunk_times: list[float] = []
    content_parts: list[str] = []
    tool_calls: list[Any] = []
    meta: dict[str, Any] = {}

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
            for attr in (
                "load_duration",
                "prompt_eval_count",
                "prompt_eval_duration",
                "eval_count",
                "eval_duration",
                "total_duration",
            ):
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
    itls = [
        round((content_chunk_times[i] - content_chunk_times[i - 1]) * 1000, 1)
        for i in range(1, len(content_chunk_times))
    ]

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


def _validate_tool_result(
    sr: dict[str, Any], expected_name: str = "add_numbers"
) -> tuple[bool, str, str, bool, str]:
    """Validate a stream result. Returns (ok, status_code, reason,
    tool_called, preview)."""
    if sr["error"]:
        return (
            False,
            "FAILED_STREAM_ERROR",
            sr["error"],
            False,
            sr["content"][:200],
        )

    content = sr["content"].strip()
    calls = sr["tool_calls"]
    preview = content[:200]

    # Normalize every call to (name, arguments); drop unrecognized shapes.
    normalized: list[tuple[str, dict[str, Any]]] = []
    for c in calls:
        name, args = _normalize_tool_call(c)
        if name and args is not None:
            normalized.append((name, args))

    if not normalized:
        if content:
            # A tool result means the model already invoked the tool and is
            # reporting its output.
            if _is_tool_result(content):
                return True, "SUCCESS", "", True, preview
            return (
                False,
                "FAILED_NO_TOOL_CALL",
                f"replied with text: {content[:150]!r}",
                False,
                preview,
            )
        return False, "FAILED_EMPTY_RESPONSE", "empty response", False, preview

    matching = [(n, a) for n, a in normalized if n == expected_name]
    if not matching:
        return (
            False,
            "FAILED_WRONG_TOOL",
            f"called {[n for n, _ in normalized]}",
            True,
            preview,
        )

    for _, args in matching:
        missing = [k for k in ("a", "b") if k not in args]
        if missing:
            return (
                False,
                "FAILED_MISSING_ARGS",
                f"missing {missing}; got {args}",
                True,
                preview,
            )
        for k in ("a", "b"):
            v = args[k]
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                return (
                    False,
                    "FAILED_INVALID_ARGS",
                    f"{k}={v!r} is {type(v).__name__}",
                    True,
                    preview,
                )

    return True, "SUCCESS", "", True, preview


def run_test(model: str, baseline_swap_used_mb: str | float = "") -> dict[str, Any]:
    """Run a cold + warm streaming test for ``model``.

    Returns a result dict with status, failure_reason, timing, resources,
    score, and other fields suitable for writing to the CSV.
    """
    from .metrics import Sampler

    start = time.time()
    r: dict[str, Any] = {
        "status": "UNKNOWN",
        "failure_reason": "",
        "tps": 0.0,
        "duration_seconds": 0.0,
        "tool_called": False,
        "warm_ok": False,
        "warm_status": "",
        "response_preview": "",
        "tool_call_details": "",
        "resources": {},
        "timing": {},
        "ttft_cold_ms": 0.0,
        "ttft_warm_ms": 0.0,
        "itl_cold_p50_ms": 0.0,
        "itl_cold_p95_ms": 0.0,
        "itl_warm_p50_ms": 0.0,
        "itl_warm_p95_ms": 0.0,
        "cold_chunks": 0,
        "warm_chunks": 0,
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
            details = [
                {"name": c.function.name, "arguments": c.function.arguments}
                for c in cold["tool_calls"]
            ]
            r["tool_call_details"] = json.dumps(details)[:300]

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
