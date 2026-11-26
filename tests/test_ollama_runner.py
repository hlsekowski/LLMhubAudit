"""Unit tests for the Ollama runner module (tool-call classification)."""

from __future__ import annotations

from llmhubaudit.ollama.runner import (
    _make_tool_call_obj,
    _normalize_tool_call,
    _parse_tool_calls_from_content,
    _validate_tool_result,
)


class TestNormalizeToolCall:
    """Tests for _normalize_tool_call (Issue #2 A9 cases)."""

    def test_object_with_function_attribute(self):
        c = _make_tool_call_obj("add_numbers", {"a": 2, "b": 3})
        name, args = _normalize_tool_call(c)
        assert name == "add_numbers"
        assert args == {"a": 2, "b": 3}

    def test_dict_name_arguments(self):
        c = {"name": "add_numbers", "arguments": {"a": 2, "b": 3}}
        name, args = _normalize_tool_call(c)
        assert name == "add_numbers"
        assert args == {"a": 2, "b": 3}

    def test_dict_function_name(self):
        c = {"function_name": "add_numbers", "arguments": {"a": 2, "b": 3}}
        name, args = _normalize_tool_call(c)
        assert name == "add_numbers"
        assert args == {"a": 2, "b": 3}

    def test_dict_type_function(self):
        c = {
            "type": "function",
            "function": {"name": "add_numbers", "arguments": {"a": 2, "b": 3}},
        }
        name, args = _normalize_tool_call(c)
        assert name == "add_numbers"
        assert args == {"a": 2, "b": 3}

    def test_dict_reversed_keys(self):
        c = {"arguments": {"a": 2, "b": 3}, "name": "add_numbers"}
        name, args = _normalize_tool_call(c)
        assert name == "add_numbers"
        assert args == {"a": 2, "b": 3}

    def test_unrecognized_shape_returns_none(self):
        assert _normalize_tool_call({"foo": "bar"}) == (None, None)
        assert _normalize_tool_call("not a dict") == (None, None)
        assert _normalize_tool_call(42) == (None, None)

    def test_non_dict_arguments_defaults_to_empty(self):
        c = {"name": "add_numbers", "arguments": "not a dict"}
        name, args = _normalize_tool_call(c)
        assert name == "add_numbers"
        assert args == {}


class TestParseToolCallsFromContent:
    """Tests for parsing tool calls from raw content text."""

    def test_plain_json(self):
        content = '{"name": "add_numbers", "arguments": {"a": 2, "b": 3}}'
        calls = _parse_tool_calls_from_content(content)
        assert len(calls) == 1
        name, args = _normalize_tool_call(calls[0])
        assert name == "add_numbers"
        assert args == {"a": 2, "b": 3}

    def _has_call(self, calls, name, args):
        return any(_normalize_tool_call(c) == (name, args) for c in calls)

    def test_tool_call_tags(self):
        content = '<tool-call>[{"name": "add_numbers", "arguments": {"a": 2, "b": 3}}]</tool-call>'
        calls = _parse_tool_calls_from_content(content)
        assert calls
        assert self._has_call(calls, "add_numbers", {"a": 2, "b": 3})

    def test_angle_bracket_json(self):
        content = '<{"name": "add_numbers", "arguments": {"a": 2, "b": 3}}>'
        calls = _parse_tool_calls_from_content(content)
        assert calls
        assert self._has_call(calls, "add_numbers", {"a": 2, "b": 3})

    def test_markdown_code_block(self):
        content = '```json\n{"name": "add_numbers", "arguments": {"a": 2, "b": 3}}\n```'
        calls = _parse_tool_calls_from_content(content)
        assert calls
        assert self._has_call(calls, "add_numbers", {"a": 2, "b": 3})

    def test_no_tool_calls(self):
        content = "The answer is 5."
        calls = _parse_tool_calls_from_content(content)
        assert calls == []


class TestValidateToolResult:
    """Tests for _validate_tool_result classification."""

    def test_success(self):
        sr = {
            "error": None,
            "content": "",
            "tool_calls": [_make_tool_call_obj("add_numbers", {"a": 2, "b": 3})],
        }
        ok, status, reason, called, preview = _validate_tool_result(sr)
        assert ok is True
        assert status == "SUCCESS"
        assert called is True

    def test_error(self):
        sr = {"error": "ConnectionError: refused", "content": "", "tool_calls": []}
        ok, status, reason, called, preview = _validate_tool_result(sr)
        assert ok is False
        assert status == "FAILED_STREAM_ERROR"
        assert called is False

    def test_empty_response(self):
        sr = {"error": None, "content": "", "tool_calls": []}
        ok, status, reason, called, preview = _validate_tool_result(sr)
        assert ok is False
        assert status == "FAILED_EMPTY_RESPONSE"

    def test_wrong_tool(self):
        sr = {
            "error": None,
            "content": "",
            "tool_calls": [_make_tool_call_obj("other_tool", {"a": 2})],
        }
        ok, status, reason, called, preview = _validate_tool_result(sr)
        assert ok is False
        assert status == "FAILED_WRONG_TOOL"

    def test_missing_args(self):
        sr = {
            "error": None,
            "content": "",
            "tool_calls": [_make_tool_call_obj("add_numbers", {"a": 2})],
        }
        ok, status, reason, called, preview = _validate_tool_result(sr)
        assert ok is False
        assert status == "FAILED_MISSING_ARGS"

    def test_invalid_args(self):
        sr = {
            "error": None,
            "content": "",
            "tool_calls": [_make_tool_call_obj("add_numbers", {"a": "x", "b": 3})],
        }
        ok, status, reason, called, preview = _validate_tool_result(sr)
        assert ok is False
        assert status == "FAILED_INVALID_ARGS"

    def test_tool_result_content(self):
        sr = {
            "error": None,
            "content": '<{"result": 42, "status": true}>',
            "tool_calls": [],
        }
        ok, status, reason, called, preview = _validate_tool_result(sr)
        assert ok is True
        assert status == "SUCCESS"
