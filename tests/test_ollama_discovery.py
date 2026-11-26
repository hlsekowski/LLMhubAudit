"""Unit tests for the Ollama discovery module."""

from __future__ import annotations

from llmhubaudit.ollama.discovery import _param_size_b, fetch_registry_models


class TestParamSizeB:
    """Tests for _param_size_b (Issue #2 A9-style normalization cases)."""

    def test_simple_billion(self):
        assert _param_size_b("llama3:8b") == 8.0

    def test_decimal_billion(self):
        assert _param_size_b("gemma2:2b") == 2.0
        assert _param_size_b("mistral:7b") == 7.0

    def test_uppercase_b(self):
        assert _param_size_b("model:12B") == 12.0

    def test_multiple_sizes_returns_max(self):
        assert _param_size_b("llama3.1:8b-instruct-70b") == 70.0

    def test_no_size_returns_none(self):
        assert _param_size_b("llama3") is None
        assert _param_size_b("model:latest") is None

    def test_float_size(self):
        assert _param_size_b("model:1.3b") == 1.3


class TestFetchRegistryModels:
    """Tests for fetch_registry_models parameter filtering."""

    def test_min_params_filters(self, monkeypatch):
        # Monkeypatch the scraping functions to return controlled data.
        import llmhubaudit.ollama.discovery as d

        monkeypatch.setattr(
            d,
            "_scrape_links",
            lambda url: [
                "/library/llama3",
                "/library/llama3:8b",
                "/library/llama3:70b",
            ],
        )
        monkeypatch.setattr(d, "_repo_has_tools", lambda repo: True)

        models = fetch_registry_models(limit=200, min_params=10, max_params=100)
        assert "llama3:70b" in models
        assert "llama3:8b" not in models

    def test_max_params_filters(self, monkeypatch):
        import llmhubaudit.ollama.discovery as d

        monkeypatch.setattr(
            d,
            "_scrape_links",
            lambda url: [
                "/library/llama3",
                "/library/llama3:8b",
                "/library/llama3:70b",
            ],
        )
        monkeypatch.setattr(d, "_repo_has_tools", lambda repo: True)

        models = fetch_registry_models(limit=200, max_params=10)
        assert "llama3:8b" in models
        assert "llama3:70b" not in models

    def test_tools_badge_required(self, monkeypatch):
        import llmhubaudit.ollama.discovery as d

        monkeypatch.setattr(
            d,
            "_scrape_links",
            lambda url: [
                "/library/llama3",
                "/library/llama3:8b",
                "/library/embed",
                "/library/embed:latest",
            ],
        )
        monkeypatch.setattr(d, "_repo_has_tools", lambda repo: repo == "llama3")

        models = fetch_registry_models(limit=200)
        assert "llama3:8b" in models
        assert not any(m.startswith("embed") for m in models)
