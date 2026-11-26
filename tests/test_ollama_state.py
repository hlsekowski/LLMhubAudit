"""Unit tests for the Ollama state module (CSV migration, history, disk guard)."""

from __future__ import annotations

import csv

from llmhubaudit.ollama import state


class TestFreePct:
    def test_returns_float(self):
        pct = state.free_pct("/")
        assert isinstance(pct, float)
        assert 0 <= pct <= 100


class TestCSVFieldsMigration:
    def test_new_csv_uses_all_required_cols(self, tmp_path, monkeypatch):
        monkeypatch.setattr(state, "CSV", str(tmp_path / "results.csv"))
        state._FIELDS = None
        fields = state.csv_fields()
        assert fields == state.ALL_REQUIRED_COLS

    def test_migration_adds_missing_cols(self, tmp_path, monkeypatch):
        csv_path = tmp_path / "results.csv"
        monkeypatch.setattr(state, "CSV", str(csv_path))
        state._FIELDS = None
        # Write an old CSV with only a subset of columns.
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["timestamp", "model", "status"])
            w.writeheader()
            w.writerow({"timestamp": "t1", "model": "m1", "status": "SUCCESS"})
        fields = state.csv_fields()
        assert "timestamp" in fields
        assert "model" in fields
        assert "status" in fields
        assert "score" in fields
        assert "ttft_cold_ms" in fields
        # Old row preserved.
        with open(csv_path, newline="") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 1
        assert rows[0]["model"] == "m1"

    def test_no_migration_when_all_cols_present(self, tmp_path, monkeypatch):
        csv_path = tmp_path / "results.csv"
        monkeypatch.setattr(state, "CSV", str(csv_path))
        state._FIELDS = None
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(state.ALL_REQUIRED_COLS))
            w.writeheader()
            w.writerow({c: "" for c in state.ALL_REQUIRED_COLS})
        fields = state.csv_fields()
        assert fields == list(state.ALL_REQUIRED_COLS)


class TestHistory:
    def test_empty_when_no_csv(self, tmp_path, monkeypatch):
        monkeypatch.setattr(state, "CSV", str(tmp_path / "results.csv"))
        assert state.history() == {}

    def test_parses_rows(self, tmp_path, monkeypatch):
        csv_path = tmp_path / "results.csv"
        monkeypatch.setattr(state, "CSV", str(csv_path))
        state._FIELDS = None
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["timestamp", "model", "status"])
            w.writeheader()
            w.writerow({"timestamp": "t1", "model": "m1", "status": "SUCCESS"})
            w.writerow({"timestamp": "t2", "model": "m2", "status": "FAILED"})
        hist = state.history()
        assert hist["m1"] == {"status": "SUCCESS", "timestamp": "t1"}
        assert hist["m2"] == {"status": "FAILED", "timestamp": "t2"}


class TestStateFile:
    def test_read_state_missing(self, tmp_path, monkeypatch):
        monkeypatch.setattr(state, "STATE", str(tmp_path / "state.json"))
        assert state.read_state() is None

    def test_read_state_malformed(self, tmp_path, monkeypatch):
        monkeypatch.setattr(state, "STATE", str(tmp_path / "state.json"))
        with open(tmp_path / "state.json", "w") as f:
            f.write("{not valid json")
        assert state.read_state() is None

    def test_write_and_read_state(self, tmp_path, monkeypatch):
        monkeypatch.setattr(state, "STATE", str(tmp_path / "state.json"))
        state.write_state("llama3:8b")
        st = state.read_state()
        assert st is not None
        assert st["model"] == "llama3:8b"
        assert "pid" in st
        assert "started_at" in st

    def test_clear_state(self, tmp_path, monkeypatch):
        monkeypatch.setattr(state, "STATE", str(tmp_path / "state.json"))
        state.write_state("llama3:8b")
        state.clear_state()
        assert state.read_state() is None


class TestDiskGuard:
    def test_free_pct_above_min(self):
        # Just sanity-check that free_pct works on the real root.
        pct = state.free_pct("/")
        assert pct >= 0
