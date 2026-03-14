from __future__ import annotations
from pathlib import Path
import pandas as pd
import pytest
from db.store import DuckDBStore


def test_replace_table_rejects_unknown_table(tmp_path: Path) -> None:
    store = DuckDBStore(tmp_path / "t.duckdb")
    with pytest.raises(ValueError, match="not in the allowed"):
        store.replace_table("evil.inject", pd.DataFrame({"x": [1]}))


def test_read_table_rejects_unknown_table(tmp_path: Path) -> None:
    store = DuckDBStore(tmp_path / "t.duckdb")
    with pytest.raises(ValueError, match="not in the allowed"):
        store.read_table("ops.nonexistent_table")


def test_append_rejects_unknown_table(tmp_path: Path) -> None:
    store = DuckDBStore(tmp_path / "t.duckdb")
    with pytest.raises(ValueError, match="not in the allowed"):
        store.append_dataframe("'; DROP TABLE ops.alerts; --", pd.DataFrame({"x": [1]}))


def test_read_table_accepts_known_table(tmp_path: Path) -> None:
    """A known-allowed table name must not raise ValueError."""
    store = DuckDBStore(tmp_path / "t.duckdb")
    # ops.alerts is a known allowed table — must not raise
    result = store.read_table("ops.alerts")
    assert isinstance(result, pd.DataFrame)
