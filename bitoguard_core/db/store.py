from __future__ import annotations

import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import duckdb
import pandas as pd

from db.schema import CANONICAL_TABLE_SPECS, FEATURE_TABLE_SPECS, OPS_TABLE_DDLS, RAW_TABLE_SPECS, TableSpec

_ALLOWED_TABLES: frozenset[str] = frozenset(
    f"{spec.schema}.{spec.name}"
    for specs in (RAW_TABLE_SPECS, CANONICAL_TABLE_SPECS, FEATURE_TABLE_SPECS)
    for spec in specs
) | frozenset({
    "ops.sync_runs", "ops.data_quality_issues", "ops.oracle_user_labels",
    "ops.oracle_scenarios", "ops.model_predictions", "ops.alerts",
    "ops.cases", "ops.case_actions", "ops.validation_reports", "ops.refresh_state",
})

_WRITE_LOCK = threading.Lock()


def _validate_table_name(table_name: str) -> None:
    if table_name not in _ALLOWED_TABLES:
        raise ValueError(f"Table '{table_name}' is not in the allowed table list")


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class DuckDBStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._bootstrap()

    @contextmanager
    def connect(self) -> Iterator[duckdb.DuckDBPyConnection]:
        conn = duckdb.connect(str(self.db_path))
        try:
            yield conn
        finally:
            conn.close()

    @contextmanager
    def transaction(self) -> Iterator[duckdb.DuckDBPyConnection]:
        """Atomic multi-statement write context manager. Acquires write lock."""
        with _WRITE_LOCK, self.connect() as conn:
            conn.execute("BEGIN TRANSACTION")
            try:
                yield conn
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise

    def _bootstrap(self) -> None:
        with self.connect() as conn:
            conn.execute("CREATE SCHEMA IF NOT EXISTS raw")
            conn.execute("CREATE SCHEMA IF NOT EXISTS canonical")
            conn.execute("CREATE SCHEMA IF NOT EXISTS features")
            conn.execute("CREATE SCHEMA IF NOT EXISTS ops")
            for spec in RAW_TABLE_SPECS:
                self._ensure_table(conn, spec, extra_columns=(("_sync_run_id", "VARCHAR"), ("_loaded_at", "TIMESTAMPTZ")))
            for spec in CANONICAL_TABLE_SPECS:
                self._ensure_table(conn, spec)
            for spec in FEATURE_TABLE_SPECS:
                self._ensure_table(conn, spec)
            for ddl in OPS_TABLE_DDLS:
                conn.execute(ddl)

    def _ensure_table(
        self,
        conn: duckdb.DuckDBPyConnection,
        spec: TableSpec,
        extra_columns: tuple[tuple[str, str], ...] = (),
    ) -> None:
        columns = list(spec.columns) + list(extra_columns)
        col_defs = ", ".join(f"{name} {dtype}" for name, dtype in columns)
        conn.execute(f"CREATE TABLE IF NOT EXISTS {spec.schema}.{spec.name} ({col_defs})")

    def replace_table(self, table_name: str, dataframe: pd.DataFrame) -> None:
        _validate_table_name(table_name)
        with _WRITE_LOCK, self.connect() as conn:
            if dataframe.empty:
                # Preserve the existing table schema; just clear all rows.
                conn.execute(f"DELETE FROM {table_name}")
                return
            conn.register("tmp_df", dataframe)
            conn.execute(f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM tmp_df")
            conn.unregister("tmp_df")

    def append_dataframe(self, table_name: str, dataframe: pd.DataFrame) -> None:
        if dataframe.empty:
            return
        _validate_table_name(table_name)
        with _WRITE_LOCK, self.connect() as conn:
            conn.register("tmp_df", dataframe)
            conn.execute(f"INSERT INTO {table_name} SELECT * FROM tmp_df")
            conn.unregister("tmp_df")

    def read_table(self, table_name: str) -> pd.DataFrame:
        _validate_table_name(table_name)
        with self.connect() as conn:
            return conn.execute(f"SELECT * FROM {table_name}").df()

    def execute(self, sql: str, params: tuple | None = None) -> None:
        with _WRITE_LOCK, self.connect() as conn:
            conn.execute(sql, params or ())

    def fetch_df(self, sql: str, params: tuple | None = None) -> pd.DataFrame:
        with self.connect() as conn:
            return conn.execute(sql, params or ()).df()


def make_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"
