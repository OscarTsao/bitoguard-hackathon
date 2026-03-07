from __future__ import annotations

import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import duckdb
import pandas as pd

from db.schema import CANONICAL_TABLE_SPECS, FEATURE_TABLE_SPECS, OPS_TABLE_DDLS, RAW_TABLE_SPECS, TableSpec


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
        with self.connect() as conn:
            conn.register("tmp_df", dataframe)
            conn.execute(f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM tmp_df")
            conn.unregister("tmp_df")

    def append_dataframe(self, table_name: str, dataframe: pd.DataFrame) -> None:
        if dataframe.empty:
            return
        with self.connect() as conn:
            conn.register("tmp_df", dataframe)
            conn.execute(f"INSERT INTO {table_name} SELECT * FROM tmp_df")
            conn.unregister("tmp_df")

    def read_table(self, table_name: str) -> pd.DataFrame:
        with self.connect() as conn:
            return conn.execute(f"SELECT * FROM {table_name}").df()

    def execute(self, sql: str, params: tuple | None = None) -> None:
        with self.connect() as conn:
            conn.execute(sql, params or ())

    def fetch_df(self, sql: str, params: tuple | None = None) -> pd.DataFrame:
        with self.connect() as conn:
            return conn.execute(sql, params or ()).df()


def make_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"
