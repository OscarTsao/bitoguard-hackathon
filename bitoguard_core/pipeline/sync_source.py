from __future__ import annotations

import argparse
import json
from datetime import datetime

import pandas as pd

from config import load_settings
from db.store import DuckDBStore, make_id, utc_now
from source_client import SOURCE_ENDPOINTS, SourceClient


def sync_source(start_time: datetime | None = None, end_time: datetime | None = None) -> str:
    settings = load_settings()
    store = DuckDBStore(settings.db_path)
    client = SourceClient(settings.source_url)
    sync_run_id = make_id("sync")
    started_at = utc_now()
    store.execute(
        """
        INSERT INTO ops.sync_runs (
            sync_run_id, started_at, finished_at, source_url, sync_mode, start_time, end_time, status, row_summary, error_message
        ) VALUES (?, ?, NULL, ?, ?, ?, ?, ?, ?, NULL)
        """,
        (
            sync_run_id,
            started_at,
            settings.source_url,
            "incremental" if (start_time or end_time) else "full",
            start_time,
            end_time,
            "running",
            json.dumps({}),
        ),
    )

    try:
        payload = client.fetch_all(start_time=start_time, end_time=end_time)
        summary: dict[str, int] = {}
        loaded_at = utc_now()
        for endpoint in SOURCE_ENDPOINTS:
            dataframe = pd.DataFrame(payload[endpoint.name])
            if dataframe.empty:
                summary[endpoint.name] = 0
                continue
            dataframe["_sync_run_id"] = sync_run_id
            dataframe["_loaded_at"] = loaded_at
            store.append_dataframe(f"raw.{endpoint.name}", dataframe)
            summary[endpoint.name] = len(dataframe)
        store.execute(
            """
            UPDATE ops.sync_runs
            SET finished_at = ?, status = ?, row_summary = ?
            WHERE sync_run_id = ?
            """,
            (utc_now(), "completed", json.dumps(summary), sync_run_id),
        )
        return sync_run_id
    except Exception as exc:  # pragma: no cover - operational path
        store.execute(
            """
            UPDATE ops.sync_runs
            SET finished_at = ?, status = ?, error_message = ?
            WHERE sync_run_id = ?
            """,
            (utc_now(), "failed", str(exc), sync_run_id),
        )
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sync source API into raw DuckDB tables.")
    parser.add_argument("--start-time")
    parser.add_argument("--end-time")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    sync_source(
        start_time=datetime.fromisoformat(args.start_time) if args.start_time else None,
        end_time=datetime.fromisoformat(args.end_time) if args.end_time else None,
    )
