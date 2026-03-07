from __future__ import annotations

import argparse
from datetime import datetime

from pipeline.load_oracle import load_oracle
from pipeline.normalize import normalize_raw_to_canonical
from pipeline.rebuild_edges import rebuild_edges
from pipeline.sync_source import sync_source


def run_sync(full: bool = False, start_time: datetime | None = None, end_time: datetime | None = None) -> str:
    sync_run_id = sync_source(start_time=None if full else start_time, end_time=None if full else end_time)
    load_oracle()
    normalize_raw_to_canonical()
    rebuild_edges()
    return sync_run_id


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Full sync from source API to canonical DuckDB tables.")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--start-time")
    parser.add_argument("--end-time")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_sync(
        full=args.full,
        start_time=datetime.fromisoformat(args.start_time) if args.start_time else None,
        end_time=datetime.fromisoformat(args.end_time) if args.end_time else None,
    )
