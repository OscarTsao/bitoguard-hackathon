from __future__ import annotations

from config import load_settings
from db.store import DuckDBStore
from oracle_client import OracleClient


def load_oracle() -> None:
    settings = load_settings()
    store = DuckDBStore(settings.db_path)
    payload = OracleClient(settings.oracle_dir).load()
    store.replace_table("ops.oracle_user_labels", payload.user_labels)
    scenarios = payload.scenarios.copy()
    for column in ("start_at", "end_at"):
        scenarios[column] = scenarios[column].pipe(lambda s: s.astype(str))
        scenarios[column] = scenarios[column].replace("nan", None)
    store.replace_table("ops.oracle_scenarios", scenarios)


if __name__ == "__main__":
    load_oracle()
