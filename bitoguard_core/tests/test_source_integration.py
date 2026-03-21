from __future__ import annotations

import json
from pathlib import Path

import httpx
import pandas as pd

from db.store import DuckDBStore
from oracle_client import OracleClient
from pipeline.normalize import normalize_raw_to_canonical
from pipeline.sync_source import sync_source
from source_client import SOURCE_ENDPOINTS, SourceClient, project_postgrest_payload


def _empty_sync_payload() -> dict[str, list[dict]]:
    return {endpoint.name: [] for endpoint in SOURCE_ENDPOINTS}


def _parse_row_summary(value: object) -> dict[str, int]:
    if isinstance(value, dict):
        return value
    if value in (None, ""):
        return {}
    return json.loads(str(value))


def test_project_postgrest_payload_scales_and_derives_views() -> None:
    payload = {
        "user_info": [
            {
                "user_id": 100,
                "sex": 1,
                "age": 30,
                "career": 22,
                "income_source": 7,
                "confirmed_at": "2025-10-01T00:00:05",
                "level1_finished_at": "2025-10-01T00:00:00",
                "level2_finished_at": None,
                "user_source": 1,
            }
        ],
        "twd_transfer": [
            {
                "id": 10,
                "user_id": 100,
                "created_at": "2025-10-02T00:00:00",
                "kind": 0,
                "ori_samount": 15000000000,
                "source_ip_hash": "ip1",
            }
        ],
        "usdt_twd_trading": [
            {
                "id": 20,
                "user_id": 100,
                "updated_at": "2025-10-03T00:00:00",
                "is_buy": 1,
                "trade_samount": 2500000000,
                "twd_srate": 3100000000,
                "source_ip_hash": "ip1",
                "is_market": 1,
                "source": 1,
            }
        ],
        "usdt_swap": [],
        "crypto_transfer": [
            {
                "id": 30,
                "user_id": 100,
                "created_at": "2025-10-04T00:00:00",
                "kind": 1,
                "sub_kind": 0,
                "ori_samount": 200000000,
                "twd_srate": 3000000000,
                "relation_user_id": None,
                "from_wallet_hash": "wallet_user",
                "to_wallet_hash": "wallet_ext",
                "protocol": 4,
                "currency": "usdt",
                "source_ip_hash": "ip2",
            }
        ],
        "train_label": [
            {
                "user_id": 100,
                "status": 1,
            }
        ],
    }

    projected = project_postgrest_payload(payload)

    assert projected["users"][0]["user_id"] == "100"
    assert projected["users"][0]["created_at"] == "2025-10-01T00:00:00+08:00"
    assert projected["users"][0]["segment"] == "app"
    assert projected["users"][0]["kyc_level"] == "level1"
    assert projected["users"][0]["activity_window"] == "2025-10-02..2025-10-04"

    fiat = projected["fiat_transactions"][0]
    assert fiat["occurred_at"] == "2025-10-02T00:00:00+08:00"
    assert fiat["direction"] == "deposit"
    assert fiat["amount_twd"] == 150.0

    trade = projected["trade_orders"][0]
    assert trade["occurred_at"] == "2025-10-03T00:00:00+08:00"
    assert trade["side"] == "buy"
    assert trade["price_twd"] == 31.0
    assert trade["quantity"] == 25.0
    assert trade["notional_twd"] == 775.0

    crypto = projected["crypto_transactions"][0]
    assert crypto["occurred_at"] == "2025-10-04T00:00:00+08:00"
    assert crypto["direction"] == "withdrawal"
    assert crypto["wallet_id"] == "wallet_user"
    assert crypto["counterparty_wallet_id"] == "wallet_ext"
    assert crypto["amount_asset"] == 2.0
    assert crypto["amount_twd_equiv"] == 60.0
    assert crypto["network"] == "TRC20"

    assert [event["occurred_at"] for event in projected["login_events"]] == [
        "2025-10-02T00:00:00+08:00",
        "2025-10-03T00:00:00+08:00",
        "2025-10-04T00:00:00+08:00",
    ]
    assert [event["is_new_device"] for event in projected["login_events"]] == [False, False, False]
    assert all(event["device_id"] is None for event in projected["login_events"])
    assert projected["devices"] == []
    assert projected["user_device_links"] == []
    assert projected["known_blacklist_users"][0]["observed_at"] == "2025-10-02T00:00:00+08:00"
    assert projected["known_blacklist_users"][0]["reason_code"] == "train_label_status_1"
    assert projected["bank_accounts"] == []
    assert projected["user_bank_links"] == []


def test_oracle_client_loads_postgrest_train_labels() -> None:
    openapi = {"paths": {"/user_info": {}, "/crypto_transfer": {}}}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/":
            return httpx.Response(200, json=openapi)
        if request.url.path == "/train_label":
            offset = int(request.url.params.get("offset", "0"))
            if offset == 0:
                return httpx.Response(200, json=[{"user_id": 100, "status": 1}, {"user_id": 200, "status": 1}])
            return httpx.Response(200, json=[])
        raise AssertionError(f"Unexpected request: {request.url}")

    client = OracleClient(
        source_url="https://aws-event-api.bitopro.com",
        transport=httpx.MockTransport(handler),
    )
    payload = client.load()

    assert list(payload.user_labels["user_id"]) == ["100", "200"]
    assert list(payload.user_labels["hidden_suspicious_label"]) == [1, 1]
    assert list(payload.user_labels["observed_blacklist_label"]) == [1, 1]
    assert payload.scenarios.empty
    assert list(payload.scenarios.columns) == ["scenario_id", "scenario_type", "start_at", "end_at", "description"]


def test_full_sync_replaces_previous_raw_rows(tmp_path: Path, monkeypatch) -> None:
    db_path = tmp_path / "bitoguard.duckdb"
    monkeypatch.setenv("BITOGUARD_DB_PATH", str(db_path))
    monkeypatch.setenv("BITOGUARD_SOURCE_URL", "https://aws-event-api.bitopro.com")

    payload_a = _empty_sync_payload()
    payload_a["users"] = [{
        "user_id": "old",
        "created_at": "2025-10-01T00:00:00+00:00",
        "segment": "web",
        "kyc_level": "level1",
        "occupation": "career_1",
        "monthly_income_twd": None,
        "expected_monthly_volume_twd": None,
        "declared_source_of_funds": "income_source_1",
        "residence_country": None,
        "residence_city": None,
        "nationality": None,
        "activity_window": "2025-10-01..2025-10-01",
    }]
    payload_b = _empty_sync_payload()
    payload_b["users"] = [{
        "user_id": "new",
        "created_at": "2025-10-02T00:00:00+00:00",
        "segment": "app",
        "kyc_level": "level2",
        "occupation": "career_2",
        "monthly_income_twd": None,
        "expected_monthly_volume_twd": None,
        "declared_source_of_funds": "income_source_2",
        "residence_country": None,
        "residence_city": None,
        "nationality": None,
        "activity_window": "2025-10-02..2025-10-02",
    }]

    payloads = [payload_a, payload_b]

    def fake_fetch_all(self, start_time=None, end_time=None, page_size=1000, progress_callback=None):  # noqa: ANN001, ANN202
        return payloads.pop(0)

    monkeypatch.setattr("pipeline.sync_source.SourceClient.fetch_all", fake_fetch_all)

    sync_source()
    sync_source()

    store = DuckDBStore(db_path)
    users = store.read_table("raw.users")
    assert users["user_id"].tolist() == ["new"]


def test_sync_reconciles_abandoned_running_rows(tmp_path: Path, monkeypatch) -> None:
    db_path = tmp_path / "bitoguard.duckdb"
    monkeypatch.setenv("BITOGUARD_DB_PATH", str(db_path))
    monkeypatch.setenv("BITOGUARD_SOURCE_URL", "https://aws-event-api.bitopro.com")

    store = DuckDBStore(db_path)
    store.execute(
        """
        INSERT INTO ops.sync_runs (
            sync_run_id, started_at, finished_at, source_url, sync_mode, start_time, end_time, status, row_summary, error_message
        ) VALUES (?, CURRENT_TIMESTAMP, NULL, ?, ?, NULL, NULL, ?, ?, NULL)
        """,
        (
            "sync_stale",
            "https://aws-event-api.bitopro.com",
            "full",
            "running",
            json.dumps({}),
        ),
    )

    def fake_fetch_all(self, start_time=None, end_time=None, page_size=1000, progress_callback=None):  # noqa: ANN001, ANN202
        return _empty_sync_payload()

    monkeypatch.setattr("pipeline.sync_source.SourceClient.fetch_all", fake_fetch_all)

    sync_run_id = sync_source()

    stale_row = store.fetch_df(
        """
        SELECT status, finished_at, error_message
        FROM ops.sync_runs
        WHERE sync_run_id = ?
        """,
        ("sync_stale",),
    ).iloc[0]
    assert stale_row["status"] == "failed"
    assert pd.notna(stale_row["finished_at"])
    assert "abandoned by newer sync run" in stale_row["error_message"]

    new_row = store.fetch_df(
        """
        SELECT status, finished_at
        FROM ops.sync_runs
        WHERE sync_run_id = ?
        """,
        (sync_run_id,),
    ).iloc[0]
    assert new_row["status"] == "completed"
    assert pd.notna(new_row["finished_at"])


def test_live_postgrest_sync_persists_fetch_progress(tmp_path: Path, monkeypatch) -> None:
    db_path = tmp_path / "bitoguard.duckdb"
    monkeypatch.setenv("BITOGUARD_DB_PATH", str(db_path))
    monkeypatch.setenv("BITOGUARD_SOURCE_URL", "https://aws-event-api.bitopro.com")

    def running_row_summary() -> dict[str, int]:
        row = DuckDBStore(db_path).fetch_df(
            """
            SELECT CAST(row_summary AS VARCHAR) AS row_summary
            FROM ops.sync_runs
            WHERE status = 'running'
            ORDER BY started_at DESC
            LIMIT 1
            """
        ).iloc[0]
        return _parse_row_summary(row["row_summary"])

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/":
            return httpx.Response(200, json={"paths": {"/user_info": {}, "/crypto_transfer": {}}})
        if request.url.path == "/user_info":
            return httpx.Response(200, json=[{"user_id": 100, "confirmed_at": "2025-10-01T00:00:00"}])
        if request.url.path == "/twd_transfer":
            assert running_row_summary() == {"user_info": 1}
            return httpx.Response(200, json=[])
        if request.url.path in {"/usdt_twd_trading", "/usdt_swap", "/crypto_transfer", "/train_label"}:
            return httpx.Response(200, json=[])
        raise AssertionError(f"Unexpected request: {request.url}")

    class PostgrestTestClient(SourceClient):
        def __init__(self, base_url: str, timeout: float = 30.0) -> None:
            super().__init__(base_url, timeout=timeout, transport=httpx.MockTransport(handler))

    monkeypatch.setattr("pipeline.sync_source.SourceClient", PostgrestTestClient)

    sync_run_id = sync_source()

    store = DuckDBStore(db_path)
    sync_row = store.fetch_df(
        """
        SELECT status, CAST(row_summary AS VARCHAR) AS row_summary
        FROM ops.sync_runs
        WHERE sync_run_id = ?
        """,
        (sync_run_id,),
    ).iloc[0]
    assert sync_row["status"] == "completed"
    assert _parse_row_summary(sync_row["row_summary"])["users"] == 1


def test_normalize_replaces_canonical_with_empty_frame(tmp_path: Path, monkeypatch) -> None:
    db_path = tmp_path / "bitoguard.duckdb"
    monkeypatch.setenv("BITOGUARD_DB_PATH", str(db_path))

    store = DuckDBStore(db_path)
    store.replace_table(
        "canonical.users",
        pd.DataFrame(
            [
                {
                    "user_id": "stale",
                    "created_at": "2025-10-01T00:00:00+00:00",
                    "segment": None,
                    "kyc_level": None,
                    "occupation": None,
                    "monthly_income_twd": None,
                    "expected_monthly_volume_twd": None,
                    "declared_source_of_funds": None,
                    "residence_country": None,
                    "residence_city": None,
                    "nationality": None,
                    "activity_window": None,
                }
            ]
        ),
    )

    normalize_raw_to_canonical()

    users = store.read_table("canonical.users")
    assert users.empty

def test_replace_table_in_transaction_rejects_unknown_table():
    """_replace_table_in_transaction must reject tables not in the allowlist."""
    import duckdb
    import pandas as pd
    from pipeline.normalize import _replace_table_in_transaction

    conn = duckdb.connect(':memory:')
    df = pd.DataFrame([{'col': 1}])
    try:
        _replace_table_in_transaction(conn, 'evil.injected_table', df)
        assert False, 'Should have raised ValueError for unknown table'
    except ValueError as e:
        assert 'not in the allowed' in str(e).lower() or 'allowlist' in str(e).lower() or 'evil' in str(e)
    finally:
        conn.close()

