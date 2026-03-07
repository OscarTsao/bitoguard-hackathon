from __future__ import annotations

import csv
from pathlib import Path


DATA_DIR = Path(__file__).resolve().parents[2] / "bitoguard_sim_output"
FORBIDDEN_FIELDS = {
    "hidden_suspicious_label",
    "risk_seed_band",
    "scenario_types",
    "evidence_tags",
    "scenario_id",
    "tags",
    "shared_group_id",
    "cluster_id",
    "risk_seed",
    "onchain_risk_seed",
    "observed_blacklist_label",
}


def collect_all_items(client, path: str, page_size: int = 1000) -> list[dict]:
    items: list[dict] = []
    page = 1
    while True:
        response = client.get(path, params={"page": page, "page_size": page_size})
        assert response.status_code == 200
        body = response.json()
        items.extend(body["items"])
        if not body["has_next"]:
            return items
        page += 1


def test_healthcheck(client) -> None:
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_users_contract_and_time_format(client) -> None:
    response = client.get("/v1/users", params={"page_size": 2})
    assert response.status_code == 200
    body = response.json()
    assert body["page"] == 1
    assert body["page_size"] == 2
    assert len(body["items"]) == 2
    for item in body["items"]:
        assert item["created_at"].endswith("+08:00")
        assert isinstance(item["monthly_income_twd"], float)
        assert FORBIDDEN_FIELDS.isdisjoint(item.keys())


def test_login_event_booleans_are_json_booleans(client) -> None:
    response = client.get("/v1/login-events", params={"page_size": 1})
    assert response.status_code == 200
    item = response.json()["items"][0]
    assert isinstance(item["is_vpn"], bool)
    assert isinstance(item["is_new_device"], bool)
    assert isinstance(item["is_geo_jump"], bool)
    assert isinstance(item["success"], bool)


def test_invalid_enum_returns_400(client) -> None:
    response = client.get("/v1/crypto-transactions", params={"asset": "DOGE"})
    assert response.status_code == 400
    body = response.json()
    assert body["error_code"] == "invalid_filter"


def test_invalid_page_returns_400(client) -> None:
    response = client.get("/v1/users", params={"page": 0})
    assert response.status_code == 400
    assert response.json()["error_code"] == "invalid_request"


def test_start_time_is_inclusive_and_end_time_exclusive(client) -> None:
    response = client.get(
        "/v1/fiat-transactions",
        params={
            "start_time": "2026-01-01T00:09:01+08:00",
            "end_time": "2026-01-01T00:09:02+08:00",
            "page_size": 10,
        },
    )
    assert response.status_code == 200
    items = response.json()["items"]
    assert items
    assert items[0]["occurred_at"] == "2026-01-01T00:09:01+08:00"
    response = client.get(
        "/v1/fiat-transactions",
        params={
            "start_time": "2026-01-01T00:09:02+08:00",
            "end_time": "2026-01-01T00:09:03+08:00",
            "page_size": 10,
        },
    )
    assert response.status_code == 200
    assert all(item["occurred_at"] != "2026-01-01T00:09:01+08:00" for item in response.json()["items"])


def test_pagination_is_stable(client) -> None:
    first_page = client.get("/v1/trade-orders", params={"page": 1, "page_size": 5}).json()["items"]
    second_page = client.get("/v1/trade-orders", params={"page": 2, "page_size": 5}).json()["items"]
    assert first_page
    assert second_page
    first_ids = {item["trade_id"] for item in first_page}
    second_ids = {item["trade_id"] for item in second_page}
    assert first_ids.isdisjoint(second_ids)


def test_join_integrity_for_login_events(client) -> None:
    devices = collect_all_items(client, "/v1/devices")
    device_ids = {item["device_id"] for item in devices}
    login_events = collect_all_items(client, "/v1/login-events")
    assert login_events
    assert all(item["device_id"] in device_ids for item in login_events)


def test_wallet_join_integrity_for_crypto_transactions(client) -> None:
    wallets = collect_all_items(client, "/v1/crypto-wallets")
    wallet_ids = {item["wallet_id"] for item in wallets}
    crypto_transactions = collect_all_items(client, "/v1/crypto-transactions")
    assert crypto_transactions
    for item in crypto_transactions:
        assert item["wallet_id"] in wallet_ids
        if item["counterparty_wallet_id"] is not None:
            assert item["counterparty_wallet_id"] in wallet_ids


def test_known_blacklist_projection_count_matches_source(client) -> None:
    response = client.get("/v1/known-blacklist-users", params={"page_size": 1000})
    assert response.status_code == 200
    api_count = response.json()["total"]

    with (DATA_DIR / "users.csv").open(newline="", encoding="utf-8") as handle:
        source_count = sum(1 for row in csv.DictReader(handle) if row["observed_blacklist_label"] == "1")
    assert api_count == source_count


def test_users_do_not_expose_blacklist_oracle_field(client) -> None:
    response = client.get("/v1/users", params={"page_size": 1})
    assert response.status_code == 200
    item = response.json()["items"][0]
    assert "observed_blacklist_label" not in item


def test_openapi_contains_expected_path(client) -> None:
    schema = client.get("/openapi.json").json()
    assert "/v1/known-blacklist-users" in schema["paths"]
    assert "/v1/crypto-wallets" in schema["paths"]
