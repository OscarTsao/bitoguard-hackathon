from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import httpx


@dataclass(frozen=True)
class SourceEndpoint:
    name: str
    path: str
    primary_key: str


SOURCE_ENDPOINTS: tuple[SourceEndpoint, ...] = (
    SourceEndpoint("users", "/v1/users", "user_id"),
    SourceEndpoint("login_events", "/v1/login-events", "login_id"),
    SourceEndpoint("fiat_transactions", "/v1/fiat-transactions", "fiat_txn_id"),
    SourceEndpoint("trade_orders", "/v1/trade-orders", "trade_id"),
    SourceEndpoint("crypto_transactions", "/v1/crypto-transactions", "crypto_txn_id"),
    SourceEndpoint("known_blacklist_users", "/v1/known-blacklist-users", "blacklist_entry_id"),
    SourceEndpoint("devices", "/v1/devices", "device_id"),
    SourceEndpoint("user_device_links", "/v1/user-device-links", "link_id"),
    SourceEndpoint("bank_accounts", "/v1/bank-accounts", "bank_account_id"),
    SourceEndpoint("user_bank_links", "/v1/user-bank-links", "link_id"),
    SourceEndpoint("crypto_wallets", "/v1/crypto-wallets", "wallet_id"),
)


class SourceClient:
    def __init__(self, base_url: str, timeout: float = 30.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def fetch_all(
        self,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        page_size: int = 1000,
    ) -> dict[str, list[dict[str, Any]]]:
        with httpx.Client(base_url=self.base_url, timeout=self.timeout) as client:
            return {
                endpoint.name: self._fetch_endpoint(client, endpoint.path, start_time, end_time, page_size)
                for endpoint in SOURCE_ENDPOINTS
            }

    def _fetch_endpoint(
        self,
        client: httpx.Client,
        path: str,
        start_time: datetime | None,
        end_time: datetime | None,
        page_size: int,
    ) -> list[dict[str, Any]]:
        page = 1
        items: list[dict[str, Any]] = []
        while True:
            params: dict[str, Any] = {"page": page, "page_size": page_size}
            if start_time is not None:
                params["start_time"] = start_time.isoformat()
            if end_time is not None:
                params["end_time"] = end_time.isoformat()
            response = client.get(path, params=params)
            response.raise_for_status()
            payload = response.json()
            items.extend(payload["items"])
            if not payload["has_next"]:
                break
            page += 1
        return items
