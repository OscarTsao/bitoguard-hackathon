from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo


TAIPEI_TZ = ZoneInfo("Asia/Taipei")
PUBLIC_DATASET_DIR = Path(__file__).resolve().parents[2] / "bitoguard_sim_output"


def parse_csv_datetime(value: str | None) -> datetime | None:
    if value in (None, ""):
        return None
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        return dt.replace(tzinfo=TAIPEI_TZ)
    return dt.astimezone(TAIPEI_TZ)


def parse_query_datetime(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=TAIPEI_TZ)
    return value.astimezone(TAIPEI_TZ)


def format_datetime(value: datetime | None) -> str | None:
    if value is None:
        return None
    return value.astimezone(TAIPEI_TZ).isoformat(timespec="seconds")


def normalize_scalar(value: str | None) -> Any:
    if value == "":
        return None
    return value


def parse_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def parse_bool(value: Any) -> bool | None:
    if value in (None, ""):
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized in {"1", "true"}:
        return True
    if normalized in {"0", "false"}:
        return False
    raise ValueError(f"Unsupported boolean value: {value!r}")


@dataclass(frozen=True)
class EndpointSpec:
    name: str
    source_name: str | None
    time_field: str
    id_field: str
    public_fields: tuple[str, ...]
    time_fields: tuple[str, ...]
    numeric_fields: tuple[str, ...]
    bool_fields: tuple[str, ...]
    filter_fields: tuple[str, ...]
    enum_fields: tuple[str, ...] = ()


@dataclass
class Record:
    payload: dict[str, Any]
    timestamps: dict[str, datetime]


ENDPOINT_SPECS: dict[str, EndpointSpec] = {
    "users": EndpointSpec(
        name="users",
        source_name="users.csv",
        time_field="created_at",
        id_field="user_id",
        public_fields=(
            "user_id",
            "created_at",
            "segment",
            "kyc_level",
            "occupation",
            "monthly_income_twd",
            "expected_monthly_volume_twd",
            "declared_source_of_funds",
            "residence_country",
            "residence_city",
            "nationality",
            "activity_window",
        ),
        time_fields=("created_at",),
        numeric_fields=("monthly_income_twd", "expected_monthly_volume_twd"),
        bool_fields=(),
        filter_fields=("user_id",),
    ),
    "login_events": EndpointSpec(
        name="login_events",
        source_name="login_events.csv",
        time_field="occurred_at",
        id_field="login_id",
        public_fields=(
            "login_id",
            "user_id",
            "occurred_at",
            "device_id",
            "ip_address",
            "ip_country",
            "ip_city",
            "is_vpn",
            "is_new_device",
            "is_geo_jump",
            "success",
        ),
        time_fields=("occurred_at",),
        numeric_fields=(),
        bool_fields=("is_vpn", "is_new_device", "is_geo_jump", "success"),
        filter_fields=("user_id", "device_id", "success"),
    ),
    "fiat_transactions": EndpointSpec(
        name="fiat_transactions",
        source_name="fiat_transactions.csv",
        time_field="occurred_at",
        id_field="fiat_txn_id",
        public_fields=(
            "fiat_txn_id",
            "user_id",
            "occurred_at",
            "direction",
            "amount_twd",
            "currency",
            "bank_account_id",
            "method",
            "status",
        ),
        time_fields=("occurred_at",),
        numeric_fields=("amount_twd",),
        bool_fields=(),
        filter_fields=("user_id", "bank_account_id", "direction", "status"),
        enum_fields=("direction", "status"),
    ),
    "trade_orders": EndpointSpec(
        name="trade_orders",
        source_name="trade_orders.csv",
        time_field="occurred_at",
        id_field="trade_id",
        public_fields=(
            "trade_id",
            "user_id",
            "occurred_at",
            "side",
            "base_asset",
            "quote_asset",
            "price_twd",
            "quantity",
            "notional_twd",
            "fee_twd",
            "order_type",
            "status",
        ),
        time_fields=("occurred_at",),
        numeric_fields=("price_twd", "quantity", "notional_twd", "fee_twd"),
        bool_fields=(),
        filter_fields=("user_id", "side", "status", "base_asset", "quote_asset"),
        enum_fields=("side", "status", "base_asset", "quote_asset"),
    ),
    "crypto_transactions": EndpointSpec(
        name="crypto_transactions",
        source_name="crypto_transactions.csv",
        time_field="occurred_at",
        id_field="crypto_txn_id",
        public_fields=(
            "crypto_txn_id",
            "user_id",
            "occurred_at",
            "direction",
            "asset",
            "network",
            "wallet_id",
            "counterparty_wallet_id",
            "amount_asset",
            "amount_twd_equiv",
            "tx_hash",
            "status",
        ),
        time_fields=("occurred_at",),
        numeric_fields=("amount_asset", "amount_twd_equiv"),
        bool_fields=(),
        filter_fields=("user_id", "wallet_id", "counterparty_wallet_id", "direction", "status", "asset", "network"),
        enum_fields=("direction", "status", "asset", "network"),
    ),
    "devices": EndpointSpec(
        name="devices",
        source_name="devices.csv",
        time_field="first_seen_at",
        id_field="device_id",
        public_fields=(
            "device_id",
            "device_type",
            "os_family",
            "app_channel",
            "device_fingerprint",
            "first_seen_at",
        ),
        time_fields=("first_seen_at",),
        numeric_fields=(),
        bool_fields=(),
        filter_fields=("device_id",),
    ),
    "user_device_links": EndpointSpec(
        name="user_device_links",
        source_name="user_device_links.csv",
        time_field="first_seen_at",
        id_field="link_id",
        public_fields=(
            "link_id",
            "user_id",
            "device_id",
            "is_primary",
            "first_seen_at",
            "last_seen_at",
        ),
        time_fields=("first_seen_at", "last_seen_at"),
        numeric_fields=(),
        bool_fields=("is_primary",),
        filter_fields=("user_id", "device_id", "is_primary"),
    ),
    "bank_accounts": EndpointSpec(
        name="bank_accounts",
        source_name="bank_accounts.csv",
        time_field="opened_at",
        id_field="bank_account_id",
        public_fields=(
            "bank_account_id",
            "bank_code",
            "bank_name",
            "country",
            "currency",
            "opened_at",
        ),
        time_fields=("opened_at",),
        numeric_fields=(),
        bool_fields=(),
        filter_fields=("bank_account_id", "country", "currency"),
        enum_fields=("country", "currency"),
    ),
    "user_bank_links": EndpointSpec(
        name="user_bank_links",
        source_name="user_bank_links.csv",
        time_field="linked_at",
        id_field="link_id",
        public_fields=(
            "link_id",
            "user_id",
            "bank_account_id",
            "is_primary",
            "linked_at",
        ),
        time_fields=("linked_at",),
        numeric_fields=(),
        bool_fields=("is_primary",),
        filter_fields=("user_id", "bank_account_id", "is_primary"),
    ),
    "crypto_wallets": EndpointSpec(
        name="crypto_wallets",
        source_name="crypto_wallets.csv",
        time_field="created_at",
        id_field="wallet_id",
        public_fields=(
            "wallet_id",
            "wallet_kind",
            "user_id",
            "asset",
            "network",
            "created_at",
        ),
        time_fields=("created_at",),
        numeric_fields=(),
        bool_fields=(),
        filter_fields=("wallet_id", "user_id", "wallet_kind", "asset", "network"),
        enum_fields=("wallet_kind", "asset", "network"),
    ),
    "known_blacklist_users": EndpointSpec(
        name="known_blacklist_users",
        source_name=None,
        time_field="observed_at",
        id_field="blacklist_entry_id",
        public_fields=(
            "blacklist_entry_id",
            "user_id",
            "observed_at",
            "source",
            "reason_code",
            "is_active",
        ),
        time_fields=("observed_at",),
        numeric_fields=(),
        bool_fields=("is_active",),
        filter_fields=("user_id", "is_active"),
    ),
}


class InvalidFilterValue(ValueError):
    pass


class DataStore:
    def __init__(self, data_dir: Path | None = None) -> None:
        self.data_dir = Path(data_dir or PUBLIC_DATASET_DIR)
        self.records: dict[str, list[Record]] = {}
        self.allowed_values: dict[str, dict[str, set[str]]] = {}
        self._load()

    def _load(self) -> None:
        for name, spec in ENDPOINT_SPECS.items():
            if spec.source_name is None:
                continue
            self.records[name] = self._load_csv_projection(spec)
        self.records["known_blacklist_users"] = self._build_blacklist_projection()
        self._validate_join_integrity()

    def _load_csv_projection(self, spec: EndpointSpec) -> list[Record]:
        path = self.data_dir / spec.source_name
        if not path.exists():
            raise FileNotFoundError(f"Missing source file: {path}")

        records: list[Record] = []
        allowed: dict[str, set[str]] = {field: set() for field in spec.enum_fields}
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                payload: dict[str, Any] = {}
                timestamps: dict[str, datetime] = {}
                for field in spec.public_fields:
                    raw_value = row.get(field)
                    if field in spec.time_fields:
                        dt = parse_csv_datetime(raw_value)
                        timestamps[field] = dt
                        payload[field] = format_datetime(dt)
                    elif field in spec.numeric_fields:
                        payload[field] = parse_float(raw_value)
                    elif field in spec.bool_fields:
                        payload[field] = parse_bool(raw_value)
                    else:
                        payload[field] = normalize_scalar(raw_value)
                        if field in spec.enum_fields and payload[field] is not None:
                            allowed[field].add(str(payload[field]))
                records.append(Record(payload=payload, timestamps=timestamps))

        self.allowed_values[spec.name] = allowed
        records.sort(key=lambda record: (record.timestamps[spec.time_field], record.payload[spec.id_field]))
        return records

    def _build_blacklist_projection(self) -> list[Record]:
        users_path = self.data_dir / "users.csv"
        observed_map: dict[str, datetime] = {}
        for endpoint in ("login_events", "fiat_transactions", "trade_orders", "crypto_transactions"):
            for record in self.records[endpoint]:
                user_id = record.payload["user_id"]
                occurred_at = record.timestamps[ENDPOINT_SPECS[endpoint].time_field]
                if user_id not in observed_map or occurred_at < observed_map[user_id]:
                    observed_map[user_id] = occurred_at

        records: list[Record] = []
        with users_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row.get("observed_blacklist_label") != "1":
                    continue
                created_at = parse_csv_datetime(row.get("created_at"))
                observed_at = observed_map.get(row["user_id"], created_at)
                payload = {
                    "blacklist_entry_id": f"kbl_{row['user_id']}",
                    "user_id": row["user_id"],
                    "observed_at": format_datetime(observed_at),
                    "source": "simulated_blacklist_feed",
                    "reason_code": "blacklist_match",
                    "is_active": True,
                }
                records.append(Record(payload=payload, timestamps={"observed_at": observed_at}))

        self.allowed_values["known_blacklist_users"] = {}
        records.sort(key=lambda record: (record.timestamps["observed_at"], record.payload["blacklist_entry_id"]))
        return records

    def _validate_join_integrity(self) -> None:
        user_ids = {record.payload["user_id"] for record in self.records["users"]}
        device_ids = {record.payload["device_id"] for record in self.records["devices"]}
        bank_account_ids = {record.payload["bank_account_id"] for record in self.records["bank_accounts"]}
        wallet_ids = {record.payload["wallet_id"] for record in self.records["crypto_wallets"]}

        self._assert_foreign_keys("login_events", "user_id", user_ids)
        self._assert_foreign_keys("login_events", "device_id", device_ids)
        self._assert_foreign_keys("fiat_transactions", "user_id", user_ids)
        self._assert_foreign_keys("fiat_transactions", "bank_account_id", bank_account_ids)
        self._assert_foreign_keys("trade_orders", "user_id", user_ids)
        self._assert_foreign_keys("crypto_transactions", "user_id", user_ids)
        self._assert_foreign_keys("crypto_transactions", "wallet_id", wallet_ids)
        self._assert_foreign_keys("crypto_transactions", "counterparty_wallet_id", wallet_ids, allow_null=True)
        self._assert_foreign_keys("user_device_links", "user_id", user_ids)
        self._assert_foreign_keys("user_device_links", "device_id", device_ids)
        self._assert_foreign_keys("user_bank_links", "user_id", user_ids)
        self._assert_foreign_keys("user_bank_links", "bank_account_id", bank_account_ids)
        self._assert_foreign_keys("known_blacklist_users", "user_id", user_ids)

    def _assert_foreign_keys(
        self,
        table_name: str,
        field_name: str,
        valid_ids: set[str],
        allow_null: bool = False,
    ) -> None:
        missing = []
        for record in self.records[table_name]:
            value = record.payload[field_name]
            if value is None and allow_null:
                continue
            if value not in valid_ids:
                missing.append(value)
        if missing:
            raise RuntimeError(f"Invalid foreign keys in {table_name}.{field_name}: {missing[:5]}")

    def _validate_enums(self, endpoint_name: str, filters: dict[str, Any]) -> None:
        allowed = self.allowed_values.get(endpoint_name, {})
        for field, allowed_values in allowed.items():
            value = filters.get(field)
            if value is None:
                continue
            if str(value) not in allowed_values:
                values = ", ".join(sorted(allowed_values))
                raise InvalidFilterValue(f"Invalid value for {field}: {value!r}. Allowed values: {values}")

    def query(
        self,
        endpoint_name: str,
        filters: dict[str, Any],
        start_time: datetime | None,
        end_time: datetime | None,
        page: int,
        page_size: int,
    ) -> dict[str, Any]:
        spec = ENDPOINT_SPECS[endpoint_name]
        start_dt = parse_query_datetime(start_time)
        end_dt = parse_query_datetime(end_time)
        if start_dt and end_dt and start_dt >= end_dt:
            raise InvalidFilterValue("start_time must be earlier than end_time")

        normalized_filters = {key: value for key, value in filters.items() if value is not None}
        self._validate_enums(endpoint_name, normalized_filters)

        filtered: list[Record] = []
        for record in self.records[endpoint_name]:
            natural_time = record.timestamps[spec.time_field]
            if start_dt and natural_time < start_dt:
                continue
            if end_dt and natural_time >= end_dt:
                continue

            matched = True
            for field, expected in normalized_filters.items():
                if record.payload.get(field) != expected:
                    matched = False
                    break
            if matched:
                filtered.append(record)

        total = len(filtered)
        start_index = (page - 1) * page_size
        end_index = start_index + page_size
        page_items = [record.payload for record in filtered[start_index:end_index]]
        return {
            "items": page_items,
            "page": page,
            "page_size": page_size,
            "total": total,
            "has_next": end_index < total,
        }
