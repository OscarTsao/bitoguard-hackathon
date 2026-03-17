from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TableSpec:
    schema: str
    name: str
    primary_key: str
    columns: tuple[tuple[str, str], ...]


RAW_TABLE_SPECS: tuple[TableSpec, ...] = (
    TableSpec("raw", "users", "user_id", (
        ("user_id", "VARCHAR"), ("created_at", "VARCHAR"), ("segment", "VARCHAR"), ("kyc_level", "VARCHAR"),
        ("occupation", "VARCHAR"), ("monthly_income_twd", "DOUBLE"), ("expected_monthly_volume_twd", "DOUBLE"),
        ("declared_source_of_funds", "VARCHAR"), ("residence_country", "VARCHAR"), ("residence_city", "VARCHAR"),
        ("nationality", "VARCHAR"), ("activity_window", "VARCHAR"),
    )),
    TableSpec("raw", "login_events", "login_id", (
        ("login_id", "VARCHAR"), ("user_id", "VARCHAR"), ("occurred_at", "VARCHAR"), ("device_id", "VARCHAR"),
        ("ip_address", "VARCHAR"), ("ip_country", "VARCHAR"), ("ip_city", "VARCHAR"),
        ("is_vpn", "BOOLEAN"), ("is_new_device", "BOOLEAN"), ("is_geo_jump", "BOOLEAN"), ("success", "BOOLEAN"),
    )),
    TableSpec("raw", "fiat_transactions", "fiat_txn_id", (
        ("fiat_txn_id", "VARCHAR"), ("user_id", "VARCHAR"), ("occurred_at", "VARCHAR"), ("direction", "VARCHAR"),
        ("amount_twd", "DOUBLE"), ("currency", "VARCHAR"), ("bank_account_id", "VARCHAR"), ("method", "VARCHAR"), ("status", "VARCHAR"),
    )),
    TableSpec("raw", "trade_orders", "trade_id", (
        ("trade_id", "VARCHAR"), ("user_id", "VARCHAR"), ("occurred_at", "VARCHAR"), ("side", "VARCHAR"),
        ("base_asset", "VARCHAR"), ("quote_asset", "VARCHAR"), ("price_twd", "DOUBLE"), ("quantity", "DOUBLE"),
        ("notional_twd", "DOUBLE"), ("fee_twd", "DOUBLE"), ("order_type", "VARCHAR"), ("status", "VARCHAR"),
    )),
    TableSpec("raw", "crypto_transactions", "crypto_txn_id", (
        ("crypto_txn_id", "VARCHAR"), ("user_id", "VARCHAR"), ("occurred_at", "VARCHAR"), ("direction", "VARCHAR"),
        ("asset", "VARCHAR"), ("network", "VARCHAR"), ("wallet_id", "VARCHAR"), ("counterparty_wallet_id", "VARCHAR"),
        ("amount_asset", "DOUBLE"), ("amount_twd_equiv", "DOUBLE"), ("tx_hash", "VARCHAR"), ("status", "VARCHAR"),
    )),
    TableSpec("raw", "known_blacklist_users", "blacklist_entry_id", (
        ("blacklist_entry_id", "VARCHAR"), ("user_id", "VARCHAR"), ("observed_at", "VARCHAR"),
        ("source", "VARCHAR"), ("reason_code", "VARCHAR"), ("is_active", "BOOLEAN"),
    )),
    TableSpec("raw", "devices", "device_id", (
        ("device_id", "VARCHAR"), ("device_type", "VARCHAR"), ("os_family", "VARCHAR"),
        ("app_channel", "VARCHAR"), ("device_fingerprint", "VARCHAR"), ("first_seen_at", "VARCHAR"),
    )),
    TableSpec("raw", "user_device_links", "link_id", (
        ("link_id", "VARCHAR"), ("user_id", "VARCHAR"), ("device_id", "VARCHAR"),
        ("is_primary", "BOOLEAN"), ("first_seen_at", "VARCHAR"), ("last_seen_at", "VARCHAR"),
    )),
    TableSpec("raw", "bank_accounts", "bank_account_id", (
        ("bank_account_id", "VARCHAR"), ("bank_code", "VARCHAR"), ("bank_name", "VARCHAR"),
        ("country", "VARCHAR"), ("currency", "VARCHAR"), ("opened_at", "VARCHAR"),
    )),
    TableSpec("raw", "user_bank_links", "link_id", (
        ("link_id", "VARCHAR"), ("user_id", "VARCHAR"), ("bank_account_id", "VARCHAR"), ("is_primary", "BOOLEAN"), ("linked_at", "VARCHAR"),
    )),
    TableSpec("raw", "crypto_wallets", "wallet_id", (
        ("wallet_id", "VARCHAR"), ("wallet_kind", "VARCHAR"), ("user_id", "VARCHAR"),
        ("asset", "VARCHAR"), ("network", "VARCHAR"), ("created_at", "VARCHAR"),
    )),
)


CANONICAL_TABLE_SPECS: tuple[TableSpec, ...] = tuple(
    TableSpec("canonical", spec.name, spec.primary_key, spec.columns) for spec in RAW_TABLE_SPECS if spec.name != "known_blacklist_users"
) + (
    TableSpec("canonical", "blacklist_feed", "blacklist_entry_id", (
        ("blacklist_entry_id", "VARCHAR"), ("user_id", "VARCHAR"), ("observed_at", "TIMESTAMPTZ"),
        ("source", "VARCHAR"), ("reason_code", "VARCHAR"), ("is_active", "BOOLEAN"),
    )),
    TableSpec("canonical", "entity_edges", "edge_id", (
        ("edge_id", "VARCHAR"), ("snapshot_time", "TIMESTAMPTZ"), ("src_type", "VARCHAR"), ("src_id", "VARCHAR"),
        ("relation_type", "VARCHAR"), ("dst_type", "VARCHAR"), ("dst_id", "VARCHAR"),
    )),
)


FEATURE_TABLE_SPECS: tuple[TableSpec, ...] = (
    TableSpec("features", "graph_features", "graph_feature_id", (
        ("graph_feature_id", "VARCHAR"), ("user_id", "VARCHAR"), ("snapshot_date", "DATE"),
        ("shared_device_count", "INTEGER"), ("shared_bank_count", "INTEGER"), ("shared_wallet_count", "INTEGER"),
        ("blacklist_1hop_count", "INTEGER"), ("blacklist_2hop_count", "INTEGER"), ("component_size", "INTEGER"),
        ("fan_out_ratio", "DOUBLE"),
    )),
    TableSpec("features", "feature_snapshots_user_day", "feature_snapshot_id", (
        ("feature_snapshot_id", "VARCHAR"), ("user_id", "VARCHAR"), ("snapshot_date", "DATE"), ("feature_version", "VARCHAR")
    )),
    TableSpec("features", "feature_snapshots_user_30d", "feature_snapshot_id", (
        ("feature_snapshot_id", "VARCHAR"), ("user_id", "VARCHAR"), ("snapshot_date", "DATE"), ("feature_version", "VARCHAR")
    )),
    # v2: 174-column expanded feature set (columns added dynamically by pandas)
    TableSpec("features", "feature_snapshots_v2", "feature_snapshot_id", (
        ("feature_snapshot_id", "VARCHAR"), ("user_id", "VARCHAR"), ("snapshot_date", "DATE"), ("feature_version", "VARCHAR")
    )),
    # anomaly: IsolationForest scores + robust z-scores per user
    TableSpec("features", "feature_snapshots_user_anomaly_30d", "feature_snapshot_id", (
        ("feature_snapshot_id", "VARCHAR"), ("user_id", "VARCHAR"), ("snapshot_date", "DATE"), ("feature_version", "VARCHAR")
    )),
)


OPS_TABLE_DDLS: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS ops.sync_runs (
        sync_run_id VARCHAR PRIMARY KEY,
        started_at TIMESTAMPTZ,
        finished_at TIMESTAMPTZ,
        source_url VARCHAR,
        sync_mode VARCHAR,
        start_time TIMESTAMPTZ,
        end_time TIMESTAMPTZ,
        status VARCHAR,
        row_summary JSON,
        error_message VARCHAR
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS ops.data_quality_issues (
        issue_id VARCHAR PRIMARY KEY,
        recorded_at TIMESTAMPTZ,
        table_name VARCHAR,
        issue_type VARCHAR,
        issue_detail VARCHAR,
        row_count INTEGER
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS ops.oracle_user_labels (
        user_id VARCHAR PRIMARY KEY,
        hidden_suspicious_label INTEGER,
        observed_blacklist_label INTEGER,
        scenario_types VARCHAR,
        evidence_tags VARCHAR
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS ops.oracle_scenarios (
        scenario_id VARCHAR PRIMARY KEY,
        scenario_type VARCHAR,
        start_at TIMESTAMPTZ,
        end_at TIMESTAMPTZ,
        description VARCHAR
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS ops.model_predictions (
        prediction_id VARCHAR PRIMARY KEY,
        user_id VARCHAR,
        snapshot_date DATE,
        prediction_time TIMESTAMPTZ,
        model_version VARCHAR,
        risk_score DOUBLE,
        risk_level VARCHAR,
        rule_hits JSON,
        top_reason_codes JSON,
        model_probability DOUBLE,
        anomaly_score DOUBLE,
        graph_risk DOUBLE
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS ops.alerts (
        alert_id VARCHAR PRIMARY KEY,
        user_id VARCHAR,
        snapshot_date DATE,
        created_at TIMESTAMPTZ,
        risk_level VARCHAR,
        status VARCHAR,
        prediction_id VARCHAR,
        report_path VARCHAR
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS ops.cases (
        case_id VARCHAR PRIMARY KEY,
        alert_id VARCHAR,
        user_id VARCHAR,
        created_at TIMESTAMPTZ,
        status VARCHAR,
        latest_decision VARCHAR
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS ops.case_actions (
        action_id VARCHAR PRIMARY KEY,
        case_id VARCHAR,
        action_type VARCHAR,
        actor VARCHAR,
        action_at TIMESTAMPTZ,
        note VARCHAR
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS ops.validation_reports (
        report_id VARCHAR PRIMARY KEY,
        created_at TIMESTAMPTZ,
        model_version VARCHAR,
        report_path VARCHAR,
        metrics_json JSON
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS ops.refresh_state (
        pipeline_name VARCHAR PRIMARY KEY,
        status VARCHAR,
        last_success_at TIMESTAMPTZ,
        last_source_event_at TIMESTAMPTZ,
        last_run_started_at TIMESTAMPTZ,
        last_run_finished_at TIMESTAMPTZ,
        last_error VARCHAR,
        details_json JSON
    )
    """,
    """
    CREATE UNIQUE INDEX IF NOT EXISTS ops_model_predictions_user_snapshot_idx
    ON ops.model_predictions (user_id, snapshot_date)
    """,
    """
    CREATE UNIQUE INDEX IF NOT EXISTS ops_alerts_user_snapshot_idx
    ON ops.alerts (user_id, snapshot_date)
    """,
    """
    CREATE UNIQUE INDEX IF NOT EXISTS ops_cases_alert_id_idx
    ON ops.cases (alert_id)
    """,
)
