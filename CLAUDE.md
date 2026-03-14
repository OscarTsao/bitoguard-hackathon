# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

All Makefile targets run from the **project root** (`bitoguard-hackathon/`). The Makefile wraps all core commands.

### Setup
```bash
make setup           # Create bitoguard_core/.venv and install Python deps (Python 3.12)
cd bitoguard_frontend && npm install  # Install Node.js deps (Node 20+)
cp bitoguard_frontend/.env.example bitoguard_frontend/.env.local  # Set BITOGUARD_INTERNAL_API_BASE=http://127.0.0.1:8001
```

### Running Tests
```bash
make test            # All 85 tests (pytest, from project root)
make test-quick      # Same, quiet output
make test-rules      # Rule engine only (tests/test_rule_engine.py)

# Run a single test file directly:
cd bitoguard_core && source .venv/bin/activate
PYTHONPATH=. pytest tests/test_smoke.py -v
PYTHONPATH=. pytest tests/test_model_pipeline.py::test_name -v
```

### Development Servers
```bash
make serve           # FastAPI backend on :8001 (hot-reload)
make frontend        # Next.js on :3000
make docker-up       # Full stack via Docker Compose
```

### Pipeline (run in order after first setup)
```bash
make sync            # Sync BitoPro data → raw.* tables in DuckDB
make features        # Build graph + tabular feature snapshots
make train           # Train LightGBM + IsolationForest
make score           # Score users → generate alerts
make drift           # Feature drift detection between latest snapshots
```

### Linting
```bash
make lint            # ruff on bitoguard_core (ignores E501)
cd bitoguard_frontend && npm run lint  # ESLint
```

## Architecture

### System Overview

BitoGuard is a 6-module AML detection system for the BitoPro cryptocurrency exchange. All pipeline scripts are run **from within `bitoguard_core/`** with `PYTHONPATH=.` (the package uses absolute imports but is not installed).

| Module | Purpose | Key Files |
|--------|---------|-----------|
| M1: Rules | 11 deterministic AML rules, severity-weighted scoring | `models/rule_engine.py` |
| M2: Statistical | Peer-deviation features, rolling windows, cohort percentiles | `features/build_features.py` |
| M3: Supervised | LightGBM with leakage-safe temporal splits | `models/train.py`, `models/validate.py` |
| M4: Anomaly | IsolationForest novelty detection | `models/anomaly.py` |
| M5: Graph | NetworkX heterogeneous graph (IP/wallet/device) | `features/graph_features.py` |
| M6: Ops | SHAP case reports, drift detection, incremental refresh | `services/`, `pipeline/refresh_live.py` |

### Data Flow

```
BitoPro API (https://aws-event-api.bitopro.com)
  └─ pipeline/sync.py
      ├─ sync_source.py       → raw.*        (users, login_events, fiat_txns, crypto_txns, trade_orders)
      ├─ load_oracle.py       → ops.oracle_user_labels  (ground truth labels)
      ├─ normalize.py         → canonical.*  (deduplicated, typed)
      └─ rebuild_edges.py     → canonical.entity_edges  (graph edges: user↔device↔wallet↔bank)
  └─ features/
      ├─ graph_features.py    → features.graph_features
      └─ build_features.py    → features.feature_snapshots_user_30d
  └─ models/
      ├─ train.py             → artifacts/models/lgbm_*.lgbm
      ├─ anomaly.py           → artifacts/models/iforest_*.joblib
      └─ score.py             → ops.model_predictions + ops.alerts
```

### Storage: DuckDB Schemas

`bitoguard_core/artifacts/bitoguard.duckdb` has 4 schemas:
- **`raw`**: Ingested directly from source API (11 tables)
- **`canonical`**: Normalized/deduped versions + `entity_edges` graph table
- **`features`**: `graph_features`, `feature_snapshots_user_day`, `feature_snapshots_user_30d`
- **`ops`**: `model_predictions`, `alerts`, `cases`, `sync_runs`, `refresh_state`, `validation_reports`

DuckDB allows **only one writer at a time**. If you see lock errors, check for other processes.

### Backend: `bitoguard_core/`

- **`config.py`**: All settings via environment variables. `load_settings()` is the central config entry point.
- **`api/main.py`**: Single FastAPI app — all endpoints in one file (~14k lines), serving 13 routes.
- **`db/store.py`**: `DuckDBStore` — thin wrapper for queries and DataFrame fetching.
- **`models/common.py`**: Shared utilities: `feature_columns`, `encode_features`, `load_lgbm`, `load_iforest`.
- **`pipeline/refresh_live.py`**: Incremental watermark-bounded refresh — updates only affected users.
- **`oracle_client.py`** / **`source_client.py`**: HTTP clients for the oracle (labels) and source (events) APIs.

### Frontend: `bitoguard_frontend/`

Next.js 16 App Router (React 19, TypeScript, Tailwind CSS v4).

**Pages** (`src/app/`):
- `/alerts` — paginated alert list with risk-level filtering
- `/alerts/[id]` — SHAP diagnosis report + case decision UI
- `/users/[id]` — 360° user view
- `/graph` — entity graph explorer (Cytoscape.js)
- `/model-ops` — validation metrics, threshold tables, drift health

**Key libs** (`src/lib/`):
- `api.ts` — all `fetch()` calls to the backend; proxied through Next.js API routes (`src/app/api/`)
- TanStack Query v5 for data fetching/caching

### Graph Feature Trust Boundary (Important)

Graph features are **split**: most are disabled by default due to known data quality issues.

- **`BITOGUARD_GRAPH_FEATURES_TRUSTED_ONLY=true`** (default) — only 3 features computed
- **Trusted**: `fan_out_ratio`, `shared_wallet_count`, `shared_bank_count`
- **Disabled**: `shared_device_count`, `component_size`, `blacklist_1hop_count`, `blacklist_2hop_count`

**Why disabled**: Artifact A7 — a placeholder device ID (`dev_cfcd208495d565ef66e7dff9f98764da`, MD5 of `"0"`) links ~78% of all users into one giant connected component, completely invalidating device-based graph features. `blacklist_1hop/2hop_count` are additionally disabled due to label leakage (Artifact A5). See `docs/GRAPH_TRUST_BOUNDARY.md` and `docs/GRAPH_RECOVERY_PLAN.md` before enabling.

`UNSAFE_GRAPH_FEATURES` and `PLACEHOLDER_DEVICE_IDS` are defined in `config.py` and enforced in `features/graph_features.py`.

### Key Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `BITOGUARD_DB_PATH` | `bitoguard_core/artifacts/bitoguard.duckdb` | DuckDB file |
| `BITOGUARD_ARTIFACT_DIR` | `bitoguard_core/artifacts/` | Model + report outputs |
| `BITOGUARD_SOURCE_URL` | `https://aws-event-api.bitopro.com` | Data source |
| `BITOGUARD_LABEL_SOURCE` | `hidden_suspicious_label` | Oracle label column |
| `BITOGUARD_GRAPH_FEATURES_TRUSTED_ONLY` | `true` | Disable unsafe graph features |
| `BITOGUARD_CORS_ORIGINS` | `http://localhost:3000` | Comma-separated CORS origins |

### Artifacts

Stored in `bitoguard_core/artifacts/`:
- `bitoguard.duckdb` — all data (gitignored)
- `models/lgbm_*.lgbm` + `lgbm_*.json` — LightGBM model + metadata
- `models/iforest_*.joblib` + `iforest_*.sha256` — IsolationForest model + integrity manifest
- `validation_report.json` — latest holdout evaluation (P@K, calibration, feature importance)

### Test Suite Structure

```
tests/
├── test_rule_engine.py        # 33 tests — all 11 AML rules, trigger/no-trigger cases
├── test_model_pipeline.py     # 15 tests — temporal splits, refresh, drift
├── test_source_integration.py # 6 tests  — canonicalization, sync lifecycle
├── test_smoke.py              # 5 tests  — API smoke, alert/case lifecycle
└── test_graph_data_quality.py # Graph data quality checks
```

Tests run against fixture data (no live API calls required by default).
