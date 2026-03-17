# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

All Makefile targets run from the **project root** (`bitoguard-hackathon/`). The Makefile wraps all core commands.

### Setup
```bash
make setup           # Create bitoguard_core/.venv and install Python deps (Python 3.13)
pip install -e bitoguard_core/  # Install bitoguard_core as editable package
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
make train           # Train CatBoost + LightGBM stacker (v2 features)
make score           # Score users → generate alerts
make drift           # Feature drift detection between latest snapshots
make refresh         # Incremental watermark-bounded refresh
```

### AWS Event Data Pipeline (offline/competition)
```bash
# Step 1: Fetch raw data from the aws-event-api endpoint into parquet files
cd bitoguard_core && source .venv/bin/activate
python ../scripts/fetch_aws_event_data.py --output-dir data/aws_event/raw

# Step 2: Clean raw parquet files → typed, labeled, scaled clean parquet
python ../scripts/clean_aws_event_data.py --raw-dir data/aws_event/raw --output-dir data/aws_event/clean

# Step 3: Run the official pipeline (features → train → validate → score)
PYTHONPATH=. python -m official.pipeline

# Or run transductive_v1 pipeline:
PYTHONPATH=. python -m transductive_v1.pipeline
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
| M3: Supervised | CatBoost + LightGBM stacker, 5-fold OOF, AUC 0.9495 | `models/stacker.py`, `models/score.py` |
| M4: Anomaly | IsolationForest novelty detection | `models/anomaly.py` |
| M5: Graph | NetworkX heterogeneous graph (IP/wallet/device) | `features/graph_features.py` |
| M6: Ops | SHAP case reports, drift detection, incremental refresh | `services/`, `pipeline/refresh_live.py` |
| V2 Features | 8-module label-free registry (~155 cols/user) | `features/registry.py`, `features/build_features_v2.py` |
| Stacker | CatBoost + LightGBM OOF → LR meta-learner (5-fold) | `models/stacker.py`, `models/train_catboost.py` |

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
      ├─ stacker.py           → artifacts/models/stacker_*.joblib
      ├─ anomaly.py           → artifacts/models/iforest_*.joblib
      └─ score.py             → ops.model_predictions + ops.alerts
```

### AWS Event Data Flow (Offline/Competition)

```
aws-event-api.bitopro.com
  └─ scripts/fetch_aws_event_data.py → data/aws_event/raw/*.parquet  (7 tables)
  └─ scripts/clean_aws_event_data.py → data/aws_event/clean/*.parquet (7 tables + user_index)
      ├─ Decimal scaling: raw s-amounts × 1e-8 → human-scale floats
      ├─ Enum decoding: integer codes → string labels (kind_label, side_label, protocol_label, etc.)
      └─ Derived columns: amount_twd_equiv, kyc_level, days_email_to_level1, etc.
  └─ official/ pipeline  → artifacts/official_features/, artifacts/models/, artifacts/predictions/
  └─ transductive_v1/    → artifacts/transductive_v1/
```

The `clean_aws_event_data.py` script produces a `user_index.parquet` that joins `user_info`, `train_label`, and `predict_label` into a single cohort-assignment table. Both `official/` and `transductive_v1/` pipelines read from the `clean/` directory via their respective `common.load_clean_table()`.

### Storage: DuckDB Schemas

`bitoguard_core/artifacts/bitoguard.duckdb` has 4 schemas:
- **`raw`**: Ingested directly from source API (11 tables)
- **`canonical`**: Normalized/deduped versions + `entity_edges` graph table
- **`features`**: `graph_features`, `feature_snapshots_user_day`, `feature_snapshots_user_30d`, `feature_snapshots_v2`
- **`ops`**: `model_predictions`, `alerts`, `cases`, `sync_runs`, `refresh_state`, `validation_reports`

DuckDB allows **only one writer at a time**. If you see lock errors, check for other processes.

### Backend: `bitoguard_core/`

- **`config.py`**: All settings via environment variables. `load_settings()` is the central config entry point. Also provides `aws_event_raw_dir` and `aws_event_clean_dir` paths used by the official and transductive pipelines.
- **`api/main.py`**: Single FastAPI app — all endpoints in one file (~14k lines), serving 13 routes.
- **`db/store.py`**: `DuckDBStore` — thin wrapper for queries and DataFrame fetching.
- **`models/common.py`**: Shared utilities: `feature_columns`, `encode_features`, `load_lgbm`, `load_iforest`.
- **`models/train.py`**: LightGBM temporal-split trainer (uses DuckDB `feature_snapshots`). Date-based: first 20 days train, next 5 validation, remaining holdout.
- **`models/validate.py`**: Holdout evaluation producing `validation_report.json` with confusion matrix, PR curve, threshold sensitivity, and per-scenario breakdown. Writes to `ops.validation_reports`.
- **`models/anomaly_common.py`**: Shared anomaly model infrastructure — cohort-aware peer-deviation z-scores (MAD-based), log-transformed features, clip bounds fitting, and `apply_anomaly_model()` for scoring.
- **`pipeline/refresh_live.py`**: Incremental watermark-bounded refresh — updates only affected users.
- **`oracle_client.py`** / **`source_client.py`**: HTTP clients for the oracle (labels) and source (events) APIs.
- **`features/registry.py`**: Assembles 8 label-free feature modules (profile, twd, crypto, swap, trading, ip, sequence, bipartite) into one ~155-column master table. Always zero-fills missing module columns to keep schema stable.
- **`features/graph_propagation.py`**: Label-aware propagation features (7 features). **Leakage contract**: must receive only training-fold labels; never called with test/validation labels. Used in the stacker per-fold, not in the static registry.

### Official Pipeline: `bitoguard_core/official/`

The official pipeline is a self-contained offline training system designed for the aws-event competition dataset. It reads from `data/aws_event/clean/*.parquet` (not DuckDB) and writes artifacts to `bitoguard_core/artifacts/`. Entry point: `official/pipeline.py` or `python -m official.pipeline`.

**Pipeline stages** (executed in order by `run_official_pipeline()`):

| Stage | Module | Output |
|-------|--------|--------|
| Data contract | `cohorts.py` | `official_features/cohorts_full.parquet`, `reports/official_data_contract_report.json` |
| Features | `features.py` | `official_features/official_user_features_full.parquet` (~150+ tabular features) |
| Graph features | `graph_features.py` | `official_features/official_graph_features_full.parquet` (IP/wallet/relation component, degree, centrality) |
| Anomaly features | `anomaly.py` | `official_features/official_anomaly_features_full.parquet` (IsolationForest scores + robust z-scores) |
| Train | `train.py` | 3 base models (CatBoost A, CatBoost B, GraphSAGE) + LR stacker + `official_bundle.json` |
| Validate | `validate.py` | `reports/official_validation_report.json` (primary + secondary group-stress metrics) |
| Score | `score.py` | `predictions/official_predict_scores.{parquet,csv}` |

**Model architecture** — 3-branch stacked ensemble:
- **Base A** (CatBoost): Label-free tabular features only (~150 columns).
- **Base B** (CatBoost): Label-free + transductive features (label-aware graph propagation: seed distances, PPR, per-edge-type positive-neighbor counts, entity reputation scores).
- **Base C** (GraphSAGE): 2-layer GNN operating on the collapsed user-user graph (relation/wallet/IP edges with type-dependent weights). Requires PyTorch.
- **Stacker**: Logistic regression over `[base_a_prob, base_b_prob, base_c_prob, rule_score, anomaly_score]`.

**Validation protocol** — dual-split:
- **Primary**: Label-mask transductive CV (5-fold `StratifiedKFold` on labeled users; all users remain in the graph). Transductive features are rebuilt per fold using only train-fold labels.
- **Secondary**: Strict group-aware stress test via `StratifiedGroupKFold`. Groups are built with `UnionFind` over shared wallets (2-10 users), shared IPs (2-5 users, min 2 events), and direct relation edges. Weak/soft edges (larger entity clusters) produce a purge map for cross-group contamination tracking.

**Calibration and thresholding**: `stacking.py` selects the best calibrator (raw/sigmoid/beta/isotonic) jointly with a threshold, optimizing for bootstrap-mean F1 with group-aware resampling. The `thresholding.py` module searches a dense grid of candidate thresholds (quantile-based + fixed grid), applies optional precision/FPR constraints, and selects from a 99%-of-best plateau using tie-breakers on stability, FPR, and precision.

**Key files**:
- `common.py`: `OfficialPaths`, `load_clean_table()`, `encode_frame()`, `feature_output_path()`, `RANDOM_SEED=42`.
- `graph_dataset.py`: `TransductiveGraph` dataclass — builds the full user-user graph with typed weighted edges and per-entity node features. Used by both the GNN and transductive feature modules.
- `transductive_features.py`: Per-fold label-aware features — BFS distances, 1-hop/2-hop propagation, PPR (alpha=0.20, 20 iterations), per-edge-type positive-neighbor counts, entity-level seed aggregates, component-level seed statistics.
- `transductive_validation.py`: `PrimarySplitSpec` and `build_primary_transductive_splits()` / `build_secondary_strict_splits()`.
- `splitters.py`: `UnionFind`-based group construction, `StratifiedGroupKFold` fold assignment with multi-seed search ensuring `min_positive_per_fold=250`.
- `rules.py`: 6 deterministic rules (fast_cashout_24h, shared_ip_ring, shared_wallet_ring, high_relation_fanout, night_trade_burst, market_order_burst) → `rule_score` (fraction of triggered rules).
- `modeling.py`: `fit_catboost()` and `fit_lgbm()` wrappers with auto-balanced class weights.
- `bundle.py`: `official_bundle.json` persistence — stores all model paths, feature columns, calibrator, threshold, and validation metadata.

**Latest results** (from `OFFICIAL_EXPERIMENT_SUMMARY_20260317.md`): Primary F1=0.363 at threshold=0.1492, isotonic calibration, AP=0.284. Secondary group-stress F1=0.351.

### Transductive V1 Pipeline: `bitoguard_core/transductive_v1/`

A competition-oriented MVP pipeline separate from `official/`. Artifacts write to `bitoguard_core/artifacts/transductive_v1/`. Entry point: `transductive_v1/pipeline.py`.

**Key differences from `official/`**:
- No GNN (Base C / GraphSAGE) — uses only 2 CatBoost branches (Base A label-free, Base B label-aware).
- Graph features come from a `GraphStore` that builds projected user-user edges (relation, wallet, IP) with typed weights and structural features (component sizes, degree centrality, shared-user counts) but does not train a GNN.
- Label-aware features use direct BFS distances, hop counts, PPR (6 iterations), entity reputation features, and a `graph_risk_score` composite.
- Stacker includes 7 candidate features: `[base_a_prob, base_b_prob, graph_risk_score, rule_score, anomaly_score, projected_component_log_size, connected_flag]`.
- Calibration jointly selects both a calibrator and a "decision rule" (which may be a threshold or other rule type) via `decision_rule.py`.
- Training uses subprocess-per-fold isolation (`fold_worker.py`) for memory management.
- Secondary validation uses `secondary_validation.py` with `StratifiedGroupKFold`.
- Scoring blends probabilities as `0.76*submission + 0.14*anomaly + 0.10*rule` (vs official's `0.72/0.16/0.12`).

### Mock API: `bitoguard_mock_api/`

Read-only FastAPI server that serves simulator-generated pseudo data through `/v1/` list endpoints. Loads CSV files from `bitoguard_sim_output/` (configurable via `BITOGUARD_DATA_DIR`). Provides 11 paginated endpoints: users, login-events, fiat-transactions, trade-orders, crypto-transactions, known-blacklist-users, devices, user-device-links, bank-accounts, user-bank-links, crypto-wallets. Supports `start_time`/`end_time` filtering and standard pagination. Run with `uvicorn app.main:app --port 8000`.

### Simulator: `bitoguard_simulator/`

Transaction simulator (`bitoguard_transaction_simulator.py`) that generates synthetic exchange data with injected AML scenarios (mule_quick_out, fan_in_hub, shared_device_ring, blacklist_2hop_chain). Produces 14 CSV files including `entity_edges.csv` for graph models. Run with `python bitoguard_simulator/bitoguard_transaction_simulator.py --n-users 1200 --days 30 --seed 42 --output-dir bitoguard_sim_output`.

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

**Note**: The `official/` and `transductive_v1/` pipelines use their own graph feature implementations that handle the trust boundary differently — they cap entity fan-out at configurable thresholds (`MAX_IP_ENTITY_USERS=200`, `MAX_WALLET_ENTITY_USERS=200` in `official/graph_features.py`) rather than using the binary trusted-only flag.

### Key Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `BITOGUARD_DB_PATH` | `bitoguard_core/artifacts/bitoguard.duckdb` | DuckDB file |
| `BITOGUARD_ARTIFACT_DIR` | `bitoguard_core/artifacts/` | Model + report outputs |
| `BITOGUARD_SOURCE_URL` | `https://aws-event-api.bitopro.com` | Data source |
| `BITOGUARD_LABEL_SOURCE` | `hidden_suspicious_label` | Oracle label column |
| `BITOGUARD_GRAPH_FEATURES_TRUSTED_ONLY` | `true` | Disable unsafe graph features |
| `BITOGUARD_CORS_ORIGINS` | `http://localhost:3000` | Comma-separated CORS origins |
| `BITOGUARD_API_KEY` | (unset) | API key for X-API-Key header auth; unset = auth disabled |
| `BITOGUARD_DATA_DIR` | `bitoguard_sim_output/` | Mock API data directory |
| `BITOGUARD_MODEL_BACKEND` | `legacy` | Model backend for scoring: `"legacy"` (DuckDB stacker) or `"official"` (pre-computed official pipeline scores) |

### Artifacts

Stored in `bitoguard_core/artifacts/`:
- `bitoguard.duckdb` — all data (gitignored)
- `models/lgbm_*.lgbm` + `lgbm_*.json` — LightGBM model + metadata
- `models/iforest_*.joblib` + `iforest_*.sha256` — IsolationForest model + integrity manifest
- `validation_report.json` — latest holdout evaluation (P@K, calibration, feature importance)
- `official_bundle.json` — official pipeline model bundle (paths, feature columns, calibrator, threshold)
- `official_features/` — parquet feature snapshots, OOF predictions, split assignments
- `models/official_catboost_base_a_*.pkl` / `official_catboost_base_b_*.pkl` — official CatBoost models
- `models/official_graphsage_*.pt` — official GraphSAGE model (PyTorch)
- `models/official_stacker_*.pkl` — official LR stacker
- `reports/official_validation_report.json` — official primary + secondary validation
- `predictions/official_predict_scores.{parquet,csv}` — final scored predictions
- `transductive_v1/` — separate artifact tree (features/, models/, reports/, predictions/, bundle.json)

### Test Suite Structure

```
tests/
├── test_rule_engine.py           # 33 tests — all 11 AML rules, trigger/no-trigger cases
├── test_model_pipeline.py        # 15 tests — temporal splits, refresh, drift
├── test_source_integration.py    # 6 tests  — canonicalization, sync lifecycle
├── test_smoke.py                 # 5 tests  — API smoke, alert/case lifecycle
├── test_graph_data_quality.py    # Graph data quality checks (placeholder device ID detection)
├── test_stacker.py               # Stacker training + OOF shape/leakage checks
├── test_feature_modules.py       # Per-module unit tests for all 8 v2 feature modules
├── test_graph_bipartite.py       # Bipartite graph feature tests
├── test_store.py                 # DuckDBStore read/write tests
├── test_anomaly_features.py      # Anomaly feature pipeline tests
├── test_iforest_ablation.py      # IsolationForest ablation tests
├── test_official_pipeline.py     # Official pipeline integration tests
└── test_transductive_v1_pipeline.py  # Transductive v1 pipeline tests
```

Tests run against fixture data (no live API calls required by default).
