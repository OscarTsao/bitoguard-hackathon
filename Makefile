# BitoGuard AML System — Task Runner
#
# Usage:
#   make help              Show this help
#   make setup             Create Python venv and install dependencies
#   make test              Run full test suite (85 tests)
#   make sync              Full data sync from live BitoPro API
#   make features          Rebuild all feature snapshots
#   make train             Train LightGBM + IsolationForest models
#   make evaluate          Run temporal holdout evaluation
#   make refresh           Incremental live refresh (watermark-based)
#   make score             Score latest snapshot + generate alerts
#   make serve             Start backend API (port 8001)
#   make frontend          Start Next.js frontend (port 3000)
#   make docker-up         Start full stack via Docker Compose
#   make docker-build      Build Docker images
#   make drift             Run feature drift detection

SHELL      := bash
PYTHON     := .venv/bin/python
ACTIVATE   := source .venv/bin/activate
CORE_DIR   := bitoguard_core

.PHONY: help setup test test-quick test-rules sync features train evaluate refresh score features-v2 train-stacker score-v2 serve frontend docker-up docker-build docker-down drift lint clean

help:
	@echo ""
	@echo "BitoGuard AML System — Make targets"
	@echo "===================================="
	@awk '/^[a-zA-Z_-]+:.*?## / { printf "  %-18s %s\n", $$1, substr($$0, index($$0, "## ")+3) }' $(MAKEFILE_LIST)
	@echo ""

# ── Environment ──────────────────────────────────────────────────────────────

setup: ## Create .venv and install Python dependencies
	cd $(CORE_DIR) && python -m venv .venv && \
	$(ACTIVATE) && pip install -r requirements.txt
	@echo "Setup complete. Activate with: source bitoguard_core/.venv/bin/activate"

# ── Tests ─────────────────────────────────────────────────────────────────────

test: ## Run full test suite (85 tests)
	cd $(CORE_DIR) && $(ACTIVATE) && PYTHONPATH=. python -m pytest tests/ -v

test-quick: ## Run tests in quiet mode
	cd $(CORE_DIR) && $(ACTIVATE) && PYTHONPATH=. python -m pytest tests/ -q

test-rules: ## Run rule engine unit tests only
	cd $(CORE_DIR) && $(ACTIVATE) && PYTHONPATH=. python -m pytest tests/test_rule_engine.py -v

# ── Data Pipeline ─────────────────────────────────────────────────────────────

sync: ## Full data sync from live BitoPro AWS Event API
	cd $(CORE_DIR) && $(ACTIVATE) && PYTHONPATH=. python pipeline/sync.py --full

features: ## Rebuild all feature snapshots (graph + tabular)
	cd $(CORE_DIR) && $(ACTIVATE) && \
	PYTHONPATH=. python features/graph_features.py && \
	PYTHONPATH=. python features/build_features.py

refresh: ## Incremental live refresh (watermark-based, bounded)
	cd $(CORE_DIR) && $(ACTIVATE) && PYTHONPATH=. python pipeline/refresh_live.py

# ── Model Training & Evaluation ───────────────────────────────────────────────

train: ## Train LightGBM classifier + IsolationForest anomaly model
	cd $(CORE_DIR) && $(ACTIVATE) && \
	PYTHONPATH=. python models/train.py && \
	PYTHONPATH=. python -c "from models.anomaly import train_anomaly_model; import json; print(json.dumps(train_anomaly_model()))"

evaluate: ## Run temporal holdout evaluation, save report
	cd $(CORE_DIR) && $(ACTIVATE) && PYTHONPATH=. python models/validate.py

score: ## Score latest snapshot + generate alerts
	cd $(CORE_DIR) && $(ACTIVATE) && PYTHONPATH=. python models/score.py

features-v2: ## Build v2 feature snapshots (~155 columns per user)
	cd $(CORE_DIR) && $(ACTIVATE) && PYTHONPATH=. python features/build_features_v2.py

train-stacker: ## Train CatBoost + LightGBM branches + LR stacker on v2 features
	cd $(CORE_DIR) && $(ACTIVATE) && PYTHONPATH=. python models/stacker.py

score-v2: ## Score latest snapshot using stacker (v2 features)
	cd $(CORE_DIR) && $(ACTIVATE) && PYTHONPATH=. python -c \
	    "from models.score import score_latest_snapshot_v2; r = score_latest_snapshot_v2(); print(f'Scored {len(r)} users')"

drift: ## Run feature drift detection between two most recent snapshots
	cd $(CORE_DIR) && $(ACTIVATE) && PYTHONPATH=. python services/drift.py

# ── API / Frontend ────────────────────────────────────────────────────────────

serve: ## Start FastAPI backend (http://localhost:8001)
	cd $(CORE_DIR) && $(ACTIVATE) && \
	PYTHONPATH=. uvicorn api.main:app --reload --port 8001

frontend: ## Start Next.js frontend (http://localhost:3000)
	cd bitoguard_frontend && npm run dev

# ── Docker ────────────────────────────────────────────────────────────────────

docker-build: ## Build backend + frontend Docker images
	docker compose build

docker-up: ## Start full stack (backend + frontend) via Docker Compose
	docker compose up --build

docker-down: ## Stop Docker Compose stack
	docker compose down

# ── Maintenance ───────────────────────────────────────────────────────────────

lint: ## Run ruff linter on bitoguard_core (if available)
	cd $(CORE_DIR) && $(ACTIVATE) && \
	python -m ruff check . --ignore E501 2>/dev/null || echo "ruff not installed; run: pip install ruff"

clean: ## Remove Python __pycache__ and .pyc files
	find $(CORE_DIR) -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find $(CORE_DIR) -name "*.pyc" -delete 2>/dev/null || true
