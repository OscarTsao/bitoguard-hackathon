# Release Verdict

**Audit date:** 2026-03-14
**Auditor:** Claude Opus 4.6

---

## 1. FINAL STATUS

**DEPLOYMENT_PREPARED**

---

## 2. ONE-SENTENCE VERDICT

BitoGuard is a well-architected, honestly documented AML screening system that works locally with 78/78 passing tests and complete deployment scaffolding, but its detection capability is limited to contemporaneous dormancy screening (not behavioral fraud prediction), and no actual cloud deployment has been executed.

---

## 3. WHAT IS VERIFIED

- **All 6 modules exist as implemented, runnable code** with appropriate tests (78/78 pass).
- **Source ingestion** correctly handles PostgREST auto-detection, pagination, amount scaling (1e8 fixed-point), and UTC normalization.
- **Feature engineering** computes 30+ behavioral features across 1d/7d/30d windows with peer-deviation percentile ranks within KYC cohorts.
- **LightGBM supervised model** uses correct temporal 70/15/15 forward splits with positive_effective_date leakage guards.
- **IsolationForest anomaly model** trains on training split only with label-prevalence-calibrated contamination.
- **Graph features** are correctly quarantined: `graph_trusted_only=True` by default zeroes unsafe features (shared_device_count, component_size, blacklist_1hop/2hop).
- **Incremental refresh** implements watermark state, no-op on unchanged data, targeted user rebuild, and stage-level logging instrumentation.
- **SHAP explanations** produce per-user top-5 factor breakdowns with Chinese-language feature names, rule hits, graph evidence, and activity timeline.
- **Alert engine** generates alerts for high/critical risk users with case management and 4 decision types.
- **Drift detection** compares consecutive snapshots using zero-rate, mean shift, and std shift metrics.
- **13 API endpoints** are wired and functional.
- **Docker packaging** (Dockerfile + compose.yaml) exists with health checks.
- **CI/CD** (GitHub Actions) covers tests, linting, Docker build, and manual AWS deploy.
- **AWS infrastructure scaffolding** (3 ECS task definitions, IAM policies, deployment scripts) is complete.
- **No hardcoded secrets** in the codebase.
- **Documentation is honest** about model limitations, data artifacts, and what the system cannot claim.

---

## 4. WHAT IS NOT VERIFIED

- **Docker image build** was not executed during this audit (syntactically valid but not tested).
- **AWS deployment** has never been executed (placeholder values remain in infrastructure templates).
- **Frontend functionality** was not tested (build artifacts present but UI not exercised).
- **Live API performance benchmarks** claimed in RUNBOOK_AWS.md were not independently verified.
- **Mock API** (`bitoguard_mock_api/`) was not inspected or tested.

---

## 5. WHAT IS STILL BLOCKED

- **No actual cloud deployment.** Maximum possible status is DEPLOYMENT_PREPARED until at least one target-environment deployment validation is executed.
- **Graph feature recovery** is blocked on the data provider supplying non-null device fingerprints. Until then, M5 graph features contribute effectively zero signal.
- **Behavioral fraud detection** is blocked on obtaining a dataset with behaviorally active fraudulent users. The current labeled population is entirely dormant.
- **API authentication** is not implemented. Production exposure without authentication is unsafe.

---

## 6. MODEL QUALITY CAVEATS THAT MUST REMAIN EXPLICIT

These caveats must be prominently displayed in any production interface, compliance report, or performance claim:

1. **The model detects dormancy, not behavioral fraud.** The top features (fiat_in_30d_peer_pct at 87.3%, fiat_out_30d_peer_pct at 12.6%) measure peer-relative inactivity. Near-perfect metrics (P=0.9984, R=1.0) reflect that 100% of blacklisted users have zero behavioral activity, making them trivially separable from active users.

2. **The dormancy baseline (M0: sum(behavioral_features)==0) achieves PR-AUC=0.9823** without any ML model. Any module that cannot beat this baseline is not adding detection signal beyond a simple zero-check.

3. **This is contemporaneous screening, not predictive detection.** The system identifies users who are currently dormant and match the blacklist profile at snapshot time. It does not predict future fraud, estimate time-to-blacklist, or provide early warning.

4. **M5 graph features are quarantined.** The placeholder device super-node (MD5("0")) connects 78% of users into a false cluster. All graph topology features are invalid until the Graph Recovery Plan is completed. Audit checks G2/G3/G4/G6 all FAIL.

5. **M1 rules have 0% trigger rate on the labeled population** because all blacklisted users are dormant and rules require active behavioral signals.

6. **M3 LightGBM results are driven by the same dormancy artifact.** The model has correctly learned to detect dormancy but should not be presented as detecting behavioral money laundering patterns.

---

## 7. WHAT MUST BE TRUE BEFORE THIS CAN BE CALLED FULLY_PRODUCTION_READY

All of the following conditions must be satisfied:

1. **At least one actual AWS (or equivalent cloud) deployment must be executed and validated** with health check, full sync, feature build, training, scoring, and refresh cycle completing successfully.

2. **API authentication must be implemented** before any production network exposure.

3. **CORS origins must be configured** for the actual production frontend domain.

4. **Graph Recovery Plan must be completed** (or M5 weight must be explicitly set to 0 rather than 10%).

5. **Monitoring and alerting infrastructure** must be operational (CloudWatch alarms, not just log groups).

6. **DuckDB concurrency strategy** must be validated under production load patterns (refresh task vs. API reads).

7. **All placeholder values** in `infra/aws/task-def-*.json` must be replaced with actual AWS resource identifiers.

---

## 8. EXACT COMMANDS RUN

See `reports/COMMANDS_RUN.log` for the complete list. Key commands:

```bash
# Test execution
cd /home/oscartsao/Developer/bitoguard-hackathon/bitoguard_core
source .venv/bin/activate
PYTHONPATH=. python -m pytest tests/ -q --tb=short
# Result: 78 passed in 27.90s

# Test enumeration
PYTHONPATH=. python -m pytest tests/ -v --co
# Result: 78 tests collected

# Repository structure
find /home/oscartsao/Developer/bitoguard-hackathon -maxdepth 3 -not -path "*/__pycache__/*" -not -path "*/.venv/*" -not -path "*/node_modules/*" -not -path "*/.git/*" | sort

# Secrets scan
grep -ri '(api_key|secret_key|password|token|bearer)' across all source files
# Result: No matches

# Model artifacts
ls -la bitoguard_core/artifacts/models/
# Result: 8 model files (4 LightGBM + 4 IsolationForest across 3 training runs)

# File reads: 40+ source files inspected across all modules
```
