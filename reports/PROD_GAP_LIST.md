# Production Gap List

**Audit date:** 2026-03-14
**Auditor:** Claude Opus 4.6

This document lists all gaps between the current system state and full production readiness, ordered by severity.

---

## BLOCKER Gaps (must resolve before any production deployment)

**None identified.** The system is deployment-prepared with safe defaults.

---

## HIGH Priority Gaps (should resolve before relying on system for compliance decisions)

### GAP-H1: Model detects dormancy, not behavioral fraud

**Evidence:** `validation_report.json` shows top features `fiat_in_30d_peer_pct` (87.3% gain) and `fiat_out_30d_peer_pct` (12.6% gain). Both are peer-percentile-rank features where zero activity = bottom percentile. 100% of blacklisted users are dormant.

**Impact:** The system cannot claim to detect behavioral fraud patterns, money laundering typologies, or provide early warning of suspicious activity. It detects dormant accounts that happen to correlate with the blacklist.

**Mitigation:** Documented in `docs/DORMANCY_BASELINE.md`, `docs/MODEL_CARD.md`, and `docs/RELEASE_READINESS_CHECKLIST.md`. The "Cannot Do" section explicitly lists this limitation.

**Resolution path:** Requires a dataset with behaviorally active fraudulent users to train and validate genuine behavioral detection.

---

### GAP-H2: Graph features are largely non-functional

**Evidence:** `config.py` sets `graph_trusted_only=True` by default. Under this setting, `shared_device_count`, `component_size`, `blacklist_1hop_count`, and `blacklist_2hop_count` are all zeroed. The remaining trusted features (`fan_out_ratio`, `shared_wallet_count`, `shared_bank_count`) have near-zero informative value in the current dataset (all wallets unique, no bank data).

**Impact:** M5 contributes effectively 0% signal to the composite risk score despite having a 10% weight allocation. Three M1 rules (shared_device_ring, blacklist_1hop, blacklist_2hop) can never trigger.

**Mitigation:** Correctly quarantined. `docs/GRAPH_RECOVERY_PLAN.md` provides a 5-step recovery plan requiring clean device IDs from the data provider.

**Resolution path:** Data provider must supply non-null, non-placeholder device fingerprints.

---

### GAP-H3: Rule engine has 0% trigger rate on labeled population

**Evidence:** `docs/RELEASE_READINESS_CHECKLIST.md` marks M1 as "FAIL - DO NOT DEPLOY" because all blacklisted users are dormant and rules require active behavioral signals to fire.

**Impact:** The 35% weight allocated to rule_score in the composite risk score contributes nothing for the known-bad population. Rules would fire on active users exhibiting suspicious patterns, but no such users exist in the labeled data to validate this.

**Mitigation:** Rules are correctly implemented and would work on active fraud. The weight allocation means non-flagged dormant users still get high composite scores from M3.

**Resolution path:** Validate rules against active fraud cases when available.

---

## MEDIUM Priority Gaps

### GAP-M1: No actual AWS deployment has been executed

**Evidence:** `infra/aws/task-def-backend.json` contains placeholder values (`<ACCOUNT_ID>`, `<REGION>`, `<EFS_FILE_SYSTEM_ID>`). No deployment logs exist. CI deploy-aws job requires manual trigger and has never been executed (github.event_name == 'workflow_dispatch').

**Impact:** Cannot verify that the deployment artifacts actually work in a real AWS environment. EFS mounting, service discovery, health checks, and scheduler timing are all untested.

**Resolution path:** Execute `scripts/deploy_aws.sh` with real AWS credentials. Verify health check, refresh cycle, and scoring end-to-end.

---

### GAP-M2: DuckDB single-writer limitation

**Evidence:** `db/store.py` opens a new DuckDB connection per operation. DuckDB supports only one concurrent writer.

**Impact:** In production, concurrent API requests that write (sync, score, refresh) will block each other. The EventBridge-scheduled refresh_live task and the backend API service share the same EFS-mounted DuckDB file.

**Resolution path:** Acceptable for the current scale (~60K users, single-task refresh). For scale-up, consider PostgreSQL or a write-ahead-log wrapper.

---

### GAP-M3: No authentication/authorization on API

**Evidence:** `api/main.py` has no authentication middleware. All endpoints are publicly accessible within the network.

**Impact:** In production, anyone with network access to the ALB can trigger sync, retrain, score, and apply case decisions.

**Resolution path:** Add API key or JWT authentication before production use. The AWS runbook mentions "Backend is not publicly accessible - only through the ALB" but ALB access is still unauthenticated.

---

### GAP-M4: CORS configured for localhost only by default

**Evidence:** `config.py` defaults `BITOGUARD_CORS_ORIGINS` to `http://localhost:3000`.

**Impact:** Production deployment must update CORS origins to the actual frontend domain. Current default is safe (restrictive) but non-functional in production.

**Resolution path:** Set `BITOGUARD_CORS_ORIGINS` in production environment config.

---

## LOW Priority Gaps

### GAP-L1: Test count discrepancy in docs

**Evidence:** README.md claims "61 tests", RUNBOOK_LOCAL.md claims "61 tests", Makefile claims "85 tests", CI workflow claims "85 tests". Actual count: 78 tests.

**Impact:** Cosmetic inconsistency. The test suite has grown but docs were not all updated.

**Resolution path:** Update all docs to reflect 78 tests.

---

### GAP-L2: No model versioning governance

**Evidence:** Model artifacts are saved with timestamp versions (e.g., `lgbm_20260313T090425Z.pkl`). Old artifacts persist in `artifacts/models/`. No automated cleanup or promotion workflow.

**Impact:** Multiple old model versions accumulate on disk. No clear rollback mechanism.

**Resolution path:** Add artifact retention policy or model registry.

---

### GAP-L3: SHAP version compatibility warning

**Evidence:** `docs/EVALUATION_REPORT.md` notes "SHAP version sensitivity: SHAP explanation requires a compatible sklearn version."

**Impact:** Model trained on one sklearn version may produce warnings when explained on another. Not a correctness issue.

**Resolution path:** Pin sklearn version in requirements.txt (already implicitly done via requirements.txt).

---

### GAP-L4: Missing bank account data

**Evidence:** `pipeline/transformers.py` returns `bank_accounts: []` and `user_bank_links: []`. `docs/EVALUATION_REPORT.md` confirms "canonical.bank_accounts and canonical.user_bank_links are empty."

**Impact:** `shared_bank_count` is always 0 for all users.

**Resolution path:** Requires data provider to expose bank identifiers in upstream API.
