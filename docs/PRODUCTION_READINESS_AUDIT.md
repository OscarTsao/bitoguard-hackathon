# Production Readiness Audit

**Audit date:** 2026-03-14
**Auditor:** Claude Opus 4.6 (automated evidence-based audit)
**Scope:** Full system audit of BitoGuard AML detection pipeline

---

## Final Verdict: DEPLOYMENT_PREPARED

BitoGuard is a complete, well-engineered AML screening system with honest documentation, 78/78 passing tests, and fully prepared deployment artifacts. The system works locally end-to-end. No actual cloud deployment has been executed.

**The system is honest about what it can and cannot do.** This is its strongest quality attribute.

---

## Audit Methodology

This audit was conducted by:
1. Reading every source file in the production codepaths (40+ files)
2. Reading all 18 documentation files
3. Executing the full test suite (78/78 pass)
4. Scanning for hardcoded secrets (none found)
5. Cross-referencing documentation claims against code implementation
6. Verifying infrastructure artifacts (Dockerfiles, task definitions, CI workflows, deployment scripts)

No production code was modified. All commands are logged in `reports/COMMANDS_RUN.log`.

---

## System Architecture Summary

| Layer | Implementation | Status |
|-------|---------------|--------|
| **Data Ingestion** | `source_client.py` + `pipeline/transformers.py` | HTTP-only, PostgREST auto-detect, paginated, 1e8 scaling |
| **Normalization** | `pipeline/normalize.py` | Raw-to-canonical dedup, UTC coercion, quality issue logging |
| **Feature Engineering** | `features/build_features.py` | 30+ features across 1d/7d/30d windows with peer-deviation |
| **Graph Features** | `features/graph_features.py` | NetworkX, trusted_only=True default, unsafe features zeroed |
| **Supervised Model** | `models/train.py` (LightGBM) | Temporal 70/15/15 split, leakage guard |
| **Anomaly Model** | `models/anomaly.py` (IsolationForest) | Unsupervised, contamination calibrated to label prevalence |
| **Rule Engine** | `models/rule_engine.py` | 11 rules, severity-weighted composite |
| **Scoring** | `models/score.py` | Composite: 35% rules + 45% LightGBM + 10% anomaly + 10% graph |
| **Explainability** | `services/explain.py` + `services/diagnosis.py` | SHAP TreeExplainer, Chinese-language factor names |
| **Ops** | `pipeline/refresh_live.py`, `services/drift.py`, `services/alert_engine.py` | Watermark refresh, drift detection, alert lifecycle |
| **API** | `api/main.py` | FastAPI, 13 endpoints, CORS configurable |
| **Database** | `db/store.py` (DuckDB) | Embedded, 4 schemas (raw, canonical, features, ops) |
| **Packaging** | Dockerfile, compose.yaml | Python 3.12-slim, health checks |
| **CI/CD** | `.github/workflows/ci.yml` | Test + lint + docker build + manual AWS deploy |
| **AWS Infra** | `infra/aws/`, `scripts/` | 3 ECS task defs, IAM policies, deploy scripts |

---

## Module-Level Assessment

### M0: Dormancy Baseline (VALID - Primary Signal)
- PR-AUC: 0.9823, ROC-AUC: 0.9882
- A trivial `sum(behavioral_features) == 0` rule
- Correctly identified as the honest primary baseline in documentation
- Every other module is honestly compared against this baseline

### M1: Rule Engine (IMPLEMENTED - 0% trigger rate on current data)
- 11 rules implemented with severity weighting
- Graph-dependent rules (3 of 11) always return False under trusted_only=True
- Behavioral rules cannot fire on dormant users (the entire labeled population)
- Rules would function correctly on active fraudulent users if such data existed

### M2: Statistical Features (IMPLEMENTED - Valid computation, confounded evaluation)
- Peer-deviation percentile ranks correctly computed within KYC cohorts
- 30-day rolling window features for fiat, trade, crypto, login activity
- Velocity features (fiat-to-crypto-out timing windows)
- Evaluation metrics invalidated by dormancy confound (A1+A2+A3)

### M3: LightGBM (IMPLEMENTED - Effectively a dormancy detector)
- Temporal split correctly implemented via `forward_date_splits()`
- Leakage guard via `positive_effective_date` in `training_dataset()` SQL
- Holdout: P=0.9984, R=1.0, F1=0.9992 (validated against `validation_report.json`)
- Top features: `fiat_in_30d_peer_pct` (87.3%), `fiat_out_30d_peer_pct` (12.6%)
- The model detects dormancy via peer-percentile features, not behavioral fraud

### M4: IsolationForest (IMPLEMENTED - Valid but below dormancy baseline)
- PR-AUC: 0.9724 (below M0's 0.9823)
- Trained on training split only (unsupervised)
- Valid non-artifact signal: identifies dormant users as anomalous
- Useful for ranking non-dormant users

### M5: Graph Features (QUARANTINED - 4/4 audit checks FAIL)
- `graph_trusted_only=True` by default
- Unsafe features zeroed: shared_device_count, component_size, blacklist_1hop/2hop
- Trusted features active but near-zero signal: fan_out_ratio, shared_wallet_count, shared_bank_count
- Placeholder device super-node (MD5("0")) affects 78% of users
- G2 ROC-AUC=0.40, G3 ROC-AUC=0.40, G4 untested, G6 marginal gain=-0.60

### M6: Operations (IMPLEMENTED - Fully functional)
- **Incremental refresh:** Watermark-based, no-op on unchanged data, targeted rebuild
- **SHAP explanations:** Top-5 factors per user with Chinese names
- **Drift detection:** Zero-rate, mean shift, std shift across consecutive snapshots
- **Alert engine:** Auto-generates alerts for high/critical, case management with 4 decisions
- **Risk diagnosis:** Full case report with summary, SHAP, rules, graph, timeline, action

---

## Key Findings

### Strengths

1. **Exceptional documentation honesty.** The system explicitly documents what it cannot do, lists data artifacts, marks modules as FAIL/QUARANTINED, and provides a dormancy baseline that every module is compared against. This is rare and valuable.

2. **Correct engineering practices.** Temporal splits, leakage guards, feature encoding with reference columns, unsupervised anomaly detection, SHAP explainability, watermark-based incremental refresh, stage-level logging.

3. **Comprehensive test suite.** 78 tests across 5 files covering rules, model pipeline, graph quality, source integration, and API smoke tests.

4. **Safe production defaults.** Graph features quarantined, CORS restricted, no secrets in code.

5. **Complete deployment preparation.** Dockerfiles, compose, CI/CD, AWS task definitions, deployment scripts, runbooks.

### Weaknesses

1. **Detection capability is limited to dormancy screening.** The model cannot claim behavioral fraud detection with the current dataset.

2. **Graph module is non-functional.** 10% of the composite score weight is allocated to features that contribute zero signal.

3. **No API authentication.** Production deployment without auth is unsafe.

4. **DuckDB single-writer limitation.** Acceptable at current scale but could cause issues with concurrent operations.

5. **Test count discrepancy in docs.** README says 61, Makefile says 85, actual is 78.

---

## Risk Assessment

| Risk | Severity | Mitigation |
|------|----------|------------|
| Overclaiming model capability | HIGH | Fully mitigated by honest documentation |
| Graph feature contamination | HIGH | Mitigated by trusted_only=True default |
| Unauthenticated API access | MEDIUM | Not mitigated (requires implementation) |
| DuckDB concurrent write conflicts | LOW | Acceptable at current scale |
| AWS deployment untested | MEDIUM | All artifacts prepared; requires execution |

---

## Supporting Evidence

| Report | Path |
|--------|------|
| Verification Matrix | `reports/VERIFICATION_MATRIX.csv` |
| Module Status Matrix | `reports/MODULE_STATUS_MATRIX.csv` |
| Production Gap List | `reports/PROD_GAP_LIST.md` |
| Claims Audit | `reports/CLAIMS_AUDIT.md` |
| Local Validation Evidence | `reports/LOCAL_VALIDATION_EVIDENCE.md` |
| AWS Deployment Preparedness | `reports/DEPLOYMENT_PREPAREDNESS_AWS.md` |
| Commands Run | `reports/COMMANDS_RUN.log` |
| Release Verdict | `docs/RELEASE_VERDICT.md` |
