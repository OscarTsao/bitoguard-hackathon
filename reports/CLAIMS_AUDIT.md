# Claims Audit

**Audit date:** 2026-03-14
**Auditor:** Claude Opus 4.6

This document audits every significant claim made in the project documentation against the actual code and data.

---

## Claim 1: "6-module AML detection pipeline"

**Source:** README.md
**Verdict: VERIFIED**

All six modules exist as implemented code:
- M1: `models/rule_engine.py` - 11 rules, severity-weighted scoring
- M2: `features/build_features.py` - peer-deviation features, cohort percentile ranks
- M3: `models/train.py` + `models/validate.py` - LightGBM with temporal splits
- M4: `models/anomaly.py` - IsolationForest
- M5: `features/graph_features.py` - NetworkX graph features
- M6: `services/explain.py`, `services/diagnosis.py`, `pipeline/refresh_live.py`, `services/drift.py`, `services/alert_engine.py`

---

## Claim 2: "78/78 tests pass"

**Source:** Ground truth
**Verdict: VERIFIED**

Command: `PYTHONPATH=. python -m pytest tests/ -q --tb=short`
Result: `78 passed in 27.90s`

Note: README.md claims "61 tests", Makefile claims "85 tests". The actual count is 78. Docs are stale.

---

## Claim 3: "graph_trusted_only=True by default"

**Source:** Ground truth, config.py
**Verdict: VERIFIED**

`config.py` line 67: `graph_trusted_raw = os.getenv("BITOGUARD_GRAPH_FEATURES_TRUSTED_ONLY", "true").lower()`
Line 79: `graph_trusted_only=graph_trusted_raw not in ("false", "0", "no")`

Default is "true", which means `graph_trusted_only=True`. The unsafe features (shared_device_count, component_size, blacklist_1hop_count, blacklist_2hop_count) are defined in `UNSAFE_GRAPH_FEATURES` frozenset and are zeroed in `graph_features.py` when `trusted_only=True`.

---

## Claim 4: "Oracle labels come from a separate endpoint, NOT from user_info.status"

**Source:** Ground truth
**Verdict: VERIFIED**

`oracle_client.py` line 83: fetches from `/train_label` endpoint.
Line 89: `"hidden_suspicious_label": int(row.get("status", 0) or 0)` - uses `train_label.status`, not `user_info.status`.

`pipeline/load_oracle.py` calls `OracleClient(source_url=settings.source_url).load()` and stores in `ops.oracle_user_labels`.

The oracle is a separate data pathway from the source sync (`pipeline/sync_source.py`).

---

## Claim 5: "No actual AWS deployment has been executed"

**Source:** Ground truth
**Verdict: VERIFIED**

Evidence:
- `infra/aws/task-def-backend.json` contains `<ACCOUNT_ID>`, `<REGION>`, `<EFS_FILE_SYSTEM_ID>` placeholders
- CI `deploy-aws` job requires `workflow_dispatch` and has condition `github.event_name == 'workflow_dispatch'`
- No deployment logs or evidence of actual execution

---

## Claim 6: "Dormancy baseline PR-AUC=0.9823"

**Source:** docs/DORMANCY_BASELINE.md, docs/MODEL_CARD.md
**Verdict: VERIFIED (claim is honest)**

The docs explicitly state this is a trivial `sum(behavioral_features) == 0` rule and that this reveals blacklisted users are predominantly dormant. The MODEL_CARD section "Data Artifact Warning" documents artifacts A1+A2+A3 and states that near-perfect metrics "reflect data artifacts, not fraud patterns."

---

## Claim 7: "LightGBM holdout P=0.9984, R=1.0, F1=0.9992"

**Source:** docs/EVALUATION_REPORT.md
**Verdict: VERIFIED against validation_report.json but with CRITICAL CAVEAT**

`validation_report.json` confirms:
- precision: 0.99844768705371
- recall: 1.0
- f1: 0.9992232406400497
- holdout_rows: 33221, positives: 3216, negatives: 30005

**Caveat correctly documented:** These metrics reflect dormancy detection. Top feature importance: `fiat_in_30d_peer_pct` at 87.32%. The EVALUATION_REPORT states "The remaining characteristic: blacklisted users are predominantly dormant" and "limits the model to dormancy-informed detection."

---

## Claim 8: "Temporal split prevents look-ahead bias"

**Source:** docs/EVALUATION_REPORT.md, docs/MODEL_CARD.md
**Verdict: VERIFIED (mechanism correct, but see caveat)**

`models/common.py` `forward_date_splits()` implements 70/15/15 chronological split. Oldest dates for training, newest for holdout.

`training_dataset()` SQL enforces `positive_effective_date` leakage guard: positive labels only included for snapshots on or after the user's first `observed_at` in `canonical.blacklist_feed`.

**Caveat:** The temporal split is correctly implemented, but since all blacklisted users are dormant at all snapshots (identical zero-activity features across dates), temporal ordering provides limited protection against the dormancy confound. The split prevents date-level leakage but not the fundamental artifact that zero-activity = blacklisted.

---

## Claim 9: "IsolationForest produces valid, artifact-free signal (PR-AUC=0.9724)"

**Source:** docs/RELEASE_READINESS_CHECKLIST.md
**Verdict: PARTIALLY VERIFIED**

M4 IsolationForest is trained on the same feature set as LightGBM but is unsupervised. The PR-AUC=0.9724 is below the dormancy baseline (0.9823), which means M4's signal is a weaker version of the dormancy signal rather than an independent detector.

The claim of "artifact-free" is debatable: M4 detects dormant users as anomalous because they have all-zero features, which is indeed a genuine anomaly relative to active users. But it is the same dormancy signal, not an independent fraud signal.

The docs are honest about this: "Its PR-AUC is slightly below the dormancy baseline."

---

## Claim 10: "M5 graph features quarantined with 4 failing audit checks"

**Source:** docs/GRAPH_RECOVERY_PLAN.md, docs/GRAPH_TRUST_BOUNDARY.md
**Verdict: VERIFIED**

The docs list:
- G2 shortcut-free ROC-AUC: 0.40 (threshold: >0.65) - FAIL
- G3 placeholder-stripped ROC-AUC: 0.40 (threshold: >0.60) - FAIL
- G4 component holdout: untested (blocked) - FAIL
- G6 marginal gain over dormancy: -0.60 (threshold: >+0.05) - FAIL

The code in `graph_features.py` zeroes unsafe features when `trusted_only=True`. The `PLACEHOLDER_DEVICE_IDS` frozenset in `config.py` lists the known bad device IDs.

---

## Claim 11: "Docker images build locally"

**Source:** docs/RELEASE_READINESS_CHECKLIST.md
**Verdict: NOT INDEPENDENTLY VERIFIED (not executed during this audit)**

The Dockerfile and compose.yaml are syntactically valid. The CI workflow includes a `docker compose build` step. We did not execute a Docker build during this audit to avoid side effects.

---

## Claim 12: "Incremental refresh is watermark-based and no-op on unchanged data"

**Source:** docs/RUNBOOK_LOCAL.md, README.md
**Verdict: VERIFIED**

`pipeline/refresh_live.py` implements:
- Watermark reading from `ops.refresh_state` table
- No-op path when `current_source_event_at <= prior_watermark`
- No-op path when `current_source_event_at is None`
- Affected user derivation via SQL UNION across 5 canonical tables
- Targeted graph + feature rebuild for only affected users
- Stage markers with elapsed time logging
- State persistence on success and failure

---

## Claim 13: "13 API endpoints"

**Source:** README.md, docs/RUNBOOK_LOCAL.md
**Verdict: VERIFIED**

`api/main.py` defines exactly 13 endpoints:
1. GET /healthz
2. POST /pipeline/sync
3. POST /features/rebuild
4. POST /model/train
5. POST /model/score
6. GET /alerts
7. GET /alerts/{alert_id}/report
8. POST /alerts/{alert_id}/decision
9. GET /users/{user_id}/360
10. GET /users/{user_id}/graph
11. GET /metrics/model
12. GET /metrics/threshold
13. GET /metrics/drift

---

## Claim 14: "SHAP case reports with rule hits, graph evidence, and timeline"

**Source:** README.md, docs/RUNBOOK_LOCAL.md
**Verdict: VERIFIED**

`services/diagnosis.py` `build_risk_diagnosis()` returns:
- summary_zh (Chinese-language summary)
- risk_summary (score, level, time)
- shap_top_factors (top 5 SHAP impacts with Chinese feature names)
- rule_hits (triggered rules)
- graph_evidence (device/bank/wallet/blacklist counts)
- timeline_summary (last 10 events across login/crypto/trade)
- recommended_action (monitor/manual_review/hold_withdrawal)

---

## Claim 15: "AWS deployment cost ~$78/month"

**Source:** docs/RUNBOOK_AWS.md
**Verdict: PLAUSIBLE BUT UNVERIFIED**

The itemized cost breakdown is reasonable for Fargate pricing in ap-northeast-1. We cannot verify actual costs without deployment.

---

## Summary

| # | Claim | Verdict |
|---|-------|---------|
| 1 | 6-module pipeline | VERIFIED |
| 2 | 78/78 tests | VERIFIED |
| 3 | graph_trusted_only=True default | VERIFIED |
| 4 | Oracle from separate endpoint | VERIFIED |
| 5 | No AWS deployment executed | VERIFIED |
| 6 | Dormancy baseline PR-AUC=0.9823 | VERIFIED (honest) |
| 7 | LightGBM holdout P=0.9984 R=1.0 | VERIFIED (dormancy caveat documented) |
| 8 | Temporal split prevents leakage | VERIFIED (mechanism correct) |
| 9 | M4 artifact-free signal | PARTIALLY VERIFIED |
| 10 | M5 quarantined 4 failing checks | VERIFIED |
| 11 | Docker images build | NOT INDEPENDENTLY VERIFIED |
| 12 | Incremental refresh watermark-based | VERIFIED |
| 13 | 13 API endpoints | VERIFIED |
| 14 | SHAP case reports | VERIFIED |
| 15 | AWS cost ~$78/month | PLAUSIBLE |

**Overall assessment:** Documentation claims are honest and well-calibrated. The system explicitly documents what it cannot do. No misleading claims were found.
