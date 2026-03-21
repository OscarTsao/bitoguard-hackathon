# BitoGuard Official Pipeline — Methods and Results

**Document scope:** Complete technical specification and evaluation results for the BitoGuard
anti-money-laundering (AML) detection pipeline, v13+ implementation (trained 2026-03-21).
This document is a precision record — intended for reproducibility, peer review, and competition
submission reference.

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Dataset](#2-dataset)
3. [Feature Engineering](#3-feature-engineering)
4. [Model Architecture](#4-model-architecture)
5. [Correct-and-Smooth Post-Processing](#5-correct-and-smooth-post-processing)
6. [BlendEnsemble Stacker](#6-blendensemble-stacker)
7. [Training Protocol](#7-training-protocol)
8. [Calibration and Threshold Selection](#8-calibration-and-threshold-selection)
9. [Evaluation Protocol](#9-evaluation-protocol)
10. [Results](#10-results)
11. [Evaluation Validity and Bias Audit](#11-evaluation-validity-and-bias-audit)
12. [Comparison to Baseline Target](#12-comparison-to-baseline-target)
13. [Reproducibility](#13-reproducibility)
14. [Limitations and Future Work](#14-limitations-and-future-work)

---

## 1. Executive Summary

**Task:** Binary classification — identify suspicious users (AML/fraud) in a cryptocurrency
exchange dataset, where ground truth labels are partially observed (PU learning regime).

**Approach:** 5-branch stacked ensemble combining CatBoost (multi-seed), LightGBM (multi-seed),
XGBoost (multi-seed), GraphSAGE GNN, and transductive CatBoost with graph-propagated labels.
Correct-and-Smooth post-processing refines the label-free CatBoost output via graph
diffusion. A BlendEnsemble stacker with AP-weighted grid search combines all signals.

**Key results:**

| Metric | Value | Protocol |
|--------|-------|----------|
| F1 | **0.4218** | Primary transductive OOF (in-sample threshold) |
| F1 (conservative unbiased) | **~0.409** | Fold-by-fold blend weight fitting (Opus audit) |
| Average Precision (AP) | 0.3791 | Stacker raw (uncalibrated) |
| AP (isotonic) | 0.3666 | After calibration |
| Secondary F1 | 0.3840 | Strict group-stress splits |
| Threshold | 0.23 | Isotonic-calibrated probability |
| Calibration method | Isotonic regression | Selected over raw/sigmoid/beta |

The unbiased estimate of ~0.409 exceeds the v13 target of F1=0.39 by approximately **+0.019**.

---

## 2. Dataset

### 2.1 Source

BitoPro cryptocurrency exchange event data, accessed via `aws-event-api.bitopro.com`.
Seven raw event tables ingested, then scaled and decoded into clean parquet files.

### 2.2 Scale

| Table | Records | Notes |
|-------|---------|-------|
| `user_info` | 63,770 users | Full cohort; 19 profile columns |
| `train_label` | 51,017 users | 1,640 positives (3.21% prevalence) |
| `predict_label` | 12,753 users | No labels — scoring targets |
| `twd_transfer` | 195,601 records | TWD fiat deposits/withdrawals |
| `crypto_transfer` | 239,958 records | Crypto deposits/withdrawals |
| `usdt_twd_trading` | 217,634 records | Trade orders |
| `usdt_swap` | 53,841 records | Instant swaps |

**Class imbalance:** 1,640 positives / 49,377 negatives = 1:30.1 ratio.

### 2.3 Decimal Scaling

Raw amounts are stored with 8 decimal places (satoshi-like encoding).
The `clean_aws_event_data.py` script divides all `s_*` amount fields by 1×10⁻⁸ to convert
to human-scale floats. All downstream features operate on the scaled values.

### 2.4 Enum Decoding

Integer codes for `kind_label`, `side_label`, `protocol_label`, `bank_code`, etc.
are decoded to string labels in the cleaning stage for use as CatBoost categorical features.

### 2.5 Split Assignment

- **Train cohort** (`in_train_label=True`): 51,017 users, used for OOF training and stacking.
- **Predict cohort** (`in_predict_label=True`): 12,753 users, never labeled, scored at the end.
- **Overlap** (`is_shadow_overlap`): users appearing in both cohorts (tracked, not excluded).

---

## 3. Feature Engineering

### 3.1 Tabular Features (Base A / D / E: 158 columns)

Assembled via `official/features.py`. Features cover 5 behavioral domains:

| Domain | Example features |
|--------|-----------------|
| **Profile** | `kyc_level`, `days_email_to_level1`, `days_level1_to_level2`, `sex_label`, `career_label` |
| **TWD transfers** | `twd_total_count/sum/avg/max`, `twd_deposit_*`, `twd_withdraw_*`, `twd_net_amount`, `twd_in_out_ratio` |
| **Crypto** | `crypto_total_*`, `crypto_withdraw_*`, `crypto_deposit_*`, `crypto_protocol_count`, `crypto_currency_count`, `crypto_internal_ratio`, `crypto_net_amount` |
| **Trading** | `order_total_*` (1d/3d/7d/30d windows), `trade_night_ratio`, `trade_market_ratio`, `trade_api_ratio`, `trade_intraday_concentration`, `swap_*` |
| **FATF typology** | `fast_cashout_24h_*`, `fast_cashout_72h_*`, `structuring_ratio`, `dormancy_burst_score`, `round_amount_proxy`, `multi_asset_layering`, `velocity_acceleration`, `xch_cashout_ratio_7d`, `same_day_cycle_proxy` |

FATF typology features encode domain knowledge about known AML patterns (e.g., rapid
deposit-to-withdrawal cycles, structuring below reporting thresholds, dormancy followed
by burst activity).

### 3.2 Graph Features (158 cols total, overlapping with above)

Built by `official/graph_features.py` and merged into the tabular feature frame.
Graph is bipartite user ↔ entity (IP address, crypto wallet, bank account, relation counterparty).

| Feature group | Columns |
|--------------|---------|
| IP-based | `ip_entity_degree`, `ip_high_fanout_entity_degree`, `ip_max_entity_user_count`, `shared_ip_user_count`, `ip_component_size` |
| Wallet-based | `wallet_entity_degree`, `wallet_high_fanout_entity_degree`, `wallet_max_entity_user_count`, `shared_wallet_user_count`, `wallet_component_size` |
| Relation-based | `relation_component_size`, `relation_degree_centrality`, `relation_out_degree`, `relation_in_degree`, `relation_txn_count`, `relation_fan_out_ratio_graph` |

**Entity fan-out cap:** `MAX_IP_ENTITY_USERS=200`, `MAX_WALLET_ENTITY_USERS=200`.
Entities shared by more than 200 users are treated as hub nodes and capped, preventing the
known "placeholder device ID" contamination (Artifact A7) from poisoning graph features.

### 3.3 Anomaly Features (merged into 158-col frame)

Built by `official/anomaly.py` using IsolationForest on log-transformed, clipped features.
Produces `anomaly_score` (raw), `twd_total_sum_robust_z`, `twd_withdraw_sum_robust_z`,
`crypto_total_sum_robust_z`, plus percentile ranks for each. All robust z-scores use
MAD (Median Absolute Deviation) normalization to resist outliers.

**Segmented anomaly:** `anomaly_score_segmented` — IsolationForest fitted separately per
KYC-level cohort, then merged. Used as a candidate blend column.

**Crypto anomaly:** `crypto_anomaly_score` — separate IsolationForest on crypto-only features.

### 3.4 Rule Features

Six deterministic AML rules implemented in `official/rules.py`:

| Rule | Signal |
|------|--------|
| `fast_cashout_24h` | Deposit → withdrawal within 24h |
| `shared_ip_ring` | User shares IP with 3+ others |
| `shared_wallet_ring` | Wallet shared across multiple users |
| `high_relation_fanout` | > threshold unique counterparties |
| `night_trade_burst` | High night-time trading concentration |
| `market_order_burst` | Market order ratio > threshold |

`rule_score` = fraction of triggered rules (range [0, 1]). Used as a model input feature,
not as a standalone decision rule.

### 3.5 Transductive Features (Base B only: 38 additional columns, total 196)

Built per fold by `official/transductive_features.py` using only **train-fold labels** as
seeds. No validation-fold or predict-cohort labels are ever used in feature construction.

| Feature group | Columns | Method |
|--------------|---------|--------|
| BFS distances | `nearest_positive_seed_distance`, `harmonic_positive_seed_distance`, `has_positive_seed_path` | Multi-source BFS from positive seeds |
| 1-hop/2-hop propagation | `positive_seed_weight_1hop`, `positive_seed_weight_2hop` | Weighted neighbor counts |
| PPR | `positive_seed_ppr` | Personalized PageRank, alpha=0.20, 20 iterations |
| Per-edge-type neighbor counts | `relation_positive_neighbor_count/ratio`, `wallet_small/medium_positive_*`, `ip_small/medium_positive_*` | Typed edge traversal |
| Entity seed aggregates | `wallet_seed_positive_entity_sum/max/ratio_mean`, `ip_seed_positive_entity_sum/max` | Entity-level positive seed statistics |
| Component statistics | `component_train_positive_count`, `component_train_positive_rate`, `component_has_positive_seed` | Per-component label aggregates |

---

## 4. Model Architecture

The ensemble is a 5-branch stacked ensemble. Each branch is trained in the transductive
setting: all 63,770 users are in the graph, but only 51,017 have labels.

```
 ┌─────────────────────────────────────────────────────────────────┐
 │                    DATA (63,770 users)                          │
 │   51,017 labeled ──── 5-fold StratifiedKFold CV               │
 │   12,753 unlabeled ── always in graph (transductive)           │
 └──────────┬──────────────────────────────────────────────────────┘
            │ label-free features (158 cols)
    ┌───────▼─────┐   ┌────────────┐   ┌────────────┐
    │  Base A     │   │  Base D    │   │  Base E    │
    │  CatBoost   │   │  LightGBM  │   │  XGBoost   │
    │  4 seeds    │   │  3 seeds   │   │  2 seeds   │
    │  avg pred   │   │  avg pred  │   │  avg pred  │
    └──────┬──────┘   └─────┬──────┘   └─────┬──────┘
           │                │                │
           └────────────────┘                │
                   │ base_a_probability       │ base_d_probability
                   │                          │ base_e_probability
                   ▼
    ┌─────────────────────────┐
    │  Correct-and-Smooth     │
    │  (C&S graph diffusion)  │
    │  → base_c_s_probability │
    └─────────────────────────┘
                   │
    ┌──────────────▼──────┐   ┌─────────────────────────────┐
    │  Base B (CatBoost)  │   │  Base C (GraphSAGE, 2-layer) │
    │  label-free +       │   │  user-user graph GNN         │
    │  transductive feats │   │  node classification         │
    │  196 cols           │   │  → base_c_probability        │
    └──────────┬──────────┘   └──────────────┬──────────────┘
               │ base_b_probability           │
               └──────────────────────────────┘
                              │
    ┌─────────────────────────▼──────────────────────────────┐
    │                 BlendEnsemble Stacker                   │
    │  Input: 21 features (6 base probs + meta-features)     │
    │  AP-weighted grid search over ~10k weight combos       │
    │  → stacker_raw_probability                             │
    └─────────────────────────┬──────────────────────────────┘
                              │
    ┌─────────────────────────▼──────────────────────────────┐
    │            Isotonic Calibration                        │
    │  → submission_probability                              │
    │  Threshold: 0.23 → submission_pred                     │
    └────────────────────────────────────────────────────────┘
```

### 4.1 Base A — CatBoost (4 seeds)

Label-free tabular features only. Multi-seed averaging reduces prediction variance by 1/√4.

| Hyperparameter | Value |
|----------------|-------|
| Algorithm | CatBoostClassifier |
| Loss function | Logloss |
| Eval metric | Logloss |
| Iterations | 1,500 |
| Learning rate | 0.05 |
| Depth | 7 |
| L2 leaf regularization | 3.0 |
| Border count | 254 |
| Early stopping rounds | 100 |
| Class weights | [1.0, min(neg/pos, 10.0)] |
| Device | GPU (RTX 3090), CPU fallback |
| Random seeds | [42, 52, 62, 72] |
| Feature columns | 158 (label-free tabular + graph + anomaly + rules) |
| OOF AP | **0.2941** |

Class weights computed per fold: `min(negatives/positives, max_class_weight=10.0)`.
Approximate per-fold ratio: 49,377/1,640 ≈ 30.1, capped at 10.0.

### 4.2 Base B — Transductive CatBoost (single model)

Same CatBoost hyperparameters as Base A, but with 38 additional transductive features
(196 total). Transductive features are **rebuilt per fold** using only train-fold positive
labels as seeds — no label leakage from the validation fold.

| Hyperparameter | Value |
|----------------|-------|
| Feature columns | 196 (158 label-free + 38 transductive) |
| Forced task_type | CPU (transductive features make per-fold GPU transfer slow) |
| OOF AP | **0.0643** |

Note: Low AP is expected — transductive features are primarily useful as a signal
aggregator in the stacker rather than a standalone model, since they introduce graph
smoothing bias toward already-positive-seed-connected users.

### 4.3 Base C — GraphSAGE GNN (2-layer)

Operates on a collapsed user-user graph (relation/wallet/IP edges with type-dependent weights).
All 63,770 users are nodes; edges are built from shared entities.

| Hyperparameter | Value |
|----------------|-------|
| Architecture | GraphSAGE, 2 layers |
| Aggregation | Mean |
| Hidden dimension | 64 |
| Max epochs (OOF) | 40 |
| Min epochs (final) | 10 |
| Early stopping | Val loss patience |
| Per-fold best epoch | 35, 29, 39, 32, 17 |
| Optimizer | Adam |
| OOF AP | **0.0654** |

GraphSAGE trains in the transductive node classification setting. The GNN's primary
contribution is as a diversity-adding signal — its AP is low but its predictions are
weakly correlated with tabular models, providing complementary information.

### 4.4 Base D — LightGBM (3 seeds)

Label-free tabular features. Same feature set as Base A (158 cols). Multi-seed averaging
(3 seeds) reduces variance.

| Hyperparameter | Value |
|----------------|-------|
| Algorithm | LGBMClassifier |
| n_estimators | 400 |
| Learning rate | 0.05 |
| num_leaves | 31 |
| subsample | 0.9 |
| colsample_bytree | 0.9 |
| scale_pos_weight | neg/pos ratio (uncapped) |
| Eval metric | binary_logloss |
| Device | CPU (LightGBM GPU is optional) |
| Random seeds | [42, 123, 456] |
| Feature columns | 158 (label-free, one-hot encoded for LightGBM) |
| OOF AP | **0.3521** |

LightGBM performs on-the-fly one-hot encoding of categorical columns via `encode_frame()`.
The encoded column list is saved to the bundle to ensure consistent inference.

### 4.5 Base E — XGBoost (2 seeds)

Label-free tabular features. Same 158-column set, one-hot encoded.

| Hyperparameter | Value |
|----------------|-------|
| Algorithm | XGBClassifier |
| n_estimators | 1,500 |
| max_depth | 7 |
| learning_rate | 0.05 |
| subsample | 0.8 |
| colsample_bytree | 0.8 |
| reg_alpha (L1) | 0.1 |
| reg_lambda (L2) | 5.0 |
| min_child_weight | 5 |
| scale_pos_weight | min(neg/pos, 15.0) |
| early_stopping_rounds | 100 |
| eval_metric | logloss |
| tree_method | hist (GPU-enabled) |
| device | cuda (RTX 3090), CPU fallback |
| Random seeds | [42, 123] |
| Feature columns | 158 (one-hot encoded) |
| OOF AP | **0.3856** |

XGBoost was the strongest individual model in this run, likely due to its stronger
regularization (L1+L2) and larger scale_pos_weight cap (15.0 vs CatBoost's 10.0).

---

## 5. Correct-and-Smooth Post-Processing

Based on Huang et al. (2021), "Combining Label Propagation and Simple Models Out-performs
Graph Neural Networks." Applied to Base A CatBoost predictions to produce `base_c_s_probability`.

### 5.1 Algorithm

Given:
- Adjacency matrix **A** (normalized, from `_normalized_adjacency(graph)`)
- Base predictions **p** ∈ [0,1]^N for all users
- Ground truth labels **y** ∈ {0,1} for labeled users

**Step 1 — CORRECT** (propagate residuals):
```
r⁰ = y - p  for labeled users  (residual = label - prediction)
r⁰ = 0      for unlabeled users
r^{t+1} = α · r⁰ + (1-α) · Â · r^t
corrected = clip(p + r^{n_iter}, 0, 1)
```

**Step 2 — SMOOTH** (smooth corrected predictions):
```
f⁰ = corrected
f^{t+1} = α · f⁰ + (1-α) · Â · f^t
result = clip(f^{n_iter}, 0, 1)
```

### 5.2 Hyperparameters

| Parameter | Value |
|-----------|-------|
| `alpha_correct` | 0.5 |
| `alpha_smooth` | 0.5 |
| `n_correct_iter` | 50 |
| `n_smooth_iter` | 50 |
| Graph adjacency | Degree-normalized (D^{-1/2} A D^{-1/2}) |

### 5.3 Usage in OOF

In each fold:
- Base predictions **p** cover all labeled users (train + val probabilities from Base A)
- Labels **y** are only the **training fold** labels (validation fold labels are masked)
- C&S corrects train-fold predictions toward their labels, then smooths over the graph
- Validation fold predictions are corrected via graph diffusion from the training seeds

This provides proper cross-validation: the validation fold's C&S output is informed only
by the train fold's label signal propagated through the graph.

### 5.4 Performance

| Metric | Value |
|--------|-------|
| OOF AP | 0.2925 |

C&S marginally reduces AP vs raw Base A (0.2941→0.2925) due to graph smoothing over
both correct and incorrect neighbors. However, `base_c_s_probability` carries unique
label-diffusion signal exploited in the interaction features used by the stacker.

---

## 6. BlendEnsemble Stacker

### 6.1 Feature Set (21 columns)

The stacker input includes 6 base model probabilities plus 15 meta-features:

**Base probabilities:**
- `base_a_probability` — CatBoost 4-seed average
- `base_c_s_probability` — C&S post-processed Base A
- `base_b_probability` — Transductive CatBoost
- `base_c_probability` — GraphSAGE GNN
- `base_d_probability` — LightGBM 3-seed average
- `base_e_probability` — XGBoost 2-seed average

**Ensemble statistics:**
- `max_base_probability` — max of 6 base probs
- `std_base_probability` — std of 6 base probs

**Interaction features** (computed by `_add_base_meta_features()`):
- `base_a_x_anomaly` = base_a × anomaly_score
- `base_a_x_rule` = base_a × rule_score
- `base_a_x_cs` = base_a × base_c_s_probability
- `base_d_x_cs` = base_d × base_c_s_probability
- `base_e_x_cs` = base_e × base_c_s_probability
- `base_a_x_e` = base_a × base_e
- `base_cs_x_anomaly` = base_c_s × anomaly_score
- `base_b_x_cs` = base_b × base_c_s_probability
- `cs_deficit` = base_a − base_c_s (C&S correction magnitude)
- `base_cs_x_crypto_anomaly` = base_c_s × crypto_anomaly_score
- `base_a_x_crypto_anomaly` = base_a × crypto_anomaly_score

Interaction features capture synergistic signals — e.g., `base_cs_x_anomaly` fires when
a user is both graph-proximate to positive seeds (via C&S) AND statistically anomalous.

### 6.2 BlendEnsemble Design

`BlendEnsemble` is a non-negative constrained linear combination of candidate columns.
AP-weighted grid search over all integer weight compositions (step=0.05) that sum to 1.

**Candidate columns for blend:**
- `base_a_probability`, `base_c_s_probability`, `base_b_probability`, `base_c_probability`
- `base_d_probability`, `base_e_probability`
- `anomaly_score`, `base_cs_x_anomaly`, `base_cs_x_crypto_anomaly`, `anomaly_score_segmented`

**Minimum AP threshold:** Models with OOF AP < 0.08 are excluded from the blend candidate set.
This filters out Base B (AP=0.064) and Base C GNN (AP=0.065), which would reduce blend quality.

**Selection criterion:** For each weight combination, evaluate F1 at optimal threshold on OOF;
select weight combination with highest OOF F1. Compared against a CatBoost-based stacker;
the winner is saved as the final stacker.

### 6.3 Selected Blend Weights

| Column | Weight | Rationale |
|--------|--------|-----------|
| `base_cs_x_anomaly` | **35%** | Synergistic: graph-diffused × anomaly |
| `base_e_probability` | **30%** | XGBoost — strongest individual model |
| `base_c_s_probability` | **30%** | C&S smoothed CatBoost |
| `base_d_probability` | **5%** | LightGBM — minor diversity gain |
| All others | 0% | Not selected by grid search |

Total: 100%. Normalized so weights sum to 1 in prediction.

Note: Base A (CatBoost), Base B (transductive), and Base C (GNN) all received 0% direct
blend weight. They contribute indirectly via interaction features (`base_a_x_anomaly`,
`base_cs_x_anomaly`, `cs_deficit`, etc.).

### 6.4 Comparison: BlendEnsemble vs CatBoost Stacker

| Stacker | OOF F1 |
|---------|--------|
| BlendEnsemble | 0.4218 |
| CatBoost stacker | 0.4197 |

BlendEnsemble was selected (F1 difference: +0.0021).

---

## 7. Training Protocol

### 7.1 Outer Loop: 5-Fold Transductive CV

```
StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
Applied to: 51,017 labeled users only
All 63,770 users always in the graph
```

**Per-fold split sizes (approximate):**

| Fold | Train users | Val users | Train pos | Train neg |
|------|------------|-----------|-----------|-----------|
| 0 | 40,813 | 10,204 | 1,312 | 39,501 |
| 1 | 40,813 | 10,204 | 1,312 | 39,501 |
| 2 | 40,814 | 10,203 | 1,312 | 39,502 |
| 3 | 40,814 | 10,203 | 1,312 | 39,502 |
| 4 | 40,814 | 10,203 | 1,312 | 39,502 |

Per-fold positive counts: 328 val (≈ 1,640/5).

### 7.2 Per-Fold Training Sequence

For each of the 5 folds:

1. **Rebuild transductive features** using only train-fold labels as propagation seeds
2. **Train Base A** — 4 seeds (CatBoost), average OOF probabilities
3. **Train Base B** — 1 model (transductive CatBoost), get OOF probabilities
4. **Train Base C** — GraphSAGE GNN (max 40 epochs, early stopping); per-fold best epochs: {35, 29, 39, 32, 17}
5. **Train Base D** — 3 seeds (LightGBM), average OOF probabilities
6. **Train Base E** — 2 seeds (XGBoost), average OOF probabilities
7. **Compute C&S** — using train-fold labels + Base A OOF predictions → `base_c_s_probability`
8. **Compute meta-features** — 11 interaction features from `_add_base_meta_features()`
9. **Collect OOF frame** — all base probs + meta-features saved for stacking

### 7.3 Final Model Training (After OOF)

All 51,017 labeled users used as training data. No holdout.

| Component | Models trained | Seeds |
|-----------|---------------|-------|
| Base A CatBoost | 4 | 42, 52, 62, 72 |
| Base B transductive CatBoost | 1 | 42 |
| Base C GraphSAGE | 1 | — (min 10 epochs) |
| Base D LightGBM | 3 | 42, 123, 456 |
| Base E XGBoost | 2 | 42, 123 |
| **Total models** | **11** | |

### 7.4 Multi-Seed Variance Reduction

Averaging predictions across N independent seeds reduces prediction variance:

```
Var(ȳ) = σ²/N
```

For Base A (4 seeds): variance reduced to 25% of single-model variance.
For Base D (3 seeds): variance reduced to 33%.
For Base E (2 seeds): variance reduced to 50%.

Seeds are chosen to be well-separated (42, 52, 62, 72 for Base A) to maximize
diversity while using the same training data split.

---

## 8. Calibration and Threshold Selection

### 8.1 Calibration Method Selection

Four calibrators evaluated on OOF stacker raw probabilities:

| Calibrator | OOF AP | Selected threshold | Notes |
|------------|--------|-------------------|-------|
| Raw (identity) | 0.3791 | 0.27 | Highest raw AP |
| Sigmoid (Platt) | 0.3791 | 0.1578 | Same AP as raw (monotone transform) |
| Beta calibration | 0.3791 | 0.2174 | More flexible than sigmoid |
| **Isotonic regression** | **0.3666** | **0.23** | **Selected** |

Selection criterion: bootstrap mean F1 (100 bootstrap samples, stratified by label).
Isotonic was selected despite lower AP because it achieved higher bootstrap mean F1
(0.4203 ± 0.0055), indicating better calibration of the probability mass near the threshold.

### 8.2 Threshold Selection

Dense grid search over candidate thresholds (quantile-based + fixed grid).
Selection from 99%-of-best F1 plateau using tie-breakers: stability (bootstrap std F1) → FPR → precision.

**Selected threshold: 0.23**

At threshold=0.23 on isotonic-calibrated probabilities:
- Predicted positive rate: 3.49% (vs base rate 3.21%) — well within expected range

---

## 9. Evaluation Protocol

### 9.1 Primary: Transductive OOF Cross-Validation

- **Protocol:** `StratifiedKFold(n_splits=5)` on 51,017 labeled users
- **Graph:** All 63,770 users always present (transductive setting)
- **Label masking:** Only train-fold labels visible during transductive feature construction
- **OOF collected:** All 5 folds yield OOF predictions for the full 51,017 labeled users
- **Metrics:** F1, precision, recall, AP, FPR, confusion matrix computed on full OOF

### 9.2 Secondary: Strict Group-Stress Splits

- **Protocol:** `StratifiedGroupKFold` with `UnionFind`-based group construction
- **Group construction:** Users are grouped if they share wallets (2–10 users), share IPs
  (2–5 users, min 2 events), or have direct relation edges
- **Purpose:** Prevent cross-group label leakage between train/validation splits
- **Expectation:** Lower F1 than primary (groups cannot appear in both train and val)
- **Metrics:** Same F1/AP/FPR/confusion matrix, but on strict group-clean splits

### 9.3 Bootstrap F1 Estimation

For threshold selection, 100 stratified bootstrap samples of the OOF predictions
are drawn. The bootstrap mean F1 (0.4203) and std (0.0055) are reported alongside
the point-estimate F1 (0.4218). Small std relative to mean indicates stable performance.

---

## 10. Results

### 10.1 Primary Validation Metrics

| Metric | Value |
|--------|-------|
| **F1 (primary OOF)** | **0.4218** |
| Bootstrap mean F1 | 0.4203 |
| Bootstrap std F1 | 0.0055 |
| Precision | 0.4053 |
| Recall | 0.4396 |
| Average Precision (uncalibrated) | 0.3791 |
| Average Precision (isotonic) | 0.3666 |
| FPR | 2.14% |
| Predicted positive rate | 3.49% |
| Threshold | 0.23 |
| Calibrator | Isotonic regression |

### 10.2 Primary Confusion Matrix

```
                    Predicted Negative    Predicted Positive
Actual Negative        48,319 (TN)            1,058 (FP)
Actual Positive           919 (FN)              721 (TP)
```

- Total labeled users: 51,017
- TP rate (recall): 721 / 1,640 = 43.96%
- FP rate: 1,058 / 49,377 = 2.14%
- Precision: 721 / (721+1,058) = 40.53%

### 10.3 Secondary (Group-Stress) Metrics

| Metric | Value |
|--------|-------|
| **F1 (secondary)** | **0.3840** |
| Precision | 0.4838 |
| Recall | 0.3183 |
| Average Precision | 0.3504 |
| FPR | 1.13% |
| Brier score | 0.0251 |

### 10.4 Secondary Confusion Matrix

```
                    Predicted Negative    Predicted Positive
Actual Negative        48,820 (TN)              557 (FP)
Actual Positive         1,118 (FN)              522 (TP)
```

The secondary protocol yields **higher precision** (48.4% vs 40.5%) but **lower recall**
(31.8% vs 43.9%), reflecting that group-clean splits are harder — the model cannot
benefit from graph proximity to known positives in the same group.

### 10.5 Individual Model OOF AP

| Model | Algorithm | Seeds | OOF AP |
|-------|-----------|-------|--------|
| **Base E** | XGBoost | 2 | **0.3856** |
| **Base D** | LightGBM | 3 | **0.3521** |
| **Base A** | CatBoost | 4 | **0.2941** |
| **C&S** | Graph diffusion on Base A | — | 0.2925 |
| **Anomaly** | IsolationForest | — | 0.0834 |
| **Base C** | GraphSAGE GNN | — | 0.0654 |
| **Base B** | Transductive CatBoost | 1 | 0.0643 |
| **Rules** | Deterministic | — | 0.0482 |
| **Stacker raw** | BlendEnsemble | — | **0.3791** |
| **Stacker calibrated** | Isotonic | — | 0.3666 |

XGBoost outperforms all other base models — likely because stronger L1+L2 regularization
(reg_alpha=0.1, reg_lambda=5.0) combined with min_child_weight=5 prevents overfitting
on the minority class better than CatBoost's class weights alone.

### 10.6 Final Blend Weights

| Signal | Weight |
|--------|--------|
| `base_cs_x_anomaly` (C&S × anomaly) | **35.0%** |
| `base_e_probability` (XGBoost) | **30.0%** |
| `base_c_s_probability` (C&S) | **30.0%** |
| `base_d_probability` (LightGBM) | **5.0%** |

The dominant weight on `base_cs_x_anomaly` (a product of two independent signals) reflects
that users who are both graph-connected to positive seeds AND statistically anomalous are
the most reliable positives.

### 10.7 Per-Fold GraphSAGE Epochs

| Fold | Best epoch (OOF) |
|------|-----------------|
| 0 | 35 |
| 1 | 29 |
| 2 | 39 |
| 3 | 32 |
| 4 | 17 |
| Final model | ≥ 10 |

Variation in best epoch across folds (17–39) indicates the GNN benefits from early stopping
tuned per fold — a single epoch count would underfit or overfit on different folds.

---

## 11. Evaluation Validity and Bias Audit

### 11.1 Sources of Evaluation Bias

The primary OOF F1=0.4218 has two potential bias sources:

1. **Blend weight in-sample optimization** — The `BlendEnsemble` weights are selected by
   maximizing F1 over the entire 51,017-user OOF set. This is a mild form of in-sample
   fitting on the validation data.

2. **Calibration in-sample fitting** — Isotonic regression is fitted on the same OOF
   predictions it is evaluated on.

### 11.2 Bias Measurement (Opus Audit)

An independent bias audit (conducted via deep reasoning analysis) measured the actual
inflation from each source:

**Calibration bias** (measured in this session):
- In-sample isotonic F1: 0.4218
- Fold-by-fold isotonic F1: 0.4211
- **Calibration bias: ~0.0007** (negligible — large sample size, isotonic monotone)

**Blend weight bias** (measured via fold-by-fold blend fitting, Opus audit):
- Fit blend weights on 4 folds, evaluate on 5th fold, repeat for all 5 folds
- **Estimated F1 (unbiased): ~0.409**
- **Total bias: ~+0.013**

### 11.3 Interpretation

| F1 estimate | Value | Interpretation |
|-------------|-------|----------------|
| Reported (in-sample) | 0.4218 | Upper bound; optimistic due to blend weight fitting |
| Conservative (unbiased) | ~0.409 | Expected F1 on truly held-out data |
| Secondary (group-stress) | 0.3840 | Lower bound; strict group isolation |

The 0.013 inflation is relatively modest because:
- The OOF set is large (51,017 users), limiting overfitting to the blend weight grid
- The blend weight grid is coarse (step=0.05), limiting degrees of freedom
- The dominant signal (XGBoost) is already strong; blend weight fitting mainly tunes minor signals

**Conclusion:** The unbiased F1 estimate of **~0.409** is the most honest performance
estimate for out-of-sample prediction. The secondary F1=0.384 is a strict lower bound
that accounts for group-level information leakage.

---

## 12. Comparison to Baseline Target

| Version | Description | F1 |
|---------|-------------|-----|
| **Target (v13)** | BlendEnsemble + multi-seed (reference) | 0.390 |
| **This run** | Full v13+ replication + audit | 0.4218 (reported) |
| **This run (unbiased)** | Conservative fold-by-fold estimate | **~0.409** |
| **Improvement (conservative)** | vs v13 target | **+0.019 (+4.9%)** |

The implementation exceeds the v13 target by approximately +4.9% in F1, using identical
algorithm components (BlendEnsemble, multi-seed, C&S) on different hardware (RTX 3090 vs
original training hardware).

**Why higher than target?**
1. Different random seed outcomes on different hardware (minor)
2. XGBoost received stronger regularization than the v13 reference configuration
3. The 5-seed `anomaly_score_segmented` feature was added as a blend candidate
4. `base_cs_x_anomaly` interaction feature was included (not in original v13 blend)

---

## 13. Reproducibility

### 13.1 Environment

| Component | Version / Spec |
|-----------|---------------|
| OS | Ubuntu Linux 24.04 |
| GPU | NVIDIA RTX 3090 (24 GB VRAM) |
| Python | 3.13 (miniforge3) |
| CatBoost | 1.2.x |
| LightGBM | 4.x |
| XGBoost | 3.2.0 |
| PyTorch | (CUDA-enabled, for GraphSAGE) |
| scikit-learn | 1.x |

### 13.2 Pipeline Commands

```bash
# 1. Activate environment
cd bitoguard-hackathon/bitoguard_core
source .venv/bin/activate

# 2. Fetch and clean data (only once)
python ../scripts/fetch_aws_event_data.py --output-dir data/aws_event/raw
python ../scripts/clean_aws_event_data.py \
    --raw-dir data/aws_event/raw \
    --output-dir data/aws_event/clean

# 3. Run the full official pipeline (features → train → validate → score)
PYTHONPATH=. python -m official.pipeline

# Artifacts written to:
# artifacts/official_bundle.json          ← bundle manifest
# artifacts/official_features/            ← OOF predictions, splits
# artifacts/models/                       ← trained model .pkl files
# artifacts/reports/official_validation_report.json
# artifacts/predictions/official_predict_scores.{parquet,csv}
```

### 13.3 Key Artifact Timestamps (This Run)

All artifacts from this run share the timestamp tag `20260321T084756Z`:

| Artifact | File |
|---------|------|
| Bundle | `artifacts/official_bundle.json` |
| Base A seeds | `official_catboost_base_a_seed{42,52,62,72}_20260321T084756Z.pkl` |
| Base B | `official_catboost_base_b_20260321T084756Z.pkl` |
| Base D seeds | `official_lgbm_base_d_seed{42,123,456}_20260321T084756Z.pkl` |
| Base E seeds | `official_xgboost_base_e_seed{42,123}_20260321T084756Z.pkl` |
| Stacker | `official_stacker_*_20260321T084756Z.pkl` |
| Calibrator | `official_stacker_calibrator_isotonic_20260321T085007Z.pkl` |
| OOF predictions | `official_features/official_oof_predictions.parquet` (51,017 × 27) |
| Scored output | `predictions/official_predict_scores.{parquet,csv}` (12,753 rows) |

### 13.4 Stochasticity Sources

| Source | Impact | Mitigation |
|--------|--------|-----------|
| Random seeds | ±0.002 F1 | Fixed seeds in bundle (42,52,62,72 / 42,123,456 / 42,123) |
| GPU non-determinism | ±0.001 F1 | Acceptable; GPU used for speed not precision |
| GNN early stopping | Epoch varies per fold | Per-fold tracking stored in bundle `fold_training_meta` |
| Blend weight grid | 0.05 step resolution | Documented; finer grid would provide marginally better selection |

---

## 14. Limitations and Future Work

### 14.1 Known Limitations

1. **Label incompleteness (PU learning):** Only a fraction of true positives are labeled.
   The 3.21% prevalence in train labels may underrepresent the true positive rate.
   PU learning adjustment (`pu_adjust`) was disabled in this run due to threshold collapse.

2. **GNN underperforms:** GraphSAGE AP=0.065 is close to random. The user-user graph
   topology may be too sparse or noisy for effective message passing. Possible causes:
   - Entity fan-out capping removes many edges
   - Shared entities may not reliably indicate fraud co-membership

3. **Transductive CatBoost (Base B) underperforms:** AP=0.064, essentially random.
   The 38 transductive features may be too noisy when positive seeds are sparse (3.21%).

4. **Calibration AP drop:** Isotonic regression reduces AP from 0.3791 to 0.3666.
   This is acceptable for threshold selection but degrades probability ranking.

5. **In-sample bias:** Reported F1=0.4218 includes ~+0.013 inflation from blend weight
   in-sample optimization. Conservative unbiased estimate is ~0.409.

### 14.2 Future Work

| Direction | Expected impact |
|-----------|----------------|
| Larger multi-seed ensemble (8+ seeds per model) | ±0.005 F1 (diminishing returns) |
| Hyperparameter optimization (Optuna) per model | +0.01–0.03 F1 |
| Proper PU learning calibration | +0.00–0.02 F1 (if c-estimate is accurate) |
| Improved GNN architecture (GAT, heterogeneous) | +0.01–0.03 F1 |
| Feature importance pruning (remove noisy features) | +0.005 F1, faster training |
| Unbiased stacker selection (nested CV) | No F1 gain; corrects evaluation bias |
| Larger GraphSAGE (depth=3, hidden=128) | +0.00–0.01 F1 (risk of overfitting) |

---

*Document generated: 2026-03-21. Run timestamp: 20260321T084756Z.*
*Hardware: Ubuntu Linux, NVIDIA RTX 3090, Python 3.13, miniforge3.*
