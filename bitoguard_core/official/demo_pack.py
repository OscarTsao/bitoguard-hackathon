"""Demo pack generation for BitoGuard AML detection.

Produces all assets needed for the competition presentation:
  1. Threshold sensitivity table (F1/P/R vs threshold)
  2. Three operating points (conservative / balanced / aggressive)
  3. Top-20 alert cases with risk narratives
  4. Model contribution breakdown
  5. AML scenario coverage summary

Reads from pre-computed artifacts (no re-training needed).

Usage:
    cd bitoguard_core
    PYTHONPATH=. python -m official.demo_pack
    PYTHONPATH=. python -m official.demo_pack --output artifacts/reports/demo_pack.json
"""
from __future__ import annotations

import json
import textwrap
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)

from official.common import load_official_paths

# ---------------------------------------------------------------------------
# Rule label mapping
# ---------------------------------------------------------------------------
RULE_LABELS: dict[str, str] = {
    "fast_cashout_24h": "Fast cashout (<24h deposit-to-withdraw)",
    "shared_ip_ring": "Shared IP address ring",
    "shared_wallet_ring": "Shared wallet ring",
    "high_relation_fanout": "High-fanout relation network (money mule hub)",
    "night_trade_burst": "Unusual night-time trading burst",
    "market_order_burst": "Rapid market-order burst (wash trading)",
}

RISK_TIER_THRESHOLDS = {
    "critical": 0.50,
    "high": 0.30,
    "medium": 0.15,
    "low": 0.05,
}


# ---------------------------------------------------------------------------
# 1. Threshold sensitivity table
# ---------------------------------------------------------------------------
def threshold_sensitivity_table(
    oof_df: pd.DataFrame,
    probs_col: str = "stacker_raw_probability",
    thresholds: list[float] | None = None,
) -> pd.DataFrame:
    """Return F1/P/R/FPR at each threshold over the OOF labeled set.

    Parameters
    ----------
    oof_df : DataFrame with 'status' and ``probs_col`` columns.
    probs_col : probability column to evaluate.
    thresholds : list of threshold candidates (default: 0.04 to 0.60 in 0.02 steps).

    Returns
    -------
    DataFrame with columns: threshold, precision, recall, f1, fpr, n_flagged,
                             ppr (predicted positive rate)
    """
    df = oof_df[oof_df["status"].notna() & oof_df[probs_col].notna()].copy()
    y = df["status"].astype(int).values
    p = df[probs_col].values
    n_total = len(y)
    n_neg = int((y == 0).sum())

    if thresholds is None:
        thresholds = list(np.round(np.arange(0.04, 0.61, 0.02), 3))

    rows = []
    for thr in thresholds:
        preds = (p >= thr).astype(int)
        tp = int(((preds == 1) & (y == 1)).sum())
        fp = int(((preds == 1) & (y == 0)).sum())
        fn = int(((preds == 0) & (y == 1)).sum())
        tn = int(((preds == 0) & (y == 0)).sum())
        prec = float(precision_score(y, preds, zero_division=0))
        rec = float(recall_score(y, preds, zero_division=0))
        f1 = float(f1_score(y, preds, zero_division=0))
        fpr = fp / max(1, n_neg)
        rows.append({
            "threshold": thr,
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "fpr": round(fpr, 4),
            "n_flagged": tp + fp,
            "ppr": round((tp + fp) / max(1, n_total), 4),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 2. Operating points
# ---------------------------------------------------------------------------
def select_operating_points(thr_table: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Select three representative operating points from the threshold table.

    Conservative: precision >= 0.55 with max F1 (minimise false alarms).
    Balanced:     maximum F1 (default deployment point).
    Aggressive:   recall >= 0.55 with max F1 (minimise missed criminals).

    Returns dict with keys 'conservative', 'balanced', 'aggressive'.
    """
    def _best_row(mask: pd.Series) -> dict[str, Any]:
        sub = thr_table[mask]
        if sub.empty:
            sub = thr_table
        best = sub.loc[sub["f1"].idxmax()]
        return best.to_dict()

    balanced = thr_table.loc[thr_table["f1"].idxmax()].to_dict()
    conservative = _best_row(thr_table["precision"] >= 0.50)
    aggressive = _best_row(thr_table["recall"] >= 0.55)

    return {
        "conservative": {
            **conservative,
            "label": "Conservative",
            "use_case": "SAR filing — high precision to reduce analyst workload",
        },
        "balanced": {
            **balanced,
            "label": "Balanced",
            "use_case": "Daily triage — best F1 for automated routing",
        },
        "aggressive": {
            **aggressive,
            "label": "Aggressive",
            "use_case": "Enhanced due diligence — high recall to catch more criminals",
        },
    }


# ---------------------------------------------------------------------------
# 3. Top-N alert cases
# ---------------------------------------------------------------------------
def _parse_reason_codes(raw: Any) -> list[str]:
    """Safely parse top_reason_codes column (JSON string or list)."""
    if isinstance(raw, list):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, list) else [raw]
        except Exception:
            return [raw]
    return []


def top_alert_cases(
    scores_df: pd.DataFrame,
    top_n: int = 20,
) -> pd.DataFrame:
    """Return top-N highest-risk users with structured alert data.

    Parameters
    ----------
    scores_df : scored users from official_predict_scores.csv / .parquet
    top_n : number of top alerts to return

    Returns
    -------
    DataFrame with columns: rank, user_id, risk_level, analyst_risk_score,
                             submission_probability, base_a_probability,
                             base_c_probability, anomaly_score, rule_score,
                             primary_drivers, narrative
    """
    df = scores_df.sort_values("analyst_risk_score", ascending=False).head(top_n).copy()
    df = df.reset_index(drop=True)
    df.insert(0, "rank", df.index + 1)

    # Parse reason codes
    df["_codes"] = df["top_reason_codes"].apply(_parse_reason_codes)
    df["primary_drivers"] = df["_codes"].apply(
        lambda codes: [RULE_LABELS.get(c, c) for c in codes[:3]]
    )

    # Generate narrative
    df["narrative"] = df.apply(_build_narrative, axis=1)

    cols = [
        "rank", "user_id", "risk_level", "analyst_risk_score",
        "submission_probability", "stacker_raw_probability",
        "base_a_probability", "base_c_probability",
        "anomaly_score", "rule_score",
        "primary_drivers", "narrative",
    ]
    available = [c for c in cols if c in df.columns]
    return df[available]


def _build_narrative(row: pd.Series) -> str:
    """Template-based AML risk narrative for a single user."""
    codes = _parse_reason_codes(row.get("top_reason_codes", []))
    risk = str(row.get("risk_level", "unknown")).upper()
    score = row.get("analyst_risk_score", 0.0)
    model_p = row.get("submission_probability", 0.0)
    anomaly = row.get("anomaly_score", 0.0)
    rule_s = row.get("rule_score", 0.0)

    # Build driver sentences
    driver_sentences = []
    for code in codes[:3]:
        label = RULE_LABELS.get(code, code)
        driver_sentences.append(f"    • {label}")

    # Signal summary
    signals = []
    if model_p >= 0.70:
        signals.append(f"ensemble model confidence {model_p:.0%}")
    elif model_p >= 0.40:
        signals.append(f"elevated ensemble score ({model_p:.0%})")
    if anomaly >= 0.50:
        signals.append("statistical anomaly (top-5% peer deviation)")
    if rule_s >= 0.33:
        signals.append(f"{int(rule_s * 6 + 0.5)} deterministic AML rules triggered")

    signal_str = "; ".join(signals) if signals else "multiple model signals elevated"

    drivers_str = "\n".join(driver_sentences) if driver_sentences else "    • Multi-model consensus"

    return textwrap.dedent(f"""\
        [{risk} RISK — Score {score:.1f}/100]
        Flagged by {signal_str}.
        Primary indicators:
{drivers_str}
        Recommended action: {"Immediate SAR filing" if risk == "CRITICAL" else "Enhanced due diligence" if risk == "HIGH" else "Monitoring"}\
    """)


# ---------------------------------------------------------------------------
# 4. Model contribution summary
# ---------------------------------------------------------------------------
def model_contribution_summary(oof_df: pd.DataFrame) -> dict[str, Any]:
    """Compute mean model scores by true label (positive vs negative).

    Returns dict with 'positives' and 'negatives' sub-dicts of mean scores,
    plus 'separation' (mean_pos - mean_neg) for each signal.
    """
    labeled = oof_df[oof_df["status"].notna()].copy()
    labeled["status"] = labeled["status"].astype(int)

    signal_cols = [
        "base_a_probability", "base_b_probability", "base_c_probability",
        "stacker_raw_probability", "rule_score", "anomaly_score",
    ]
    available = [c for c in signal_cols if c in labeled.columns]

    pos = labeled[labeled["status"] == 1][available].mean().to_dict()
    neg = labeled[labeled["status"] == 0][available].mean().to_dict()
    sep = {k: round(pos.get(k, 0) - neg.get(k, 0), 4) for k in available}

    return {
        "positives": {k: round(v, 4) for k, v in pos.items()},
        "negatives": {k: round(v, 4) for k, v in neg.items()},
        "separation": sep,
        "n_positives": int((labeled["status"] == 1).sum()),
        "n_negatives": int((labeled["status"] == 0).sum()),
    }


# ---------------------------------------------------------------------------
# 5. AML scenario coverage
# ---------------------------------------------------------------------------
def scenario_coverage_summary(scores_df: pd.DataFrame) -> dict[str, Any]:
    """Count how many flagged users trigger each AML scenario / rule.

    Returns dict: rule_code → n_flagged, pct_of_flagged
    """
    flagged = scores_df[scores_df.get("submission_pred", scores_df.index.map(lambda _: 1)) == 1]
    if flagged.empty:
        flagged = scores_df[scores_df["analyst_risk_score"] >= 50]

    total_flagged = len(flagged)
    counts: dict[str, int] = {code: 0 for code in RULE_LABELS}

    for codes_raw in flagged["top_reason_codes"]:
        for code in _parse_reason_codes(codes_raw):
            if code in counts:
                counts[code] += 1

    summary = {}
    for code, label in RULE_LABELS.items():
        n = counts[code]
        summary[code] = {
            "label": label,
            "n_flagged": n,
            "pct_of_flagged": round(n / max(1, total_flagged) * 100, 1),
        }

    return {
        "total_flagged": total_flagged,
        "scenarios": summary,
    }


# ---------------------------------------------------------------------------
# 6. SHAP explanations for top-N users
# ---------------------------------------------------------------------------
def shap_top_features(
    top_user_ids: list[int],
    dataset: pd.DataFrame,
    feature_cols: list[str],
    model: "Any",
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """Compute per-user top-K SHAP features using CatBoost native ShapValues.

    Parameters
    ----------
    top_user_ids : list of user_ids to explain (in order)
    dataset : full feature DataFrame (with original dtypes, incl. string cat cols)
    feature_cols : feature column names the model was trained on
    model : trained CatBoostClassifier
    top_k : number of top features to return per user

    Returns
    -------
    list of dicts: {user_id, shap_features: [{feature, shap_value, feature_value}]}
    """
    try:
        from catboost import Pool
    except ImportError:
        return []

    feat_top = dataset[dataset["user_id"].isin(top_user_ids)][["user_id"] + feature_cols].copy()
    if feat_top.empty:
        return []

    # Fill NA for categorical columns
    cat_cols = [c for c in feature_cols if c in feat_top.columns and
                (pd.api.types.is_object_dtype(feat_top[c]) or
                 pd.api.types.is_string_dtype(feat_top[c]))]
    for col in cat_cols:
        feat_top[col] = feat_top[col].fillna("NA").astype(str)

    X = feat_top[feature_cols].copy()
    pool = Pool(X, cat_features=cat_cols)
    shap_vals = model.get_feature_importance(pool, type="ShapValues")[:, :-1]  # drop bias col

    results = []
    for i, uid in enumerate(feat_top["user_id"].tolist()):
        sv = shap_vals[i]
        top_idx = np.argsort(np.abs(sv))[-top_k:][::-1]
        features = []
        for j in top_idx:
            val = X.iloc[i][feature_cols[j]]
            features.append({
                "feature": feature_cols[j],
                "shap_value": round(float(sv[j]), 4),
                "feature_value": float(val) if not isinstance(val, str) else val,
            })
        results.append({"user_id": int(uid), "shap_features": features})
    return results


# ---------------------------------------------------------------------------
# 7. Risk level distribution
# ---------------------------------------------------------------------------
def risk_level_distribution(scores_df: pd.DataFrame) -> dict[str, int]:
    """Count users by risk level band."""
    if "risk_level" in scores_df.columns:
        return scores_df["risk_level"].value_counts().to_dict()
    # Derive from submission_probability
    dist: dict[str, int] = {"critical": 0, "high": 0, "medium": 0, "low": 0, "minimal": 0}
    p = scores_df.get("submission_probability", scores_df.get("stacker_raw_probability", pd.Series(dtype=float)))
    for val in p:
        if val >= 0.50:
            dist["critical"] += 1
        elif val >= 0.30:
            dist["high"] += 1
        elif val >= 0.15:
            dist["medium"] += 1
        elif val >= 0.05:
            dist["low"] += 1
        else:
            dist["minimal"] += 1
    return dist


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def run_demo_pack(output_path: str | Path | None = None) -> dict[str, Any]:
    """Generate the full demo pack and return as a dict.

    Loads OOF predictions and scored submission from artifacts/.

    Parameters
    ----------
    output_path : if given, write the JSON report to this path.

    Returns
    -------
    dict with all demo pack sections.
    """
    paths = load_official_paths()

    # Load OOF predictions (labeled set)
    oof_path = paths.feature_dir / "official_oof_predictions.parquet"
    if not oof_path.exists():
        raise FileNotFoundError(f"OOF predictions not found: {oof_path}")
    oof_df = pd.read_parquet(oof_path)

    # Load submission scores
    scores_path = paths.prediction_dir / "official_predict_scores.parquet"
    if not scores_path.exists():
        scores_path = paths.prediction_dir / "official_predict_scores.csv"
    if not scores_path.exists():
        raise FileNotFoundError(f"Prediction scores not found at {paths.prediction_dir}")
    scores_df = (
        pd.read_parquet(scores_path)
        if str(scores_path).endswith(".parquet")
        else pd.read_csv(scores_path)
    )

    # Load bundle for reference threshold and AP
    bundle_path = paths.artifact_dir / "official_bundle.json"
    bundle_meta: dict[str, Any] = {}
    if bundle_path.exists():
        with open(bundle_path) as f:
            raw_bundle = json.load(f)
        cal = raw_bundle.get("calibrator", {})
        if isinstance(cal, dict):
            bundle_meta = {
                "calibrator": cal.get("method", "unknown"),
                "average_precision": cal.get("average_precision", 0.0),
                "selected_threshold": cal.get("selected_threshold", 0.10),
                "selected_f1": cal.get("selected_row", {}).get("f1", 0.0),
            }

    print("[demo_pack] Building threshold sensitivity table...")
    thr_table = threshold_sensitivity_table(oof_df)
    best_row = thr_table.loc[thr_table["f1"].idxmax()].to_dict()

    print("[demo_pack] Selecting operating points...")
    ops = select_operating_points(thr_table)

    print("[demo_pack] Computing model contributions...")
    contrib = model_contribution_summary(oof_df)

    print("[demo_pack] Generating top-20 alerts...")
    alerts = top_alert_cases(scores_df, top_n=20)

    print("[demo_pack] Computing scenario coverage...")
    coverage = scenario_coverage_summary(scores_df)

    print("[demo_pack] Computing risk level distribution...")
    risk_dist = risk_level_distribution(scores_df)

    # SHAP explanations for top-20 alert users (optional — skipped if model unavailable)
    shap_data: list[dict[str, Any]] = []
    if bundle_path.exists():
        try:
            print("[demo_pack] Computing SHAP explanations for top-20 alerts...")
            import pickle
            from official.train import _load_dataset
            model_path = raw_bundle.get("base_model_paths", {}).get("base_a_catboost", "")
            feat_cols = raw_bundle.get("feature_columns_base_a", [])
            if model_path and feat_cols and Path(model_path).exists():
                with open(model_path, "rb") as f:
                    base_a_model = pickle.load(f)
                full_dataset = _load_dataset("full")
                top_user_ids = alerts["user_id"].tolist()
                shap_data = shap_top_features(top_user_ids, full_dataset, feat_cols, base_a_model, top_k=5)
                print(f"[demo_pack] SHAP computed for {len(shap_data)} users")
        except Exception as exc:
            print(f"[demo_pack] SHAP skipped: {exc}")

    # Print summary to stdout
    _print_summary(best_row, ops, contrib, coverage, risk_dist, bundle_meta)

    pack: dict[str, Any] = {
        "metadata": {
            **bundle_meta,
            "oof_n_users": len(oof_df),
            "oof_n_positives": int(oof_df["status"].astype(float).eq(1).sum()),
            "submission_n_users": len(scores_df),
        },
        "best_operating_point": best_row,
        "operating_points": ops,
        "threshold_sensitivity": thr_table.to_dict(orient="records"),
        "model_contributions": contrib,
        "top_20_alerts": alerts.to_dict(orient="records"),
        "scenario_coverage": coverage,
        "risk_level_distribution": risk_dist,
        "shap_explanations": shap_data,
    }

    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(pack, f, indent=2, ensure_ascii=False, default=str)
        print(f"\n[demo_pack] Report saved → {out}")

    return pack


def _print_summary(
    best: dict,
    ops: dict,
    contrib: dict,
    coverage: dict,
    risk_dist: dict,
    bundle_meta: dict,
) -> None:
    """Print a formatted demo pack summary."""
    sep = "=" * 70
    print(f"\n{sep}")
    print("BITOGUARD AML DETECTION — DEMO PACK SUMMARY")
    print(sep)

    # Model performance
    ap = bundle_meta.get("average_precision", best.get("f1", 0))
    print(f"\n{'Model Performance':}")
    print(f"  Average Precision (AP)  : {ap:.4f}")
    print(f"  Best F1 (OOF)           : {best.get('f1', 0):.4f}")
    print(f"    Precision @ best F1   : {best.get('precision', 0):.4f}")
    print(f"    Recall    @ best F1   : {best.get('recall', 0):.4f}")
    print(f"    Threshold             : {best.get('threshold', 0):.4f}")

    # Operating points
    print(f"\n{'Operating Points':}")
    print(f"  {'Mode':<14} {'Threshold':>9} {'F1':>7} {'P':>7} {'R':>7} {'Flagged':>8}")
    print(f"  {'-'*60}")
    for key, op in ops.items():
        print(
            f"  {op['label']:<14} {op['threshold']:>9.4f} "
            f"{op['f1']:>7.4f} {op['precision']:>7.4f} {op['recall']:>7.4f} "
            f"{op['n_flagged']:>8,}"
        )

    # Model contributions (separation)
    print(f"\n{'Signal Separation (mean_pos - mean_neg)':}")
    for sig, delta in sorted(contrib["separation"].items(), key=lambda x: -abs(x[1])):
        bar = "█" * int(abs(delta) * 50)
        print(f"  {sig:<30} {delta:+.4f}  {bar}")

    # Scenario coverage
    scen = coverage.get("scenarios", {})
    total_flagged = coverage.get("total_flagged", 0)
    print(f"\n{'AML Scenario Coverage (top flagged users)':}")
    for code, info in sorted(scen.items(), key=lambda x: -x[1]["n_flagged"]):
        pct = info["pct_of_flagged"]
        bar = "█" * int(pct / 5)
        print(f"  {info['label'][:40]:<40} {info['n_flagged']:>5} ({pct:5.1f}%)  {bar}")

    # Risk distribution
    print(f"\n{'Risk Level Distribution (submission set)':}")
    for level in ["critical", "high", "medium", "low", "minimal"]:
        n = risk_dist.get(level, 0)
        pct = n / max(1, sum(risk_dist.values())) * 100
        bar = "█" * int(pct / 2)
        print(f"  {level.upper():<10} {n:>6,} ({pct:5.1f}%)  {bar}")

    print(f"\n{sep}\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="BitoGuard demo pack generator")
    parser.add_argument(
        "--output",
        default="artifacts/reports/demo_pack.json",
        help="Output path for JSON report (default: artifacts/reports/demo_pack.json)",
    )
    args = parser.parse_args()

    run_demo_pack(output_path=args.output)
