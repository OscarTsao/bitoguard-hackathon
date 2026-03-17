from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from transductive_v1.common import bundle_path, feature_path, load_pickle, prediction_path
from transductive_v1.decision_rule import apply_rule


def score_transductive_v1(cutoff_tag: str = "full") -> pd.DataFrame:
    bundle = json.loads(bundle_path().read_text(encoding="utf-8"))
    if not bundle.get("calibrator") or not bundle.get("decision_rule"):
        raise ValueError("Bundle is not ready for scoring")
    scoring = pd.read_parquet(feature_path("full_scoring_frame", cutoff_tag))
    stacker_model = load_pickle(Path(bundle["stacker_path"]))
    calibrator = load_pickle(Path(bundle["calibrator"]["calibrator_path"]))
    predict_frame = scoring[scoring["needs_prediction"].eq(True)].copy()
    stacker_columns = bundle["stacker_feature_columns"]
    predict_frame["stacker_raw_probability"] = stacker_model.predict_proba(predict_frame[stacker_columns])[:, 1]
    predict_frame["submission_probability"] = calibrator.predict(predict_frame["stacker_raw_probability"].to_numpy())
    predict_frame["submission_pred"] = apply_rule(predict_frame["submission_probability"].to_numpy(), bundle["decision_rule"])
    predict_frame["analyst_risk_score"] = (
        0.76 * predict_frame["submission_probability"]
        + 0.14 * predict_frame["anomaly_score"]
        + 0.10 * predict_frame["rule_score"]
    ) * 100.0
    predict_frame["risk_rank"] = predict_frame["analyst_risk_score"].rank(method="first", ascending=False).astype(int)
    predict_frame["risk_level"] = pd.cut(
        predict_frame["analyst_risk_score"],
        bins=[-1, 35, 60, 80, 100],
        labels=["low", "medium", "high", "critical"],
    ).astype(str)
    output = predict_frame[[
        "user_id",
        "submission_probability",
        "submission_pred",
        "stacker_raw_probability",
        "base_a_probability",
        "base_b_probability",
        "graph_risk_score",
        "rule_score",
        "anomaly_score",
        "analyst_risk_score",
        "risk_rank",
        "risk_level",
        "top_reason_codes",
    ]].sort_values("risk_rank")
    output.to_parquet(prediction_path("predict_scores.parquet"), index=False)
    output.to_csv(prediction_path("predict_scores.csv"), index=False)
    return output
