from __future__ import annotations

from pathlib import Path

import pandas as pd

from official.bundle import load_selected_bundle
from official.common import load_clean_table, load_official_paths, load_pickle
from official.graph_dataset import build_transductive_graph
from official.graph_model import load_graph_model, predict_graph_model
from official.stacking import STACKER_FEATURE_COLUMNS
from official.train import _label_frame, _load_dataset, _prepare_base_frames
from official.transductive_features import build_transductive_feature_frame


def score_official_predict() -> pd.DataFrame:
    bundle = load_selected_bundle(require_ready=True)
    dataset = _load_dataset("full")
    graph = build_transductive_graph(dataset)
    full_label_frame = _label_frame(dataset)
    transductive_features = build_transductive_feature_frame(graph, full_label_frame)
    label_free_frame, with_transductive_frame = _prepare_base_frames(dataset, transductive_features)

    predict_users = set(load_clean_table("predict_label")["user_id"].astype(int).tolist())
    scoring_label_free = label_free_frame[label_free_frame["user_id"].astype(int).isin(predict_users)].copy()
    scoring_transductive = with_transductive_frame[with_transductive_frame["user_id"].astype(int).isin(predict_users)].copy()

    base_a_model = load_pickle(Path(bundle["base_model_paths"]["base_a_catboost"]))
    base_b_model = load_pickle(Path(bundle["base_model_paths"]["base_b_catboost"]))
    graph_model_state = load_graph_model(Path(bundle["graph_model_path"]))
    stacker_model = load_pickle(Path(bundle["stacker_path"]))
    calibrator = load_pickle(Path(bundle["calibrator"]["calibrator_path"]))

    base_a_probability = base_a_model.predict_proba(scoring_label_free[bundle["feature_columns_base_a"]])[:, 1]
    base_b_probability = base_b_model.predict_proba(scoring_transductive[bundle["feature_columns_base_b"]])[:, 1]
    graph_probability_frame = predict_graph_model(graph, graph_model_state)
    scoring = scoring_label_free.merge(graph_probability_frame, on="user_id", how="left")
    scoring["base_a_probability"] = base_a_probability
    scoring["base_b_probability"] = base_b_probability
    scoring["base_c_probability"] = scoring["graph_probability"].fillna(0.0)
    scoring["stacker_raw_probability"] = stacker_model.predict_proba(scoring[STACKER_FEATURE_COLUMNS])[:, 1]
    scoring["submission_probability"] = calibrator.predict(scoring["stacker_raw_probability"].to_numpy())
    scoring["submission_pred"] = (scoring["submission_probability"] >= float(bundle["selected_threshold"])).astype(int)
    scoring["analyst_risk_score"] = (
        0.72 * scoring["submission_probability"]
        + 0.16 * scoring["anomaly_score"]
        + 0.12 * scoring["rule_score"]
    ) * 100.0
    scoring["risk_rank"] = scoring["analyst_risk_score"].rank(method="first", ascending=False).astype(int)
    scoring["risk_level"] = pd.cut(
        scoring["analyst_risk_score"],
        bins=[-1, 35, 60, 80, 100],
        labels=["low", "medium", "high", "critical"],
    ).astype(str)
    output = scoring[[
        "user_id",
        "submission_probability",
        "submission_pred",
        "stacker_raw_probability",
        "base_a_probability",
        "base_b_probability",
        "base_c_probability",
        "anomaly_score",
        "rule_score",
        "analyst_risk_score",
        "risk_rank",
        "risk_level",
        "top_reason_codes",
        "is_shadow_overlap",
    ]].sort_values("risk_rank")
    paths = load_official_paths()
    output.to_parquet(paths.prediction_dir / "official_predict_scores.parquet", index=False)
    output.to_csv(paths.prediction_dir / "official_predict_scores.csv", index=False)
    return output


def main() -> None:
    print(score_official_predict().head())


if __name__ == "__main__":
    main()
