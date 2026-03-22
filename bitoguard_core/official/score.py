from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from official.bundle import load_selected_bundle
from official.common import load_clean_table, load_official_paths, load_pickle
from official.correct_and_smooth import correct_and_smooth
from official.graph_dataset import build_transductive_graph
from official.graph_model import load_graph_model, predict_graph_model
from official.stacking import STACKER_FEATURE_COLUMNS, _add_base_meta_features
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

    # Load models — multi-seed for Base A/D/E, single for B
    base_a_paths = bundle["base_model_paths"].get("base_a_catboost_seeds") or [bundle["base_model_paths"]["base_a_catboost"]]
    base_a_models = [load_pickle(Path(p)) for p in base_a_paths]
    base_b_model = load_pickle(Path(bundle["base_model_paths"]["base_b_catboost"]))
    _skip_gnn = __import__("os").environ.get("SKIP_GNN", "0") == "1"
    _graph_path = Path(bundle["graph_model_path"])
    if _skip_gnn or not _graph_path.exists():
        graph_model_state = None
    else:
        graph_model_state = load_graph_model(_graph_path)
    stacker_model = load_pickle(Path(bundle["stacker_path"]))
    calibrator = load_pickle(Path(bundle["calibrator"]["calibrator_path"]))

    base_a_cols = bundle["feature_columns_base_a"]
    base_b_cols = bundle["feature_columns_base_b"]

    # Base A: average over seeds
    base_a_probability = np.mean(
        [m.predict_proba(scoring_label_free[base_a_cols])[:, 1] for m in base_a_models],
        axis=0,
    )

    # Base B: transductive CatBoost
    base_b_probability = base_b_model.predict_proba(scoring_transductive[base_b_cols])[:, 1]

    # Base C: GraphSAGE
    if graph_model_state is not None:
        graph_probability_frame = predict_graph_model(graph, graph_model_state)
    else:
        graph_probability_frame = pd.DataFrame({"user_id": scoring_label_free["user_id"], "graph_probability": 0.0})

    # Base D: LightGBM multi-seed
    base_d_paths = bundle["base_model_paths"].get("base_d_lgbm_seeds") or [bundle["base_model_paths"].get("base_d_lgbm")]
    base_d_paths = [p for p in base_d_paths if p]
    if base_d_paths:
        from official.common import encode_frame
        base_d_cols = bundle.get("feature_columns_base_d", base_a_cols)
        enc_cols = bundle.get("encoded_columns_base_d") or None
        base_d_models = [load_pickle(Path(p)) for p in base_d_paths]
        x_score_d, _ = encode_frame(scoring_label_free, base_d_cols, reference_columns=enc_cols)
        base_d_probability = np.mean(
            [m.predict_proba(x_score_d)[:, 1] for m in base_d_models],
            axis=0,
        )
    else:
        base_d_probability = np.zeros(len(scoring_label_free))

    # Base E: XGBoost multi-seed
    base_e_paths = bundle["base_model_paths"].get("base_e_xgboost_seeds") or [bundle["base_model_paths"].get("base_e_xgboost")]
    base_e_paths = [p for p in base_e_paths if p]
    if base_e_paths:
        from official.common import encode_frame
        base_e_cols = bundle.get("feature_columns_base_e", base_a_cols)
        enc_cols_e = bundle.get("encoded_columns_base_e") or None
        base_e_models = [load_pickle(Path(p)) for p in base_e_paths]
        x_score_e, _ = encode_frame(scoring_label_free, base_e_cols, reference_columns=enc_cols_e)
        base_e_probability = np.mean(
            [m.predict_proba(x_score_e)[:, 1] for m in base_e_models],
            axis=0,
        )
    else:
        base_e_probability = np.zeros(len(scoring_label_free))

    # C&S post-processing on Base A predictions
    # Use all labeled users' labels as train_labels (no leakage — scoring users are never labeled)
    _cs_base_probs: dict[int, float] = {}
    for _uid, _prob in zip(label_free_frame["user_id"].astype(int), np.mean(
        [m.predict_proba(label_free_frame[base_a_cols])[:, 1] for m in base_a_models], axis=0
    )):
        _cs_base_probs[int(_uid)] = float(_prob)
    _cs_train_labels: dict[int, float] = dict(zip(
        full_label_frame["user_id"].astype(int),
        full_label_frame["status"].astype(float),
    ))
    _cs_result = correct_and_smooth(
        graph, _cs_train_labels, _cs_base_probs,
        alpha_correct=0.5, alpha_smooth=0.5,
        n_correct_iter=50, n_smooth_iter=50,
    )
    base_c_s_probability = np.array(
        [_cs_result.get(int(uid), float(p)) for uid, p in zip(
            scoring_label_free["user_id"].astype(int), base_a_probability
        )],
        dtype=float,
    )

    scoring = scoring_label_free.merge(graph_probability_frame, on="user_id", how="left")
    scoring["base_a_probability"] = base_a_probability
    scoring["base_c_s_probability"] = base_c_s_probability
    scoring["base_b_probability"] = base_b_probability
    scoring["base_c_probability"] = scoring["graph_probability"].fillna(0.0)
    scoring["base_d_probability"] = base_d_probability
    scoring["base_e_probability"] = base_e_probability

    # Add interaction features
    scoring = _add_base_meta_features(scoring)

    available_cols = [c for c in STACKER_FEATURE_COLUMNS if c in scoring.columns]
    scoring["stacker_raw_probability"] = stacker_model.predict_proba(scoring[available_cols])[:, 1]
    scoring["submission_probability"] = calibrator.predict(scoring["stacker_raw_probability"].to_numpy())
    scoring["submission_pred"] = (scoring["submission_probability"] >= float(bundle["selected_threshold"])).astype(int)
    scoring["analyst_risk_score"] = (
        0.72 * scoring["submission_probability"]
        + 0.16 * scoring["anomaly_score"].fillna(0.0)
        + 0.12 * scoring["rule_score"].fillna(0.0)
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
