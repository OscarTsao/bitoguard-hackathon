from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from official.bundle import load_selected_bundle
from official.common import encode_frame, load_clean_table, load_official_paths, load_pickle
from official.correct_and_smooth import correct_and_smooth
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

    _base_a_seed_paths = bundle["base_model_paths"].get("base_a_catboost_seeds")
    if _base_a_seed_paths:
        base_a_models = [load_pickle(Path(p)) for p in _base_a_seed_paths]
    else:
        base_a_models = [load_pickle(Path(bundle["base_model_paths"]["base_a_catboost"]))]
    base_b_model = load_pickle(Path(bundle["base_model_paths"]["base_b_catboost"]))
    graph_model_state = load_graph_model(Path(bundle["graph_model_path"]))
    stacker_model = load_pickle(Path(bundle["stacker_path"]))
    calibrator = load_pickle(Path(bundle["calibrator"]["calibrator_path"]))
    # v42: Load multi-seed models for Base D and E. Falls back to single model if seeds not in bundle.
    base_d_models: list = []
    _base_d_seed_paths = bundle.get("base_model_paths", {}).get("base_d_lgbm_seeds")
    if _base_d_seed_paths:
        base_d_models = [load_pickle(Path(p)) for p in _base_d_seed_paths]
    elif "base_d_lgbm" in bundle.get("base_model_paths", {}):
        base_d_models = [load_pickle(Path(bundle["base_model_paths"]["base_d_lgbm"]))]
    base_e_models: list = []
    _base_e_seed_paths = bundle.get("base_model_paths", {}).get("base_e_xgboost_seeds")
    if _base_e_seed_paths:
        base_e_models = [load_pickle(Path(p)) for p in _base_e_seed_paths]
    elif "base_e_xgboost" in bundle.get("base_model_paths", {}):
        base_e_models = [load_pickle(Path(bundle["base_model_paths"]["base_e_xgboost"]))]
    # If the bundle specifies blend_weights, reconstruct BlendEnsemble over the
    # pickled stacker (BlendEnsemble is stateless — weights are stored in bundle).
    if bundle.get("blend_weights"):
        from official.stacking import BlendEnsemble
        stacker_model = BlendEnsemble(bundle["blend_weights"])

    import numpy as _np
    # Base A: predict on scoring users only (for stacker input)
    base_a_probability = _np.mean(
        [m.predict_proba(scoring_label_free[bundle["feature_columns_base_a"]])[:, 1] for m in base_a_models],
        axis=0,
    )
    # C&S: apply graph correction to Base A probabilities.
    # Uses ALL labeled users as training seeds (no leakage — scoring users have no labels).
    # All labeled users' Base A predictions are computed once here for C&S propagation.
    _all_labeled_ids = label_free_frame["user_id"].astype(int).tolist()
    _all_a_probs = _np.mean(
        [m.predict_proba(label_free_frame[bundle["feature_columns_base_a"]])[:, 1] for m in base_a_models],
        axis=0,
    )
    _cs_base_probs: dict[int, float] = {
        int(_uid): float(_p) for _uid, _p in zip(_all_labeled_ids, _all_a_probs)
    }
    # Add scoring users to base_probs (they override labeled users if overlapping)
    for _uid, _p in zip(scoring_label_free["user_id"].astype(int), base_a_probability):
        _cs_base_probs[int(_uid)] = float(_p)
    _cs_train_labels: dict[int, float] = dict(zip(
        full_label_frame["user_id"].astype(int),
        full_label_frame["status"].astype(float),
    ))
    _cs_result = correct_and_smooth(
        graph, _cs_train_labels, _cs_base_probs,
        alpha_correct=0.5, alpha_smooth=0.5,
        n_correct_iter=50, n_smooth_iter=50,
    )
    _scoring_ids = scoring_label_free["user_id"].astype(int).tolist()
    base_c_s_probability = _np.array(
        [_cs_result.get(int(_uid), float(_p)) for _uid, _p in zip(_scoring_ids, base_a_probability)],
        dtype=float,
    )

    base_b_probability = base_b_model.predict_proba(scoring_transductive[bundle["feature_columns_base_b"]])[:, 1]
    graph_probability_frame = predict_graph_model(graph, graph_model_state)
    scoring = scoring_label_free.merge(graph_probability_frame, on="user_id", how="left")
    scoring["base_a_probability"] = base_a_probability
    scoring["base_c_s_probability"] = base_c_s_probability
    scoring["base_b_probability"] = base_b_probability
    scoring["base_c_probability"] = scoring["graph_probability"].fillna(0.0)
    if base_d_models:
        x_d, _ = encode_frame(
            scoring_label_free,
            bundle["feature_columns_base_d"],
            reference_columns=bundle.get("encoded_columns_base_d"),
        )
        # v42: Average predictions across all seed models for variance reduction.
        scoring["base_d_probability"] = _np.mean(
            [m.predict_proba(x_d)[:, 1] for m in base_d_models], axis=0
        )
    else:
        scoring["base_d_probability"] = 0.0
    if base_e_models:
        x_e, _ = encode_frame(
            scoring_label_free,
            bundle.get("feature_columns_base_e", bundle["feature_columns_base_a"]),
            reference_columns=bundle.get("encoded_columns_base_e"),
        )
        # v42: Average predictions across all seed models for variance reduction.
        scoring["base_e_probability"] = _np.mean(
            [m.predict_proba(x_e)[:, 1] for m in base_e_models], axis=0
        )
    else:
        scoring["base_e_probability"] = 0.0
    # Anomaly subscores (LOF, OCSVM) — present in the dataset if anomaly module produced them.
    scoring["lof_score"] = (
        pd.to_numeric(scoring_label_free["lof_score"], errors="coerce").fillna(0.0).to_numpy()
        if "lof_score" in scoring_label_free.columns else np.zeros(len(scoring))
    )
    scoring["ocsvm_score"] = (
        pd.to_numeric(scoring_label_free["ocsvm_score"], errors="coerce").fillna(0.0).to_numpy()
        if "ocsvm_score" in scoring_label_free.columns else np.zeros(len(scoring))
    )
    # Individual AML rule binary flags for the non-linear stacker.
    # Rules require columns from the full dataset (graph features like shared_ip_user_count).
    from official.rules import evaluate_official_rules, RULE_DEFINITIONS
    rule_frame = evaluate_official_rules(dataset)
    scoring = scoring.merge(rule_frame[["user_id"] + list(RULE_DEFINITIONS)], on="user_id", how="left")
    for _rule in RULE_DEFINITIONS:
        if _rule in scoring.columns:
            scoring[_rule] = scoring[_rule].fillna(False).astype(bool)
    # Enrich with base-probability meta-features used by v29+ stacker.
    try:
        from official.stacking import _add_base_meta_features
        scoring = _add_base_meta_features(scoring)
    except (ImportError, AttributeError):
        pass  # Backward compat: older stacking.py without meta-feature support
    stacker_cols = bundle.get("stacker_feature_columns", STACKER_FEATURE_COLUMNS)
    # Only use columns that actually exist in scoring (graceful backward compat).
    stacker_cols = [c for c in stacker_cols if c in scoring.columns]
    # Convert bool columns to int for stacker inference compatibility.
    scoring_stacker = scoring[stacker_cols].copy()
    for c in stacker_cols:
        if c in scoring_stacker.columns and (scoring_stacker[c].dtype == bool or str(scoring_stacker[c].dtype) == "bool"):
            scoring_stacker[c] = scoring_stacker[c].astype(int)
    scoring["stacker_raw_probability"] = stacker_model.predict_proba(scoring_stacker)[:, 1]
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
        "base_d_probability",
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
