from __future__ import annotations

from pathlib import Path

import pandas as pd

from official.bundle import load_selected_bundle
from official.common import encode_frame, load_clean_table, load_official_paths, load_pickle
from official.train import _load_dataset


def _load_bundle_model(bundle: dict) -> object:
    if bundle["selected_model"] != "lgbm":
        raise NotImplementedError(f"Selected model not yet supported: {bundle['selected_model']}")
    model_path = Path(bundle["model_paths"]["lgbm"])
    return load_pickle(model_path)


def score_official_predict() -> pd.DataFrame:
    bundle = load_selected_bundle(require_ready=True)
    dataset = _load_dataset("full")
    model = _load_bundle_model(bundle)
    calibrator = load_pickle(Path(bundle["calibrator"]["calibrator_path"]))

    predict_users = set(load_clean_table("predict_label")["user_id"].astype(int).tolist())
    scoring = dataset[dataset["user_id"].astype(int).isin(predict_users)].copy()
    encoded, _ = encode_frame(scoring, bundle["feature_columns_lgbm"], reference_columns=bundle["encoded_columns_lgbm"])
    scoring["model_probability_raw"] = model.predict_proba(encoded)[:, 1]
    scoring["submission_probability"] = calibrator.predict(scoring["model_probability_raw"].to_numpy())
    scoring["submission_pred"] = (scoring["submission_probability"] >= float(bundle["selected_threshold"])).astype(int)
    scoring["analyst_risk_score"] = (
        0.80 * scoring["submission_probability"]
        + 0.12 * scoring["anomaly_score"]
        + 0.08 * scoring["rule_score"]
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
        "model_probability_raw",
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
