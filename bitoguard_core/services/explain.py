from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from config import load_settings
from models.common import encode_features, feature_columns, load_feature_table, load_pickle


def explain_user(user_id: str) -> list[dict]:
    import shap

    settings = load_settings()
    features = load_feature_table("features.feature_snapshots_user_day")
    latest_date = features["snapshot_date"].max()
    frame = features[(features["snapshot_date"] == latest_date) & (features["user_id"] == user_id)].copy()
    if frame.empty:
        return []

    model_files = sorted((settings.artifact_dir / "models").glob("lgbm_*.pkl"))
    if not model_files:
        return []
    model_path = model_files[-1]
    meta_path = model_path.with_suffix(".json")
    if not meta_path.exists():
        return []
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    model = load_pickle(model_path)
    cols = feature_columns(frame)
    encoded, encoded_columns = encode_features(frame, cols, reference_columns=meta["encoded_columns"])
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(encoded)
    if isinstance(shap_values, list):
        shap_vector = shap_values[1][0]
    else:
        shap_vector = shap_values[0]
    factors = pd.DataFrame({
        "feature": encoded_columns,
        "value": encoded.iloc[0].tolist(),
        "impact": shap_vector,
    })
    factors["abs_impact"] = factors["impact"].abs()
    top = factors.sort_values("abs_impact", ascending=False).head(10)
    return top[["feature", "value", "impact"]].to_dict(orient="records")
