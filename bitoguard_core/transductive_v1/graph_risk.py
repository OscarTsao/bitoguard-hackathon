from __future__ import annotations

import numpy as np
import pandas as pd


GRAPH_RISK_COMPONENTS = (
    "direct_positive_relation_count",
    "wallet_positive_entity_sum",
    "ip_positive_entity_sum",
    "positive_exposure_1hop_count",
    "positive_exposure_2hop_count",
    "positive_seed_propagation",
    "positive_seed_walk2",
    "positive_seed_harmonic_distance",
)


def build_graph_risk_features(label_aware_frame: pd.DataFrame) -> pd.DataFrame:
    result = label_aware_frame.copy()
    scores = []
    for column in GRAPH_RISK_COMPONENTS:
        if column not in result.columns:
            continue
        series = pd.to_numeric(result[column], errors="coerce").fillna(0.0)
        scores.append(series.rank(method="average", pct=True))
    if scores:
        stacked = pd.concat(scores, axis=1)
        result["graph_risk_score"] = stacked.mean(axis=1).fillna(0.0)
    else:
        result["graph_risk_score"] = 0.0
    return result
