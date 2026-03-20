"""Experiment tracking: logs every run with config, metrics, and artifacts.

Each experiment gets a unique ID and is logged to a JSON-lines file.
Supports ablation comparisons and metric aggregation.

P0-3: Every experiment includes cohort_metrics with dormant/active breakdown.
Dormant users have zero events across all behavioral tables.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from official.common import load_official_paths

EXPERIMENT_LOG = "experiment_log.jsonl"

# P0-3: Cached set of active user IDs (populated lazily).
_active_user_ids_cache: set[int] | None = None


def _get_active_user_ids() -> set[int]:
    """Return IDs of users with at least one behavioral event (lazy-loaded, cached)."""
    global _active_user_ids_cache
    if _active_user_ids_cache is not None:
        return _active_user_ids_cache
    from official.common import load_clean_table
    active: set[int] = set()
    for table_name in ("twd_transfer", "crypto_transfer", "usdt_swap", "usdt_twd_trading"):
        try:
            t = load_clean_table(table_name)
            if "user_id" in t.columns:
                active.update(t["user_id"].dropna().astype(int).unique().tolist())
        except Exception:
            pass
    _active_user_ids_cache = active
    return active


def compute_cohort_metrics(
    oof_frame: pd.DataFrame,
    threshold: float,
    prob_col: str = "stacker_raw_probability",
) -> dict[str, Any]:
    """P0-3: Compute F1/P/R breakdown for dormant and active user cohorts.

    Dormant = user has zero rows in all behavioral event tables.
    Active  = user has ≥1 row in at least one event table.

    Args:
        oof_frame: OOF evaluation frame with 'user_id', 'status', and prob_col.
        threshold: Decision threshold for binary classification.
        prob_col: Column containing predicted probabilities.

    Returns:
        dict with keys 'all', 'dormant', 'active', each containing:
            f1, precision, recall, n_users, n_pos
    """
    from sklearn.metrics import f1_score, precision_score, recall_score

    if oof_frame.empty or prob_col not in oof_frame.columns:
        return {}

    active_ids = _get_active_user_ids()
    frame = oof_frame.copy()
    frame["_is_dormant"] = ~frame["user_id"].astype(int).isin(active_ids)

    results: dict[str, Any] = {}
    for cohort_name, mask in [
        ("all", pd.Series(True, index=frame.index)),
        ("dormant", frame["_is_dormant"]),
        ("active", ~frame["_is_dormant"]),
    ]:
        sub = frame[mask]
        if len(sub) == 0:
            results[cohort_name] = {"f1": 0.0, "precision": 0.0, "recall": 0.0, "n_users": 0, "n_pos": 0}
            continue
        labels = sub["status"].astype(int).values
        n_pos = int(labels.sum())
        if n_pos == 0:
            results[cohort_name] = {"f1": 0.0, "precision": 0.0, "recall": 0.0, "n_users": len(sub), "n_pos": 0}
            continue
        preds = (pd.to_numeric(sub[prob_col], errors="coerce").fillna(0.0).values >= threshold).astype(int)
        results[cohort_name] = {
            "f1": float(f1_score(labels, preds, zero_division=0)),
            "precision": float(precision_score(labels, preds, zero_division=0)),
            "recall": float(recall_score(labels, preds, zero_division=0)),
            "n_users": int(len(sub)),
            "n_pos": n_pos,
        }
    return results


def _log_path() -> Path:
    paths = load_official_paths()
    paths.report_dir.mkdir(parents=True, exist_ok=True)
    return paths.report_dir / EXPERIMENT_LOG


def log_experiment(
    experiment_id: str,
    config: dict[str, Any],
    metrics: dict[str, float],
    notes: str = "",
    artifacts: list[str] | None = None,
    cohort_metrics: dict[str, Any] | None = None,
) -> None:
    entry = {
        "experiment_id": experiment_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "metrics": metrics,
        "notes": notes,
        "artifacts": artifacts or [],
    }
    if cohort_metrics:
        entry["cohort_metrics"] = cohort_metrics
    path = _log_path()
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    f1 = metrics.get("f1", metrics.get("oof_f1", "N/A"))
    f1_str = f"{f1:.4f}" if isinstance(f1, float) else str(f1)
    # P0-3: Print cohort breakdown if available
    cohort_str = ""
    if cohort_metrics:
        d_f1 = cohort_metrics.get("dormant", {}).get("f1", 0.0)
        a_f1 = cohort_metrics.get("active", {}).get("f1", 0.0)
        cohort_str = f" [dormant={d_f1:.4f}, active={a_f1:.4f}]"
    print(f"[EXP] {experiment_id}: F1={f1_str}{cohort_str} | {notes}")


def load_experiments() -> list[dict[str, Any]]:
    path = _log_path()
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def get_experiment(experiment_id: str) -> dict[str, Any] | None:
    for exp in load_experiments():
        if exp["experiment_id"] == experiment_id:
            return exp
    return None


def print_leaderboard(top_n: int = 20) -> None:
    experiments = load_experiments()
    if not experiments:
        print("No experiments logged yet.")
        return
    sorted_exps = sorted(
        experiments,
        key=lambda e: e["metrics"].get("f1", e["metrics"].get("oof_f1", 0)),
        reverse=True,
    )
    has_cohort = any("cohort_metrics" in e for e in sorted_exps[:top_n])
    if has_cohort:
        print(f"\n{'='*110}")
        print(f"{'Rank':<5} {'Experiment ID':<42} {'F1':>8} {'P':>8} {'R':>8} {'AP':>8} {'Thr':>8} {'Dorm-F1':>9} {'Act-F1':>8}")
        print(f"{'='*110}")
    else:
        print(f"\n{'='*90}")
        print(f"{'Rank':<5} {'Experiment ID':<45} {'F1':>8} {'P':>8} {'R':>8} {'AP':>8} {'Thr':>8}")
        print(f"{'='*90}")
    for i, exp in enumerate(sorted_exps[:top_n], 1):
        m = exp["metrics"]
        f1 = m.get("f1", m.get("oof_f1", 0))
        row = (
            f"{i:<5} {exp['experiment_id']:<42} "
            f"{f1:>8.4f} {m.get('precision', 0):>8.4f} "
            f"{m.get('recall', 0):>8.4f} {m.get('pr_auc', 0):>8.4f} "
            f"{m.get('threshold', 0):>8.4f}"
        )
        if has_cohort:
            cm = exp.get("cohort_metrics", {})
            d_f1 = cm.get("dormant", {}).get("f1", float("nan"))
            a_f1 = cm.get("active", {}).get("f1", float("nan"))
            row += f" {d_f1:>9.4f} {a_f1:>8.4f}"
        print(row)
    w = 110 if has_cohort else 90
    print(f"{'='*w}\n")
    best = sorted_exps[0]
    print(f"Best: {best['experiment_id']} — F1={best['metrics'].get('f1', best['metrics'].get('oof_f1', 0)):.4f}")
    print(f"  Config: {best['config']}\n")
