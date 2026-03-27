from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from official import nested_hpo as nh
from official.calibration import IsotonicCalibrator
from official.common import load_clean_table, load_official_paths, save_json, save_pickle
from official.correct_and_smooth import correct_and_smooth
from official.graph_dataset import build_transductive_graph
from official.graph_model import save_graph_model, train_graphsage_model
from official.modeling_xgb import fit_xgboost
from official.stacking import BlendEnsemble, STACKER_FEATURE_COLUMNS, _add_base_meta_features, tune_blend_weights
from official.train import PRIMARY_GRAPH_MAX_EPOCHS, _label_frame, _label_free_feature_columns, _load_dataset
from official.transductive_features import build_transductive_feature_frame


ROOT = Path(__file__).resolve().parent
MAINLINE_DIR = ROOT / "artifacts" / "official_features" / "hybrid_primary_locked_eval_with_graphsage"
MAINLINE_REPORT = MAINLINE_DIR / "hybrid_primary_locked_eval_with_graphsage_report.json"
MAINLINE_OOF = MAINLINE_DIR / "hybrid_primary_locked_eval_with_graphsage_oof.parquet"
OUT_DIR = ROOT / "artifacts" / "final_submission_current_mainline"
MODELS_DIR = OUT_DIR / "models"
OUT_CSV = OUT_DIR / "submission_provisional_graphsage_mainline.csv"
OUT_JSON = OUT_DIR / "submission_provisional_graphsage_mainline.json"
OUT_MD = OUT_DIR / "submission_provisional_graphsage_mainline.md"
OUT_SCORES = OUT_DIR / "submission_provisional_graphsage_mainline_scores.parquet"


def _git(cmd: list[str]) -> str:
    return subprocess.check_output(cmd, cwd=ROOT.parent, text=True).strip()


def _require_mainline_sources() -> tuple[dict, pd.DataFrame]:
    missing = [str(p) for p in (MAINLINE_REPORT, MAINLINE_OOF) if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing mainline sources: " + ", ".join(missing))
    report = json.loads(MAINLINE_REPORT.read_text(encoding="utf-8"))
    oof = pd.read_parquet(MAINLINE_OOF)
    return report, oof


def _train_final_models(dataset: pd.DataFrame, report: dict) -> dict[str, object]:
    graph = build_transductive_graph(dataset)
    label_frame = _label_frame(dataset)
    feature_columns = _label_free_feature_columns(dataset)
    trans_df = build_transductive_feature_frame(graph, label_frame)
    trans_cols = [c for c in trans_df.columns if c != "user_id"]

    label_free_frame = dataset.copy()
    with_transductive_frame = dataset.merge(trans_df, on="user_id", how="left")
    with_transductive_frame[trans_cols] = with_transductive_frame[trans_cols].fillna(0.0)

    labeled_user_ids = set(label_frame["user_id"].astype(int).tolist())
    predict_user_ids = set(load_clean_table("predict_label")["user_id"].astype(int).tolist())

    labeled_lf = label_free_frame[label_free_frame["user_id"].astype(int).isin(labeled_user_ids)].copy()
    labeled_td = with_transductive_frame[with_transductive_frame["user_id"].astype(int).isin(labeled_user_ids)].copy()
    predict_lf = label_free_frame[label_free_frame["user_id"].astype(int).isin(predict_user_ids)].copy()
    predict_td = with_transductive_frame[with_transductive_frame["user_id"].astype(int).isin(predict_user_ids)].copy()

    cb_params = dict(report["frozen_catboost_params"])
    lgbm_params = dict(report["frozen_lightgbm_params"])
    xgb_params = dict(report["frozen_xgboost_params"])

    base_a_fits = [
        nh._fit_catboost_with_params(
            labeled_lf,
            None,
            feature_columns,
            dict(cb_params),
            seed,
            thread_count=nh.GPU_STREAM_CPU_THREADS,
        )
        for seed in nh.FINAL_SEEDS_A
    ]
    base_b_fit = nh._fit_catboost_with_params(
        labeled_td,
        None,
        feature_columns + trans_cols,
        {"task_type": "CPU", "l2_leaf_reg": 5.0},
        nh.HPO_SEED,
        force_cpu=True,
        thread_count=nh.CPU_STREAM_THREADS,
    )
    base_d_fits = [
        nh._fit_lgbm_cpu(labeled_lf, None, feature_columns, dict(lgbm_params), seed, n_jobs=nh.CPU_STREAM_THREADS)
        for seed in nh.FINAL_SEEDS_D
    ]
    base_e_fits = [
        fit_xgboost(labeled_lf, None, feature_columns, params=dict(xgb_params), random_seed=seed)
        for seed in nh.FINAL_SEEDS_E
    ]
    graph_fit = train_graphsage_model(
        graph,
        label_frame=label_frame,
        train_user_ids=labeled_user_ids,
        valid_user_ids=None,
        max_epochs=PRIMARY_GRAPH_MAX_EPOCHS,
        hidden_dim=128,
    )

    return {
        "graph": graph,
        "label_frame": label_frame,
        "feature_columns": feature_columns,
        "trans_cols": trans_cols,
        "labeled_lf": labeled_lf,
        "labeled_td": labeled_td,
        "predict_lf": predict_lf,
        "predict_td": predict_td,
        "base_a_fits": base_a_fits,
        "base_b_fit": base_b_fit,
        "base_d_fits": base_d_fits,
        "base_e_fits": base_e_fits,
        "graph_fit": graph_fit,
        "xgb_params": xgb_params,
        "cb_params": cb_params,
        "lgbm_params": lgbm_params,
    }


def _fit_submission_stacker(oof: pd.DataFrame) -> tuple[BlendEnsemble, IsotonicCalibrator]:
    frame = _add_base_meta_features(oof.copy())
    weights = tune_blend_weights(frame)
    blend = BlendEnsemble(weights)
    stacker_cols = [c for c in STACKER_FEATURE_COLUMNS if c in frame.columns]
    raw_probs = blend.predict_proba(frame[stacker_cols])[:, 1]
    calibrator = IsotonicCalibrator().fit(raw_probs, frame["status"].astype(int).to_numpy())
    return blend, calibrator


def _score_predict_only(trained: dict[str, object], blend: BlendEnsemble, calibrator: IsotonicCalibrator, threshold: float) -> pd.DataFrame:
    graph = trained["graph"]
    label_frame = trained["label_frame"]
    feature_columns = trained["feature_columns"]
    predict_lf = trained["predict_lf"]
    predict_td = trained["predict_td"]
    base_a_fits = trained["base_a_fits"]
    base_b_fit = trained["base_b_fit"]
    base_d_fits = trained["base_d_fits"]
    base_e_fits = trained["base_e_fits"]
    graph_fit = trained["graph_fit"]

    base_a_probs = np.mean(
        [nh._predict_fit_result(fit, predict_lf) for fit in base_a_fits],
        axis=0,
    )
    base_b_probs = base_b_fit.model.predict_proba(predict_td[base_b_fit.feature_columns])[:, 1]
    base_d_probs = np.mean(
        [nh._predict_fit_result(fit, predict_lf) for fit in base_d_fits],
        axis=0,
    )
    base_e_probs = np.mean(
        [nh._predict_fit_result(fit, predict_lf) for fit in base_e_fits],
        axis=0,
    )

    all_label_free = pd.concat([trained["labeled_lf"], predict_lf], ignore_index=True).drop_duplicates(subset=["user_id"])
    all_base_a_probs = np.mean(
        [nh._predict_fit_result(fit, all_label_free) for fit in base_a_fits],
        axis=0,
    )
    all_base_probs = dict(zip(all_label_free["user_id"].astype(int).tolist(), all_base_a_probs.tolist()))
    train_labels = dict(zip(label_frame["user_id"].astype(int).tolist(), label_frame["status"].astype(float).tolist()))
    cs_result = correct_and_smooth(
        graph,
        train_labels,
        all_base_probs,
        alpha_correct=0.5,
        alpha_smooth=0.5,
        n_correct_iter=50,
        n_smooth_iter=50,
    )
    predict_ids = predict_lf["user_id"].astype(int).tolist()
    base_cs_probs = np.array([cs_result.get(uid, float(prob)) for uid, prob in zip(predict_ids, base_a_probs)], dtype=float)

    graph_probs = pd.DataFrame({"user_id": graph.user_ids, "graph_probability": graph_fit.full_probabilities})
    scored = predict_lf.merge(graph_probs, on="user_id", how="left")
    scored["base_a_probability"] = base_a_probs
    scored["base_c_s_probability"] = base_cs_probs
    scored["base_b_probability"] = base_b_probs
    scored["base_c_probability"] = scored["graph_probability"].fillna(0.0)
    scored["base_d_probability"] = base_d_probs
    scored["base_e_probability"] = base_e_probs
    for col in ("rule_score", "anomaly_score", "crypto_anomaly_score", "anomaly_score_segmented"):
        if col not in scored.columns:
            scored[col] = 0.0
        scored[col] = pd.to_numeric(scored[col], errors="coerce").fillna(0.0)

    scored = _add_base_meta_features(scored)
    stacker_cols = [c for c in STACKER_FEATURE_COLUMNS if c in scored.columns]
    scored["stacker_raw_probability"] = blend.predict_proba(scored[stacker_cols])[:, 1]
    scored["submission_probability"] = calibrator.predict(scored["stacker_raw_probability"].to_numpy())
    scored["status"] = (scored["submission_probability"] >= threshold).astype(int)
    return scored


def _save_artifacts(report: dict, trained: dict[str, object], blend: BlendEnsemble, calibrator: IsotonicCalibrator, scored: pd.DataFrame) -> dict:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).isoformat()

    base_a_paths = []
    for seed, fit in zip(nh.FINAL_SEEDS_A, trained["base_a_fits"]):
        path = MODELS_DIR / f"base_a_seed{seed}.pkl"
        save_pickle(fit.model, path)
        base_a_paths.append(str(path))

    base_d_paths = []
    for seed, fit in zip(nh.FINAL_SEEDS_D, trained["base_d_fits"]):
        path = MODELS_DIR / f"base_d_seed{seed}.pkl"
        save_pickle(fit.model, path)
        base_d_paths.append(str(path))

    base_e_paths = []
    for seed, fit in zip(nh.FINAL_SEEDS_E, trained["base_e_fits"]):
        path = MODELS_DIR / f"base_e_seed{seed}.pkl"
        save_pickle(fit.model, path)
        base_e_paths.append(str(path))

    base_b_path = MODELS_DIR / "base_b.pkl"
    blend_path = MODELS_DIR / "blend_model.pkl"
    calibrator_path = MODELS_DIR / "calibrator.pkl"
    graph_path = MODELS_DIR / "graphsage.pt"
    save_pickle(trained["base_b_fit"].model, base_b_path)
    save_pickle(blend, blend_path)
    save_pickle(calibrator, calibrator_path)
    save_graph_model(trained["graph_fit"].model_state, graph_path)

    submission = scored[["user_id", "status"]].copy().sort_values("user_id").reset_index(drop=True)
    submission.to_csv(OUT_CSV, index=False)
    scored.to_parquet(OUT_SCORES, index=False)

    branch = _git(["git", "branch", "--show-current"])
    commit = _git(["git", "rev-parse", "HEAD"])
    metadata = {
        "status": "PROVISIONAL_SUBMISSION_READY",
        "branch": branch,
        "commit": commit,
        "repo_path": str(ROOT),
        "config_source": str(MAINLINE_REPORT),
        "report_source": str(MAINLINE_REPORT),
        "oof_source": str(MAINLINE_OOF),
        "frozen_params": {
            "catboost": report["frozen_catboost_params"],
            "lightgbm": report["frozen_lightgbm_params"],
            "xgboost": report["frozen_xgboost_params"],
        },
        "graphsage_enabled": True,
        "threshold_used": float(report["honest_threshold"]),
        "train_rows": int(len(trained["labeled_lf"])),
        "predict_rows": int(len(submission)),
        "output_file_path": str(OUT_CSV),
        "scores_path": str(OUT_SCORES),
        "generation_timestamp": timestamp,
        "note": [
            "THIS_IS_PROVISIONAL",
            "XGB_GRID_FOLDS_2_3_4_STILL_PENDING_ON_OTHER_MACHINE",
        ],
        "validation": {
            "row_count_ok": int(len(submission)) == 12753,
            "user_id_unique": bool(not submission["user_id"].duplicated().any()),
            "no_nan_status": bool(not submission["status"].isna().any()),
            "status_binary": bool(set(submission["status"].unique()).issubset({0, 1})),
            "columns": list(submission.columns),
        },
        "artifacts": {
            "base_a_paths": base_a_paths,
            "base_b_path": str(base_b_path),
            "base_d_paths": base_d_paths,
            "base_e_paths": base_e_paths,
            "graph_path": str(graph_path),
            "blend_path": str(blend_path),
            "calibrator_path": str(calibrator_path),
        },
    }
    save_json(metadata, OUT_JSON)
    OUT_MD.write_text(
        "\n".join(
            [
                "# Provisional Submission Status",
                "",
                "- status: `PROVISIONAL_SUBMISSION_READY`",
                f"- branch: `{branch}`",
                f"- commit: `{commit}`",
                f"- repo_path: `{ROOT}`",
                f"- mainline_config_source: `{MAINLINE_REPORT}`",
                f"- mainline_oof_source: `{MAINLINE_OOF}`",
                f"- threshold_used: `{report['honest_threshold']}`",
                f"- train_rows: `{len(trained['labeled_lf'])}`",
                f"- predict_rows: `{len(submission)}`",
                f"- submission_path: `{OUT_CSV}`",
                f"- note: `THIS_IS_PROVISIONAL`",
                f"- note: `XGB_GRID_FOLDS_2_3_4_STILL_PENDING_ON_OTHER_MACHINE`",
                f"- recommendation: `wait_for_xgb_grid_decision_before_final_upload`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return metadata


def main() -> None:
    report, oof = _require_mainline_sources()
    trained = _train_final_models(_load_dataset("full"), report)
    blend, calibrator = _fit_submission_stacker(oof)
    scored = _score_predict_only(trained, blend, calibrator, float(report["honest_threshold"]))
    metadata = _save_artifacts(report, trained, blend, calibrator, scored)
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
