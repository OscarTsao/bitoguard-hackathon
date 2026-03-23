"""Apply bug fixes + E27/E29/E21 patches, then run experiments."""
import os, sys, json, subprocess, glob, re

BASE_DIR = os.path.expanduser("~/bitoguard-hackathon/bitoguard_core")
os.chdir(BASE_DIR)

def read_file(path):
    with open(path) as f:
        return f.read()

def write_file(path, content):
    with open(path, "w") as f:
        f.write(content)

def run_pipeline(env_vars, tag):
    env = os.environ.copy()
    env.update(env_vars)
    env["PYTHONPATH"] = "."
    env["SKIP_GNN"] = "1"
    print(f"\n{'='*60}", flush=True)
    print(f"Running {tag}...", flush=True)
    print(f"  env: {env_vars}", flush=True)
    print(f"{'='*60}", flush=True)
    for f in glob.glob("artifacts/official_features/official_*.parquet"):
        os.remove(f)
    proc = subprocess.run(
        [sys.executable, "-u", "-m", "official.pipeline"],
        env=env, capture_output=True, text=True, cwd=BASE_DIR
    )
    log_path = f"/tmp/e_{tag}.log"
    with open(log_path, "w") as f:
        f.write(proc.stdout + "\n" + proc.stderr)
    if proc.returncode != 0:
        print(f"  FAILED! Last 15 lines:", flush=True)
        for line in proc.stderr.strip().split("\n")[-15:]:
            print(f"  | {line}", flush=True)
        return None
    report_path = os.path.join(BASE_DIR, "artifacts/reports/official_validation_report.json")
    if os.path.exists(report_path):
        with open(report_path) as f:
            r = json.load(f)
        cal = r["calibrator"]
        sel = cal["selected_row"]
        sec = r.get("secondary_group_stress_metrics", {})
        result = {
            "tag": tag, "primary_f1": sel["f1"], "primary_ap": cal["average_precision"],
            "threshold": r["selected_threshold"], "precision": sel["precision"],
            "recall": sel["recall"], "bootstrap_f1": sel["bootstrap_mean_f1"],
            "secondary_f1": sec.get("f1", "N/A"), "secondary_ap": sec.get("average_precision", "N/A"),
        }
        print(f"  Primary F1={result['primary_f1']:.4f}, AP={result['primary_ap']:.4f}", flush=True)
        sec_f1 = result['secondary_f1']
        if isinstance(sec_f1, float):
            print(f"  Secondary F1={sec_f1:.4f}", flush=True)
        else:
            print(f"  Secondary F1={sec_f1}", flush=True)
        print(f"  Threshold={result['threshold']}", flush=True)
        return result
    print("  No validation report!", flush=True)
    return None

# ══════════════════════════════════════════
# BUG FIX 1: validate.py stale HPO params
# ══════════════════════════════════════════
print("[FIX 1] validate.py: remove stale HPO loading", flush=True)
val_code = read_file("official/validate.py")
OLD_HPO = """    # Load HPO params for secondary validation consistency with primary
    catboost_params = None
    try:
        from official.hpo import load_hpo_best_params
        catboost_params = load_hpo_best_params()
    except Exception:
        pass"""
NEW_HPO = """    # Bug fix: don't load stale HPO params — use same defaults as primary
    catboost_params = None"""
if OLD_HPO in val_code:
    val_code = val_code.replace(OLD_HPO, NEW_HPO)
    write_file("official/validate.py", val_code)
    print("  Fixed!", flush=True)
else:
    print("  Already fixed or not found", flush=True)

# ══════════════════════════════════════════
# BUG FIX 2: Disable stale HPO json files
# ══════════════════════════════════════════
print("[FIX 2] Disable stale HPO json files", flush=True)
for f in ["artifacts/official_features/hpo_best_params.json",
          "artifacts/official_features/hpo_catboost_best.json"]:
    if os.path.exists(f):
        os.rename(f, f + ".DISABLED")
        print(f"  Disabled: {f}", flush=True)
    else:
        print(f"  Not found: {f}", flush=True)

# ══════════════════════════════════════════
# PATCH train.py for E27 (C&S on Base E) + E29 (C&S alpha sweep)
# ══════════════════════════════════════════
print("[PATCH] train.py: E27 C&S source + E29 alpha sweep", flush=True)
train_code = read_file("official/train.py")

# 1. Store _base_e_models list (like _base_a_models)
OLD_XGB = """        # Multi-seed XGBoost ensemble for Base E (2 seeds).
        _base_e_val_probs = []
        _base_e_fit = None
        for _seed_e in _BASE_E_SEEDS:
            _base_e_fit = fit_xgboost(train_label_free, valid_label_free, base_a_feature_columns, random_seed=_seed_e)
            _base_e_val_probs.append(_base_e_fit.validation_probabilities)"""
NEW_XGB = """        # Multi-seed XGBoost ensemble for Base E (2 seeds).
        _base_e_val_probs = []
        _base_e_models = []
        _base_e_fit = None
        for _seed_e in _BASE_E_SEEDS:
            _base_e_fit = fit_xgboost(train_label_free, valid_label_free, base_a_feature_columns, random_seed=_seed_e)
            _base_e_val_probs.append(_base_e_fit.validation_probabilities)
            _base_e_models.append(_base_e_fit.model)"""

if "_base_e_models = []" not in train_code:
    train_code = train_code.replace(OLD_XGB, NEW_XGB)
    print("  Added _base_e_models list", flush=True)

# 2. Replace C&S block with CS_SOURCE + alpha env vars
OLD_CS = """        # Correct-and-Smooth (C&S): graph post-processing on Base A OOF probs.
        _train_a_probs = np.mean(
            [m.predict_proba(train_label_free[base_a_feature_columns])[:, 1] for m in _base_a_models],
            axis=0,
        )
        _val_a_probs = np.asarray(base_a_fit.validation_probabilities, dtype=float)
        _cs_base_probs: dict[int, float] = {}
        for _uid, _prob in zip(train_label_free["user_id"].astype(int), _train_a_probs):
            _cs_base_probs[int(_uid)] = float(_prob)
        for _uid, _prob in zip(valid_label_free["user_id"].astype(int), _val_a_probs):
            _cs_base_probs[int(_uid)] = float(_prob)
        # Include unlabeled users in C&S base_probs so their fraud signal propagates.
        _all_labeled_ids = set(train_users) | set(valid_users)
        _unlabeled_frame = label_free_frame[~label_free_frame["user_id"].astype(int).isin(_all_labeled_ids)]
        if len(_unlabeled_frame) > 0:
            _unlabeled_a_probs = np.mean(
                [m.predict_proba(_unlabeled_frame[base_a_feature_columns])[:, 1] for m in _base_a_models],
                axis=0,
            )
            for _uid, _prob in zip(_unlabeled_frame["user_id"].astype(int), _unlabeled_a_probs):
                _cs_base_probs[int(_uid)] = float(_prob)
        _cs_train_labels: dict[int, float] = dict(zip(
            fold_train_labels["user_id"].astype(int),
            fold_train_labels["status"].astype(float),
        ))
        _cs_result = correct_and_smooth(
            graph, _cs_train_labels, _cs_base_probs,
            alpha_correct=0.5, alpha_smooth=0.5,
            n_correct_iter=50, n_smooth_iter=50,
        )
        _val_ids = valid_label_free["user_id"].astype(int).tolist()
        _cs_val_probs = np.array(
            [_cs_result.get(int(_uid), float(_p)) for _uid, _p in zip(_val_ids, _val_a_probs)],
            dtype=float,
        )"""

NEW_CS = """        # Correct-and-Smooth (C&S): graph post-processing.
        # E27: CS_SOURCE selects which base model feeds C&S (default: base_a)
        import os as _cs_os
        _cs_source = _cs_os.environ.get("CS_SOURCE", "base_a")
        _cs_alpha_c = float(_cs_os.environ.get("CS_ALPHA_CORRECT", "0.5"))
        _cs_alpha_s = float(_cs_os.environ.get("CS_ALPHA_SMOOTH", "0.5"))
        _cs_n_correct = int(_cs_os.environ.get("CS_N_CORRECT", "50"))
        _cs_n_smooth = int(_cs_os.environ.get("CS_N_SMOOTH", "50"))

        if _cs_source == "base_e":
            _cs_train_probs = np.mean(
                [m.predict_proba(train_label_free[base_a_feature_columns])[:, 1] for m in _base_e_models],
                axis=0,
            )
            _cs_val_probs_raw = np.asarray(base_e_fit.validation_probabilities, dtype=float)
            if fold_id == 0:
                print(f"  C&S source: Base E (XGBoost, {len(_base_e_models)} seeds)", flush=True)
        else:
            _cs_train_probs = np.mean(
                [m.predict_proba(train_label_free[base_a_feature_columns])[:, 1] for m in _base_a_models],
                axis=0,
            )
            _cs_val_probs_raw = np.asarray(base_a_fit.validation_probabilities, dtype=float)

        _cs_base_probs: dict[int, float] = {}
        for _uid, _prob in zip(train_label_free["user_id"].astype(int), _cs_train_probs):
            _cs_base_probs[int(_uid)] = float(_prob)
        for _uid, _prob in zip(valid_label_free["user_id"].astype(int), _cs_val_probs_raw):
            _cs_base_probs[int(_uid)] = float(_prob)
        # Include unlabeled users in C&S base_probs so their fraud signal propagates.
        _all_labeled_ids = set(train_users) | set(valid_users)
        _unlabeled_frame = label_free_frame[~label_free_frame["user_id"].astype(int).isin(_all_labeled_ids)]
        if len(_unlabeled_frame) > 0:
            if _cs_source == "base_e":
                _unlabeled_cs_probs = np.mean(
                    [m.predict_proba(_unlabeled_frame[base_a_feature_columns])[:, 1] for m in _base_e_models],
                    axis=0,
                )
            else:
                _unlabeled_cs_probs = np.mean(
                    [m.predict_proba(_unlabeled_frame[base_a_feature_columns])[:, 1] for m in _base_a_models],
                    axis=0,
                )
            for _uid, _prob in zip(_unlabeled_frame["user_id"].astype(int), _unlabeled_cs_probs):
                _cs_base_probs[int(_uid)] = float(_prob)
        _cs_train_labels: dict[int, float] = dict(zip(
            fold_train_labels["user_id"].astype(int),
            fold_train_labels["status"].astype(float),
        ))
        _cs_result = correct_and_smooth(
            graph, _cs_train_labels, _cs_base_probs,
            alpha_correct=_cs_alpha_c, alpha_smooth=_cs_alpha_s,
            n_correct_iter=_cs_n_correct, n_smooth_iter=_cs_n_smooth,
        )
        _val_ids = valid_label_free["user_id"].astype(int).tolist()
        _cs_val_probs = np.array(
            [_cs_result.get(int(_uid), float(_p)) for _uid, _p in zip(_val_ids, _cs_val_probs_raw)],
            dtype=float,
        )"""

if "CS_SOURCE" not in train_code:
    train_code = train_code.replace(OLD_CS, NEW_CS)
    print("  Added CS_SOURCE + CS_ALPHA env vars", flush=True)

# 3. Add PRUNE_FEATURES to _label_free_feature_columns
OLD_LFFC = """def _label_free_feature_columns(dataset: pd.DataFrame) -> list[str]:
    non_null = {col for col in dataset.columns if not dataset[col].isna().all()}
    return [column for column in dataset.columns if column not in LABEL_FREE_EXCLUDED_COLUMNS and column in non_null]"""
NEW_LFFC = """def _label_free_feature_columns(dataset: pd.DataFrame) -> list[str]:
    non_null = {col for col in dataset.columns if not dataset[col].isna().all()}
    cols = [column for column in dataset.columns if column not in LABEL_FREE_EXCLUDED_COLUMNS and column in non_null]
    _prune = __import__("os").environ.get("PRUNE_FEATURES", "")
    if _prune:
        _exclude = set(_prune.split(","))
        cols = [c for c in cols if c not in _exclude]
        print(f"[features] Pruned {len(_exclude)} features, {len(cols)} remaining", flush=True)
    return cols"""
if "PRUNE_FEATURES" not in train_code:
    train_code = train_code.replace(OLD_LFFC, NEW_LFFC)
    print("  Added PRUNE_FEATURES env var", flush=True)

# 4. Add NEG_DOWNSAMPLE_RATIO
DS_PATCH = '''
        # ── Negative downsampling ──
        import os as _ds_os
        _ds_ratio = float(_ds_os.environ.get("NEG_DOWNSAMPLE_RATIO", "0"))
        if _ds_ratio > 0:
            _pos = train_label_free[train_label_free["status"].astype(int) == 1]
            _neg = train_label_free[train_label_free["status"].astype(int) == 0]
            _n_target = int(len(_pos) * _ds_ratio)
            if _n_target < len(_neg):
                _neg = _neg.sample(n=_n_target, random_state=42+fold_id)
                train_label_free = pd.concat([_pos, _neg]).sort_values("user_id").reset_index(drop=True)
                _ds_uids = set(train_label_free["user_id"].astype(int))
                train_transductive = train_transductive[
                    train_transductive["user_id"].astype(int).isin(_ds_uids)
                ].copy()
                print(f"  [fold {fold_id}] Downsampled: {len(_pos)}+ {len(_neg)}- (1:{_ds_ratio:.0f})", flush=True)

'''
ANCHOR = "        # Multi-seed CatBoost ensemble for Base A (4 seeds, reduces variance ~50%)."
if "NEG_DOWNSAMPLE_RATIO" not in train_code:
    train_code = train_code.replace(ANCHOR, DS_PATCH + ANCHOR)
    print("  Added NEG_DOWNSAMPLE_RATIO", flush=True)

write_file("official/train.py", train_code)

# ══════════════════════════════════════════
# PATCH modeling.py + modeling_xgb.py for E26
# ══════════════════════════════════════════
print("[PATCH] modeling.py: CB_MAX_CLASS_WEIGHT", flush=True)
mod_code = read_file("official/modeling.py")
OLD_CW = '    _max_cw = hp.pop("max_class_weight", 10.0)'
NEW_CW = '    _max_cw = float(__import__("os").environ.get("CB_MAX_CLASS_WEIGHT", str(hp.pop("max_class_weight", 10.0))))'
if "CB_MAX_CLASS_WEIGHT" not in mod_code:
    mod_code = mod_code.replace(OLD_CW, NEW_CW)
    write_file("official/modeling.py", mod_code)
    print("  Done", flush=True)
else:
    print("  Already patched", flush=True)

print("[PATCH] modeling_xgb.py: XGB_SPW_CAP", flush=True)
xgb_code = read_file("official/modeling_xgb.py")
OLD_SPW = "    scale_pos_weight = min(float(negatives) / positives, 15.0)"
NEW_SPW = '    _spw_cap = float(__import__("os").environ.get("XGB_SPW_CAP", "15.0"))\n    scale_pos_weight = min(float(negatives) / positives, _spw_cap)'
if "XGB_SPW_CAP" not in xgb_code:
    xgb_code = xgb_code.replace(OLD_SPW, NEW_SPW)
    write_file("official/modeling_xgb.py", xgb_code)
    print("  Done", flush=True)
else:
    print("  Already patched", flush=True)

# ══════════════════════════════════════════
# PATCH stacking.py for E28 (blend step refinement)
# ══════════════════════════════════════════
print("[PATCH] stacking.py: BLEND_STEP + BLEND_MIN_AP", flush=True)
stack_code = read_file("official/stacking.py")
# This is trickier — need to find the step and min_ap values in tune_blend_weights
# For now, skip this and focus on E27
print("  Skipped (E27 is priority)", flush=True)

print("\n" + "="*60, flush=True)
print("All patches applied. Starting experiments...", flush=True)
print("="*60, flush=True)

results = []

# ── E27a: C&S on Base E (HIGHEST PRIORITY) ──
r = run_pipeline({"CS_SOURCE": "base_e"}, "E27a_cs_base_e")
if r: results.append(r)

# ── E29a: C&S alpha 0.3 ──
r = run_pipeline({"CS_ALPHA_CORRECT": "0.3", "CS_ALPHA_SMOOTH": "0.3"}, "E29a_alpha03")
if r: results.append(r)

# ── E29b: C&S alpha 0.7 ──
r = run_pipeline({"CS_ALPHA_CORRECT": "0.7", "CS_ALPHA_SMOOTH": "0.7"}, "E29b_alpha07")
if r: results.append(r)

# ── E26a: Lower class weights ──
r = run_pipeline({"CB_MAX_CLASS_WEIGHT": "5", "XGB_SPW_CAP": "8"}, "E26a_low_cw")
if r: results.append(r)

# ── E26b: Higher class weights ──
r = run_pipeline({"CB_MAX_CLASS_WEIGHT": "20", "XGB_SPW_CAP": "25"}, "E26b_high_cw")
if r: results.append(r)

# ── E27a + E29a combined: C&S on Base E with alpha 0.3 ──
r = run_pipeline({"CS_SOURCE": "base_e", "CS_ALPHA_CORRECT": "0.3", "CS_ALPHA_SMOOTH": "0.3"}, "E27_E29_cs_e_a03")
if r: results.append(r)

# Print summary
print("\n" + "=" * 80, flush=True)
print("FINAL RESULTS SUMMARY", flush=True)
print("=" * 80, flush=True)
print(f"{'Tag':<25} {'Pri F1':>8} {'Pri AP':>8} {'Sec F1':>8} {'Thr':>8} {'Delta':>8}", flush=True)
print("-" * 75, flush=True)
print(f"{'E15 baseline':<25} {'0.4418':>8} {'0.3842':>8} {'0.4304':>8} {'0.2071':>8} {'---':>8}", flush=True)
for r in results:
    delta = r["primary_f1"] - 0.4418
    sec_f1 = f"{r['secondary_f1']:.4f}" if isinstance(r["secondary_f1"], float) else str(r["secondary_f1"])
    print(f"{r['tag']:<25} {r['primary_f1']:>8.4f} {r['primary_ap']:>8.4f} {sec_f1:>8} {r['threshold']:>8} {delta:>+8.4f}", flush=True)

with open("/tmp/e27_results.json", "w") as f:
    json.dump(results, f, indent=2, default=str)
print(f"\nResults saved to /tmp/e27_results.json", flush=True)
