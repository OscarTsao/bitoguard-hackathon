#!/usr/bin/env python3
"""
讀取 HPO 結果並產出更新後的超參數，用於最終 pipeline 重跑。

用法：
  # 查看 HPO 結果（不修改任何檔案）
  python scripts/apply_hpo_results.py --show

  # 產出 env vars（可直接 source）
  python scripts/apply_hpo_results.py --env > /tmp/hpo_env.sh && source /tmp/hpo_env.sh

  # 直接更新 modeling.py / modeling_xgb.py 的預設值
  python scripts/apply_hpo_results.py --apply

搜尋以下檔案的 HPO 結果：
  - artifacts/official_features/hpo_catboost_e15_cpu.json  (140.123 CatBoost v1)
  - artifacts/official_features/hpo_catboost_v2.json       (SageMaker CatBoost v2)
  - artifacts/official_features/hpo_xgb_e15.json           (XGBoost)
  - artifacts/official_features/cs_grid_search.json        (C&S grid search)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _find_best_catboost(feature_dir: Path) -> dict | None:
    """合併兩個 CatBoost HPO 結果，取最佳。"""
    candidates = []
    for name in ["hpo_catboost_e15_cpu.json", "hpo_catboost_v2.json",
                  "hpo_best_params.json", "hpo_catboost_e15.json"]:
        data = _load_json(feature_dir / name)
        if data and "best_f1" in data:
            candidates.append({
                "source": name,
                "f1": data["best_f1"],
                "params": data["best_params"],
            })
    if not candidates:
        return None
    return max(candidates, key=lambda c: c["f1"])


def _find_best_xgb(feature_dir: Path) -> dict | None:
    for name in ["hpo_xgb_e15.json", "hpo_xgb_best.json"]:
        data = _load_json(feature_dir / name)
        if data and "best_f1" in data:
            return {"source": name, "f1": data["best_f1"], "params": data["best_params"]}
    return None


def _find_best_cs(feature_dir: Path) -> dict | None:
    data = _load_json(feature_dir / "cs_grid_search.json")
    if data and "best" in data:
        return data["best"]
    return None


def show_results(feature_dir: Path) -> None:
    print("=" * 60)
    print("HPO Results Summary")
    print("=" * 60)

    cb = _find_best_catboost(feature_dir)
    if cb:
        print(f"\n[CatBoost Base A] Best F1={cb['f1']:.4f} (from {cb['source']})")
        for k, v in cb["params"].items():
            print(f"  {k}: {v}")
    else:
        print("\n[CatBoost] No HPO results found")

    xgb = _find_best_xgb(feature_dir)
    if xgb:
        print(f"\n[XGBoost Base E] Best F1={xgb['f1']:.4f} (from {xgb['source']})")
        for k, v in xgb["params"].items():
            print(f"  {k}: {v}")
    else:
        print("\n[XGBoost] No HPO results found")

    cs = _find_best_cs(feature_dir)
    if cs:
        print(f"\n[C&S] Best F1={cs['f1']:.4f}")
        print(f"  alpha_correct: {cs['ac']}")
        print(f"  alpha_smooth: {cs['as']}")
        print(f"  n_correct_iter: {cs['nc']}")
        print(f"  n_smooth_iter: {cs['ns']}")
    else:
        print("\n[C&S] No grid search results found")

    # 和現有預設值比較
    print("\n" + "=" * 60)
    print("Current defaults vs HPO best")
    print("=" * 60)
    if cb:
        print("\nCatBoost (modeling.py defaults → HPO best):")
        defaults = {"depth": 7, "learning_rate": 0.05, "l2_leaf_reg": 3.0,
                     "border_count": 254, "iterations": 1500}
        for k, v in defaults.items():
            new = cb["params"].get(k, v)
            changed = "  ← CHANGED" if new != v else ""
            print(f"  {k}: {v} → {new}{changed}")

    if xgb:
        print("\nXGBoost (modeling_xgb.py defaults → HPO best):")
        defaults = {"n_estimators": 1500, "max_depth": 6, "learning_rate": 0.0585,
                     "subsample": 0.812, "colsample_bytree": 0.881,
                     "reg_alpha": 0.061, "reg_lambda": 5.707, "min_child_weight": 5.185}
        for k, v in defaults.items():
            new = xgb["params"].get(k, v)
            changed = "  ← CHANGED" if abs(float(new) - float(v)) > 0.001 else ""
            print(f"  {k}: {v} → {new}{changed}")


def print_env(feature_dir: Path) -> None:
    """產出環境變數格式，可直接 source。"""
    cb = _find_best_catboost(feature_dir)
    xgb = _find_best_xgb(feature_dir)
    cs = _find_best_cs(feature_dir)

    print("# HPO best params — source this file before running the pipeline")
    if cb:
        p = cb["params"]
        print(f'export CB_DEPTH={p.get("depth", 7)}')
        print(f'export CB_LEARNING_RATE={p.get("learning_rate", 0.05)}')
        print(f'export CB_L2_LEAF_REG={p.get("l2_leaf_reg", 3.0)}')
        print(f'export CB_ITERATIONS={p.get("iterations", 1500)}')
        print(f'export CB_BORDER_COUNT={p.get("border_count", 254)}')
        print(f'export CB_RANDOM_STRENGTH={p.get("random_strength", 1.0)}')
        print(f'export CB_BAGGING_TEMPERATURE={p.get("bagging_temperature", 1.0)}')
        print(f'export CB_MIN_DATA_IN_LEAF={p.get("min_data_in_leaf", 1)}')
        print(f'export CB_MAX_CLASS_WEIGHT={p.get("max_class_weight", 10.0)}')

    if xgb:
        p = xgb["params"]
        print(f'export XGB_N_ESTIMATORS={p.get("n_estimators", 1500)}')
        print(f'export XGB_MAX_DEPTH={p.get("max_depth", 6)}')
        print(f'export XGB_LEARNING_RATE={p.get("learning_rate", 0.0585)}')
        print(f'export XGB_SUBSAMPLE={p.get("subsample", 0.812)}')
        print(f'export XGB_COLSAMPLE_BYTREE={p.get("colsample_bytree", 0.881)}')
        print(f'export XGB_REG_ALPHA={p.get("reg_alpha", 0.061)}')
        print(f'export XGB_REG_LAMBDA={p.get("reg_lambda", 5.707)}')
        print(f'export XGB_MIN_CHILD_WEIGHT={p.get("min_child_weight", 5.185)}')

    if cs:
        print(f'export CS_ALPHA_CORRECT={cs["ac"]}')
        print(f'export CS_ALPHA_SMOOTH={cs["as"]}')
        print(f'export CS_N_CORRECT_ITER={cs["nc"]}')
        print(f'export CS_N_SMOOTH_ITER={cs["ns"]}')


def apply_to_code(feature_dir: Path, code_dir: Path) -> None:
    """直接更新 modeling.py 和 modeling_xgb.py 的預設值。"""
    cb = _find_best_catboost(feature_dir)
    xgb = _find_best_xgb(feature_dir)

    if cb:
        modeling_path = code_dir / "official" / "modeling.py"
        if modeling_path.exists():
            content = modeling_path.read_text(encoding="utf-8")
            p = cb["params"]
            # 更新預設值
            import re
            replacements = {
                r'(hp\.pop\("iterations",\s*)\d+\)': f'hp.pop("iterations", {p.get("iterations", 1500)})',
                r'(hp\.pop\("depth",\s*)\d+\)': f'hp.pop("depth", {p.get("depth", 7)})',
                r'(hp\.pop\("learning_rate",\s*)[0-9.]+\)': f'hp.pop("learning_rate", {p.get("learning_rate", 0.05)})',
                r'(hp\.pop\("l2_leaf_reg",\s*)[0-9.]+\)': f'hp.pop("l2_leaf_reg", {p.get("l2_leaf_reg", 3.0)})',
                r'(hp\.pop\("border_count",\s*)\d+\)': f'hp.pop("border_count", {p.get("border_count", 254)})',
            }
            for pattern, replacement in replacements.items():
                content = re.sub(pattern, replacement, content)
            modeling_path.write_text(content, encoding="utf-8")
            print(f"[CatBoost] Updated {modeling_path}")
            print(f"  Best F1={cb['f1']:.4f} from {cb['source']}")
        else:
            print(f"[CatBoost] {modeling_path} not found")

    if xgb:
        xgb_path = code_dir / "official" / "modeling_xgb.py"
        if xgb_path.exists():
            content = xgb_path.read_text(encoding="utf-8")
            p = xgb["params"]
            import re
            replacements = {
                r'(p\.get\("n_estimators",\s*)\d+\)': f'p.get("n_estimators", {p.get("n_estimators", 1500)})',
                r'(p\.get\("max_depth",\s*)\d+\)': f'p.get("max_depth", {p.get("max_depth", 6)})',
                r'(p\.get\("learning_rate",\s*)[0-9.]+\)': f'p.get("learning_rate", {p.get("learning_rate", 0.0585)})',
                r'(p\.get\("subsample",\s*)[0-9.]+\)': f'p.get("subsample", {p.get("subsample", 0.812)})',
                r'(p\.get\("colsample_bytree",\s*)[0-9.]+\)': f'p.get("colsample_bytree", {p.get("colsample_bytree", 0.881)})',
                r'(p\.get\("reg_alpha",\s*)[0-9.]+\)': f'p.get("reg_alpha", {p.get("reg_alpha", 0.061)})',
                r'(p\.get\("reg_lambda",\s*)[0-9.]+\)': f'p.get("reg_lambda", {p.get("reg_lambda", 5.707)})',
                r'(p\.get\("min_child_weight",\s*)[0-9.]+\)': f'p.get("min_child_weight", {p.get("min_child_weight", 5.185)})',
            }
            for pattern, replacement in replacements.items():
                content = re.sub(pattern, replacement, content)
            xgb_path.write_text(content, encoding="utf-8")
            print(f"[XGBoost] Updated {xgb_path}")
            print(f"  Best F1={xgb['f1']:.4f} from {xgb['source']}")
        else:
            print(f"[XGBoost] {xgb_path} not found")


def main():
    parser = argparse.ArgumentParser(description="Apply HPO results to pipeline code")
    parser.add_argument("--show", action="store_true", help="Show HPO results (no changes)")
    parser.add_argument("--env", action="store_true", help="Print env vars to source")
    parser.add_argument("--apply", action="store_true", help="Update modeling.py/modeling_xgb.py defaults")
    parser.add_argument("--feature-dir", type=str, default=None,
                        help="Path to official_features dir (default: auto-detect)")
    parser.add_argument("--code-dir", type=str, default=None,
                        help="Path to bitoguard_core dir (default: auto-detect)")
    args = parser.parse_args()

    if not any([args.show, args.env, args.apply]):
        args.show = True

    # Auto-detect paths
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    code_dir = Path(args.code_dir) if args.code_dir else repo_root / "bitoguard_core"
    feature_dir = Path(args.feature_dir) if args.feature_dir else code_dir / "artifacts" / "official_features"

    if args.show:
        show_results(feature_dir)
    if args.env:
        print_env(feature_dir)
    if args.apply:
        apply_to_code(feature_dir, code_dir)


if __name__ == "__main__":
    main()
