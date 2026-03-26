"""
SageMaker Training 入口 — E15 AML 管線。

此腳本由 SageMaker Training Job 執行，完整跑 run_official_pipeline()：
  1. 資料品質檢查
  2. 表格特徵 (~110 cols)
  3. 圖特徵 (~17 cols)
  4. 異常偵測特徵 (~29 cols)
  5. 序列特徵 (20 cols)
  6. 時序特徵 (23 cols)
  7. 訓練 CatBoost×4 + XGBoost×2 + LR stacker + 校準器
  8. OOF 驗證
  9. 產出 submission CSV (user_id, status)

SageMaker 目錄對應：
  /opt/ml/input/data/raw/    ← S3 上的 7 張 parquet 表
  /opt/ml/work/clean/        ← 清洗後的表（訓練中產生）
  /opt/ml/work/artifacts/    ← 模型、特徵、報告
  /opt/ml/model/             ← 訓練完成後複製 artifacts → model.tar.gz
  /opt/ml/output/data/       ← submission CSV
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

# ── 確保 bitoguard_core 在 sys.path ──────────────────────────────────────────
sys.path.insert(0, "/opt/ml/code")

# ── SageMaker 目錄常數 ───────────────────────────────────────────────────────
SM_INPUT_RAW = Path("/opt/ml/input/data/raw")
SM_WORK_DIR = Path("/opt/ml/work")
SM_CLEAN_DIR = SM_WORK_DIR / "clean"
SM_ARTIFACT_DIR = SM_WORK_DIR / "artifacts"
SM_MODEL_DIR = Path("/opt/ml/model")
SM_OUTPUT_DIR = Path("/opt/ml/output/data")


def _setup_env() -> None:
    """設定 E15 pipeline 所需的環境變數。

    official/common.py 的 load_official_paths() 透過 config.load_settings()
    讀取這些 env vars 來決定所有 I/O 路徑。
    """
    os.environ["BITOGUARD_AWS_EVENT_RAW_DIR"] = str(SM_INPUT_RAW)
    os.environ["BITOGUARD_AWS_EVENT_CLEAN_DIR"] = str(SM_CLEAN_DIR)
    os.environ["BITOGUARD_ARTIFACT_DIR"] = str(SM_ARTIFACT_DIR)
    # GNN 已停用（_DummyGNN），跳過 torch import 避免缺套件
    os.environ["SKIP_GNN"] = "1"
    # 跳過 secondary validation（省 ~50% 時間），後續可單獨跑
    os.environ["SKIP_SECONDARY_OOF"] = "1"
    # 停用平行化（128GB 不夠同時跑多個 model），用全部 32 cores 依序跑
    os.environ["PARALLEL_SEEDS"] = "0"

    # 確保目錄存在
    for d in (SM_CLEAN_DIR, SM_ARTIFACT_DIR, SM_MODEL_DIR, SM_OUTPUT_DIR):
        d.mkdir(parents=True, exist_ok=True)

    # 不需要 GPU（比賽帳號 GPU quota = 0）
    os.environ["BITOGUARD_USE_GPU"] = "0"


def _prepare_clean_tables() -> None:
    """從 raw parquet 建立 clean 表 + user_index。

    E15 pipeline 的 load_clean_table() 從 CLEAN_DIR 讀取。
    raw 表直接複製到 clean（金額欄位已在上傳時處理），
    並建立 user_index（合併 user_info + train_label + predict_label）。
    """
    import pandas as pd

    print("[資料準備] 建立 clean 表 ...")
    raw_dir = SM_INPUT_RAW
    clean_dir = SM_CLEAN_DIR
    clean_dir.mkdir(parents=True, exist_ok=True)

    def _scale(val):
        if val is None or val == "":
            return None
        return float(val) / 1e8

    # ── user_info → clean (加 has_profile, has_email, has_kyc 等) ──
    ui = pd.read_parquet(raw_dir / "user_info.parquet")
    ui["user_id"] = pd.to_numeric(ui["user_id"], errors="coerce").astype("Int64")
    # has_profile: 使用者是否填寫了個人資料（sex 或 age 非空）
    ui["has_profile"] = ui.get("sex", pd.Series(None, index=ui.index)).notna() | ui.get("age", pd.Series(None, index=ui.index)).notna()
    # confirmed_at → has_email_confirmation
    ui["has_email_confirmation"] = ui.get("confirmed_at", pd.Series(None, index=ui.index)).notna()
    # level1_finished_at → has_level1_kyc（API 欄位名稱不同於 clean 表）
    l1_col = "level1_finished_at" if "level1_finished_at" in ui.columns else "kyc_level1_at"
    l2_col = "level2_finished_at" if "level2_finished_at" in ui.columns else "kyc_level2_at"
    ui["has_level1_kyc"] = ui.get(l1_col, pd.Series(None, index=ui.index)).notna()
    ui["has_level2_kyc"] = ui.get(l2_col, pd.Series(None, index=ui.index)).notna()
    # kyc_level: 0=none, 1=level1, 2=level2
    ui["kyc_level"] = 0
    ui.loc[ui["has_level1_kyc"], "kyc_level"] = 1
    ui.loc[ui["has_level2_kyc"], "kyc_level"] = 2
    # 計算天數差
    confirmed_ts = pd.to_datetime(ui.get("confirmed_at"), errors="coerce", utc=True)
    l1_ts = pd.to_datetime(ui.get(l1_col), errors="coerce", utc=True)
    l2_ts = pd.to_datetime(ui.get(l2_col), errors="coerce", utc=True)
    ui["days_email_to_level1"] = (l1_ts - confirmed_ts).dt.total_seconds() / 86400
    ui["days_level1_to_level2"] = (l2_ts - l1_ts).dt.total_seconds() / 86400
    # 標籤化欄位
    ui["sex_label"] = ui.get("sex", pd.Series(None, index=ui.index)).fillna("unknown")
    ui["career_label"] = ui.get("career", pd.Series(None, index=ui.index)).fillna("unknown")
    ui["income_source_label"] = ui.get("income_source", pd.Series(None, index=ui.index)).fillna("unknown")
    ui["user_source_label"] = ui.get("user_source", pd.Series(None, index=ui.index)).fillna("unknown")
    ui.to_parquet(clean_dir / "user_info.parquet", index=False)
    print(f"  user_info.parquet → clean/ ({len(ui):,} rows)")

    # ── twd_transfer → clean (amount_twd, kind_label) ──
    twd = pd.read_parquet(raw_dir / "twd_transfer.parquet")
    twd["ori_samount_raw"] = twd.get("ori_samount", pd.Series(0, index=twd.index)).copy()
    twd["amount_twd"] = pd.to_numeric(twd.get("ori_samount", 0), errors="coerce") * 1e-8
    twd["kind_label"] = twd.get("kind", pd.Series(0, index=twd.index)).apply(
        lambda x: "deposit" if int(x) == 0 else "withdrawal"
    )
    twd.to_parquet(clean_dir / "twd_transfer.parquet", index=False)
    print(f"  twd_transfer.parquet → clean/ ({len(twd):,} rows)")

    # ── crypto_transfer → clean (amount_asset, amount_twd_equiv, kind_label) ──
    crypto = pd.read_parquet(raw_dir / "crypto_transfer.parquet")
    crypto["ori_samount_raw"] = crypto.get("ori_samount", pd.Series(0, index=crypto.index)).copy()
    crypto["amount_asset"] = pd.to_numeric(crypto.get("ori_samount", 0), errors="coerce") * 1e-8
    crypto["twd_srate_raw"] = crypto.get("twd_srate", pd.Series(0, index=crypto.index)).copy()
    crypto["twd_rate"] = pd.to_numeric(crypto.get("twd_srate", 0), errors="coerce") * 1e-8
    crypto["amount_twd_equiv"] = crypto["amount_asset"] * crypto["twd_rate"]
    crypto["kind_label"] = crypto.get("kind", pd.Series(0, index=crypto.index)).apply(
        lambda x: "deposit" if int(x) == 0 else "withdrawal"
    )
    if "sub_kind" in crypto.columns:
        crypto["sub_kind_label"] = crypto["sub_kind"].apply(
            lambda x: "internal" if int(x) == 0 else "external"
        )
    # protocol_label: 0=SELF, 1=ERC20, 2=OMNI, 3=BNB, 4=TRC20, 5=BSC, 6=POLYGON
    _PROTOCOL_MAP = {0:"SELF",1:"ERC20",2:"OMNI",3:"BNB",4:"TRC20",5:"BSC",6:"POLYGON"}
    if "protocol" in crypto.columns:
        crypto["protocol_label"] = crypto["protocol"].apply(
            lambda x: _PROTOCOL_MAP.get(int(x), "UNKNOWN") if pd.notna(x) else "UNKNOWN"
        )
    else:
        crypto["protocol_label"] = "UNKNOWN"
    # is_internal_transfer: sub_kind == 0 (internal)
    if "sub_kind" in crypto.columns:
        crypto["is_internal_transfer"] = crypto["sub_kind"].apply(lambda x: int(x) == 0 if pd.notna(x) else False)
    else:
        crypto["is_internal_transfer"] = False
    crypto.to_parquet(clean_dir / "crypto_transfer.parquet", index=False)
    print(f"  crypto_transfer.parquet → clean/ ({len(crypto):,} rows)")

    # ── usdt_twd_trading → clean (trade_amount_usdt, twd_rate, kind_label) ──
    trading = pd.read_parquet(raw_dir / "usdt_twd_trading.parquet")
    trading["trade_samount_raw"] = trading.get("trade_samount", pd.Series(0, index=trading.index)).copy()
    trading["trade_amount_usdt"] = pd.to_numeric(trading.get("trade_samount", 0), errors="coerce") * 1e-8
    trading["twd_srate_raw"] = trading.get("twd_srate", pd.Series(0, index=trading.index)).copy()
    trading["twd_rate"] = pd.to_numeric(trading.get("twd_srate", 0), errors="coerce") * 1e-8
    trading["trade_notional_twd"] = trading["trade_amount_usdt"] * trading["twd_rate"]
    trading["kind_label"] = trading.get("is_buy", pd.Series(0, index=trading.index)).apply(
        lambda x: "buy" if int(x) == 1 else "sell"
    )
    trading["side_label"] = trading.get("is_buy", pd.Series(0, index=trading.index)).apply(
        lambda x: "buy_usdt_with_twd" if int(x) == 1 else "sell_usdt_for_twd"
    )
    trading["order_type_label"] = trading.get("is_market", pd.Series(0, index=trading.index)).apply(
        lambda x: "market" if int(x) == 1 else "limit"
    )
    _SOURCE_MAP = {0: "web", 1: "app", 2: "api"}
    trading["source_label"] = trading.get("source", pd.Series(0, index=trading.index)).apply(
        lambda x: _SOURCE_MAP.get(int(x), "unknown") if pd.notna(x) else "unknown"
    )
    trading.to_parquet(clean_dir / "usdt_twd_trading.parquet", index=False)
    print(f"  usdt_twd_trading.parquet → clean/ ({len(trading):,} rows)")

    # ── usdt_swap → clean (twd_amount, currency_amount, kind_label) ──
    swap = pd.read_parquet(raw_dir / "usdt_swap.parquet")
    swap["twd_samount_raw"] = swap.get("twd_samount", pd.Series(0, index=swap.index)).copy()
    swap["twd_amount"] = pd.to_numeric(swap.get("twd_samount", 0), errors="coerce") * 1e-8
    swap["currency_samount_raw"] = swap.get("currency_samount", pd.Series(0, index=swap.index)).copy()
    swap["currency_amount"] = pd.to_numeric(swap.get("currency_samount", 0), errors="coerce") * 1e-8
    # kind_label: buy_usdt_with_twd / sell_usdt_for_twd
    swap["kind_label"] = swap.get("is_buy", pd.Series(0, index=swap.index)).apply(
        lambda x: "buy_usdt_with_twd" if int(x) == 1 else "sell_usdt_for_twd"
    )
    swap.to_parquet(clean_dir / "usdt_swap.parquet", index=False)
    print(f"  usdt_swap.parquet → clean/ ({len(swap):,} rows)")

    # ── train_label, predict_label → 直接複製 ──
    for name in ("train_label", "predict_label"):
        src = raw_dir / f"{name}.parquet"
        if src.exists():
            import shutil as _shutil
            _shutil.copy2(src, clean_dir / f"{name}.parquet")
            print(f"  {name}.parquet → clean/")

    # 建立 user_index：合併 clean user_info（含 has_profile 等衍生欄位）+ train_label + predict_label
    user_info = pd.read_parquet(clean_dir / "user_info.parquet")
    train_label = pd.read_parquet(raw_dir / "train_label.parquet")
    predict_label = pd.read_parquet(raw_dir / "predict_label.parquet")

    # user_info 為基底
    user_index = user_info[["user_id"]].copy()
    user_index["user_id"] = pd.to_numeric(user_index["user_id"], errors="coerce").astype("Int64")

    # 合併 train_label → status 欄
    train_label["user_id"] = pd.to_numeric(train_label["user_id"], errors="coerce").astype("Int64")
    if "status" in train_label.columns:
        label_col = "status"
    elif "hidden_suspicious_label" in train_label.columns:
        label_col = "hidden_suspicious_label"
    else:
        label_col = train_label.columns[-1]
    user_index = user_index.merge(
        train_label[["user_id", label_col]].rename(columns={label_col: "status"}),
        on="user_id", how="left",
    )

    # 合併 predict_label → needs_prediction 欄
    predict_label["user_id"] = pd.to_numeric(predict_label["user_id"], errors="coerce").astype("Int64")
    predict_label["needs_prediction"] = True
    user_index = user_index.merge(
        predict_label[["user_id", "needs_prediction"]],
        on="user_id", how="left",
    )
    user_index["needs_prediction"] = user_index["needs_prediction"].fillna(False)

    # 合併 user_info 其他欄位
    user_info["user_id"] = pd.to_numeric(user_info["user_id"], errors="coerce").astype("Int64")
    extra_cols = [c for c in user_info.columns if c != "user_id" and c not in user_index.columns]
    if extra_cols:
        user_index = user_index.merge(user_info[["user_id"] + extra_cols], on="user_id", how="left")

    user_index.to_parquet(clean_dir / "user_index.parquet", index=False)
    print(f"  user_index.parquet: {len(user_index):,} users, {int(user_index['status'].notna().sum()):,} labeled, {int(user_index['needs_prediction'].sum()):,} predict")


def _verify_input_data() -> list[str]:
    """驗證 S3 輸入的 7 張 parquet 表都存在。

    Returns:
        找到的 parquet 檔案名稱清單。

    Raises:
        FileNotFoundError: 如果 /opt/ml/input/data/raw/ 不存在或無 parquet 檔。
    """
    if not SM_INPUT_RAW.exists():
        raise FileNotFoundError(
            f"輸入資料目錄不存在: {SM_INPUT_RAW}\n"
            "請確認 launch_training.py 的 S3 channel 設定正確。"
        )

    parquet_files = sorted(SM_INPUT_RAW.glob("*.parquet"))
    if not parquet_files:
        # 有些 S3 結構會多一層子目錄
        parquet_files = sorted(SM_INPUT_RAW.rglob("*.parquet"))

    if not parquet_files:
        raise FileNotFoundError(
            f"在 {SM_INPUT_RAW} 找不到任何 .parquet 檔。\n"
            "預期 7 張表: user_info, twd_transfer, crypto_transfer, "
            "usdt_twd_trading, usdt_swap, train_label, predict_label"
        )

    names = [f.name for f in parquet_files]
    print(f"[資料驗證] 找到 {len(parquet_files)} 個 parquet 檔:")
    for n in names:
        print(f"  - {n}")

    expected = {
        "user_info", "twd_transfer", "crypto_transfer",
        "usdt_twd_trading", "usdt_swap", "train_label", "predict_label",
    }
    found_stems = {f.stem for f in parquet_files}
    missing = expected - found_stems
    if missing:
        print(f"[警告] 缺少表: {missing}")

    return names


def _copy_artifacts_to_model_dir() -> None:
    """將 artifacts 複製到 /opt/ml/model/，讓 SageMaker 打包成 model.tar.gz。

    model.tar.gz 內容（推論時解壓到 /opt/ml/model/）：
      - official_bundle.json       ← 模型路徑 + 特徵欄位 + 編碼資訊
      - models/*.pkl               ← 所有模型檔（CatBoost, XGBoost, stacker, calibrator）
      - serve_e15.py               ← Flask 推論伺服器
      - bitoguard_core/            ← 完整原始碼（推論需要 import）
    """
    print("\n[複製 Artifacts] 開始複製到 /opt/ml/model/ ...")

    # 1. 複製 bundle JSON
    bundle_path = SM_ARTIFACT_DIR / "official_bundle.json"
    if bundle_path.exists():
        shutil.copy2(bundle_path, SM_MODEL_DIR / "official_bundle.json")
        print(f"  ✓ official_bundle.json")
    else:
        print(f"  ✗ 找不到 official_bundle.json（路徑: {bundle_path}）")

    # 2. 複製所有模型檔
    models_src = SM_ARTIFACT_DIR / "models"
    models_dst = SM_MODEL_DIR / "models"
    if models_src.exists():
        shutil.copytree(models_src, models_dst, dirs_exist_ok=True)
        n_files = len(list(models_dst.rglob("*")))
        print(f"  ✓ models/ ({n_files} 個檔案)")
    else:
        print(f"  ✗ 找不到 models/ 目錄（路徑: {models_src}）")

    # 3. 複製特徵 parquet（推論時 C&S 需要完整 graph + label_free_frame）
    feat_src = SM_ARTIFACT_DIR / "official_features"
    feat_dst = SM_MODEL_DIR / "official_features"
    if feat_src.exists():
        shutil.copytree(feat_src, feat_dst, dirs_exist_ok=True)
        print(f"  ✓ official_features/")

    # 4. 複製報告
    report_src = SM_ARTIFACT_DIR / "reports"
    report_dst = SM_MODEL_DIR / "reports"
    if report_src.exists():
        shutil.copytree(report_src, report_dst, dirs_exist_ok=True)
        print(f"  ✓ reports/")

    # 5. 複製 serve_e15.py（推論容器入口）
    serve_src = Path("/opt/ml/code/ml_pipeline/serve_e15.py")
    if serve_src.exists():
        shutil.copy2(serve_src, SM_MODEL_DIR / "serve_e15.py")
        print(f"  ✓ serve_e15.py")

    # 6. 複製 bitoguard_core 原始碼（推論需要 import official/、shared/ 等模組）
    code_src = Path("/opt/ml/code")
    code_dst = SM_MODEL_DIR / "code"
    # 只複製 Python 原始碼，跳過 .venv、__pycache__、.git
    ignore = shutil.ignore_patterns(
        ".venv", "__pycache__", ".git", ".pytest_cache",
        "*.egg-info", "catboost_info", "artifacts", "tests",
    )
    shutil.copytree(code_src, code_dst, ignore=ignore, dirs_exist_ok=True)
    print(f"  ✓ code/ (bitoguard_core 原始碼)")


def _copy_predictions_to_output() -> None:
    """將 submission CSV 複製到 /opt/ml/output/data/。"""
    pred_src = SM_ARTIFACT_DIR / "predictions"
    if not pred_src.exists():
        print("[警告] 找不到 predictions/ 目錄，跳過 submission 輸出")
        return

    for csv_file in pred_src.glob("*.csv"):
        dst = SM_OUTPUT_DIR / csv_file.name
        shutil.copy2(csv_file, dst)
        print(f"  ✓ 已複製 {csv_file.name} → {dst}")


def _write_training_metadata(result: dict, elapsed_s: float) -> None:
    """寫入訓練 metadata JSON 到 /opt/ml/model/。"""
    metadata = {
        "pipeline": "E15",
        "training_date": datetime.now(timezone.utc).isoformat(),
        "sagemaker_job": os.environ.get("TRAINING_JOB_NAME", "local"),
        "elapsed_seconds": round(elapsed_s, 1),
        "result_summary": {
            k: v for k, v in result.items()
            if isinstance(v, (str, int, float, bool))
        },
    }
    meta_path = SM_MODEL_DIR / "training_metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
    print(f"  ✓ training_metadata.json")


def main() -> None:
    """E15 SageMaker Training 主入口。"""
    print("=" * 72)
    print("BitoGuard E15 — SageMaker Training Entry Point")
    print(f"時間: {datetime.now(timezone.utc).isoformat()}")
    print(f"Job: {os.environ.get('TRAINING_JOB_NAME', 'local')}")
    print("=" * 72)

    # Step 1: 設定環境
    print("\n[Step 1/5] 設定環境變數 ...")
    _setup_env()
    print(f"  RAW_DIR      = {os.environ['BITOGUARD_AWS_EVENT_RAW_DIR']}")
    print(f"  CLEAN_DIR    = {os.environ['BITOGUARD_AWS_EVENT_CLEAN_DIR']}")
    print(f"  ARTIFACT_DIR = {os.environ['BITOGUARD_ARTIFACT_DIR']}")

    # Step 2: 驗證輸入資料
    print("\n[Step 2/6] 驗證輸入資料 ...")
    _verify_input_data()

    # Step 2.5: 建立 clean 表
    print("\n[Step 3/6] 建立 clean 表 + user_index ...")
    _prepare_clean_tables()

    # Step 3: 跑完整 E15 pipeline
    print("\n[Step 3/5] 執行 E15 Pipeline ...")
    print("=" * 72)
    t0 = time.time()

    try:
        from official.pipeline import run_official_pipeline

        result = run_official_pipeline()
    except Exception:
        print("\n[錯誤] E15 Pipeline 執行失敗！")
        traceback.print_exc()
        sys.exit(1)

    elapsed = time.time() - t0
    print(f"\n[Pipeline 完成] 耗時 {elapsed:.1f} 秒")

    # Step 4: 複製 artifacts 到 SageMaker 輸出目錄
    print("\n[Step 4/5] 複製 Artifacts ...")
    _copy_artifacts_to_model_dir()
    _write_training_metadata(result, elapsed)

    # Step 5: 複製 submission CSV
    print("\n[Step 5/5] 複製 Submission CSV ...")
    _copy_predictions_to_output()

    # 完成
    print("\n" + "=" * 72)
    print("E15 SageMaker Training 完成！")
    print(f"  模型產出: {SM_MODEL_DIR}")
    print(f"  Submission: {SM_OUTPUT_DIR}")
    print("=" * 72)


if __name__ == "__main__":
    main()
