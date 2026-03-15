"""Tests for ml_pipeline/train_entrypoint.py"""
from __future__ import annotations
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def test_train_entrypoint_imports_without_error():
    """F1: train_entrypoint must import cleanly (no ImportError on train_catboost)."""
    import importlib
    try:
        mod = importlib.import_module("ml_pipeline.train_entrypoint")
        assert mod is not None
    except ImportError as e:
        raise AssertionError(f"train_entrypoint import failed: {e}") from e


def test_parse_args_model_type():
    """parse_args accepts all four model types."""
    from ml_pipeline.train_entrypoint import parse_args
    for model in ["lgbm", "catboost", "iforest", "stacker"]:
        args = parse_args(["--model_type", model])
        assert args.model_type == model


def test_parse_args_use_s3_data_flag():
    """F4: --use_s3_data flag is parsed correctly."""
    from ml_pipeline.train_entrypoint import parse_args

    args_with = parse_args(["--model_type", "lgbm", "--use_s3_data"])
    assert args_with.use_s3_data is True

    args_without = parse_args(["--model_type", "lgbm"])
    assert args_without.use_s3_data is False


def test_load_training_data_from_s3_path(tmp_path):
    """F4: load_training_data_from_path loads Parquet from the given directory."""
    import pandas as pd
    from ml_pipeline.train_entrypoint import load_training_data_from_path

    df = pd.DataFrame({"user_id": ["u1", "u2"], "feat_a": [1.0, 2.0],
                       "hidden_suspicious_label": [0, 1]})
    parquet_path = tmp_path / "data.parquet"
    df.to_parquet(parquet_path, index=False)

    result = load_training_data_from_path(str(tmp_path))
    assert len(result) == 2
    assert "user_id" in result.columns
    assert "hidden_suspicious_label" in result.columns
