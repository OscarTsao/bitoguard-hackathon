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
