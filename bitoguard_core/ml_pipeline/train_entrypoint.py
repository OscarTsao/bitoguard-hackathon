"""
SageMaker training entry point.

This script is executed by SageMaker training jobs to train models.
It integrates with existing training modules and saves artifacts to S3.
"""
import os
import sys
import json
import shutil
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

# Add parent directory to path for imports
sys.path.insert(0, '/opt/ml/code')

from models.train import train_model
from models.train_catboost import train_catboost_model as train_catboost
from models.anomaly import train_anomaly_model as train_anomaly
from ml_pipeline.config_loader import get_config


def parse_args(args=None):
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='SageMaker training entry point')
    
    # Model type
    parser.add_argument(
        '--model_type',
        type=str,
        required=True,
        choices=['lgbm', 'catboost', 'iforest', 'stacker'],
        help='Type of model to train'
    )
    
    # SageMaker directories
    parser.add_argument(
        '--input_data',
        type=str,
        default='/opt/ml/input/data/training',
        help='Directory containing training data'
    )
    
    parser.add_argument(
        '--model_dir',
        type=str,
        default='/opt/ml/model',
        help='Directory to save model artifacts'
    )
    
    parser.add_argument(
        '--output_data',
        type=str,
        default='/opt/ml/output',
        help='Directory for additional output'
    )

    parser.add_argument(
        '--use_s3_data',
        action='store_true',
        default=False,
        help='Load training data from S3 input path instead of DuckDB'
    )

    # Optional: Override hyperparameters
    parser.add_argument(
        '--hyperparameters',
        type=str,
        default='{}',
        help='JSON string of hyperparameters to override config'
    )
    
    # K-fold CV for stacker
    parser.add_argument(
        '--n_folds',
        type=int,
        default=5,
        help='Number of folds for stacker cross-validation (default: 5)'
    )
    
    # Hyperparameters for tuning (LightGBM and CatBoost)
    parser.add_argument('--learning_rate', type=float, default=None, help='Learning rate')
    parser.add_argument('--num_leaves', type=int, default=None, help='Number of leaves (LightGBM)')
    parser.add_argument('--n_estimators', type=int, default=None, help='Number of estimators')
    parser.add_argument('--subsample', type=float, default=None, help='Subsample ratio')
    parser.add_argument('--colsample_bytree', type=float, default=None, help='Column subsample ratio')
    parser.add_argument('--min_data_in_leaf', type=int, default=None, help='Minimum data in leaf (LightGBM)')
    parser.add_argument('--max_depth', type=int, default=None, help='Maximum tree depth')
    parser.add_argument('--reg_alpha', type=float, default=None, help='L1 regularization')
    parser.add_argument('--reg_lambda', type=float, default=None, help='L2 regularization')
    
    # CatBoost specific
    parser.add_argument('--depth', type=int, default=None, help='Tree depth (CatBoost)')
    parser.add_argument('--l2_leaf_reg', type=float, default=None, help='L2 regularization (CatBoost)')
    
    return parser.parse_args(args)


def load_hyperparameters(model_type: str, override_json: str, args) -> Dict[str, Any]:
    """
    Load hyperparameters from config and apply overrides.
    
    Args:
        model_type: Model type (lgbm, catboost, iforest)
        override_json: JSON string of hyperparameters to override
        args: Command line arguments with hyperparameter values
        
    Returns:
        Dictionary of hyperparameters
    """
    # Load from SSM Parameter Store
    try:
        config = get_config()
        hyperparams = config.get_training_config(model_type)
    except Exception as e:
        print(f"Warning: Failed to load config from SSM: {e}")
        hyperparams = {}
    
    # Apply JSON overrides
    if override_json and override_json != '{}':
        overrides = json.loads(override_json)
        hyperparams.update(overrides)
    
    # Apply command-line hyperparameter overrides (for tuning)
    tuning_params = {
        'learning_rate': args.learning_rate,
        'num_leaves': args.num_leaves,
        'n_estimators': args.n_estimators,
        'subsample': args.subsample,
        'colsample_bytree': args.colsample_bytree,
        'min_data_in_leaf': args.min_data_in_leaf,
        'max_depth': args.max_depth,
        'reg_alpha': args.reg_alpha,
        'reg_lambda': args.reg_lambda,
        'depth': args.depth,
        'l2_leaf_reg': args.l2_leaf_reg
    }
    
    # Only add non-None values
    for key, value in tuning_params.items():
        if value is not None:
            hyperparams[key] = value
    
    return hyperparams


def train_lgbm(hyperparams: Dict[str, Any]) -> Dict[str, Any]:
    """
    Train LightGBM model.
    
    Args:
        hyperparams: Hyperparameters for training
        
    Returns:
        Training result dictionary
    """
    print(f"Training LightGBM with hyperparameters: {hyperparams}")
    
    # Set environment variables for hyperparameters if needed
    # The existing train_model() function may read from config
    for key, value in hyperparams.items():
        os.environ[f"LGBM_{key.upper()}"] = str(value)
    
    # Call existing training function
    result = train_model()
    
    # Print metrics in SageMaker-compatible format for hyperparameter tuning
    if 'precision_at_100' in result:
        print(f"precision_at_100: {result['precision_at_100']}")
    if 'validation_logloss' in result:
        print(f"valid_logloss: {result['validation_logloss']}")
    if 'auc' in result:
        print(f"auc: {result['auc']}")
    
    return result


def train_catboost_model(hyperparams: Dict[str, Any]) -> Dict[str, Any]:
    """
    Train CatBoost model.
    
    Args:
        hyperparams: Hyperparameters for training
        
    Returns:
        Training result dictionary
    """
    print(f"Training CatBoost with hyperparameters: {hyperparams}")
    
    # Set environment variables for hyperparameters if needed
    for key, value in hyperparams.items():
        os.environ[f"CATBOOST_{key.upper()}"] = str(value)
    
    # Call existing training function
    result = train_catboost()
    
    # Print metrics in SageMaker-compatible format for hyperparameter tuning
    if 'precision_at_100' in result:
        print(f"precision_at_100: {result['precision_at_100']}")
    if 'validation_logloss' in result:
        print(f"valid_logloss: {result['validation_logloss']}")
    if 'auc' in result:
        print(f"auc: {result['auc']}")
    
    return result


def train_iforest(hyperparams: Dict[str, Any]) -> Dict[str, Any]:
    """
    Train IsolationForest model.
    
    Args:
        hyperparams: Hyperparameters for training
        
    Returns:
        Training result dictionary
    """
    print(f"Training IsolationForest with hyperparameters: {hyperparams}")
    
    # Call existing training function
    result = train_anomaly()
    
    return result


def train_stacker_model(n_folds: int = 5) -> Dict[str, Any]:
    """
    Train ensemble stacker with k-fold cross-validation.
    
    Args:
        n_folds: Number of folds for cross-validation
        
    Returns:
        Training result dictionary
    """
    print(f"Training Stacker with {n_folds}-fold cross-validation")
    
    # Import stacker training function
    from models.stacker import train_stacker
    
    # Call existing stacker training function
    result = train_stacker(n_folds=n_folds)
    
    # Print metrics in SageMaker-compatible format
    print(f"stacker_version: {result['stacker_version']}")
    print(f"n_folds: {n_folds}")
    
    return result


def save_model_artifacts(model_type: str, model_dir: Path, result: Dict[str, Any]):
    """
    Save model artifacts to SageMaker model directory.
    
    Args:
        model_type: Model type (lgbm, catboost, iforest)
        model_dir: Directory to save artifacts
        result: Training result dictionary
    """
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy trained model files from artifacts directory
    artifacts_dir = Path('artifacts/models')
    
    if not artifacts_dir.exists():
        raise FileNotFoundError(f"Artifacts directory not found: {artifacts_dir}")
    
    # Find model files matching the type
    model_files = list(artifacts_dir.glob(f"{model_type}_*"))
    
    if not model_files:
        raise FileNotFoundError(f"No model files found for {model_type} in {artifacts_dir}")
    
    print(f"Found {len(model_files)} model artifact files")
    
    # Copy all matching files
    for artifact_file in model_files:
        dest_file = model_dir / artifact_file.name
        shutil.copy2(artifact_file, dest_file)
        print(f"Copied {artifact_file.name} to {dest_file}")
    
    # Generate model version
    model_version = f"{model_type}_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}"
    
    # Write training metadata
    metadata = {
        "model_type": model_type,
        "model_version": model_version,
        "training_date": datetime.utcnow().isoformat(),
        "sagemaker_job": os.environ.get('TRAINING_JOB_NAME', 'local'),
        "training_result": result
    }
    
    metadata_file = model_dir / 'metadata.json'
    metadata_file.write_text(json.dumps(metadata, indent=2))
    print(f"Wrote metadata to {metadata_file}")
    
    # Write success marker
    success_file = model_dir / 'SUCCESS'
    success_file.write_text(f"Training completed successfully at {datetime.utcnow().isoformat()}")
    
    print(f"Model artifacts saved to {model_dir}")


def load_training_data_from_path(input_data_path: str) -> 'pd.DataFrame':
    """Load training DataFrame from Parquet files at the given directory.

    SageMaker places channel data at /opt/ml/input/data/<channel>.
    Ref: https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-training-algo-running-container.html
    """
    import glob
    import pandas as pd

    parquet_files = glob.glob(f"{input_data_path}/**/*.parquet", recursive=True)
    if not parquet_files:
        parquet_files = glob.glob(f"{input_data_path}/*.parquet")
    if not parquet_files:
        raise FileNotFoundError(f"No Parquet files found in {input_data_path}")
    return pd.concat([pd.read_parquet(f) for f in parquet_files], ignore_index=True)


def main():
    """Main training entry point."""
    print("=" * 80)
    print("BitoGuard SageMaker Training Entry Point")
    print("=" * 80)
    
    # Parse arguments
    args = parse_args()

    training_df = None
    if args.use_s3_data:
        training_df = load_training_data_from_path(args.input_data)
        print(f"Loaded {len(training_df)} rows from S3 input: {args.input_data}")

    print(f"\nTraining Configuration:")
    print(f"  Model Type: {args.model_type}")
    print(f"  Input Data: {args.input_data}")
    print(f"  Model Dir: {args.model_dir}")
    print(f"  Output Data: {args.output_data}")
    
    # Load hyperparameters
    try:
        hyperparams = load_hyperparameters(args.model_type, args.hyperparameters, args)
        print(f"\nHyperparameters: {json.dumps(hyperparams, indent=2)}")
    except Exception as e:
        print(f"Warning: Failed to load hyperparameters from config: {e}")
        print("Using default hyperparameters from training modules")
        hyperparams = {}
    
    # Train model based on type
    print(f"\n{'=' * 80}")
    print(f"Starting {args.model_type.upper()} training...")
    print(f"{'=' * 80}\n")
    
    try:
        if args.model_type == 'lgbm':
            result = train_lgbm(hyperparams)
        elif args.model_type == 'catboost':
            result = train_catboost_model(hyperparams)
        elif args.model_type == 'iforest':
            result = train_iforest(hyperparams)
        elif args.model_type == 'stacker':
            result = train_stacker_model(n_folds=args.n_folds)
        else:
            raise ValueError(f"Unknown model type: {args.model_type}")
        
        print(f"\n{'=' * 80}")
        print("Training completed successfully!")
        print(f"{'=' * 80}\n")
        print(f"Training result: {json.dumps(result, indent=2)}")
        
    except Exception as e:
        print(f"\n{'=' * 80}")
        print(f"ERROR: Training failed!")
        print(f"{'=' * 80}\n")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Save model artifacts to SageMaker model directory
    print(f"\n{'=' * 80}")
    print("Saving model artifacts...")
    print(f"{'=' * 80}\n")
    
    try:
        model_dir = Path(args.model_dir)
        save_model_artifacts(args.model_type, model_dir, result)
        
        print(f"\n{'=' * 80}")
        print("Model artifacts saved successfully!")
        print(f"{'=' * 80}\n")
        
    except Exception as e:
        print(f"\n{'=' * 80}")
        print(f"ERROR: Failed to save model artifacts!")
        print(f"{'=' * 80}\n")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print(f"\n{'=' * 80}")
    print("SageMaker training completed successfully!")
    print(f"{'=' * 80}\n")


if __name__ == '__main__':
    main()
