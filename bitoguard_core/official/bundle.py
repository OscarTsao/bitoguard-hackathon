from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from config import load_settings
from official.common import bundle_file_path, save_json


def _remap_path(value: str, artifact_dir: Path, *fallback_dirs: Path) -> str:
    """Resolve a stored path to an absolute path on this machine.

    Handles three cases:
    1. Relative path (stored as 'models/foo.pkl') — resolve against artifact_dir.
    2. Absolute path that exists — return as-is.
    3. Absolute path from a foreign machine — try filename-only lookup in known local artifact directories.
    """
    p = Path(value)
    if not p.is_absolute():
        candidate = artifact_dir / p
        if candidate.exists():
            return str(candidate)
    if p.exists():
        return value
    for directory in fallback_dirs:
        candidate = directory / p.name
        if candidate.exists():
            return str(candidate)
    return value


REQUIRED_BUNDLE_KEYS = {
    "bundle_version",
    "selected_model",
    "primary_validation_protocol",
    "base_model_paths",
    "graph_model_path",
    "stacker_path",
    "shadow_protocol",
    "grouping_params",
}
READY_BUNDLE_KEYS = {
    "calibrator",
    "selected_threshold",
}


def save_selected_bundle(bundle: dict[str, Any], path: Path | None = None) -> Path:
    target = bundle_file_path(path)
    save_json(bundle, target)
    return target


def load_selected_bundle(path: Path | None = None, require_ready: bool = False) -> dict[str, Any]:
    target = bundle_file_path(path)
    if not target.exists():
        raise FileNotFoundError(target)
    bundle = json.loads(target.read_text(encoding="utf-8"))
    missing = sorted(REQUIRED_BUNDLE_KEYS.difference(bundle))
    if missing:
        raise ValueError(f"Bundle missing required keys: {', '.join(missing)}")
    if require_ready:
        missing_ready = sorted(key for key in READY_BUNDLE_KEYS if key not in bundle or bundle[key] in (None, "", {}))
        if missing_ready:
            raise ValueError(f"Bundle is not ready for scoring: missing {', '.join(missing_ready)}")
    # Remap hard-coded absolute paths from foreign machines to local artifact_dir
    settings = load_settings()
    artifact_dir = settings.artifact_dir
    model_dir = artifact_dir / "models"
    feature_dir = artifact_dir / "official_features"
    _path_keys = (
        "graph_model_path",
        "stacker_path",
        "oof_predictions_path",
        "primary_split_path",
        "primary_labeled_split_path",
        "model_meta_path",
    )
    for key in _path_keys:
        if isinstance(bundle.get(key), str):
            bundle[key] = _remap_path(bundle[key], artifact_dir, model_dir, feature_dir)
    if isinstance(bundle.get("base_model_paths"), dict):
        bundle["base_model_paths"] = {
            k: ([_remap_path(p, artifact_dir, model_dir) for p in v] if isinstance(v, list) else _remap_path(v, artifact_dir, model_dir))
            for k, v in bundle["base_model_paths"].items()
        }
    if isinstance(bundle.get("calibrator"), dict) and isinstance(bundle["calibrator"].get("calibrator_path"), str):
        bundle["calibrator"]["calibrator_path"] = _remap_path(
            bundle["calibrator"]["calibrator_path"], artifact_dir, model_dir
        )
    if isinstance(bundle.get("secondary_stress_summary"), dict):
        secondary_oof = bundle["secondary_stress_summary"].get("secondary_oof_predictions_path")
        if isinstance(secondary_oof, str):
            bundle["secondary_stress_summary"]["secondary_oof_predictions_path"] = _remap_path(
                secondary_oof,
                artifact_dir,
                feature_dir,
            )
    return bundle
