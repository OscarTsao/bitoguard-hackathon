from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from official.common import bundle_file_path, save_json


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
    return bundle
