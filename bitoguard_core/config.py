from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_SOURCE_URL = "http://127.0.0.1:8000"
DEFAULT_ORACLE_DIR = ROOT_DIR.parent / "bitoguard_sim_output"
DEFAULT_DB_PATH = ROOT_DIR / "artifacts" / "bitoguard.duckdb"
DEFAULT_ARTIFACT_DIR = ROOT_DIR / "artifacts"


@dataclass(frozen=True)
class Settings:
    source_url: str
    oracle_dir: Path
    db_path: Path
    artifact_dir: Path
    label_source: str
    internal_api_port: int


def load_settings() -> Settings:
    artifact_dir = Path(os.getenv("BITOGUARD_ARTIFACT_DIR", str(DEFAULT_ARTIFACT_DIR))).resolve()
    artifact_dir.mkdir(parents=True, exist_ok=True)
    db_path = Path(os.getenv("BITOGUARD_DB_PATH", str(DEFAULT_DB_PATH))).resolve()
    if not db_path.parent.exists():
        db_path.parent.mkdir(parents=True, exist_ok=True)
    return Settings(
        source_url=os.getenv("BITOGUARD_SOURCE_URL", DEFAULT_SOURCE_URL).rstrip("/"),
        oracle_dir=Path(os.getenv("BITOGUARD_ORACLE_DIR", str(DEFAULT_ORACLE_DIR))).resolve(),
        db_path=db_path,
        artifact_dir=artifact_dir,
        label_source=os.getenv("BITOGUARD_LABEL_SOURCE", "hidden_suspicious_label"),
        internal_api_port=int(os.getenv("BITOGUARD_INTERNAL_API_PORT", "8001")),
    )
