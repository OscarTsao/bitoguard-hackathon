from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_SOURCE_URL = "https://aws-event-api.bitopro.com"
DEFAULT_DB_PATH = ROOT_DIR / "artifacts" / "bitoguard.duckdb"
DEFAULT_ARTIFACT_DIR = ROOT_DIR / "artifacts"


@dataclass(frozen=True)
class Settings:
    source_url: str
    db_path: Path
    artifact_dir: Path
    aws_event_raw_dir: Path
    aws_event_clean_dir: Path
    label_source: str
    internal_api_port: int
    cors_origins: list[str]
    graph_max_nodes: int
    graph_max_edges: int
    # Graph feature safety flag.
    # When True (default), only trusted graph features are computed.
    # Unsafe features (shared_device_count, component_size, blacklist_1hop_count,
    # blacklist_2hop_count) are disabled until the graph recovery plan is executed.
    # See docs/GRAPH_TRUST_BOUNDARY.md and docs/GRAPH_RECOVERY_PLAN.md.
    graph_trusted_only: bool
    # Optional API key for X-API-Key header authentication.
    # When None (BITOGUARD_API_KEY unset), auth is disabled (dev mode).
    api_key: str | None
    # Module toggles for the deployed scoring path.
    m0_enabled: bool
    m1_enabled: bool
    m3_enabled: bool
    m4_enabled: bool
    m5_enabled: bool
    # Model backend selector: "legacy" (DuckDB stacker) or "official" (pre-computed official pipeline scores).
    model_backend: str


# Graph features disabled by default due to placeholder-device artifact (A7).
# See docs/GRAPH_TRUST_BOUNDARY.md for the full trust boundary specification.
UNSAFE_GRAPH_FEATURES: frozenset[str] = frozenset({
    "shared_device_count",
    "component_size",
    "blacklist_1hop_count",
    "blacklist_2hop_count",
})

TRUSTED_GRAPH_FEATURES: frozenset[str] = frozenset({
    "fan_out_ratio",
    "shared_wallet_count",
    "shared_bank_count",
})

# Known placeholder device IDs that must never become real graph identity nodes.
# MD5 hashes of sentinel values (0, "", "null", "unknown", "none").
PLACEHOLDER_DEVICE_IDS: frozenset[str] = frozenset({
    "dev_cfcd208495d565ef66e7dff9f98764da",  # MD5("0")
    "dev_d41d8cd98f00b204e9800998ecf8427e",  # MD5("")
    "dev_37a6259cc0c1dae299a7866489dff0bd",  # MD5("null")
    "dev_d8e8fca2dc0f896fd7cb4cb0031ba249",  # MD5("unknown")
})

# Maximum fraction of total users a single graph node may connect before being
# treated as a super-node artifact (default 1%).
SUPERNODE_USER_FRACTION_THRESHOLD: float = 0.01


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"false", "0", "no", "off"}


_LEGAL_MODEL_BACKENDS: frozenset[str] = frozenset({"legacy", "official"})


def _validated_model_backend(value: str) -> str:
    """Return *value* if it is a recognised model backend, otherwise raise ValueError."""
    normalised = value.strip().lower()
    if normalised not in _LEGAL_MODEL_BACKENDS:
        raise ValueError(
            f"Unrecognised BITOGUARD_MODEL_BACKEND={value!r}. "
            f"Legal values: {sorted(_LEGAL_MODEL_BACKENDS)}"
        )
    return normalised


def load_settings() -> Settings:
    artifact_dir = Path(os.getenv("BITOGUARD_ARTIFACT_DIR", str(DEFAULT_ARTIFACT_DIR))).resolve()
    artifact_dir.mkdir(parents=True, exist_ok=True)
    db_path = Path(os.getenv("BITOGUARD_DB_PATH", str(DEFAULT_DB_PATH))).resolve()
    if not db_path.parent.exists():
        db_path.parent.mkdir(parents=True, exist_ok=True)
    aws_event_root = ROOT_DIR.parent / "data" / "aws_event"
    aws_event_raw_dir = Path(
        os.getenv("BITOGUARD_AWS_EVENT_RAW_DIR", str(aws_event_root / "raw"))
    ).resolve()
    aws_event_clean_dir = Path(
        os.getenv("BITOGUARD_AWS_EVENT_CLEAN_DIR", str(aws_event_root / "clean"))
    ).resolve()
    aws_event_raw_dir.mkdir(parents=True, exist_ok=True)
    aws_event_clean_dir.mkdir(parents=True, exist_ok=True)
    graph_trusted_raw = os.getenv("BITOGUARD_GRAPH_FEATURES_TRUSTED_ONLY", "true").lower()
    cors_raw = os.getenv("BITOGUARD_CORS_ORIGINS", "http://localhost:3000")
    cors_origins = [origin.strip() for origin in cors_raw.split(",") if origin.strip()]
    return Settings(
        source_url=os.getenv("BITOGUARD_SOURCE_URL", DEFAULT_SOURCE_URL).rstrip("/"),
        db_path=db_path,
        artifact_dir=artifact_dir,
        aws_event_raw_dir=aws_event_raw_dir,
        aws_event_clean_dir=aws_event_clean_dir,
        label_source=os.getenv("BITOGUARD_LABEL_SOURCE", "hidden_suspicious_label"),
        internal_api_port=int(os.getenv("BITOGUARD_INTERNAL_API_PORT", "8001")),
        cors_origins=cors_origins,
        graph_max_nodes=int(os.getenv("BITOGUARD_GRAPH_MAX_NODES", "120")),
        graph_max_edges=int(os.getenv("BITOGUARD_GRAPH_MAX_EDGES", "240")),
        graph_trusted_only=graph_trusted_raw not in ("false", "0", "no"),
        api_key=os.getenv("BITOGUARD_API_KEY") or None,
        m0_enabled=_env_flag("BITOGUARD_M0_ENABLED", True),
        m1_enabled=_env_flag("BITOGUARD_M1_ENABLED", True),
        m3_enabled=_env_flag("BITOGUARD_M3_ENABLED", True),
        # M4 explicitly disabled: IsolationForest trained on v1 schema, incompatible with v2.
        # Re-enable after retraining on v2 features (negatives-only). See docs/GRAPH_RECOVERY_PLAN.md.
        m4_enabled=_env_flag("BITOGUARD_M4_ENABLED", False),
        m5_enabled=_env_flag("BITOGUARD_M5_ENABLED", False),
        model_backend=_validated_model_backend(os.getenv("BITOGUARD_MODEL_BACKEND", "legacy")),
    )
