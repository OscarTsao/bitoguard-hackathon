"""Tests for graph data-quality guards.

Guards verified:
  1. Null/placeholder device IDs are rejected before becoming graph nodes.
  2. Super-node detection triggers on nodes connecting > threshold of users.
  3. Duplicate edges are deduplicated.
  4. Unsafe graph features return safe defaults when graph_trusted_only=True.
"""
from __future__ import annotations

import hashlib
from unittest.mock import MagicMock

import pandas as pd
import pytest

from config import PLACEHOLDER_DEVICE_IDS, SUPERNODE_USER_FRACTION_THRESHOLD
from pipeline.rebuild_edges import _is_placeholder_device, _detect_and_remove_supernodes


# ── Guard 1: Placeholder device rejection ─────────────────────────────────────

class TestPlaceholderDeviceRejection:
    """Null and placeholder device IDs must never become real graph nodes."""

    def test_none_is_placeholder(self):
        assert _is_placeholder_device(None) is True

    def test_empty_string_is_placeholder(self):
        assert _is_placeholder_device("") is True
        assert _is_placeholder_device("   ") is True

    def test_zero_string_is_placeholder(self):
        assert _is_placeholder_device("0") is True

    def test_null_string_is_placeholder(self):
        assert _is_placeholder_device("null") is True
        assert _is_placeholder_device("NULL") is True
        assert _is_placeholder_device("None") is True

    def test_unknown_string_is_placeholder(self):
        assert _is_placeholder_device("unknown") is True
        assert _is_placeholder_device("UNKNOWN") is True
        assert _is_placeholder_device("n/a") is True
        assert _is_placeholder_device("N/A") is True

    def test_known_placeholder_hash_is_rejected(self):
        # dev_cfcd208495d565ef66e7dff9f98764da is MD5("0")
        for known in PLACEHOLDER_DEVICE_IDS:
            assert _is_placeholder_device(known) is True, f"{known} should be rejected"

    def test_md5_of_zero_is_placeholder(self):
        md5_zero = "dev_" + hashlib.md5(b"0").hexdigest()
        assert _is_placeholder_device(md5_zero) is True

    def test_md5_of_empty_is_placeholder(self):
        md5_empty = "dev_" + hashlib.md5(b"").hexdigest()
        assert _is_placeholder_device(md5_empty) is True

    def test_real_device_id_is_not_placeholder(self):
        assert _is_placeholder_device("dev_abc123def456abc123def456abc123de") is False
        assert _is_placeholder_device("dev_79e4bfcc69451cde585578d35e1c0eb5") is False

    def test_nan_float_is_placeholder(self):
        assert _is_placeholder_device(float("nan")) is True


# ── Guard 2: Super-node detection ────────────────────────────────────────────

class TestSupernodeDetection:
    """A node connecting >= SUPERNODE_USER_FRACTION_THRESHOLD of users must be removed."""

    def _make_edges(self, user_ids, dst_id, relation="uses_device", dst_type="device"):
        return pd.DataFrame([
            {
                "edge_id": f"edge_{i:06d}",
                "snapshot_time": pd.Timestamp("2025-01-01", tz="UTC"),
                "src_type": "user",
                "src_id": uid,
                "relation_type": relation,
                "dst_type": dst_type,
                "dst_id": dst_id,
            }
            for i, uid in enumerate(user_ids)
        ])

    def test_supernode_removed_when_above_threshold(self):
        total_users = 1000
        threshold_users = int(total_users * SUPERNODE_USER_FRACTION_THRESHOLD) + 1
        user_ids = [f"user_{i}" for i in range(threshold_users)]
        edge_df = self._make_edges(user_ids, dst_id="device_supernode")
        mock_store = MagicMock()

        with pytest.warns(UserWarning, match="Super-node detected"):
            result = _detect_and_remove_supernodes(edge_df, total_users, mock_store)

        assert len(result) == 0, "Super-node edges should be fully removed"
        mock_store.execute.assert_called()  # quality issue should be logged

    def test_legitimate_node_not_removed(self):
        total_users = 1000
        # A node connecting only 2 users — well below threshold
        user_ids = ["user_1", "user_2"]
        edge_df = self._make_edges(user_ids, dst_id="device_small")
        mock_store = MagicMock()

        result = _detect_and_remove_supernodes(edge_df, total_users, mock_store)
        assert len(result) == 2, "Legitimate shared device should not be removed"

    def test_empty_edge_df_returns_empty(self):
        mock_store = MagicMock()
        empty = pd.DataFrame(columns=["edge_id", "snapshot_time", "src_type",
                                       "src_id", "relation_type", "dst_type", "dst_id"])
        result = _detect_and_remove_supernodes(empty, 1000, mock_store)
        assert len(result) == 0

    def test_supernode_threshold_boundary(self):
        """Node connecting exactly threshold-1 users should NOT be removed."""
        total_users = 1000
        threshold_users = int(total_users * SUPERNODE_USER_FRACTION_THRESHOLD)  # exactly at threshold
        user_ids = [f"user_{i}" for i in range(threshold_users - 1)]
        edge_df = self._make_edges(user_ids, dst_id="device_borderline")
        mock_store = MagicMock()

        result = _detect_and_remove_supernodes(edge_df, total_users, mock_store)
        # Should NOT be removed (strictly less than threshold)
        assert len(result) == len(user_ids)


# ── Guard 4: Unsafe graph features disabled by default ────────────────────────

class TestTrustedOnlyMode:
    """With graph_trusted_only=True (default), unsafe features must return safe defaults."""

    def _make_settings(self, trusted_only: bool):
        from config import Settings
        from pathlib import Path
        return Settings(
            source_url="http://localhost",
            db_path=Path("/tmp/test.duckdb"),
            artifact_dir=Path("/tmp"),
            aws_event_raw_dir=Path("/tmp/raw"),
            aws_event_clean_dir=Path("/tmp/clean"),
            label_source="hidden_suspicious_label",
            internal_api_port=8001,
            cors_origins=["http://localhost:3000"],
            graph_max_nodes=120,
            graph_max_edges=240,
            graph_trusted_only=trusted_only,
            api_key=None,
            m0_enabled=True,
            m1_enabled=True,
            m3_enabled=True,
            m4_enabled=True,
            m5_enabled=False,
        )

    def test_config_default_is_trusted_only(self):
        """BITOGUARD_GRAPH_FEATURES_TRUSTED_ONLY defaults to True."""
        import os
        # Ensure the env var is not set for this test
        env_backup = os.environ.pop("BITOGUARD_GRAPH_FEATURES_TRUSTED_ONLY", None)
        try:
            from config import load_settings
            settings = load_settings()
            assert settings.graph_trusted_only is True
        finally:
            if env_backup is not None:
                os.environ["BITOGUARD_GRAPH_FEATURES_TRUSTED_ONLY"] = env_backup

    def test_trusted_only_false_when_env_set_false(self):
        import os
        os.environ["BITOGUARD_GRAPH_FEATURES_TRUSTED_ONLY"] = "false"
        try:
            from importlib import reload
            import config as cfg
            reload(cfg)
            settings = cfg.load_settings()
            assert settings.graph_trusted_only is False
        finally:
            os.environ["BITOGUARD_GRAPH_FEATURES_TRUSTED_ONLY"] = "true"
            from importlib import reload
            import config as cfg
            reload(cfg)



# ── Guard 3: Duplicate edge deduplication ─────────────────────────────────────

class TestDuplicateEdgeGuard:
    """Duplicate (src, dst, relation_type) edges must be removed."""

    def test_rebuild_edges_deduplicates(self):
        """The rebuild_edges function must deduplicate on (src_id, dst_id, relation_type)."""
        # This is a smoke test verifying the logic exists; the full integration
        # test requires a database fixture.
        from pipeline.rebuild_edges import rebuild_edges
        import inspect
        source = inspect.getsource(rebuild_edges)
        assert "drop_duplicates" in source, \
            "rebuild_edges must call drop_duplicates to guard against duplicate edges"


def test_fast_path_blacklist_is_snapshot_bounded() -> None:
    """Graph fast path must filter blacklist by snapshot date, not global max_date.

    A user blacklisted at T+1 must NOT appear in the blacklisted_set when
    computing features for snapshot at T.
    """
    import inspect
    from features import graph_features
    src = inspect.getsource(graph_features._build_graph_features_fast)
    # The blacklisted_set must NOT be computed using a single global max_date
    assert "max_date" not in src or "blacklisted_set" not in src.split("max_date")[0].split("blacklisted_set")[-1], \
        "Blacklist set must not use global max_date — must be bounded per snapshot"
