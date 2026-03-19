"""One-time script: rebuild entity_edges with placeholder device filtering.

Reads current canonical tables, re-derives edges with Guard 1 (placeholder
device rejection) and Guard 2 (super-node detection) active, and overwrites
canonical.entity_edges.

After running this script:
  1. Verify shared_device_count distribution is reasonable
  2. Set BITOGUARD_GRAPH_FEATURES_TRUSTED_ONLY=false (or m5_enabled=true)
  3. Re-run feature build and training

Usage:
    cd bitoguard_core && PYTHONPATH=. python scripts/fix_graph_a7.py
"""
from __future__ import annotations

import sys
sys.path.insert(0, ".")

from config import PLACEHOLDER_DEVICE_IDS, load_settings
from db.store import DuckDBStore


def main() -> None:
    settings = load_settings()
    store = DuckDBStore(settings.db_path)

    # Pre-check: count edges linked to placeholder devices
    try:
        before_df = store.fetch_df("""
            SELECT dst_id, COUNT(DISTINCT src_id) as user_count
            FROM canonical.entity_edges
            WHERE dst_type = 'device'
            GROUP BY dst_id
            ORDER BY user_count DESC
            LIMIT 10
        """)
        print("Top 10 device nodes by user count BEFORE fix:")
        print(before_df.to_string(index=False))

        total_edges = store.fetch_df("SELECT COUNT(*) as n FROM canonical.entity_edges")["n"].iloc[0]
        print(f"\nTotal edges before: {total_edges:,}")
    except Exception as e:
        print(f"Pre-check skipped: {e}")

    # Rebuild edges with guards active
    print("\nRebuilding entity_edges with placeholder filtering...")
    try:
        from pipeline.rebuild_edges import rebuild_entity_edges
        rebuild_entity_edges(store)
    except Exception as e:
        print(f"rebuild_entity_edges failed: {e}")
        print("Falling back to direct SQL deletion of placeholder device edges...")
        placeholder_list = ", ".join(f"'{p}'" for p in PLACEHOLDER_DEVICE_IDS)
        store.execute(f"""
            DELETE FROM canonical.entity_edges
            WHERE dst_type = 'device'
              AND dst_id IN ({placeholder_list})
        """)
        print("Direct deletion complete.")

    # Post-check
    try:
        after_df = store.fetch_df("""
            SELECT dst_id, COUNT(DISTINCT src_id) as user_count
            FROM canonical.entity_edges
            WHERE dst_type = 'device'
            GROUP BY dst_id
            ORDER BY user_count DESC
            LIMIT 10
        """)
        print("\nTop 10 device nodes by user count AFTER fix:")
        print(after_df.to_string(index=False))

        total_edges_after = store.fetch_df("SELECT COUNT(*) as n FROM canonical.entity_edges")["n"].iloc[0]
        print(f"\nTotal edges after: {total_edges_after:,}")

        # Check no placeholder devices remain
        placeholders = store.fetch_df(f"""
            SELECT dst_id, COUNT(*) as n
            FROM canonical.entity_edges
            WHERE dst_type = 'device'
              AND dst_id IN ({', '.join(f"'{p}'" for p in PLACEHOLDER_DEVICE_IDS)})
            GROUP BY dst_id
        """)
        if placeholders.empty:
            print("\nSUCCESS: No placeholder device IDs remain in entity_edges")
        else:
            print(f"\nWARNING: Placeholder devices still present:\n{placeholders}")
    except Exception as e:
        print(f"Post-check error: {e}")


if __name__ == "__main__":
    main()
