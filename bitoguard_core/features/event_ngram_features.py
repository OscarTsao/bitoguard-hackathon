"""Event n-gram and transition matrix features.

Encodes each user's event history as an ordered sequence of (event_type, time_bucket)
tokens, then extracts:
  - Bigram/trigram frequency features (most common AML-relevant patterns)
  - Transition probability matrix entropy (chaotic vs predictable behavior)
  - Longest streak of same-type events
  - Directional flow pattern (in→out ratio in sequence)

These features capture temporal ordering that aggregate statistics miss.

Column name handling:
  - fiat: uses 'occurred_at' or 'created_at'; 'direction' or 'kind_label' for type
  - crypto: uses 'occurred_at' or 'created_at'; 'direction' or 'kind_label'
  - trades: uses 'occurred_at', 'updated_at', or 'created_at'; 'side' or 'side_label';
            'order_type' or 'order_type_label' (instant_swap → swap, else trade)
"""
from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd


# Canonical event type tokens
INFLOW_TOKENS = frozenset({"FD", "CD", "TB", "SB"})
OUTFLOW_TOKENS = frozenset({"FW", "CW", "TS", "SS"})

# AML-relevant bigrams to count explicitly
AML_BIGRAMS: list[tuple[str, str]] = [
    ("FD", "SB"),  # fiat deposit → swap buy (classic layering)
    ("FD", "CW"),  # fiat deposit → crypto withdrawal (cash-out)
    ("SB", "CW"),  # swap buy → crypto withdrawal (convert & move)
    ("FD", "FW"),  # fiat deposit → fiat withdrawal (pass-through)
    ("CD", "FW"),  # crypto deposit → fiat withdrawal (cash-out reverse)
    ("CD", "TS"),  # crypto deposit → trade sell
    ("FD", "TB"),  # fiat deposit → trade buy
]

# AML-relevant trigrams
AML_TRIGRAMS: list[tuple[str, str, str]] = [
    ("FD", "SB", "CW"),  # fiat → swap → crypto out (full layering chain)
    ("FD", "TB", "CW"),  # fiat → trade → crypto out
    ("CD", "TS", "FW"),  # crypto in → sell → fiat out
    ("FD", "FW", "FD"),  # repeated fiat pass-through
]

# Column name aliases
_TIME_COLS = ("occurred_at", "created_at", "updated_at")
_DIR_COLS_FIAT = ("direction", "kind_label")
_DIR_COLS_TRADE = ("side", "side_label")
_TYPE_COLS_TRADE = ("order_type", "order_type_label")

_FIAT_DIR_MAP = {"deposit": "FD", "withdrawal": "FW"}
_CRYPTO_DIR_MAP = {"deposit": "CD", "withdrawal": "CW"}
_TRADE_SIDE_BUY = {"buy", "buy_usdt_with_twd"}
_TRADE_SIDE_SELL = {"sell", "sell_usdt_for_twd"}
_SWAP_ORDER_TYPES = {"instant_swap", "swap"}


def _find_col(df: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _build_event_index(
    fiat: pd.DataFrame,
    crypto: pd.DataFrame,
    trades: pd.DataFrame,
) -> dict[object, list[tuple[np.int64, str]]]:
    """Build a per-user event index: user_id → sorted list of (timestamp_ns, token).

    Uses vectorized operations + groupby for efficiency.
    """
    events_parts: list[pd.DataFrame] = []

    # Fiat events
    if not fiat.empty and "user_id" in fiat.columns:
        time_col = _find_col(fiat, _TIME_COLS)
        dir_col = _find_col(fiat, _DIR_COLS_FIAT)
        if time_col and dir_col:
            f = fiat[["user_id", time_col, dir_col]].copy()
            f["ts"] = pd.to_datetime(f[time_col], utc=True, errors="coerce")
            f["token"] = f[dir_col].map(_FIAT_DIR_MAP)
            events_parts.append(f[["user_id", "ts", "token"]].dropna(subset=["ts", "token"]))

    # Crypto events
    if not crypto.empty and "user_id" in crypto.columns:
        time_col = _find_col(crypto, _TIME_COLS)
        dir_col = _find_col(crypto, _DIR_COLS_FIAT)
        if time_col and dir_col:
            c = crypto[["user_id", time_col, dir_col]].copy()
            c["ts"] = pd.to_datetime(c[time_col], utc=True, errors="coerce")
            c["token"] = c[dir_col].map(_CRYPTO_DIR_MAP)
            events_parts.append(c[["user_id", "ts", "token"]].dropna(subset=["ts", "token"]))

    # Trade events
    if not trades.empty and "user_id" in trades.columns:
        time_col = _find_col(trades, _TIME_COLS)
        side_col = _find_col(trades, _DIR_COLS_TRADE)
        type_col = _find_col(trades, _TYPE_COLS_TRADE)
        if time_col and side_col:
            t = trades[["user_id", time_col, side_col] + ([type_col] if type_col else [])].copy()
            t["ts"] = pd.to_datetime(t[time_col], utc=True, errors="coerce")
            side_vals = t[side_col].str.lower() if t[side_col].dtype == object else t[side_col]
            is_buy = side_vals.isin(_TRADE_SIDE_BUY)
            is_sell = side_vals.isin(_TRADE_SIDE_SELL)
            if type_col:
                is_swap = t[type_col].str.lower().isin(_SWAP_ORDER_TYPES)
            else:
                is_swap = pd.Series(False, index=t.index)
            t["token"] = np.where(
                is_swap & is_buy, "SB",
                np.where(is_swap & is_sell, "SS",
                np.where(is_buy, "TB",
                np.where(is_sell, "TS", None)))
            )
            events_parts.append(t[["user_id", "ts", "token"]].dropna(subset=["ts", "token"]))

    if not events_parts:
        return {}

    all_events = pd.concat(events_parts, ignore_index=True)
    all_events = all_events.sort_values(["user_id", "ts"])

    index: dict[object, list[tuple[np.int64, str]]] = {}
    for uid, grp in all_events.groupby("user_id"):
        index[uid] = list(zip(grp["ts"].values, grp["token"].values))
    return index


def _transition_entropy(sequence: list[str]) -> float:
    """Shannon entropy of the transition probability matrix (averaged over source states)."""
    if len(sequence) < 2:
        return 0.0
    transitions: dict[str, Counter] = {}
    for i in range(len(sequence) - 1):
        src, dst = sequence[i], sequence[i + 1]
        if src not in transitions:
            transitions[src] = Counter()
        transitions[src][dst] += 1
    total_entropy = 0.0
    for counter in transitions.values():
        total = sum(counter.values())
        for count in counter.values():
            if count > 0:
                p = count / total
                total_entropy -= p * np.log2(p)
    return total_entropy / max(1, len(transitions))


def _longest_same_streak(sequence: list[str]) -> int:
    """Longest consecutive run of the same event type."""
    if not sequence:
        return 0
    max_streak, current = 1, 1
    for i in range(1, len(sequence)):
        if sequence[i] == sequence[i - 1]:
            current += 1
            max_streak = max(max_streak, current)
        else:
            current = 1
    return max_streak


def compute_event_ngram_features(
    fiat: pd.DataFrame,
    crypto: pd.DataFrame,
    trades: pd.DataFrame,
) -> pd.DataFrame:
    """Compute event n-gram features for all users.

    Returns DataFrame with one row per user_id, columns:
      seq_length, bg_*, tg_*, seq_transition_entropy,
      seq_longest_streak, seq_inflow_outflow_ratio,
      seq_n_unique_types, seq_outflow_fraction
    """
    event_index = _build_event_index(fiat, crypto, trades)
    if not event_index:
        return pd.DataFrame()

    rows: list[dict] = []
    for uid, events in sorted(event_index.items(), key=lambda x: x[0]):
        seq = [tok for _, tok in events]
        row: dict = {"user_id": uid, "seq_length": len(seq)}

        # Bigram counts
        bigrams: Counter = Counter()
        for i in range(len(seq) - 1):
            bigrams[(seq[i], seq[i + 1])] += 1
        for bg in AML_BIGRAMS:
            row[f"bg_{bg[0]}_{bg[1]}"] = bigrams.get(bg, 0)

        # Trigram counts
        trigrams: Counter = Counter()
        for i in range(len(seq) - 2):
            trigrams[(seq[i], seq[i + 1], seq[i + 2])] += 1
        for tg in AML_TRIGRAMS:
            row[f"tg_{tg[0]}_{tg[1]}_{tg[2]}"] = trigrams.get(tg, 0)

        # Sequence statistics
        row["seq_transition_entropy"] = _transition_entropy(seq)
        row["seq_longest_streak"] = _longest_same_streak(seq)
        n_inflow = sum(1 for e in seq if e in INFLOW_TOKENS)
        n_outflow = sum(1 for e in seq if e in OUTFLOW_TOKENS)
        row["seq_inflow_outflow_ratio"] = n_inflow / max(1, n_outflow)
        row["seq_n_unique_types"] = len(set(seq))
        row["seq_outflow_fraction"] = n_outflow / max(1, len(seq))
        rows.append(row)

    return pd.DataFrame(rows).fillna(0)
