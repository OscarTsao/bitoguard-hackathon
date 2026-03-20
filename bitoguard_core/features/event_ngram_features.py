"""Event n-gram and transition matrix features.

Encodes each user's event history as an ordered sequence of event tokens, then extracts:
  - Bigram/trigram frequency features for AML-relevant patterns
  - Transition probability matrix entropy (chaotic vs predictable behavior)
  - Longest streak of same-type events
  - Directional flow features in the sequence

Column name handling:
  - fiat: uses "occurred_at" or "created_at"; "direction" or "kind_label" for type
  - crypto: uses "occurred_at" or "created_at"; "direction" or "kind_label"
  - trades: uses "occurred_at", "updated_at", or "created_at"; "side" or "side_label";
            "order_type" or "order_type_label" (instant_swap -> swap, else trade)
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# Canonical event type tokens
INFLOW_TOKENS = frozenset({"FD", "CD", "TB", "SB"})
OUTFLOW_TOKENS = frozenset({"FW", "CW", "TS", "SS"})

# AML-relevant bigrams to count explicitly
AML_BIGRAMS: list[tuple[str, str]] = [
    ("FD", "SB"),  # fiat deposit -> swap buy (classic layering)
    ("FD", "CW"),  # fiat deposit -> crypto withdrawal (cash-out)
    ("SB", "CW"),  # swap buy -> crypto withdrawal (convert & move)
    ("FD", "FW"),  # fiat deposit -> fiat withdrawal (pass-through)
    ("CD", "FW"),  # crypto deposit -> fiat withdrawal (cash-out reverse)
    ("CD", "TS"),  # crypto deposit -> trade sell
    ("FD", "TB"),  # fiat deposit -> trade buy
]

# AML-relevant trigrams
AML_TRIGRAMS: list[tuple[str, str, str]] = [
    ("FD", "SB", "CW"),  # fiat -> swap -> crypto out (full layering chain)
    ("FD", "TB", "CW"),  # fiat -> trade -> crypto out
    ("CD", "TS", "FW"),  # crypto in -> sell -> fiat out
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
_TOKEN_CATEGORIES = ["FD", "FW", "CD", "CW", "TB", "TS", "SB", "SS"]

_BIGRAM_COLUMNS = [f"bg_{src}_{dst}" for src, dst in AML_BIGRAMS]
_TRIGRAM_COLUMNS = [f"tg_{a}_{b}_{c}" for a, b, c in AML_TRIGRAMS]
_INT_COLUMNS = [
    "seq_length",
    *_BIGRAM_COLUMNS,
    *_TRIGRAM_COLUMNS,
    "seq_longest_streak",
    "seq_n_unique_types",
]
_FLOAT_COLUMNS = [
    "seq_transition_entropy",
    "seq_inflow_outflow_ratio",
    "seq_outflow_fraction",
]
_OUTPUT_COLUMNS = [
    "user_id",
    "seq_length",
    *_BIGRAM_COLUMNS,
    *_TRIGRAM_COLUMNS,
    "seq_transition_entropy",
    "seq_longest_streak",
    "seq_inflow_outflow_ratio",
    "seq_n_unique_types",
    "seq_outflow_fraction",
]


def _find_col(df: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    return None


def _empty_feature_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=_OUTPUT_COLUMNS)


def _prepare_events(
    fiat: pd.DataFrame,
    crypto: pd.DataFrame,
    trades: pd.DataFrame,
) -> pd.DataFrame:
    """Normalize all event sources into one sorted event frame."""
    events_parts: list[pd.DataFrame] = []

    if not fiat.empty and "user_id" in fiat.columns:
        time_col = _find_col(fiat, _TIME_COLS)
        dir_col = _find_col(fiat, _DIR_COLS_FIAT)
        if time_col and dir_col:
            fiat_events = fiat.loc[:, ["user_id", time_col, dir_col]].copy()
            fiat_events["ts"] = pd.to_datetime(fiat_events[time_col], utc=True, errors="coerce")
            fiat_events["token"] = fiat_events[dir_col].astype("string").str.lower().map(_FIAT_DIR_MAP)
            fiat_events["_source_rank"] = 0
            fiat_events["_source_order"] = np.arange(len(fiat_events), dtype=np.int64)
            events_parts.append(
                fiat_events[["user_id", "ts", "token", "_source_rank", "_source_order"]]
                .dropna(subset=["user_id", "ts", "token"])
            )

    if not crypto.empty and "user_id" in crypto.columns:
        time_col = _find_col(crypto, _TIME_COLS)
        dir_col = _find_col(crypto, _DIR_COLS_FIAT)
        if time_col and dir_col:
            crypto_events = crypto.loc[:, ["user_id", time_col, dir_col]].copy()
            crypto_events["ts"] = pd.to_datetime(crypto_events[time_col], utc=True, errors="coerce")
            crypto_events["token"] = crypto_events[dir_col].astype("string").str.lower().map(_CRYPTO_DIR_MAP)
            crypto_events["_source_rank"] = 1
            crypto_events["_source_order"] = np.arange(len(crypto_events), dtype=np.int64)
            events_parts.append(
                crypto_events[["user_id", "ts", "token", "_source_rank", "_source_order"]]
                .dropna(subset=["user_id", "ts", "token"])
            )

    if not trades.empty and "user_id" in trades.columns:
        time_col = _find_col(trades, _TIME_COLS)
        side_col = _find_col(trades, _DIR_COLS_TRADE)
        type_col = _find_col(trades, _TYPE_COLS_TRADE)
        if time_col and side_col:
            keep_cols = ["user_id", time_col, side_col]
            if type_col:
                keep_cols.append(type_col)
            trade_events = trades.loc[:, keep_cols].copy()
            trade_events["ts"] = pd.to_datetime(trade_events[time_col], utc=True, errors="coerce")
            side_values = trade_events[side_col].astype("string").str.lower()
            is_buy = side_values.isin(_TRADE_SIDE_BUY)
            is_sell = side_values.isin(_TRADE_SIDE_SELL)
            if type_col:
                is_swap = trade_events[type_col].astype("string").str.lower().isin(_SWAP_ORDER_TYPES)
            else:
                is_swap = pd.Series(False, index=trade_events.index)
            trade_events["token"] = np.where(
                is_swap & is_buy,
                "SB",
                np.where(
                    is_swap & is_sell,
                    "SS",
                    np.where(is_buy, "TB", np.where(is_sell, "TS", None)),
                ),
            )
            trade_events["_source_rank"] = 2
            trade_events["_source_order"] = np.arange(len(trade_events), dtype=np.int64)
            events_parts.append(
                trade_events[["user_id", "ts", "token", "_source_rank", "_source_order"]]
                .dropna(subset=["user_id", "ts", "token"])
            )

    if not events_parts:
        return pd.DataFrame(columns=["user_id", "ts", "token"])

    all_events = pd.concat(events_parts, ignore_index=True)
    all_events = all_events.sort_values(
        ["user_id", "ts", "_source_rank", "_source_order"],
        kind="mergesort",
    ).reset_index(drop=True)
    all_events["token"] = pd.Categorical(all_events["token"], categories=_TOKEN_CATEGORIES)
    return all_events[["user_id", "ts", "token"]]


def _transition_entropy(sequence: list[str]) -> float:
    """Shannon entropy of the transition probability matrix (averaged over source states)."""
    if len(sequence) < 2:
        return 0.0

    pairs = pd.DataFrame({"token": sequence[:-1], "next_token": sequence[1:]})
    counts = (
        pairs.groupby(["token", "next_token"], sort=False)
        .size()
        .rename("cnt")
        .reset_index()
    )
    counts["total"] = counts.groupby("token", sort=False)["cnt"].transform("sum")
    probs = counts["cnt"] / counts["total"]
    per_source = (-probs * np.log2(probs.clip(lower=np.finfo(float).tiny))).groupby(
        counts["token"],
        sort=False,
    ).sum()
    return float(per_source.mean()) if not per_source.empty else 0.0


def _longest_same_streak(sequence: list[str]) -> int:
    """Longest consecutive run of the same event type."""
    if not sequence:
        return 0

    seq = pd.Series(sequence)
    run_id = seq.ne(seq.shift()).cumsum()
    return int(seq.groupby(run_id, sort=False).size().max())


def _count_selected_ngrams(
    events: pd.DataFrame,
    token_cols: list[str],
    patterns: list[tuple[str, ...]],
    prefix: str,
) -> pd.DataFrame:
    """Count selected n-grams per user without any per-user Python iteration."""
    feature_names = [f"{prefix}_{'_'.join(pattern)}" for pattern in patterns]
    ngram_events = events.dropna(subset=token_cols)
    if ngram_events.empty:
        return pd.DataFrame(columns=feature_names)

    lookup = pd.DataFrame(patterns, columns=token_cols)
    lookup["feature"] = feature_names
    matched = ngram_events[["user_id", *token_cols]].merge(lookup, on=token_cols, how="inner", sort=False)
    if matched.empty:
        return pd.DataFrame(columns=feature_names)

    counts = matched.groupby(["user_id", "feature"], sort=False).size().unstack("feature", fill_value=0)
    return counts.reindex(columns=feature_names, fill_value=0)


def _vectorized_transition_entropy(events: pd.DataFrame) -> pd.Series:
    """Compute Shannon entropy of transition probabilities per user."""
    pairs = events.dropna(subset=["next_token"])[["user_id", "token", "next_token"]]
    if pairs.empty:
        return pd.Series(dtype=float, name="seq_transition_entropy")

    counts = (
        pairs.groupby(["user_id", "token", "next_token"], sort=False, observed=True)
        .size()
        .rename("cnt")
        .reset_index()
    )
    counts["total"] = counts.groupby(["user_id", "token"], sort=False, observed=True)["cnt"].transform("sum")
    probs = counts["cnt"] / counts["total"]
    counts["entropy"] = -probs * np.log2(probs.clip(lower=np.finfo(float).tiny))
    per_source = counts.groupby(["user_id", "token"], sort=False, observed=True)["entropy"].sum()
    return per_source.groupby(level="user_id", sort=False, observed=True).mean().rename(
        "seq_transition_entropy"
    )


def compute_event_ngram_features(
    fiat: pd.DataFrame,
    crypto: pd.DataFrame,
    trades: pd.DataFrame,
) -> pd.DataFrame:
    """Compute event n-gram features for all users with vectorized pandas operations."""
    all_events = _prepare_events(fiat, crypto, trades)
    if all_events.empty:
        return _empty_feature_frame()

    all_events = all_events.copy()
    all_events["next_token"] = all_events.groupby("user_id", sort=False)["token"].shift(-1)
    all_events["next2_token"] = all_events.groupby("user_id", sort=False)["token"].shift(-2)

    user_index = pd.Index(all_events["user_id"].drop_duplicates(), name="user_id")
    result = pd.DataFrame(index=user_index)

    seq_length = all_events.groupby("user_id", sort=False).size().rename("seq_length")
    seq_n_unique_types = all_events.groupby("user_id", sort=False)["token"].nunique().rename("seq_n_unique_types")

    inflow_counts = all_events["token"].isin(INFLOW_TOKENS).groupby(all_events["user_id"], sort=False).sum()
    outflow_counts = all_events["token"].isin(OUTFLOW_TOKENS).groupby(all_events["user_id"], sort=False).sum()

    change_points = all_events["user_id"].ne(all_events["user_id"].shift()) | all_events["token"].ne(
        all_events["token"].shift()
    )
    run_id = change_points.cumsum()
    longest_streak = (
        all_events.groupby(["user_id", run_id], sort=False)
        .size()
        .groupby(level="user_id", sort=False)
        .max()
        .rename("seq_longest_streak")
    )

    result = result.join(seq_length)
    result = result.join(_count_selected_ngrams(all_events, ["token", "next_token"], AML_BIGRAMS, "bg"))
    result = result.join(
        _count_selected_ngrams(all_events, ["token", "next_token", "next2_token"], AML_TRIGRAMS, "tg")
    )
    result = result.join(_vectorized_transition_entropy(all_events))
    result = result.join(longest_streak)
    result["seq_inflow_outflow_ratio"] = inflow_counts / outflow_counts.clip(lower=1)
    result["seq_n_unique_types"] = seq_n_unique_types
    result["seq_outflow_fraction"] = outflow_counts / seq_length

    for column in _INT_COLUMNS:
        if column not in result:
            result[column] = 0
        result[column] = pd.to_numeric(result[column], errors="coerce").fillna(0).astype(np.int64)
    for column in _FLOAT_COLUMNS:
        if column not in result:
            result[column] = 0.0
        result[column] = pd.to_numeric(result[column], errors="coerce").fillna(0.0).astype(float)

    result = result.reset_index()
    return result.reindex(columns=_OUTPUT_COLUMNS, fill_value=0)
