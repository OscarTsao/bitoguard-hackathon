from __future__ import annotations

"""GRU-based per-user transaction sequence encoder (Base D features).

Produces a fixed-size embedding from each user's transaction history.  When
PyTorch is available, a 1-layer GRU autoencoder is trained in unsupervised
reconstruction mode and the final hidden states are returned as dense features.
When PyTorch is not available, the GRU embedding columns are filled with zeros
and only the statistical sequence features are returned.

Since the dataset contains only aggregate tabular features (not raw per-event
rows), per-user "sequences" are constructed from rolling time-window bucket
columns (e.g. `twd_dep_count_7d`, `twd_dep_count_30d`, …).  This gives a
coarse 4-step pseudo-sequence that captures temporal dynamics well enough to
feed into the GRU.

Features produced per user
--------------------------
GRU embedding (8 dims, zeros if torch unavailable):
    seq_gru_emb_0 … seq_gru_emb_7

Statistical sequence features (always computed):
    seq_last_amount_log    - log1p of the most-recent transaction amount proxy
    seq_velocity_trend     - linear slope of inter-window transaction counts
    seq_burst_score        - fraction of transactions in the busiest time bucket
    seq_weekend_ratio      - fraction of transactions on weekends (from `weekend_*` columns)
    seq_night_ratio        - fraction of transactions between 22:00-06:00 proxy
    seq_amount_cv          - coefficient of variation of transaction amounts
"""

import logging
import warnings
from math import log1p

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Optional PyTorch import ────────────────────────────────────────────────────
_TORCH_AVAILABLE = False
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    _TORCH_AVAILABLE = True
except ImportError:
    pass

# ── Constants ──────────────────────────────────────────────────────────────────
# GRU input dimensionality: (amount_log, is_deposit_fraction, is_crypto_fraction,
# time_bucket_index_normalised) — 4 features per step.
_GRU_INPUT_DIM = 4
# Column name prefix for GRU embedding outputs.
_EMB_PREFIX = "seq_gru_emb_"
# Statistical feature names (always present).
_STAT_FEATURE_NAMES = [
    "seq_last_amount_log",
    "seq_velocity_trend",
    "seq_burst_score",
    "seq_weekend_ratio",
    "seq_night_ratio",
    "seq_amount_cv",
]


# ── GRU Autoencoder (only defined when torch is available) ──────────────────────

if _TORCH_AVAILABLE:

    class _GRUAutoencoder(nn.Module):
        """Shallow GRU autoencoder for per-user transaction sequences.

        Architecture:
            Encoder: GRU(input_dim → hidden_dim), takes final hidden state h_T.
            Decoder: Linear(hidden_dim → input_dim * max_seq_len), reshapes to
                     (max_seq_len, input_dim) for MSE reconstruction.
        """

        def __init__(self, input_dim: int, hidden_dim: int, max_seq_len: int) -> None:
            super().__init__()
            self.hidden_dim = hidden_dim
            self.max_seq_len = max_seq_len
            self.encoder = nn.GRU(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=1,
                batch_first=True,
            )
            self.decoder = nn.Linear(hidden_dim, input_dim * max_seq_len)

        def forward(self, x: "torch.Tensor") -> tuple["torch.Tensor", "torch.Tensor"]:
            # x: (batch, seq_len, input_dim)
            _, h_n = self.encoder(x)
            # h_n: (1, batch, hidden_dim) → (batch, hidden_dim)
            h = h_n.squeeze(0)
            reconstruction = self.decoder(h).view(-1, self.max_seq_len, x.shape[2])
            return reconstruction, h

        def encode(self, x: "torch.Tensor") -> "torch.Tensor":
            _, h_n = self.encoder(x)
            return h_n.squeeze(0)  # (batch, hidden_dim)


def _build_pseudo_sequences(
    dataset: pd.DataFrame,
    max_seq_len: int,
) -> tuple[np.ndarray, list[int]]:
    """Construct synthetic 4-feature × max_seq_len sequences from tabular columns.

    Uses rolling-window bucket columns to simulate a temporal sequence.  The
    bucketing strategy converts aggregate counts/sums across 1d/7d/30d/all
    windows into a coarse 4-step trajectory for each user.

    Sequence steps (indices 0..3, oldest to newest proxy):
        step 0: all-time aggregates (oldest "background")
        step 1: 30d aggregates (medium-term)
        step 2: 7d aggregates (short-term)
        step 3: 1d/latest aggregates (most recent)

    Feature channels (index → meaning):
        0: amount_log     — log1p(avg amount for that time window)
        1: deposit_frac   — fraction of activity that is deposits
        2: crypto_frac    — fraction of activity that is crypto-related
        3: count_norm     — normalised transaction count for that window
    """
    n_users = len(dataset)
    # Output: (n_users, max_seq_len, _GRU_INPUT_DIM)
    sequences = np.zeros((n_users, max_seq_len, _GRU_INPUT_DIM), dtype=np.float32)
    user_ids: list[int] = []

    # Helper: safely pull a column as float array, default 0.
    def _col(name: str) -> np.ndarray:
        if name in dataset.columns:
            return pd.to_numeric(dataset[name], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
        return np.zeros(n_users, dtype=np.float32)

    # ── Amount features (log-scaled) ──────────────────────────────────────────
    # We use twd + crypto sums/counts for 1d, 7d, 30d, all.
    windows = ["1d", "7d", "30d"]  # ordered from newest → oldest (then "all")

    # Gather count and sum for each window bucket.
    # Column naming convention: twd_dep_count_7d, twd_dep_sum_7d, crypto_dep_count_7d …
    def _window_stats(suffix: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return (total_count, twd_dep_count, crypto_count, total_sum) for a suffix."""
        twd_dep_cnt = _col(f"twd_dep_count_{suffix}") + _col(f"twd_withdraw_count_{suffix}")
        crypto_cnt = _col(f"crypto_dep_count_{suffix}") + _col(f"crypto_withdraw_count_{suffix}")
        total_cnt = twd_dep_cnt + crypto_cnt
        # Prefer twd_dep_sum_Xd; fall back to more generic names.
        twd_sum = _col(f"twd_dep_sum_{suffix}") + _col(f"twd_withdraw_sum_{suffix}")
        crypto_sum = _col(f"crypto_dep_sum_{suffix}") + _col(f"crypto_withdraw_sum_{suffix}")
        total_sum = twd_sum + crypto_sum
        return total_cnt, twd_dep_cnt, crypto_cnt, total_sum

    # All-time stats (step 0 = oldest background).
    all_cnt = (
        _col("twd_total_count") + _col("crypto_total_count")
    )
    # Fall back: some feature schemas use different names.
    if np.all(all_cnt == 0):
        all_cnt = _col("twd_count_all") + _col("crypto_count_all")
    all_sum = _col("twd_total_sum") + _col("crypto_total_sum")
    if np.all(all_sum == 0):
        all_sum = _col("twd_sum_all") + _col("crypto_sum_all")

    twd_all_cnt = _col("twd_total_count")
    crypto_all_cnt = _col("crypto_total_count")
    total_all = np.maximum(all_cnt, 1.0)

    # Build per-step vectors: steps 0=all, 1=30d, 2=7d, 3=1d.
    step_data: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []

    # Step 0: all-time background.
    step_data.append((all_cnt, twd_all_cnt, crypto_all_cnt, all_sum))

    # Steps 1, 2, 3: 30d, 7d, 1d.
    for window in ["30d", "7d", "1d"]:
        total_cnt_w, twd_cnt_w, crypto_cnt_w, total_sum_w = _window_stats(window)
        step_data.append((total_cnt_w, twd_cnt_w, crypto_cnt_w, total_sum_w))

    # Global max count for normalisation (per user across all steps).
    max_cnts = np.stack([sd[0] for sd in step_data], axis=1).max(axis=1).clip(min=1.0)  # (n_users,)
    # Global max sum for amount normalisation.
    max_sums = np.stack([sd[3] for sd in step_data], axis=1).max(axis=1).clip(min=1.0)  # (n_users,)

    # Fill in sequence steps 0..3 (pad with zeros for steps beyond 3 if max_seq_len > 4).
    n_steps = min(len(step_data), max_seq_len)
    for step_idx in range(n_steps):
        total_cnt_s, twd_cnt_s, crypto_cnt_s, total_sum_s = step_data[step_idx]
        safe_cnt = np.maximum(total_cnt_s, 1.0)

        # Channel 0: amount_log — log1p of average transaction amount.
        avg_amount = total_sum_s / safe_cnt
        sequences[:, step_idx, 0] = np.log1p(avg_amount.clip(min=0.0))

        # Channel 1: deposit_frac — twd deposit fraction of total.
        sequences[:, step_idx, 1] = (twd_cnt_s / safe_cnt).clip(0.0, 1.0)

        # Channel 2: crypto_frac — crypto fraction of total.
        sequences[:, step_idx, 2] = (crypto_cnt_s / safe_cnt).clip(0.0, 1.0)

        # Channel 3: count_norm — normalised total count (relative to user's own max).
        sequences[:, step_idx, 3] = (total_cnt_s / max_cnts).clip(0.0, 1.0)

    user_ids = pd.to_numeric(dataset["user_id"], errors="coerce").fillna(-1).astype(int).tolist()
    return sequences, user_ids


def _train_gru_autoencoder(
    sequences: np.ndarray,
    embed_dim: int,
    n_epochs: int,
    device: str,
) -> np.ndarray:
    """Train GRU autoencoder and return encoder hidden states.

    Parameters
    ----------
    sequences:
        (n_users, max_seq_len, _GRU_INPUT_DIM) float32 array.
    embed_dim:
        Hidden state dimension of the GRU.
    n_epochs:
        Number of training epochs.
    device:
        'cpu' or 'cuda'.

    Returns
    -------
    np.ndarray of shape (n_users, embed_dim)
        Encoder hidden states (GRU embeddings).
    """
    n_users, max_seq_len, input_dim = sequences.shape
    torch_device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")

    x_tensor = torch.from_numpy(sequences).to(torch_device)
    dataset_torch = TensorDataset(x_tensor)
    batch_size = min(256, n_users)
    loader = DataLoader(dataset_torch, batch_size=batch_size, shuffle=True)

    model = _GRUAutoencoder(input_dim=input_dim, hidden_dim=embed_dim, max_seq_len=max_seq_len)
    model = model.to(torch_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(n_epochs):
        total_loss = 0.0
        for (batch_x,) in loader:
            optimizer.zero_grad()
            reconstruction, _ = model(batch_x)
            loss = criterion(reconstruction, batch_x)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
        avg_loss = total_loss / max(len(loader), 1)
        if (epoch + 1) % max(1, n_epochs // 5) == 0:
            logger.debug("GRU autoencoder epoch %d/%d — loss=%.6f", epoch + 1, n_epochs, avg_loss)

    # Extract embeddings (no gradient needed).
    model.eval()
    all_embeddings: list[np.ndarray] = []
    inference_loader = DataLoader(TensorDataset(x_tensor), batch_size=512, shuffle=False)
    with torch.no_grad():
        for (batch_x,) in inference_loader:
            h = model.encode(batch_x)  # (batch, embed_dim)
            all_embeddings.append(h.cpu().numpy())

    embeddings = np.concatenate(all_embeddings, axis=0)  # (n_users, embed_dim)
    return embeddings


def _compute_statistical_features(
    dataset: pd.DataFrame,
    sequences: np.ndarray,
) -> dict[str, np.ndarray]:
    """Compute statistical sequence features from aggregate columns.

    These features do not require PyTorch and capture temporal dynamics using
    the 4-step pseudo-sequence constructed from rolling bucket columns.

    Parameters
    ----------
    dataset:
        Original input DataFrame (tabular features).
    sequences:
        (n_users, max_seq_len, _GRU_INPUT_DIM) array from _build_pseudo_sequences.

    Returns
    -------
    dict mapping feature name → np.ndarray of shape (n_users,).
    """
    n_users = len(dataset)

    def _col(name: str) -> np.ndarray:
        if name in dataset.columns:
            return pd.to_numeric(dataset[name], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
        return np.zeros(n_users, dtype=np.float64)

    feats: dict[str, np.ndarray] = {}

    # ── seq_last_amount_log ──────────────────────────────────────────────────
    # Channel 0 of the last (most recent) sequence step contains log1p(avg_amount).
    # We use step 3 (index 3 = 1d bucket) as the "last" proxy.
    last_step_idx = min(3, sequences.shape[1] - 1)
    feats["seq_last_amount_log"] = sequences[:, last_step_idx, 0].astype(np.float64)

    # ── seq_velocity_trend ───────────────────────────────────────────────────
    # Linear regression slope across count_norm (channel 3) over the 4 steps.
    # A positive slope means activity is accelerating (increasing counts recently).
    count_seq = sequences[:, :, 3].astype(np.float64)  # (n_users, seq_len)
    n_steps = count_seq.shape[1]
    if n_steps >= 2:
        x_idx = np.arange(n_steps, dtype=np.float64)
        x_mean = x_idx.mean()
        x_centered = x_idx - x_mean
        denom = float((x_centered ** 2).sum()) or 1.0
        # slope = (X - X_mean)·Y / ||X-X_mean||^2 broadcast over users.
        numerator = (count_seq * x_centered[np.newaxis, :]).sum(axis=1)
        feats["seq_velocity_trend"] = numerator / denom
    else:
        feats["seq_velocity_trend"] = np.zeros(n_users, dtype=np.float64)

    # ── seq_burst_score ──────────────────────────────────────────────────────
    # Fraction of total count in the highest-count bucket (= max step / sum steps).
    # Captures concentration of activity in a single time window.
    total_count_seq = count_seq.sum(axis=1).clip(min=1.0)
    max_step_count = count_seq.max(axis=1)
    feats["seq_burst_score"] = (max_step_count / total_count_seq).clip(0.0, 1.0)

    # ── seq_weekend_ratio ────────────────────────────────────────────────────
    # Proxy from any `*_weekend_*` or `*_weekday_*` columns.
    weekend_cnt = _col("twd_weekend_count") + _col("crypto_weekend_count")
    weekday_cnt = _col("twd_weekday_count") + _col("crypto_weekday_count")
    total_day_cnt = weekend_cnt + weekday_cnt
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        feats["seq_weekend_ratio"] = np.where(
            total_day_cnt > 0, weekend_cnt / total_day_cnt, 0.0
        ).clip(0.0, 1.0)

    # ── seq_night_ratio ──────────────────────────────────────────────────────
    # Proxy: look for `*_night_*` columns; fall back to `*_hour_22*` / `*_hour_0*` etc.
    night_cnt = (
        _col("twd_night_count") + _col("crypto_night_count")
        + _col("twd_dep_night_count") + _col("crypto_dep_night_count")
    )
    day_cnt = (
        _col("twd_day_count") + _col("crypto_day_count")
        + _col("twd_dep_count_30d") + _col("crypto_dep_count_30d")
    ).clip(min=0.0)
    # If we have night counts use them; otherwise set to 0.
    total_for_night = (night_cnt + day_cnt).clip(min=1.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        feats["seq_night_ratio"] = np.where(
            night_cnt > 0, night_cnt / total_for_night, 0.0
        ).clip(0.0, 1.0)

    # ── seq_amount_cv ────────────────────────────────────────────────────────
    # Coefficient of variation of the amount_log sequence (channel 0) across steps.
    # High CV = very uneven transaction amounts across time windows.
    amount_seq = sequences[:, :, 0].astype(np.float64)  # (n_users, seq_len)
    amount_mean = amount_seq.mean(axis=1)
    amount_std = amount_seq.std(axis=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        feats["seq_amount_cv"] = np.where(
            amount_mean > 1e-6, amount_std / amount_mean, 0.0
        ).clip(0.0, 10.0)

    return feats


def build_sequence_features(
    dataset: pd.DataFrame,
    max_seq_len: int = 30,
    embed_dim: int = 8,
    n_epochs: int = 10,
    device: str = "cpu",
) -> pd.DataFrame:
    """Build GRU sequence embedding + statistical features for all users.

    Encodes each user's transaction history as a fixed-size embedding using a
    GRU autoencoder trained on pseudo-sequences derived from rolling-window
    aggregate columns.  The final hidden state of the GRU encoder is returned
    as 8 dense embedding features (`seq_gru_emb_0` … `seq_gru_emb_7`).

    If PyTorch is not available, GRU embedding columns are filled with zeros.
    Statistical features (seq_velocity_trend, seq_burst_score, etc.) are always
    computed from the tabular data.

    The entire function is wrapped in a try/except: on any failure an empty
    DataFrame is returned (caller should handle this gracefully).

    Parameters
    ----------
    dataset:
        Full dataset with a `user_id` column and any available rolling-window
        aggregate columns (e.g. `twd_dep_count_7d`, `crypto_total_sum`, …).
        The function is tolerant of missing columns (fills with zeros).
    max_seq_len:
        Number of sequence steps to use.  Steps beyond 4 are zero-padded since
        only 4 pseudo-steps are constructed from the bucket columns.
    embed_dim:
        GRU hidden state dimension (= number of `seq_gru_emb_*` features).
    n_epochs:
        Training epochs for the GRU autoencoder.
    device:
        Compute device for PyTorch: 'cpu' or 'cuda'.

    Returns
    -------
    pd.DataFrame
        One row per user in `dataset`, columns:
        user_id,
        seq_gru_emb_0 … seq_gru_emb_{embed_dim-1},
        seq_last_amount_log, seq_velocity_trend, seq_burst_score,
        seq_weekend_ratio, seq_night_ratio, seq_amount_cv.
        Returns an empty DataFrame with correct column names on failure.
    """
    # Define expected output columns for the empty-fallback case.
    emb_cols = [f"{_EMB_PREFIX}{i}" for i in range(embed_dim)]
    all_output_cols = ["user_id", *emb_cols, *_STAT_FEATURE_NAMES]

    if dataset.empty or "user_id" not in dataset.columns:
        return pd.DataFrame(columns=all_output_cols)

    try:
        n_users = len(dataset)
        logger.info(
            "build_sequence_features: n_users=%d, max_seq_len=%d, embed_dim=%d, torch=%s",
            n_users, max_seq_len, embed_dim, _TORCH_AVAILABLE,
        )

        # ── Step 1: Build pseudo-sequences ───────────────────────────────────
        sequences, user_ids = _build_pseudo_sequences(dataset, max_seq_len=max_seq_len)
        # sequences: (n_users, max_seq_len, _GRU_INPUT_DIM)

        # ── Step 2: Statistical features ─────────────────────────────────────
        stat_feats = _compute_statistical_features(dataset, sequences)

        # ── Step 3: GRU embeddings ───────────────────────────────────────────
        if _TORCH_AVAILABLE:
            try:
                embeddings = _train_gru_autoencoder(
                    sequences=sequences,
                    embed_dim=embed_dim,
                    n_epochs=n_epochs,
                    device=device,
                )
                # embeddings: (n_users, embed_dim)
            except Exception as torch_exc:
                logger.warning(
                    "GRU autoencoder training failed (%s); using zero embeddings.", torch_exc
                )
                embeddings = np.zeros((n_users, embed_dim), dtype=np.float32)
        else:
            logger.debug("torch not available; GRU embeddings will be zero.")
            embeddings = np.zeros((n_users, embed_dim), dtype=np.float32)

        # ── Step 4: Assemble output DataFrame ────────────────────────────────
        output: dict[str, np.ndarray | list] = {"user_id": user_ids}

        for i in range(embed_dim):
            output[f"{_EMB_PREFIX}{i}"] = embeddings[:, i].astype(np.float32)

        for feat_name in _STAT_FEATURE_NAMES:
            output[feat_name] = stat_feats.get(feat_name, np.zeros(n_users, dtype=np.float64))

        result = pd.DataFrame(output)
        result["user_id"] = pd.to_numeric(result["user_id"], errors="coerce").astype("Int64")

        logger.info(
            "build_sequence_features: produced %d rows, %d columns (torch=%s)",
            len(result), len(result.columns), _TORCH_AVAILABLE,
        )
        return result

    except Exception as exc:
        logger.error(
            "build_sequence_features: unrecoverable error (%s); returning empty DataFrame.",
            exc,
            exc_info=True,
        )
        return pd.DataFrame(columns=all_output_cols)
