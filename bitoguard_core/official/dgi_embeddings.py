from __future__ import annotations

"""Deep Graph Infomax (DGI) self-supervised graph embeddings.

DGI maximises mutual information between local node representations and a global
graph summary without requiring labels.  In the PU-learning setting of this
pipeline (34.8 % of positives are unlabelled) DGI can surface structural fraud
patterns that supervised GNNs miss when the labelled training set is small.

Public API
----------
build_dgi_embeddings(graph, embed_dim, n_epochs, lr, device) -> DataFrame
    Returns user_id + dgi_emb_0...(embed_dim-1)

build_dgi_features(graph, label_frame, embed_dim, n_epochs) -> DataFrame
    Wraps build_dgi_embeddings and appends a Random Forest label-predictive score
    (rf_dgi_score) trained on the labelled subset.
"""

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from official.graph_dataset import TransductiveGraph

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_node_feature_matrix(
    graph: "TransductiveGraph",
    label_frame: pd.DataFrame | None,
) -> np.ndarray:
    """Build a float32 node feature matrix in graph.user_index order.

    If graph.user_feature_frame has numeric columns beyond user_id those are
    used directly.  Otherwise a 4-dimensional fallback is computed:
        [degree_normalised, is_labeled, is_positive_seed, ppr_approximate]
    """
    user_ids = graph.user_ids  # sorted list of int
    n = len(user_ids)

    # ------------------------------------------------------------------
    # Option A: use tabular features already attached to the graph
    # ------------------------------------------------------------------
    feature_cols = [
        col
        for col in graph.user_feature_frame.columns
        if col != "user_id"
    ]
    if feature_cols:
        # Align to user_index order
        frame = graph.user_feature_frame.set_index("user_id").reindex(user_ids)
        mat = frame[feature_cols].fillna(0.0).to_numpy(dtype=np.float32)
        # Replace any remaining NaN/Inf with 0
        mat = np.nan_to_num(mat, nan=0.0, posinf=0.0, neginf=0.0)
        return mat

    # ------------------------------------------------------------------
    # Option B: 4-dimensional fallback features
    # ------------------------------------------------------------------
    uid_to_idx = graph.user_index  # {user_id: row_index}

    # degree_normalised
    degree = np.zeros(n, dtype=np.float32)
    if not graph.collapsed_edges.empty:
        for uid, neighbors in graph.combined_neighbors.items():
            idx = uid_to_idx.get(uid)
            if idx is not None:
                degree[idx] = float(len(neighbors))
    max_degree = degree.max()
    if max_degree > 0.0:
        degree = degree / max_degree

    # is_labeled / is_positive_seed
    is_labeled = np.zeros(n, dtype=np.float32)
    is_positive_seed = np.zeros(n, dtype=np.float32)
    if label_frame is not None and not label_frame.empty:
        lf = label_frame.copy()
        lf["user_id"] = pd.to_numeric(lf["user_id"], errors="coerce").astype("Int64")
        lf = lf.dropna(subset=["user_id"])
        labeled_ids = set(lf["user_id"].astype(int).tolist())
        pos_ids: set[int] = set()
        if "status" in lf.columns:
            lf["status"] = pd.to_numeric(lf["status"], errors="coerce")
            pos_ids = set(lf[lf["status"] == 1]["user_id"].astype(int).tolist())
        for uid in labeled_ids:
            idx = uid_to_idx.get(uid)
            if idx is not None:
                is_labeled[idx] = 1.0
        for uid in pos_ids:
            idx = uid_to_idx.get(uid)
            if idx is not None:
                is_positive_seed[idx] = 1.0

    # ppr_approximate: 1-hop neighbours of positives = 0.5, 2-hop = 0.25, else 0
    ppr_approx = np.zeros(n, dtype=np.float32)
    positive_indices = set(np.where(is_positive_seed > 0.0)[0].tolist())
    # Build index-level neighbour map for fast lookup
    idx_neighbors: dict[int, set[int]] = {}
    for uid, neighbors in graph.combined_neighbors.items():
        row_idx = uid_to_idx.get(uid)
        if row_idx is None:
            continue
        idx_neighbors[row_idx] = {
            uid_to_idx[nb] for nb, _ in neighbors if nb in uid_to_idx
        }
    one_hop: set[int] = set()
    two_hop: set[int] = set()
    for pidx in positive_indices:
        for nb_idx in idx_neighbors.get(pidx, set()):
            if nb_idx not in positive_indices:
                one_hop.add(nb_idx)
    for hop1_idx in one_hop:
        for nb_idx in idx_neighbors.get(hop1_idx, set()):
            if nb_idx not in positive_indices and nb_idx not in one_hop:
                two_hop.add(nb_idx)
    for idx in one_hop:
        ppr_approx[idx] = 0.5
    for idx in two_hop:
        ppr_approx[idx] = 0.25

    mat = np.column_stack([degree, is_labeled, is_positive_seed, ppr_approx])
    return mat.astype(np.float32)


def _build_sparse_adjacency(graph: "TransductiveGraph") -> "scipy.sparse.csr_matrix":
    """Return a row-normalised sparse adjacency matrix in user_index order."""
    from scipy import sparse

    n = len(graph.user_ids)
    if graph.collapsed_edges.empty:
        return sparse.csr_matrix((n, n), dtype=np.float32)

    src = graph.collapsed_edges["src_user_id"].astype(int).map(graph.user_index).to_numpy()
    dst = graph.collapsed_edges["dst_user_id"].astype(int).map(graph.user_index).to_numpy()
    weight = graph.collapsed_edges["weight"].astype(float).to_numpy(dtype=np.float32)

    # Symmetric normalisation D^{-1/2} A D^{-1/2}
    adj = sparse.csr_matrix((weight, (src, dst)), shape=(n, n), dtype=np.float32)
    degree = np.asarray(adj.sum(axis=1)).ravel()
    degree[degree == 0.0] = 1.0
    d_inv_sqrt = sparse.diags(1.0 / np.sqrt(degree))
    return d_inv_sqrt @ adj @ d_inv_sqrt


# ---------------------------------------------------------------------------
# Spectral fallback (scipy, no PyTorch required)
# ---------------------------------------------------------------------------


def _spectral_embeddings(
    graph: "TransductiveGraph",
    label_frame: pd.DataFrame | None,
    embed_dim: int,
) -> np.ndarray:
    """Compute the top embed_dim eigenvectors of the normalised Laplacian.

    These approximate DGI embeddings without a neural network.
    Isolated (zero-degree) nodes receive zero embeddings.
    """
    from scipy import sparse
    from scipy.sparse.linalg import eigsh

    n = len(graph.user_ids)
    if n == 0:
        return np.zeros((0, embed_dim), dtype=np.float32)

    adj = _build_sparse_adjacency(graph)

    # Symmetrised Laplacian: L = I - D^{-1/2} A D^{-1/2}
    laplacian = sparse.eye(n, format="csr") - adj

    # How many eigenvectors can we safely request?
    k = min(embed_dim, n - 1)
    if k <= 0:
        return np.zeros((n, embed_dim), dtype=np.float32)

    try:
        # which="SM" → smallest magnitude eigenvalues → smoothest eigenvectors
        _, eigvecs = eigsh(laplacian, k=k, which="SM", tol=1e-4, maxiter=1000)
        eigvecs = eigvecs.astype(np.float32)
    except Exception as exc:
        logger.warning("eigsh failed (%s); using zero spectral embeddings", exc)
        eigvecs = np.zeros((n, k), dtype=np.float32)

    # Pad to embed_dim columns if needed
    if k < embed_dim:
        pad = np.zeros((n, embed_dim - k), dtype=np.float32)
        eigvecs = np.concatenate([eigvecs, pad], axis=1)

    # L2-normalise each row so magnitudes are comparable across experiments
    row_norms = np.linalg.norm(eigvecs, axis=1, keepdims=True)
    row_norms[row_norms == 0.0] = 1.0
    eigvecs = eigvecs / row_norms

    return eigvecs


# ---------------------------------------------------------------------------
# PyTorch DGI implementation
# ---------------------------------------------------------------------------


def _torch_dgi_embeddings(
    graph: "TransductiveGraph",
    label_frame: pd.DataFrame | None,
    embed_dim: int,
    n_epochs: int,
    lr: float,
    device_str: str,
) -> np.ndarray:
    """Train a 2-layer GCN encoder with DGI objective and return embeddings.

    Architecture
    ------------
    Encoder:  GCNConv(input_dim → 256) → PReLU → GCNConv(256 → embed_dim)
    Discriminator: bilinear(h_i, s) → scalar  where s = sigmoid(mean(H))

    DGI loss: maximise MI between node embeddings H and global summary s,
    while minimising MI between corrupted embeddings H~ and s.
    Loss = -E[log D(h, s)] - E[log(1 - D(h~, s))]  (binary cross-entropy)
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from scipy import sparse

    n = len(graph.user_ids)
    if n == 0:
        return np.zeros((0, embed_dim), dtype=np.float32)

    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device(device_str if torch.cuda.is_available() or device_str == "cpu" else "cpu")

    # ── Node features ──────────────────────────────────────────────────────
    X_np = _build_node_feature_matrix(graph, label_frame)  # (n, d_in)
    d_in = X_np.shape[1]
    X = torch.tensor(X_np, dtype=torch.float32, device=device)

    # ── Adjacency (sparse COO for efficient GCN propagation) ──────────────
    adj_sp = _build_sparse_adjacency(graph)  # scipy csr
    adj_coo = adj_sp.tocoo()
    indices = torch.tensor(
        np.vstack([adj_coo.row, adj_coo.col]), dtype=torch.long, device=device
    )
    values = torch.tensor(adj_coo.data, dtype=torch.float32, device=device)
    adj_t = torch.sparse_coo_tensor(indices, values, size=(n, n)).coalesce()

    # ── GCN propagation helper ─────────────────────────────────────────────
    def gcn_conv(h: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        # Standard GCN: A_norm @ H @ W + b
        propagated = torch.sparse.mm(adj_t, h)
        return F.linear(propagated, weight, bias)

    # ── Model parameters ──────────────────────────────────────────────────
    hidden_dim = 256
    W1 = nn.Parameter(torch.empty(hidden_dim, d_in, device=device))
    b1 = nn.Parameter(torch.zeros(hidden_dim, device=device))
    W2 = nn.Parameter(torch.empty(embed_dim, hidden_dim, device=device))
    b2 = nn.Parameter(torch.zeros(embed_dim, device=device))
    # Bilinear discriminator weight: (embed_dim, embed_dim)
    W_disc = nn.Parameter(torch.empty(embed_dim, embed_dim, device=device))
    b_disc = nn.Parameter(torch.zeros(1, device=device))

    nn.init.xavier_uniform_(W1)
    nn.init.xavier_uniform_(W2)
    nn.init.xavier_uniform_(W_disc)

    # PReLU learnable negative slope
    prelu_w = nn.Parameter(torch.tensor(0.25, device=device))

    params = [W1, b1, W2, b2, W_disc, b_disc, prelu_w]
    optimiser = torch.optim.Adam(params, lr=lr)

    def encode(x_in: torch.Tensor) -> torch.Tensor:
        h = gcn_conv(x_in, W1, b1)
        h = F.prelu(h, prelu_w)
        h = gcn_conv(h, W2, b2)
        return h  # (n, embed_dim)

    def discriminate(h: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """Bilinear discriminator score: h @ W_disc @ s^T + b"""
        # s: (embed_dim,), h: (n, embed_dim)
        Ws = s @ W_disc.t()  # (embed_dim,)
        return (h * Ws.unsqueeze(0)).sum(dim=1) + b_disc.squeeze()  # (n,)

    # ── Training loop ─────────────────────────────────────────────────────
    for epoch in range(n_epochs):
        optimiser.zero_grad()

        # Positive branch: real node features
        H_real = encode(X)  # (n, embed_dim)
        # Global summary: sigmoid of mean embedding
        s = torch.sigmoid(H_real.mean(dim=0))  # (embed_dim,)

        # Negative branch: corrupt by shuffling rows of X
        perm = torch.randperm(n, device=device)
        X_corrupt = X[perm]
        H_corrupt = encode(X_corrupt)  # (n, embed_dim)

        # Discriminator scores
        logits_real = discriminate(H_real, s)       # (n,)
        logits_corrupt = discriminate(H_corrupt, s)  # (n,)

        # Binary cross-entropy loss: real → 1, corrupt → 0
        loss_real = F.binary_cross_entropy_with_logits(
            logits_real, torch.ones(n, device=device)
        )
        loss_corrupt = F.binary_cross_entropy_with_logits(
            logits_corrupt, torch.zeros(n, device=device)
        )
        loss = loss_real + loss_corrupt
        loss.backward()
        optimiser.step()

        if (epoch + 1) % 10 == 0:
            logger.debug("DGI epoch %d/%d  loss=%.4f", epoch + 1, n_epochs, loss.item())

    # ── Extract embeddings ────────────────────────────────────────────────
    with torch.no_grad():
        H_final = encode(X).cpu().numpy()  # (n, embed_dim)

    return H_final.astype(np.float32)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_dgi_embeddings(
    graph: "TransductiveGraph",
    embed_dim: int = 16,
    n_epochs: int = 50,
    lr: float = 0.001,
    device: str = "cpu",
    *,
    label_frame: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Return DGI (or spectral fallback) embeddings for all users.

    Parameters
    ----------
    graph:      TransductiveGraph from official.graph_dataset.
    embed_dim:  Number of embedding dimensions.
    n_epochs:   Training epochs (ignored for spectral fallback).
    lr:         Adam learning rate (ignored for spectral fallback).
    device:     Torch device string; "cpu" or "cuda".
    label_frame: Optional DataFrame with user_id + status columns used to
                 compute fallback node features when graph has no tabular
                 features attached.

    Returns
    -------
    DataFrame with columns: user_id, dgi_emb_0, ..., dgi_emb_{embed_dim-1}
    All users in graph.user_ids receive a row; isolated nodes get zeros.
    """
    user_ids = graph.user_ids
    n = len(user_ids)
    emb_cols = [f"dgi_emb_{i}" for i in range(embed_dim)]

    if n == 0:
        return pd.DataFrame(columns=["user_id", *emb_cols])

    try:
        import torch  # noqa: F401 — check availability only
        logger.info("PyTorch available — running DGI neural encoder (%d epochs)", n_epochs)
        embeddings = _torch_dgi_embeddings(
            graph, label_frame, embed_dim, n_epochs, lr, device
        )
    except ImportError:
        logger.info("PyTorch not available — falling back to spectral embeddings")
        embeddings = _spectral_embeddings(graph, label_frame, embed_dim)
    except Exception as exc:
        logger.warning("DGI training failed (%s); falling back to spectral embeddings", exc)
        try:
            embeddings = _spectral_embeddings(graph, label_frame, embed_dim)
        except Exception as exc2:
            logger.warning("Spectral fallback also failed (%s); returning zeros", exc2)
            embeddings = np.zeros((n, embed_dim), dtype=np.float32)

    # Sanity-check shape
    if embeddings.shape != (n, embed_dim):
        logger.warning(
            "Embedding shape mismatch: expected (%d, %d), got %s — zeroing out",
            n, embed_dim, embeddings.shape,
        )
        embeddings = np.zeros((n, embed_dim), dtype=np.float32)

    # Replace any remaining non-finite values
    embeddings = np.nan_to_num(embeddings, nan=0.0, posinf=0.0, neginf=0.0)

    out = pd.DataFrame(embeddings, columns=emb_cols)
    out.insert(0, "user_id", pd.array(user_ids, dtype="Int64"))
    return out


def build_dgi_features(
    graph: "TransductiveGraph",
    label_frame: pd.DataFrame,
    embed_dim: int = 16,
    n_epochs: int = 50,
) -> pd.DataFrame:
    """Return DGI embeddings + an RF-based label-predictive scalar score.

    The Random Forest is trained on labelled users (those present in
    label_frame with a valid status column) and then predicts a fraud
    probability for every user using the DGI embeddings as features.  The
    result, ``rf_dgi_score``, summarises the label-predictive content of the
    DGI representation as a single feature that can be plugged into the
    stacker alongside tabular features.

    Parameters
    ----------
    graph:       TransductiveGraph.
    label_frame: DataFrame with user_id (Int64) and status (0/1) columns.
    embed_dim:   DGI embedding dimension.
    n_epochs:    DGI training epochs.

    Returns
    -------
    DataFrame: user_id, dgi_emb_0...(embed_dim-1), rf_dgi_score
    """
    try:
        emb_df = build_dgi_embeddings(
            graph,
            embed_dim=embed_dim,
            n_epochs=n_epochs,
            label_frame=label_frame,
        )

        emb_cols = [f"dgi_emb_{i}" for i in range(embed_dim)]

        # Default: rf_dgi_score = 0.0
        emb_df["rf_dgi_score"] = np.float32(0.0)

        # ── Prepare labelled subset ───────────────────────────────────────
        if label_frame is None or label_frame.empty:
            logger.info("build_dgi_features: no label_frame; skipping RF score")
            return emb_df

        lf = label_frame.copy()
        lf["user_id"] = pd.to_numeric(lf["user_id"], errors="coerce").astype("Int64")
        if "status" not in lf.columns:
            logger.info("build_dgi_features: label_frame has no 'status' column; skipping RF score")
            return emb_df

        lf["status"] = pd.to_numeric(lf["status"], errors="coerce")
        lf = lf.dropna(subset=["user_id", "status"])

        # Merge embeddings into label_frame to get training set
        lf["user_id"] = lf["user_id"].astype(int)
        emb_for_merge = emb_df.copy()
        emb_for_merge["user_id"] = emb_for_merge["user_id"].astype(int)
        train_df = lf.merge(emb_for_merge[["user_id", *emb_cols]], on="user_id", how="inner")

        if len(train_df) < 10:
            logger.info(
                "build_dgi_features: only %d labelled users with embeddings; skipping RF score",
                len(train_df),
            )
            return emb_df

        X_train = train_df[emb_cols].fillna(0.0).to_numpy(dtype=np.float32)
        y_train = train_df["status"].astype(int).to_numpy()

        n_pos = int(y_train.sum())
        n_neg = len(y_train) - n_pos
        if n_pos < 2 or n_neg < 2:
            logger.info(
                "build_dgi_features: insufficient class balance (%d pos, %d neg); skipping RF score",
                n_pos, n_neg,
            )
            return emb_df

        # ── Train Random Forest ───────────────────────────────────────────
        try:
            from sklearn.ensemble import RandomForestClassifier

            rf = RandomForestClassifier(
                n_estimators=100,
                max_depth=6,
                min_samples_leaf=5,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            )
            rf.fit(X_train, y_train)

            # Score all users
            X_all = emb_df[emb_cols].fillna(0.0).to_numpy(dtype=np.float32)
            rf_scores = rf.predict_proba(X_all)[:, 1].astype(np.float32)
            emb_df["rf_dgi_score"] = rf_scores

            logger.info(
                "build_dgi_features: RF trained on %d labelled users; "
                "rf_dgi_score mean=%.4f",
                len(train_df),
                float(rf_scores.mean()),
            )
        except ImportError:
            logger.warning("scikit-learn not available; rf_dgi_score left as 0.0")
        except Exception as exc:
            logger.warning("RF score computation failed (%s); rf_dgi_score left as 0.0", exc)

        return emb_df

    except Exception as exc:
        logger.error("build_dgi_features failed (%s); returning empty DataFrame", exc)
        emb_cols = [f"dgi_emb_{i}" for i in range(embed_dim)]
        empty = pd.DataFrame(columns=["user_id", *emb_cols, "rf_dgi_score"])
        empty["user_id"] = pd.array([], dtype="Int64")
        return empty
