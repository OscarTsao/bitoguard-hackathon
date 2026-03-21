from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from official.common import RANDOM_SEED
from official.graph_dataset import TransductiveGraph


def _require_torch() -> tuple[Any, Any, Any]:
    try:
        import torch
        import torch.nn.functional as F
        from torch import nn
    except Exception as exc:  # pragma: no cover - dependency guard
        raise ImportError(
            "Graph model dependencies are missing. Install torch to enable transductive graph training."
        ) from exc
    return torch, F, nn


@dataclass
class GraphModelFitResult:
    model_state: dict[str, Any]
    model_meta: dict[str, Any]
    validation_probabilities: np.ndarray | None
    full_probabilities: np.ndarray


def _seed_everything(torch: Any) -> None:
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)


def _device(torch: Any) -> Any:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _user_feature_columns(graph: TransductiveGraph) -> list[str]:
    return [column for column in graph.user_feature_frame.columns if column != "user_id"]


def _feature_tensor(graph: TransductiveGraph, torch: Any, feature_columns: list[str]) -> Any:
    frame = graph.user_feature_frame[["user_id", *feature_columns]].copy()
    return torch.tensor(frame[feature_columns].fillna(0.0).to_numpy(dtype=np.float32), dtype=torch.float32)


def _normalized_adjacency_tensor(graph: TransductiveGraph, torch: Any) -> Any:
    user_count = len(graph.user_ids)
    if graph.collapsed_edges.empty:
        indices = torch.zeros((2, 0), dtype=torch.long)
        values = torch.zeros((0,), dtype=torch.float32)
        return torch.sparse_coo_tensor(indices, values, size=(user_count, user_count)).coalesce()
    src = graph.collapsed_edges["src_user_id"].astype(int).map(graph.user_index).to_numpy(dtype=np.int64)
    dst = graph.collapsed_edges["dst_user_id"].astype(int).map(graph.user_index).to_numpy(dtype=np.int64)
    weight = graph.collapsed_edges["weight"].astype(float).to_numpy(dtype=np.float32)
    # v36: Symmetric normalization D^{-1/2} A D^{-1/2} (standard GCN, Kipf & Welling 2017).
    # Previous source-only D^{-1} A caused over-smoothing: low-degree nodes connected to hub
    # (degree=849) received hub embeddings directly (A[node,hub]=1/1*hub_feat), making all
    # hub-connected nodes look identical. With symmetric norm, hub influence scales as
    # 1/sqrt(849*degree_dst) — drastically reducing hub dominance.
    degree_src = np.zeros(user_count, dtype=np.float32)
    degree_dst = np.zeros(user_count, dtype=np.float32)
    np.add.at(degree_src, src, weight)
    np.add.at(degree_dst, dst, weight)
    degree_src[degree_src == 0.0] = 1.0
    degree_dst[degree_dst == 0.0] = 1.0
    norm_weight = weight / np.sqrt(degree_src[src] * degree_dst[dst])
    indices = torch.tensor(np.vstack([src, dst]), dtype=torch.long)
    values = torch.tensor(norm_weight, dtype=torch.float32)
    return torch.sparse_coo_tensor(indices, values, size=(user_count, user_count)).coalesce()


def train_graphsage_model(
    graph: TransductiveGraph,
    label_frame: pd.DataFrame,
    train_user_ids: set[int],
    valid_user_ids: set[int] | None = None,
    max_epochs: int = 40,
    hidden_dim: int = 96,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 12,
) -> GraphModelFitResult:
    # Skip GNN entirely when max_epochs=0 (graphsage_3layer disabled) — avoids torch import.
    if max_epochs <= 0:
        n_users = len(graph.user_ids)
        return GraphModelFitResult(
            model_state={"metadata": {"user_ids": graph.user_ids, "max_epochs": 0, "hidden_dim": hidden_dim, "user_feature_columns": [], "best_epoch": 0}},
            model_meta={"user_ids": graph.user_ids, "max_epochs": 0, "hidden_dim": hidden_dim, "user_feature_columns": [], "best_epoch": 0},
            validation_probabilities=None,
            full_probabilities=np.zeros(n_users, dtype=np.float32),
        )
    torch, F, nn = _require_torch()
    _seed_everything(torch)
    feature_columns = _user_feature_columns(graph)
    x = _feature_tensor(graph, torch, feature_columns)
    adjacency = _normalized_adjacency_tensor(graph, torch)

    class UserGraphSAGE(nn.Module):
        def __init__(self, input_dim: int, hidden_dim: int, prior_logit: float) -> None:
            super().__init__()
            # v43: Input feature BatchNorm — normalizes ~150 raw features (different scales:
            # account_age_days=0-3650, crypto_txn_count=0-1000+, etc.) to prevent large-scale
            # features dominating gradient flow through the neighborhood aggregation step.
            # BatchNorm over all N nodes gives stable per-feature statistics across the graph.
            self.input_bn = nn.BatchNorm1d(input_dim)
            self.self_linear_1 = nn.Linear(input_dim, hidden_dim)
            self.neighbor_linear_1 = nn.Linear(input_dim, hidden_dim)
            self.self_linear_2 = nn.Linear(hidden_dim, hidden_dim)
            self.neighbor_linear_2 = nn.Linear(hidden_dim, hidden_dim)
            # v47: 3rd aggregation layer — captures 3-hop neighborhoods (fraud chains where
            # A→B→C→D all participate in the same ring). With temporal co-occurrence edges
            # added to the graph, a 3rd layer allows signal to propagate through:
            # direct_relation → wallet → temporal_cluster paths that are invisible to 2 layers.
            # hidden_dim=96 (was 64): more capacity needed for 3-layer aggregation without
            # information bottleneck; 50% increase from 64→96 adds +6k parameters per layer.
            self.self_linear_3 = nn.Linear(hidden_dim, hidden_dim)
            self.neighbor_linear_3 = nn.Linear(hidden_dim, hidden_dim)
            # v36: LayerNorm after each aggregation layer — stabilizes training on heterogeneous
            # user graphs where node degree varies 1–849. Without normalization, high-degree
            # hub nodes produce large activation magnitudes that destabilize gradient flow.
            # LayerNorm normalizes per-node independently, preventing hub-induced gradient explosion.
            self.norm_1 = nn.LayerNorm(hidden_dim)
            self.norm_2 = nn.LayerNorm(hidden_dim)
            self.norm_3 = nn.LayerNorm(hidden_dim)
            self.dropout = nn.Dropout(0.20)
            self.classifier = nn.Linear(hidden_dim, 1)
            # v43: Initialize classifier bias to log(pi/(1-pi)) so the model starts at the
            # true positive rate (~0.03) rather than 0.5. This prevents the class-imbalance
            # gradient flood that collapses GNN probabilities to a constant.
            with torch.no_grad():
                self.classifier.bias.fill_(prior_logit)

        def forward(self, x_tensor: Any, adjacency_tensor: Any) -> Any:
            # Normalize input features before aggregation.
            x_normed = self.input_bn(x_tensor)
            neighbor_1 = torch.sparse.mm(adjacency_tensor, x_normed)
            hidden = self.norm_1(F.relu(self.self_linear_1(x_normed) + self.neighbor_linear_1(neighbor_1)))
            hidden = self.dropout(hidden)
            neighbor_2 = torch.sparse.mm(adjacency_tensor, hidden)
            hidden = self.norm_2(F.relu(self.self_linear_2(hidden) + self.neighbor_linear_2(neighbor_2)))
            hidden = self.dropout(hidden)
            # v47: Layer 3 — 3-hop aggregation.
            neighbor_3 = torch.sparse.mm(adjacency_tensor, hidden)
            hidden = self.norm_3(F.relu(self.self_linear_3(hidden) + self.neighbor_linear_3(neighbor_3)))
            hidden = self.dropout(hidden)
            return self.classifier(hidden).squeeze(-1)

    # Build labels and train_mask BEFORE creating model (prior_logit needs them).
    labels = torch.full((len(graph.user_ids),), -1.0, dtype=torch.float32)
    label_frame = label_frame.copy()
    label_frame["user_id"] = pd.to_numeric(label_frame["user_id"], errors="coerce").astype("Int64")
    label_frame["status"] = pd.to_numeric(label_frame["status"], errors="coerce").astype("Int64")
    label_frame = label_frame.dropna(subset=["user_id", "status"])
    for _, row in label_frame.iterrows():
        user_id = int(row["user_id"])
        if user_id in graph.user_index:
            labels[graph.user_index[user_id]] = float(int(row["status"]))

    train_mask = torch.zeros(len(graph.user_ids), dtype=torch.bool)
    for user_id in train_user_ids:
        if user_id in graph.user_index:
            train_mask[graph.user_index[user_id]] = True

    # Compute positive prior from training labels for bias initialization.
    # Standard practice for class-imbalanced classification: initialize the output
    # bias to log(pi/(1-pi)) so sigmoid(bias)=pi (true positive rate, ~0.03).
    # Without this, default bias=0 → sigmoid(0)=0.5, forcing gradients to push ALL
    # users toward 0. With 30:1 imbalance, this gradient flood dominates and the
    # model collapses to predicting a constant ~0.5 or overshoots to constant near 0.
    _label_vals = labels[train_mask]
    _pi = max(0.001, min(0.999, float((_label_vals == 1.0).float().mean().item())))
    _prior_logit = float(np.log(_pi / (1.0 - _pi)))  # ≈ -3.48 for pi=0.03

    model = UserGraphSAGE(x.size(1), hidden_dim, _prior_logit)
    valid_mask = torch.zeros(len(graph.user_ids), dtype=torch.bool)
    for user_id in valid_user_ids or set():
        if user_id in graph.user_index:
            valid_mask[graph.user_index[user_id]] = True
    has_valid = bool(valid_mask.any().item())

    device = _device(torch)
    model = model.to(device)
    x = x.to(device)
    adjacency = adjacency.to(device)
    labels = labels.to(device)
    train_mask = train_mask.to(device)
    valid_mask = valid_mask.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    best_score = -1.0
    best_epoch = 0
    wait = 0

    # v44: Focal loss for GNN training — addresses the 30:1 class imbalance more
    # effectively than weighted BCE.
    #
    # Weighted BCE (pos_weight=15) scales the loss uniformly for all positives.
    # Focal loss (Lin et al. 2017, RetinaNet) additionally DOWN-WEIGHTS easy
    # negatives by (1-p)^gamma, focusing training on hard misclassified samples.
    #
    # For AML detection with 3% fraud rate:
    # - Many negatives are confidently predicted as negative (p≈0.02) → (1-p)^2 ≈ 0.96 → full loss
    # - Hard negatives (p≈0.3) → (1-p)^2 = 0.49 → 51% weight reduction
    # - Hard positives (p≈0.4) → p^2 = 0.16 → focuses gradient on these
    # gamma=2.0 is standard; alpha=0.75 gives positive class 75% weight.
    # Combined with prior_logit bias initialization and BatchNorm, focal loss
    # prevents GNN collapse while providing better gradient signal than BCE.
    _train_labels = labels[train_mask]
    _alpha_focal = 0.75  # positive class weight in focal loss
    _gamma_focal = 2.0   # focusing parameter

    def _focal_loss(logits_train: "Any", labels_train: "Any") -> "Any":
        """Focal loss for binary classification (Lin et al. 2017).

        FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
        where alpha_t = alpha for positives, (1-alpha) for negatives.
        """
        probs = torch.sigmoid(logits_train)
        # Clamp for numerical stability
        probs_clamped = probs.clamp(1e-7, 1.0 - 1e-7)
        pos_mask = labels_train == 1.0
        # Per-sample BCE loss (unreduced)
        bce = -labels_train * torch.log(probs_clamped) - (1.0 - labels_train) * torch.log(1.0 - probs_clamped)
        # Focal modulation: down-weight easy samples
        p_t = torch.where(pos_mask, probs_clamped, 1.0 - probs_clamped)
        focal_weight = (1.0 - p_t) ** _gamma_focal
        # Alpha weighting: higher weight for positives
        alpha_t = torch.where(pos_mask,
                              torch.tensor(_alpha_focal, device=device),
                              torch.tensor(1.0 - _alpha_focal, device=device))
        return (alpha_t * focal_weight * bce).mean()

    for epoch in range(max_epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(x, adjacency)
        loss = _focal_loss(logits[train_mask], labels[train_mask])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            logits = model(x, adjacency)
            probabilities = torch.sigmoid(logits)
            if has_valid:
                valid_probs = probabilities[valid_mask].detach().cpu().numpy()
                valid_labels = labels[valid_mask].detach().cpu().numpy()
                positive_mask = valid_labels.astype(int) == 1
                # v43: Use AP (average precision) instead of mean probability difference.
                # AP directly measures ranking quality — a key determinant of blend eligibility
                # (AP >= 0.08 threshold). Mean-prob-diff can be maximized by outputting a large
                # constant for all positives, which inflates the metric but gives poor AP.
                # With AP as early stopping target, the model is guided toward better ranking.
                if positive_mask.any() and (~positive_mask).any():
                    try:
                        from sklearn.metrics import average_precision_score as _ap
                        score = float(_ap(valid_labels.astype(int), valid_probs))
                    except Exception:
                        # Fallback: mean probability difference if sklearn unavailable
                        score = float(valid_probs[positive_mask].mean() - valid_probs[~positive_mask].mean())
                else:
                    score = float(valid_probs.mean()) if len(valid_probs) else 0.0
            else:
                score = float(probabilities[train_mask].mean().item())
        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if has_valid and wait >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits = model(x, adjacency)
        full_probabilities = torch.sigmoid(logits).detach().cpu().numpy()
        validation_probabilities = full_probabilities[valid_mask.detach().cpu().numpy()] if has_valid else None

    # Free GPU memory immediately after extracting numpy arrays — prevents
    # VRAM staying allocated during subsequent CPU-bound CatBoost training.
    del model, x, adjacency, logits
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    model_state = {
        "state_dict": {key: value.cpu() for key, value in best_state.items()},
        "metadata": {
            "user_feature_columns": feature_columns,
            "hidden_dim": hidden_dim,
            "max_epochs": max_epochs,
            "best_epoch": best_epoch,
            "user_ids": graph.user_ids,
        },
    }
    return GraphModelFitResult(
        model_state=model_state,
        model_meta=model_state["metadata"],
        validation_probabilities=validation_probabilities,
        full_probabilities=full_probabilities,
    )


def save_graph_model(model_state: dict[str, Any], path: Path) -> None:
    torch, _, _ = _require_torch()
    torch.save(model_state, path)


def load_graph_model(path: Path) -> dict[str, Any]:
    torch, _, _ = _require_torch()
    return torch.load(path, map_location="cpu")


def predict_graph_model(graph: TransductiveGraph, model_state: dict[str, Any]) -> pd.DataFrame:
    torch, F, nn = _require_torch()
    metadata = model_state["metadata"]
    feature_columns = metadata["user_feature_columns"]
    x = _feature_tensor(graph, torch, feature_columns)
    adjacency = _normalized_adjacency_tensor(graph, torch)

    class UserGraphSAGE(nn.Module):
        # v47: Mirror train architecture exactly — 3-layer GNN with hidden_dim=96.
        # Old saved models (pre-v47, 2-layer) will load with strict=False; missing
        # layer-3 weights are randomly initialized (neutral: they're retrained on next run).
        def __init__(self, input_dim: int, hidden_dim: int) -> None:
            super().__init__()
            self.input_bn = nn.BatchNorm1d(input_dim)
            self.self_linear_1 = nn.Linear(input_dim, hidden_dim)
            self.neighbor_linear_1 = nn.Linear(input_dim, hidden_dim)
            self.self_linear_2 = nn.Linear(hidden_dim, hidden_dim)
            self.neighbor_linear_2 = nn.Linear(hidden_dim, hidden_dim)
            self.self_linear_3 = nn.Linear(hidden_dim, hidden_dim)
            self.neighbor_linear_3 = nn.Linear(hidden_dim, hidden_dim)
            self.norm_1 = nn.LayerNorm(hidden_dim)
            self.norm_2 = nn.LayerNorm(hidden_dim)
            self.norm_3 = nn.LayerNorm(hidden_dim)
            self.dropout = nn.Dropout(0.20)
            self.classifier = nn.Linear(hidden_dim, 1)

        def forward(self, x_tensor: Any, adjacency_tensor: Any) -> Any:
            x_normed = self.input_bn(x_tensor)
            neighbor_1 = torch.sparse.mm(adjacency_tensor, x_normed)
            hidden = self.norm_1(F.relu(self.self_linear_1(x_normed) + self.neighbor_linear_1(neighbor_1)))
            hidden = self.dropout(hidden)
            neighbor_2 = torch.sparse.mm(adjacency_tensor, hidden)
            hidden = self.norm_2(F.relu(self.self_linear_2(hidden) + self.neighbor_linear_2(neighbor_2)))
            hidden = self.dropout(hidden)
            neighbor_3 = torch.sparse.mm(adjacency_tensor, hidden)
            hidden = self.norm_3(F.relu(self.self_linear_3(hidden) + self.neighbor_linear_3(neighbor_3)))
            hidden = self.dropout(hidden)
            return self.classifier(hidden).squeeze(-1)

    model = UserGraphSAGE(x.size(1), int(metadata["hidden_dim"]))
    # strict=False for backward compatibility: old models (pre-v43) lack input_bn/norm
    # weights; PyTorch initialises them to identity so inference degrades gracefully.
    missing, unexpected = model.load_state_dict(model_state["state_dict"], strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected keys in GNN state_dict: {unexpected}")
    model.eval()
    with torch.no_grad():
        probabilities = torch.sigmoid(model(x, adjacency)).detach().cpu().numpy()
    return pd.DataFrame({"user_id": graph.user_ids, "graph_probability": probabilities})
