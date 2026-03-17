from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from hardware import hardware_profile
from official.common import RANDOM_SEED
from official.graph_dataset import TransductiveGraph


def _require_torch() -> tuple[Any, Any, Any]:
    try:
        # Pre-load libcuda.so before torch initializes to avoid "CUDA unknown error"
        # on machines where the CUDA driver library path isn't in the default ld cache.
        import ctypes
        try:
            ctypes.CDLL("libcuda.so")
        except OSError:
            pass  # Not available or not needed — torch will handle it
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
    profile = hardware_profile()
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    if profile.gpu_enabled and torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)


def _device(torch: Any) -> Any:
    profile = hardware_profile()
    if profile.gpu_enabled and torch.cuda.is_available():
        device_id = profile.gpu_device_id if profile.gpu_device_id is not None else 0
        if device_id >= torch.cuda.device_count():
            device_id = 0
        return torch.device(f"cuda:{device_id}")
    return torch.device("cpu")


def _user_feature_columns(graph: TransductiveGraph) -> list[str]:
    return [column for column in graph.user_feature_frame.columns if column != "user_id"]


def _feature_tensor(
    graph: TransductiveGraph,
    torch: Any,
    feature_columns: list[str],
    col_mean: np.ndarray | None = None,
    col_std: np.ndarray | None = None,
) -> tuple[Any, np.ndarray, np.ndarray]:
    frame = graph.user_feature_frame[["user_id", *feature_columns]].copy()
    values = frame[feature_columns].fillna(0.0).to_numpy(dtype=np.float32)
    if col_mean is None or col_std is None:
        # Standardize each feature column: prevents exploding hidden representations
        # when 150+ raw features span vastly different scales during 2-hop aggregation.
        col_mean = values.mean(axis=0, keepdims=True)
        col_std = values.std(axis=0, keepdims=True)
        col_std[col_std < 1e-8] = 1.0
    values = (values - col_mean) / col_std
    return torch.tensor(values, dtype=torch.float32), col_mean, col_std


def _normalized_adjacency_tensor(graph: TransductiveGraph, torch: Any) -> Any:
    user_count = len(graph.user_ids)
    if graph.collapsed_edges.empty:
        indices = torch.zeros((2, 0), dtype=torch.long)
        values = torch.zeros((0,), dtype=torch.float32)
        return torch.sparse_coo_tensor(indices, values, size=(user_count, user_count)).coalesce()
    src = graph.collapsed_edges["src_user_id"].astype(int).map(graph.user_index).to_numpy(dtype=np.int64)
    dst = graph.collapsed_edges["dst_user_id"].astype(int).map(graph.user_index).to_numpy(dtype=np.int64)
    weight = graph.collapsed_edges["weight"].astype(float).to_numpy(dtype=np.float32)
    degree = np.zeros(user_count, dtype=np.float32)
    np.add.at(degree, src, weight)
    degree[degree == 0.0] = 1.0
    norm_weight = weight / degree[src]
    indices = torch.tensor(np.vstack([src, dst]), dtype=torch.long)
    values = torch.tensor(norm_weight, dtype=torch.float32)
    return torch.sparse_coo_tensor(indices, values, size=(user_count, user_count)).coalesce()


def train_graphsage_model(
    graph: TransductiveGraph,
    label_frame: pd.DataFrame,
    train_user_ids: set[int],
    valid_user_ids: set[int] | None = None,
    max_epochs: int = 40,
    hidden_dim: int = 64,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 8,
) -> GraphModelFitResult:
    torch, F, nn = _require_torch()
    profile = hardware_profile()
    torch.set_num_threads(profile.cpu_threads)
    _seed_everything(torch)
    feature_columns = _user_feature_columns(graph)
    x, col_mean, col_std = _feature_tensor(graph, torch, feature_columns)
    adjacency = _normalized_adjacency_tensor(graph, torch)

    class UserGraphSAGE(nn.Module):
        def __init__(self, input_dim: int, hidden_dim: int) -> None:
            super().__init__()
            self.self_linear_1 = nn.Linear(input_dim, hidden_dim)
            self.neighbor_linear_1 = nn.Linear(input_dim, hidden_dim)
            self.norm_1 = nn.LayerNorm(hidden_dim)
            self.self_linear_2 = nn.Linear(hidden_dim, hidden_dim)
            self.neighbor_linear_2 = nn.Linear(hidden_dim, hidden_dim)
            self.norm_2 = nn.LayerNorm(hidden_dim)
            self.dropout = nn.Dropout(0.20)
            self.classifier = nn.Linear(hidden_dim, 1)

        def forward(self, x_tensor: Any, adjacency_tensor: Any) -> Any:
            neighbor_1 = torch.sparse.mm(adjacency_tensor, x_tensor)
            hidden = self.norm_1(F.relu(self.self_linear_1(x_tensor) + self.neighbor_linear_1(neighbor_1)))
            hidden = self.dropout(hidden)
            neighbor_2 = torch.sparse.mm(adjacency_tensor, hidden)
            hidden = self.norm_2(F.relu(self.self_linear_2(hidden) + self.neighbor_linear_2(neighbor_2)))
            hidden = self.dropout(hidden)
            return self.classifier(hidden).squeeze(-1)

    model = UserGraphSAGE(x.size(1), hidden_dim)
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
    valid_mask = torch.zeros(len(graph.user_ids), dtype=torch.bool)
    for user_id in valid_user_ids or set():
        if user_id in graph.user_index:
            valid_mask[graph.user_index[user_id]] = True
    has_valid = bool(valid_mask.any().item())

    device = _device(torch)
    if device.type == "cuda":
        torch.cuda.set_device(device.index or 0)
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

    for epoch in range(max_epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(x, adjacency)
        n_pos = labels[train_mask].sum().item()
        n_neg = (train_mask.sum() - n_pos).item()
        pos_weight = torch.tensor([max(1.0, n_neg / max(n_pos, 1))], device=device)
        loss = F.binary_cross_entropy_with_logits(
            logits[train_mask], labels[train_mask], pos_weight=pos_weight
        )
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            logits = model(x, adjacency)
            probabilities = torch.sigmoid(logits)
            if has_valid:
                valid_probs_np = probabilities[valid_mask].detach().cpu().numpy()
                valid_labels_np = labels[valid_mask].detach().cpu().numpy().astype(int)
                if valid_labels_np.sum() > 0:
                    score = average_precision_score(valid_labels_np, valid_probs_np)
                else:
                    score = 0.0
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

    model_state = {
        "state_dict": {key: value.cpu() for key, value in model.state_dict().items()},
        "metadata": {
            "user_feature_columns": feature_columns,
            "hidden_dim": hidden_dim,
            "max_epochs": max_epochs,
            "best_epoch": best_epoch,
            "user_ids": graph.user_ids,
            "feature_mean": col_mean.tolist(),
            "feature_std": col_std.tolist(),
            "hardware_profile": profile.to_dict(),
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
    # Use training-time statistics for consistent inference-time standardization
    feature_mean = np.array(metadata.get("feature_mean", [[0.0] * len(feature_columns)]))
    feature_std = np.array(metadata.get("feature_std", [[1.0] * len(feature_columns)]))
    feature_std[feature_std < 1e-8] = 1.0
    x, _, _ = _feature_tensor(graph, torch, feature_columns, col_mean=feature_mean, col_std=feature_std)
    adjacency = _normalized_adjacency_tensor(graph, torch)

    class UserGraphSAGE(nn.Module):
        def __init__(self, input_dim: int, hidden_dim: int) -> None:
            super().__init__()
            self.self_linear_1 = nn.Linear(input_dim, hidden_dim)
            self.neighbor_linear_1 = nn.Linear(input_dim, hidden_dim)
            self.norm_1 = nn.LayerNorm(hidden_dim)
            self.self_linear_2 = nn.Linear(hidden_dim, hidden_dim)
            self.neighbor_linear_2 = nn.Linear(hidden_dim, hidden_dim)
            self.norm_2 = nn.LayerNorm(hidden_dim)
            self.dropout = nn.Dropout(0.20)
            self.classifier = nn.Linear(hidden_dim, 1)

        def forward(self, x_tensor: Any, adjacency_tensor: Any) -> Any:
            neighbor_1 = torch.sparse.mm(adjacency_tensor, x_tensor)
            hidden = self.norm_1(F.relu(self.self_linear_1(x_tensor) + self.neighbor_linear_1(neighbor_1)))
            hidden = self.dropout(hidden)
            neighbor_2 = torch.sparse.mm(adjacency_tensor, hidden)
            hidden = self.norm_2(F.relu(self.self_linear_2(hidden) + self.neighbor_linear_2(neighbor_2)))
            hidden = self.dropout(hidden)
            return self.classifier(hidden).squeeze(-1)

    device = _device(torch)
    model = UserGraphSAGE(x.size(1), int(metadata["hidden_dim"]))
    model.load_state_dict(model_state["state_dict"])
    model = model.to(device)
    x = x.to(device)
    adjacency = adjacency.to(device)
    model.eval()
    with torch.no_grad():
        probabilities = torch.sigmoid(model(x, adjacency)).detach().cpu().numpy()
    return pd.DataFrame({"user_id": graph.user_ids, "graph_probability": probabilities})
