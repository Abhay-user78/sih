"""Phase 2: message-passing GNN anomaly detector.

Runs on the icosahedral-mesh graph (nodes = mesh vertices, edges = mesh
edges). For each timestep it scores every node with an anomaly probability;
high-scoring nodes are then clustered into a macro-scale bounding region.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GraphConv

from .config import CONFIG, model_dir
from . import physics


class AnomalyGNN(nn.Module):
    """3-layer GraphConv with an MLP readout head."""

    def __init__(self, in_dim: int, hidden: int = 48, dropout: float = 0.3):
        super().__init__()
        self.conv1 = GraphConv(in_dim, hidden)
        self.conv2 = GraphConv(hidden, hidden)
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, data) -> torch.Tensor:
        x = data.x
        ei = data.edge_index
        x = F.relu(self.conv1(x, ei))
        x = F.relu(self.conv2(x, ei))
        return self.head(x).squeeze(-1)


def build_graphs(feats: np.ndarray, labels: np.ndarray,
                 edge_index: np.ndarray,
                 weights: np.ndarray | None = None) -> list[Data]:
    """Convert (T, N, F) features + (T, N) labels into PyG Data graphs."""
    ei = torch.tensor(edge_index, dtype=torch.long)
    graphs = []
    for t in range(feats.shape[0]):
        g = Data(
            x=torch.tensor(feats[t], dtype=torch.float),
            y=torch.tensor(labels[t], dtype=torch.float),
            edge_index=ei,
        )
        if weights is not None:
            g.w = torch.tensor(weights[t], dtype=torch.float)
        graphs.append(g)
    return graphs


def make_batch(graphs: list[Data]) -> Batch:
    return Batch.from_data_list(graphs)


def train(dataset: dict, split=(0.8, 0.1, 0.1), epochs: int = 40,
          hidden: int = 32, lr: float = 3e-4, wd: float = 1e-3,
          patience: int = 8, late_epoch: int = 12,
          seed: int = 42, device: str | None = None,
          label_mask=None,
          save_path: str | None = None):
    """Train/validate the GNN. Data split is over timesteps (contiguous).

    Adds early stopping on validation loss after `late_epoch` warm-up epochs.
    `label_mask` (T,N) multiplies target labels before training; this is how
    human feedback (operator corrections) is injected.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = (device or CONFIG.device)
    device = device if torch.cuda.is_available() else "cpu"

    feats = dataset["features"]
    labels = dataset["labels"]
    if label_mask is not None:
        labels = labels * np.asarray(label_mask, dtype=np.float32)
    T, N, Fdim = feats.shape

    n_tr = int(T * split[0])
    n_va = int(T * split[1])
    w = 1.0 + 4.0 * np.clip(labels[:n_tr], 0.0, 1.0)
    tr = build_graphs(feats[:n_tr], labels[:n_tr], dataset["edge_index"],
                      weights=w)
    va = build_graphs(feats[n_tr:n_tr + n_va], labels[n_tr:n_tr + n_va],
                      dataset["edge_index"])
    node_conv = dataset.get("node_conv")
    physics_on = (CONFIG.use_physics_loss and node_conv is not None)
    if physics_on:
        for i, g in enumerate(tr):
            g.sup = torch.tensor(node_conv[i], dtype=torch.float)
        for i, g in enumerate(va):
            g.sup = torch.tensor(node_conv[n_tr + i], dtype=torch.float)
        print("[physics] GNN loss: anomaly-score x convergence-support "
              "penalty ENABLED")

    model = AnomalyGNN(Fdim, hidden=hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    def crit(logit, g):
        bce = F.binary_cross_entropy_with_logits(logit, g.y, reduction="none")
        base = (bce * g.w).mean() if hasattr(g, "w") else bce.mean()
        if physics_on and hasattr(g, "sup"):
            base = base + physics.gnn_physics_penalty(
                logit, g.sup, weight=CONFIG.physics_gnn_weight)
        return base

    def run(loader, train_mode):
        model.train(train_mode)
        tot, cnt, n = 0.0, 0, 0
        for g in (tr if train_mode else va):
            g = g.to(device)
            logit = model(g)
            loss = crit(logit, g)
            if train_mode:
                opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * g.num_nodes
            n += g.num_nodes
            with torch.no_grad():
                pred = (torch.sigmoid(logit) > 0.5).float()
                cnt += int((pred == (g.y > 0.5).float()).sum())
        return tot / n, cnt / n

    history = {"train_loss": [], "val_acc": [], "val_loss": []}
    best, best_state, no_improve = None, None, 0
    for ep in range(1, epochs + 1):
        tl, ta = run(None, True)
        vl, va_ = run(None, False)
        history["train_loss"].append(tl)
        history["val_loss"].append(vl)
        history["val_acc"].append(va_)
        if ep >= late_epoch:
            if best is None or vl < best:
                best, no_improve = vl, 0
                best_state = {k: v.detach().clone() for k, v in
                              model.state_dict().items()}
            else:
                no_improve += 1
        if ep % 5 == 0 or ep == epochs:
            print(f"  ep {ep:3d}  train_loss {tl:.4f}  "
                  f"val_loss {vl:.4f}  val_acc {va_:.4f}")
        if no_improve >= patience:
            print(f"  early stop at epoch {ep}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    torch.save({"state": model.state_dict(), "feature_dim": Fdim,
                "hidden": hidden, "history": history},
               save_path or (model_dir("gnn") / "gnn_anomaly.pt"))
    return model, history


def predict(dataset: dict, model: AnomalyGNN,
            start: int, end: int, device: str | None = None) -> np.ndarray:
    """Return (n_steps, N) anomaly probabilities for timesteps [start, end)."""
    device = (device or CONFIG.device)
    device = device if torch.cuda.is_available() else "cpu"
    model.eval().to(device)
    feats = dataset["features"]
    out = np.zeros((end - start, dataset["n_nodes"]), dtype=np.float32)
    for s in range(start, end):
        g = build_graphs(feats[s:s + 1], np.zeros((1, dataset["n_nodes"])),
                         dataset["edge_index"])[0].to(device)
        with torch.no_grad():
            out[s - start] = torch.sigmoid(model(g)).cpu().numpy()
    return out


def bounding_region(scores: np.ndarray, thr: float = None, node_lat=None,
                    node_lon=None, min_size: int = 5,
                    max_gap_km: float = 260.0,
                    abs_floor: float = 0.10,
                    hotspot_q: float = 97.5, **kw) -> dict:
    """Cluster high-scoring nodes (DBSCAN on sphere coords via xyz) and
    return the macro bounding region of the dominant cluster.

    If `thr` is None, an adaptive hotspot threshold is used:
      thr = max(abs_floor, <hotspot_q>-th percentile of scores)
    which selects the relative anomaly hotspots regardless of model
    calibration. Returns dict with cluster centres, bbox, radius estimate.
    """
    from scipy.spatial.distance import pdist, squareform
    from sklearn.cluster import DBSCAN

    if thr is None:
        thr = max(abs_floor, float(np.percentile(scores, hotspot_q)))

    hit = np.where(scores >= thr)[0]
    if len(hit) < min_size:
        return {"n_hits": int(len(hit)), "clusters": [], "bbox": None,
                "thr": float(thr)}

    node_xyz = np.column_stack([
        np.cos(np.deg2rad(node_lat)) * np.cos(np.deg2rad(node_lon)),
        np.cos(np.deg2rad(node_lat)) * np.sin(np.deg2rad(node_lon)),
        np.sin(np.deg2rad(node_lat))])

    d = squareform(pdist(node_xyz[hit])) * 6371.0  # chord-based km approx.
    del d  # cluster in 3-D on the scaled sphere (km units) below
    eps = max_gap_km
    labels_ = DBSCAN(eps=eps, min_samples=3).fit(
        node_xyz[hit] * 6371.0).labels_
    clusters = sorted(
        [lab for lab in set(labels_) if lab >= 0],
        key=lambda lb: -int((labels_ == lb).sum()))

    out = {"n_hits": int(len(hit)), "clusters": [], "bbox": None,
           "thr": float(thr)}
    for lb in clusters[:2]:
        idx = hit[labels_ == lb]
        c_lat = node_lat[idx]; c_lon = node_lon[idx]
        out["clusters"].append({
            "n_nodes": int(len(idx)),
            "lat": float(node_lat[idx].mean()),
            "lon": float(np.mean(np.mod(node_lon[idx], 360.0))),
            "mean_score": float(scores[idx].mean()),
            "max_score": float(scores[idx].max()),
            "bbox": (float(c_lat.min()), float(c_lat.max()),
                     float(c_lon.min()), float(c_lon.max())),
        })
    if out["clusters"]:
        c = out["clusters"][0]
        out["bbox"] = c["bbox"]
        out["centre"] = {"lat": c["lat"], "lon": c["lon"]}
        out["node_mask"] = hit[np.where(labels_ >= 0)[0]]
    return out