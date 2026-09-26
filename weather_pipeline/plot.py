"""matplotlib/cartopy plotting helpers."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _save(fig, path: str | Path, dpi: int = 110) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] saved {path}")


def plot_grid_field(lat, lon, field, title: str, path: str,
                    vmin=None, vmax=None, cmap="turbo") -> None:
    """Simple pcolormesh on a rectilinear (lat desc, lon asc) grid."""
    fig, ax = plt.subplots(figsize=(9, 6))
    m = ax.pcolormesh(lon, lat, field, shading="auto",
                      vmin=vmin, vmax=vmax, cmap=cmap)
    fig.colorbar(m, ax=ax, label=title)
    ax.set_title(title)
    ax.set_xlabel("longitude (deg E)")
    ax.set_ylabel("latitude (deg N)")
    _save(fig, path)


def plot_mesh(node_lat, node_lon, values, title: str, path: str,
              cmap="turbo", node_size: float = 8.0) -> None:
    """Scatter mesh-node values; mark the edges lightly if provided."""
    fig, ax = plt.subplots(figsize=(9, 6))
    sc = ax.scatter(node_lon, node_lat, c=values, s=node_size, cmap=cmap)
    fig.colorbar(sc, ax=ax, label=title)
    ax.set_title(title)
    ax.set_xlabel("longitude (deg E)")
    ax.set_ylabel("latitude (deg N)")
    _save(fig, path)


def plot_mesh_edges(mesh, path: str = "outputs/mesh_graph.png") -> None:
    """Visualise nodes + edges of a mesh subgraph (Phase 1c check)."""
    fig, ax = plt.subplots(figsize=(9, 6))
    e = mesh["edge_index"]
    v = mesh["vertices"]
    for u, w in e.T:
        ax.plot([v[u, 0], v[w, 0]], [v[u, 1], v[w, 1]], "-",
                color="0.75", lw=0.4)
    ax.scatter(v[:, 0], v[:, 1], s=4, c="navy")
    ax.set_title(f"icosphere subgraph: {v.shape[0]} nodes, {e.shape[1]} edges")
    ax.set_aspect("equal")
    _save(fig, path)