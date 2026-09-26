"""Phase 1c: icosahedral (icosphere) mesh construction and regridding.

Nodes live on the unit sphere; subdivision level `L` yields
10*4**L + 2 vertices and 20*4**L triangular faces. Vertex angular spacing
is ~63.4 deg / 2**L.

The GNN in Phase 2 operates on this graph: nodes = mesh vertices, edges =
mesh edges.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def _base_icosahedron():
    """Return (vertices (12,3), faces (20,3)) of a unit icosahedron."""
    t = (1.0 + 5.0 ** 0.5) / 2.0
    v = np.array([
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1],
    ], dtype=float)
    v = v / np.linalg.norm(v, axis=1, keepdims=True)
    faces = np.array([
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 11, 2], [4, 9, 5], [2, 11, 10], [6, 7, 8], [6, 10, 2],
        [9, 1, 5], [1, 9, 7], [9, 8, 4], [8, 6, 4], [5, 4, 8],
    ], dtype=np.int64)
    return v, faces


def subdivide(v: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """One level of icosphere subdivision. Each triangle -> 4 triangles."""
    mid_cache: dict[tuple[int, int], int] = {}
    new_v = list(v)
    out_faces = []
    for a, b, c in faces:
        ns = []
        for i, j in ((a, b), (b, c), (c, a)):
            key = tuple(sorted((int(i), int(j))))
            if key not in mid_cache:
                m = (v[key[0]] + v[key[1]]) / 2.0
                m = m / np.linalg.norm(m)
                mid_cache[key] = len(new_v)
                new_v.append(m)
            ns.append(mid_cache[key])
        i, j, k = ns
        out_faces.extend([
            (int(a), i, k),
            (int(b), j, i),
            (int(c), k, j),
            (i, j, k),
        ])
    return np.asarray(new_v), np.asarray(out_faces, dtype=np.int64)


def build_mesh(level: int = 5) -> dict:
    """Build a unit icosphere of subdivision `level`."""
    v, f = _base_icosahedron()
    for _ in range(level):
        v, f = subdivide(v, f)
    # unique undirected edges of the triangulation
    edges = set()
    for a, b, _c in f:
        edges.add((int(a), int(b)))
        edges.add((int(a), int(_c)))
        edges.add((int(b), int(_c)))
    return {"vertices": v, "faces": f, "edge_index": np.array(sorted(edges)).T}


def xyz(lat: np.ndarray | float, lon: np.ndarray | float) -> np.ndarray:
    """Convert latitude/longitude (lon in [0,360) or [-180,180)) to xyz on
    the unit sphere."""
    lat = np.deg2rad(lat)
    lon = np.deg2rad(lon)
    clat = np.cos(lat)
    x = clat * np.cos(lon)
    y = clat * np.sin(lon)
    z = np.sin(lat)
    return np.stack([x, y, z], axis=-1)


def latlon(v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """xyz -> latitude, longitude (both degrees, lon in [0,360))."""
    x, y, z = v[:, 0], v[:, 1], v[:, 2]
    lat = np.rad2deg(np.arcsin(np.clip(z, -1, 1)))
    lon = np.rad2deg(np.arctan2(y, x)) % 360.0
    return lat, lon


def subset_to_bbox(mesh: dict, domain) -> dict:
    """Keep mesh vertices inside the domain bbox, one ring of neighbours
    beyond it, and stich the induced subgraph edges."""
    lat, lon = latlon(mesh["vertices"])
    n = mesh["vertices"].shape[0]
    inside = ((lat >= domain.south) & (lat <= domain.north)
              & (np.mod(lon - domain.west, 360.0) <= (domain.east - domain.west)))
    ei = mesh["edge_index"]
    # one ring: vertices connected to any inside vertex
    neigh = np.zeros(n, dtype=bool)
    neigh[ei[0][inside[ei[1]]]] = True
    neigh[ei[1][inside[ei[0]]]] = True
    keep = np.where(inside | neigh)[0]
    keep_set = set(keep.tolist())
    old2new = {int(oi): ni for ni, oi in enumerate(keep)}
    mask = np.array([u in keep_set and v in keep_set for u, v in ei.T])
    sub_ei = ei[:, mask]
    sub_ei = np.stack([np.array([old2new[int(u)] for u in sub_ei[0]]),
                       np.array([old2new[int(v)] for v in sub_ei[1]])])
    sub = {
        "global_index": keep,
        "vertices": mesh["vertices"][keep],
        "edge_index": sub_ei,
        "lat": lat[keep],
        "lon": lon[keep],
    }
    sub["faces_global"] = mesh["faces"]
    return sub


def interpolate_field(sample_values: np.ndarray,
                      sample_lat: np.ndarray,
                      sample_lon: np.ndarray,
                      node_lat: np.ndarray,
                      node_lon: np.ndarray,
                      k: int = 4,
                      p: float = 2.0) -> np.ndarray:
    """Inverse-distance-weighted interpolation of a field sampled on a
    rectilinear lat/lon grid onto mesh nodes (scipy KD-tree on the unit
    sphere). `sample_values` must be shaped (nlat, nlon)."""
    sample_lat = np.asarray(sample_lat, dtype=float)
    sample_lon = np.asarray(sample_lon, dtype=float)
    llat, llon = np.meshgrid(sample_lat, sample_lon, indexing="ij")
    pts = xyz(llat.ravel(), llon.ravel())
    tree = cKDTree(pts)
    q = xyz(np.asarray(node_lat, dtype=float),
            np.asarray(node_lon, dtype=float))
    dists, idx = tree.query(q, k=min(k, len(pts)))
    if dists.ndim == 1:
        dists = dists[:, None]
        idx = idx[:, None]
    w = 1.0 / np.maximum(dists, 1e-9) ** p
    w /= w.sum(axis=1, keepdims=True)
    vals = np.asarray(sample_values).ravel()[idx]
    return (vals * w).sum(axis=1)