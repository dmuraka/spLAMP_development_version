"""Spatial neighbor utilities (frNN-style fixed-radius search and KNN).

Uses sklearn ``BallTree`` for radius queries because its
``query_radius(return_distance=True)`` returns indices *and* distances in a
single C-level call, sparing the Python-side distance recomputation that the
scipy cKDTree path required.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
from scipy.spatial import cKDTree
from sklearn.neighbors import BallTree


def build_tree(coords: np.ndarray) -> BallTree:
    """Build a reusable BallTree on ``coords`` (Euclidean, 2D)."""
    coords = np.ascontiguousarray(coords, dtype=float)
    return BallTree(coords, metric="euclidean")


def frnn_csr_from_tree(
    tree: BallTree, query: np.ndarray, eps: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fixed-radius NN against a pre-built tree, CSR-flattened output."""
    query = np.ascontiguousarray(query, dtype=float)
    ids, dists = tree.query_radius(query, r=eps, return_distance=True)
    sizes = np.fromiter((a.size for a in ids), dtype=np.int64, count=len(ids))
    offsets = np.empty(len(ids) + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(sizes, out=offsets[1:])
    if offsets[-1] == 0:
        return (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=float),
            offsets,
        )
    id_flat = np.concatenate(ids).astype(np.int64, copy=False)
    dist_flat = np.concatenate(dists).astype(float, copy=False)
    return id_flat, dist_flat, offsets


def frnn_csr(
    coords: np.ndarray, query: np.ndarray, eps: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One-shot frNN: builds a tree and queries it. Prefer caching the tree."""
    return frnn_csr_from_tree(build_tree(coords), query, eps)


def knnx(coords: np.ndarray, query: np.ndarray, k: int = 1) -> np.ndarray:
    """K nearest neighbor indices (FNN::get.knnx replacement). Returns (n_query, k)."""
    tree = cKDTree(np.ascontiguousarray(coords, dtype=float))
    _, idx = tree.query(np.ascontiguousarray(query, dtype=float), k=k)
    if k == 1:
        idx = idx[:, None] if idx.ndim == 1 else idx
    return np.asarray(idx, dtype=np.int64)


def frnn(
    coords: np.ndarray, query: np.ndarray, eps: float
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Back-compat list-of-arrays variant. New code should use frnn_csr*."""
    tree = build_tree(coords)
    query = np.ascontiguousarray(query, dtype=float)
    ids, dists = tree.query_radius(query, r=eps, return_distance=True)
    return (
        [a.astype(np.int64, copy=False) for a in ids],
        [d.astype(float, copy=False) for d in dists],
    )
