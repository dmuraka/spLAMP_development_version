"""Internal utilities for the coarse-to-fine spatial downscaling model.

Ports of ``R/internal_utils_ds.R`` and ``src/lwr_ds_chunk.cpp``.

Area indexing convention
------------------------
The R code assumes the user's ``agg_id`` values are ``1..N`` and uses them
directly as row indices into the areal vectors.  Here we map ``agg_id`` to a
dense 0-based area index via ``np.unique(agg_id, return_inverse=True)`` (the
``inverse`` array, ``agg_inv``).  Areal vectors are ordered by sorted-unique
``agg_id`` — exactly what R's ``aggregate(..., by = list(agg_id))`` produces —
so the two representations coincide when ``agg_id`` is a contiguous ``1..N``
label, and this variant is additionally robust to non-contiguous labels.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.cluster import KMeans

from ._kernels import lwr_ds_chunk
from ._neighbors import build_tree, frnn_csr_from_tree


def _agg_sum(values: np.ndarray, agg_inv: np.ndarray, N: int) -> np.ndarray:
    """aggregate(values, by=list(agg_id), sum)[, 2], ordered by sorted-unique id."""
    return np.bincount(agg_inv, weights=np.asarray(values, dtype=float), minlength=N)


def _wls_coef(X: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Weighted least squares coefficients: solve (X^T W X) b = X^T W y."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    w = np.asarray(w, dtype=float).ravel()
    Xw = X * w[:, None]
    XtWX = X.T @ Xw
    XtWy = X.T @ (w * y)
    try:
        return np.linalg.solve(XtWX, XtWy)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(np.sqrt(w)[:, None] * X, np.sqrt(w) * y, rcond=None)[0]


def _kmeans_centers(X: np.ndarray, k: int, seed: int = 4321,
                    max_iter: int = 10) -> np.ndarray:
    k = max(1, int(k))
    km = KMeans(n_clusters=k, n_init=1, max_iter=int(max_iter), random_state=seed)
    km.fit(np.asarray(X, dtype=float))
    return np.asarray(km.cluster_centers_, dtype=float)


def select_opt_id(SSE_valid) -> int:
    """Pick the scale with the smallest validation SSE (R ``select_opt_id``)."""
    sse = np.asarray(SSE_valid, dtype=float)
    if sse.size < 2 or not np.any(np.isfinite(sse)):
        return 0
    finite = np.where(np.isfinite(sse))[0]
    best = finite[int(np.argmin(sse[finite]))]
    return max(int(best), 0)


def multiplicative_pycnophylactic(pred, Y, agg_inv, a, N):
    """Scale per area so that aggregate(a*pred, agg_id, sum) == Y exactly."""
    pred = np.asarray(pred, dtype=float).copy()
    Y = np.asarray(Y, dtype=float)
    a = np.asarray(a, dtype=float)
    Pred = _agg_sum(a * pred, agg_inv, N)
    zero_mask = (Pred == 0) & (Y != 0)
    if np.any(zero_mask):
        a_sum = _agg_sum(a, agg_inv, N)
        for i_area in np.where(zero_mask)[0]:
            in_area = np.where(agg_inv == i_area)[0]
            if in_area.size > 0 and a_sum[i_area] > 0:
                pred[in_area] = Y[i_area] / a_sum[i_area]
        Pred = _agg_sum(a * pred, agg_inv, N)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(Pred > 0, Y / Pred, 0.0)
    return pred * ratio[agg_inv]


@dataclass
class InitialDSState:
    beta_int: np.ndarray          # nx x 1
    beta: np.ndarray              # n x nx
    coords_uni: np.ndarray
    Coords_uni: np.ndarray
    pred: np.ndarray              # n (disaggregate)
    Pred: np.ndarray              # N (areal)
    Resid: np.ndarray             # N
    X: np.ndarray                 # N x nx
    W: np.ndarray                 # N
    W_glob: np.ndarray            # N
    eps_auto: float
    sig2_hat: float
    s2_area: np.ndarray
    Agg_id: np.ndarray            # sorted-unique agg_id values
    agg_inv: np.ndarray           # n, 0-based dense area index
    x: np.ndarray                 # n x nx
    a: np.ndarray                 # n
    xname: list
    n: int
    nx: int
    N: int
    Id_train: np.ndarray          # 0-based area indices


def initial_ds_fun(Y, Y_type, x, a, coords, train_rat, agg_id, Id_train=None,
                   seed: Optional[int] = None):
    """Initial areal fit for the downscaling model. Mirrors R ``initial_ds_fun``."""
    coords = np.asarray(coords, dtype=float)
    Y = np.asarray(Y, dtype=float).ravel()
    N = Y.shape[0]
    n = coords.shape[0]

    # Design matrix: ensure a constant intercept column at position 0.
    xname_in = None
    if x is None:
        x_mat = np.ones((n, 1))
    else:
        if hasattr(x, "columns"):
            xname_in = [str(c) for c in list(x.columns)]
        x_arr = np.asarray(x, dtype=float)
        if x_arr.ndim == 1:
            x_arr = x_arr.reshape(-1, 1)
        if not np.all(x_arr[:, 0] == 1):
            x_mat = np.hstack([np.ones((n, 1)), x_arr])
            if xname_in is not None:
                xname_in = None  # intercept prepended; fall back to generic names
        else:
            x_mat = x_arr
    nx = x_mat.shape[1]
    if xname_in is not None and len(xname_in) == nx:
        xname = xname_in
    else:
        xname = ["Intercept"] + [f"x{i}" for i in range(1, nx)]

    if a is None:
        a = np.ones(n)
    a = np.asarray(a, dtype=float).ravel()

    # 0-based dense area index, ordered by sorted-unique agg_id.
    Agg_id, agg_inv = np.unique(np.asarray(agg_id), return_inverse=True)
    agg_inv = agg_inv.astype(np.int64)

    if Y_type == "sum":
        W = np.ones(N)
    elif Y_type == "mean":
        A = _agg_sum(a, agg_inv, N)
        a = a / A[agg_inv]
        W = np.ones(N)
    else:
        raise ValueError(f"Y_type must be 'sum' or 'mean', got {Y_type!r}")

    # Area-level coordinates (mean of member points) and areal design.
    Coords = np.column_stack([
        _agg_sum(coords[:, 0], agg_inv, N) / np.maximum(np.bincount(agg_inv, minlength=N), 1),
        _agg_sum(coords[:, 1], agg_inv, N) / np.maximum(np.bincount(agg_inv, minlength=N), 1),
    ])
    X = np.column_stack([_agg_sum(a * x_mat[:, j], agg_inv, N) for j in range(nx)])

    from ._utils import _unique_rows_with_inverse
    Coords_uni, inv_C = _unique_rows_with_inverse(Coords)   # inv_C: area -> unique-centroid id
    coords_uni, _ = _unique_rows_with_inverse(coords)
    N_uni = Coords_uni.shape[0]

    rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
    if Id_train is None:
        if train_rat < 1:
            if N_uni <= 1000:
                K = int(round(N_uni * train_rat))
                centers = _kmeans_centers(coords_uni, K, seed=seed if seed is not None else 4321)
                from ._neighbors import knnx
                nn = knnx(Coords_uni, centers, k=1).ravel()
                Id_train_uni = np.unique(nn)  # sorted-unique nearest areas
            else:
                Id_train_uni = np.sort(rng.choice(N_uni, size=int(round(N_uni * train_rat)),
                                                  replace=False))
        else:
            Id_train_uni = np.arange(N_uni)
        # Map unique-centroid training ids back to area indices (R:
        # which(Id_uni %in% Id_train_uni)). Identical to Id_train_uni when every
        # area has a distinct centroid; robust to shared centroids otherwise.
        mask = np.zeros(N_uni, dtype=bool)
        mask[Id_train_uni] = True
        Id_train = np.where(mask[inv_C])[0].astype(np.int64)
    else:
        Id_train = np.asarray(Id_train, dtype=np.int64)

    # Variance-component (nugget) areal weight W_glob.
    s2_area = _agg_sum(a ** 2, agg_inv, N)
    beta_ols = _wls_coef(X, Y, W)
    r0_area = Y - X @ beta_ols
    # vc_fit: lm(r0^2 ~ s2)   over training areas
    tr = Id_train
    if tr.size >= 2:
        A_vc = np.column_stack([np.ones(tr.size), s2_area[tr]])
        vc = np.linalg.lstsq(A_vc, r0_area[tr] ** 2, rcond=None)[0]
        tau2_hat = max(float(vc[0]), 0.0)
        sig2_hat = float(vc[1])
    else:
        tau2_hat = 0.0
        sig2_hat = float("nan")

    if np.isfinite(sig2_hat) and sig2_hat > 0:
        eps_auto = tau2_hat / sig2_hat
        denom_floor = max(1e-6 * float(np.max(s2_area)), np.finfo(float).eps)
        W_glob = 1.0 / np.maximum(s2_area + eps_auto, denom_floor)
    else:
        eps_auto = float("inf")
        W_glob = np.ones(N)

    beta_int = _wls_coef(X, Y, W_glob).reshape(-1, 1)
    Pred = (X @ beta_int).ravel()
    Resid = Y - Pred
    pred = (x_mat @ beta_int).ravel()
    beta = np.tile(beta_int.ravel(), (n, 1))

    return InitialDSState(
        beta_int=beta_int, beta=beta, coords_uni=coords_uni, Coords_uni=Coords_uni,
        pred=pred, Pred=Pred, Resid=Resid, X=X, W=W, W_glob=W_glob,
        eps_auto=eps_auto, sig2_hat=sig2_hat, s2_area=s2_area, Agg_id=Agg_id,
        agg_inv=agg_inv, x=x_mat, a=a, xname=xname, n=n, nx=nx, N=N,
        Id_train=Id_train,
    )


def lwr_ds(coords, coords_uni, Resid, beta_int, Coords_uni, Y, X, W, x, a, band,
           b_old, kernel, Id_train, agg_inv, N, sel_id=None, sse_hv0=None,
           pred_sp=0.0, ridge=False, func="cf_downscale_hv", c_shrink=0.0,
           knots_train_only=False, seed=4321):
    """Per-knot areal weighted local regression. Mirrors R ``lwr_ds``."""
    coords = np.ascontiguousarray(coords, dtype=float)
    n = coords.shape[0]
    nx = x.shape[1]
    x = np.ascontiguousarray(x, dtype=float)

    if kernel == "gau":
        threshold = math.sqrt(-math.log(0.05)) * band
        kernel_id = 2
    elif kernel == "exp":
        threshold = -math.log(0.05) * band
        kernel_id = 1
    else:
        raise ValueError(f"unknown kernel: {kernel}")

    rng = np.random.default_rng(seed)
    Id_train = np.asarray(Id_train, dtype=np.int64)

    # ---- knot selection (sel_id is None on every real call path) ----
    if sel_id is None:
        if knots_train_only:
            pool_area_ids = Id_train
            pool_point_mask = np.isin(agg_inv, Id_train)
        else:
            pool_area_ids = np.arange(N, dtype=np.int64)
            pool_point_mask = np.ones(n, dtype=bool)
        pool_point_idx = np.where(pool_point_mask)[0]
        n_pool_pts = pool_point_idx.size
        n_pool_area = pool_area_ids.size
        coords_pool_pts = coords[pool_point_idx]

        area = (np.ptp(coords[:, 0])) ** 2 + (np.ptp(coords[:, 1])) ** 2
        n_knot = int(round(1.5 * area / (band ** 2)))
        if n_knot < n_pool_area:
            coords_cent = _kmeans_centers(coords_pool_pts, n_knot, seed=seed)
        elif n_knot == n_pool_area:
            agg_sel = np.array([rng.choice(np.where(agg_inv == aid)[0])
                                for aid in pool_area_ids], dtype=np.int64)
            coords_cent = coords[agg_sel]
        elif n_pool_pts > n_knot and n_knot > n_pool_area:
            agg_sel_a = np.array([rng.choice(np.where(agg_inv == aid)[0])
                                  for aid in pool_area_ids], dtype=np.int64)
            n_knot_add = n_knot - n_pool_area
            remaining = np.setdiff1d(pool_point_idx, agg_sel_a)
            agg_sel_b = rng.choice(remaining, size=n_knot_add, replace=False)
            agg_sel = np.concatenate([agg_sel_a, agg_sel_b])
            coords_cent = coords[agg_sel]
        else:
            coords_cent = coords_pool_pts
        n_knot = coords_cent.shape[0]
    elif np.asarray(sel_id).dtype.kind == "f" and np.isnan(np.asarray(sel_id).flat[0]):
        coords_cent = coords_uni
        n_knot = coords_cent.shape[0]
    else:
        sel_id = np.asarray(sel_id, dtype=np.int64)
        coords_cent = coords_uni[sel_id]
        n_knot = coords_cent.shape[0]

    # Prior coefficient variance (inf = no ridge).
    B_var = float("inf")
    if b_old is not None and ridge:
        B_var = float(np.mean(np.asarray(b_old) ** 2))

    id_train_flag = np.isin(agg_inv, Id_train)  # point in a training area

    # Neighbor lists; keep knots with >= 3 neighbors.
    tree = build_tree(coords)
    id_flat, dist_flat, offsets = frnn_csr_from_tree(tree, coords_cent, threshold)
    sizes = np.diff(offsets)
    keep = np.where(sizes >= 3)[0]
    if keep.size == 0:
        return {"run": False}

    # Re-pack CSR to only the kept knots.
    keep_id = [id_flat[offsets[k]:offsets[k + 1]] for k in keep]
    keep_dist = [dist_flat[offsets[k]:offsets[k + 1]] for k in keep]
    ksizes = np.fromiter((a_.size for a_ in keep_id), dtype=np.int64, count=len(keep_id))
    koff = np.empty(len(keep_id) + 1, dtype=np.int64)
    koff[0] = 0
    np.cumsum(ksizes, out=koff[1:])
    kid_flat = np.concatenate(keep_id).astype(np.int64) if keep_id else np.zeros(0, np.int64)
    kdist_flat = np.concatenate(keep_dist).astype(float) if keep_dist else np.zeros(0)

    n_kept = keep.size
    b_all = np.zeros((n, nx))
    bv_inv_all = np.zeros((n, nx))
    pv_inv_all = np.zeros((n, nx))
    b_old_mat = np.zeros((n_kept, nx))
    sel_chunk = np.arange(n_kept, dtype=np.int64)
    local_bands = np.full(n_kept, float(band))
    b_var_col = np.full(n_kept, B_var)

    lwr_ds_chunk(
        kid_flat, koff, kdist_flat,
        sel_chunk, local_bands, int(kernel_id), 0,
        np.ascontiguousarray(Resid, dtype=float),
        np.ascontiguousarray(X[:, 0], dtype=float),
        np.ascontiguousarray(W, dtype=float),
        np.ascontiguousarray(a, dtype=float),
        np.ascontiguousarray(agg_inv, dtype=np.int64),
        id_train_flag,
        x,
        b_var_col,
        float(c_shrink),
        b_all, bv_inv_all, pv_inv_all, b_old_mat,
    )
    b_old_out = b_old_mat[:, 0]

    vc = 0  # 0-based spatial-process column
    pred_sp = np.asarray(pred_sp, dtype=float) if np.ndim(pred_sp) else float(pred_sp)

    with np.errstate(divide="ignore", invalid="ignore"):
        b_all[:, vc] = np.where(pv_inv_all[:, vc] != 0, b_all[:, vc] / pv_inv_all[:, vc], 0.0)
    b_all = np.nan_to_num(b_all, nan=0.0)
    pred_sp_add = x[:, vc] * b_all[:, vc]
    bv_all = bv_inv_all.copy()

    if func == "cf_downscale_hv":
        pred_sp_new = pred_sp + pred_sp_add
        pred_sp_new = pred_sp_new - np.mean(pred_sp_new)
        Pred_sp = _agg_sum(a * pred_sp_new, agg_inv, N)
        coef = _wls_coef(X, Y - Pred_sp, W)
        resid_area = (Y - Pred_sp) - X @ coef
        not_train = np.ones(N, dtype=bool)
        not_train[Id_train] = False
        sse_hv_val = float(np.sum(W[not_train] * resid_area[not_train] ** 2))
        sse_hv = sse_hv_val
        run = sse_hv < (sse_hv0 if sse_hv0 is not None else float("inf"))
        if not run:
            sse_hv = sse_hv0
    else:
        pred_sp_new = pred_sp + pred_sp_add
        Pred_sp = _agg_sum(a * pred_sp_new, agg_inv, N)
        sse_hv = float("nan")
        run = True

    if not run:
        return {"run": False, "sse_hv": sse_hv}

    with np.errstate(divide="ignore", invalid="ignore"):
        bv_inv_all[:, vc] = np.where(pv_inv_all[:, vc] != 0,
                                     bv_inv_all[:, vc] / pv_inv_all[:, vc], 0.0)
        bv_all[:, vc] = np.where(bv_inv_all[:, vc] != 0, 1.0 / bv_inv_all[:, vc], np.inf)
    not_vc = np.ones(nx, dtype=bool)
    not_vc[vc] = False
    bv_all[:, not_vc] = np.nan

    return {"beta": b_all, "beta_v": bv_all, "sel_id": sel_id,
            "coords_cent": coords_cent, "pred_sp": pred_sp_new, "Pred_sp": Pred_sp,
            "b_old": b_old_out, "run": True, "sse_hv": sse_hv, "vc_sel": vc,
            "sse_hv0": sse_hv0}
