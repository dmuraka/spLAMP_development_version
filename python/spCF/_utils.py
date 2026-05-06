"""Internal utilities for the Gaussian (LM) coarse-to-fine model.

Ports of `R/internal_utils_lm.R`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.cluster import KMeans

from ._kernels import lwr_chunk
from ._neighbors import build_tree, frnn_csr, frnn_csr_from_tree, knnx


def _unique_rows_with_inverse(coords: np.ndarray):
    """Return (unique_rows, inverse_index) preserving first-seen row order.

    np.unique sorts; this implementation matches R's `unique()` which
    returns rows in the order of first occurrence.
    """
    coords = np.ascontiguousarray(coords)
    view = np.ascontiguousarray(coords).view(
        np.dtype((np.void, coords.dtype.itemsize * coords.shape[1]))
    ).ravel()
    _, idx_first, inv = np.unique(view, return_index=True, return_inverse=True)
    order = np.argsort(idx_first)
    rank = np.empty_like(order)
    rank[order] = np.arange(order.size)
    inv_remap = rank[inv]
    coords_uni = coords[idx_first[order]]
    return coords_uni, inv_remap


def _kmeans_centers(X: np.ndarray, k: int, max_iter: int, seed: int = 4321) -> np.ndarray:
    km = KMeans(n_clusters=int(k), n_init=1, max_iter=int(max_iter), random_state=seed)
    km.fit(X)
    return np.asarray(km.cluster_centers_, dtype=float)


def _train_split(coords_uni: np.ndarray, train_rat: float, seed: Optional[int]):
    n_uni = coords_uni.shape[0]
    rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
    if n_uni > 30000:
        idx = np.sort(rng.choice(n_uni, size=int(round(n_uni * train_rat)), replace=False))
        return idx
    K_train = int(round(n_uni * train_rat))
    K_val = n_uni - K_train
    pick_train = K_train <= K_val
    K = K_train if pick_train else K_val
    if K <= 0:
        return np.arange(n_uni)
    iter_max = 3 if n_uni > 20000 else (5 if n_uni > 5000 else 10)
    centers = _kmeans_centers(coords_uni, K, iter_max, seed if seed is not None else 4321)
    sel_uni = np.sort(knnx(coords_uni, centers, k=1).ravel())
    if pick_train:
        return sel_uni
    return np.setdiff1d(np.arange(n_uni), sel_uni)


@dataclass
class InitialState:
    xx_inv: np.ndarray
    beta_int: np.ndarray  # nx x 1
    x: np.ndarray
    id_train: np.ndarray
    beta: np.ndarray
    beta_v: np.ndarray
    pred: np.ndarray
    resid: np.ndarray
    n: int
    nx: int
    x_sel: np.ndarray
    xname: list
    coords: np.ndarray


def initial_fun(
    x,
    y,
    coords,
    x_sel=None,
    train_rat: float = 0.75,
    id_train=None,
    seed: Optional[int] = None,
) -> InitialState:
    coords = np.asarray(coords, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError("coords must be (N, 2)")
    y = np.asarray(y, dtype=float).ravel()
    n = y.shape[0]

    coords_uni, inv = _unique_rows_with_inverse(coords)

    if id_train is None:
        if train_rat < 1:
            id_train_uni = _train_split(coords_uni, train_rat, seed)
        else:
            id_train_uni = np.arange(coords_uni.shape[0])
        mask = np.zeros(coords_uni.shape[0], dtype=bool)
        mask[id_train_uni] = True
        id_train = np.where(mask[inv])[0]
    id_train = np.asarray(id_train, dtype=np.int64)

    # Build design matrix with intercept
    one = np.ones((n, 1))
    if x is None:
        x_pre = one
    else:
        x_arr = np.asarray(x, dtype=float)
        if x_arr.ndim == 1:
            x_arr = x_arr.reshape(-1, 1)
        x_pre = np.hstack([one, x_arr])

    if x_pre.shape[1] > 1:
        if x_sel is None:
            sd = np.std(x_pre, axis=0, ddof=1)
            x_sel = (sd != 0)[1:]
        else:
            x_sel = np.asarray(x_sel, dtype=bool)
    else:
        x_sel = np.zeros(0, dtype=bool)

    xname = ["Intercept"]
    if x_sel.size > 0 and x_sel.sum() > 0:
        try:
            import pandas as pd  # type: ignore
            if hasattr(x, "columns"):
                cols = list(pd.DataFrame(x).columns)
                xname += [str(c) for i, c in enumerate(cols) if x_sel[i]]
            else:
                xname += [f"x{i+1}" for i, sel in enumerate(x_sel) if sel]
        except Exception:
            xname += [f"x{i+1}" for i, sel in enumerate(x_sel) if sel]

    keep = np.concatenate([[True], x_sel])
    x_mat = np.ascontiguousarray(x_pre[:, keep], dtype=float)
    nx = x_mat.shape[1]
    xtx = x_mat.T @ x_mat
    xx_inv = np.linalg.solve(xtx, np.eye(nx))
    beta_int = (xx_inv @ x_mat.T @ y).reshape(-1, 1)
    pred = x_mat @ beta_int.ravel()
    resid = y - pred
    sig2 = float(np.sum(resid ** 2) / (n - nx)) if n > nx else 0.0
    beta_int_vcov = sig2 * xx_inv
    beta = np.tile(beta_int.ravel(), (n, 1))
    beta_v = np.tile(np.diag(beta_int_vcov), (n, 1))

    return InitialState(
        xx_inv=xx_inv,
        beta_int=beta_int,
        x=x_mat,
        id_train=id_train,
        beta=beta,
        beta_v=beta_v,
        pred=pred,
        resid=resid,
        n=n,
        nx=nx,
        x_sel=x_sel,
        xname=xname,
        coords=coords,
    )


def bopt_core(par, bands, Z, beta_int, nx, x, y, n_bid, id_train=None):
    """Optimize the spatial-process coefficient via WLS scalar regression.

    Mirrors `bopt_core` in internal_utils_lm.R.
    """
    n = x.shape[0]
    xbeta = np.zeros((n, nx))
    par = float(np.atleast_1d(par)[0])
    bands = np.asarray(bands, dtype=float)
    w = np.exp(-par / bands)
    w = w / w[0]
    bbb = np.zeros(n)
    for i in range(n_bid):
        bbb += w[i] * Z[:, i]
    xbeta[:, 0] = x[:, 0] * (beta_int[0, 0] + bbb)
    if nx > 2:
        for j in range(1, nx):
            xbeta[:, j] = x[:, j] * beta_int[j, 0]
    elif nx == 2:
        xbeta[:, 1] = x[:, 1] * beta_int[1, 0]

    if nx > 1:
        resid = y - np.sum(xbeta[:, 1:], axis=1)
    else:
        resid = y.copy()
    xbeta_tt = xbeta[:, 0:1]
    A = xbeta_tt.T @ xbeta_tt
    bvec = xbeta_tt.T @ resid
    vpar = np.linalg.solve(A, bvec)  # (1, ) or (1, 1)
    vpar = np.asarray(vpar).reshape(-1)
    if id_train is not None:
        mask = np.ones(n, dtype=bool)
        mask[id_train] = False
        diff = resid[mask] - (xbeta_tt[mask] @ vpar).ravel()
        sse = float(np.sum(diff ** 2))
    else:
        diff = resid - (xbeta_tt @ vpar).ravel()
        sse = float(np.sum(diff ** 2))
    return {"sse": sse, "vpar": vpar.reshape(-1, 1)}


def _is_nan_sentinel(sel_id) -> bool:
    if sel_id is None:
        return False
    arr = np.asarray(sel_id)
    if arr.size == 0:
        return False
    if not np.issubdtype(arr.dtype, np.floating):
        return False
    return bool(np.isnan(arr.flat[0]))


def _select_knots(coords_uni, band, sel_id, seed=4321):
    """Choose knot indices into coords_uni.

    sel_id semantics (matching R `lwr`):
      * sel_id=None    : choose afresh (cf_*_hv path).
      * NaN sentinel   : use all unique coords.
      * int array      : use those unique-coord indices.

    Returns (sel_id_for_caller, coords_cent).
    """
    n_uni = coords_uni.shape[0]
    if sel_id is None:
        area = (np.ptp(coords_uni[:, 0])) ** 2 + (np.ptp(coords_uni[:, 1])) ** 2
        n_knot_target = int(round(1.5 * area / (band ** 2)))
        if n_knot_target < n_uni:
            if n_knot_target > 1000:
                rng = np.random.default_rng(seed)
                sel_id = np.sort(rng.choice(n_uni, size=n_knot_target, replace=False))
            else:
                iter_max = 5 if n_uni > 5000 else 10
                centers = _kmeans_centers(coords_uni, n_knot_target, iter_max, seed)
                sel_id = knnx(coords_uni, centers, k=1).ravel()
            coords_cent = coords_uni[sel_id]
            return sel_id, coords_cent
        return np.array([np.nan]), coords_uni
    if _is_nan_sentinel(sel_id):
        return np.array([np.nan]), coords_uni
    sel_id = np.asarray(sel_id, dtype=np.int64).ravel()
    return sel_id, coords_uni[sel_id]


def lwr(
    coords,
    coords_uni,
    resid,
    x,
    band,
    b_old,
    vc,
    ridge,
    kernel,
    id_train,
    y=None,
    beta=None,
    coords0=None,
    x0=None,
    sel_id=None,
    func: str = "cf_lm",
    tree=None,
    tree0=None,
):
    """Driver for the LM lwr loop. Mirrors `lwr` in internal_utils_lm.R.

    ``tree`` / ``tree0`` are optional pre-built ``BallTree`` objects. Passing
    them avoids per-band tree construction and is the recommended path when
    calling ``lwr`` repeatedly inside a band loop.
    """
    coords = np.ascontiguousarray(coords, dtype=float)
    n = coords.shape[0]
    nx = x.shape[1]
    x = np.ascontiguousarray(x, dtype=float)
    if tree is None:
        tree = build_tree(coords)

    if kernel == "gau":
        threshold = math.sqrt(-math.log(0.05)) * band
        kernel_id = 2
    elif kernel == "exp":
        threshold = -math.log(0.05) * band
        kernel_id = 1
    else:
        raise ValueError(f"unknown kernel: {kernel}")

    sel_id_out, coords_cent = _select_knots(coords_uni, band, sel_id)
    n_knot = coords_cent.shape[0]

    # Prior coefficient variance
    B_var = np.full((n_knot, nx), np.inf, dtype=float)
    if b_old is not None and ridge:
        for i in range(nx):
            B_var[:, i] = float(np.mean(b_old[:, i] ** 2))

    n0 = 0 if coords0 is None else int(np.asarray(coords0).shape[0])
    has0 = coords0 is not None

    id_train_flag = np.zeros(n, dtype=bool)
    id_train_flag[np.asarray(id_train, dtype=np.int64)] = True
    vc_int = np.atleast_1d(np.asarray(vc, dtype=np.int64)).ravel()

    b_all = np.zeros((n, nx), dtype=float)
    bv_inv_all = np.zeros((n, nx), dtype=float)
    pv_inv_all = np.zeros((n, nx), dtype=float)
    b_old_out = np.zeros((n_knot, nx), dtype=float)

    if has0:
        coords0 = np.ascontiguousarray(coords0, dtype=float)
        x0 = np.ascontiguousarray(x0, dtype=float)
        b_all0 = np.zeros((n0, nx), dtype=float)
        bv_inv_all0 = np.zeros((n0, nx), dtype=float)
        pv_inv_all0 = np.zeros((n0, nx), dtype=float)
        if tree0 is None:
            tree0 = build_tree(coords0)
    else:
        # placeholders for numba's strict typing
        b_all0 = np.zeros((1, nx), dtype=float)
        bv_inv_all0 = np.zeros((1, nx), dtype=float)
        pv_inv_all0 = np.zeros((1, nx), dtype=float)
        x0 = np.zeros((1, nx), dtype=float)

    sel_list = np.arange(n_knot, dtype=np.int64)
    chunk_size = max(1, min(n_knot, int(math.ceil(1e8 / max(n, 1)))))

    for cs in range(0, n_knot, chunk_size):
        ce = min(cs + chunk_size, n_knot)
        sel_chunk = sel_list[cs:ce]
        query = coords_cent[sel_chunk]
        id_flat, dist_flat, offsets = frnn_csr_from_tree(tree, query, threshold)
        if has0:
            id0_flat, dist0_flat, offsets0 = frnn_csr_from_tree(tree0, query, threshold)
        else:
            id0_flat = np.zeros(0, dtype=np.int64)
            dist0_flat = np.zeros(0, dtype=float)
            offsets0 = np.zeros(1, dtype=np.int64)

        lwr_chunk(
            id_flat, offsets, dist_flat,
            id0_flat, offsets0, dist0_flat,
            sel_chunk,
            id_train_flag,
            np.ascontiguousarray(resid, dtype=float),
            x,
            bool(has0),
            x0,
            B_var,
            vc_int,
            float(band),
            int(kernel_id),
            b_all, bv_inv_all, pv_inv_all,
            b_all0, bv_inv_all0, pv_inv_all0,
            b_old_out,
        )

    # Selection of vc through CV (only used in cf_lm_hv path)
    run = False
    sse_hv = float("nan")
    if func == "cf_lm_hv":
        not_train = np.ones(n, dtype=bool)
        not_train[np.asarray(id_train, dtype=np.int64)] = False
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(
                pv_inv_all[np.ix_(not_train, vc_int)] != 0,
                b_all[np.ix_(not_train, vc_int)] / pv_inv_all[np.ix_(not_train, vc_int)],
                0.0,
            )
        ratio = np.nan_to_num(ratio, nan=0.0)
        pred_hv = np.sum(x[not_train][:, vc_int] * ratio, axis=1)
        resid_hv = resid[not_train] - pred_hv
        sse_hv0 = float(np.sum(resid[not_train] ** 2))
        sse_hv_now = float(np.sum(resid_hv ** 2))
        run = sse_hv_now < sse_hv0
        sse_hv = sse_hv_now if run else sse_hv0
    else:
        run = True

    if not run:
        return {"run": False, "sse_hv": sse_hv}

    bv_all = bv_inv_all.copy()
    not_vc_mask = np.ones(nx, dtype=bool)
    not_vc_mask[vc_int] = False
    with np.errstate(divide="ignore", invalid="ignore"):
        b_all[:, vc_int] = np.where(
            pv_inv_all[:, vc_int] != 0,
            b_all[:, vc_int] / pv_inv_all[:, vc_int],
            0.0,
        )
    b_all[:, not_vc_mask] = 0.0
    b_all = np.nan_to_num(b_all, nan=0.0)

    with np.errstate(divide="ignore", invalid="ignore"):
        bv_inv_all[:, vc_int] = np.where(
            pv_inv_all[:, vc_int] != 0,
            bv_inv_all[:, vc_int] / pv_inv_all[:, vc_int],
            0.0,
        )
        bv_all[:, vc_int] = np.where(
            bv_inv_all[:, vc_int] != 0,
            1.0 / bv_inv_all[:, vc_int],
            np.inf,
        )
    bv_all[:, not_vc_mask] = np.nan
    bv_all = np.where(np.isnan(bv_all) & ~np.isnan(bv_all), bv_all, bv_all)
    bv_all[np.isnan(bv_inv_all)] = np.inf

    pred = np.sum(x * b_all, axis=1)

    if has0:
        with np.errstate(divide="ignore", invalid="ignore"):
            b_all0[:, vc_int] = np.where(
                pv_inv_all0[:, vc_int] != 0,
                b_all0[:, vc_int] / pv_inv_all0[:, vc_int],
                0.0,
            )
        b_all0[:, not_vc_mask] = 0.0
        b_all0 = np.nan_to_num(b_all0, nan=0.0)
        bv_all0 = bv_inv_all0.copy()
        with np.errstate(divide="ignore", invalid="ignore"):
            bv_inv_all0[:, vc_int] = np.where(
                pv_inv_all0[:, vc_int] != 0,
                bv_inv_all0[:, vc_int] / pv_inv_all0[:, vc_int],
                0.0,
            )
            bv_all0[:, vc_int] = np.where(
                bv_inv_all0[:, vc_int] != 0,
                1.0 / bv_inv_all0[:, vc_int],
                np.inf,
            )
        bv_all0[:, not_vc_mask] = np.nan
        pred0 = np.sum(x0 * b_all0, axis=1)
    else:
        b_all0 = bv_all0 = pred0 = None

    return {
        "beta": b_all,
        "beta_v": bv_all,
        "pred": pred,
        "sel_id": sel_id_out,
        "coords_cent": coords_cent,
        "beta0": b_all0,
        "beta0_v": bv_all0,
        "pred0": pred0,
        "b_old": b_old_out,
        "run": True,
        "sse_hv": sse_hv,
        "vc_sel": vc_int,
    }


def sample_from_qrf(rf_qmat: np.ndarray, qs: np.ndarray, n: int, n_draw: int = 100, rng=None):
    """Linear interpolation between quantiles to draw samples (R::approx)."""
    if rng is None:
        rng = np.random.default_rng()
    U = rng.random((n, n_draw))
    draws = np.empty((n, n_draw))
    qs = np.asarray(qs, dtype=float)
    for i in range(n):
        qi = rf_qmat[i, :]
        if not np.all(qi[:-1] <= qi[1:]):
            qi = np.sort(qi)
        draws[i, :] = np.interp(U[i, :], qs, qi)
    return draws


def add_mod_lm(
    add_learn: str,
    train: bool,
    resid: Optional[np.ndarray],
    x: np.ndarray,
    coords: np.ndarray,
    x0: Optional[np.ndarray] = None,
    coords0: Optional[np.ndarray] = None,
    id_train: Optional[np.ndarray] = None,
    nx: int = 0,
    xname: Optional[list] = None,
    seed: int = 123,
    sse_hv: Optional[float] = None,
    a_par: Optional[dict] = None,
    sd_method: str = "residual",
):
    """Additional learning via random forest. Mirrors `add_mod` in internal_utils_lm.R.

    The predict path returns ``pred_v`` / ``pred0_v`` (per-point predictive
    variance) computed according to ``sd_method``:

    * ``"residual"`` (default, back-compat): scalar ``Var(resid)`` broadcast
      to every point. Cheap but homoskedastic and biased high.
    * ``"tree_var"``: per-point variance across the 500 RF trees' predictions
      (sklearn-only, no extra dependency). Heteroskedastic.
    * ``"qrf"``: trains a ``quantile_forest.RandomForestQuantileRegressor``
      with the same hyperparameters, predicts 201 quantiles per point, draws
      200 samples via inverse-CDF and reports their per-point variance.
      This mirrors the R ``ranger(quantreg=TRUE)`` path exactly.
    """
    a_xname = (xname[1:] if xname else []) + ["px", "py"]
    if x.shape[1] > 1:
        a_X = np.hstack([x[:, 1:], coords])
    else:
        a_X = coords.copy()

    if add_learn == "none":
        if train:
            return {"sse_hv": sse_hv, "a_par": None, "a_run": False, "add_learn": add_learn, "a_xname": a_xname}
        return {"mod": None, "pred": 0.0, "pred0": 0.0, "add_learn": add_learn, "a_xname": a_xname}

    if add_learn != "rf":
        raise ValueError(f"add_learn must be 'rf' or 'none', got {add_learn!r}")

    from sklearn.ensemble import RandomForestRegressor

    if train:
        a_run = False
        a_par_out = {"mtry": None, "min_node_size": None}
        mtry_all = sorted({max(1, round((nx + 1) / 5)), max(1, round((nx + 1) / 3)), max(1, round((nx + 1) / 2))})
        param_grid = [(m, mns) for m in mtry_all for mns in (1, 5, 10)]
        for mtry, mns in param_grid:
            rf = RandomForestRegressor(
                n_estimators=500,
                max_features=int(mtry),
                min_samples_leaf=int(mns),
                random_state=seed,
                n_jobs=-1,
            )
            rf.fit(a_X[id_train], resid[id_train])
            mask = np.ones(a_X.shape[0], dtype=bool)
            mask[id_train] = False
            pred_test = rf.predict(a_X[mask])
            sse_rf = float(np.sum((resid[mask] - pred_test) ** 2))
            if sse_hv is None or sse_rf < sse_hv:
                sse_hv = sse_rf
                a_par_out = {"mtry": int(mtry), "min_node_size": int(mns)}
                a_run = True
        return {"sse_hv": sse_hv, "a_par": a_par_out, "a_run": a_run, "add_learn": add_learn, "a_xname": a_xname}

    # predict path
    a_par = a_par or {"mtry": max(1, round((nx + 1) / 3)), "min_node_size": 5}
    if coords0 is not None:
        if x0 is not None and x0.shape[1] > 1:
            a_X0 = np.hstack([x0[:, 1:], coords0])
        else:
            a_X0 = np.asarray(coords0, dtype=float)
    else:
        a_X0 = None

    if sd_method == "qrf":
        try:
            from quantile_forest import RandomForestQuantileRegressor
        except ImportError as e:
            raise ImportError(
                "sd_method='qrf' requires `pip install quantile-forest`"
            ) from e
        rf = RandomForestQuantileRegressor(
            n_estimators=500,
            max_features=int(a_par["mtry"]),
            min_samples_leaf=int(a_par["min_node_size"]),
            random_state=seed,
            n_jobs=-1,
        )
        rf.fit(a_X, np.asarray(resid, dtype=float))
        pred = rf.predict(a_X, quantiles="mean")
        qs = np.linspace(0.0, 1.0, 201)
        qmat = np.asarray(rf.predict(a_X, quantiles=list(qs)))
        pred_v = sample_from_qrf(qmat, qs, a_X.shape[0],
                                 n_draw=200,
                                 rng=np.random.default_rng(seed)).var(axis=1, ddof=1)
        pred0 = 0.0
        pred0_v = 0.0
        if a_X0 is not None:
            pred0 = rf.predict(a_X0, quantiles="mean")
            qmat0 = np.asarray(rf.predict(a_X0, quantiles=list(qs)))
            pred0_v = sample_from_qrf(qmat0, qs, a_X0.shape[0],
                                      n_draw=200,
                                      rng=np.random.default_rng(seed)).var(axis=1, ddof=1)
        return {"mod": rf, "pred": pred, "pred0": pred0, "pred_v": pred_v,
                "pred0_v": pred0_v, "a_xname": a_xname,
                "add_learn": add_learn, "sd_method": sd_method}

    rf = RandomForestRegressor(
        n_estimators=500,
        max_features=int(a_par["mtry"]),
        min_samples_leaf=int(a_par["min_node_size"]),
        random_state=seed,
        n_jobs=-1,
    )
    rf.fit(a_X, np.asarray(resid, dtype=float))
    pred = rf.predict(a_X)
    pred0 = 0.0 if a_X0 is None else rf.predict(a_X0)

    if sd_method == "tree_var":
        # Per-point variance across the trees' individual predictions.
        tree_preds = np.stack([t.predict(a_X) for t in rf.estimators_], axis=0)
        pred_v = tree_preds.var(axis=0, ddof=1)
        pred0_v = 0.0
        if a_X0 is not None:
            tree_preds0 = np.stack([t.predict(a_X0) for t in rf.estimators_], axis=0)
            pred0_v = tree_preds0.var(axis=0, ddof=1)
    elif sd_method == "residual":
        # Back-compat: scalar Var(resid) broadcast to every point.
        v = float(np.var(np.asarray(resid), ddof=1))
        pred_v = np.full(a_X.shape[0], v)
        pred0_v = np.full(a_X0.shape[0], v) if a_X0 is not None else 0.0
    else:
        raise ValueError(
            f"sd_method must be 'residual', 'tree_var', or 'qrf', got {sd_method!r}"
        )

    return {"mod": rf, "pred": pred, "pred0": pred0, "pred_v": pred_v,
            "pred0_v": pred0_v, "a_xname": a_xname,
            "add_learn": add_learn, "sd_method": sd_method}
