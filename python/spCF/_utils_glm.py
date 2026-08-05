"""Internal utilities for the GLM (non-Gaussian) coarse-to-fine model.

Ports of `R/internal_utils_glm.R`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import statsmodels.api as sm

from ._kernels import lwr_chunk_glm
from ._neighbors import build_tree, frnn_csr_from_tree
from ._utils import _train_split, _unique_rows_with_inverse, _select_knots
from .families import Family


def spcf_clip_l(l, family: Optional[Family] = None, cap: Optional[float] = 20.0):
    """Soft-clip log-scale linear predictor to [-cap, cap]."""
    if family is not None and family.link == "identity":
        return l
    if cap is None or not np.isfinite(cap):
        return l
    return np.minimum(np.maximum(l, -cap), cap)


def link_fun(mu, family: Family):
    return family.linkfun(mu)


def inv_link_fun(eta, family: Family):
    return family.linkinv(eta)


def response_se(pred_lin, pred_lin_sd, family: Family):
    dmu_det = family.mu_eta(pred_lin)
    return np.abs(dmu_det) * pred_lin_sd


@dataclass
class InitialStateGLM:
    beta_int: np.ndarray
    x: np.ndarray
    id_train: np.ndarray
    offset: np.ndarray
    beta: np.ndarray
    beta_v: np.ndarray
    n: int
    nx: int
    pred: np.ndarray  # link scale
    resid: np.ndarray  # working residual: (y - mu)/mu_eta(eta)
    weights: np.ndarray
    x_sel: np.ndarray
    xname: list
    coords: np.ndarray
    glm_results: object = field(default=None)


def _fit_glm(y, x, family: Family, offset=None):
    fam = family.family
    glm = sm.GLM(y, x, family=fam, offset=offset)
    res = glm.fit()
    return res


def initial_fun_glm(
    x,
    y,
    coords,
    family: Family,
    offset=None,
    x_sel=None,
    train_rat: float = 0.75,
    id_train=None,
    seed: Optional[int] = None,
) -> InitialStateGLM:
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

    xname = ["Intercept"]
    if x is None:
        x_mat = np.ones((n, 1))
        x_sel = np.zeros(0, dtype=bool)
    else:
        x_arr = np.asarray(x, dtype=float)
        if x_arr.ndim == 1:
            x_arr = x_arr.reshape(-1, 1)
        if x_sel is None:
            sd = np.std(x_arr, axis=0, ddof=1)
            x_sel = (sd != 0)
        else:
            x_sel = np.asarray(x_sel, dtype=bool)
        if x_sel.sum() == 1:
            xname = ["Intercept", "x1"]
        elif x_sel.sum() > 1:
            try:
                import pandas as pd  # type: ignore
                if hasattr(x, "columns"):
                    cols = list(pd.DataFrame(x).columns)
                    xname = ["Intercept"] + [str(c) for i, c in enumerate(cols) if x_sel[i]]
                else:
                    xname = ["Intercept"] + [f"x{i+1}" for i, sel in enumerate(x_sel) if sel]
            except Exception:
                xname = ["Intercept"] + [f"x{i+1}" for i, sel in enumerate(x_sel) if sel]
        x_keep = x_arr[:, x_sel]
        x_mat = np.hstack([np.ones((n, 1)), x_keep])
    nx = x_mat.shape[1]

    if offset is None:
        offset = np.zeros(n)
    offset = np.asarray(offset, dtype=float).ravel()

    res = _fit_glm(y, x_mat, family, offset=offset)
    eta = res.predict(which="linear")
    mu = res.predict(which="mean")
    mu_eta = family.mu_eta(eta)
    # working residual matches R `glm$residuals`
    with np.errstate(divide="ignore", invalid="ignore"):
        working_resid = (y - mu) / np.where(mu_eta != 0, mu_eta, np.nan)
    working_resid = np.nan_to_num(working_resid, nan=0.0, posinf=0.0, neginf=0.0)

    weights = _glm_iwls_weights(res, family)

    beta_int = np.asarray(res.params, dtype=float).reshape(-1, 1)
    beta_int_vcov = np.asarray(res.cov_params(), dtype=float)
    beta = np.tile(beta_int.ravel(), (n, 1))
    beta_v = np.tile(np.diag(beta_int_vcov), (n, 1))

    return InitialStateGLM(
        beta_int=beta_int,
        x=x_mat,
        id_train=id_train,
        offset=offset,
        beta=beta,
        beta_v=beta_v,
        n=n,
        nx=nx,
        pred=eta,
        resid=working_resid,
        weights=weights,
        x_sel=np.asarray(x_sel, dtype=bool),
        xname=xname,
        coords=coords,
        glm_results=res,
    )


def _glm_iwls_weights(res, family: Family) -> np.ndarray:
    """Replicate R's `glm$weights` (final IRLS working weights)."""
    eta = np.asarray(res.predict(which="linear"), dtype=float)
    mu_eta = family.mu_eta(eta)
    var = np.maximum(family.variance(family.linkinv(eta)), 1e-300)
    return (mu_eta ** 2) / var


def _deviance_residuals(y, mu, family: Family):
    fam = family.family
    return fam.resid_dev(y, mu)


def lwr_glm(
    coords,
    coords_uni,
    resid,
    x,
    family: Family,
    band,
    b_old,
    vc,
    ridge,
    kernel,
    id_train,
    y,
    offset,
    weights=None,
    coords0=None,
    x0=None,
    sel_id=None,
    sse_hv0: Optional[float] = None,
    l_pred=None,
    func: str = "cf_glm",
    tree=None,
    tree0=None,
):
    coords = np.ascontiguousarray(coords, dtype=float)
    n = coords.shape[0]
    nx = x.shape[1]
    x = np.ascontiguousarray(x, dtype=float)
    if tree is None:
        tree = build_tree(coords)

    if weights is None:
        weights = np.ones(n)

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

    B_var = np.full((n_knot, nx), np.inf, dtype=float)
    if b_old is not None and ridge:
        for i in range(nx):
            B_var[:, i] = float(np.mean(b_old[:, i] ** 2))

    has0 = coords0 is not None
    n0 = 0 if not has0 else int(np.asarray(coords0).shape[0])

    id_train_flag = np.zeros(n, dtype=bool)
    id_train_flag[np.asarray(id_train, dtype=np.int64)] = True
    vc_int = np.atleast_1d(np.asarray(vc, dtype=np.int64)).ravel()

    b_all = np.zeros((n, nx))
    bv_inv_all = np.zeros((n, nx))
    pv_inv_all = np.zeros((n, nx))
    b_old_out = np.zeros((n_knot, nx))

    if has0:
        coords0 = np.ascontiguousarray(coords0, dtype=float)
        x0 = np.ascontiguousarray(x0, dtype=float)
        b_all0 = np.zeros((n0, nx))
        bv_inv_all0 = np.zeros((n0, nx))
        pv_inv_all0 = np.zeros((n0, nx))
        if tree0 is None:
            tree0 = build_tree(coords0)
    else:
        b_all0 = np.zeros((1, nx))
        bv_inv_all0 = np.zeros((1, nx))
        pv_inv_all0 = np.zeros((1, nx))
        x0 = np.zeros((1, nx))

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

        lwr_chunk_glm(
            id_flat, offsets, dist_flat,
            id0_flat, offsets0, dist0_flat,
            sel_chunk,
            id_train_flag,
            np.ascontiguousarray(resid, dtype=float),
            np.ascontiguousarray(weights, dtype=float),
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

    not_vc_mask = np.ones(nx, dtype=bool)
    not_vc_mask[vc_int] = False

    bv_all = bv_inv_all.copy()
    with np.errstate(divide="ignore", invalid="ignore"):
        b_all[:, vc_int] = np.where(
            pv_inv_all[:, vc_int] != 0,
            b_all[:, vc_int] / pv_inv_all[:, vc_int],
            0.0,
        )
    b_all[:, not_vc_mask] = 0.0
    b_all = np.nan_to_num(b_all, nan=0.0)
    pred = np.sum(x * b_all, axis=1)

    run = False
    sse_hv = float("nan")
    if func in ("cf_glm_hv", "cf_lm_hv"):
        l_pred_now = (l_pred if l_pred is not None else 0.0) + pred
        l_pred_off = spcf_clip_l(l_pred_now, family) + offset
        # Refit GLM with new offset
        res = _fit_glm(y, x, family, offset=l_pred_off)
        mu = res.predict(which="mean")
        d = _deviance_residuals(y, mu, family)
        not_train = np.ones(n, dtype=bool)
        not_train[np.asarray(id_train, dtype=np.int64)] = False
        sse_hv_now = float(np.sum(d[not_train] ** 2))
        run = sse_hv_now < (sse_hv0 if sse_hv0 is not None else float("inf"))
        sse_hv = sse_hv_now if run else sse_hv0
    else:
        run = True

    if not run:
        return {"run": False, "sse_hv": sse_hv}

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
    bv_all[np.isnan(bv_inv_all)] = np.inf

    # Predictive variance per eq.(10) on the link scale: 1 / sum_k (w_k / pv_k).
    with np.errstate(divide="ignore", invalid="ignore"):
        pv_all = np.where(pv_inv_all != 0, 1.0 / pv_inv_all, np.inf)
    pv_all[:, not_vc_mask] = np.nan

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
        with np.errstate(divide="ignore", invalid="ignore"):
            pv_all0 = np.where(pv_inv_all0 != 0, 1.0 / pv_inv_all0, np.inf)
        pv_all0[:, not_vc_mask] = np.nan
        pred0 = np.sum(x0 * b_all0, axis=1)
    else:
        b_all0 = bv_all0 = pv_all0 = pred0 = None

    return {
        "beta": b_all,
        "beta_v": bv_all,
        "beta_pv": pv_all,
        "pred": pred,
        "sel_id": sel_id_out,
        "coords_cent": coords_cent,
        "beta0": b_all0,
        "beta0_v": bv_all0,
        "beta0_pv": pv_all0,
        "pred0": pred0,
        "b_old": b_old_out,
        "run": True,
        "sse_hv": sse_hv,
        "sse_hv0": sse_hv0,
        "vc_sel": vc_int,
    }
