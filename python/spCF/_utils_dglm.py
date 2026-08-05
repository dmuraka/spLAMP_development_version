"""Internal utilities for the coarse-to-fine *dynamic* (space-time) spatial GLMM.

Port of ``R/internal_utils_dglm.R`` and ``src/dglm_chunk.cpp``.

The model is a separable space-time decomposition on the link scale:
``g(mu_{i,t}) = x_{i,t}'beta + sum_k f_k(s_i, t) + offset``, where each scale-k
field ``f_k`` is a per-knot AR(1) Kalman smoother in time combined with kernel
kriging in space (generalized Product of Experts recombination).
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
from scipy.optimize import minimize, minimize_scalar

from ._kernels import dglm_scale_chunk
from ._neighbors import build_tree, frnn_csr_from_tree, knnx
from ._utils import _kmeans_centers, _unique_rows_with_inverse
from ._utils_glm import spcf_clip_l as dglm_clip_l


def dglm_kfun(d, band, kernel="exp"):
    d = np.asarray(d, dtype=float)
    if kernel == "gau":
        return np.exp(-(d / band) ** 2)
    return np.exp(-d / band)


def dglm_work(family, eta, y, offset=0.0):
    """Generic IRLS working response / weight for a glm() family object."""
    eta = dglm_clip_l(eta, family)
    mu = np.asarray(family.linkinv(eta), dtype=float)
    me = np.asarray(family.mu_eta(eta), dtype=float)
    v = np.maximum(np.asarray(family.variance(mu), dtype=float), 1e-8)
    y = np.asarray(y, dtype=float)
    offset = np.asarray(offset, dtype=float)
    z = (eta - offset) + (y - mu) / me
    w = np.maximum(me ** 2 / v, 1e-8)
    return {"mu": mu, "z": z, "w": w}


def dglm_panel(coords, time, vals=None, time_levels=None):
    """Build a (location x time) panel from long-format vectors."""
    coords = np.asarray(coords, dtype=float)
    C, lk = _unique_rows_with_inverse(coords)     # lk: 0-based location index
    nL = C.shape[0]
    time = np.asarray(time)
    if time_levels is None:
        time_levels = np.unique(time)
    else:
        time_levels = np.asarray(time_levels)
    T = time_levels.shape[0]
    tk = np.searchsorted(time_levels, time).astype(np.int64)   # 0-based time index
    M = None
    if vals is not None:
        M = np.full((nL, T), np.nan)
        M[lk, tk] = np.asarray(vals, dtype=float)
    return {"M": M, "C": C, "lk": lk.astype(np.int64), "tk": tk,
            "nL": nL, "T": T, "time_levels": time_levels}


def dglm_nbr(query, knots, band, kernel="exp"):
    """CSR neighbour lists (0-based knot indices) + kernel weights for one band."""
    query = np.asarray(query, dtype=float)
    knots = np.asarray(knots, dtype=float)
    rad = (math.sqrt(-math.log(1e-3)) * band if kernel == "gau"
           else -math.log(1e-3) * band)
    tree = build_tree(knots)
    idx, dist, offsets = frnn_csr_from_tree(tree, query, rad)
    w = dglm_kfun(dist, band, kernel)
    return {"ptr": offsets.astype(np.int64), "idx": idx.astype(np.int64),
            "w": np.asarray(w, dtype=float)}


def dglm_knots(coords_uni, band, seed=4321):
    """Knot coordinates for a bandwidth (n_knot = round(1.5*area/band^2), capped)."""
    coords_uni = np.asarray(coords_uni, dtype=float)
    area = (np.ptp(coords_uni[:, 0])) ** 2 + (np.ptp(coords_uni[:, 1])) ** 2
    n_uni = coords_uni.shape[0]
    n_knot = max(8, int(min(round(1.5 * area / band ** 2), n_uni)))
    if n_knot >= n_uni:
        return coords_uni
    if n_knot > 1000:
        rng = np.random.default_rng(seed)
        sel = np.sort(rng.choice(n_uni, size=n_knot, replace=False))
        return coords_uni[sel]
    iter_max = 5 if n_uni > 5000 else 10
    ck = _kmeans_centers(coords_uni, n_knot, iter_max, seed)
    nn = knnx(coords_uni, ck, k=1).ravel()
    return coords_uni[nn]


def _dglm_aggregate_dense(nb, W0, R0, K):
    """Kernel-weighted aggregation of a working-residual panel onto knots.

    Returns (Z, Rmat) each K x T, matching R ``.dglm_aggregate`` (den<1e-12 ->
    Z=0, Rmat=Inf). ``nb`` is the CSR neighbour structure (site -> knots).
    """
    nL, T = W0.shape
    ptr, idx, w = nb["ptr"], nb["idx"], nb["w"]
    den = np.zeros((K, T))
    Znum = np.zeros((K, T))
    Rnum = np.zeros((K, T))
    for i in range(nL):
        for nz in range(ptr[i], ptr[i + 1]):
            k = idx[nz]
            wik = w[nz]
            row_w = W0[i]
            den[k] += wik * row_w
            Znum[k] += wik * row_w * R0[i]
            Rnum[k] += (wik * wik) * row_w
    miss = ~np.isfinite(den) | (den < 1e-12)
    with np.errstate(divide="ignore", invalid="ignore"):
        Z = Znum / den
        Rmat = Rnum / den ** 2
    Z[miss] = 0.0
    Rmat[miss] = np.inf
    return Z, Rmat


def dglm_ar1_ml(Z, Rmat, rho0=0.7, Q0=1.0):
    """ML estimate of a global AR(1) (rho, Q) from knot-aggregated residuals."""
    Z = np.asarray(Z, dtype=float)
    Rmat = np.asarray(Rmat, dtype=float)
    K, T = Z.shape

    def nll(par):
        rho = math.tanh(par[0])
        Q = math.exp(par[1])
        a = np.zeros(K)
        P = np.full(K, Q / (1 - rho ** 2))
        ll = 0.0
        for t in range(T):
            ap = rho * a
            Pp = rho * rho * P + Q
            ob = np.isfinite(Rmat[:, t])
            S = Pp + Rmat[:, t]
            v = Z[:, t] - ap
            if np.any(ob):
                with np.errstate(divide="ignore", invalid="ignore"):
                    term = np.log(2 * np.pi * S) + v ** 2 / S
                ll -= 0.5 * np.sum(term[ob])
            with np.errstate(divide="ignore", invalid="ignore"):
                Kg = np.where(ob, Pp / S, 0.0)
            a = ap + Kg * v
            P = np.where(ob, (1 - Kg) * Pp, Pp)
        return 1e10 if not np.isfinite(ll) else -ll

    x0 = [math.atanh(rho0), math.log(Q0)]
    try:
        opt = minimize(nll, x0, method="Nelder-Mead",
                       options={"maxiter": 400, "xatol": 1e-5, "fatol": 1e-6})
        sol = opt.x
    except Exception:
        sol = x0
    rho = max(min(math.tanh(sol[0]), 0.999), -0.999)
    Q = math.exp(sol[1])
    return {"rho": rho, "Q": Q}


def dglm_scale_setup(Ctr, band, kernel, seed, Cpr=None):
    """Knot placement + (train + optional prediction) neighbourhoods for one band."""
    Ctr = np.asarray(Ctr, dtype=float)
    coords_uni, _ = _unique_rows_with_inverse(Ctr)
    knots = dglm_knots(coords_uni, band, seed)
    nb = dglm_nbr(Ctr, knots, band, kernel)
    if Cpr is not None:
        Cpr = np.asarray(Cpr, dtype=float)
        pb = dglm_nbr(Cpr, knots, band, kernel)
        n0 = Cpr.shape[0]
    else:
        pb = {"ptr": np.array([0, 0], dtype=np.int64),
              "idx": np.zeros(0, dtype=np.int64), "w": np.zeros(0)}
        n0 = 0
    return {"knots": knots, "K": knots.shape[0], "nb": nb, "pb": pb, "n0": n0}


def dglm_scale_apply(su, Rtr, Wtr, rho, Q, predict=True):
    """Residual-dependent part of one cascade scale (given a pre-built setup)."""
    W0 = np.nan_to_num(np.asarray(Wtr, dtype=float), nan=0.0)
    R0 = np.nan_to_num(np.asarray(Rtr, dtype=float), nan=0.0)
    use_pr = predict and su["n0"] > 0
    if use_pr:
        pb = su["pb"]
        n0 = su["n0"]
    else:
        pb = {"ptr": np.array([0, 0], dtype=np.int64),
              "idx": np.zeros(0, dtype=np.int64), "w": np.zeros(0)}
        n0 = 0
    nb = su["nb"]
    Ftr, Vtr, Fpr, Vpr = dglm_scale_chunk(
        nb["ptr"], nb["idx"], nb["w"],
        np.ascontiguousarray(W0), np.ascontiguousarray(R0),
        int(su["K"]), float(rho), float(Q),
        pb["ptr"], pb["idx"], pb["w"], int(n0),
    )
    out = {"Ftr": Ftr, "Vtr": Vtr, "peeled": np.asarray(Rtr, dtype=float) - Ftr,
           "knots": su["knots"]}
    if n0 > 0:
        out["Fpr"] = Fpr
        out["Vpr"] = Vpr
    return out


def dglm_scale(Ctr, Rtr, Wtr, band, rho, Q, kernel, seed, Cpr=None):
    su = dglm_scale_setup(Ctr, band, kernel, seed, Cpr)
    return dglm_scale_apply(su, Rtr, Wtr, rho, Q, predict=Cpr is not None)


def dglm_dynreg(r, Xtv, w, tk, T, q=None, P0=1e4, rg=1e-6):
    """Dynamic regression: time-varying coefficients via Kalman filter + RTS."""
    r = np.asarray(r, dtype=float)
    Xtv = np.asarray(Xtv, dtype=float)
    w = np.asarray(w, dtype=float)
    tk = np.asarray(tk, dtype=np.int64)
    d = Xtv.shape[1]
    Id = np.eye(d)
    bt = [None] * T
    Rt = [None] * T
    has = np.zeros(T, dtype=bool)
    for t in range(T):
        idx = np.where(tk == t)[0]
        if idx.size == 0:
            continue
        Xt = Xtv[idx]
        wt = w[idx]
        L = Xt.T @ (wt[:, None] * Xt) + rg * Id
        Ri = np.linalg.solve(L, Id)
        Rt[t] = Ri
        bt[t] = Ri @ (Xt.T @ (wt * r[idx]))
        has[t] = True

    def mkQ(qv):
        qv = np.broadcast_to(np.asarray(qv, dtype=float).ravel(), (d,)) if np.size(qv) == 1 \
            else np.asarray(qv, dtype=float).ravel()
        if qv.size != d:
            qv = np.resize(qv, d)
        return np.diag(qv)

    def run(qv, smooth=False):
        Qm = mkQ(qv)
        a_p = [None] * T
        P_p = [None] * T
        a_f = [None] * T
        P_f = [None] * T
        a = np.zeros(d)
        P = P0 * np.eye(d)
        ll = 0.0
        for t in range(T):
            ap = a
            Pp = P + Qm
            a_p[t] = ap
            P_p[t] = Pp
            if has[t]:
                S = Pp + Rt[t]
                Si = np.linalg.solve(S, Id)
                v = bt[t] - ap
                sign, logdet = np.linalg.slogdet(S)
                ll -= 0.5 * (logdet + float(v @ Si @ v))
                Kk = Pp @ Si
                a = ap + Kk @ v
                P = (Id - Kk) @ Pp
            else:
                a = ap
                P = Pp
            a_f[t] = a
            P_f[t] = P
        if not smooth:
            return ll
        a_s = list(a_f)
        P_s = list(P_f)
        for t in range(T - 2, -1, -1):
            G = P_f[t] @ np.linalg.solve(P_p[t + 1], Id)
            a_s[t] = a_f[t] + G @ (a_s[t + 1] - a_p[t + 1])
            P_s[t] = P_f[t] + G @ (P_s[t + 1] - P_p[t + 1]) @ G.T
        return a_s, P_s

    if q is None:
        try:
            op = minimize(lambda lq: -run(np.exp(lq)), np.full(d, math.log(1e-3)),
                          method="L-BFGS-B",
                          bounds=[(math.log(1e-8), math.log(1e3))] * d)
            q = np.exp(op.x) if np.isfinite(op.fun) else None
        except Exception:
            q = None
        if q is None:
            res = minimize_scalar(lambda lq: -run(np.full(d, math.exp(lq))),
                                  bounds=(math.log(1e-8), math.log(1e3)),
                                  method="bounded")
            q = np.full(d, math.exp(res.x))
    else:
        q = np.broadcast_to(np.asarray(q, dtype=float).ravel(), (d,)).copy() \
            if np.size(q) == 1 else np.resize(np.asarray(q, dtype=float).ravel(), d)

    a_s, P_s = run(q, smooth=True)
    beta = np.array([np.asarray(a_s[t]).ravel() for t in range(T)])   # T x d
    return {"beta": beta, "V": P_s, "q": q}
