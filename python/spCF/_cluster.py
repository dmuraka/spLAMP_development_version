"""Spatial-block cluster-robust covariance for CF spatial LM/GLMM coefficients.

Port of ``.spcf_clusterSE`` (in ``R/internal_utils_glm.R``). The model-based
covariance treats the fitted spatial field as a known offset and so understates
``Var(beta-hat)`` (the residual is a spatially correlated random field). Here
the field is put back into the working residual and a cluster-robust sandwich
is taken over spatial blocks (a per-axis grid sized so each block exceeds the
field's correlation length, proxied by the median committed bandwidth).

Reduces to the field-in-error OLS sandwich for gaussian/identity.
"""
from __future__ import annotations

import numpy as np


class GaussianIdentity:
    """Minimal family shim (identity-link Gaussian) for the LM cluster SE.

    Avoids a hard statsmodels dependency on the LM path.
    """

    link = "identity"
    family_name = "gaussian"

    @staticmethod
    def linkinv(eta):
        return np.asarray(eta, dtype=float)

    @staticmethod
    def mu_eta(eta):
        return np.ones_like(np.asarray(eta, dtype=float))

    @staticmethod
    def variance(mu):
        return np.ones_like(np.asarray(mu, dtype=float))


def _clip_l(l, family, cap: float = 20.0):
    link = getattr(family, "link", None)
    if link == "identity":
        return l
    if cap is None or not np.isfinite(cap):
        return l
    return np.minimum(np.maximum(l, -cap), cap)


def _fam_name(family) -> str:
    return (getattr(family, "family_name", None)
            or getattr(family, "name", None) or "")


def spcf_cluster_se(y, X, beta, field, offset, family, coords, bands,
                    c_guard: float = 1.0):
    """Return ``(V, G)``: p x p cluster-robust covariance and block count.

    ``family`` must expose ``link`` (str), ``linkinv``, ``mu_eta`` and
    ``variance``.  ``offset`` may be ``None``.
    """
    X = np.asarray(X, dtype=float)
    beta = np.asarray(beta, dtype=float).ravel()
    field = np.asarray(field, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    coords = np.asarray(coords, dtype=float)
    off = 0.0 if offset is None else np.asarray(offset, dtype=float).ravel()

    eta = _clip_l(X @ beta + field + off, family)
    mu = np.asarray(family.linkinv(eta), dtype=float)
    mup = np.asarray(family.mu_eta(eta), dtype=float)
    v = np.maximum(np.asarray(family.variance(mu), dtype=float), 1e-8)
    W = np.maximum(mup ** 2 / v, 1e-8)
    with np.errstate(divide="ignore", invalid="ignore"):
        e = field + (y - mu) / mup                 # working residual WITH the field
    e = np.nan_to_num(e, nan=0.0, posinf=0.0, neginf=0.0)

    # Correlation-length proxy: median committed bandwidth.
    bands_arr = np.asarray(bands, dtype=float)
    rng = float(np.nanquantile(bands_arr, 0.5)) if bands_arr.size else float("nan")
    span = np.array([np.ptp(coords[:, 0]), np.ptp(coords[:, 1])])
    if not np.isfinite(rng) or rng <= 0:
        rng = float(np.mean(span)) / 8.0
    Gxy = np.clip(np.floor(span / (c_guard * rng)).astype(int), 2, 8)

    qx = np.quantile(coords[:, 0], np.linspace(0, 1, Gxy[0] + 1))
    qy = np.quantile(coords[:, 1], np.linspace(0, 1, Gxy[1] + 1))
    bx = _bin_index(coords[:, 0], np.unique(qx))
    by = _bin_index(coords[:, 1], np.unique(qy))
    nby = int(by.max()) + 1
    blk_raw = bx * nby + by
    # Relabel present blocks to 0..G-1 (interaction(..., drop = TRUE)).
    _, blk = np.unique(blk_raw, return_inverse=True)
    G = int(blk.max()) + 1

    XtWX = X.T @ (W[:, None] * X)
    XtWXi = np.linalg.solve(XtWX, np.eye(X.shape[1]))
    We = W * e
    p = X.shape[1]
    S = np.zeros((G, p))
    XWe = X * We[:, None]
    for j in range(p):
        S[:, j] = np.bincount(blk, weights=XWe[:, j], minlength=G)
    V = (G / (G - 1)) * XtWXi @ (S.T @ S) @ XtWXi
    return V, G


def _spatial_blocks(coords, bands, c_guard: float = 1.0):
    """Return ``(blk, G, rng)``: 0-based block labels, block count, and the
    correlation-length proxy ``rng`` (median committed bandwidth). Matches the
    per-axis quantile grid used by :func:`spcf_cluster_se`.
    """
    coords = np.asarray(coords, dtype=float)
    bands_arr = np.asarray(bands, dtype=float)
    rng = float(np.nanquantile(bands_arr, 0.5)) if bands_arr.size else float("nan")
    span = np.array([np.ptp(coords[:, 0]), np.ptp(coords[:, 1])])
    if not np.isfinite(rng) or rng <= 0:
        rng = float(np.mean(span)) / 8.0
    Gxy = np.clip(np.floor(span / (c_guard * rng)).astype(int), 2, 8)
    qx = np.quantile(coords[:, 0], np.linspace(0, 1, Gxy[0] + 1))
    qy = np.quantile(coords[:, 1], np.linspace(0, 1, Gxy[1] + 1))
    bx = _bin_index(coords[:, 0], np.unique(qx))
    by = _bin_index(coords[:, 1], np.unique(qy))
    nby = int(by.max()) + 1
    _, blk = np.unique(bx * nby + by, return_inverse=True)
    G = int(blk.max()) + 1
    return blk, G, rng


def levloo_meat(X, W, r, coords, bands, blk, Ai):
    """Leverage-LOO score-variance meat (port of ``.spcf_levloo_meat``).

    Estimates the TOTAL score variance (noise + field error) at full ``n`` via
    an approximate leverage-corrected LOO residual, used only as an upper cap in
    :func:`optfield_SE`.
    """
    X = np.asarray(X, dtype=float)
    coords = np.asarray(coords, dtype=float)
    W = np.asarray(W, dtype=float).ravel()
    r = np.asarray(r, dtype=float).ravel()
    npp = X.shape[0]
    bands = np.asarray(bands, dtype=float)
    bands = bands[np.isfinite(bands) & (bands > 0)]
    hX = W * ((X @ Ai) * X).sum(axis=1)                 # fixed-effect leverage
    hfield = np.zeros(npp)
    if bands.size:
        from sklearn.neighbors import NearestNeighbors
        k = int(min(npp - 1, 200))
        if k >= 1:
            nn = NearestNeighbors(n_neighbors=k + 1).fit(coords)
            kn = nn.kneighbors(coords, return_distance=True)[0][:, 1:]  # drop self
            loglev = np.zeros(npp)
            for hr in bands:
                s_ir = 1.0 + np.exp(-kn / hr).sum(axis=1)
                loglev += np.log1p(-np.minimum(1.0 / s_ir, 0.999))
            hfield = 1.0 - np.exp(loglev)
    h_ii = np.minimum(np.maximum(hX, 0.0) + hfield, 0.99)
    r_loo = r / (1.0 - h_ii)
    G = int(blk.max()) + 1
    p = X.shape[1]
    S = np.zeros((G, p))
    np.add.at(S, blk, X * (W * r_loo)[:, None])
    return (G / (G - 1)) * (S.T @ S)


def optfield_SE(y, X, beta, field, s_f, offset, family, coords, bands,
                c_guard: float = 1.0):
    """opt+field cluster-robust coefficient covariance (default ``se_method``).

    Port of ``.spcf_optfield_SE`` (``R/internal_utils_glm.R``); also covers the
    dynamic GLMM case (``.dglm_optfield_SE``) when the caller folds the
    time-varying part into ``field`` (``field = f_obs + tvpart``). The clustered
    meat is split into a field-removed observation-noise piece and a calibrated
    field-uncertainty piece with an explicit within-block ``exp(-d/rng)``
    correlation, then capped from above by the leverage-LOO ceiling. Returns
    ``(V, G)``.
    """
    X = np.asarray(X, dtype=float)
    beta = np.asarray(beta, dtype=float).ravel()
    field = np.asarray(field, dtype=float).ravel()
    s_f = np.asarray(s_f, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    coords = np.asarray(coords, dtype=float)
    off = 0.0 if offset is None else np.asarray(offset, dtype=float).ravel()

    eta = _clip_l(X @ beta + field + off, family)
    mu = np.asarray(family.linkinv(eta), dtype=float)
    mup = np.asarray(family.mu_eta(eta), dtype=float)
    v = np.maximum(np.asarray(family.variance(mu), dtype=float), 1e-8)
    W = np.maximum(mup ** 2 / v, 1e-8)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = (y - mu) / np.where(np.abs(mup) < 1e-8, 1e-8, mup)   # field-removed
    r = np.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0)

    blk, G, rng = _spatial_blocks(coords, bands, c_guard)
    p = X.shape[1]
    XtWX = X.T @ (W[:, None] * X)
    Ai = np.linalg.solve(XtWX, np.eye(p))

    # (i) observation-noise meat
    S = np.zeros((G, p))
    np.add.at(S, blk, X * (W * r)[:, None])
    Bnoise = (G / (G - 1)) * (S.T @ S)

    # (ii) field meat with within-block exp(-d/rng) correlation
    from scipy.spatial.distance import cdist
    U = X * (W * s_f)[:, None]
    Bfield = np.zeros((p, p))
    for g in range(G):
        ix = np.where(blk == g)[0]
        if ix.size == 1:
            Bfield += np.outer(U[ix[0]], U[ix[0]])
            continue
        Dg = cdist(coords[ix], coords[ix])
        Rg = np.exp(-Dg / rng)
        Ug = U[ix]
        Bfield += Ug.T @ (Rg @ Ug)

    Vof = Ai @ (Bnoise + Bfield) @ Ai

    # leverage-LOO ceiling (correlation-preserving diagonal rescale)
    try:
        Bloo = levloo_meat(X, W, r, coords, bands, blk, Ai)
    except Exception:
        Bloo = None
    if Bloo is not None:
        Ve = Ai @ Bnoise @ Ai
        Vloo = Ai @ Bloo @ Ai
        d = np.maximum(np.diag(Ve), np.minimum(np.diag(Vof), np.diag(Vloo)))
        sof = np.sqrt(np.maximum(np.diag(Vof), np.finfo(float).eps))
        Rc = Vof / np.outer(sof, sof)
        sdn = np.sqrt(np.maximum(d, 0.0))
        Vof = Rc * np.outer(sdn, sdn)
    return Vof, G


def _bin_index(z: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Assign each value to a bin defined by ``edges``, matching R's
    ``cut(..., right = TRUE, include.lowest = TRUE)``: interior intervals are
    right-closed ``(b_{k-1}, b_k]`` and the first interval also includes its
    left endpoint. ``side="left"`` reproduces the right-closed rule (a point
    exactly on an interior edge b_k lands in the left block), and the clip to 0
    handles the include-lowest left endpoint.
    """
    z = np.asarray(z, dtype=float)
    edges = np.asarray(edges, dtype=float)
    nbin = edges.shape[0] - 1
    idx = np.searchsorted(edges, z, side="left") - 1
    idx = np.clip(idx, 0, nbin - 1)
    return idx.astype(np.int64)
