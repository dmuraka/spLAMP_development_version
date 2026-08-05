"""Core LWR chunk kernels.

Ports of `src/lwr_chunk.cpp` and `src/lwr_chunk_glm.cpp` from the R package.
Uses numba when available; otherwise falls back to pure-Python loops.
"""
from __future__ import annotations

import numpy as np

try:
    from numba import njit, prange  # type: ignore
    _HAS_NUMBA = True
except Exception:  # pragma: no cover - exercised only without numba
    _HAS_NUMBA = False

    def njit(*args, **kwargs):  # type: ignore
        def deco(f):
            return f
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return deco

    def prange(n):  # type: ignore
        return range(n)


# kernel_id: 1 = exp, 2 = gau
@njit(cache=True, fastmath=True)
def _kfun_scalar(d, band, kernel_id):
    if kernel_id == 2:
        return np.exp(-(d * d) / (band * band))
    return np.exp(-d / band)


@njit(cache=True, fastmath=True)
def lwr_chunk(
    nb_id_flat,        # int64[:]   concatenated 0-based neighbor indices
    nb_id_offsets,     # int64[:]   length chunk_sz+1 (CSR-style offsets)
    nb_dist_flat,      # float64[:] distances aligned with nb_id_flat
    nb_id0_flat,       # int64[:] or empty
    nb_id0_offsets,    # int64[:] or length-1 empty
    nb_dist0_flat,     # float64[:]
    sel_chunk,         # int64[:]   0-based row indices into b_old
    id_train_flag,     # bool[:]    length n
    resid,             # float64[:] length n
    x,                 # float64[:, :]  n x nx
    has0,              # bool
    x0,                # float64[:, :]  n0 x nx (or zeros if has0=False)
    B_var,             # float64[:, :]  n_knot x nx
    vc_cols,           # int64[:]   0-based columns to process
    band,              # float64
    kernel_id,         # int64
    b_all,             # float64[:, :]  n x nx, accumulated in place
    bv_inv_all,        # float64[:, :]  n x nx
    pv_inv_all,        # float64[:, :]  n x nx
    b_all0,            # float64[:, :]  n0 x nx
    bv_inv_all0,       # float64[:, :]
    pv_inv_all0,       # float64[:, :]
    b_old,             # float64[:, :]  n_knot x nx, filled in place
):
    """Direct port of lwr_chunk_cpp."""
    nx = x.shape[1]
    chunk_sz = sel_chunk.shape[0]
    n_vc = vc_cols.shape[0]

    for k in range(chunk_sz):
        sel = sel_chunk[k]
        i_start = nb_id_offsets[k]
        i_end = nb_id_offsets[k + 1]
        m = i_end - i_start
        if m == 0:
            continue

        m_hv = 0
        wxy_sel_csum = 0.0
        wxxw_sel_csum = 0.0
        wei = np.empty(m)

        for i in range(m):
            d = nb_dist_flat[i_start + i]
            if kernel_id == 2:
                w = np.exp(-(d * d) / (band * band))
            else:
                w = np.exp(-d / band)
            wei[i] = w
            sidx = nb_id_flat[i_start + i]
            if id_train_flag[sidx]:
                ww = w * w
                wxy_sel_csum += ww * resid[sidx]
                wxxw_sel_csum += ww
                m_hv += 1

        if m_hv <= 5:
            continue
        if wxxw_sel_csum <= 0.0:
            continue

        b_sel0 = wxy_sel_csum / wxxw_sel_csum
        for j in range(nx):
            b_old[sel, j] = b_sel0

        # coords0 neighbors
        m0 = 0
        i0_start = 0
        i0_end = 0
        if has0:
            i0_start = nb_id0_offsets[k]
            i0_end = nb_id0_offsets[k + 1]
            m0 = i0_end - i0_start
        wei0 = np.empty(m0) if m0 > 0 else np.empty(0)
        if has0 and m0 > 0:
            for i in range(m0):
                d0 = nb_dist0_flat[i0_start + i]
                if kernel_id == 2:
                    wei0[i] = np.exp(-(d0 * d0) / (band * band))
                else:
                    wei0[i] = np.exp(-d0 / band)

        for vi in range(n_vc):
            j = vc_cols[vi]

            sigma = 0.0
            for i in range(m):
                sidx = nb_id_flat[i_start + i]
                rs = resid[sidx] - x[sidx, j] * b_sel0
                v = wei[i] * rs
                sigma += v * v
            sigma /= (m - 1)

            B_var_sj = B_var[sel, j]
            lam = sigma / B_var_sj  # Inf -> 0
            wxxw_lam = wxxw_sel_csum + lam
            b_sel_val = wxy_sel_csum / wxxw_lam
            bv_sel = sigma / wxxw_lam
            inv_bv_sel = 1.0 / bv_sel
            inv_wxxw = 1.0 / wxxw_sel_csum

            for i in range(m):
                sidx = nb_id_flat[i_start + i]
                w = wei[i]
                ws = w * w
                xv = x[sidx, j]
                pv_sel = (xv * xv * inv_wxxw) * sigma + sigma / w
                wei2_pv = ws / pv_sel
                b_all[sidx, j] += wei2_pv * b_sel_val
                bv_inv_all[sidx, j] += wei2_pv * inv_bv_sel
                pv_inv_all[sidx, j] += wei2_pv

            if has0 and m0 > 0:
                for i in range(m0):
                    sidx0 = nb_id0_flat[i0_start + i]
                    w0 = wei0[i]
                    w0s = w0 * w0
                    xv0 = x0[sidx0, j]
                    pv_sel0 = (xv0 * xv0 * inv_wxxw) * sigma + sigma / w0
                    wei2_pv0 = w0s / pv_sel0
                    b_all0[sidx0, j] += wei2_pv0 * b_sel_val
                    bv_inv_all0[sidx0, j] += wei2_pv0 * inv_bv_sel
                    pv_inv_all0[sidx0, j] += wei2_pv0


@njit(cache=True, fastmath=True)
def lwr_chunk_glm(
    nb_id_flat,
    nb_id_offsets,
    nb_dist_flat,
    nb_id0_flat,
    nb_id0_offsets,
    nb_dist0_flat,
    sel_chunk,
    id_train_flag,
    resid,
    w_obs,             # float64[:]  IRLS weights, length n
    x,
    has0,
    x0,
    B_var,
    vc_cols,
    band,
    kernel_id,
    b_all,
    bv_inv_all,
    pv_inv_all,
    b_all0,
    bv_inv_all0,
    pv_inv_all0,
    b_old,
):
    """Direct port of lwr_chunk_glm_cpp.

    Differences from lwr_chunk:
    * sigma over TRAIN neighbors only, weighted by w_obs.
    * Sums-of-squares accumulators are weighted by w_obs at training points.
    """
    nx = x.shape[1]
    chunk_sz = sel_chunk.shape[0]
    n_vc = vc_cols.shape[0]

    for k in range(chunk_sz):
        sel = sel_chunk[k]
        i_start = nb_id_offsets[k]
        i_end = nb_id_offsets[k + 1]
        m = i_end - i_start
        if m == 0:
            continue

        m_hv = 0
        wxy_sel_csum = 0.0
        wxxw_sel_csum = 0.0
        wei = np.empty(m)

        for i in range(m):
            d = nb_dist_flat[i_start + i]
            if kernel_id == 2:
                w_ker = np.exp(-(d * d) / (band * band))
            else:
                w_ker = np.exp(-d / band)
            wei[i] = w_ker
            sidx = nb_id_flat[i_start + i]
            if id_train_flag[sidx]:
                ww = w_ker * w_ker
                w_o = w_obs[sidx]
                wxy_sel_csum += ww * w_o * resid[sidx]
                wxxw_sel_csum += ww * w_o
                m_hv += 1

        if m_hv <= 5:
            continue
        if wxxw_sel_csum <= 0.0:
            continue

        b_sel0 = wxy_sel_csum / wxxw_sel_csum
        for j in range(nx):
            b_old[sel, j] = b_sel0

        m0 = 0
        i0_start = 0
        i0_end = 0
        if has0:
            i0_start = nb_id0_offsets[k]
            i0_end = nb_id0_offsets[k + 1]
            m0 = i0_end - i0_start
        wei0 = np.empty(m0) if m0 > 0 else np.empty(0)
        if has0 and m0 > 0:
            for i in range(m0):
                d0 = nb_dist0_flat[i0_start + i]
                if kernel_id == 2:
                    wei0[i] = np.exp(-(d0 * d0) / (band * band))
                else:
                    wei0[i] = np.exp(-d0 / band)

        for vi in range(n_vc):
            j = vc_cols[vi]

            # sigma over TRAIN neighbors only, IRLS-weighted
            sigma = 0.0
            for i in range(m):
                sidx = nb_id_flat[i_start + i]
                if not id_train_flag[sidx]:
                    continue
                rs = resid[sidx] - x[sidx, j] * b_sel0
                v = wei[i] * rs
                sigma += w_obs[sidx] * v * v
            if m_hv <= 1:
                continue
            sigma /= (m_hv - 1)

            B_var_sj = B_var[sel, j]
            lam = sigma / B_var_sj
            wxxw_lam = wxxw_sel_csum + lam
            b_sel_val = wxy_sel_csum / wxxw_lam
            bv_sel = sigma / wxxw_lam
            inv_bv_sel = 1.0 / bv_sel
            inv_wxxw = 1.0 / wxxw_sel_csum

            for i in range(m):
                sidx = nb_id_flat[i_start + i]
                w_ker = wei[i]
                ws = w_ker * w_ker
                xv = x[sidx, j]
                pv_sel = (xv * xv * inv_wxxw) * sigma + sigma / w_ker
                wei2_pv = ws / pv_sel
                b_all[sidx, j] += wei2_pv * b_sel_val
                bv_inv_all[sidx, j] += wei2_pv * inv_bv_sel
                pv_inv_all[sidx, j] += wei2_pv

            if has0 and m0 > 0:
                for i in range(m0):
                    sidx0 = nb_id0_flat[i0_start + i]
                    w0 = wei0[i]
                    w0s = w0 * w0
                    xv0 = x0[sidx0, j]
                    pv_sel0 = (xv0 * xv0 * inv_wxxw) * sigma + sigma / w0
                    wei2_pv0 = w0s / pv_sel0
                    b_all0[sidx0, j] += wei2_pv0 * b_sel_val
                    bv_inv_all0[sidx0, j] += wei2_pv0 * inv_bv_sel
                    pv_inv_all0[sidx0, j] += wei2_pv0


@njit(cache=True, fastmath=True)
def lwr_ds_chunk(
    nb_id_flat,        # int64[:]   concatenated 0-based neighbor indices
    nb_id_offsets,     # int64[:]   length chunk_sz+1 (CSR offsets)
    nb_dist_flat,      # float64[:] distances aligned with nb_id_flat
    sel_chunk,         # int64[:]   0-based knot rows into b_old
    local_bands,       # float64[:] band per knot in chunk
    kernel_id,         # int64      1 = exp, 2 = gau
    vc0,               # int64      0-based x column of the spatial process
    resid_area,        # float64[:] length N_area
    x_area,            # float64[:] length N_area  (= X[:, vc0])
    w_area,            # float64[:] length N_area  (area weight)
    a,                 # float64[:] length n       (point weight)
    agg_id0,           # int64[:]   length n       (0-based area id per point)
    id_train_flag,     # bool[:]    length n
    x,                 # float64[:, :] n x nx
    b_var_col,         # float64[:] length n_knot   (may be inf = no ridge)
    c_shrink,          # float64
    b_all,             # float64[:, :] n x nx, accumulated in place
    bv_inv_all,        # float64[:, :] n x nx
    pv_inv_all,        # float64[:, :] n x nx
    b_old,             # float64[:, :] n_knot x nx, filled in place
):
    """Direct port of lwr_ds_chunk_cpp (downscaling variant).

    Areal-observation local WLS: residuals ``resid_area`` are per area, each
    point carries a weight ``a`` and area id ``agg_id0``. Per knot, training
    points in the kernel window are aggregated to areas with
    ``V_i = sum_{j in i} a_j^2 / w_j`` (strict aggregated variance), a WLS
    slope ``b_sel`` is fit, and the predictive quantities are scattered back
    onto every neighbor point (train + test).
    """
    eps = np.finfo(np.float64).eps
    chunk_sz = sel_chunk.shape[0]

    for k in range(chunk_sz):
        sel_knot = sel_chunk[k]
        band_k = local_bands[k]
        i_start = nb_id_offsets[k]
        i_end = nb_id_offsets[k + 1]
        m = i_end - i_start
        if m < 3:
            continue

        wei = np.empty(m)
        # Per-knot areal accumulators, built with an append-if-new scan.
        u_area = np.empty(m, dtype=np.int64)   # unique area index (0-based)
        u_ax = np.zeros(m)                      # sum a*x over training pts
        u_a2w = np.zeros(m)                     # sum a^2 / w  (V_i)
        nu = 0
        m_hv = 0
        sum_wei_hv = 0.0
        sum_wei_hv_sq = 0.0

        for i in range(m):
            d = nb_dist_flat[i_start + i]
            if kernel_id == 2:
                w = np.exp(-(d * d) / (band_k * band_k))
            else:
                w = np.exp(-d / band_k)
            wei[i] = w
            sidx = nb_id_flat[i_start + i]
            if id_train_flag[sidx]:
                m_hv += 1
                sum_wei_hv += w
                sum_wei_hv_sq += w * w
                aid = agg_id0[sidx]
                aj = a[sidx]
                ax = aj * x[sidx, vc0]
                a2w = (aj * aj) / (w if w > eps else eps)
                # find aid in u_area[0:nu]
                pos = -1
                for t in range(nu):
                    if u_area[t] == aid:
                        pos = t
                        break
                if pos < 0:
                    u_area[nu] = aid
                    u_ax[nu] = ax
                    u_a2w[nu] = a2w
                    nu += 1
                else:
                    u_ax[pos] += ax
                    u_a2w[pos] += a2w

        if m_hv <= 5:
            continue
        n_train_areas = nu

        # ---- WLS at area level
        wxy_sel_csum = 0.0
        wxxw_sel_csum = 0.0
        for t in range(nu):
            V_i = u_a2w[t]
            if V_i <= 0.0:
                continue
            aidx = u_area[t]
            x_sel_area = u_ax[t]
            weight_i = w_area[aidx] / V_i
            wxy_sel_csum += weight_i * x_sel_area * resid_area[aidx]
            wxxw_sel_csum += weight_i * x_sel_area * x_sel_area
        if wxxw_sel_csum <= 0.0:
            continue
        b_sel0 = wxy_sel_csum / wxxw_sel_csum
        b_old[sel_knot, vc0] = b_sel0

        # ---- sigma^2
        sigma_sum = 0.0
        for t in range(nu):
            V_i = u_a2w[t]
            if V_i <= 0.0:
                continue
            aidx = u_area[t]
            r_sub = resid_area[aidx] - x_area[aidx] * b_sel0
            sigma_sum += (w_area[aidx] / V_i) * r_sub * r_sub
        denom = n_train_areas - 1
        if denom < 1:
            denom = 1
        sigma = sigma_sum / denom
        if sigma <= 0.0:
            sigma = eps

        # ---- ridge via b_var (may be inf for no ridge)
        B_var_s = b_var_col[sel_knot]
        if np.isinf(B_var_s) or B_var_s <= 0.0:
            lam = 0.0
        else:
            lam = sigma / B_var_s
        wxxw_lambda = wxxw_sel_csum + lam
        b_sel = wxy_sel_csum / wxxw_lambda
        bv_sel = sigma / wxxw_lambda

        # ---- ESS-based shrinkage (c_shrink > 0)
        if c_shrink > 0.0:
            n_eff = (sum_wei_hv * sum_wei_hv) / (sum_wei_hv_sq if sum_wei_hv_sq > eps else eps)
            shrink = n_eff / (n_eff + c_shrink)
            b_sel *= shrink
            bv_sel *= shrink * shrink
        if bv_sel <= 0.0:
            bv_sel = eps

        # ---- accumulate onto every neighbor point
        inv_wxxw = 1.0 / wxxw_sel_csum
        inv_bv_sel = 1.0 / bv_sel
        for i in range(m):
            sidx = nb_id_flat[i_start + i]
            w = wei[i]
            ws = w * w
            xv = x[sidx, vc0]
            pv_sel = (xv * xv * inv_wxxw) * sigma + sigma / (w if w > eps else eps)
            if pv_sel <= 0.0:
                continue
            wei2_pv_sel = ws / pv_sel
            b_all[sidx, vc0] += wei2_pv_sel * b_sel
            bv_inv_all[sidx, vc0] += wei2_pv_sel * inv_bv_sel
            pv_inv_all[sidx, vc0] += wei2_pv_sel


@njit(cache=True, fastmath=True)
def _dglm_gpoe(P_ptr, P_idx, P_w, n, T, invP, mP):
    """Generalized Product-of-Experts recombination (port of the gpoe lambda)."""
    F = np.zeros((n, T))
    V = np.zeros((n, T))
    gden = np.zeros((n, T))
    gnum = np.zeros((n, T))
    sumw = np.zeros(n)
    for i in range(n):
        sw = 0.0
        for nz in range(P_ptr[i], P_ptr[i + 1]):
            k = P_idx[nz]
            wik = P_w[nz]
            sw += wik
            for t in range(T):
                gden[i, t] += wik * invP[k, t]
                gnum[i, t] += wik * mP[k, t]
        sumw[i] = sw
    vmx = 1.0
    for i in range(n):
        for t in range(T):
            g = gden[i, t]
            if g > 0.0:
                v = sumw[i] / g
                if v > vmx:
                    vmx = v
    for i in range(n):
        sw = sumw[i]
        for t in range(T):
            g = gden[i, t]
            if g > 0.0:
                F[i, t] = gnum[i, t] / g
                V[i, t] = sw / g
            else:
                F[i, t] = 0.0
                V[i, t] = vmx
    return F, V


@njit(cache=True, fastmath=True)
def dglm_scale_chunk(ptr, idx, w, W0, R0, K, rho, Q, pptr, pidx, pw, n0):
    """Fused per-scale operator for CF-DGLMM (port of ``dglm_scale_chunk_cpp``).

    Neighbour-limited kernel aggregation to knots -> per-knot AR(1) Kalman
    filter + RTS smoother -> gPoE recombination at training (and prediction)
    sites. ``W0``/``R0`` are ``nL x T`` working-weight / working-residual
    panels (0 where unobserved). Returns ``(Ftr, Vtr, Fpr, Vpr)`` (nL x T and
    n0 x T; ``Fpr``/``Vpr`` empty when ``n0 == 0``).
    """
    nL = W0.shape[0]
    T = W0.shape[1]
    eps = 1e-12

    # 1. aggregate working residual to knots (K x T)
    den = np.zeros((K, T))
    Znum = np.zeros((K, T))
    Rnum = np.zeros((K, T))
    for i in range(nL):
        for nz in range(ptr[i], ptr[i + 1]):
            k = idx[nz]
            wik = w[nz]
            wik2 = wik * wik
            for t in range(T):
                w0 = W0[i, t]
                if w0 == 0.0:
                    continue
                wW = wik * w0
                den[k, t] += wW
                Znum[k, t] += wW * R0[i, t]
                Rnum[k, t] += wik2 * w0

    # 2. per-knot AR(1) Kalman filter + RTS smoother -> m, P (K x T)
    m = np.zeros((K, T))
    P = np.zeros((K, T))
    ap = np.zeros(T)
    Pp = np.zeros(T)
    af = np.zeros(T)
    Pf = np.zeros(T)
    P0 = Q / (1.0 - rho * rho)
    for k in range(K):
        a = 0.0
        p = P0
        for t in range(T):
            a_pred = rho * a
            p_pred = rho * rho * p + Q
            ap[t] = a_pred
            Pp[t] = p_pred
            d = den[k, t]
            if d > eps:
                Zkt = Znum[k, t] / d
                Rkt = Rnum[k, t] / (d * d)
                Kg = p_pred / (p_pred + Rkt)
                a = a_pred + Kg * (Zkt - a_pred)
                p = (1.0 - Kg) * p_pred
            else:
                a = a_pred
                p = p_pred
            af[t] = a
            Pf[t] = p
        ms = af[T - 1]
        ps = Pf[T - 1]
        m[k, T - 1] = ms
        P[k, T - 1] = ps if ps > 1e-8 else 1e-8
        for t in range(T - 2, -1, -1):
            pp1 = Pp[t + 1]
            if pp1 < 1e-12:
                pp1 = 1e-12
            G = rho * Pf[t] / pp1
            ms = af[t] + G * (ms - ap[t + 1])
            ps = Pf[t] + G * G * (ps - Pp[t + 1])
            m[k, t] = ms
            P[k, t] = ps if ps > 1e-8 else 1e-8

    invP = 1.0 / P
    mP = m * invP

    Ftr, Vtr = _dglm_gpoe(ptr, idx, w, nL, T, invP, mP)
    if n0 > 0:
        Fpr, Vpr = _dglm_gpoe(pptr, pidx, pw, n0, T, invP, mP)
    else:
        Fpr = np.zeros((0, T))
        Vpr = np.zeros((0, T))
    return Ftr, Vtr, Fpr, Vpr


def kfun(dist, band, kernel="exp"):
    """Vectorized kernel weight."""
    dist = np.asarray(dist, dtype=float)
    if kernel == "gau":
        return np.exp(-(dist ** 2) / (band ** 2))
    return np.exp(-dist / band)
