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
                pv_sel = (xv * xv * inv_wxxw) * sigma + sigma / ws
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
                    pv_sel0 = (xv0 * xv0 * inv_wxxw) * sigma + sigma / w0s
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
                pv_sel = (xv * xv * inv_wxxw) * sigma + sigma / ws
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
                    pv_sel0 = (xv0 * xv0 * inv_wxxw) * sigma + sigma / w0s
                    wei2_pv0 = w0s / pv_sel0
                    b_all0[sidx0, j] += wei2_pv0 * b_sel_val
                    bv_inv_all0[sidx0, j] += wei2_pv0 * inv_bv_sel
                    pv_inv_all0[sidx0, j] += wei2_pv0


def kfun(dist, band, kernel="exp"):
    """Vectorized kernel weight."""
    dist = np.asarray(dist, dtype=float)
    if kernel == "gau":
        return np.exp(-(dist ** 2) / (band ** 2))
    return np.exp(-dist / band)
