"""Coarse-to-fine spatial downscaling (CF-DS).

Python port of ``R/cf_downscale.R`` and ``R/cf_downscale_hv.R``.

``cf_downscale_hv`` trains the CF-DS model and selects the number of spatial
scales by sequential holdout validation; ``cf_downscale`` produces the final
disaggregate-level predictions (pycnophylactically adjusted so they aggregate
exactly to the observed aggregate-level ``Y``).
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ._utils_ds import (
    _agg_sum,
    _wls_coef,
    initial_ds_fun,
    lwr_ds,
    multiplicative_pycnophylactic,
)


@dataclass
class CFDownscaleHV:
    """Output of :func:`cf_downscale_hv`."""
    sse_hv: float
    sse_hv_all: list  # list of (name, value)
    id_train: np.ndarray
    other: dict = field(default_factory=dict)
    call: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        lines = ["cf_downscale_hv result",
                 "----Sum-of-squares errors for validation samples-----"]
        for name, val in self.sse_hv_all:
            lines.append(f"  {name:30s}  {val:.7g}")
        return "\n".join(lines)


@dataclass
class CFDownscale:
    """Output of :func:`cf_downscale`."""
    beta: dict
    sd_summary: list
    e_summary: list
    pred: dict            # {'pred', 'pred_sd'}
    bands: Optional[np.ndarray]
    Z: Optional[np.ndarray]
    Z_sd: Optional[np.ndarray]
    other: dict = field(default_factory=dict)
    call: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        out = ["cf_downscale result", "----Coefficients----"]
        for name, c, se, lo, hi in zip(
            self.beta["xname"], self.beta["coef"], self.beta["coef_se"],
            self.beta["lower_95CI"], self.beta["upper_95CI"]
        ):
            out.append(f"  {name:15s}  coef={c:+.5g}  se={se:.4g}  95% CI=[{lo:+.4g},{hi:+.4g}]")
        out.append("----Standard deviations (influential elements only)----")
        for name, sd in self.sd_summary:
            out.append(f"  {name:25s}  {sd:.5g}")
        out.append("----Error statistics----")
        for name, val in self.e_summary:
            out.append(f"  {name:25s}  {val:.5g}")
        return "\n".join(out)


def cf_downscale_hv(
    Y,
    Y_type: str = "sum",
    x=None,
    prop_weight=None,
    coords=None,
    agg_id=None,
    train_rat: float = 0.75,
    id_train=None,
    alpha: float = 0.9,
    kernel: str = "exp",
    rel_tol: float = 1e-4,
    seed: Optional[int] = 123,
    verbose: bool = True,
) -> CFDownscaleHV:
    """Holdout validation for coarse-to-fine spatial downscaling."""
    a = prop_weight
    stop_k = 5
    max_iter = 100
    agg_tol = 0.10
    agg_q = 0.95

    init = initial_ds_fun(Y=Y, Y_type=Y_type, x=x, a=a, coords=coords,
                          train_rat=train_rat, agg_id=agg_id, Id_train=id_train,
                          seed=seed)
    coords = np.asarray(coords, dtype=float)
    coords_uni = init.coords_uni
    Coords_uni = init.Coords_uni
    X = init.X
    W = init.W
    W_glob = init.W_glob
    x_mat = init.x
    a = init.a
    xname = init.xname
    n = init.n
    nx = init.nx
    N = init.N
    agg_inv = init.agg_inv
    id_train = init.Id_train
    Y = np.asarray(Y, dtype=float).ravel()

    use_valid = id_train.size < N
    not_train = np.ones(N, dtype=bool)
    not_train[id_train] = False

    # Refit the initial beta on training-only areas (nested chain -> monotone SSE).
    Xmat = np.asarray(X, dtype=float)
    beta_int = _wls_coef(Xmat[id_train], Y[id_train], W_glob[id_train]).reshape(-1, 1)
    Resid = Y - (Xmat @ beta_int).ravel()
    sse_tr0 = float(np.sum(W_glob[id_train] * Resid[id_train] ** 2))
    sse_va0 = float(np.sum(W_glob[not_train] * Resid[not_train] ** 2)) if use_valid else float("nan")

    max_d = math.sqrt(np.ptp(coords[:, 0]) ** 2 + np.ptp(coords[:, 1]) ** 2) / 3
    Bands = max_d * (alpha ** np.arange(1, max_iter + 1))

    b_old = None
    bands: list = []
    bid: list = []
    pred_sp = np.zeros(n)
    SSE_train = [sse_tr0]
    SSE_valid = [sse_va0]
    count_norun = 0
    pred_sp_add_list: list = []

    # Phase-1 aggregation gate metric (multi-point training areas only).
    area_size = np.bincount(agg_inv, minlength=N)
    Id_train_agg = np.intersect1d(id_train, np.where(area_size > 1)[0])
    if Id_train_agg.size == 0:
        Id_train_agg = id_train
    Y_sd_const = float(np.std(Y, ddof=1))
    if not np.isfinite(Y_sd_const) or Y_sd_const <= 0:
        Y_sd_const = 1.0
    AGG_err_tr = [float(np.quantile(np.abs(Resid[Id_train_agg]) / Y_sd_const, agg_q))]

    if verbose:
        print("--- SSE: Linear regression ---")
    SSE_init = sse_va0 if use_valid else sse_tr0
    if verbose:
        print(f"  {SSE_init:.7g}")
    SSE = [SSE_init]
    SSE_name = ["linear regression"]

    if verbose:
        print("--- SSE: Learning multi-scale spatial process ---")

    agg_satisfied = False
    plateau_va = 0
    Pred_sp_areal = np.zeros(N)
    gamma_list: list = []
    beta = np.asarray(beta_int).ravel()

    for i, band in enumerate(Bands):
        lmod = lwr_ds(coords=coords, coords_uni=coords_uni, Resid=Resid,
                      beta_int=beta_int, Coords_uni=Coords_uni, Y=Y, X=X, W=W,
                      x=x_mat, a=a, band=band, b_old=b_old, ridge=False,
                      kernel=kernel, Id_train=id_train, agg_inv=agg_inv, N=N,
                      sel_id=None, sse_hv0=None, pred_sp=pred_sp,
                      func="cf_downscale", knots_train_only=True, c_shrink=0.0)
        if not lmod.get("run", False):
            count_norun += 1
            if count_norun >= stop_k:
                break
            continue
        count_norun = 0

        incr_raw = lmod["pred_sp"] - pred_sp
        incr_pt = incr_raw - np.mean(incr_raw)
        Pred_sp_add = _agg_sum(a * incr_pt, agg_inv, N)
        Y_lhs = Y - Pred_sp_areal
        design_tr = np.hstack([Xmat[id_train], Pred_sp_add[id_train][:, None]])
        coef_full = _wls_coef(design_tr, Y_lhs[id_train], W_glob[id_train])
        beta_cand = coef_full[:nx].copy()
        gamma_k = coef_full[nx]
        if not np.isfinite(gamma_k):
            gamma_k = 0.0
        if gamma_k > 1:
            gamma_k = 1.0
            beta_cand = _wls_coef(Xmat[id_train], Y_lhs[id_train] - Pred_sp_add[id_train],
                                  W_glob[id_train])
        elif gamma_k < 0:
            gamma_k = 0.0
            beta_cand = _wls_coef(Xmat[id_train], Y_lhs[id_train], W_glob[id_train])

        pred_sp_cand = pred_sp + gamma_k * incr_pt
        Pred_sp_areal_cand = Pred_sp_areal + gamma_k * Pred_sp_add
        Resid_cand = Y - (Xmat @ beta_cand) - Pred_sp_areal_cand
        sse_tr_cand = float(np.sum(W_glob[id_train] * Resid_cand[id_train] ** 2))
        sse_va_cand = (float(np.sum(W_glob[not_train] * Resid_cand[not_train] ** 2))
                       if use_valid else float("nan"))
        agg_err_tr_cand = float(np.quantile(np.abs(Resid_cand[Id_train_agg]) / Y_sd_const, agg_q))

        b_old = lmod["b_old"]
        pred_sp = pred_sp_cand
        Pred_sp_areal = Pred_sp_areal_cand
        Resid = Resid_cand
        beta = beta_cand
        pred_sp_add_list.append(incr_pt)
        gamma_list.append(gamma_k)
        bands.append(band)
        bid.append(i)
        SSE_train.append(sse_tr_cand)
        SSE_valid.append(sse_va_cand)
        AGG_err_tr.append(agg_err_tr_cand)

        if not agg_satisfied and agg_err_tr_cand <= agg_tol:
            agg_satisfied = True

        if use_valid:
            prev = np.asarray(SSE_valid[:-1], dtype=float)
            prev_min = float(np.nanmin(prev)) if np.any(np.isfinite(prev)) else float("inf")
            improved = np.isfinite(sse_va_cand) and sse_va_cand < prev_min * (1 - rel_tol)
        else:
            prev = np.asarray(SSE_train[:-1], dtype=float)
            prev_min = float(np.nanmin(prev)) if np.any(np.isfinite(prev)) else float("inf")
            improved = np.isfinite(sse_tr_cand) and sse_tr_cand < prev_min * (1 - rel_tol)
        plateau_va = 0 if improved else plateau_va + 1

        sse_show = sse_va_cand if use_valid else sse_tr_cand
        if verbose:
            tag = " " * (1 if (i + 1) >= 10 else 2)
            comment = "" if agg_satisfied else " agg constraint not yet satisfied"
            print(f"  {sse_show:.7g} (Scale{tag}{i+1}){comment}")
        SSE.append(sse_show)
        SSE_name.append(f"scale {i+1}")

        if agg_satisfied and plateau_va >= stop_k:
            break

    K = len(bands)
    if verbose:
        if K > 0:
            print()
            print(f"-> Selected finest scale: {K} (bandwidth: {bands[K-1]:.7g})")
            print()
        else:
            print("Warning: No residual spatial process was detected.")

    if K > 0:
        sse_hv = (float(np.sum(W_glob[not_train] * Resid[not_train] ** 2)) if use_valid
                  else float(np.sum(W_glob[id_train] * Resid[id_train] ** 2)))
    else:
        sse_hv = sse_va0 if use_valid else sse_tr0

    sse_hv_all = list(zip(SSE_name, SSE))

    pred = (x_mat @ beta) + pred_sp
    pred = np.where(pred < 0, 0.0, pred)
    if Y_type == "sum":
        pred = a * pred

    Pred_areal = (Xmat @ beta) + Pred_sp_areal

    other = {
        "bands": np.asarray(bands) if bands else None,
        "bands_all": Bands, "alpha": alpha, "x": x_mat, "X": X, "xname": xname,
        "kernel": kernel, "coords_uni": coords_uni, "Coords_uni": Coords_uni,
        "bid": bid, "pred": pred, "Pred_areal": Pred_areal, "a": a,
        "Y_type": Y_type, "rel_tol": rel_tol, "agg_tol": agg_tol, "agg_q": agg_q,
        "SSE_train": SSE_train, "SSE_valid": SSE_valid, "AGG_err_tr": AGG_err_tr,
        "gamma_list": gamma_list, "agg_inv": agg_inv, "N": N,
    }
    return CFDownscaleHV(
        sse_hv=sse_hv, sse_hv_all=sse_hv_all, id_train=id_train, other=other,
        call={"Y": "Y", "Y_type": Y_type, "kernel": kernel, "alpha": alpha,
              "rel_tol": rel_tol},
    )


def cf_downscale(
    Y,
    x=None,
    prop_weight=None,
    coords=None,
    agg_id=None,
    *,
    mod_hv: CFDownscaleHV,
    adj: bool = True,
    nonneg: bool = True,
    verbose: bool = True,
) -> CFDownscale:
    """Downscale with a trained CF-DS model."""
    a = prop_weight
    adj_method = "none" if adj is False else "mult"

    bands = mod_hv.other["bands"]
    xname = mod_hv.other["xname"]
    kernel = mod_hv.other["kernel"]
    Y_type = mod_hv.other["Y_type"]
    id_train_hv = mod_hv.id_train

    Y = np.asarray(Y, dtype=float).ravel()
    N = Y.shape[0]
    init = initial_ds_fun(Y=Y, Y_type=Y_type, x=x, a=a, coords=coords,
                          train_rat=1, agg_id=agg_id, Id_train=None)
    beta_int = init.beta_int
    X = init.X
    Coords_uni = init.Coords_uni
    coords_uni = init.coords_uni
    W = init.W
    W_glob = init.W_glob
    x_mat = init.x
    a = init.a
    Resid = init.Resid
    agg_inv = init.agg_inv
    Id_train = init.Id_train
    n = init.n
    nx = init.nx
    coords = np.asarray(coords, dtype=float)

    Bands = bands if bands is not None else np.empty(0)
    n_bands = len(Bands)
    Z = np.zeros((n, n_bands))
    Z_sd = np.zeros((n, n_bands))

    if verbose:
        print("--- Learning multi-scale spatial process ---")

    Xmat = np.asarray(X, dtype=float)
    Pred_sp_areal = np.zeros(N)
    b_old = None
    beta = np.asarray(beta_int).ravel().copy()
    gamma_vec = np.zeros(n_bands)
    pred_sp = np.zeros(n)

    for i in range(n_bands):
        band = Bands[i]
        lmod = lwr_ds(coords=coords, coords_uni=coords_uni, Resid=Resid,
                      beta_int=beta_int, Coords_uni=Coords_uni, Y=Y, X=X, W=W,
                      x=x_mat, a=a, band=band, b_old=b_old, ridge=False,
                      kernel=kernel, Id_train=Id_train, agg_inv=agg_inv, N=N,
                      sel_id=None, sse_hv0=None, pred_sp=pred_sp,
                      func="cf_downscale", knots_train_only=True, c_shrink=0.0)
        b_old = lmod.get("b_old", b_old)

        incr_raw = lmod["pred_sp"] - pred_sp
        incr_pt = incr_raw - np.mean(incr_raw)
        Pred_sp_add = _agg_sum(a * incr_pt, agg_inv, N)
        design = np.hstack([Xmat, Pred_sp_add[:, None]])
        coef_full = _wls_coef(design, Y - Pred_sp_areal, W_glob)
        beta = coef_full[:nx].copy()
        gamma_k = coef_full[nx]
        if not np.isfinite(gamma_k):
            gamma_k = 0.0
        if gamma_k > 1:
            gamma_k = 1.0
            beta = _wls_coef(Xmat, (Y - Pred_sp_areal) - Pred_sp_add, W_glob)
        elif gamma_k < 0:
            gamma_k = 0.0
            beta = _wls_coef(Xmat, Y - Pred_sp_areal, W_glob)
        gamma_vec[i] = gamma_k

        pred_sp = pred_sp + gamma_k * incr_pt
        Pred_sp_areal = Pred_sp_areal + gamma_k * Pred_sp_add
        Resid = Y - (Xmat @ beta) - Pred_sp_areal

        Z[:, i] = gamma_k * incr_pt
        bv = lmod.get("beta_v")
        if bv is not None:
            bvc = bv[:, 0].copy()
            bvc[np.isinf(bvc)] = 0.0
            bvc[bvc < 0] = 0.0
            Z_sd[:, i] = np.abs(x_mat[:, 0]) * np.sqrt(bvc) * gamma_k

        if verbose:
            tag = " " * (1 if (i + 1) >= 10 else 2)
            print(f"  Scale{tag}{i+1} (bandwidth:{band:.7g})")

    pred = (x_mat @ beta) + pred_sp
    if nonneg:
        pred = np.where(pred < 0, 0.0, pred)
    pred_naive = a * pred

    if adj_method == "mult":
        pred = multiplicative_pycnophylactic(pred, Y, agg_inv, a, N)
    if Y_type == "sum":
        pred = a * pred
    if np.any(~np.isfinite(pred)):
        nbad = int(np.sum(~np.isfinite(pred)))
        warnings.warn(f"{nbad} non-finite point predictions replaced with 0.")
        pred = np.where(np.isfinite(pred), pred, 0.0)

    # Coefficient summary (W_glob-weighted GLS covariance).
    XtWX = Xmat.T @ (W_glob[:, None] * Xmat)
    try:
        XtWX_inv = np.linalg.solve(XtWX, np.eye(nx))
    except np.linalg.LinAlgError:
        XtWX_inv = np.full((nx, nx), np.nan)
    resid_areal = Y - (Xmat @ beta) - Pred_sp_areal
    sigma2_hat = float(np.sum(W_glob * resid_areal ** 2) / max(N - nx, 1))
    beta_vmat = sigma2_hat * XtWX_inv
    beta_se = np.sqrt(np.maximum(np.diag(beta_vmat), 0.0))
    beta_summary = {
        "xname": xname,
        "coef": beta,
        "coef_se": beta_se,
        "lower_95CI": beta - 1.96 * beta_se,
        "upper_95CI": beta + 1.96 * beta_se,
    }

    pred_sd_reg = np.sqrt(np.maximum(((x_mat @ beta_vmat) * x_mat).sum(axis=1), 0.0))
    pred_sd_sp = np.sqrt((Z_sd ** 2).sum(axis=1))

    # Holdout tau calibration of the spatial-process variance.
    Pred_areal_hv = mod_hv.other.get("Pred_areal")
    tau = 1.0
    if id_train_hv is not None and np.asarray(id_train_hv).size < N and Pred_areal_hv is not None:
        val = np.setdiff1d(np.arange(N), np.asarray(id_train_hv))
        if val.size >= 2:
            Vsp_area = _agg_sum((a * pred_sd_sp) ** 2, agg_inv, N)
            e2 = (Y[val] - Pred_areal_hv[val]) ** 2
            Wv = W_glob[val]
            verr = float(np.sum(Wv * e2) / np.sum(Wv))
            vfld = float(np.sum(Wv * Vsp_area[val]) / np.sum(Wv))
            tr = np.asarray(id_train_hv)
            sig2 = float(np.sum(W_glob[tr] * resid_areal[tr] ** 2) / np.sum(W_glob[tr]))
            num = verr - sig2
            se = math.sqrt(2.0 / val.size) * verr
            rel = num ** 2 / (num ** 2 + se ** 2) if (num > 0 and np.isfinite(se) and se > 0) else 0.0
            tau_raw = max(num, 1e-6) / vfld if vfld > 0 else 1.0
            tau = min(max(math.exp(math.log(tau_raw) * rel), 1e-2), 1e2)
            if not np.isfinite(tau):
                tau = 1.0

    pred_sd_density = np.sqrt(pred_sd_reg ** 2 + tau * pred_sd_sp ** 2)
    pred_sd = a * pred_sd_density if Y_type == "sum" else pred_sd_density
    pred_dict = {"pred": pred, "pred_sd": pred_sd}

    # sd_summary
    sd_summary = [("xb", float(np.std(x_mat @ beta, ddof=1)))]
    Z_out = Z if n_bands > 0 else None
    Z_sd_out = Z_sd if n_bands > 0 else None
    if n_bands > 0:
        for k in range(n_bands):
            sd_summary.append((f"spatial_scale{k+1}", float(np.std(Z[:, k], ddof=1))))
    sd_summary.append(("residuals", float(np.std(resid_areal, ddof=1))))

    # Areal holdout validation stats.
    r2 = rmse = mae = float("nan")
    if id_train_hv is not None and np.asarray(id_train_hv).size < N and Pred_areal_hv is not None:
        val_idx = np.setdiff1d(np.arange(N), np.asarray(id_train_hv))
        if val_idx.size >= 2:
            if np.std(Y[val_idx], ddof=1) > 0 and np.std(Pred_areal_hv[val_idx], ddof=1) > 0:
                r2 = float(np.corrcoef(Y[val_idx], Pred_areal_hv[val_idx])[0, 1] ** 2)
            rmse = float(math.sqrt(np.mean((Y[val_idx] - Pred_areal_hv[val_idx]) ** 2)))
            mae = float(np.mean(np.abs(Y[val_idx] - Pred_areal_hv[val_idx])))
    e_summary = [("validation_R2", r2), ("validation_RMSE", rmse), ("validation_MAE", mae)]

    Pred_agg = (_agg_sum(pred, agg_inv, N) if Y_type == "sum"
                else _agg_sum(a * pred, agg_inv, N))
    other = {
        "Y": Y, "x": x_mat, "a": a, "agg_inv": agg_inv, "N": N, "X": X,
        "beta_vmat": beta_vmat, "sigma2_hat": sigma2_hat, "pred_naive": pred_naive,
        "pred_sp": pred_sp, "Pred_agg": Pred_agg, "Pred_areal_hv": Pred_areal_hv,
        "tau": tau, "gamma_list": gamma_vec, "Y_type": Y_type, "sse_hv": mod_hv.sse_hv,
    }
    return CFDownscale(
        beta=beta_summary, sd_summary=sd_summary, e_summary=e_summary,
        pred=pred_dict, bands=bands, Z=Z_out, Z_sd=Z_sd_out, other=other,
    )
