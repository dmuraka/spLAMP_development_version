"""Coarse-to-fine dynamic (space-time) spatial GLMMs (CF-DGLMMs).

Python port of ``R/cf_dglm.R`` and ``R/cf_dglm_hv.R``.

The link-scale linear predictor is decomposed as
``g(mu_{i,t}) = x_{i,t}'beta + sum_k f_k(s_i,t) + offset``, where each scale-k
field ``f_k`` couples a per-knot AR(1) Kalman smoother in time with kernel
kriging in space. ``cf_dglm_hv`` selects the spatial scales by holdout
validation; ``cf_dglm`` refits the selected structure on the full sample and
predicts.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.stats import norm
import statsmodels.api as sm

from .families import Family, as_family
from ._cluster import spcf_cluster_se, optfield_SE
from ._prediction_se import obs_predict, apply_obs
from ._utils_ds import _wls_coef
from ._utils_glm import _deviance_residuals, spcf_clip_l
from ._utils_dglm import (
    dglm_ar1_ml,
    dglm_dynreg,
    dglm_knots,
    dglm_nbr,
    dglm_panel,
    dglm_scale,
    dglm_scale_apply,
    dglm_scale_setup,
    dglm_work,
    _dglm_aggregate_dense,
)
from ._neighbors import knnx

_QS = np.array([0.005, 0.025, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,
                0.9, 0.95, 0.975, 0.995])


@dataclass
class CFDGLMHV:
    loss_hv: float
    loss_hv_all: list
    e_summary: list
    id_train: np.ndarray
    other: dict = field(default_factory=dict)
    call: dict = field(default_factory=dict)

    def __repr__(self):
        out = ["cf_dglm_hv result", "---- Validation SSE trace over scales ----"]
        for name, val in self.loss_hv_all:
            out.append(f"  {name:30s}  {val:.7g}")
        out.append(f"Selected scales: {len(self.other['bands'])}  | "
                   f"AR(1): rho={self.other['rho']:.3f}, Q={self.other['Q']:.3g}")
        out.append("---- Out-of-sample validation metrics ----")
        for name, val in self.e_summary:
            out.append(f"  {name:25s}  {val:.5g}")
        return "\n".join(out)


@dataclass
class CFDGLM:
    beta: dict
    beta_tv: Optional[dict]
    beta_tv_sd: Optional[dict]
    sd_summary: list
    e_summary: list
    pred: dict
    pred0: Optional[dict]
    pred_q: dict
    pred0_q: Optional[dict]
    bands: Optional[np.ndarray]
    Z: Optional[np.ndarray]
    Z_sd: Optional[np.ndarray]
    Z0: Optional[np.ndarray]
    Z0_sd: Optional[np.ndarray]
    other: dict = field(default_factory=dict)
    call: dict = field(default_factory=dict)

    def __repr__(self):
        out = ["cf_dglm result", "---- Coefficients ----"]
        for name, c, se, lo, hi in zip(
            self.beta["xname"], self.beta["coef"], self.beta["coef_se"],
            self.beta["lower_95CI"], self.beta["upper_95CI"]
        ):
            out.append(f"  {name:15s}  coef={c:+.5g}  se={se:.4g}  95% CI=[{lo:+.4g},{hi:+.4g}]")
        out.append("---- Standard deviations (model elements) ----")
        for name, sd in self.sd_summary:
            out.append(f"  {name:25s}  {sd:.5g}")
        out.append("---- Error statistics ----")
        for name, val in self.e_summary:
            out.append(f"  {name:25s}  {val:.5g}")
        return "\n".join(out)


def _design_glm(x, n):
    """Intercept + non-constant covariate columns; returns (X, x_sel, xname)."""
    xname = ["Intercept"]
    if x is None:
        return np.ones((n, 1)), np.zeros(0, dtype=bool), xname
    cols = list(x.columns) if hasattr(x, "columns") else None
    x_arr = np.asarray(x, dtype=float)
    if x_arr.ndim == 1:
        x_arr = x_arr.reshape(-1, 1)
    x_sel = np.std(x_arr, axis=0, ddof=1) != 0
    if x_sel.sum() >= 1:
        if cols is not None:
            xname = ["Intercept"] + [str(cols[i]) for i in range(len(cols)) if x_sel[i]]
        else:
            xname = ["Intercept"] + [f"x{i+1}" for i in range(x_sel.size) if x_sel[i]]
    X = np.hstack([np.ones((n, 1)), x_arr[:, x_sel]])
    return X, x_sel, xname


def _glm_fit_params(X, y, family: Family, offset=None, maxiter=100):
    res = sm.GLM(np.asarray(y, dtype=float), np.asarray(X, dtype=float),
                 family=family.family, offset=offset).fit(maxiter=maxiter)
    b = np.asarray(res.params, dtype=float)
    b[~np.isfinite(b)] = 0.0
    return b


def cf_dglm_hv(y, x=None, coords=None, time=None, offset=None,
               train_rat: float = 0.75, id_train=None, alpha: float = 0.9,
               kernel: str = "exp", family=None, rho=None, Q=None,
               tvc=None, q_tvc=None, seed: Optional[int] = 1234,
               verbose: bool = True) -> CFDGLMHV:
    family = as_family(family) if family is not None else as_family("gaussian")
    y = np.asarray(y, dtype=float).ravel()
    n = y.shape[0]
    coords = np.asarray(coords, dtype=float)
    offset = np.zeros(n) if offset is None else np.asarray(offset, dtype=float).ravel()

    X, x_sel, xname = _design_glm(x, n)
    nx = X.shape[1]

    # time-varying columns (default: none)
    tv_cols = _resolve_tvc(tvc, x_sel, xname, nx)
    has_tv = tv_cols.size > 0
    const_cols = np.setdiff1d(np.arange(nx), tv_cols)

    pn = dglm_panel(coords, time)
    nL, T, lev = pn["nL"], pn["T"], pn["time_levels"]
    lk, tk, C = pn["lk"], pn["tk"], pn["C"]

    if id_train is None:
        K_tr = max(1, round(nL * train_rat))
        rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
        if nL > 30000 or K_tr >= nL:
            tr_loc = np.sort(rng.choice(nL, size=K_tr, replace=False))
        else:
            from ._utils import _kmeans_centers
            iter_max = 5 if nL > 5000 else 10
            ck = _kmeans_centers(C, min(K_tr, nL - 1), iter_max, seed if seed is not None else 4321)
            tr_loc = np.sort(knnx(C, ck, k=1).ravel())
            tr_loc = np.unique(tr_loc)
        id_train = np.where(np.isin(lk, tr_loc))[0]
    else:
        id_train = np.asarray(id_train, dtype=np.int64)
        tr_loc = np.unique(lk[id_train])
    id_train = np.asarray(id_train, dtype=np.int64)

    # pooled GLM linearization
    beta = _glm_fit_params(X, y, family, offset=offset)
    eta = X @ beta + offset
    zw = dglm_work(family, eta, y, offset)
    beta_c = beta.copy()
    if has_tv:
        beta_c[tv_cols] = 0
    r = zw["z"] - X @ beta_c
    Rp = np.full((nL, T), np.nan)
    Rp[lk, tk] = r
    Wp = np.full((nL, T), np.nan)
    Wp[lk, tk] = zw["w"]

    fi = np.sort(tr_loc)
    vi = np.setdiff1d(np.arange(nL), fi)
    if vi.size == 0:
        vi = fi
    Cfit, Cval = C[fi], C[vi]
    Rfit, Rval = Rp[fi].copy(), Rp[vi].copy()
    Wfit, Wval = Wp[fi].copy(), Wp[vi].copy()

    obs_f = np.where(np.isin(lk, fi))[0]
    fi_row = np.searchsorted(fi, lk[obs_f])
    fi_col = tk[obs_f]
    obs_v = np.where(np.isin(lk, vi))[0]
    vi_row = np.searchsorted(vi, lk[obs_v])
    vi_col = tk[obs_v]
    xval = X[obs_v][:, const_cols]

    # time-varying coefficient pre-estimate (skipped for the default no-tv case)
    tvp = np.zeros(n)
    if has_tv:
        dr = dglm_dynreg(r[obs_f], X[obs_f][:, tv_cols], zw["w"][obs_f],
                         tk[obs_f], T, q=q_tvc)
        tvbeta = dr["beta"]
        q_tvc = dr["q"]
        tvp = (X[:, tv_cols] * tvbeta[tk]).sum(axis=1)
        Rfit[fi_row, fi_col] = Rfit[fi_row, fi_col] - tvp[obs_f]
        Rval[vi_row, vi_col] = Rval[vi_row, vi_col] - tvp[obs_v]

    # bandwidth grid with a resolution floor
    max_d = math.sqrt(np.ptp(C[:, 0]) ** 2 + np.ptp(C[:, 1]) ** 2) / 3
    Bands = max_d * (alpha ** np.arange(1, 101))
    from scipy.spatial import cKDTree
    per_time_nn = []
    for t in range(T):
        lt = np.unique(lk[tk == t])
        if lt.size < 2:
            per_time_nn.append(np.nan)
        else:
            # per-time median nearest-neighbour distance (FNN::get.knn, k=1)
            dd, _ = cKDTree(C[lt]).query(C[lt], k=2)
            per_time_nn.append(float(np.median(dd[:, 1])))
    band_min = 0.5 * np.nanmean(per_time_nn) if np.any(np.isfinite(per_time_nn)) else np.nan
    if not np.isfinite(band_min):
        band_min = 0.0
    Bands = Bands[Bands >= band_min]
    if Bands.size == 0:
        Bands = np.array([max_d * alpha])

    sk = seed if seed is not None else 4321

    # global AR(1) (rho, Q)
    if rho is None or Q is None:
        Cfit_uni, _ = _unique(Cfit)
        kn = dglm_knots(Cfit_uni, Bands[0], seed=sk)
        nb = dglm_nbr(Cfit, kn, Bands[0], kernel)
        W0f = np.nan_to_num(Wfit, nan=0.0)
        R0f = np.nan_to_num(Rfit, nan=0.0)
        Zag, Rag = _dglm_aggregate_dense(nb, W0f, R0f, kn.shape[0])
        ml = dglm_ar1_ml(Zag, Rag)
        if rho is None:
            rho = ml["rho"]
        if Q is None:
            Q = ml["Q"]

    # greedy scale selection on holdout deviance
    yv_obs = y[obs_v]
    off_v = offset[obs_v]
    tvp_v = tvp[obs_v] if has_tv else np.zeros(obs_v.size)
    tvpf = tvp[obs_f] if has_tv else np.zeros(obs_f.size)
    mu_floor = 1e-8 if family.name == "poisson" else 0.0
    b_const = beta[const_cols].copy()
    xfc = X[obs_f][:, const_cols]
    offf = offset[obs_f]

    def dev_of(field_mat):
        fld = field_mat[vi_row, vi_col]
        eta_ = xval @ b_const + off_v + tvp_v + fld
        mu = np.maximum(family.linkinv(spcf_clip_l(eta_, family)), mu_floor)
        return float(np.nansum(_deviance_residuals(yv_obs, mu, family) ** 2))

    if verbose:
        print("--- Validation deviance: Basic GLM ---")
    pred_val = np.zeros((vi.size, T))
    Vval = np.zeros((vi.size, T))
    committed: list = []
    cumF_fit = np.zeros((fi.size, T))
    best = dev_of(pred_val)
    if verbose:
        print(f"  {best:.7g}")
        print("--- Validation deviance: Learning multi-scale space-time process ---")
    count = 0
    accept_num = 5
    Loss = [best]
    Loss_name = ["basic GLM"]
    for i, b in enumerate(Bands):
        sc = dglm_scale(Cfit, Rfit, Wfit, b, rho, Q, kernel, sk, Cpr=Cval)
        cumF_fit_try = cumF_fit + sc["Ftr"]
        Of_try = offf + cumF_fit_try[fi_row, fi_col] + tvpf
        b_try = _glm_fit_params(xfc, y[obs_f], family, offset=Of_try, maxiter=25)
        cumF_val_try = pred_val + sc["Fpr"]
        eta_v = xval @ b_try + off_v + tvp_v + cumF_val_try[vi_row, vi_col]
        mu_v = np.maximum(family.linkinv(spcf_clip_l(eta_v, family)), mu_floor)
        trial = float(np.nansum(_deviance_residuals(yv_obs, mu_v, family) ** 2))
        if trial < best - 1e-8:
            pred_val = cumF_val_try
            Vval = Vval + sc["Vpr"]
            cumF_fit = cumF_fit_try
            b_const = b_try
            committed.append(b)
            count = 0
            comment = ""
            eta_f = xfc @ b_const + Of_try
            zwf = dglm_work(family, eta_f, y[obs_f], offf)
            Rfit[:] = np.nan
            Wfit[:] = np.nan
            Rfit[fi_row, fi_col] = zwf["z"] - (eta_f - offf)
            Wfit[fi_row, fi_col] = zwf["w"]
            best = trial
        else:
            if (i + 1) > 10:
                count += 1
            comment = " no improvement"
        Loss.append(best)
        Loss_name.append(f"scale {i+1}")
        if verbose:
            print(f"  {best:.7g} (Scale {i+1}){comment}")
        if count == accept_num:
            break

    if has_tv:
        clean_fit = Rfit[fi_row, fi_col] + tvp[obs_f]
        q_tvc = dglm_dynreg(clean_fit, X[obs_f][:, tv_cols], zw["w"][obs_f],
                            tk[obs_f], T, q=None)["q"]

    loss_hv = dev_of(pred_val)
    loss_hv_all = list(zip(Loss_name, Loss))

    # tau calibration
    okf = np.isfinite(Rfit) & np.isfinite(Wfit) & (Wfit > 0)
    sig2 = float(np.sum(Wfit[okf] * Rfit[okf] ** 2) / np.sum(Wfit[okf])) if np.any(okf) else 0.0
    eta_v = xval @ b_const + off_v + tvp_v + pred_val[vi_row, vi_col]
    zwv = dglm_work(family, eta_v, yv_obs, off_v)
    Rval[:] = np.nan
    Wval[:] = np.nan
    Rval[vi_row, vi_col] = zwv["z"] - (eta_v - off_v)
    Wval[vi_row, vi_col] = zwv["w"]
    okv = (np.isfinite(Rval) & np.isfinite(Wval) & (Wval > 0)
           & np.isfinite(Vval) & (Vval > 0))
    if np.any(okv):
        Wv = Wval[okv]
        verr = float(np.sum(Wv * Rval[okv] ** 2) / np.sum(Wv))
        vfld = float(np.sum(Wv * Vval[okv]) / np.sum(Wv))
        num = verr - sig2
        se = math.sqrt(2.0 / max(int(okv.sum()), 1)) * verr
        rel = num ** 2 / (num ** 2 + se ** 2) if (num > 0 and np.isfinite(se) and se > 0) else 0.0
        tau_raw = max(num, 1e-6) / vfld if vfld > 0 else 1.0
        tau = min(max(math.exp(math.log(tau_raw) * rel), 1e-2), 1e2)
    else:
        tau = 1.0
    if not np.isfinite(tau):
        tau = 1.0

    # out-of-sample validation metrics
    muv_obs = family.linkinv(spcf_clip_l(eta_v, family))
    okp = np.isfinite(muv_obs) & np.isfinite(yv_obs)
    mu_lo = 1e-8 if family.name == "poisson" else 0.0
    rmse_hv = float(math.sqrt(np.mean((yv_obs[okp] - muv_obs[okp]) ** 2)))
    mae_hv = float(np.mean(np.abs(yv_obs[okp] - muv_obs[okp])))
    try:
        null_glm = sm.GLM(yv_obs[okp], np.ones((int(okp.sum()), 1)), family=family.family).fit()
        dnull = float(null_glm.null_deviance)
        dres = float(np.sum(_deviance_residuals(yv_obs[okp], np.maximum(muv_obs[okp], mu_lo), family) ** 2))
        r2_hv = 1 - dres / dnull if (np.isfinite(dnull) and dnull > 0) else float("nan")
    except Exception:
        r2_hv = float("nan")
    e_summary = [("validation_Pseudo-R2", r2_hv), ("validation_RMSE", rmse_hv),
                 ("validation_MAE", mae_hv)]

    other = {
        "bands": np.asarray(committed, dtype=float),
        "bands_all": Bands, "alpha": alpha, "kernel": kernel, "family": family,
        "rho": rho, "Q": Q, "sigma": math.sqrt(max(sig2, 0.0)),
        "x_sel": x_sel, "xname": xname, "seed": seed, "time_levels": lev,
        "tau": tau, "tv_cols": tv_cols, "q_tvc": q_tvc,
    }
    return CFDGLMHV(loss_hv=loss_hv, loss_hv_all=loss_hv_all, e_summary=e_summary,
                    id_train=id_train, other=other,
                    call={"family": family.name, "kernel": kernel, "alpha": alpha})


def cf_dglm(y, x=None, coords=None, time=None, offset=None,
            x0=None, coords0=None, time0=None, offset0=None,
            *, mod_hv: CFDGLMHV, robust_se: bool = True, sill_cap: bool = True,
            se_type: str = "prediction", se_method: str = "opt",
            verbose: bool = True) -> CFDGLM:
    family: Family = mod_hv.other["family"]
    bands = np.asarray(mod_hv.other["bands"], dtype=float)
    kernel = mod_hv.other["kernel"]
    rho, Q = mod_hv.other["rho"], mod_hv.other["Q"]
    x_sel = mod_hv.other["x_sel"]
    xname = mod_hv.other["xname"]
    lev = mod_hv.other["time_levels"]
    sk = mod_hv.other["seed"] if mod_hv.other["seed"] is not None else 4321
    tau = mod_hv.other["tau"]
    if tau is None or not np.isfinite(tau) or tau <= 0:
        tau = 1.0
    tv_cols = np.asarray(mod_hv.other.get("tv_cols", np.zeros(0, dtype=np.int64)), dtype=np.int64)
    q_tvc = mod_hv.other.get("q_tvc")

    y = np.asarray(y, dtype=float).ravel()
    n = y.shape[0]
    coords = np.asarray(coords, dtype=float)
    offset = np.zeros(n) if offset is None else np.asarray(offset, dtype=float).ravel()
    has0 = coords0 is not None

    if has0:
        if offset is not None and offset0 is None and np.any(offset != 0):
            raise ValueError("offset0 must be provided when offset is specified")
        if x is not None and x0 is None:
            raise ValueError("x0 must be provided when x is specified")

    lev_work = lev
    if has0 and time0 is not None:
        lev_work = np.unique(np.concatenate([np.asarray(lev), np.asarray(time0)]))

    # design
    if x is None:
        X = np.ones((n, 1))
    else:
        x_arr = np.asarray(x, dtype=float)
        if x_arr.ndim == 1:
            x_arr = x_arr.reshape(-1, 1)
        X = np.hstack([np.ones((n, 1)), x_arr[:, x_sel]])
    nx = X.shape[1]
    tv_cols = tv_cols[(tv_cols >= 1) & (tv_cols <= nx - 1) & (tv_cols != 0)]
    has_tv = tv_cols.size > 0
    const_cols = np.setdiff1d(np.arange(nx), tv_cols)

    pn = dglm_panel(coords, time, time_levels=lev_work)
    nL, T, Ctr = pn["nL"], pn["T"], pn["C"]
    lk, tk = pn["lk"], pn["tk"]

    if has0:
        coords0 = np.asarray(coords0, dtype=float)
        n0 = coords0.shape[0]
        offset0 = np.zeros(n0) if offset0 is None else np.asarray(offset0, dtype=float).ravel()
        if time0 is None:
            raise ValueError("time0 must be provided for prediction sites")
        if x0 is None:
            X0 = np.ones((n0, 1))
        else:
            x0_arr = np.asarray(x0, dtype=float)
            if x0_arr.ndim == 1:
                x0_arr = x0_arr.reshape(-1, 1)
            X0 = np.hstack([np.ones((n0, 1)), x0_arr[:, x_sel]])
        pn0 = dglm_panel(coords0, time0, time_levels=lev_work)
        Cpr = pn0["C"]
    else:
        n0 = None
        Cpr = None

    Xc = X[:, const_cols]
    Xtv = X[:, tv_cols]

    def f_obs(field):
        return field[lk, tk]

    def tvpart_of(tvbeta):
        if not has_tv:
            return np.zeros(n)
        return (Xtv * tvbeta[tk]).sum(axis=1)

    setups = [dglm_scale_setup(Ctr, bands[k], kernel, sk, Cpr=Cpr) for k in range(bands.size)]

    def casc_sweep(beta, tvbeta, q_cur, predict=False):
        tvpart = tvpart_of(tvbeta)
        f = np.zeros((nL, T))
        Ftr_sum = np.zeros((nL, T))
        Fpr_sum = np.zeros((Cpr.shape[0], T)) if (has0 and predict) else None
        sc_list = [None] * bands.size
        z = w = None
        for k in range(bands.size):
            fobs = f_obs(f)
            eta = spcf_clip_l(X @ beta + tvpart + fobs + offset, family)
            zw = dglm_work(family, eta, y, offset)
            z, w = zw["z"], zw["w"]
            resid = z - X @ beta - tvpart - fobs
            Rp = np.full((nL, T), np.nan)
            Rp[lk, tk] = resid
            Wp = np.full((nL, T), np.nan)
            Wp[lk, tk] = w
            sc = dglm_scale_apply(setups[k], Rp, Wp, rho, Q, predict=predict)
            f = f + sc["Ftr"]
            Ftr_sum = Ftr_sum + sc["Ftr"]
            if has0 and predict:
                Fpr_sum = Fpr_sum + sc["Fpr"]
            sc_list[k] = sc
            robs = (Rp - sc["Ftr"])[lk, tk]
            ba = _wls_coef(Xc, robs, w)
            ba[~np.isfinite(ba)] = 0.0
            beta = beta.copy()
            beta[const_cols] = beta[const_cols] + ba
            if has_tv:
                r_tv = z - X @ beta - f_obs(f)
                dr = dglm_dynreg(r_tv, Xtv, w, tk, T,
                                 q=None if _anynan(q_cur) else q_cur)
                tvbeta = dr["beta"]
                if _anynan(q_cur):
                    q_cur = dr["q"]
                tvpart = tvpart_of(tvbeta)
        return {"beta": beta, "tvbeta": tvbeta, "q_cur": q_cur, "Ftr": Ftr_sum,
                "Fpr": Fpr_sum, "scales": sc_list, "z": z, "w": w}

    beta = _glm_fit_params(X, y, family, offset=offset)
    tvbeta = np.zeros((T, tv_cols.size)) if has_tv else None
    if has_tv:
        beta[tv_cols] = 0
    if has_tv and q_tvc is not None and np.all(np.isfinite(q_tvc)) and np.all(np.asarray(q_tvc) > 0):
        q_cur = np.asarray(q_tvc, dtype=float)
    else:
        q_cur = np.array([np.nan])

    nb_scale = max(bands.size, 1)
    Z = np.zeros((n, nb_scale))
    Z_sd = np.zeros((n, nb_scale))
    Z0 = np.zeros((n0, nb_scale)) if has0 else None
    Z0_sd = np.zeros((n0, nb_scale)) if has0 else None
    f_tr = np.zeros(n)
    f0_obs = np.zeros(n0) if has0 else None
    tvpart = tvpart_of(tvbeta)
    z = w = None

    if verbose:
        print("--- Learning multi-scale space-time process ---")
    if bands.size == 0:
        eta = spcf_clip_l(X @ beta + tvpart + offset, family)
        zw = dglm_work(family, eta, y, offset)
        z, w = zw["z"], zw["w"]
        beta[const_cols] = _wls_coef(Xc, z - tvpart, w)
    else:
        sw = casc_sweep(beta, tvbeta, q_cur, predict=True)
        beta, tvbeta, q_cur, z, w = sw["beta"], sw["tvbeta"], sw["q_cur"], sw["z"], sw["w"]
        for k in range(bands.size):
            Z[:, k] = sw["scales"][k]["Ftr"][lk, tk]
            Z_sd[:, k] = np.sqrt(np.maximum(sw["scales"][k]["Vtr"][lk, tk], 0))
            if has0:
                Z0[:, k] = sw["scales"][k]["Fpr"][pn0["lk"], pn0["tk"]]
                Z0_sd[:, k] = np.sqrt(np.maximum(sw["scales"][k]["Vpr"][pn0["lk"], pn0["tk"]], 0))
        zmean = Z[:, :bands.size].mean(axis=0)
        Z[:, :bands.size] = Z[:, :bands.size] - zmean
        if has0:
            Z0[:, :bands.size] = Z0[:, :bands.size] - zmean
        f_tr = Z[:, :bands.size].sum(axis=1)
        if has0:
            f0_obs = Z0[:, :bands.size].sum(axis=1)

    tvV = None
    if has_tv:
        from ._utils_dglm import dglm_dynreg as _dr
        dr = _dr(z - X @ beta - f_tr, Xtv, w, tk, T, q=None if _anynan(q_cur) else q_cur)
        tvbeta, tvV = dr["beta"], dr["V"]
        tvpart = tvpart_of(tvbeta)
        q_tvc = dr["q"]

    # final GLM with the cascade field (and tv part) as offset
    const_cov = const_cols[const_cols != 0]
    ncv = const_cov.size
    off_tr = spcf_clip_l(f_tr, family) + tvpart + offset
    if ncv > 0:
        Xg = np.hstack([np.ones((n, 1)), X[:, const_cov]])
    else:
        Xg = np.ones((n, 1))
    gmod = sm.GLM(y, Xg, family=family.family, offset=off_tr).fit()
    beta_int = np.asarray(gmod.params, dtype=float)
    Vbeta = np.asarray(gmod.cov_params(), dtype=float)
    G_block = None
    if robust_se and bands.size > 0:
        try:
            V, G_block = spcf_cluster_se(y=y, X=Xg, beta=beta_int, field=f_tr,
                                         offset=offset + tvpart, family=family,
                                         coords=coords, bands=bands)
            Vbeta = V
        except Exception:
            pass
    beta_int_se = np.sqrt(np.maximum(np.diag(Vbeta), 0))
    beta_summary = {
        "xname": ["Intercept"] + [xname[i] for i in const_cov],
        "coef": beta_int,
        "coef_se": beta_int_se,
        "lower_95CI": beta_int - 1.96 * beta_int_se,
        "upper_95CI": beta_int + 1.96 * beta_int_se,
    }

    def tvvar(Xt, tk_):
        if not has_tv:
            return np.zeros(Xt.shape[0])
        out = np.array([float(Xt[i] @ tvV[tk_[i]] @ Xt[i]) for i in range(Xt.shape[0])])
        return np.maximum(out, 0)

    if sill_cap and family.name != "binomial" and bands.size > 0:
        sv = float(np.var(Z[:, :bands.size].sum(axis=1), ddof=1))
        sill = sv if (np.isfinite(sv) and sv > 0) else np.inf
    else:
        sill = np.inf

    # ---- opt+field coefficient covariance (default se_method="opt") ----
    # Recomputed once the calibrated per-point field SD s_f is available,
    # replacing the classic field-retained cluster-robust covariance. The
    # time-varying part is folded into the field so the shared optfield_SE
    # reproduces .dglm_optfield_SE.
    if robust_se and se_method == "opt" and bands.size > 0:
        try:
            s_f = np.sqrt(np.maximum(
                np.minimum(tau * (Z_sd[:, :bands.size] ** 2).sum(axis=1), sill), 0.0))
            V, G2 = optfield_SE(y=y, X=Xg, beta=beta_int, field=f_tr + tvpart,
                                s_f=s_f, offset=offset, family=family,
                                coords=coords, bands=bands)
            if np.all(np.isfinite(np.diag(V))) and np.all(np.diag(V) > 0):
                Vbeta = V
                G_block = G2
                beta_int_se = np.sqrt(np.maximum(np.diag(V), 0))
                beta_summary["coef_se"] = beta_int_se
                beta_summary["lower_95CI"] = beta_int - 1.96 * beta_int_se
                beta_summary["upper_95CI"] = beta_int + 1.96 * beta_int_se
        except Exception:
            pass

    pred = np.asarray(gmod.predict(which="mean"), dtype=float)
    pred_lin = np.asarray(gmod.predict(which="linear"), dtype=float)
    fieldvar = np.minimum(tau * (Z_sd[:, :bands.size] ** 2).sum(axis=1) if bands.size > 0 else 0.0, sill)
    pred_lin_sd = np.sqrt(np.maximum(((Xg @ Vbeta) * Xg).sum(axis=1)
                                     + tvvar(Xtv, tk) + fieldvar, 0))
    pred_sd = np.abs(family.mu_eta(pred_lin)) * pred_lin_sd
    qn = norm.ppf(_QS)
    pred_q_lin = pred_lin[:, None] + np.outer(pred_lin_sd, qn)
    pred_q = {f"q{q}": np.asarray(family.linkinv(pred_q_lin[:, k])) for k, q in enumerate(_QS)}
    pred_dict = {"pred": pred, "pred_sd": pred_sd}

    pred0_dict = pred0_q = None
    if has0:
        X0tv = X0[:, tv_cols]
        tvpart0 = (X0tv * tvbeta[pn0["tk"]]).sum(axis=1) if has_tv else np.zeros(n0)
        off0 = spcf_clip_l(f0_obs, family) + tvpart0 + offset0
        if ncv > 0:
            Xg0 = np.hstack([np.ones((n0, 1)), X0[:, const_cov]])
        else:
            Xg0 = np.ones((n0, 1))
        pred0_lin = Xg0 @ beta_int + off0
        pred0 = np.asarray(family.linkinv(pred0_lin), dtype=float)
        fieldvar0 = np.minimum(tau * (Z0_sd[:, :bands.size] ** 2).sum(axis=1) if bands.size > 0 else 0.0, sill)
        pred0_lin_sd = np.sqrt(np.maximum(((Xg0 @ Vbeta) * Xg0).sum(axis=1)
                                          + tvvar(X0tv, pn0["tk"]) + fieldvar0, 0))
        pred0_sd = np.abs(family.mu_eta(pred0_lin)) * pred0_lin_sd
        pred0_dict = {"pred": pred0, "pred_sd": pred0_sd}
        pred0_q_lin = pred0_lin[:, None] + np.outer(pred0_lin_sd, qn)
        pred0_q = {f"q{q}": np.asarray(family.linkinv(pred0_q_lin[:, k])) for k, q in enumerate(_QS)}

    Zc = Z[:, :bands.size] if bands.size > 0 else None
    Zc_sd = Z_sd[:, :bands.size] if bands.size > 0 else None
    Z0c = Z0[:, :bands.size] if (bands.size > 0 and has0) else None
    Z0c_sd = Z0_sd[:, :bands.size] if (bands.size > 0 and has0) else None

    # time-varying coefficients output
    beta_tv = beta_tv_sd = None
    if has_tv:
        beta_tv = {xname[tv_cols[j]]: tvbeta[:, j] for j in range(tv_cols.size)}
        beta_tv["time"] = lev_work
        beta_tv_sd = {xname[tv_cols[j]]: np.array([math.sqrt(max(tvV[t][j, j], 0)) for t in range(T)])
                      for j in range(tv_cols.size)}
        beta_tv_sd["time"] = lev_work

    # sd summary
    sd_summary = [("xb", float(np.std(Xg @ beta_int, ddof=1)))]
    if bands.size > 0:
        for k in range(bands.size):
            sd_summary.append((f"spatial_scale{k+1}", float(np.std(Zc[:, k], ddof=1))))
    if has_tv:
        for j in range(tv_cols.size):
            sd_summary.append((f"tv_{xname[tv_cols[j]]}", float(np.std(tvbeta[:, j], ddof=1))))

    # validation error statistics (holdout via mod_hv)
    idt = mod_hv.id_train
    mask = np.ones(n, dtype=bool)
    mask[idt] = False
    yt = y[mask]
    yp = pred[mask]
    if family.name == "binomial":
        yp_c = np.minimum(np.maximum(yp, 1e-6), 1 - 1e-6)
    elif family.name == "poisson":
        yp_c = np.maximum(yp, 1e-8)
    else:
        yp_c = yp
    try:
        null_glm = sm.GLM(yt, np.ones((yt.size, 1)), family=family.family).fit()
        dnull = float(null_glm.null_deviance)
        mu_fix = np.asarray(family.linkinv(family.linkfun(yp_c)), dtype=float)
        dres = float(np.sum(_deviance_residuals(yt, mu_fix, family) ** 2))
        r2 = 1 - dres / dnull if dnull > 0 else float("nan")
    except Exception:
        r2 = float("nan")
    rmse = float(math.sqrt(np.mean((yt - yp) ** 2)))
    mae = float(abs(np.mean(yt - yp)))
    e_summary = [("validation_Pseudo-R2", r2), ("validation_RMSE", rmse),
                 ("validation_MAE", mae)]

    other = {
        "n": n, "n0": n0, "nx": nx, "y": y, "coords": coords, "coords0": coords0,
        "rho": rho, "Q": Q, "kernel": kernel, "beta_int_vmat": Vbeta,
        "loss_hv": mod_hv.loss_hv, "tau": tau, "tv_cols": tv_cols, "q_tvc": q_tvc,
        "time": np.asarray(time), "time0": np.asarray(time0) if has0 else None,
        "time_levels": lev_work, "time_levels_train": lev,
        "robust_se": robust_se, "se_blocks": G_block,
    }
    result = CFDGLM(beta=beta_summary, beta_tv=beta_tv, beta_tv_sd=beta_tv_sd,
                    sd_summary=sd_summary, e_summary=e_summary,
                    pred=pred_dict, pred0=pred0_dict, pred_q=pred_q, pred0_q=pred0_q,
                    bands=bands if bands.size > 0 else None,
                    Z=Zc, Z_sd=Zc_sd, Z0=Z0c, Z0_sd=Z0c_sd, other=other)

    # ---- observation (data-distribution) predictive (default se_type) ----
    # cf_dglm_hv keeps no out-of-fold prediction vector in `other`, so this
    # calibrates on the in-sample noise floor (hvp=None), matching R.
    if se_type == "prediction":
        try:
            ob = obs_predict(
                family=family, y=y, hvp=None, id_train=mod_hv.id_train,
                pred_in=result.pred["pred"], predq_in=result.pred_q,
                pred_out=(result.pred0["pred"] if result.pred0 is not None else None),
                predq_out=result.pred0_q,
            )
        except Exception:
            ob = None
        result = apply_obs(result, ob)
    else:
        result.other["se_type"] = "mean"
    return result


# ---------------------------------------------------------------------------
def _resolve_tvc(tvc, x_sel, xname, nx):
    """Resolve time-varying-coefficient design columns as 0-based indices into
    the design matrix (never the intercept, column 0).

    ``tvc`` may be covariate names (matched against ``xname``) or 0-based
    integer indices into the original covariate matrix ``x``.
    """
    if tvc is None or int(np.sum(x_sel)) == 0:
        return np.zeros(0, dtype=np.int64)
    kept = np.where(x_sel)[0]                                  # 0-based selected covariate cols
    if isinstance(tvc, str) or (isinstance(tvc, (list, tuple, np.ndarray))
                                and len(tvc) and isinstance(tvc[0], str)):
        names = [tvc] if isinstance(tvc, str) else list(tvc)
        cols0 = [xname.index(nm) for nm in names if nm in xname]   # 0-based design col
    else:
        sel = [c for c in np.atleast_1d(tvc) if c in kept]
        cols0 = [1 + int(np.where(kept == c)[0][0]) for c in sel]  # 0-based design col
    cols0 = sorted({c for c in cols0 if 1 <= c <= nx - 1 and c != 0})
    return np.array(cols0, dtype=np.int64)


def _anynan(v):
    return bool(np.any(~np.isfinite(np.asarray(v, dtype=float))))


def _unique(a):
    from ._utils import _unique_rows_with_inverse
    return _unique_rows_with_inverse(np.asarray(a, dtype=float))
