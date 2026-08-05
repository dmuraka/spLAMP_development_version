"""Coarse-to-fine spatial GLMM for non-Gaussian responses."""
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
from ._neighbors import build_tree
from ._utils import _unique_rows_with_inverse
from ._utils_glm import (
    _deviance_residuals,
    _fit_glm,
    _glm_iwls_weights,
    initial_fun_glm,
    lwr_glm,
    response_se,
    spcf_clip_l,
)

# Predictive quantile levels (matches R cf_glm).
_QS = np.array([0.005, 0.025, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,
                0.9, 0.95, 0.975, 0.995])


@dataclass
class CFGLMHV:
    loss_hv: float
    loss_hv_all: list
    id_train: np.ndarray
    other: dict = field(default_factory=dict)
    call: dict = field(default_factory=dict)

    def __repr__(self):
        out = ["cf_glm_hv result", "---- Deviance losses for validation samples ----"]
        for name, val in self.loss_hv_all:
            out.append(f"  {name:30s}  {val:.7g}")
        return "\n".join(out)


@dataclass
class CFGLM:
    beta: dict
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
        out = ["cf_glm result", "---- Coefficients ----"]
        for name, c, se, lo, hi in zip(
            self.beta["xname"], self.beta["coef"], self.beta["coef_se"],
            self.beta["lower_95CI"], self.beta["upper_95CI"]
        ):
            out.append(f"  {name:15s}  coef={c:+.5g}  se={se:.4g}  95% CI=[{lo:+.4g},{hi:+.4g}]")
        out.append("---- Deviance losses (influential elements only) ----")
        for name, sd in self.sd_summary:
            out.append(f"  {name:25s}  {sd:.5g}")
        out.append("---- Error statistics ----")
        for name, val in self.e_summary:
            out.append(f"  {name:25s}  {val:.5g}")
        return "\n".join(out)


def cf_glm_hv(
    y,
    x=None,
    coords=None,
    offset=None,
    train_rat: float = 0.75,
    id_train=None,
    alpha: float = 0.9,
    kernel: str = "exp",
    family=None,
    seed: Optional[int] = 1234,
    verbose: bool = True,
) -> CFGLMHV:
    family = as_family(family) if family is not None else as_family("gaussian")

    init = initial_fun_glm(
        x=x, y=y, coords=coords, offset=offset, family=family,
        train_rat=train_rat, x_sel=None, id_train=id_train, seed=seed,
    )
    beta_int = init.beta_int
    beta = init.beta
    coords_arr = init.coords
    coords_uni, _ = _unique_rows_with_inverse(coords_arr)
    resid = init.resid.copy()
    x_mat = init.x
    x_sel = init.x_sel
    xname = init.xname
    offset_v = init.offset
    n = init.n
    nx = init.nx
    id_train = init.id_train
    w = init.weights
    vc = np.array([0], dtype=np.int64)
    ridge = True

    Bands_max = 100
    Z = np.zeros((n, Bands_max))
    max_d = math.sqrt(np.ptp(coords_uni[:, 0]) ** 2 + np.ptp(coords_uni[:, 1]) ** 2) / 3
    Bands = max_d * (alpha ** np.arange(1, Bands_max + 1))
    accept_num = 5

    sel_id_list: list = [None]
    b_old = None
    bands: list = []

    not_train = np.ones(n, dtype=bool)
    not_train[id_train] = False

    if verbose:
        print("--- Deviance: Basic GLM ---")
    glm0_res = init.glm_results
    mu0 = glm0_res.predict(which="mean")
    dev0 = _deviance_residuals(np.asarray(y, dtype=float), mu0, family)
    Loss = [float(np.sum(dev0[not_train] ** 2))]
    sse_hv0 = Loss[0]
    Loss_name = ["basic GLM"]
    if verbose:
        print(f"  {Loss[0]:.7g}")

    if verbose:
        print("--- Deviance: Learning multi-scale spatial process ---")
    l_pred = np.zeros(n)
    count = 0
    VCmat: list = []
    tree = build_tree(coords_arr)
    for i, band in enumerate(Bands):
        lmod = lwr_glm(
            coords=coords_arr, coords_uni=coords_uni, resid=resid, x=x_mat, weights=w,
            offset=offset_v, family=family, band=band, b_old=b_old, vc=vc,
            ridge=ridge, kernel=kernel, id_train=id_train, y=np.asarray(y, dtype=float),
            coords0=None, x0=None, sel_id=None, sse_hv0=sse_hv0, l_pred=l_pred,
            func="cf_glm_hv", tree=tree,
        )
        run = lmod.get("run", False)
        if run:
            bands.append(band)
            b_old = lmod["b_old"]
            sse_hv0 = lmod["sse_hv"]
            l_pred_add = lmod["pred"]
            l_pred = l_pred + l_pred_add
            l_bias = float(np.mean(l_pred))
            l_pred = l_pred - l_bias

            beta_add = lmod["beta"]
            beta_add[:, 0] = beta_add[:, 0] - l_bias
            beta = beta + beta_add
            Z[:, i] = beta_add[:, 0]
            sel_id_list.append(lmod["sel_id"])

            l_pred_off = spcf_clip_l(l_pred, family) + offset_v
            res = _fit_glm(np.asarray(y, dtype=float), x_mat, family, offset=l_pred_off)
            mu = res.predict(which="mean")
            mu_eta = family.mu_eta(res.predict(which="linear"))
            with np.errstate(divide="ignore", invalid="ignore"):
                resid = (np.asarray(y, dtype=float) - mu) / np.where(mu_eta != 0, mu_eta, np.nan)
            resid = np.nan_to_num(resid, nan=0.0, posinf=0.0, neginf=0.0)
            w = _glm_iwls_weights(res, family)
            beta_int_new = np.asarray(res.params, dtype=float).reshape(-1, 1)
            for jj in range(nx):
                beta[:, jj] = beta[:, jj] - beta_int[jj, 0] + beta_int_new[jj, 0]
            beta_int = beta_int_new
            d = _deviance_residuals(np.asarray(y, dtype=float), mu, family)
            loss_new = float(np.sum(d[not_train] ** 2))
            Loss.append(loss_new)
            vc_sel = lmod["vc_sel"]
            vcrow = np.zeros(nx)
            vcrow[vc_sel] = 1
            VCmat.append(vcrow)
            count = 0
            comment = ""
        else:
            if i + 1 > 10:
                count += 1
            if count == accept_num:
                break
            VCmat.append(np.zeros(nx))
            # Keep sel_id_list band-indexed (matches R; see cf_lm_hv note).
            sel_id_list.append(None)
            Loss.append(Loss[-1])
            comment = " no improvement"
        Loss_name.append(f"scale {i+1}")
        if verbose:
            tag = " " * (1 if (i + 1) >= 10 else 2)
            print(f"  {Loss[-1]:.7g} (Scale{tag}{i+1}){comment}")

    Z_sd_axis = Z.std(axis=0, ddof=1)
    nonzero = Z_sd_axis > 0
    if nonzero.sum() > 0:
        bid = np.where(nonzero)[0]
        max_bid = int(bid.max())
        Z = Z[:, : max_bid + 1]
        n_bid = bid.size
        z_pred = Z[:, bid].sum(axis=1)
        if verbose:
            print()
            print(f"-> Selected finest scale: {max_bid + 1} (bandwidth: {Bands[max_bid]:.7g})")
            print()
    else:
        Z = None
        n_bid = 0
        z_pred = np.zeros(n)

    xbeta = np.zeros(n)
    for j in range(nx):
        xbeta = xbeta + x_mat[:, j] * beta_int[j, 0]
    xbeta = xbeta + z_pred
    xbeta_off = spcf_clip_l(xbeta, family) + offset_v
    mu1 = family.linkinv(xbeta_off)
    d1 = _deviance_residuals(np.asarray(y, dtype=float), mu1, family)
    loss_hv = float(np.sum(d1[not_train] ** 2))

    loss_hv_all = list(zip(Loss_name, Loss))

    other = {
        "bands": np.asarray(bands) if bands else None,
        "bands_all": Bands,
        "alpha": alpha,
        "ridge": ridge,
        "vc": vc,
        "x_sel": x_sel,
        "sel_id_list": sel_id_list,
        "Loss": Loss,
        "coords_uni": coords_uni,
        "VCmat": np.asarray(VCmat) if VCmat else np.zeros((0, nx)),
        "kernel": kernel,
        "pred": mu1,
        "family": family,
    }
    return CFGLMHV(
        loss_hv=loss_hv,
        loss_hv_all=loss_hv_all,
        id_train=id_train,
        other=other,
        call={"y": "y", "x": "x", "coords": "coords", "kernel": kernel,
              "alpha": alpha, "family": family.name},
    )


def cf_glm(
    y,
    x=None,
    coords=None,
    offset=None,
    x0=None,
    coords0=None,
    offset0=None,
    *,
    mod_hv: CFGLMHV,
    robust_se: bool = True,
    se_type: str = "prediction",
    se_method: str = "opt",
    verbose: bool = True,
) -> CFGLM:
    family: Family = mod_hv.other["family"]
    bands = mod_hv.other["bands"]
    bands_all = mod_hv.other["bands_all"]
    coords_uni = mod_hv.other["coords_uni"]
    sel_id_list = mod_hv.other["sel_id_list"]
    ridge = mod_hv.other["ridge"]
    x_sel = mod_hv.other["x_sel"]
    VCmat = mod_hv.other["VCmat"]
    kernel = mod_hv.other["kernel"]

    if coords0 is not None:
        if offset is not None and offset0 is None:
            raise ValueError("offset0 must be provided when offset is specified")
        if x is not None and x0 is None:
            raise ValueError("x0 must be provided when x is specified")

    init = initial_fun_glm(
        x=x, y=y, coords=coords, offset=offset, family=family,
        x_sel=x_sel, train_rat=1,
    )
    beta_int = init.beta_int
    beta = init.beta
    coords_arr = init.coords
    resid = init.resid.copy()
    x_mat = init.x
    xname = init.xname
    offset_v = init.offset
    n = init.n
    nx = init.nx
    id_train = init.id_train
    w = init.weights
    y_arr = np.asarray(y, dtype=float)
    gmod0_res = init.glm_results

    nb = len(bands) if bands is not None else 0
    if coords0 is not None:
        coords0 = np.asarray(coords0, dtype=float)
        n0 = coords0.shape[0]
        one0 = np.ones((n0, 1))
        if x_sel is None or x_sel.size == 0 or int(x_sel.sum()) == 0:
            x0_full = one0
        else:
            x0_arr = np.asarray(x0, dtype=float)
            if x0_arr.ndim == 1:
                x0_arr = x0_arr.reshape(-1, 1)
            x0_full = np.hstack([one0, x0_arr[:, x_sel]])
        if offset0 is None:
            offset0 = np.zeros(n0)
        else:
            offset0 = np.asarray(offset0, dtype=float).ravel()
        beta0 = np.tile(beta_int.ravel(), (n0, 1))
        l_pred0 = np.zeros(n0)
        Z0 = np.zeros((n0, nb))
        Z0_sd = np.zeros_like(Z0)
        Z0_pv = np.zeros_like(Z0)
    else:
        n0 = None
        x0_full = None
        offset0 = None
        l_pred0 = None
        Z0 = Z0_sd = Z0_pv = None

    if verbose:
        print("--- Learning multi-scale spatial process ---")
    bands_scale = np.where(VCmat[:, 0] == 1)[0] if VCmat.size else np.array([], dtype=np.int64)
    b_old = None
    Z = np.zeros((n, nb))
    Z_sd = np.zeros_like(Z)
    Z_pv = np.zeros_like(Z)
    l_pred = np.zeros(n)

    tree = build_tree(coords_arr)
    tree0 = build_tree(coords0) if coords0 is not None else None

    if bands is not None and len(bands) > 0:
        max_i = int(bands_scale.max())
        for i in range(max_i + 1):
            vc_idx = np.where(VCmat[i] == 1)[0]
            lmod = lwr_glm(
                coords=coords_arr, coords_uni=coords_uni, resid=resid, x=x_mat, weights=w,
                offset=offset_v, family=family, band=bands_all[i], b_old=b_old,
                vc=vc_idx if vc_idx.size else np.array([0]), ridge=ridge, kernel=kernel,
                id_train=id_train, y=y_arr, x0=x0_full, coords0=coords0,
                sel_id=sel_id_list[i + 1] if (i + 1) < len(sel_id_list) else None,
                l_pred=l_pred, func="cf_glm", tree=tree, tree0=tree0,
            )
            b_old = lmod.get("b_old", b_old)
            if vc_idx.size > 0 and lmod.get("run", False):
                l_pred_add = lmod["pred"]
                l_pred = l_pred + l_pred_add
                l_bias = float(np.mean(l_pred))
                l_pred = l_pred - l_bias

                beta_add = lmod["beta"]
                beta_add[:, 0] = beta_add[:, 0] - l_bias
                beta = beta + beta_add
                beta_v_add = lmod["beta_v"]
                beta_v_add[np.isinf(beta_v_add)] = 0
                ii = int(np.where(bands_scale == i)[0][0])
                Z[:, ii] = beta_add[:, 0]
                Z_sd[:, ii] = np.sqrt(np.maximum(beta_v_add[:, 0], 0))
                bpv = lmod["beta_pv"][:, 0].copy()
                bpv[~np.isfinite(bpv)] = 0
                Z_pv[:, ii] = np.sqrt(np.maximum(bpv, 0))

                l_pred_off = spcf_clip_l(l_pred, family) + offset_v
                res = _fit_glm(y_arr, x_mat, family, offset=l_pred_off)
                gmod0_res = res
                mu = res.predict(which="mean")
                mu_eta = family.mu_eta(res.predict(which="linear"))
                with np.errstate(divide="ignore", invalid="ignore"):
                    resid = (y_arr - mu) / np.where(mu_eta != 0, mu_eta, np.nan)
                resid = np.nan_to_num(resid, nan=0.0, posinf=0.0, neginf=0.0)
                w = _glm_iwls_weights(res, family)
                beta_int_new = np.asarray(res.params, dtype=float).reshape(-1, 1)
                for jj in range(nx):
                    beta[:, jj] = beta[:, jj] - beta_int[jj, 0] + beta_int_new[jj, 0]
                beta_int = beta_int_new

                if coords0 is not None:
                    l_pred0_add = lmod["pred0"]
                    l_pred0 = l_pred0 + l_pred0_add
                    l_pred0 = l_pred0 - l_bias
                    beta0_add = lmod["beta0"]
                    beta0_add[:, 0] = beta0_add[:, 0] - l_bias
                    beta0 = beta0 + beta0_add
                    beta0_v_add = lmod["beta0_v"]
                    beta0_v_add[np.isinf(beta0_v_add)] = 0
                    Z0[:, ii] = beta0_add[:, 0]
                    Z0_sd[:, ii] = np.sqrt(np.maximum(beta0_v_add[:, 0], 0))
                    b0pv = lmod["beta0_pv"][:, 0].copy()
                    b0pv[~np.isfinite(b0pv)] = 0
                    Z0_pv[:, ii] = np.sqrt(np.maximum(b0pv, 0))
                comment = ""
            else:
                comment = " no improvement (skipped)"

            if verbose:
                tag = " " * (1 if (i + 1) >= 10 else 2)
                print(f"  Scale{tag}{i+1} (bandwidth:{bands_all[i]:.7g}){comment}")

    pred_pre = np.asarray(gmod0_res.predict(which="mean"), dtype=float)

    # ---- coefficient summary (naive SE from last loop GLM) ----
    beta_int_vec = beta_int.ravel()
    beta_int_se = np.asarray(gmod0_res.bse, dtype=float)
    beta_summary = {
        "xname": xname,
        "coef": beta_int_vec,
        "coef_se": beta_int_se,
        "lower_95CI": beta_int_vec - 1.96 * beta_int_se,
        "upper_95CI": beta_int_vec + 1.96 * beta_int_se,
    }

    beta = np.tile(beta_int_vec, (n, 1))
    if coords0 is not None:
        beta0 = np.tile(beta_int_vec, (n0, 1))

    n_bid = len(bands) if bands is not None else 0
    b_field = np.zeros(n)
    if n_bid > 0:
        b_field = Z.sum(axis=1)
        beta[:, 0] = beta[:, 0] + b_field
        if coords0 is not None:
            b0 = Z0.sum(axis=1)
            beta0[:, 0] = beta0[:, 0] + b0

    # ---- refinement fit for predictions (spatial field as offset) ----
    gmod_off = spcf_clip_l(beta[:, 0], family) + offset_v
    gmod_res = _fit_glm(y_arr, x_mat, family, offset=gmod_off)
    pred_resp = np.asarray(gmod_res.predict(which="mean"), dtype=float)
    pred_lin = np.asarray(gmod_res.predict(which="linear"), dtype=float)
    beta_int_vmat = np.asarray(gmod_res.cov_params(), dtype=float)

    # ---- spatial-block cluster-robust coefficient covariance (default) ----
    if robust_se and bands is not None and len(bands) > 0:
        try:
            V, _G = spcf_cluster_se(y=y_arr, X=x_mat, beta=beta_int_vec, field=b_field,
                                    offset=offset_v, family=family,
                                    coords=coords_arr, bands=np.asarray(bands))
            beta_int_vmat = V
            beta_int_se = np.sqrt(np.maximum(np.diag(V), 0))
            beta_summary["coef_se"] = beta_int_se
            beta_summary["lower_95CI"] = beta_int_vec - 1.96 * beta_int_se
            beta_summary["upper_95CI"] = beta_int_vec + 1.96 * beta_int_se
        except Exception:
            pass

    qn = norm.ppf(_QS)

    # ---- holdout tau calibration of the spatial-process variance (link scale) ----
    field_var = (Z_pv ** 2).sum(axis=1)
    sill = float(np.var(Z.sum(axis=1), ddof=1)) if n_bid > 0 else 0.0
    if not np.isfinite(sill) or sill <= 0:
        sill = np.inf
    if family.name == "binomial":
        sill = np.inf

    tau = 1.0
    idt = mod_hv.id_train
    mh_pred = mod_hv.other.get("pred")
    if idt is not None and len(idt) < n and mh_pred is not None:
        val = np.setdiff1d(np.arange(n), idt)
        # in-sample working residual of the full model -> noise floor
        eta_f = spcf_clip_l(pred_lin, family)
        me_f = family.mu_eta(eta_f)
        v_f = np.maximum(family.variance(pred_resp), 1e-8)
        with np.errstate(divide="ignore", invalid="ignore"):
            r_f = (y_arr - pred_resp) / np.where(np.abs(me_f) < 1e-8, 1e-8, me_f)
        w_f = me_f ** 2 / v_f
        in_train = np.zeros(n, dtype=bool)
        in_train[idt] = True
        self_mask = np.isfinite(r_f) & np.isfinite(w_f) & (w_f > 0) & in_train
        sig2 = (float(np.sum(w_f[self_mask] * r_f[self_mask] ** 2) / np.sum(w_f[self_mask]))
                if np.any(self_mask) else 0.0)
        # holdout working residual at validation samples
        if family.name == "binomial":
            mu_h = np.minimum(np.maximum(mh_pred, 1e-6), 1 - 1e-6)
        elif family.name == "poisson":
            mu_h = np.maximum(mh_pred, 1e-8)
        else:
            mu_h = np.asarray(mh_pred, dtype=float)
        eta_h = spcf_clip_l(family.linkfun(mu_h), family)
        me_h = family.mu_eta(eta_h)
        v_h = np.maximum(family.variance(mu_h), 1e-8)
        with np.errstate(divide="ignore", invalid="ignore"):
            r_h = (y_arr - mu_h) / np.where(np.abs(me_h) < 1e-8, 1e-8, me_h)
        w_h = me_h ** 2 / v_h
        in_val = np.zeros(n, dtype=bool)
        in_val[val] = True
        okv = in_val & np.isfinite(r_h) & np.isfinite(w_h) & (w_h > 0) & np.isfinite(field_var) & (field_var > 0)
        if okv.sum() >= 2:
            Wv = w_h[okv]
            verr = float(np.sum(Wv * r_h[okv] ** 2) / np.sum(Wv))
            vfld = float(np.sum(Wv * field_var[okv]) / np.sum(Wv))
            num = verr - sig2
            se = math.sqrt(2.0 / okv.sum()) * verr
            rel = num ** 2 / (num ** 2 + se ** 2) if (num > 0 and np.isfinite(se) and se > 0) else 0.0
            tau_raw = max(num, 1e-6) / vfld if vfld > 0 else 1.0
            tau = min(max(math.exp(math.log(tau_raw) * rel), 1e-2), 1e2)
            if not np.isfinite(tau):
                tau = 1.0

    fv_cal = np.minimum(tau * field_var, sill)
    with np.errstate(divide="ignore", invalid="ignore"):
        Z_sd = Z_pv * np.sqrt(np.where(field_var > 0, fv_cal / field_var, 1.0))[:, None]

    # ---- opt+field coefficient covariance (default se_method="opt") ----
    # Recomputed once the calibrated per-point field SD s_f = sqrt(fv_cal) is
    # available, replacing the classic field-retained cluster-robust covariance.
    if robust_se and se_method == "opt" and n_bid > 0:
        try:
            V, _G = optfield_SE(y=y_arr, X=x_mat, beta=beta_int_vec, field=b_field,
                                s_f=np.sqrt(fv_cal), offset=offset_v, family=family,
                                coords=coords_arr, bands=np.asarray(bands))
            if np.all(np.isfinite(np.diag(V))) and np.all(np.diag(V) > 0):
                beta_int_vmat = V
                beta_int_se = np.sqrt(np.maximum(np.diag(V), 0))
                beta_summary["coef_se"] = beta_int_se
                beta_summary["lower_95CI"] = beta_int_vec - 1.96 * beta_int_se
                beta_summary["upper_95CI"] = beta_int_vec + 1.96 * beta_int_se
        except Exception:
            pass

    pred_lin_sd = np.sqrt(((x_mat @ beta_int_vmat) * x_mat).sum(axis=1) + fv_cal)
    pred_sd = response_se(pred_lin, pred_lin_sd, family)

    pred_q_lin = pred_lin[:, None] + np.outer(pred_lin_sd, qn)
    pred_q = {f"q{q}": np.asarray(family.linkinv(pred_q_lin[:, k])) for k, q in enumerate(_QS)}
    pred_dict = {"pred": pred_resp, "pred_sd": pred_sd}

    pred0_dict = None
    pred0_q = None
    if coords0 is not None:
        gmod0_off = spcf_clip_l(beta0[:, 0], family) + offset0
        pred0_lin = (x0_full @ np.asarray(gmod_res.params, dtype=float)) + gmod0_off
        pred0_resp = np.asarray(family.linkinv(pred0_lin), dtype=float)
        field_var0 = (Z0_pv ** 2).sum(axis=1)
        fv0_cal = np.minimum(tau * field_var0, sill)
        with np.errstate(divide="ignore", invalid="ignore"):
            Z0_sd = Z0_pv * np.sqrt(np.where(field_var0 > 0, fv0_cal / field_var0, 1.0))[:, None]
        pred0_lin_sd = np.sqrt(((x0_full @ beta_int_vmat) * x0_full).sum(axis=1) + fv0_cal)
        pred0_sd = response_se(pred0_lin, pred0_lin_sd, family)
        pred0_dict = {"pred": pred0_resp, "pred_sd": pred0_sd}
        pred0_q_lin = pred0_lin[:, None] + np.outer(pred0_lin_sd, qn)
        pred0_q = {f"q{q}": np.asarray(family.linkinv(pred0_q_lin[:, k])) for k, q in enumerate(_QS)}

    Z_out = Z if n_bid > 0 else None
    Z_sd_out = Z_sd if n_bid > 0 else None
    Z0_out = Z0 if (n_bid > 0 and coords0 is not None) else None
    Z0_sd_out = Z0_sd if (n_bid > 0 and coords0 is not None) else None

    # ---- sd_summary (no residuals row for GLM) ----
    sd_summary = [("xb", float(np.std(x_mat @ beta_int_vec, ddof=1)))]
    if Z_out is not None:
        for k, sc in enumerate(bands_scale):
            sd_summary.append((f"spatial_scale{int(sc)+1}", float(np.std(Z_out[:, k], ddof=1))))

    # ---- error statistics (holdout validation via mod_hv.pred) ----
    ival = np.setdiff1d(np.arange(n), mod_hv.id_train)
    r2 = rmse = mae = float("nan")
    if ival.size >= 2 and mh_pred is not None:
        y_test = y_arr[ival]
        y_pred = np.asarray(mh_pred, dtype=float)[ival]
        # Deviance-based pseudo-R2: 1 - dev(y ~ 0 + offset(link(y_pred))) / null_dev.
        try:
            mu_fix = np.asarray(family.linkinv(family.linkfun(y_pred)), dtype=float)
            fix_dev = float(np.sum(_deviance_residuals(y_test, mu_fix, family) ** 2))
            null_glm = sm.GLM(y_test, np.ones((y_test.size, 1)), family=family.family).fit()
            null_dev = float(null_glm.null_deviance)
            r2 = 1 - fix_dev / null_dev if null_dev > 0 else float("nan")
        except Exception:
            r2 = float("nan")
        rmse = float(math.sqrt(np.mean((y_test - y_pred) ** 2)))
        mae = float(np.mean(np.abs(y_test - y_pred)))
    e_summary = [
        ("validation_Pseudo-R2", r2),
        ("validation_RMSE", rmse),
        ("validation_MAE", mae),
    ]

    other = {
        "n": n, "n0": n0, "nx": nx, "y": y, "x": x_mat, "x0": x0_full, "VCmat": VCmat,
        "coords": coords_arr, "coords0": coords0, "pred_pre": pred_pre,
        "loss_hv": mod_hv.loss_hv, "family": family, "tau": tau,
        "Z_pv": Z_pv, "Z0_pv": Z0_pv,
    }
    result = CFGLM(
        beta=beta_summary, sd_summary=sd_summary, e_summary=e_summary,
        pred=pred_dict, pred0=pred0_dict, pred_q=pred_q, pred0_q=pred0_q,
        bands=np.asarray(bands) if bands is not None else None,
        Z=Z_out, Z_sd=Z_sd_out, Z0=Z0_out, Z0_sd=Z0_sd_out, other=other,
    )

    # ---- observation (data-distribution) predictive (default se_type) ----
    if se_type == "prediction":
        try:
            ob = obs_predict(
                family=family, y=y_arr, hvp=mh_pred, id_train=mod_hv.id_train,
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
