"""Coarse-to-fine spatial modeling for Gaussian response."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import norm

from ._cluster import GaussianIdentity, spcf_cluster_se, optfield_SE
from ._prediction_se import obs_predict, apply_obs
from ._neighbors import build_tree
from ._utils import (
    add_mod_lm,
    apply_cqr,
    bopt_core,
    cqr_offsets,
    initial_fun,
    lwr,
    total_qmat,
)

# Predictive quantile levels (matches R cf_lm).
_QLEV_OUT = np.array([0.005, 0.025, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7,
                      0.8, 0.9, 0.95, 0.975, 0.995])


@dataclass
class CFLMHV:
    """Output of `cf_lm_hv`."""
    sse_hv: float
    sse_hv_all: list  # list of (name, value)
    id_train: np.ndarray
    other: dict = field(default_factory=dict)
    call: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        lines = ["cf_lm_hv result", "----Sum-of-squares errors for validation samples-----"]
        for name, val in self.sse_hv_all:
            lines.append(f"  {name:30s}  {val:.7g}")
        return "\n".join(lines)


@dataclass
class CFLM:
    """Output of `cf_lm`."""
    beta: object  # data frame-like (dict of arrays)
    sd_summary: list
    e_summary: list
    pred: dict          # {'pred', 'pred_sd'}
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

    def __repr__(self) -> str:
        out = ["cf_lm result", "----Coefficients----"]
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


def cf_lm_hv(
    y,
    x=None,
    coords=None,
    train_rat: float = 0.75,
    id_train=None,
    alpha: float = 0.9,
    kernel: str = "exp",
    add_learn: str = "none",
    seed: Optional[int] = 123,
    verbose: bool = True,
) -> CFLMHV:
    """Holdout validation for the Gaussian CF spatial model."""
    init = initial_fun(
        x=x, y=y, coords=coords, x_sel=None,
        train_rat=train_rat, id_train=id_train, seed=seed,
    )
    xx_inv = init.xx_inv
    beta_int = init.beta_int
    beta = init.beta
    coords_arr = init.coords
    from ._utils import _unique_rows_with_inverse
    coords_uni, _ = _unique_rows_with_inverse(coords_arr)

    pred = init.pred
    resid = init.resid.copy()
    x_mat = init.x
    x_sel = init.x_sel
    xname = init.xname
    n = init.n
    nx = init.nx
    id_train = init.id_train
    vc = np.array([0], dtype=np.int64)
    ridge = True

    Z = np.zeros((n, 100))
    max_d = math.sqrt(np.ptp(coords_arr[:, 0]) ** 2 + np.ptp(coords_arr[:, 1]) ** 2) / 3
    Bands = max_d * (alpha ** np.arange(1, 101))
    accept_num = 5

    sel_id_list: list = [None]
    b_old = None
    bands: list = []
    SSE: list = []
    SSE_name: list = []

    tree = build_tree(coords_arr)

    not_train = np.ones(n, dtype=bool)
    not_train[id_train] = False

    sse0 = float(np.sum(resid[not_train] ** 2))
    if verbose:
        print("--- SSE: Linear regression ---")
        print(f"  {sse0:.7g}")
    SSE.append(sse0)
    SSE_name.append("linear regression")

    if verbose:
        print("--- SSE: Learning multi-scale spatial process ---")
    count = 0
    VCmat: list = []
    for i, band in enumerate(Bands):
        lmod = lwr(
            coords=coords_arr, coords_uni=coords_uni, resid=resid, x=x_mat,
            band=band, b_old=b_old, vc=vc, id_train=id_train, ridge=ridge,
            kernel=kernel, y=y, beta=beta, coords0=None, x0=None,
            sel_id=None, func="cf_lm_hv", tree=tree,
        )
        run = lmod["run"]
        if run:
            bands.append(band)
            b_old = lmod["b_old"]
            beta_add = lmod["beta"]
            pred_add = lmod["pred"]
            beta = beta + beta_add
            pred = pred + pred_add
            resid = np.asarray(y, dtype=float) - pred
            SSE.append(float(lmod["sse_hv"]))
            vc_sel = lmod["vc_sel"]
            vcrow = np.zeros(nx)
            vcrow[vc_sel] = 1
            VCmat.append(vcrow)

            beta_int_add = (xx_inv @ x_mat.T @ resid).reshape(-1, 1)
            pred0_add = (x_mat @ beta_int_add).ravel()
            beta = beta + beta_int_add.ravel()
            pred = pred + pred0_add
            resid = resid - pred0_add

            beta_add_m = beta_add.mean(axis=0)
            Z[:, i] = beta_add[:, 0] - beta_add_m[0]
            sel_id_list.append(lmod["sel_id"])
            beta_int = beta_int + beta_int_add + beta_add_m.reshape(-1, 1)
            count = 0
            comment = ""
        else:
            if i + 1 > 10:
                count += 1
            if count == accept_num:
                break
            VCmat.append(np.zeros(nx))
            # Keep sel_id_list band-indexed (matches R: sel_id_list[[i]] with NULL
            # gaps for skipped bands). Appending here alongside VCmat guarantees
            # sel_id_list[k+1] <-> band k, so cf_lm reads the right knots for a
            # committed band that follows a skipped one.
            sel_id_list.append(None)
            SSE.append(SSE[-1])
            comment = " no improvement"
        SSE_name.append(f"scale {i+1}")
        if verbose:
            tag = " " * (1 if (i + 1) >= 10 else 2)
            print(f"  {SSE[-1]:.7g} (Scale{tag}{i+1}){comment}")

    Z_sd_axis = Z.std(axis=0, ddof=1)
    nonzero = Z_sd_axis > 0
    if nonzero.sum() > 0:
        bid = np.where(nonzero)[0]
        max_bid = int(bid.max())
        Z = Z[:, : max_bid + 1]
        n_bid = bid.size
        if verbose:
            print()
            print(f"-> Selected finest scale: {max_bid + 1} (bandwidth: {Bands[max_bid]:.7g})")
            print()
    else:
        bid = None
        Z = None
        n_bid = 0

    bands_arr = np.asarray(bands, dtype=float) if bands else None

    if n_bid is not None and n_bid > 1:
        if verbose:
            print("--- SSE: After coefficient adjustment ---")
        ZZ = Z[:, bid]

        def obj(par):
            try:
                out = bopt_core(par, bands=bands_arr, Z=ZZ, beta_int=beta_int,
                                nx=nx, x=x_mat, y=np.asarray(y, dtype=float),
                                n_bid=n_bid, id_train=id_train)
                if not np.isfinite(out["sse"]):
                    return np.finfo(float).max
                return float(out["sse"])
            except Exception:
                return np.finfo(float).max

        try:
            import nlopt  # type: ignore

            opt = nlopt.opt(nlopt.LN_BOBYQA, 1)
            opt.set_min_objective(lambda x_, grad: obj(x_[0]))
            opt.set_maxeval(500)
            opt.set_lower_bounds([-1e6])
            opt.set_upper_bounds([1e6])
            sol = opt.optimize([0.0])
            v_opt0 = sol[0]
        except Exception:
            res = minimize_scalar(obj, method="brent", options={"maxiter": 500})
            v_opt0 = float(res.x)

        v_test = bopt_core(v_opt0, bands=bands_arr, Z=ZZ, beta_int=beta_int,
                           nx=nx, x=x_mat, y=np.asarray(y, dtype=float),
                           n_bid=n_bid, id_train=id_train)
        if v_test["sse"] < SSE[-1]:
            vpar = np.array([float(v_test["vpar"][0, 0]), float(v_opt0)])
        else:
            vpar = np.array([1.0, 0.0])
    else:
        if n_bid == 1:
            vpar = np.array([1.0, 0.0])
        else:
            vpar = np.array([np.nan, np.nan])
            if verbose:
                print("Warning: No residual spatial process was detected.")

    xbeta = np.zeros((n, nx))
    for j in range(nx):
        xbeta[:, j] = x_mat[:, j] * beta_int[j, 0]
    if not np.isnan(vpar[0]) and bid is not None and Z is not None and bands_arr is not None:
        w_0 = np.exp(-vpar[1] / bands_arr)
        w = vpar[0] * w_0 / w_0[0]
        w[w < 0] = 0
        b = Z[:, bid] @ w
        xbeta[:, 0] = xbeta[:, 0] + x_mat[:, 0] * b

    pred = xbeta.sum(axis=1)
    resid = np.asarray(y, dtype=float) - pred
    sse_hv = float(np.sum(resid[not_train] ** 2))
    SSE.append(sse_hv)
    SSE_name.append("coef. adjustment")
    if n_bid > 1 and verbose:
        print(f"  {sse_hv:.7g}")

    # Holdout prediction of the selected model at all samples (out-of-sample on
    # the validation complement); folds in the additional learner if present.
    pred_hv = pred.copy()

    if add_learn == "rf":
        if verbose:
            print("--- SSE: After additional learning ---")
        a_mod0 = add_mod_lm(
            add_learn="rf", train=True, resid=resid, x=x_mat, coords=coords_arr,
            id_train=id_train, nx=nx, xname=xname, sse_hv=sse_hv,
        )
        sse_hv = a_mod0["sse_hv"]
        SSE.append(sse_hv)
        SSE_name.append("additional learning")
        if verbose:
            print(f"  {sse_hv:.7g}")
        if a_mod0.get("a_run") and a_mod0.get("a_pred_hv") is not None:
            pred_hv[not_train] = pred_hv[not_train] + a_mod0["a_pred_hv"]
    else:
        a_mod0 = {"a_par": None, "a_run": False, "add_learn": add_learn}

    sse_hv_all = list(zip(SSE_name, SSE))

    other = {
        "bands": bands_arr,
        "bands_all": Bands,
        "vpar": vpar,
        "alpha": alpha,
        "ridge": ridge,
        "vc": vc,
        "x_sel": x_sel,
        "sel_id_list": sel_id_list,
        "coords_uni": coords_uni,
        "VCmat": np.asarray(VCmat) if VCmat else np.zeros((0, nx)),
        "kernel": kernel,
        "a_mod0": a_mod0,
        "pred_hv": pred_hv,
    }
    return CFLMHV(
        sse_hv=sse_hv,
        sse_hv_all=sse_hv_all,
        id_train=id_train,
        other=other,
        call={"y": "y", "x": "x", "coords": "coords", "kernel": kernel,
              "alpha": alpha, "add_learn": add_learn},
    )


def cf_lm(
    y,
    x=None,
    coords=None,
    x0=None,
    coords0=None,
    *,
    mod_hv: CFLMHV,
    robust_se: bool = True,
    sd_method: str = "qrf",
    se_type: str = "prediction",
    se_method: str = "opt",
    verbose: bool = True,
) -> CFLM:
    """Spatial regression / prediction with the trained CF-LM.

    Parameters
    ----------
    robust_se : bool
        If ``True`` (default), coefficient standard errors and the
        coefficient-uncertainty term of the predictive SD use a spatial-block
        cluster-robust sandwich accounting for local spatial correlation.
    sd_method : {"qrf", "tree_var", "residual"}
        Predictive-SD construction when ``add_learn="rf"`` was used at training.
        ``"qrf"`` (default) mirrors R's ranger(quantreg=TRUE) total-CQR path:
        the combined predictive distribution is simulated and conformalized on
        the validation samples. ``"tree_var"`` / ``"residual"`` are cheaper
        Gaussian approximations (per-point tree variance / homoskedastic).
    se_type : {"prediction", "mean"}
        Type of predictive uncertainty carried in ``pred``/``pred_q``.
        ``"prediction"`` (default) returns the observation (data-distribution)
        predictive, holdout-calibrated so its intervals cover NEW observations;
        the signal (mean) versions are kept in ``other['pred_signal']`` /
        ``other['pred_q_signal']``. ``"mean"`` returns only the signal predictive.
    se_method : {"opt", "classic"}
        Cluster-robust coefficient-SE estimator (used when ``robust_se=True``).
        ``"opt"`` (default) is the opt+field sandwich with a leverage-LOO ceiling
        (removes the classic conservatism); ``"classic"`` keeps the realised
        field inside the working residual.
    """
    if coords0 is not None and x is not None and x0 is None:
        raise ValueError("x0 must be provided when x is specified")

    bands = mod_hv.other["bands"]
    bands_all = mod_hv.other["bands_all"]
    coords_uni = mod_hv.other["coords_uni"]
    vpar = mod_hv.other["vpar"]
    sel_id_list = mod_hv.other["sel_id_list"]
    ridge = mod_hv.other["ridge"]
    x_sel = mod_hv.other["x_sel"]
    VCmat = mod_hv.other["VCmat"]
    kernel = mod_hv.other["kernel"]
    a_mod0 = mod_hv.other["a_mod0"]
    a_par = a_mod0.get("a_par")
    a_run = a_mod0.get("a_run", False)
    add_learn = a_mod0.get("add_learn", "none")

    init = initial_fun(x=x, y=y, coords=coords, x_sel=x_sel, train_rat=1)
    xx_inv = init.xx_inv
    beta_int = init.beta_int
    beta_int0 = beta_int.copy()          # initial OLS beta (for noise-floor sig_pre)
    coords_arr = init.coords
    pred = init.pred
    resid = init.resid.copy()
    x_mat = init.x
    xname = init.xname
    n = init.n
    nx = init.nx
    id_train = init.id_train
    y_arr = np.asarray(y, dtype=float)

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
        pred0 = (x0_full @ beta_int).ravel()
        nb = len(bands) if bands is not None else 0
        Z0 = np.zeros((n0, nb))
        Z0_sd = np.zeros_like(Z0)
        Z0_pv = np.zeros_like(Z0)
    else:
        n0 = None
        x0_full = None
        pred0 = None
        Z0 = Z0_sd = Z0_pv = None

    if verbose:
        print("--- Learning multi-scale spatial process ---")

    bands_scale = np.where(VCmat[:, 0] == 1)[0] if VCmat.size else np.array([], dtype=np.int64)
    b_old = None
    nb = len(bands) if bands is not None else 0
    Z = np.zeros((n, nb))
    Z_sd = np.zeros_like(Z)
    Z_pv = np.zeros_like(Z)

    tree = build_tree(coords_arr)
    tree0 = build_tree(coords0) if coords0 is not None else None

    if bands is not None and len(bands) > 0:
        max_i = int(bands_scale.max())
        for i in range(max_i + 1):
            vc_idx = np.where(VCmat[i] == 1)[0]
            lmod = lwr(
                coords=coords_arr, coords_uni=coords_uni, resid=resid, x=x_mat,
                band=bands_all[i], b_old=b_old, vc=vc_idx if vc_idx.size else np.array([0]),
                id_train=id_train, ridge=ridge, kernel=kernel,
                x0=x0_full, coords0=coords0,
                sel_id=sel_id_list[i + 1] if (i + 1) < len(sel_id_list) else None,
                func="cf_lm", tree=tree, tree0=tree0,
            )
            b_old = lmod.get("b_old", b_old)
            if vc_idx.size > 0 and lmod.get("run", False):
                beta_add = lmod["beta"]
                beta_v_add = lmod["beta_v"]
                beta_v_add[np.isinf(beta_v_add)] = 0
                pred_add = lmod["pred"]
                pred = pred + pred_add
                resid = y_arr - pred
                beta_int_add = (xx_inv @ x_mat.T @ resid).reshape(-1, 1)
                pred_int_add = (x_mat @ beta_int_add).ravel()
                pred = pred + pred_int_add
                resid = resid - pred_int_add

                ii = int(np.where(bands_scale == i)[0][0])
                beta_add_m = beta_add.mean(axis=0)
                Z[:, ii] = beta_add[:, 0] - beta_add_m[0]
                Z_sd[:, ii] = np.sqrt(np.maximum(beta_v_add[:, 0], 0))
                bpv = lmod["beta_pv"][:, 0].copy()
                bpv[~np.isfinite(bpv)] = 0
                Z_pv[:, ii] = np.sqrt(np.maximum(bpv, 0))
                beta_int = beta_int + beta_int_add + beta_add_m.reshape(-1, 1)

                if coords0 is not None:
                    beta0_add = lmod["beta0"]
                    beta0_v_add = lmod["beta0_v"]
                    beta0_v_add[np.isinf(beta0_v_add)] = 0
                    pred0_add = lmod["pred0"]
                    pred0 = pred0 + pred0_add
                    pred0_int_add = (x0_full @ beta_int_add).ravel()
                    pred0 = pred0 + pred0_int_add
                    Z0[:, ii] = beta0_add[:, 0] - beta_add_m[0]
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

    # ---- coefficient GLS covariance (diagonal field-variance) ----
    pred_pre = (x_mat @ beta_int0).ravel()
    sig_pre = float(np.sum((y_arr - pred_pre) ** 2) / max(n - nx, 1))
    v_diag = (Z_sd ** 2).sum(axis=1) + sig_pre
    inv_v = 1.0 / v_diag
    Xt_W_X = (x_mat.T * inv_v) @ x_mat
    beta_int_vmat = np.linalg.solve(Xt_W_X, np.eye(nx))
    beta_int_se = np.sqrt(np.maximum(np.diag(beta_int_vmat), 0))

    beta_int_vec = beta_int.ravel()
    beta_summary = {
        "xname": xname,
        "coef": beta_int_vec,
        "coef_se": beta_int_se,
        "lower_95CI": beta_int_vec - 1.96 * beta_int_se,
        "upper_95CI": beta_int_vec + 1.96 * beta_int_se,
    }

    # ---- tuning: weighted spatial-process correction ----
    beta = np.tile(beta_int_vec, (n, 1))
    if coords0 is not None:
        beta0 = np.tile(beta_int_vec, (n0, 1))

    n_bid = len(bands) if bands is not None else 0
    b_field = None
    if n_bid > 0:
        vpar_coef = bopt_core(vpar[1], bands=np.asarray(bands), Z=Z,
                              beta_int=beta_int, nx=nx, x=x_mat,
                              y=y_arr, n_bid=n_bid, id_train=None)["vpar"][0, 0]
        w_0 = np.exp(-vpar[1] / np.asarray(bands))
        w = float(vpar_coef) * w_0 / w_0[0]
        b_field = Z @ w
        beta[:, 0] = beta[:, 0] + b_field
        if coords0 is not None:
            b0 = Z0 @ w
            beta0[:, 0] = beta0[:, 0] + b0

    # ---- spatial-block cluster-robust coefficient covariance (default) ----
    if robust_se and n_bid > 0 and b_field is not None:
        try:
            V, _G = spcf_cluster_se(y=y_arr, X=x_mat, beta=beta_int_vec, field=b_field,
                                    offset=None, family=GaussianIdentity(),
                                    coords=coords_arr, bands=np.asarray(bands))
            beta_int_vmat = V
            beta_int_se = np.sqrt(np.maximum(np.diag(V), 0))
            beta_summary["coef_se"] = beta_int_se
            beta_summary["lower_95CI"] = beta_int_vec - 1.96 * beta_int_se
            beta_summary["upper_95CI"] = beta_int_vec + 1.96 * beta_int_se
        except Exception:
            pass

    # ---- additional learning ----
    a_mod = {"add_learn": "none"}
    a_pred = a_pred0 = 0.0
    a_pred_v = a_pred0_v = 0.0
    if a_run:
        a_mod = add_mod_lm(
            add_learn=add_learn, train=False, resid=resid,
            x=x_mat, coords=coords_arr, x0=x0_full, coords0=coords0,
            nx=nx, xname=xname, a_par=a_par, sd_method=sd_method,
        )
        a_pred = a_mod["pred"]
        a_pred0 = a_mod["pred0"] if coords0 is not None else 0.0
        a_pred_v = a_mod.get("pred_v", 0.0)
        a_pred0_v = a_mod.get("pred0_v", 0.0) if coords0 is not None else 0.0

    # ---- prediction ----
    pred = (x_mat * beta).sum(axis=1) + a_pred
    coef_var = ((x_mat @ beta_int_vmat) * x_mat).sum(axis=1)
    field_var = (Z_pv ** 2).sum(axis=1)
    sill = float(np.var(Z.sum(axis=1), ddof=1)) if n_bid > 0 else 0.0
    if not np.isfinite(sill) or sill <= 0:
        sill = np.inf
    qn_hi = float(norm.ppf(_QLEV_OUT[-1]))
    if coords0 is not None:
        pred0 = (x0_full * beta0).sum(axis=1) + a_pred0
        coef_var0 = ((x0_full @ beta_int_vmat) * x0_full).sum(axis=1)
        field_var0 = (Z0_pv ** 2).sum(axis=1)

    # ---- holdout tau calibration of the spatial-process variance ----
    tau = 1.0
    idt = mod_hv.id_train
    pred_hv = mod_hv.other.get("pred_hv")
    if idt is not None and len(idt) < n and pred_hv is not None:
        val = np.setdiff1d(np.arange(n), idt)
        sig2 = float(np.mean((y_arr[idt] - pred[idt]) ** 2))
        e2 = (y_arr[val] - pred_hv[val]) ** 2
        fv = field_var[val]
        okv = np.isfinite(e2) & np.isfinite(fv) & (fv > 0)
        if okv.sum() >= 2:
            verr = float(np.mean(e2[okv]))
            vfld = float(np.mean(fv[okv]))
            num = verr - sig2
            se = math.sqrt(2.0 / okv.sum()) * verr
            rel = num ** 2 / (num ** 2 + se ** 2) if (num > 0 and np.isfinite(se) and se > 0) else 0.0
            tau_raw = max(num, 1e-6) / vfld if vfld > 0 else 1.0
            tau = min(max(math.exp(math.log(tau_raw) * rel), 1e-2), 1e2)
            if not np.isfinite(tau):
                tau = 1.0

    fv_cal = np.minimum(tau * field_var, sill)
    fv0_cal = np.minimum(tau * field_var0, sill) if coords0 is not None else None

    # ---- opt+field coefficient covariance (default se_method="opt") ----
    # Recomputed once the calibrated per-point field SD s_f = sqrt(fv_cal) is
    # available, replacing the classic field-retained cluster-robust covariance
    # above. Updates the reported SEs and the coefficient-uncertainty term
    # coef_var of the predictive SE.
    if robust_se and se_method == "opt" and n_bid > 0 and b_field is not None:
        try:
            V, _G = optfield_SE(y=y_arr, X=x_mat, beta=beta_int_vec, field=b_field,
                                s_f=np.sqrt(fv_cal), offset=None,
                                family=GaussianIdentity(), coords=coords_arr,
                                bands=np.asarray(bands))
            if np.all(np.isfinite(np.diag(V))) and np.all(np.diag(V) > 0):
                beta_int_vmat = V
                beta_int_se = np.sqrt(np.maximum(np.diag(V), 0))
                beta_summary["coef_se"] = beta_int_se
                beta_summary["lower_95CI"] = beta_int_vec - 1.96 * beta_int_se
                beta_summary["upper_95CI"] = beta_int_vec + 1.96 * beta_int_se
                coef_var = ((x_mat @ beta_int_vmat) * x_mat).sum(axis=1)
                if coords0 is not None:
                    coef_var0 = ((x0_full @ beta_int_vmat) * x0_full).sum(axis=1)
        except Exception:
            pass

    # per-scale Z_sd / Z0_sd on the pv/tau/sill footing (used by sp_scalewise)
    with np.errstate(divide="ignore", invalid="ignore"):
        sf_pt = np.sqrt(np.where(field_var > 0, fv_cal / field_var, 1.0))
    Z_sd = Z_pv * sf_pt[:, None]
    if coords0 is not None:
        with np.errstate(divide="ignore", invalid="ignore"):
            sf0_pt = np.sqrt(np.where(field_var0 > 0, fv0_cal / field_var0, 1.0))
        Z0_sd = Z0_pv * sf0_pt[:, None]

    # ---- predictive quantiles ----
    use_cqr = a_run and ("qmat" in a_mod)
    if use_cqr:
        Qtot = total_qmat(pred, np.sqrt(coef_var + fv_cal),
                          a_mod["qmat"], a_mod["qlevels"], _QLEV_OUT)
        off = None
        if idt is not None and len(idt) < n and pred_hv is not None:
            val = np.setdiff1d(np.arange(n), idt)
            Qcal = Qtot[val] - pred[val, None] + pred_hv[val, None]
            off = cqr_offsets(Qcal, y_arr[val], _QLEV_OUT)
            Qtot = apply_cqr(Qtot, _QLEV_OUT, off)
        pred_q = {f"q{q}": Qtot[:, k] for k, q in enumerate(_QLEV_OUT)}
        pred_sd = (Qtot[:, -1] - Qtot[:, 0]) / (2 * qn_hi)
        pred0_q = None
        if coords0 is not None:
            Qtot0 = total_qmat(pred0, np.sqrt(coef_var0 + fv0_cal),
                               a_mod["qmat0"], a_mod["qlevels"], _QLEV_OUT)
            if off is not None:
                Qtot0 = apply_cqr(Qtot0, _QLEV_OUT, off)
            pred0_q = {f"q{q}": Qtot0[:, k] for k, q in enumerate(_QLEV_OUT)}
            pred0_sd = (Qtot0[:, -1] - Qtot0[:, 0]) / (2 * qn_hi)
    else:
        pred_sd = np.sqrt(coef_var + fv_cal + a_pred_v)
        pred_q = {f"q{q}": pred + pred_sd * float(norm.ppf(q)) for q in _QLEV_OUT}
        pred0_q = None
        if coords0 is not None:
            pred0_sd = np.sqrt(coef_var0 + fv0_cal + a_pred0_v)
            pred0_q = {f"q{q}": pred0 + pred0_sd * float(norm.ppf(q)) for q in _QLEV_OUT}

    pred_dict = {"pred": pred, "pred_sd": pred_sd}
    pred0_dict = None
    if coords0 is not None:
        pred0_dict = {"pred": pred0, "pred_sd": pred0_sd}

    Z_out = Z if n_bid > 0 else None
    Z_sd_out = Z_sd if n_bid > 0 else None
    Z0_out = Z0 if (n_bid > 0 and coords0 is not None) else None
    Z0_sd_out = Z0_sd if (n_bid > 0 and coords0 is not None) else None

    # ---- standard deviations of model elements ----
    resid_sd = float(np.std(y_arr - pred, ddof=1))
    sd_summary = [("xb", float(np.std(x_mat @ beta_int_vec, ddof=1)))]
    if Z_out is not None:
        for k, sc in enumerate(bands_scale):
            sd_summary.append((f"spatial_scale{int(sc)+1}", float(np.std(Z_out[:, k], ddof=1))))
    if a_run:
        sd_summary.append((f"additional learning ({add_learn})", float(np.std(a_pred, ddof=1))))
    sd_summary.append(("residuals", resid_sd))

    # ---- error statistics (holdout validation) ----
    ival = np.setdiff1d(np.arange(n), mod_hv.id_train)
    r2 = rmse = mae = float("nan")
    if ival.size >= 2 and pred_hv is not None:
        r2 = float(np.corrcoef(y_arr[ival], pred_hv[ival])[0, 1] ** 2)
        rmse = math.sqrt(mod_hv.sse_hv / ival.size)
        mae = float(np.mean(np.abs(y_arr[ival] - pred_hv[ival])))
    e_summary = [
        ("validation_R2", r2),
        ("validation_RMSE", rmse),
        ("validation_MAE", mae),
    ]

    other = {
        "n": n, "n0": n0, "nx": nx, "y": y, "x": x_mat, "x0": x0_full, "VCmat": VCmat,
        "coords": coords_arr, "coords0": coords0, "vpar": vpar, "xx_inv": xx_inv,
        "a_mod": a_mod, "pred_pre": pred_pre, "sse_hv": mod_hv.sse_hv, "tau": tau,
        "Z_pv": Z_pv, "Z0_pv": Z0_pv,
    }
    result = CFLM(
        beta=beta_summary, sd_summary=sd_summary, e_summary=e_summary,
        pred=pred_dict, pred0=pred0_dict, pred_q=pred_q, pred0_q=pred0_q,
        bands=np.asarray(bands) if bands is not None else None,
        Z=Z_out, Z_sd=Z_sd_out, Z0=Z0_out, Z0_sd=Z0_sd_out, other=other,
    )

    # ---- observation (data-distribution) predictive (default se_type) ----
    if se_type == "prediction":
        try:
            ob = obs_predict(
                family=GaussianIdentity(), y=y_arr, hvp=pred_hv,
                id_train=mod_hv.id_train,
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
