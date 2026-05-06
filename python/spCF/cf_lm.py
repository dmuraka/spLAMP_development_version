"""Coarse-to-fine spatial modeling for Gaussian response."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.optimize import minimize_scalar

from ._neighbors import build_tree
from ._utils import (
    add_mod_lm,
    bopt_core,
    initial_fun,
    lwr,
    sample_from_qrf,
)


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
    coords_uni = np.unique(coords_arr, axis=0)  # sorted; matches R unique() row-set
    # We need order-preserving unique to match initial_fun mapping. Use the
    # same helper as initial_fun:
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

    coords_old = None
    sel_id_list: list = [None]
    b_old = None
    bands: list = []
    SSE: list = []
    SSE_name: list = []

    # Build the BallTree on coords once; reuse it across all band iterations.
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
            coords_old = lmod["coords_cent"]
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
            beta = beta + beta_int_add.ravel()  # broadcast adds along columns
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

        # NLOPT_LN_BOBYQA in R ≈ scipy bounded scalar minimizer.
        # The R code uses x0=0; we mirror that by using BOBYQA-equivalent.
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
    sd_method: str = "qrf",
    verbose: bool = True,
) -> CFLM:
    """Predict / regress with a trained CF-LM.

    Parameters
    ----------
    sd_method : {"qrf", "tree_var", "residual"}
        Predictive-SD estimation when ``add_learn="rf"`` was used at training.
        Default ``"qrf"`` mirrors R's quantile-RF approach (200 draws from a
        201-quantile prediction per point) and captures both model and
        observation-noise variance — required for honest predictive intervals
        on heteroskedastic data.  ``"tree_var"`` is a cheap heteroskedastic
        approximation (variance across the RF trees only); good when noise is
        homoskedastic.  ``"residual"`` broadcasts a single ``Var(resid)`` to
        every point (homoskedastic, biased ~2× high) and is kept only for
        backward compatibility.
    """
    """Spatial regression / prediction with the trained CF-LM."""
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
    coords_arr = init.coords
    pred = init.pred
    resid = init.resid.copy()
    x_mat = init.x
    xname = init.xname
    n = init.n
    nx = init.nx
    id_train = init.id_train

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
        Z0 = np.zeros((n0, len(bands) if bands is not None else 0))
        Z0_sd = np.zeros_like(Z0)
    else:
        n0 = None
        x0_full = None
        pred0 = None
        Z0 = Z0_sd = None

    if verbose:
        print("--- Learning multi-scale spatial process ---")

    bands_scale = np.where(VCmat[:, 0] == 1)[0] if VCmat.size else np.array([], dtype=np.int64)
    b_old = None
    Z = np.zeros((n, len(bands) if bands is not None else 0))
    Z_sd = np.zeros_like(Z)

    # One BallTree per coords / coords0; reused across all band iterations.
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
                resid = np.asarray(y, dtype=float) - pred
                beta_int_add = (xx_inv @ x_mat.T @ resid).reshape(-1, 1)
                pred_int_add = (x_mat @ beta_int_add).ravel()
                pred = pred + pred_int_add
                resid = resid - pred_int_add

                ii = int(np.where(bands_scale == i)[0][0])
                beta_add_m = beta_add.mean(axis=0)
                Z[:, ii] = beta_add[:, 0] - beta_add_m[0]
                Z_sd[:, ii] = np.sqrt(np.maximum(beta_v_add[:, 0], 0))
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
                comment = ""
            else:
                comment = " no improvement (skipped)"

            if verbose:
                tag = " " * (1 if (i + 1) >= 10 else 2)
                print(f"  Scale{tag}{i+1} (bandwidth:{bands_all[i]:.7g}){comment}")

    pred_pre = (x_mat * np.tile(np.tile(beta_int.ravel(), (n, 1))[0], (n, 1))).sum(axis=1)
    pred_pre = (x_mat @ beta_int).ravel()
    sig_pre = float(np.sum((np.asarray(y) - pred_pre) ** 2) / max(n - nx, 1))
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

    # Tuning -- weighted spatial-process correction
    beta = np.tile(beta_int_vec, (n, 1))
    if coords0 is not None:
        beta0 = np.tile(beta_int_vec, (n0, 1))

    n_bid = len(bands) if bands is not None else 0
    if n_bid > 0:
        vpar_coef = bopt_core(vpar[1], bands=np.asarray(bands), Z=Z,
                              beta_int=beta_int, nx=nx, x=x_mat,
                              y=np.asarray(y, dtype=float), n_bid=n_bid,
                              id_train=None)["vpar"][0, 0]
        w_0 = np.exp(-vpar[1] / np.asarray(bands))
        w = float(vpar_coef) * w_0 / w_0[0]
        b = Z @ w
        beta[:, 0] = beta[:, 0] + b
        if coords0 is not None:
            b0 = Z0 @ w
            beta0[:, 0] = beta0[:, 0] + b0

    a_pred = a_pred0 = a_pred_v = a_pred0_v = 0.0
    a_mod = {"add_learn": "none"}
    if a_run:
        a_mod = add_mod_lm(
            add_learn=add_learn, train=False, resid=resid,
            x=x_mat, coords=coords_arr,
            x0=x0_full, coords0=coords0,
            nx=nx, xname=xname, a_par=a_par,
            sd_method=sd_method,
        )
        a_pred = a_mod["pred"]
        a_pred0 = a_mod["pred0"] if coords0 is not None else 0.0
        a_pred_v = a_mod["pred_v"]
        a_pred0_v = a_mod["pred0_v"] if coords0 is not None else 0.0

    pred = (x_mat * beta).sum(axis=1) + a_pred
    pred_sd = np.sqrt(((x_mat @ beta_int_vmat) * x_mat).sum(axis=1) + (Z_sd ** 2).sum(axis=1) + a_pred_v)
    pred_dict = {"pred": pred, "pred_sd": pred_sd}
    pred0_dict = None
    if coords0 is not None:
        pred0_arr = (x0_full * beta0).sum(axis=1) + a_pred0
        pred0_sd = np.sqrt(((x0_full @ beta_int_vmat) * x0_full).sum(axis=1) + (Z0_sd ** 2).sum(axis=1) + a_pred0_v)
        pred0_dict = {"pred": pred0_arr, "pred_sd": pred0_sd}

    Z_out = Z if (bands is not None and len(bands) > 0) else None
    Z_sd_out = Z_sd if (bands is not None and len(bands) > 0) else None
    Z0_out = Z0 if (bands is not None and len(bands) > 0 and coords0 is not None) else None
    Z0_sd_out = Z0_sd if (bands is not None and len(bands) > 0 and coords0 is not None) else None

    resid_sd = float(np.std(np.asarray(y) - pred, ddof=1))
    sd_summary = [("xb", float(np.std(x_mat @ beta_int_vec, ddof=1)))]
    if Z_out is not None:
        for k, sc in enumerate(bands_scale):
            sd_summary.append((f"spatial_scale{int(sc)+1}", float(np.std(Z_out[:, k], ddof=1))))
    if a_run:
        sd_summary.append((f"additional learning ({add_learn})", float(np.std(a_pred, ddof=1))))
    sd_summary.append(("residuals", resid_sd))

    not_train = np.ones(n, dtype=bool)
    not_train[mod_hv.id_train] = False
    if not_train.sum() > 0:
        r2 = float(np.corrcoef(np.asarray(y)[not_train], pred[not_train])[0, 1] ** 2)
    else:
        r2 = float("nan")
    rmse = math.sqrt(mod_hv.sse_hv / max(n - len(mod_hv.id_train), 1))

    e_summary = [
        ("validation_R2", r2),
        ("validation_RMSE", rmse),
        ("residual_SD", resid_sd),
    ]

    other = {
        "n": n, "n0": n0, "nx": nx, "y": y, "x": x_mat, "x0": x0_full, "VCmat": VCmat,
        "coords": coords_arr, "coords0": coords0, "vpar": vpar, "xx_inv": xx_inv,
        "a_mod": a_mod, "pred_pre": pred_pre, "sse_hv": mod_hv.sse_hv,
    }
    return CFLM(
        beta=beta_summary, sd_summary=sd_summary, e_summary=e_summary,
        pred=pred_dict, pred0=pred0_dict, bands=np.asarray(bands) if bands is not None else None,
        Z=Z_out, Z_sd=Z_sd_out, Z0=Z0_out, Z0_sd=Z0_sd_out, other=other,
    )
