"""Observation (data-distribution) predictive with holdout calibration.

Python port of ``R/internal_prediction_se.R`` (``.spcf_obs_predict`` /
``.spcf_apply_obs``). Converts the SIGNAL predictive (mean uncertainty, carried
in ``pred_q``) into the OBSERVATION predictive for a new data point, and
calibrates it on the ``cf_*_hv`` holdout (out-of-fold) samples so that it is
verifiable on data:

* gaussian : ``N(mean, signal_var + sigma^2)`` with split-conformal SD scaling
* poisson  : negative-binomial count predictive (Poisson x lognormal-lambda),
             holdout scaling of the mean-uncertainty component
* binomial : ``Bernoulli(p_calibrated)``; ``p`` from holdout temperature
             scaling, ``pred_sd = sqrt(p(1-p))`` (interval coverage is
             degenerate for binary, so calibration is on the probability)

Works purely from ``(pred, pred_q)`` + the family name + the holdout out-of-fold
predictions ``hvp`` (and ``id_train``), so it is shared unchanged by
cf_lm / cf_glm / cf_dglm and does not touch their internal variance
construction. Returns replacement ``pred_sd`` / ``pred_q`` (and, for binomial,
a calibrated ``pred``); ``apply_obs`` keeps the signal versions in separate
``other['*_signal']`` fields so nothing is lost (non-destructive).
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm, nbinom
from scipy.special import expit, logit
from scipy.optimize import minimize_scalar

# Same 15 quantile levels used across cf_lm / cf_glm / cf_dglm.
_QS = np.array([0.005, 0.025, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,
                0.9, 0.95, 0.975, 0.995])
_ZNM = [f"q{q}" for q in _QS]


def _fam_name(family) -> str:
    return (getattr(family, "name", None)
            or getattr(family, "family_name", None) or "")


def _clp(p):
    return np.minimum(np.maximum(p, 1e-8), 1 - 1e-8)


def _link_of(fam):
    if fam == "gaussian":
        return lambda p: np.asarray(p, dtype=float)
    if fam == "poisson":
        return lambda p: np.log(np.maximum(np.asarray(p, dtype=float), 1e-8))
    if fam == "binomial":
        return lambda p: logit(_clp(np.asarray(p, dtype=float)))
    return None


def _slink(q, lk):
    """Recover the link-scale signal SD from the signal quantiles."""
    if q is None or "q0.975" not in q or "q0.025" not in q:
        return None
    return (lk(q["q0.975"]) - lk(q["q0.025"])) / (2 * 1.96)


def obs_predict(family, y, hvp, id_train, pred_in, predq_in,
                pred_out=None, predq_out=None):
    """Return a dict of replacement predictive fields, or ``None``.

    Parameters mirror ``.spcf_obs_predict``: ``hvp`` is the cf_*_hv out-of-fold
    prediction vector (``None`` -> in-sample fallback), ``id_train`` the training
    index array. ``pred_in``/``predq_in`` are the in-sample signal predictive
    mean and quantile dict; ``pred_out``/``predq_out`` the out-of-sample ones.
    """
    fam = _fam_name(family)
    lk = _link_of(fam)
    if lk is None:
        return None
    y = np.asarray(y, dtype=float)
    pred_in = np.asarray(pred_in, dtype=float)

    s_in = _slink(predq_in, lk)
    s_out = _slink(predq_out, lk)
    if s_in is None:
        return None
    s_in = np.asarray(s_in, dtype=float)
    if s_out is not None:
        s_out = np.asarray(s_out, dtype=float)

    n = pred_in.shape[0]
    if id_train is not None:
        val = np.setdiff1d(np.arange(n), np.asarray(id_train))
    else:
        val = np.arange(0)
    hvp_arr = None if hvp is None else np.asarray(hvp, dtype=float)
    haveval = (val.size >= 10 and hvp_arr is not None
               and np.all(np.isfinite(hvp_arr[val])))

    out = {"type": fam}

    if fam == "gaussian":
        if haveval:
            sig2 = float(np.mean((y[val] - hvp_arr[val]) ** 2))
        else:
            sig2 = float(np.mean((y - pred_in) ** 2))
        obs_in = np.sqrt(s_in ** 2 + sig2)
        obs_out = np.sqrt(s_out ** 2 + sig2) if s_out is not None else None
        cc = 1.0
        if haveval:
            sv = np.sqrt(s_in[val] ** 2 + sig2)
            cc = float(np.quantile(np.abs(y[val] - hvp_arr[val]) / sv, 0.95)) / 1.96
            if not np.isfinite(cc) or cc <= 0:
                cc = 1.0
        qn = norm.ppf(_QS)

        def mkq(mu, s):
            d = mu[:, None] + np.outer(cc * s, qn)
            return {nm: d[:, k] for k, nm in enumerate(_ZNM)}

        out["pred_sd"] = cc * obs_in
        out["pred_q"] = mkq(pred_in, obs_in)
        if obs_out is not None:
            out["pred0_sd"] = cc * obs_out
            out["pred0_q"] = mkq(np.asarray(pred_out, dtype=float), obs_out)
        out["calib"] = {"type": "gaussian_conformal", "sigma2": sig2, "scale": cc}

    elif fam == "poisson":
        cc = 1.0
        if haveval:
            sv = s_in[val]
            lamv = np.maximum(hvp_arr[val], 1e-8)

            def cov_at(c):
                sz = 1.0 / np.maximum((c * sv) ** 2, 1e-8)
                p = sz / (sz + lamv)
                lo = nbinom.ppf(0.025, sz, p)
                hi = nbinom.ppf(0.975, sz, p)
                return float(np.mean((y[val] >= lo) & (y[val] <= hi)))

            grid = np.arange(0.05, 1.5 + 1e-9, 0.05)
            covs = np.array([cov_at(c) for c in grid])
            cc = float(grid[int(np.argmin(np.abs(covs - 0.95)))])

        def nb_sd(mu, s):
            return np.sqrt(mu + (mu * cc * s) ** 2)

        def mkq(mu, s):
            sz = 1.0 / np.maximum((cc * s) ** 2, 1e-8)
            p = sz / (sz + mu)
            d = np.column_stack([nbinom.ppf(a, sz, p) for a in _QS])
            return {nm: d[:, k] for k, nm in enumerate(_ZNM)}

        out["pred_sd"] = nb_sd(pred_in, s_in)
        out["pred_q"] = mkq(pred_in, s_in)
        if s_out is not None:
            po = np.asarray(pred_out, dtype=float)
            out["pred0_sd"] = nb_sd(po, s_out)
            out["pred0_q"] = mkq(po, s_out)
        out["calib"] = {"type": "poisson_negbin", "scale": cc}

    elif fam == "binomial":
        Tt = 1.0
        if haveval:
            mv = logit(_clp(hvp_arr[val]))
            yv = y[val]

            def nll(T):
                pc = _clp(expit(mv / T))
                return -float(np.mean(yv * np.log(pc) + (1 - yv) * np.log(1 - pc)))

            try:
                Tt = float(minimize_scalar(nll, bounds=(0.3, 3.0),
                                           method="bounded").x)
            except Exception:
                Tt = 1.0

        def pcal(mu):
            return expit(logit(_clp(np.asarray(mu, dtype=float))) / Tt)

        p_in = pcal(pred_in)
        out["pred"] = p_in
        out["pred_sd"] = np.sqrt(p_in * (1 - p_in))
        if pred_out is not None:
            p_out = pcal(pred_out)
            out["pred0"] = p_out
            out["pred0_sd"] = np.sqrt(p_out * (1 - p_out))
        out["calib"] = {"type": "binomial_temperature", "temperature": Tt}
        out["binary"] = True
    else:
        return None
    return out


def apply_obs(res, ob):
    """Apply ``obs_predict`` output onto a cf_* result object in place.

    The signal versions are preserved under ``other['*_signal']`` (non
    destructive). ``res`` is a CFLM / CFGLM / CFDGLM dataclass exposing ``pred``,
    ``pred0``, ``pred_q``, ``pred0_q`` and ``other``. When ``ob is None`` the
    result is left as the signal (mean) predictive.
    """
    if ob is None:
        res.other["se_type"] = "mean"
        return res
    res.other["pred_signal"] = res.pred.get("pred")
    res.other["pred_q_signal"] = res.pred_q
    if res.pred0 is not None:
        res.other["pred0_signal"] = res.pred0.get("pred")
    if res.pred0_q is not None:
        res.other["pred0_q_signal"] = res.pred0_q
    # in-sample
    if ob.get("pred") is not None:
        res.pred["pred"] = ob["pred"]            # binomial: calibrated prob
    if ob.get("pred_sd") is not None:
        res.pred["pred_sd"] = ob["pred_sd"]
    if ob.get("pred_q") is not None:
        res.pred_q = ob["pred_q"]
    # out-of-sample
    if res.pred0 is not None:
        if ob.get("pred0") is not None:
            res.pred0["pred"] = ob["pred0"]
        if ob.get("pred0_sd") is not None:
            res.pred0["pred_sd"] = ob["pred0_sd"]
        if ob.get("pred0_q") is not None:
            res.pred0_q = ob["pred0_q"]
    res.other["se_type"] = "prediction"
    res.other["calibration"] = ob.get("calib")
    res.other["binary_pred"] = bool(ob.get("binary", False))
    return res
