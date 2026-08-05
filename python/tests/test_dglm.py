"""Tests for the coarse-to-fine dynamic (space-time) GLMM (cf_dglm[_hv])."""
from __future__ import annotations

import warnings

import numpy as np
import pytest

import spCF

warnings.filterwarnings("ignore")

_QKEYS = ["q0.005", "q0.025", "q0.05", "q0.1", "q0.2", "q0.3", "q0.4", "q0.5",
          "q0.6", "q0.7", "q0.8", "q0.9", "q0.95", "q0.975", "q0.995"]


def _panel(ns=50, nt=5, seed=1):
    rng = np.random.default_rng(seed)
    sites = rng.uniform(0, 10, size=(ns, 2))
    K = 10
    cen = rng.uniform(0, 10, size=(K, 2))
    Dc = np.sqrt(((cen[:, None, :] - cen[None, :, :]) ** 2).sum(-1))
    Wc = np.exp(-Dc / 3.0)
    Wc = Wc / np.sqrt((Wc ** 2).sum(1, keepdims=True))
    a = np.zeros((nt, K))
    a[0] = Wc @ rng.normal(size=K)
    for t in range(1, nt):
        a[t] = 0.7 * a[t - 1] + Wc @ rng.normal(size=K)
    Dm = np.sqrt(((sites[:, None, :] - cen[None, :, :]) ** 2).sum(-1))
    Wp = np.exp(-Dm / 3.0)
    Wp = Wp / Wp.sum(1, keepdims=True)
    coords = np.repeat(sites, nt, axis=0)
    time = np.tile(np.arange(1, nt + 1), ns)
    site_idx = np.repeat(np.arange(ns), nt)
    field = (Wp[site_idx] * a[time - 1]).sum(1)
    x1 = rng.normal(size=ns * nt)
    x2 = rng.normal(size=ns * nt)
    return coords, time, x1, x2, field, rng


def make_gauss(seed=1):
    coords, time, x1, x2, field, rng = _panel(seed=seed)
    y = 1.0 + 1.5 * x1 - 0.5 * x2 + field + rng.normal(0, 0.4, size=x1.size)
    return y, np.column_stack([x1, x2]), coords, time


def make_poisson(seed=2):
    coords, time, x1, x2, field, rng = _panel(seed=seed)
    y = rng.poisson(np.exp(0.3 + 0.3 * x1 + 0.5 * field)).astype(float)
    return y, np.column_stack([x1, x2]), coords, time


class TestCFDGLM:
    def test_gaussian_basic(self):
        y, x, coords, time = make_gauss()
        mhv = spCF.cf_dglm_hv(y=y, x=x, coords=coords, time=time,
                              family=spCF.families.gaussian(), seed=1234, verbose=False)
        m = spCF.cf_dglm(y=y, x=x, coords=coords, time=time, mod_hv=mhv, verbose=False)
        assert m.pred["pred"].shape == (y.size,)
        assert np.all(m.pred["pred_sd"] >= 0)
        # Gaussian-identity final GLM has an intercept -> mean-calibrated.
        assert abs(m.pred["pred"].mean() - y.mean()) < 1e-6
        # coefficient recovery
        assert abs(m.beta["coef"][1] - 1.5) < 0.4
        assert list(m.pred_q.keys()) == _QKEYS

    def test_poisson_calibration_and_quantiles(self):
        y, x, coords, time = make_poisson()
        mhv = spCF.cf_dglm_hv(y=y, x=x, coords=coords, time=time,
                              family=spCF.families.poisson(), seed=1234, verbose=False)
        m = spCF.cf_dglm(y=y, x=x, coords=coords, time=time, mod_hv=mhv, verbose=False)
        assert np.all(m.pred["pred"] >= 0)
        assert abs(m.pred["pred"].mean() - y.mean()) < 1e-4     # Poisson MLE calibration
        Q = np.column_stack([m.pred_q[k] for k in _QKEYS])
        assert np.all(Q >= 0)
        assert np.all(np.diff(Q, axis=1) >= -1e-6)

    def test_ar1_and_tau(self):
        y, x, coords, time = make_gauss()
        mhv = spCF.cf_dglm_hv(y=y, x=x, coords=coords, time=time,
                              family=spCF.families.gaussian(), seed=1234, verbose=False)
        assert -0.999 <= mhv.other["rho"] <= 0.999
        assert mhv.other["Q"] > 0
        m = spCF.cf_dglm(y=y, x=x, coords=coords, time=time, mod_hv=mhv, verbose=False)
        assert 1e-2 <= m.other["tau"] <= 1e2

    def test_e_summary_fields(self):
        y, x, coords, time = make_gauss()
        mhv = spCF.cf_dglm_hv(y=y, x=x, coords=coords, time=time,
                              family=spCF.families.gaussian(), seed=1234, verbose=False)
        m = spCF.cf_dglm(y=y, x=x, coords=coords, time=time, mod_hv=mhv, verbose=False)
        assert [n for n, _ in m.e_summary] == ["validation_Pseudo-R2", "validation_RMSE", "validation_MAE"]
        assert [n for n, _ in mhv.e_summary] == ["validation_Pseudo-R2", "validation_RMSE", "validation_MAE"]

    def test_prediction_sites(self):
        y, x, coords, time = make_gauss()
        rng = np.random.default_rng(3)
        n0 = 40
        coords0 = rng.uniform(0, 10, size=(n0, 2))
        time0 = rng.integers(1, 6, size=n0)
        x0 = rng.normal(size=(n0, 2))
        mhv = spCF.cf_dglm_hv(y=y, x=x, coords=coords, time=time,
                              family=spCF.families.gaussian(), seed=1234, verbose=False)
        m = spCF.cf_dglm(y=y, x=x, coords=coords, time=time,
                         x0=x0, coords0=coords0, time0=time0, mod_hv=mhv, verbose=False)
        assert m.pred0["pred"].shape == (n0,)
        assert np.all(m.pred0["pred_sd"] >= 0)
        assert m.pred0_q["q0.5"].shape == (n0,)

    def test_robust_se_toggle(self):
        y, x, coords, time = make_gauss()
        mhv = spCF.cf_dglm_hv(y=y, x=x, coords=coords, time=time,
                              family=spCF.families.gaussian(), seed=1234, verbose=False)
        m_r = spCF.cf_dglm(y=y, x=x, coords=coords, time=time, mod_hv=mhv,
                           robust_se=True, verbose=False)
        m_n = spCF.cf_dglm(y=y, x=x, coords=coords, time=time, mod_hv=mhv,
                           robust_se=False, verbose=False)
        assert np.all(m_r.beta["coef_se"] > 0) and np.all(m_n.beta["coef_se"] > 0)
        assert np.allclose(m_r.beta["coef"], m_n.beta["coef"])

    def test_repr(self):
        y, x, coords, time = make_gauss()
        mhv = spCF.cf_dglm_hv(y=y, x=x, coords=coords, time=time,
                              family=spCF.families.gaussian(), seed=1234, verbose=False)
        m = spCF.cf_dglm(y=y, x=x, coords=coords, time=time, mod_hv=mhv, verbose=False)
        assert "cf_dglm" in repr(mhv)
        assert "Coefficients" in repr(m)


class TestSpScalewiseDGLM:
    def test_time_range_average(self):
        y, x, coords, time = make_gauss()
        mhv = spCF.cf_dglm_hv(y=y, x=x, coords=coords, time=time,
                              family=spCF.families.gaussian(), seed=1234, verbose=False)
        m = spCF.cf_dglm(y=y, x=x, coords=coords, time=time, mod_hv=mhv, verbose=False)
        out = spCF.sp_scalewise(m, bw_range=(0, np.inf), time_range=(3, 3))
        p = out["pred"]
        assert "px" in p and "py" in p and "n_time" in p
        # 50 unique locations, each observed once at time 3
        assert p["pred"].shape[0] == 50
        assert np.all(p["n_time"] == 1)

    def test_half_open_partition(self):
        y, x, coords, time = make_gauss()
        mhv = spCF.cf_dglm_hv(y=y, x=x, coords=coords, time=time,
                              family=spCF.families.gaussian(), seed=1234, verbose=False)
        m = spCF.cf_dglm(y=y, x=x, coords=coords, time=time, mod_hv=mhv, verbose=False)
        if m.bands is None or len(m.bands) < 2:
            pytest.skip("not enough bands")
        thr = float(np.median(m.bands))
        # full = small [0,thr) + large [thr, inf); no scale double counted
        full = spCF.sp_scalewise(m, bw_range=(0, np.inf))["pred"]["pred"]
        small = spCF.sp_scalewise(m, bw_range=(0, thr))["pred"]["pred"]
        large = spCF.sp_scalewise(m, bw_range=(thr, np.inf))["pred"]["pred"]
        assert np.allclose(full, small + large, atol=1e-9)


def test_time_range_warning_non_dglm():
    rng = np.random.default_rng(0)
    n = 150
    coords = rng.uniform(0, 10, size=(n, 2))
    x = rng.normal(size=(n, 2))
    z = np.sin(coords[:, 0] / 2) * np.cos(coords[:, 1] / 2)
    y = 1 + 2 * x[:, 0] + 1.5 * z + rng.normal(0, 0.3, n)
    mhv = spCF.cf_lm_hv(y=y, x=x, coords=coords, seed=42, verbose=False)
    m = spCF.cf_lm(y=y, x=x, coords=coords, mod_hv=mhv, verbose=False)
    with pytest.warns(UserWarning, match="time_range is ignored"):
        spCF.sp_scalewise(m, bw_range=(0, np.inf), time_range=(1, 2))
