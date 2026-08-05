"""Tests for the statistical machinery added in spCF 0.1.2.

Covers predictive quantiles (pred_q/pred0_q), holdout tau calibration,
cluster-robust standard errors (robust_se), and the validation_MAE metric,
for both cf_lm and cf_glm.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

import spCF

warnings.filterwarnings("ignore")

_QKEYS = ["q0.005", "q0.025", "q0.05", "q0.1", "q0.2", "q0.3", "q0.4", "q0.5",
          "q0.6", "q0.7", "q0.8", "q0.9", "q0.95", "q0.975", "q0.995"]


def _gauss(n=250, seed=1):
    rng = np.random.default_rng(seed)
    coords = rng.uniform(0, 10, size=(n, 2))
    x = rng.normal(size=(n, 2))
    z = np.sin(coords[:, 0] / 2) * np.cos(coords[:, 1] / 2)
    y = 1.0 + 2.0 * x[:, 0] - 0.5 * x[:, 1] + 1.5 * z + rng.normal(0, 0.3, n)
    return y, x, coords


def _poisson(n=220, seed=2):
    rng = np.random.default_rng(seed)
    coords = rng.uniform(0, 10, size=(n, 2))
    x1 = rng.normal(size=n)
    z = np.sin(coords[:, 0] / 2) * np.cos(coords[:, 1] / 2)
    y = rng.poisson(np.exp(0.5 + 0.3 * x1 + 0.5 * z))
    return y, x1, coords


class TestLMFeatures:
    def test_pred_q_keys_and_monotone(self):
        y, x, coords = _gauss()
        mhv = spCF.cf_lm_hv(y=y, x=x, coords=coords, seed=42, verbose=False)
        m = spCF.cf_lm(y=y, x=x, coords=coords, mod_hv=mhv, verbose=False)
        assert list(m.pred_q.keys()) == _QKEYS
        Q = np.column_stack([m.pred_q[k] for k in _QKEYS])
        # Quantiles non-decreasing across levels for every point.
        assert np.all(np.diff(Q, axis=1) >= -1e-9)

    def test_pred0_q_present_with_sites(self):
        y, x, coords = _gauss()
        rng = np.random.default_rng(9)
        coords0 = rng.uniform(0, 10, size=(40, 2))
        x0 = rng.normal(size=(40, 2))
        mhv = spCF.cf_lm_hv(y=y, x=x, coords=coords, seed=7, verbose=False)
        m = spCF.cf_lm(y=y, x=x, coords=coords, x0=x0, coords0=coords0,
                       mod_hv=mhv, verbose=False)
        assert m.pred0_q is not None
        assert m.pred0_q["q0.5"].shape == (40,)

    def test_validation_mae_present(self):
        y, x, coords = _gauss()
        mhv = spCF.cf_lm_hv(y=y, x=x, coords=coords, seed=42, verbose=False)
        m = spCF.cf_lm(y=y, x=x, coords=coords, mod_hv=mhv, verbose=False)
        names = [nm for nm, _ in m.e_summary]
        assert names == ["validation_R2", "validation_RMSE", "validation_MAE"]

    def test_tau_stored(self):
        y, x, coords = _gauss()
        mhv = spCF.cf_lm_hv(y=y, x=x, coords=coords, seed=42, verbose=False)
        m = spCF.cf_lm(y=y, x=x, coords=coords, mod_hv=mhv, verbose=False)
        assert 1e-2 <= m.other["tau"] <= 1e2
        assert "Z_pv" in m.other

    def test_robust_se_toggle(self):
        y, x, coords = _gauss()
        mhv = spCF.cf_lm_hv(y=y, x=x, coords=coords, seed=42, verbose=False)
        m_rob = spCF.cf_lm(y=y, x=x, coords=coords, mod_hv=mhv,
                           robust_se=True, verbose=False)
        m_naive = spCF.cf_lm(y=y, x=x, coords=coords, mod_hv=mhv,
                             robust_se=False, verbose=False)
        assert np.all(m_rob.beta["coef_se"] > 0)
        assert np.all(m_naive.beta["coef_se"] > 0)
        # Point estimates identical regardless of SE method.
        assert np.allclose(m_rob.beta["coef"], m_naive.beta["coef"])


class TestGLMFeatures:
    def test_poisson_pred_q_nonneg_monotone(self):
        y, x1, coords = _poisson()
        mhv = spCF.cf_glm_hv(y=y, x=x1, coords=coords,
                             family=spCF.families.poisson(), seed=1234, verbose=False)
        m = spCF.cf_glm(y=y, x=x1, coords=coords, mod_hv=mhv, verbose=False)
        assert list(m.pred_q.keys()) == _QKEYS
        Q = np.column_stack([m.pred_q[k] for k in _QKEYS])
        assert np.all(Q >= 0)                       # Poisson response scale
        assert np.all(np.diff(Q, axis=1) >= -1e-9)

    def test_validation_mae_present(self):
        y, x1, coords = _poisson()
        mhv = spCF.cf_glm_hv(y=y, x=x1, coords=coords,
                             family=spCF.families.poisson(), seed=1234, verbose=False)
        m = spCF.cf_glm(y=y, x=x1, coords=coords, mod_hv=mhv, verbose=False)
        names = [nm for nm, _ in m.e_summary]
        assert names == ["validation_Pseudo-R2", "validation_RMSE", "validation_MAE"]

    def test_robust_se_toggle(self):
        y, x1, coords = _poisson()
        mhv = spCF.cf_glm_hv(y=y, x=x1, coords=coords,
                             family=spCF.families.poisson(), seed=1234, verbose=False)
        m_rob = spCF.cf_glm(y=y, x=x1, coords=coords, mod_hv=mhv,
                            robust_se=True, verbose=False)
        m_naive = spCF.cf_glm(y=y, x=x1, coords=coords, mod_hv=mhv,
                              robust_se=False, verbose=False)
        assert np.all(m_rob.beta["coef_se"] > 0)
        assert np.all(m_naive.beta["coef_se"] > 0)

    def test_tau_stored(self):
        y, x1, coords = _poisson()
        mhv = spCF.cf_glm_hv(y=y, x=x1, coords=coords,
                             family=spCF.families.poisson(), seed=1234, verbose=False)
        m = spCF.cf_glm(y=y, x=x1, coords=coords, mod_hv=mhv, verbose=False)
        assert 1e-2 <= m.other["tau"] <= 1e2


class TestSelIdAlignment:
    """Regression guard: sel_id_list must stay band-indexed (one entry per
    processed band, plus the leading placeholder), so that a committed band
    following a skipped ("no improvement") band gets the correct knots.
    """

    def test_lm_sel_id_band_indexed(self):
        y, x, coords = _gauss(n=250, seed=1)
        mhv = spCF.cf_lm_hv(y=y, x=x, coords=coords, seed=42, verbose=False)
        VC = mhv.other["VCmat"]
        sel = mhv.other["sel_id_list"]
        assert len(sel) == VC.shape[0] + 1          # leading None + one per band
        # If gaps exist, each committed band index must map into sel_id_list.
        committed = np.where(VC[:, 0] == 1)[0]
        for i in committed:
            assert i + 1 < len(sel)
        # Runs and is finite (exercises the post-gap knot lookup).
        m = spCF.cf_lm(y=y, x=x, coords=coords, mod_hv=mhv, verbose=False)
        assert np.all(np.isfinite(m.pred["pred"]))

    def test_glm_sel_id_band_indexed(self):
        y, x1, coords = _poisson(n=220, seed=2)
        mhv = spCF.cf_glm_hv(y=y, x=x1, coords=coords,
                             family=spCF.families.poisson(), seed=1234, verbose=False)
        VC = mhv.other["VCmat"]
        assert len(mhv.other["sel_id_list"]) == VC.shape[0] + 1
        m = spCF.cf_glm(y=y, x=x1, coords=coords, mod_hv=mhv, verbose=False)
        assert np.all(np.isfinite(m.pred["pred"]))
