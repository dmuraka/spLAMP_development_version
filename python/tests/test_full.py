"""Comprehensive test suite for the Python `spCF` package.

Covers every public entry point (cf_lm[+_hv], cf_glm[+_hv], sp_scalewise),
both kernels, both add_learn settings, every sd_method, every supported GLM
family, with and without prediction sites, plus repr() formatting.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

import spCF

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Synthetic-data helpers
# ---------------------------------------------------------------------------
def _spatial(coords):
    return np.sin(coords[:, 0] / 2) * np.cos(coords[:, 1] / 2)


def make_gauss(n=200, seed=0):
    rng = np.random.default_rng(seed)
    coords = rng.uniform(0, 10, size=(n, 2))
    x = rng.normal(size=(n, 2))
    z = _spatial(coords)
    y = 1.0 + 2.0 * x[:, 0] - 0.5 * x[:, 1] + 1.5 * z + rng.normal(0, 0.3, n)
    return y, x, coords


def make_nonlinear(n=300, seed=0):
    """Dataset with a nonlinear interaction so RF actually activates."""
    rng = np.random.default_rng(seed)
    coords = rng.uniform(0, 10, size=(n, 2))
    x = rng.normal(size=(n, 2))
    z = _spatial(coords)
    nonlin = (
        0.7 * np.tanh(x[:, 0]) * (x[:, 1] ** 2 - 1.0)
        + 1.5 * np.exp(-((coords[:, 0] - 7) ** 2 + (coords[:, 1] - 3) ** 2) / 1.5)
    )
    y = 1.0 + 2.0 * x[:, 0] - 0.5 * x[:, 1] + 1.5 * z + nonlin + rng.normal(0, 0.3, n)
    return y, x, coords


def make_poisson(n=200, seed=1):
    rng = np.random.default_rng(seed)
    coords = rng.uniform(0, 10, size=(n, 2))
    x1 = rng.normal(size=n)
    z = _spatial(coords)
    mu = np.exp(0.5 + 0.3 * x1 + 0.5 * z)
    y = rng.poisson(mu)
    return y, x1, coords


def make_binomial(n=200, seed=2):
    rng = np.random.default_rng(seed)
    coords = rng.uniform(0, 10, size=(n, 2))
    x1 = rng.normal(size=n)
    z = _spatial(coords)
    eta = -0.2 + 0.5 * x1 + 1.0 * z
    p = 1 / (1 + np.exp(-eta))
    y = rng.binomial(1, p).astype(float)
    return y, x1, coords


def make_gamma(n=200, seed=3):
    rng = np.random.default_rng(seed)
    coords = rng.uniform(0, 10, size=(n, 2))
    x1 = rng.normal(size=n)
    z = _spatial(coords)
    mu = np.exp(1.0 + 0.2 * x1 + 0.3 * z)  # positive
    shape = 5.0
    y = rng.gamma(shape=shape, scale=mu / shape)
    return y, x1, coords


# ---------------------------------------------------------------------------
# Core flow: cf_lm_hv → cf_lm
# ---------------------------------------------------------------------------
class TestCFLM:
    def test_basic(self):
        y, x, coords = make_gauss()
        mhv = spCF.cf_lm_hv(y=y, x=x, coords=coords, kernel="exp",
                            verbose=False, seed=42)
        m = spCF.cf_lm(y=y, x=x, coords=coords, mod_hv=mhv, verbose=False)
        assert m.pred["pred"].shape == (200,)
        assert np.all(m.pred["pred_sd"] >= 0)
        # Coefficient recovery within reasonable tolerance.
        coef = m.beta["coef"]
        assert abs(coef[0] - 1.0) < 0.5  # intercept
        assert abs(coef[1] - 2.0) < 0.3  # x1
        assert abs(coef[2] - (-0.5)) < 0.3  # x2

    def test_with_prediction_sites(self):
        y, x, coords = make_gauss(n=200)
        rng = np.random.default_rng(99)
        coords0 = rng.uniform(0, 10, size=(50, 2))
        x0 = rng.normal(size=(50, 2))
        mhv = spCF.cf_lm_hv(y=y, x=x, coords=coords, verbose=False, seed=7)
        m = spCF.cf_lm(y=y, x=x, coords=coords, x0=x0, coords0=coords0,
                       mod_hv=mhv, verbose=False)
        assert m.pred0["pred"].shape == (50,)
        assert m.pred0["pred_sd"].shape == (50,)
        assert np.all(m.pred0["pred_sd"] >= 0)

    def test_intercept_only(self):
        """No covariates besides the intercept."""
        y, _x, coords = make_gauss(n=150)
        mhv = spCF.cf_lm_hv(y=y, x=None, coords=coords, verbose=False, seed=3)
        m = spCF.cf_lm(y=y, x=None, coords=coords, mod_hv=mhv, verbose=False)
        assert m.pred["pred"].shape == (150,)
        assert len(m.beta["coef"]) == 1  # only intercept

    @pytest.mark.parametrize("kernel", ["exp", "gau"])
    def test_kernels(self, kernel):
        y, x, coords = make_gauss(n=150)
        mhv = spCF.cf_lm_hv(y=y, x=x, coords=coords, kernel=kernel,
                            verbose=False, seed=5)
        m = spCF.cf_lm(y=y, x=x, coords=coords, mod_hv=mhv, verbose=False)
        assert m.pred["pred"].shape == (150,)
        assert m.other["VCmat"].shape[1] == 3  # intercept + 2 covariates

    def test_explicit_id_train(self):
        y, x, coords = make_gauss(n=200)
        rng = np.random.default_rng(0)
        id_train = np.sort(rng.choice(200, 150, replace=False))
        mhv = spCF.cf_lm_hv(y=y, x=x, coords=coords,
                            id_train=id_train, verbose=False)
        m = spCF.cf_lm(y=y, x=x, coords=coords, mod_hv=mhv, verbose=False)
        assert np.array_equal(mhv.id_train, id_train)
        assert m.pred["pred"].shape == (200,)


class TestAddLearnRF:
    """add_learn='rf' should activate on nonlinear data."""

    @pytest.fixture(scope="class")
    def fitted_rf(self):
        y, x, coords = make_nonlinear(n=400)
        mhv = spCF.cf_lm_hv(y=y, x=x, coords=coords,
                            add_learn="rf", verbose=False, seed=42)
        return y, x, coords, mhv

    def test_rf_activated(self, fitted_rf):
        _, _, _, mhv = fitted_rf
        a = mhv.other["a_mod0"]
        assert a["a_run"] is True
        assert a["a_par"] is not None

    @pytest.mark.parametrize("sd_method", ["qrf", "tree_var", "residual"])
    def test_sd_methods(self, fitted_rf, sd_method):
        y, x, coords, mhv = fitted_rf
        m = spCF.cf_lm(y=y, x=x, coords=coords, mod_hv=mhv,
                       sd_method=sd_method, verbose=False)
        assert m.pred["pred"].shape == (400,)
        assert np.all(m.pred["pred_sd"] >= 0)
        # Must be non-degenerate.
        assert m.pred["pred_sd"].std() > 0 or sd_method == "residual"

    def test_invalid_sd_method(self, fitted_rf):
        y, x, coords, mhv = fitted_rf
        with pytest.raises(ValueError, match="sd_method"):
            spCF.cf_lm(y=y, x=x, coords=coords, mod_hv=mhv,
                      sd_method="bogus", verbose=False)


class TestSpScalewise:
    @pytest.fixture(scope="class")
    def fitted(self):
        y, x, coords = make_gauss(n=300)
        rng = np.random.default_rng(0)
        coords0 = rng.uniform(0, 10, size=(50, 2))
        x0 = rng.normal(size=(50, 2))
        mhv = spCF.cf_lm_hv(y=y, x=x, coords=coords, verbose=False, seed=42)
        m = spCF.cf_lm(y=y, x=x, coords=coords, x0=x0, coords0=coords0,
                       mod_hv=mhv, verbose=False)
        return m

    def test_full_range(self, fitted):
        out = spCF.sp_scalewise(fitted, bw_range=(0, np.inf))
        assert out["pred"]["pred"].shape == (300,)
        assert out["pred0"]["pred"].shape == (50,)

    def test_split_range(self, fitted):
        if fitted.bands is None or len(fitted.bands) < 2:
            pytest.skip("not enough bands")
        # Threshold strictly between two bands so neither side double-counts.
        sb = np.sort(fitted.bands)
        k = len(sb) // 2
        thresh_lo = float((sb[k - 1] + sb[k]) / 2)
        large = spCF.sp_scalewise(fitted, bw_range=(thresh_lo, np.inf))
        small = spCF.sp_scalewise(fitted, bw_range=(0.0, thresh_lo))
        full = spCF.sp_scalewise(fitted, bw_range=(0.0, np.inf))
        recon = large["pred"]["pred"] + small["pred"]["pred"]
        assert np.allclose(full["pred"]["pred"], recon, atol=1e-9)

    def test_empty_range_errors(self, fitted):
        with pytest.raises(ValueError, match="bw_range"):
            spCF.sp_scalewise(fitted, bw_range=(1e30, 2e30))


# ---------------------------------------------------------------------------
# GLM flow
# ---------------------------------------------------------------------------
class TestCFGLM:
    def test_gaussian(self):
        y, x, coords = make_gauss(n=200)
        mhv = spCF.cf_glm_hv(y=y, x=x, coords=coords,
                              family=spCF.families.gaussian(),
                              verbose=False, seed=42)
        m = spCF.cf_glm(y=y, x=x, coords=coords, mod_hv=mhv, verbose=False)
        assert m.pred["pred"].shape == (200,)
        assert all(k in m.pred_q for k in ("q0.025", "q0.5", "q0.975"))

    def test_poisson(self):
        y, x1, coords = make_poisson(n=200)
        mhv = spCF.cf_glm_hv(y=y, x=x1, coords=coords,
                              family=spCF.families.poisson(),
                              verbose=False, seed=42)
        m = spCF.cf_glm(y=y, x=x1, coords=coords, mod_hv=mhv, verbose=False)
        assert np.all(m.pred["pred"] >= 0)
        # Quantiles must be non-negative for Poisson.
        for q in ("q0.025", "q0.5", "q0.975"):
            assert np.all(m.pred_q[q] >= 0), q

    def test_binomial(self):
        y, x1, coords = make_binomial(n=250)
        rng = np.random.default_rng(0)
        coords0 = rng.uniform(0, 10, size=(60, 2))
        x10 = rng.normal(size=60)
        mhv = spCF.cf_glm_hv(y=y, x=x1, coords=coords,
                              family=spCF.families.binomial(),
                              verbose=False, seed=42)
        m = spCF.cf_glm(y=y, x=x1, coords=coords, x0=x10, coords0=coords0,
                        mod_hv=mhv, verbose=False)
        assert np.all(m.pred["pred"] >= 0) and np.all(m.pred["pred"] <= 1)
        assert np.all(m.pred0["pred"] >= 0) and np.all(m.pred0["pred"] <= 1)

    def test_gamma(self):
        y, x1, coords = make_gamma(n=200)
        mhv = spCF.cf_glm_hv(y=y, x=x1, coords=coords,
                              family=spCF.families.gamma(link="log"),
                              verbose=False, seed=42)
        m = spCF.cf_glm(y=y, x=x1, coords=coords, mod_hv=mhv, verbose=False)
        assert np.all(m.pred["pred"] > 0)

    @pytest.mark.parametrize("kernel", ["exp", "gau"])
    def test_kernels(self, kernel):
        y, x1, coords = make_poisson(n=180)
        mhv = spCF.cf_glm_hv(y=y, x=x1, coords=coords,
                              family=spCF.families.poisson(),
                              kernel=kernel, verbose=False, seed=5)
        m = spCF.cf_glm(y=y, x=x1, coords=coords, mod_hv=mhv, verbose=False)
        assert m.pred["pred"].shape == (180,)


class TestPoissonOffset:
    def test_offset_runs(self):
        rng = np.random.default_rng(0)
        n = 200
        coords = rng.uniform(0, 10, size=(n, 2))
        x1 = rng.normal(size=n)
        z = _spatial(coords)
        log_exposure = rng.uniform(-1, 1, size=n)
        mu = np.exp(log_exposure + 0.4 + 0.3 * x1 + 0.5 * z)
        y = rng.poisson(mu)
        mhv = spCF.cf_glm_hv(y=y, x=x1, coords=coords, offset=log_exposure,
                              family=spCF.families.poisson(),
                              verbose=False, seed=42)
        m = spCF.cf_glm(y=y, x=x1, coords=coords, offset=log_exposure,
                        mod_hv=mhv, verbose=False)
        assert m.pred["pred"].shape == (n,)


# ---------------------------------------------------------------------------
# Result-class formatting
# ---------------------------------------------------------------------------
class TestRepr:
    def test_lm_repr(self):
        y, x, coords = make_gauss(n=120)
        mhv = spCF.cf_lm_hv(y=y, x=x, coords=coords, verbose=False)
        m = spCF.cf_lm(y=y, x=x, coords=coords, mod_hv=mhv, verbose=False)
        s_hv = repr(mhv)
        s_m = repr(m)
        assert "cf_lm_hv" in s_hv
        assert "Coefficients" in s_m

    def test_glm_repr(self):
        y, x1, coords = make_poisson(n=120)
        mhv = spCF.cf_glm_hv(y=y, x=x1, coords=coords,
                              family=spCF.families.poisson(),
                              verbose=False, seed=1)
        m = spCF.cf_glm(y=y, x=x1, coords=coords, mod_hv=mhv, verbose=False)
        assert "cf_glm_hv" in repr(mhv)
        assert "Coefficients" in repr(m)
