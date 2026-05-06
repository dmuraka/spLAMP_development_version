"""Smoke tests for cf_glm_hv / cf_glm with Poisson and Binomial families."""
import numpy as np
import pytest

import spCF


def _spatial_signal(coords):
    return np.sin(coords[:, 0] / 2) * np.cos(coords[:, 1] / 2)


def test_cf_glm_poisson():
    rng = np.random.default_rng(1)
    n = 200
    coords = rng.uniform(0, 10, size=(n, 2))
    x1 = rng.normal(size=n)
    z = _spatial_signal(coords)
    mu = np.exp(0.5 + 0.3 * x1 + 0.5 * z)
    y = rng.poisson(mu)

    mod_hv = spCF.cf_glm_hv(
        y=y, x=x1, coords=coords, train_rat=0.75,
        family=spCF.families.poisson(), verbose=False, seed=42,
    )
    mod = spCF.cf_glm(y=y, x=x1, coords=coords, mod_hv=mod_hv, verbose=False)
    assert mod.pred["pred"].shape == (n,)
    assert mod.pred["pred_sd"].shape == (n,)
    assert np.all(mod.pred["pred"] >= 0)


def test_cf_glm_binomial_with_prediction_sites():
    rng = np.random.default_rng(2)
    n = 250
    coords = rng.uniform(0, 10, size=(n, 2))
    x1 = rng.normal(size=n)
    z = _spatial_signal(coords)
    eta = -0.2 + 0.5 * x1 + 1.0 * z
    p = 1 / (1 + np.exp(-eta))
    y = rng.binomial(1, p).astype(float)

    coords0 = rng.uniform(0, 10, size=(60, 2))
    x10 = rng.normal(size=60)

    mod_hv = spCF.cf_glm_hv(
        y=y, x=x1, coords=coords, train_rat=0.75,
        family=spCF.families.binomial(), verbose=False, seed=42,
    )
    mod = spCF.cf_glm(
        y=y, x=x1, coords=coords, x0=x10, coords0=coords0,
        mod_hv=mod_hv, verbose=False,
    )
    assert mod.pred0 is not None
    assert mod.pred0["pred"].shape == (60,)
    assert np.all(mod.pred0["pred"] >= 0) and np.all(mod.pred0["pred"] <= 1)


def test_cf_glm_gaussian_matches_lm_pattern():
    rng = np.random.default_rng(3)
    n = 200
    coords = rng.uniform(0, 10, size=(n, 2))
    x1 = rng.normal(size=n)
    z = _spatial_signal(coords)
    y = 1.0 + 2.0 * x1 + 1.5 * z + rng.normal(0, 0.3, size=n)

    mod_hv = spCF.cf_glm_hv(
        y=y, x=x1, coords=coords, train_rat=0.75,
        family=spCF.families.gaussian(), verbose=False, seed=42,
    )
    mod = spCF.cf_glm(y=y, x=x1, coords=coords, mod_hv=mod_hv, verbose=False)
    # We expect coefficient on x1 close to 2.0 (loosely; spatial trend overlaps).
    coef = mod.beta["coef"]
    assert abs(coef[1] - 2.0) < 0.5
