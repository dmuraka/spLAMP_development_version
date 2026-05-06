"""Smoke test for cf_lm_hv / cf_lm on synthetic Gaussian data."""
import numpy as np
import pytest

import spCF


def _make_data(n=300, seed=0):
    rng = np.random.default_rng(seed)
    coords = rng.uniform(0, 10, size=(n, 2))
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    # spatial trend
    z = np.sin(coords[:, 0] / 2) * np.cos(coords[:, 1] / 2)
    y = 1.0 + 2.0 * x1 - 0.5 * x2 + 1.5 * z + rng.normal(0, 0.3, size=n)
    return y, np.column_stack([x1, x2]), coords


def test_cf_lm_smoke():
    y, x, coords = _make_data(n=200, seed=0)
    mod_hv = spCF.cf_lm_hv(y=y, x=x, coords=coords, train_rat=0.75,
                              kernel="exp", verbose=False, seed=42)
    assert mod_hv.sse_hv > 0
    assert len(mod_hv.sse_hv_all) >= 2

    mod = spCF.cf_lm(y=y, x=x, coords=coords, mod_hv=mod_hv, verbose=False)
    assert mod.pred["pred"].shape == (200,)
    assert mod.pred["pred_sd"].shape == (200,)
    assert np.all(mod.pred["pred_sd"] >= 0)


def test_cf_lm_with_prediction_sites():
    y, x, coords = _make_data(n=200, seed=1)
    rng = np.random.default_rng(2)
    coords0 = rng.uniform(0, 10, size=(50, 2))
    x0 = rng.normal(size=(50, 2))

    mod_hv = spCF.cf_lm_hv(y=y, x=x, coords=coords, train_rat=0.75,
                              kernel="exp", verbose=False, seed=7)
    mod = spCF.cf_lm(y=y, x=x, coords=coords, x0=x0, coords0=coords0,
                       mod_hv=mod_hv, verbose=False)
    assert mod.pred0 is not None
    assert mod.pred0["pred"].shape == (50,)
    assert mod.pred0["pred_sd"].shape == (50,)


def test_sp_scalewise():
    y, x, coords = _make_data(n=200, seed=3)
    mod_hv = spCF.cf_lm_hv(y=y, x=x, coords=coords, train_rat=0.75, verbose=False, seed=11)
    mod = spCF.cf_lm(y=y, x=x, coords=coords, mod_hv=mod_hv, verbose=False)
    if mod.bands is None or len(mod.bands) == 0:
        pytest.skip("no bands selected; cannot test sp_scalewise")
    out = spCF.sp_scalewise(mod, bw_range=(0, np.inf))
    assert out["pred"]["pred"].shape == (200,)
    assert np.all(out["pred"]["pred_sd"] >= 0)
