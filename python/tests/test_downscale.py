"""Tests for coarse-to-fine spatial downscaling (cf_downscale[_hv])."""
from __future__ import annotations

import warnings

import numpy as np
import pytest

import spCF

warnings.filterwarnings("ignore")


def _make_ds(n=300, n_area=40, seed=123):
    rng = np.random.default_rng(seed)
    coords = rng.uniform(0, 10, size=(n, 2))
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    z = np.sin(coords[:, 0] / 2) * np.cos(coords[:, 1] / 2)
    val = np.exp(0.5 + 0.3 * x1 + 0.2 * x2 + 0.8 * z) + rng.uniform(0, 0.5, n)
    prop_weight = rng.uniform(0.5, 2.0, n)
    from sklearn.cluster import KMeans
    agg_id = KMeans(n_clusters=n_area, n_init=10, random_state=0).fit_predict(coords)
    x = np.column_stack([x1, x2])
    return val, x, prop_weight, coords, agg_id


def _fit(Y_type="sum", adj=True, seed=123):
    val, x, pw, coords, agg_id = _make_ds()
    if Y_type == "sum":
        Y = np.bincount(agg_id, weights=val)
    else:  # mean
        cnt = np.bincount(agg_id)
        Y = np.bincount(agg_id, weights=val) / np.maximum(cnt, 1)
    mh = spCF.cf_downscale_hv(Y=Y, Y_type=Y_type, x=x, prop_weight=pw,
                              coords=coords, agg_id=agg_id, seed=seed, verbose=False)
    md = spCF.cf_downscale(Y=Y, x=x, prop_weight=pw, coords=coords,
                           agg_id=agg_id, mod_hv=mh, adj=adj, verbose=False)
    return Y, agg_id, pw, mh, md


class TestDownscale:
    def test_basic_shapes(self):
        Y, agg_id, pw, mh, md = _fit()
        n = agg_id.shape[0]
        assert md.pred["pred"].shape == (n,)
        assert md.pred["pred_sd"].shape == (n,)
        assert np.all(md.pred["pred_sd"] >= 0)
        assert len(md.beta["coef"]) == 3  # intercept + x1 + x2

    def test_pycnophylactic_exact(self):
        """adj=True: cell predictions aggregate exactly to Y (Y_type='sum')."""
        Y, agg_id, pw, mh, md = _fit(Y_type="sum", adj=True)
        Pred_agg = np.bincount(agg_id, weights=md.pred["pred"])
        assert np.max(np.abs(Pred_agg - Y)) < 1e-8

    def test_mean_type_pycnophylactic(self):
        Y, agg_id, pw, mh, md = _fit(Y_type="mean", adj=True)
        # For Y_type="mean", aggregate(a*pred) == Y with the normalized weights.
        a = md.other["a"]
        Pred_agg = np.bincount(agg_id, weights=a * md.pred["pred"])
        assert np.max(np.abs(Pred_agg - Y)) < 1e-8

    def test_no_adjust_runs(self):
        Y, agg_id, pw, mh, md = _fit(adj=False)
        assert md.pred["pred"].shape[0] == agg_id.shape[0]

    def test_e_summary_fields(self):
        Y, agg_id, pw, mh, md = _fit()
        names = [nm for nm, _ in md.e_summary]
        assert names == ["validation_R2", "validation_RMSE", "validation_MAE"]

    def test_tau_and_gamma(self):
        Y, agg_id, pw, mh, md = _fit()
        assert 1e-2 <= md.other["tau"] <= 1e2
        assert np.all(np.asarray(md.other["gamma_list"]) >= 0)
        assert np.all(np.asarray(md.other["gamma_list"]) <= 1)

    def test_repr(self):
        Y, agg_id, pw, mh, md = _fit()
        assert "cf_downscale" in repr(mh)
        assert "Coefficients" in repr(md)

    def test_explicit_id_train(self):
        val, x, pw, coords, agg_id = _make_ds()
        Y = np.bincount(agg_id, weights=val)
        N = Y.shape[0]
        rng = np.random.default_rng(0)
        idt = np.sort(rng.choice(N, int(N * 0.75), replace=False))
        mh = spCF.cf_downscale_hv(Y=Y, Y_type="sum", x=x, prop_weight=pw,
                                  coords=coords, agg_id=agg_id, id_train=idt,
                                  verbose=False)
        assert np.array_equal(mh.id_train, idt)
        md = spCF.cf_downscale(Y=Y, x=x, prop_weight=pw, coords=coords,
                               agg_id=agg_id, mod_hv=mh, verbose=False)
        Pred_agg = np.bincount(agg_id, weights=md.pred["pred"])
        assert np.max(np.abs(Pred_agg - Y)) < 1e-8
