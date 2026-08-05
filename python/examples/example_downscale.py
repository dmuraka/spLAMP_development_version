"""Coarse-to-fine spatial downscaling (CF-DS) example.

Downscales an aggregate-level (areal) response ``Y`` to disaggregate-level
point predictions that aggregate exactly back to ``Y`` (pycnophylactic).
"""
import numpy as np

import spCF


def main():
    rng = np.random.default_rng(123)
    n = 300
    coords = rng.uniform(0, 10, size=(n, 2))
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    z = np.sin(coords[:, 0] / 2) * np.cos(coords[:, 1] / 2)

    # Positive cell-level values and proportional allocation weights.
    val = np.exp(0.5 + 0.3 * x1 + 0.2 * x2 + 0.8 * z) + rng.uniform(0, 0.5, n)
    prop_weight = rng.uniform(0.5, 2.0, n)

    # 40 aggregate areas via k-means on the coordinates.
    from sklearn.cluster import KMeans
    agg_id = KMeans(n_clusters=40, n_init=10, random_state=0).fit_predict(coords)

    # Aggregate-level response (Y_type="sum": extensive/count-like).
    Y = np.bincount(agg_id, weights=val)

    x = np.column_stack([x1, x2])

    print("Training (cf_downscale_hv) ...")
    mod_hv = spCF.cf_downscale_hv(
        Y=Y, Y_type="sum", x=x, prop_weight=prop_weight,
        coords=coords, agg_id=agg_id, seed=123, verbose=False,
    )
    print(f"  selected scales : {0 if mod_hv.other['bands'] is None else len(mod_hv.other['bands'])}")
    print(f"  validation SSE  : {mod_hv.sse_hv:.4g}")

    print("Downscaling (cf_downscale) ...")
    mod = spCF.cf_downscale(
        Y=Y, x=x, prop_weight=prop_weight, coords=coords, agg_id=agg_id,
        mod_hv=mod_hv, verbose=False,
    )
    print(f"  coefficients      : {mod.beta['coef']}")
    print(f"  validation R2     : {mod.e_summary[0][1]:.4g}")
    print(f"  pred mean (first 3): {mod.pred['pred'][:3]}")

    # Pycnophylactic constraint: predictions aggregate exactly to Y.
    Pred_agg = np.bincount(agg_id, weights=mod.pred["pred"])
    print(f"  max |aggregate(pred) - Y| : {np.max(np.abs(Pred_agg - Y)):.2e}")


if __name__ == "__main__":
    main()
