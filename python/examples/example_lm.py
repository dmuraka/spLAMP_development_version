"""Gaussian CFSM example with multiscale spatial process extraction."""
import numpy as np

import spCF


def main():
    rng = np.random.default_rng(0)
    n = 600
    coords = rng.uniform(0, 10, size=(n, 2))
    x = rng.normal(size=(n, 2))
    z = np.sin(coords[:, 0] / 2) * np.cos(coords[:, 1] / 2)
    y = 1.0 + 2.0 * x[:, 0] - 0.5 * x[:, 1] + 1.5 * z + rng.normal(0, 0.3, n)

    coords0 = rng.uniform(0, 10, size=(100, 2))
    x0 = rng.normal(size=(100, 2))

    print("Training (cf_lm_hv) ...")
    mod_hv = spCF.cf_lm_hv(
        y=y, x=x, coords=coords, kernel="exp", seed=42, train_rat=0.75,
    )
    print(f"  selected bandwidths: {len(mod_hv.other['bands'])}")
    print(f"  validation SSE     : {mod_hv.sse_hv:.4g}")

    print("Predicting (cf_lm) ...")
    mod = spCF.cf_lm(
        y=y, x=x, coords=coords, x0=x0, coords0=coords0, mod_hv=mod_hv,
        verbose=False,
    )
    print(f"  estimated coefficients: {mod.beta['coef']}")
    print(f"  validation R2         : {mod.e_summary[0][1]:.4g}")
    print(f"  pred mean (first 3)   : {mod.pred['pred'][:3]}")
    print(f"  pred sd   (first 3)   : {mod.pred['pred_sd'][:3]}")

    print("Multiscale extraction ...")
    if mod.bands is not None and len(mod.bands) > 1:
        thresh = float(np.median(mod.bands))
        large = spCF.sp_scalewise(mod, bw_range=(thresh, np.inf))
        small = spCF.sp_scalewise(mod, bw_range=(0.0, thresh))
        print(f"  large-scale sd: {np.std(large['pred']['pred']):.4g}")
        print(f"  small-scale sd: {np.std(small['pred']['pred']):.4g}")


if __name__ == "__main__":
    main()
