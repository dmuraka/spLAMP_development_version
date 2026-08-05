"""Coarse-to-fine dynamic (space-time) spatial GLMM example (cf_dglm).

Fits a separable space-time cascade (per-knot AR(1) Kalman smoother in time +
kernel kriging in space) to a balanced Gaussian panel, then extracts the
temporally-averaged spatial process.
"""
import numpy as np

import spCF


def main():
    rng = np.random.default_rng(1)
    ns, nt = 60, 6                      # 60 sites observed at each of 6 time points
    sites = rng.uniform(0, 10, size=(ns, 2))

    # latent AR(1) spatial field over time
    K = 12
    cen = rng.uniform(0, 10, size=(K, 2))
    Dc = np.sqrt(((cen[:, None] - cen[None]) ** 2).sum(-1))
    Wc = np.exp(-Dc / 3.0); Wc = Wc / np.sqrt((Wc ** 2).sum(1, keepdims=True))
    a = np.zeros((nt, K)); a[0] = Wc @ rng.normal(size=K)
    for t in range(1, nt):
        a[t] = 0.7 * a[t - 1] + Wc @ rng.normal(size=K)
    Dm = np.sqrt(((sites[:, None] - cen[None]) ** 2).sum(-1))
    Wp = np.exp(-Dm / 3.0); Wp = Wp / Wp.sum(1, keepdims=True)

    coords = np.repeat(sites, nt, axis=0)
    time = np.tile(np.arange(1, nt + 1), ns)
    field = (Wp[np.repeat(np.arange(ns), nt)] * a[time - 1]).sum(1)
    x1 = rng.normal(size=ns * nt)
    x2 = rng.normal(size=ns * nt)
    y = 1.0 + 1.5 * x1 - 0.5 * x2 + field + rng.normal(0, 0.4, size=ns * nt)
    x = np.column_stack([x1, x2])

    print("Training (cf_dglm_hv) ...")
    mod_hv = spCF.cf_dglm_hv(y=y, x=x, coords=coords, time=time,
                             family=spCF.families.gaussian(), seed=1234, verbose=False)
    print(f"  selected scales : {0 if mod_hv.other['bands'] is None else len(mod_hv.other['bands'])}")
    print(f"  AR(1)           : rho={mod_hv.other['rho']:.3f}, Q={mod_hv.other['Q']:.3g}")
    print(f"  OOS validation  : {[(n, round(v, 4)) for n, v in mod_hv.e_summary]}")

    print("Space-time modeling / prediction (cf_dglm) ...")
    mod = spCF.cf_dglm(y=y, x=x, coords=coords, time=time, mod_hv=mod_hv, verbose=False)
    print(f"  coefficients      : {mod.beta['coef']}  (true: 1, 1.5, -0.5)")
    print(f"  pred mean (first 3): {mod.pred['pred'][:3]}")
    print(f"  mean(pred)={mod.pred['pred'].mean():.4f}  mean(y)={y.mean():.4f}")

    print("Multiscale extraction, averaged over time 5 ...")
    if mod.bands is not None and len(mod.bands) > 1:
        thr = float(np.median(mod.bands))
        large = spCF.sp_scalewise(mod, bw_range=(thr, np.inf), time_range=(5, 5))
        small = spCF.sp_scalewise(mod, bw_range=(0, thr), time_range=(5, 5))
        print(f"  large-scale sd: {np.std(large['pred']['pred']):.4g}")
        print(f"  small-scale sd: {np.std(small['pred']['pred']):.4g}")


if __name__ == "__main__":
    main()
