"""Poisson and Binomial CFSM examples."""
import numpy as np

import spCF


def poisson_demo():
    rng = np.random.default_rng(1)
    n = 400
    coords = rng.uniform(0, 10, size=(n, 2))
    x1 = rng.normal(size=n)
    z = np.sin(coords[:, 0] / 2) * np.cos(coords[:, 1] / 2)
    mu = np.exp(0.5 + 0.3 * x1 + 0.5 * z)
    y = rng.poisson(mu)

    print("=== Poisson ===")
    mod_hv = spCF.cf_glm_hv(
        y=y, x=x1, coords=coords, family=spCF.families.poisson(),
        seed=42, train_rat=0.75,
    )
    mod = spCF.cf_glm(y=y, x=x1, coords=coords, mod_hv=mod_hv, verbose=False)
    print(f"  coefs: {mod.beta['coef']}  (true: 0.5, 0.3)")
    print(f"  validation Pseudo-R2: {mod.e_summary[0][1]:.4g}")
    print(f"  pred mean[:3]: {mod.pred['pred'][:3]}")


def binomial_demo():
    rng = np.random.default_rng(2)
    n = 400
    coords = rng.uniform(0, 10, size=(n, 2))
    x1 = rng.normal(size=n)
    z = np.sin(coords[:, 0] / 2) * np.cos(coords[:, 1] / 2)
    eta = -0.2 + 0.5 * x1 + 1.0 * z
    p = 1 / (1 + np.exp(-eta))
    y = rng.binomial(1, p).astype(float)

    coords0 = rng.uniform(0, 10, size=(80, 2))
    x10 = rng.normal(size=80)

    print("=== Binomial ===")
    mod_hv = spCF.cf_glm_hv(
        y=y, x=x1, coords=coords, family=spCF.families.binomial(),
        seed=42, train_rat=0.75,
    )
    mod = spCF.cf_glm(
        y=y, x=x1, coords=coords, x0=x10, coords0=coords0,
        mod_hv=mod_hv, verbose=False,
    )
    print(f"  coefs: {mod.beta['coef']}  (true: -0.2, 0.5)")
    print(f"  validation Pseudo-R2: {mod.e_summary[0][1]:.4g}")
    print(f"  pred0 prob[:3]: {mod.pred0['pred'][:3]}")


if __name__ == "__main__":
    poisson_demo()
    binomial_demo()
