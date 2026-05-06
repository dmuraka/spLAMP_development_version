# spCF (Python)

Python port of the R package
[**spCF**](https://cran.r-project.org/package=spCF) — Coarse-to-Fine Spatial
Modeling (CFSM) for fast spatial prediction, regression, and uncertainty
quantification on moderate-to-large samples.

Reference: Murakami, D., Comber, A., Yoshida, T., Tsutsumida, N., Brunsdon, C.,
& Nakaya, T. (2026). *Coarse-to-fine spatial modeling: A scalable,
machine-learning-compatible framework.* Geographical Analysis, 58(2), e70034.
[doi:10.1111/gean.70034](https://onlinelibrary.wiley.com/doi/10.1111/gean.70034)

## Install

The Python port lives under the `python/` subdirectory of the
[`dmuraka/spCF`](https://github.com/dmuraka/spCF) repository, alongside the
R source.

```bash
# Install the latest from GitHub
pip install "git+https://github.com/dmuraka/spCF#subdirectory=python"

# With the JIT-accelerated kernels (recommended; ~10× faster):
pip install "git+https://github.com/dmuraka/spCF#subdirectory=python[fast]"

# With NLOPT_LN_BOBYQA for R-parity optimization:
pip install "git+https://github.com/dmuraka/spCF#subdirectory=python[nlopt]"
```

For local development:

```bash
git clone https://github.com/dmuraka/spCF
cd spCF/python
pip install -e .[dev]   # numba + nlopt + pytest
```

The importable name is `spCF`, matching the R package.

## Quick start — Gaussian response

```python
import numpy as np
import spCF

rng = np.random.default_rng(0)
n = 500
coords = rng.uniform(0, 10, size=(n, 2))
x = rng.normal(size=(n, 2))
z = np.sin(coords[:, 0] / 2) * np.cos(coords[:, 1] / 2)
y = 1.0 + 2.0 * x[:, 0] - 0.5 * x[:, 1] + 1.5 * z + rng.normal(0, 0.3, n)

# Holdout-validation training
mod_hv = spCF.cf_lm_hv(y=y, x=x, coords=coords, kernel="exp", seed=42)

# Spatial regression / prediction at the sample sites
mod = spCF.cf_lm(y=y, x=x, coords=coords, mod_hv=mod_hv)

print(mod.beta["coef"])            # estimated coefficients
print(mod.pred["pred"][:5])        # predictive mean
print(mod.pred["pred_sd"][:5])     # predictive sd
```

## Prediction at new sites

```python
coords0 = rng.uniform(0, 10, size=(100, 2))
x0 = rng.normal(size=(100, 2))
mod = spCF.cf_lm(y=y, x=x, coords=coords,
                 x0=x0, coords0=coords0, mod_hv=mod_hv)
print(mod.pred0["pred"].shape)     # (100,)
print(mod.pred0["pred_sd"].shape)
```

## Multiscale spatial-process extraction

```python
big   = spCF.sp_scalewise(mod, bw_range=(2.0, np.inf))
small = spCF.sp_scalewise(mod, bw_range=(0.0, 2.0))
```

## Non-Gaussian responses (cf_glm)

Supports any GLM family from statsmodels via the `spCF.families` shim
(`gaussian`, `poisson`, `binomial`, `gamma`).

```python
y = rng.poisson(np.exp(0.5 + 0.3 * x[:, 0] + 0.5 * z))

mod_hv = spCF.cf_glm_hv(
    y=y, x=x[:, 0], coords=coords,
    family=spCF.families.poisson(), seed=42,
)
mod = spCF.cf_glm(y=y, x=x[:, 0], coords=coords, mod_hv=mod_hv)

print(mod.beta["coef"])
print(mod.pred_q["q0.05"][:5])     # 5%-tile on the response scale
```

## Public API

| Function | Purpose |
| --- | --- |
| `cf_lm_hv(...)` | Train & holdout-validate the Gaussian CF spatial model |
| `cf_lm(...)` | Predict / regress with a trained Gaussian CF model |
| `cf_glm_hv(...)` | Train & holdout-validate a CF spatial GLMM |
| `cf_glm(...)` | Predict / regress with a trained CF GLMM |
| `sp_scalewise(mod, bw_range)` | Extract spatial process for a bandwidth range |
| `spCF.families.{gaussian,poisson,binomial,gamma}` | GLM family helpers |

## Differences from the R package

* **Numerical kernels**: the C++ `lwr_chunk_cpp` / `lwr_chunk_glm_cpp` are
  ported to Python and accelerated via `numba` when installed (otherwise
  pure Python — slower but pure-pip).
* **Random forest add-learn**: uses `sklearn.ensemble.RandomForestRegressor`
  for the point predictions, and `quantile-forest`'s
  `RandomForestQuantileRegressor` for the predictive standard deviation —
  the same 201-quantile / 200-draw construction that R's `ranger(quantreg=TRUE)`
  uses, so `pred_sd` matches R to within ~5% on typical workloads. Two cheaper
  alternatives are exposed via ``cf_lm(..., sd_method=...)``: ``"tree_var"``
  (variance across RF trees, no quantile-forest needed) and ``"residual"``
  (legacy homoskedastic approximation).
* **Optimizer**: defaults to `scipy.optimize.minimize_scalar` (Brent). Install
  the optional `nlopt` extra to recover R parity (`NLOPT_LN_BOBYQA`).
* **RNG**: `numpy.random.default_rng`. Stochastic outputs (train/test split,
  fine-scale knot sampling) match R only structurally, not bit-for-bit.

## License

GPL-2.0-or-later, matching the upstream R package.
