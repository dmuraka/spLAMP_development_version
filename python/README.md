# spCF (Python)

Python port of the R package
[**spCF**](https://cran.r-project.org/package=spCF) (version 0.1.3) —
Coarse-to-Fine Spatial Modeling (CFSM) for fast spatial prediction, regression,
downscaling, space-time modeling, and uncertainty quantification on
moderate-to-large samples.

References:

* Gaussian CFSM — Murakami, D., Comber, A., Yoshida, T., Tsutsumida, N.,
  Brunsdon, C., & Nakaya, T. (2026). *Coarse-to-fine spatial modeling: A
  scalable, machine-learning-compatible framework.* Geographical Analysis,
  58(2), e70034.
  [doi:10.1111/gean.70034](https://onlinelibrary.wiley.com/doi/10.1111/gean.70034)
* CF-GLMMs — Murakami et al. (2025).
  [arXiv:2605.01157](https://doi.org/10.48550/arXiv.2605.01157)
* CF spatial downscaling — Murakami, Chun, Yoshida & Seya (2026).

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
print(mod.beta["coef_se"])         # cluster-robust SEs (robust_se=True default)
print(mod.pred["pred"][:5])        # predictive mean
print(mod.pred["pred_sd"][:5])     # predictive sd (tau-calibrated)
print(mod.pred_q["q0.95"][:5])     # 95%-tile predictive quantile
print(mod.e_summary)               # validation_R2 / RMSE / MAE
```

Coefficient standard errors default to a spatial-block cluster-robust
estimator (`robust_se=True`); the spatial-process contribution to `pred_sd`
is rescaled by a holdout-calibrated factor `tau` (stored in `mod.other["tau"]`)
and capped at the marginal field variance.

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

## Spatial downscaling (cf_downscale)

Predict a disaggregate-level (point) response from an aggregate-level (areal)
response `Y`, with predictions that aggregate **exactly** back to `Y`
(pycnophylactic constraint).

```python
# Y: areal response (length N), agg_id: area id per cell (length n),
# prop_weight: proportional-allocation weights (length n), x/coords: cell-level
mod_hv = spCF.cf_downscale_hv(
    Y=Y, Y_type="sum", x=x, prop_weight=prop_weight,
    coords=coords, agg_id=agg_id, seed=123,
)
mod = spCF.cf_downscale(
    Y=Y, x=x, prop_weight=prop_weight, coords=coords, agg_id=agg_id, mod_hv=mod_hv,
)

print(mod.pred["pred"][:5])        # disaggregate predictive mean
print(mod.pred["pred_sd"][:5])     # disaggregate predictive sd (tau-calibrated)
# aggregate(a*pred) == Y exactly on every area when adj=True (default)
```

Use `Y_type="sum"` for extensive/count-like data (e.g. population) and
`Y_type="mean"` for intensive/density-like data.

## Space-time modeling (cf_dglm)

Dynamic (space-time) spatial GLMM: a separable cascade combining a per-knot
AR(1) Kalman smoother in time with kernel kriging in space. Rows sharing the
same `coords` are repeated observations of one location across `time`; the
panel may be unbalanced (observed locations may differ across time points).

```python
# coords/time per row; the AR(1) parameters (rho, Q) are estimated internally
mod_hv = spCF.cf_dglm_hv(y=y, x=x, coords=coords, time=time,
                         family=spCF.families.poisson(), offset=offset)
mod = spCF.cf_dglm(y=y, x=x, coords=coords, time=time,
                   offset=offset, mod_hv=mod_hv)

print(mod.beta["coef"])                     # constant coefficients (cluster-robust SEs)
print(mod.other["rho"], mod.other["Q"])     # estimated AR(1) parameters
print(mod.pred["pred"][:5])                 # space-time predictive mean

# multiscale spatial process, temporally averaged over a time window
big = spCF.sp_scalewise(mod, bw_range=(4000, np.inf), time_range=(2010, 2011))
```

Prediction at new sites/times uses `x0`, `coords0`, `time0` (and `offset0`);
interior missing times are interpolated and future times are forecast via the
per-knot AR(1) predict step.

## Public API

| Function | Purpose |
| --- | --- |
| `cf_lm_hv(...)` / `cf_lm(...)` | Train / predict — Gaussian CF spatial model |
| `cf_glm_hv(...)` / `cf_glm(...)` | Train / predict — CF spatial GLMM |
| `cf_downscale_hv(...)` / `cf_downscale(...)` | Train / downscale an areal response |
| `cf_dglm_hv(...)` / `cf_dglm(...)` | Train / predict — dynamic (space-time) CF GLMM |
| `sp_scalewise(mod, bw_range, time_range)` | Extract spatial process for a bandwidth (and time) range |
| `spCF.families.{gaussian,poisson,binomial,gamma}` | GLM family helpers |

`sp_scalewise`'s `bw_range` is the half-open interval `[low, high)`, so
contiguous ranges partition the scales without double-counting a shared
endpoint. `time_range` applies only to `cf_dglm` fits.

## Differences from the R package

Tracks R **spCF 0.2.0** (Python package `0.1.4`). The dynamic space-time module (`cf_dglm` /
`cf_dglm_hv`, with the fused `dglm_scale_chunk` AR(1)-Kalman + gPoE kernel) is
ported and numerically matches R closely (with matched `id_train`, coefficients
and validation metrics agree to ~3 decimals); the global AR(1) `(rho, Q)` is
estimated with `scipy.optimize` (Nelder-Mead) in place of R's `nloptr`
BOBYQA, so `rho`/`Q` differ slightly. `sp_scalewise` gains a half-open
`bw_range` and a `time_range` for `cf_dglm` fits.

The 0.1.2 statistical machinery is also tracked: eq.(10)
predictive variance (`Z_pv`), holdout `tau` variance calibration with a
marginal-field-variance (sill) cap, spatial-block cluster-robust coefficient
standard errors (`robust_se=True`), and predictive quantiles (`pred_q` /
`pred0_q`) — including total conformalized quantile regression (CQR) when
`add_learn="rf"` is active for `cf_lm`, and link-scale Gaussian quantiles for
`cf_glm`.

* **Predictive uncertainty defaults (matches R)**: `cf_lm` / `cf_glm` /
  `cf_dglm` default to `se_type="prediction"` and `se_method="opt"`, matching
  the R package. `se_type="prediction"` returns the holdout-calibrated
  *observation* predictive — Gaussian signal-variance + noise with
  split-conformal scaling, a negative-binomial count predictive for Poisson,
  and temperature-scaled Bernoulli for binomial — so `pred_q`/`pred0_q` cover
  new observations rather than the signal mean. The signal (mean) versions are
  preserved in `other['pred_signal']` / `other['pred_q_signal']`; pass
  `se_type="mean"` to return only the signal. `se_method="opt"` recomputes the
  coefficient covariance with the opt+field sandwich (leverage-LOO ceiling),
  removing the conservatism of the classic field-retained estimator
  (`se_method="classic"`).

* **Numerical kernels**: the C++ `lwr_chunk_cpp` / `lwr_chunk_glm_cpp` /
  `lwr_ds_chunk_cpp` are ported to Python and accelerated via `numba` when
  installed (otherwise pure Python — slower but pure-pip).
* **Random forest add-learn**: uses `sklearn.ensemble.RandomForestRegressor`
  for the point predictions, and `quantile-forest`'s
  `RandomForestQuantileRegressor` for the 201-quantile / 200-draw construction
  that R's `ranger(quantreg=TRUE)` uses, feeding the total-CQR predictive
  distribution. Two cheaper `pred_sd` alternatives are exposed via
  `cf_lm(..., sd_method=...)`: `"tree_var"` (variance across RF trees, no
  quantile-forest needed) and `"residual"` (homoskedastic approximation).
  The R `add_learn="lightgbm"` learner is **not** ported.
* **Optimizer**: defaults to `scipy.optimize.minimize_scalar` (Brent). Install
  the optional `nlopt` extra to recover R parity (`NLOPT_LN_BOBYQA`).
* **RNG**: `numpy.random.default_rng` / scikit-learn `KMeans`. Stochastic
  outputs (train/test split, k-means knot selection, CQR draws) match R only
  structurally, not bit-for-bit; validation metrics and total-calibrated
  predictions agree closely, while per-point values differ within the
  knot-selection variability.

## License

GPL-2.0-or-later, matching the upstream R package.
