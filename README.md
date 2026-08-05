# spCF — Coarse-to-Fine Spatial Modeling

A scalable, machine-learning–compatible framework for spatial regression,
prediction, and uncertainty quantification on moderate-to-large samples.
Available as **both an R package and a Python package** sharing the same
algorithm and reference implementation.

## What it does

Given response `y`, covariates `x`, and 2-D coordinates, spCF jointly fits a
linear / GLM regression and a multiscale spatial process. Holdout validation
selects the spatial scales adaptively, and the fit produces predictive means
and standard deviations at sample and unseen sites. A scale-wise
decomposition (`sp_scalewise`) lets you isolate large-, medium-, and
small-scale spatial structure for interpretation.

Key features:

- Coarse-to-fine bandwidth search that is scalable for large samples.
- Optional **add-learn** with **random forest** or **LightGBM** that captures
  nonlinearities the linear part cannot absorb, with quantile-based predictive
  intervals.
- GLM support for **Gaussian, Poisson, Binomial, Gamma**, plus quasi-likelihood
  families on the R side.
- **Spatial downscaling** (`cf_downscale`) that disaggregates aggregate-level
  responses to a finer grid under a pycnophylactic (mass-preserving) constraint,
  with a GLMM extension (`cf_dglm`) for count/spatiotemporal data.
- **Verifiable predictive intervals**: predictions default to the
  holdout-calibrated *observation* predictive (`se_type="prediction"`) so
  `pred_q` covers new observations — Gaussian conformal, Poisson
  negative-binomial, binomial temperature scaling — together with opt+field
  cluster-robust coefficient standard errors (`se_method="opt"`).

## Repository layout

```
spCF/
├── R/, src/, man/, vignettes/   # R package source
├── DESCRIPTION, NAMESPACE       # R package metadata
└── python/                      # Python port (see python/README.md)
    ├── spCF/                    # importable package
    ├── examples/, tests/
    └── pyproject.toml
```

## Install

### R

```r
# CRAN
install.packages("spCF")

# Or the development version straight from GitHub:
remotes::install_github("dmuraka/spCF")
```

### Python

```bash
pip install "git+https://github.com/dmuraka/spCF#subdirectory=python"
```

The Python importable name is also `spCF`.

## Quick start

### R

```r
library(spCF)
library(sf); library(sp)
data(meuse); data(meuse.grid)

y      <- log(meuse[, "zinc"])
coords <- meuse[, c("x", "y")]
x      <- data.frame(dist = meuse[, "dist"])
x0     <- data.frame(dist = meuse.grid[, "dist"])
coords0<- meuse.grid[, c("x", "y")]

mod_hv <- cf_lm_hv(y = y, x = x, coords = coords)
mod    <- cf_lm   (y = y, x = x, x0 = x0,
                   coords = coords, coords0 = coords0,
                   mod_hv = mod_hv)
mod
```

### Python

```python
import numpy as np
import spCF

rng = np.random.default_rng(0)
n = 500
coords = rng.uniform(0, 10, size=(n, 2))
x = rng.normal(size=(n, 2))
z = np.sin(coords[:, 0] / 2) * np.cos(coords[:, 1] / 2)
y = 1.0 + 2.0 * x[:, 0] - 0.5 * x[:, 1] + 1.5 * z + rng.normal(0, 0.3, n)

mod_hv = spCF.cf_lm_hv(y=y, x=x, coords=coords, kernel="exp", seed=42)
mod    = spCF.cf_lm   (y=y, x=x, coords=coords, mod_hv=mod_hv)
print(mod.beta["coef"])
print(mod.pred["pred"][:5])
```

## Public API (both languages)

| Function | Purpose |
|---|---|
| `cf_lm_hv` / `cf_lm` | Train & holdout-validate / predict with the Gaussian CF spatial model |
| `cf_glm_hv` / `cf_glm` | Train & holdout-validate / predict with a CF spatial GLMM |
| `cf_downscale_hv` / `cf_downscale` | Train & holdout-validate / predict spatial downscaling (areal → fine grid) |
| `cf_dglm_hv` / `cf_dglm` | Downscaling GLMM for count / spatiotemporal data |
| `sp_scalewise` | Extract the spatial process for a given bandwidth range |
| `spCFmap` | Interactive Shiny map explorer for CF outputs (**R only**) |

See the R walk-throughs in
[`vignettes/spCF_lm.Rmd`](vignettes/spCF_lm.Rmd),
[`vignettes/spCF_glm.Rmd`](vignettes/spCF_glm.Rmd),
[`vignettes/spCF_downscale.Rmd`](vignettes/spCF_downscale.Rmd), and
[`vignettes/spCF_dglm.Rmd`](vignettes/spCF_dglm.Rmd),
and [`python/README.md`](python/README.md) for Python-specific notes
(including the `sd_method`, `se_type`, and `se_method` options that control the
predictive SD and coefficient-SE estimators).

## Citation

```bibtex
@article{Murakami2026,
  author  = {Murakami, Daisuke and Comber, Alexis and Yoshida, Takahiro and
             Tsutsumida, Narumasa and Brunsdon, Chris and Nakaya, Tomoki},
  title   = {Coarse-to-fine spatial modeling: A scalable,
             machine-learning-compatible framework},
  journal = {Geographical Analysis},
  volume  = {58},
  number  = {2},
  pages   = {e70034},
  year    = {2026},
  doi     = {10.1111/gean.70034}
}

@article{Murakami2026b,
  author  = {Murakami, Daisuke and Comber, Alexis and Yoshida, Takahiro and
             Tsutsumida, Narumasa and Brunsdon, Chris and Nakaya, Tomoki},
  title   = {Coarse-to-fine spatial GLMM for scalable prediction 
             and multiscale analysis},
  journal = {ArXiv},
  number  = {2605.01157},
  year    = {2026},
}
```

## License

GPL (≥ 2). See `LICENSE` (added via the GitHub license template).
