#' Coarse-to-fine dynamic (space-time) spatial GLMMs (CF-DGLMMs)
#'
#' Prediction and regression via a separable space-time cascade. Given the
#' scales selected by \code{\link{cf_dglm_hv}}, the model is refitted on the
#' full sample and predictions (with standard deviations) are produced at sample
#' and, optionally, prediction sites. The link-scale linear predictor is
#' \eqn{g(\mu_{i,t}) = x_{i,t}'\beta + \sum_k f_k(s_i,t) + } offset, where each
#' scale-\eqn{k} field \eqn{f_k} couples a per-knot AR(1) Kalman smoother in time
#' with kernel kriging in space.
#'
#' The full-sample fit is a SINGLE coarse-to-fine cascade sweep, mirroring the
#' relationship between \code{cf_glm} and \code{cf_glm_hv}: it reuses the same
#' single greedy sweep that \code{\link{cf_dglm_hv}} performs for scale
#' selection, plus prediction. Within the sweep, for each band (coarse to fine)
#' the GLM working response/weights are refreshed (IRLS folded into the sweep, as
#' \code{cf_glm}'s per-band \code{glm()} does), the scale is fit and accumulated,
#' and the constant and time-varying coefficients are backfit. (The earlier
#' outer-IRLS implementation is archived as \code{cf_dglm_iter} under misc/.)
#'
#' @param y Vector of response variables (N x 1).
#' @param x Matrix of covariates (N x K).
#' @param coords Matrix of 2-dimensional point coordinates (N x 2). The
#'   space-time panel may be unbalanced (observed locations may differ across
#'   time points).
#' @param time Vector of time indices (N x 1); must use the same time points as
#'   in \code{\link{cf_dglm_hv}}.
#' @param offset Optional. Offset variable (N x 1), consistent with \code{glm}.
#' @param x0 Optional. Matrix of covariates at prediction sites (N0 x K).
#' @param coords0 Optional. Coordinates at prediction sites (N0 x 2).
#' @param time0 Optional. Time indices at prediction sites (N0 x 1). May include
#'   time points with no observations: interior time points absent from the
#'   training data are interpolated, and time points beyond the last observed one
#'   are forecast, via the per-knot AR(1) predict step (the Kalman gain is zero
#'   where a time column carries no data). Such time points are added to the
#'   working time grid, so predicting at interior gaps slightly re-spaces the
#'   AR(1) grid; forecasting beyond the last observed point leaves the
#'   training-time fit unchanged.
#' @param offset0 Optional. Offset at prediction sites (N0 x 1).
#' @param mod_hv Output object of \code{\link{cf_dglm_hv}}.
#' @param robust_se Logical; if \code{TRUE} (default), the constant-coefficient
#'   standard errors (and the coefficient-uncertainty term of the predictive SE)
#'   use a spatial-block cluster-robust sandwich that accounts for the cascade
#'   field being a correlated random component. The naive model-based covariance
#'   treats the field as a known offset and severely understates the SEs; the
#'   robust version restores near-nominal coverage. Set \code{FALSE} for the
#'   naive \code{vcov(glm)} SEs.
#' @param sill_cap Logical; if \code{TRUE} (default), the field component of the
#'   predictive variance is capped at the marginal variance of the fitted total
#'   field (the "sill"), \code{var(sum_r z_r)} on the link scale. This prevents
#'   the gPoE variance from diverging in deep extrapolation, mirroring a
#'   stationary GP that reverts to the prior marginal variance far from data. The
#'   cap is applied to the total field variance (never per scale) and leaves the
#'   predictive mean, RMSE, and the coefficient-uncertainty term unchanged. It is
#'   disabled automatically for \code{binomial} responses. Set \code{FALSE} to
#'   leave the field variance uncapped.
#' @param se_type Type of predictive uncertainty in \code{pred}/\code{pred_q}.
#'   \code{"prediction"} (default) returns the holdout-calibrated OBSERVATION
#'   predictive for a new data point (Gaussian: mean uncertainty + residual
#'   variance; Poisson: negative-binomial count predictive; binomial:
#'   temperature-calibrated probability with \code{pred_sd = sqrt(p(1-p))}); the
#'   signal versions are kept in \code{pred_signal}/\code{pred_q_signal}.
#'   \code{"mean"} returns the signal (mean) uncertainty only (previous
#'   behaviour). See \code{other$calibration}.
#' @param se_method Cluster-robust coefficient-SE estimator (used when
#'   \code{robust_se = TRUE}). \code{"opt"} (default) splits the sandwich
#'   meat into a field-removed observation-noise part and a field part that adds
#'   the calibrated field variance back with a within-block \code{exp(-d/h)}
#'   correlation (\code{h} = median committed bandwidth); this is near-nominal. A refit-free leverage leave-one-out ceiling then caps the field term, preventing over-coverage for count (Poisson) responses while leaving already-calibrated families unchanged.
#'   \code{"classic"} keeps the realised field inside the working residual (the
#'   previous behaviour), which is valid but conservative.
#'
#' @return A list (class \code{"cf_dglm"}) mirroring \code{\link{cf_glm}}:
#'   \code{beta}, \code{sd_summary}, \code{e_summary}, \code{pred}, \code{pred0},
#'   \code{pred_q}, \code{pred0_q}, \code{bands}, \code{Z}, \code{Z_sd},
#'   \code{Z0}, \code{Z0_sd}, \code{other}, \code{call}, plus
#'   \describe{
#'     \item{beta_tv, beta_tv_sd}{Time-varying coefficients and their standard
#'     deviations, one row per time point and one column per covariate named in
#'     \code{tvc} (plus a \code{time} column). \code{NULL} when \code{tvc} was
#'     not used in \code{\link{cf_dglm_hv}}.}
#'     \item{pred_signal, pred_q_signal}{The signal (mean) predictive kept
#'     alongside the observation predictive when \code{se_type = "prediction"}.}
#'   }
#'   The temporal parameters of the fitted cascade are in \code{other$rho}
#'   (AR(1) autocorrelation), \code{other$Q} (innovation variance) and
#'   \code{other$tau} (holdout-calibrated field-variance factor); the first two
#'   are shown by \code{print}.
#'
#' @references
#' Murakami, D. (2026).
#' Fast covariance-free spatiotemporal modeling via coarse-to-fine learning.
#' *ArXiv preprint*.
#'
#' @seealso \code{\link{cf_dglm_hv}}, \code{\link{cf_glm}}
#' @author Daisuke Murakami
#'
#' @examples
#' ### Monthly PM10 at 63 German background stations, 2001-2005 (the data set
#' ### behind the "Demo (air, space-time)" entry of spCFmap(); see the
#' ### spCF_dglm vignette for a fuller walk-through).
#' require(sf)
#' air    <- read.csv(system.file("shiny", "spCFmap",
#'                                "example_spacetime_air.csv", package = "spCF"))
#' pts    <- st_as_sf(air, coords = c("lon", "lat"), crs = 4326)
#' coords <- st_coordinates(st_transform(pts, 25832))  # UTM 32N, in metres
#'
#' ### The annual cycle is a fixed effect; the space-time process takes the rest
#' x      <- data.frame(sin12 = sin(2 * pi * air$month / 12),
#'                      cos12 = cos(2 * pi * air$month / 12))
#'
#' ### Holdout validation optimizing the number of spatial scales
#' mod_hv <- cf_dglm_hv(y = air$pm10, x = x, coords = coords, time = air$time)
#'
#' ### Space-time modeling; the 63 stations are also predicted one month beyond
#' ### the data (time = 61), which the AR(1) predict step turns into a forecast
#' uni    <- !duplicated(air$station)
#' n0     <- sum(uni)
#' mod    <- cf_dglm(y = air$pm10, x = x, coords = coords, time = air$time,
#'                   x0 = data.frame(sin12 = rep(sin(2 * pi / 12), n0),
#'                                   cos12 = rep(cos(2 * pi / 12), n0)),
#'                   coords0 = coords[uni, ], time0 = rep(61, n0),
#'                   mod_hv = mod_hv)
#' mod
#'
#' round(mod$bands / 1000, 1)              # accepted bandwidths, in km
#' round(c(rho = mod$other$rho, Q = mod$other$Q), 3)  # AR(1) parameters
#'
#' ### Mapping the forecast for January 2006 at the station locations
#' fc     <- st_as_sf(data.frame(pred = mod$pred0$pred, coords[uni, ]),
#'                    coords = c("X", "Y"), crs = 25832)
#' plot(fc[, "pred"], pch = 20, cex = 1.3, axes = TRUE, key.pos = 4, nbreaks = 20)
#'
#' ### Multiscale extraction, averaged over the observed months
#' mod_s1 <- sp_scalewise(mod, bw_range = c(150000, Inf))  # large scale
#' mod_s2 <- sp_scalewise(mod, bw_range = c(0, 150000))    # small scale
#'
#' ### The same fit, explored interactively over a basemap
#' # spCFmap(mod, crs = 25832)
#'
#' @importFrom fields rdist
#' @importFrom stats glm gaussian predict vcov qnorm sd as.formula glm.fit lm.wfit
#' @export
cf_dglm <- function(y, x = NULL, coords, time, offset = NULL,
                    x0 = NULL, coords0 = NULL, time0 = NULL, offset0 = NULL,
                    mod_hv, robust_se = TRUE, sill_cap = TRUE,
                    se_type = c("prediction", "mean"),
                    se_method = c("opt", "classic")) {
  se_method <- match.arg(se_method)
  se_type <- match.arg(se_type)

  .spcf_check_mod_hv(mod_hv, "cf_dglm_hv", "cf_dglm_hv")
  .spcf_check_data(y = y, x = x, coords = coords, offset = offset, time = time)
  .spcf_check_newdata(x = x, x0 = x0, coords0 = coords0, time0 = time0,
                      offset0 = offset0)
  if (!is.null(coords0) && is.null(time0))
    .spcf_stop("'time0' must be supplied together with 'coords0': every prediction site needs a time point.")

  family <- mod_hv$other$family
  bands  <- mod_hv$other$bands
  kernel <- mod_hv$other$kernel
  rho    <- mod_hv$other$rho; Q <- mod_hv$other$Q
  x_sel  <- mod_hv$other$x_sel; xname <- mod_hv$other$xname
  lev    <- mod_hv$other$time_levels
  sk     <- ifelse(is.null(mod_hv$other$seed), 4321, mod_hv$other$seed)
  tau    <- mod_hv$other$tau; if (is.null(tau) || !is.finite(tau) || tau <= 0) tau <- 1
  tv_cols <- mod_hv$other$tv_cols           # design-column indices (in X) with time-varying coefficient
  if (is.null(tv_cols)) tv_cols <- integer(0)
  q_tvc  <- mod_hv$other$q_tvc              # drift variance for the time-varying coefficients
  has_tv <- length(tv_cols) > 0
  n      <- length(y)
  coords <- as.matrix(coords)
  if (is.null(offset)) offset <- rep(0, n)
  has0   <- !is.null(coords0)

  if (has0) {
    if (!is.null(offset) && is.null(offset0) && any(offset != 0))
      stop("offset0 must be provided when offset is specified")
    if (!is.null(x) && is.null(x0)) stop("x0 must be provided when x is specified")
  }

  ## working time grid: extend to include any requested prediction times so that
  ## prediction is possible at time points with no observations. Columns absent
  ## from the training data carry no observation, so each per-knot AR(1) Kalman
  ## uses gain 0 there (predict only): interior gaps are interpolated and times
  ## beyond the last observed point are forecast (variance grows with horizon).
  lev_work <- lev
  if (has0 && !is.null(time0)) lev_work <- sort(unique(c(lev, time0)))

  ## ---- design matrices (intercept + selected covariates)
  if (is.null(x)) X <- matrix(1, n, 1) else { x <- as.matrix(x); X <- cbind(1, x[, x_sel, drop = FALSE]) }
  nx <- ncol(X)
  tv_cols <- tv_cols[tv_cols >= 1 & tv_cols <= nx & tv_cols != 1]  # never the intercept
  has_tv  <- length(tv_cols) > 0
  const_cols <- setdiff(seq_len(nx), tv_cols)                      # intercept + non-tv covariates

  ## ---- panels (training)
  pn  <- .dglm_panel(coords, time, time_levels = lev_work)
  nL  <- pn$nL; nT <- pn$nT
  Ctr <- pn$C

  ## ---- prediction-site panel (locations x same time grid)
  if (has0) {
    n0 <- nrow(coords0); coords0 <- as.matrix(coords0)
    if (is.null(offset0)) offset0 <- rep(0, n0)
    if (is.null(time0)) stop("time0 must be provided for prediction sites")
    X0  <- if (is.null(x)) matrix(1, n0, 1) else cbind(1, as.matrix(x0)[, x_sel, drop = FALSE])
    pn0 <- .dglm_panel(coords0, time0, time_levels = lev_work)
    Cpr <- pn0$C
  } else { Cpr <- NULL }

  f_obs <- function(field) field[cbind(pn$lk, pn$tk)]
  Xc <- X[, const_cols, drop = FALSE]                     # columns whose coef is constant
  Xtv <- X[, tv_cols, drop = FALSE]                       # columns whose coef is time-varying
  ## Knots and neighbourhoods depend only on the coordinates, so build them ONCE
  ## (train + grid) and reuse; the single sweep then only re-runs the residual-
  ## dependent C++ kernel per band.
  setups <- lapply(seq_along(bands), function(k)
    .dglm_scale_setup(Ctr, bands[k], kernel, sk, Cpr = Cpr))
  tvpart_of <- function(tvbeta) if (has_tv) rowSums(Xtv * tvbeta[pn$tk, , drop = FALSE]) else rep(0, n)

  ## ---- one coarse-to-fine cascade sweep (cf_glm-style, single pass). For each
  ## band, coarse to fine: (i) refresh the GLM working response/weights at the
  ## current linear predictor (IRLS folded into the sweep, as cf_glm's per-band
  ## glm() does); (ii) fit that scale to the working residual and accumulate its
  ## field; (iii) backfit the constant coefficients; (iv) re-estimate the
  ## time-varying coefficients. No outer iteration. `predict` toggles the
  ## prediction-site recombination (TRUE for the single final pass).
  casc_sweep <- function(beta, tvbeta, q_cur, predict = FALSE) {
    tvpart  <- tvpart_of(tvbeta)
    f       <- matrix(0, nL, nT)                            # cumulative field (link scale)
    Ftr_sum <- matrix(0, nL, nT)
    Fpr_sum <- if (has0 && predict) matrix(0, nrow(Cpr), nT) else NULL
    sc_list <- vector("list", length(bands)); z <- w <- NULL
    for (k in seq_along(bands)) {
      fobs <- f_obs(f)
      eta  <- .dglm_clip_l(drop(X %*% beta) + tvpart + fobs + offset, family)
      zw   <- .dglm_work(family, eta, y, offset); z <- zw$z; w <- zw$w
      resid <- z - drop(X %*% beta) - tvpart - fobs  # working residual (z is already offset-free)
      Rp <- matrix(NA_real_, nL, nT); Rp[cbind(pn$lk, pn$tk)] <- resid
      Wp <- matrix(NA_real_, nL, nT); Wp[cbind(pn$lk, pn$tk)] <- w
      sc <- .dglm_scale_apply(setups[[k]], Rp, Wp, rho, Q, predict = predict)
      f <- f + sc$Ftr; Ftr_sum <- Ftr_sum + sc$Ftr
      if (has0 && predict) Fpr_sum <- Fpr_sum + sc$Fpr
      sc_list[[k]] <- sc
      ## constant-coefficient backfit on the peeled residual
      robs <- (Rp - sc$Ftr)[cbind(pn$lk, pn$tk)]
      ba   <- stats::lm.wfit(Xc, robs, w)$coefficients; ba[!is.finite(ba)] <- 0
      beta[const_cols] <- beta[const_cols] + ba
      ## time-varying-coefficient update (field + constant part removed)
      if (has_tv) {
        r_tv <- z - drop(X %*% beta) - f_obs(f)   # z is already offset-free
        dr   <- .dglm_dynreg(r_tv, Xtv, w, pn$tk, nT, q = if (anyNA(q_cur)) NULL else q_cur)
        tvbeta <- dr$beta; if (anyNA(q_cur)) q_cur <- dr$q
        tvpart <- tvpart_of(tvbeta)
      }
    }
    list(beta = beta, tvbeta = tvbeta, q_cur = q_cur,
         Ftr = Ftr_sum, Fpr = Fpr_sum, scales = sc_list, z = z, w = w)
  }

  ## ---- initialize and run the single sweep
  beta <- stats::glm.fit(X, y, offset = offset, family = family)$coefficients
  tvbeta <- if (has_tv) matrix(0, nT, length(tv_cols)) else NULL
  if (has_tv) beta[tv_cols] <- 0                          # constant part holds const_cols only
  q_cur <- if (has_tv && !is.null(q_tvc) && all(is.finite(q_tvc)) && all(q_tvc > 0)) q_tvc else NA_real_

  Z <- Z_sd <- matrix(0, n, max(length(bands), 1L))
  Z0 <- Z0_sd <- if (has0) matrix(0, n0, max(length(bands), 1L)) else NULL
  f_tr <- rep(0, n); f0_obs <- if (has0) rep(0, n0) else NULL
  tvpart <- tvpart_of(tvbeta); z <- w <- NULL
  if (length(bands) == 0) {
    eta <- .dglm_clip_l(drop(X %*% beta) + tvpart + offset, family)
    zw  <- .dglm_work(family, eta, y, offset); z <- zw$z; w <- zw$w
    beta[const_cols] <- stats::lm.wfit(Xc, z - tvpart, w)$coefficients
  } else {
    sw <- casc_sweep(beta, tvbeta, q_cur, predict = TRUE)
    beta <- sw$beta; tvbeta <- sw$tvbeta; q_cur <- sw$q_cur; z <- sw$z; w <- sw$w
    for (k in seq_along(bands)) {
      Z[, k]    <- sw$scales[[k]]$Ftr[cbind(pn$lk, pn$tk)]
      Z_sd[, k] <- sqrt(sw$scales[[k]]$Vtr[cbind(pn$lk, pn$tk)])
      if (has0) {
        Z0[, k]    <- sw$scales[[k]]$Fpr[cbind(pn0$lk, pn0$tk)]
        Z0_sd[, k] <- sqrt(sw$scales[[k]]$Vpr[cbind(pn0$lk, pn0$tk)])
      }
    }
    ## center each scale to zero mean (folding the bias into the intercept via
    ## the final GLM), as cf_dglm / cf_glm do. Prediction is unchanged.
    zmean <- colMeans(Z); Z <- sweep(Z, 2, zmean)
    if (has0) Z0 <- sweep(Z0, 2, zmean)   # center by TRAINING means for consistency
    f_tr <- rowSums(Z); if (has0) f0_obs <- rowSums(Z0)
  }

  ## ---- final time-varying coefficients (smoothed value + per-time covariance)
  tvV <- NULL
  if (has_tv) {
    dr <- .dglm_dynreg(z - drop(X %*% beta) - f_tr, Xtv, w, pn$tk, nT,
                       q = if (anyNA(q_cur)) NULL else q_cur)
    tvbeta <- dr$beta; tvV <- dr$V; tvpart <- tvpart_of(tvbeta); q_tvc <- dr$q
  }

  ## ---- final GLM with the cascade field (and the time-varying part) as offset
  const_cov <- const_cols[const_cols != 1]                 # constant covariate columns (no intercept)
  ncv <- length(const_cov)
  cov_df <- if (ncv > 0) as.data.frame(X[, const_cov, drop = FALSE]) else NULL
  off_tr <- .dglm_clip_l(f_tr, family) + tvpart + offset
  dat <- data.frame(y = y, .off = off_tr)
  if (!is.null(cov_df)) { names(cov_df) <- xname[const_cov]; dat <- cbind(dat, cov_df) }
  form <- if (ncv > 0) as.formula(paste0("y ~ offset(.off) + ", paste(xname[const_cov], collapse = "+"))) else as.formula("y ~ offset(.off)")
  gmod <- stats::glm(form, data = dat, family = family)
  beta_int <- matrix(gmod$coefficients)
  ## design ordered as gmod (intercept, constant covariates) for variance propagation
  Xg <- if (ncv > 0) cbind(1, X[, const_cov, drop = FALSE]) else matrix(1, n, 1)
  Vbeta <- vcov(gmod)
  ## spatial-block cluster-robust covariance (default): the model-based vcov treats
  ## the cascade field as a known offset and badly understates Var(beta) because the
  ## residual is a correlated random field; .dglm_clusterSE puts the field back into
  ## the working residual and clusters over spatial blocks. Used for both the
  ## coefficient SEs and the coefficient-uncertainty term of the predictive SE.
  V_rob <- Vbeta; G_block <- NA_integer_
  if (robust_se && length(bands) > 0) {
    cse <- tryCatch(.dglm_clusterSE(y, Xg, beta_int, f_tr, tvpart, offset, family,
                                    coords, mod_hv$other$bands),
                    error = function(e) NULL)
    if (!is.null(cse)) { V_rob <- cse$V; G_block <- cse$G }
  }
  Vbeta <- V_rob
  beta_int_se <- sqrt(diag(Vbeta))
  beta_summ <- data.frame(coef = beta_int, coef_se = beta_int_se,
                          lower_95CI = beta_int - 1.96 * beta_int_se,
                          upper_95CI = beta_int + 1.96 * beta_int_se)
  row.names(beta_summ) <- c("Intercept", xname[const_cov])

  ## time-varying-coefficient variance contribution x_tv' V_t x_tv at each obs
  tvvar <- function(Xt, tk_) {
    if (!has_tv) return(rep(0, nrow(Xt)))
    v <- vapply(seq_len(nrow(Xt)), function(i) drop(Xt[i, ] %*% tvV[[tk_[i]]] %*% Xt[i, ]), numeric(1))
    pmax(v, 0)
  }

  ## sill cap: the field predictive variance cannot exceed the marginal variance
  ## of the fitted total field (link scale). Without it the gPoE variance
  ## V = 1/sum(phi/P) diverges where neighbours vanish (deep extrapolation). The
  ## cap is on the TOTAL field variance (never per scale -- scales are positively
  ## correlated, so sum_k var(Z_k) << var(sum_k Z_k) and per-scale caps undershoot).
  ## Disabled for binomial (bounded response, weak logit field) and when no scale.
  sill <- if (isTRUE(sill_cap) && family$family != "binomial" && length(bands) > 0) {
    sv <- stats::var(rowSums(Z)); if (is.finite(sv) && sv > 0) sv else Inf
  } else Inf
  ## opt+field coefficient covariance (default se_method): recomputed once the
  ## calibrated per-point field SD s_f = sqrt(pmin(tau*rowSums(Z_sd^2), sill)) is
  ## available, replacing the classic field-retained cluster-robust covariance.
  if (robust_se && se_method == "opt" && length(bands) > 0) {
    s_f <- sqrt(pmax(pmin(tau * rowSums(Z_sd^2), sill), 0))
    ofse <- tryCatch(.dglm_optfield_SE(y, Xg, beta_int, f_tr, s_f, tvpart, offset,
                                       family, coords, mod_hv$other$bands),
                     error = function(e) NULL)
    if (!is.null(ofse) && all(is.finite(diag(ofse$V))) && all(diag(ofse$V) > 0)) {
      Vbeta <- ofse$V; G_block <- ofse$G
      beta_int_se <- sqrt(diag(Vbeta))
      beta_summ <- data.frame(coef = beta_int, coef_se = beta_int_se,
                              lower_95CI = beta_int - 1.96 * beta_int_se,
                              upper_95CI = beta_int + 1.96 * beta_int_se)
      row.names(beta_summ) <- c("Intercept", xname[const_cov])
    }
  }
  pred     <- predict(gmod, type = "response")
  pred_lin <- predict(gmod, type = "link")
  pred_lin_sd <- sqrt(pmax(rowSums((Xg %*% Vbeta) * Xg) + tvvar(Xtv, pn$tk) +
                           pmin(tau * rowSums(Z_sd^2), sill), 0))
  pred_sd  <- abs(family$mu.eta(pred_lin)) * pred_lin_sd
  qs <- c(0.005, 0.025, 0.05, seq(0.1, 0.9, 0.1), 0.95, 0.975, 0.995)
  pred_q <- data.frame(family$linkinv(pred_lin + outer(pred_lin_sd, qnorm(qs), "*")))
  names(pred_q) <- paste0("q", qs)
  pred_ms <- data.frame(pred = pred, pred_sd = pred_sd)

  pred0_ms <- pred0_q <- NULL
  if (has0) {
    X0tv    <- X0[, tv_cols, drop = FALSE]
    tvpart0 <- if (has_tv) rowSums(X0tv * tvbeta[pn0$tk, , drop = FALSE]) else rep(0, n0)
    off0 <- .dglm_clip_l(f0_obs, family) + tvpart0 + offset0
    cov0 <- if (ncv > 0) as.data.frame(X0[, const_cov, drop = FALSE]) else NULL
    dat0 <- data.frame(.off = off0); if (!is.null(cov0)) { names(cov0) <- xname[const_cov]; dat0 <- cbind(dat0, cov0) }
    pred0     <- predict(gmod, newdata = dat0, type = "response")
    pred0_lin <- predict(gmod, newdata = dat0, type = "link")
    Xg0 <- if (ncv > 0) cbind(1, X0[, const_cov, drop = FALSE]) else matrix(1, n0, 1)
    pred0_lin_sd <- sqrt(pmax(rowSums((Xg0 %*% Vbeta) * Xg0) + tvvar(X0tv, pn0$tk) +
                              pmin(tau * rowSums(Z0_sd^2), sill), 0))
    pred0_sd  <- abs(family$mu.eta(pred0_lin)) * pred0_lin_sd
    pred0_q <- data.frame(family$linkinv(pred0_lin + outer(pred0_lin_sd, qnorm(qs), "*")))
    names(pred0_q) <- paste0("q", qs)
    pred0_ms <- data.frame(pred = pred0, pred_sd = pred0_sd)
  }

  ## ---- spatial-process objects (per scale)
  Zdf <- Zsd_df <- Z0df <- Z0sd_df <- NULL
  if (length(bands) > 0) {
    Zdf <- as.data.frame(Z); Zsd_df <- as.data.frame(Z_sd)
    names(Zdf) <- names(Zsd_df) <- paste0("scale", seq_along(bands))
    if (has0) {
      Z0df <- as.data.frame(Z0); Z0sd_df <- as.data.frame(Z0_sd)
      names(Z0df) <- names(Z0sd_df) <- paste0("scale", seq_along(bands))
    }
  }

  ## ---- time-varying coefficients (smoothed value and per-time SD), if any
  beta_tv <- beta_tv_sd <- NULL
  if (has_tv) {
    d_tv <- length(tv_cols)
    beta_tv    <- as.data.frame(matrix(tvbeta, nrow = nT, ncol = d_tv))
    beta_tv_sd <- as.data.frame(matrix(unlist(lapply(tvV, function(V) sqrt(pmax(diag(V), 0)))),
                                       nrow = nT, ncol = d_tv, byrow = TRUE))
    names(beta_tv) <- names(beta_tv_sd) <- xname[tv_cols]
    beta_tv$time <- beta_tv_sd$time <- lev_work
  }

  ## ---- sd summary
  if (length(bands) > 0) {
    elements <- c("xb", paste0("spatial_scale", seq_along(bands)))
    standard_deviation <- c(sd(Xg %*% beta_int), apply(Z, 2, sd))
  } else { elements <- "xb"; standard_deviation <- sd(Xg %*% beta_int) }
  if (has_tv) {
    elements <- c(elements, paste0("tv_", xname[tv_cols]))
    standard_deviation <- c(standard_deviation, apply(tvbeta, 2, sd))
  }
  sd_summary <- data.frame(elements, standard_deviation); row.names(sd_summary) <- NULL

  ## ---- validation error statistics (holdout from mod_hv)
  idt <- mod_hv$id_train
  yt <- y[-idt]; yp <- pred[-idt]
  yp_c <- switch(family$family,
                 binomial = pmin(pmax(yp, 1e-6), 1 - 1e-6),
                 poisson  = pmax(yp, 1e-8),
                 yp)
  gmod_null <- stats::glm(yt ~ 1, family = family)
  gmod_fix  <- stats::glm(yt ~ 0 + offset(family$linkfun(yp_c)), family = family)
  r2  <- 1 - gmod_fix$deviance / gmod_null$null.deviance
  rmse <- sqrt(mean((yt - yp)^2)); mae <- abs(mean(yt - yp))
  e_summary <- data.frame(stat = c("validation_Pseudo-R2", "validation_RMSE", "validation_MAE"),
                          value = c(r2, rmse, mae))

  other <- list(n = n, n0 = if (has0) n0 else NA, nx = nx, y = y,
                coords = coords, coords0 = coords0, rho = rho, Q = Q,
                kernel = kernel, beta_int_vmat = Vbeta, loss_hv = mod_hv$loss_hv,
                tau = tau, tv_cols = tv_cols, q_tvc = q_tvc,
                time = time, time0 = if (has0) time0 else NULL,
                x = x, x0 = if (has0) x0 else NULL,
                time_levels = lev_work, time_levels_train = lev,
                robust_se = robust_se, se_blocks = G_block)
  result <- list(beta = beta_summ, beta_tv = beta_tv, beta_tv_sd = beta_tv_sd,
                 sd_summary = sd_summary, e_summary = e_summary,
                 pred = pred_ms, pred0 = pred0_ms, pred_q = pred_q, pred0_q = pred0_q,
                 bands = bands, Z = Zdf, Z_sd = Zsd_df, Z0 = Z0df, Z0_sd = Z0sd_df,
                 other = other, call = match.call())
  if (identical(se_type, "prediction")) {
    ob <- tryCatch(.spcf_obs_predict(family = family, y = y, mod_hv = mod_hv,
                     pred_in = result$pred$pred, predq_in = result$pred_q,
                     pred_out = result$pred0$pred, predq_out = result$pred0_q),
                   error = function(e) NULL)
    result <- .spcf_apply_obs(result, ob)
  } else result$other$se_type <- "mean"
  class(result) <- "cf_dglm"
  result
}

#' @noRd
#' @export
print.cf_dglm <- function(x, ...) {
  cat("Call:\n"); print(x$call)
  cat("\n---- Coefficients -------------------------------------\n")
  print(x$beta)
  st  <- x$other[c("rho", "Q")]
  ok  <- vapply(st, function(z) is.numeric(z) && length(z) == 1L && is.finite(z),
                logical(1))
  if (any(ok)) {
    lab <- c(rho = "Temporal autocorrelation, AR(1) (rho)",
             Q   = "Temporal innovation variance (Q)")
    cat("\n---- Space-time parameters ----------------------------\n")
    print(data.frame(parameter = unname(lab[names(st)[ok]]),
                     value     = round(unlist(st[ok]), 4)),
          row.names = FALSE, right = FALSE)
  }
  cat("\n---- Standard deviations (model elements) -------------\n")
  print(x$sd_summary)
  cat("\n---- Error statistics ---------------------------------\n")
  print(x$e_summary)
  invisible(x)
}
