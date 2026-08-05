#' Holdout validation for coarse-to-fine dynamic (space-time) spatial GLMMs
#'
#' Trains a coarse-to-fine dynamic spatial GLMM (CF-DGLMM) and selects the
#' spatial scales of a separable space-time cascade through progressive holdout
#' validation. The companion \code{\link{cf_dglm}} refits the selected structure
#' on the full sample and predicts. The model decomposes the link-scale linear
#' predictor as \eqn{g(\mu_{i,t}) = x_{i,t}'\beta + \sum_k f_k(s_i,t) + }
#' offset, where each scale-\eqn{k} field \eqn{f_k} is a per-knot AR(1) Kalman
#' smoother in time combined with kernel kriging in space.
#'
#' @param y Vector of response variables (N x 1) including continuous, count,
#'   and binary responses, following an exponential family distribution.
#' @param x Matrix of covariates (N x K).
#' @param coords Matrix of 2-dimensional point coordinates (N x 2). Rows sharing
#'   the same coordinates are treated as repeated observations of one location
#'   across time. The space-time panel may be unbalanced: the set of observed
#'   locations is allowed to differ from one time point to another (knots are
#'   placed on the union of locations and the per-knot AR(1) smoother bridges
#'   time points at which a knot has no nearby observation).
#' @param time Vector of time indices (N x 1) identifying the time point of each
#'   observation. Any sortable type (integer, numeric, Date) is accepted.
#' @param offset Optional. Vector of offset variable (N x 1) to be included in
#'   the linear predictor, consistent with \code{\link{glm}}.
#' @param train_rat Training sample ratio (default: 0.75). Holdout is performed
#'   at the \emph{location} level: a subset of locations (and all their time
#'   points) is held out for validation.
#' @param id_train Optional. If specified, the corresponding samples are used as
#'   training samples; otherwise locations are chosen based on \code{train_rat}.
#' @param alpha Decay ratio of the kernel bandwidth in the coarse-to-fine
#'   training (default: 0.9).
#' @param kernel Kernel type for spatial dependence. \code{"exp"} for the
#'   exponential kernel (default) and \code{"gau"} for the Gaussian kernel.
#' @param family Error distribution and link function, consistent with the
#'   \code{family} argument of \code{\link{glm}}. Functionality has been
#'   confirmed for \code{gaussian()}, \code{poisson()}, and \code{binomial()}.
#' @param rho,Q Optional AR(1) temporal parameters (autocorrelation and
#'   innovation variance). When \code{NULL} (default) a single global
#'   \code{(rho, Q)} is estimated by maximum marginal likelihood.
#' @param tvc Optional. Covariates whose regression coefficients are allowed to
#'   vary over time, given as covariate names or as integer column indices into
#'   \code{x}. The remaining coefficients are constant. The intercept is always
#'   kept constant (a time-varying intercept is confounded with the temporal mean
#'   of the spatial field). \code{NULL} (default) keeps all coefficients constant.
#' @param q_tvc Optional. Innovation (drift) variance of the random walk followed
#'   by the time-varying coefficients. When \code{NULL} (default) it is estimated
#'   from the data.
#' @param seed Random seed for the training/validation split and knot placement
#'   (default \code{1234}). Set to \code{NULL} for a random split.
#'
#' @return A list of class \code{"cf_dglm_hv"} with the following elements:
#' \describe{
#'   \item{loss_hv}{Holdout deviance of the selected model, evaluated at the
#'   validation locations. Fits of the same data share the same split, so this
#'   value compares models directly, whatever number of scales each selected.}
#'   \item{loss_hv_all}{The validation loss after every learning step.}
#'   \item{e_summary}{Out-of-sample accuracy at the validation locations of the
#'   model trained on the training locations only: deviance-based pseudo
#'   R-squared (\code{validation_Pseudo-R2}, ordinary R-squared in the Gaussian
#'   case), \code{validation_RMSE} and \code{validation_MAE}. Unlike the
#'   \code{e_summary} of \code{\link{cf_dglm}}, which scores the full-sample
#'   refit at those same points, this one never saw them.}
#'   \item{val_pred}{The validation predictions behind \code{e_summary}: one row
#'   per held-out observation with its location index (\code{loc}), time point
#'   (\code{time}), observed response (\code{y}) and predicted mean
#'   (\code{pred}) on the response scale.}
#'   \item{id_train}{Row indices of the training observations.}
#'   \item{other}{Internal objects reused by \code{\link{cf_dglm}}.}
#'   \item{call}{The matched call.}
#' }
#'
#' @references
#' Murakami, D. (2026).
#' Fast covariance-free spatiotemporal modeling via coarse-to-fine learning.
#' *ArXiv preprint*.
#'
#' @seealso \code{\link{cf_dglm}}, \code{\link{cf_glm_hv}}
#' @author Daisuke Murakami
#'
#' @importFrom fields rdist
#' @importFrom nloptr nloptr
#' @importFrom stats glm gaussian predict residuals sd kmeans
#' @export
cf_dglm_hv <- function(y, x = NULL, coords, time, offset = NULL,
                       train_rat = 0.75, id_train = NULL, alpha = 0.9,
                       kernel = "exp", family = gaussian(),
                       rho = NULL, Q = NULL, tvc = NULL, q_tvc = NULL, seed = 1234) {

  .spcf_check_data(y = y, x = x, coords = coords, offset = offset, time = time)
  .spcf_check_hv_args(length(y), train_rat, id_train, alpha, kernel)
  if (!is.null(q_tvc) && (!is.numeric(q_tvc) || anyNA(q_tvc) ||
                          any(!is.finite(q_tvc)) || any(q_tvc <= 0)))
    .spcf_stop("'q_tvc' must be positive and finite (a scalar, or one value per covariate in 'tvc').")
  q_user <- q_tvc                    # NULL unless the drift is fixed by the user

  n      <- length(y)
  coords <- as.matrix(coords)
  if (is.null(offset)) offset <- rep(0, n)

  ## ---- covariates (intercept + non-constant columns), as in initial_fun_glm
  xname <- "Intercept"; x_sel <- logical(0)
  if (is.null(x)) { x <- matrix(1, n, 1) } else {
    x <- as.matrix(x); x_sel <- apply(x, 2, sd) != 0
    if (sum(x_sel) >= 1) xname <- c("Intercept", colnames(x)[x_sel])
    if (is.null(colnames(x))) xname <- c("Intercept", paste0("x", which(x_sel)))
    x <- cbind(1, x[, x_sel, drop = FALSE])
  }
  nx <- ncol(x)

  ## ---- resolve which design columns have a time-varying coefficient. `tvc` may
  ## be covariate names or integer indices into the original covariate matrix.
  ## The intercept is ALWAYS kept constant: a time-varying intercept is a global
  ## temporal trend that is confounded with the temporal mean of the spatial
  ## field, so it is never allowed (requests for it are dropped with a message).
  tv_cols <- integer(0)
  if (!is.null(tvc) && sum(x_sel) > 0) {
    kept <- which(x_sel)
    if (is.character(tvc)) {
      if (any(tvc %in% c("Intercept", "(Intercept)")))
        message("cf_dglm_hv: the intercept is kept constant (not time-varying) ",
                "to avoid confounding with the spatial field.")
      tv_cols <- match(tvc, xname)                         # positions in the design (incl intercept)
      tv_cols <- tv_cols[!is.na(tv_cols)]
    } else {
      sel <- tvc[tvc %in% kept]
      tv_cols <- 1L + match(sel, kept)                     # +1 for the intercept column
    }
    tv_cols <- sort(unique(tv_cols[tv_cols >= 2 & tv_cols <= nx]))  # never the intercept (col 1)
  }
  has_tv <- length(tv_cols) > 0
  const_cols <- setdiff(seq_len(nx), tv_cols)              # intercept is always here

  ## ---- panels keyed on (location, time); location-level holdout split
  pn   <- .dglm_panel(coords, time)               # gives lk (loc id), tk, levels
  nL   <- pn$nL; nT <- pn$nT; lev <- pn$time_levels
  if (is.null(id_train)) {
    K_tr <- max(1L, round(nL * train_rat))
    do_split <- function() {
      if (nL > 30000 || K_tr >= nL) sort(sample.int(nL, K_tr)) else {
        ## kmeans the smaller of training/validation and derive the other via
        ## set complement, as initial_fun()/initial_fun_glm() do. Clustering the
        ## larger side means running kmeans with K close to nL, where the
        ## partition is near-degenerate: the selected locations (and hence the
        ## whole holdout split) then flip under coordinate changes as small as
        ## the last floating-point digit, e.g. between coords/1000 and
        ## coords*1e-3.
        K_val      <- nL - K_tr
        pick_train <- K_tr <= K_val
        K          <- max(1L, if (pick_train) K_tr else K_val)
        suppressWarnings(ck <- stats::kmeans(pn$C, min(K, nL - 1L),
                                             iter.max = ifelse(nL > 5000, 5L, 10L))$centers)
        sel <- sort(FNN::get.knnx(pn$C, ck, 1)$nn.index)
        if (pick_train) sel else setdiff(seq_len(nL), sel)
      }
    }
    tr_loc   <- if (is.null(seed)) do_split() else withr::with_seed(seed, do_split())
    id_train <- which(pn$lk %in% tr_loc)
  } else {
    tr_loc <- sort(unique(pn$lk[id_train]))
  }

  ## ---- linearize once: pooled GLM working response/weight (Gaussian surrogate)
  beta <- stats::glm.fit(x, y, offset = offset, family = family)$coefficients
  eta  <- drop(x %*% beta) + offset
  zw   <- .dglm_work(family, eta, y, offset)
  beta_c <- beta; if (has_tv) beta_c[tv_cols] <- 0   # keep tv covariate effect in the residual
  r    <- zw$z - drop(x %*% beta_c)                  # link-scale residual
  Rp   <- matrix(NA_real_, nL, nT); Rp[cbind(pn$lk, pn$tk)] <- r
  Wp   <- matrix(NA_real_, nL, nT); Wp[cbind(pn$lk, pn$tk)] <- zw$w

  fi <- which(seq_len(nL) %in% tr_loc); vi <- setdiff(seq_len(nL), fi)
  if (length(vi) == 0) { vi <- fi }                 # degenerate (train_rat=1)
  Cfit <- pn$C[fi, , drop = FALSE]; Cval <- pn$C[vi, , drop = FALSE]
  Rfit <- Rp[fi, , drop = FALSE];   Rval <- Rp[vi, , drop = FALSE]
  Wfit <- Wp[fi, , drop = FALSE];   Wval <- Wp[vi, , drop = FALSE]

  ## observation <-> fit/val panel maps (for the per-scale GLM re-linearization)
  obs_f <- which(pn$lk %in% fi); fi_row <- match(pn$lk[obs_f], fi); fi_col <- pn$tk[obs_f]
  obs_v <- which(pn$lk %in% vi); vi_row <- match(pn$lk[obs_v], vi); vi_col <- pn$tk[obs_v]
  xval <- x[obs_v, const_cols, drop = FALSE]

  ## ---- time-varying coefficients: estimate (and the drift q) on the fit set by
  ## dynamic regression, then peel from both fit and validation residuals so the
  ## (rho,Q) estimate and the scale selection see the tv-adjusted residual.
  tvbeta <- NULL
  if (has_tv) {
    dr <- .dglm_dynreg(r[obs_f], x[obs_f, tv_cols, drop = FALSE], zw$w[obs_f],
                       pn$tk[obs_f], nT, q = q_user)
    tvbeta <- dr$beta; q_tvc <- dr$q
    tvp <- rowSums(x[, tv_cols, drop = FALSE] * tvbeta[pn$tk, , drop = FALSE])
    Rfit[cbind(fi_row, fi_col)] <- Rfit[cbind(fi_row, fi_col)] - tvp[obs_f]
    Rval[cbind(vi_row, vi_col)] <- Rval[cbind(vi_row, vi_col)] - tvp[obs_v]
  }

  ## ---- bandwidth grid (geometric), consistent with cf_glm_hv
  max_d     <- sqrt(diff(range(pn$C[, 1]))^2 + diff(range(pn$C[, 2]))^2) / 3
  Bands_max <- 100L
  Bands     <- max_d * alpha^(1:Bands_max)
  ## floor the bandwidth grid at ~half the typical inter-point spacing: bands
  ## finer than the data resolution carry no information and make the kernel
  ## exp(-d/b) underflow to 0 (empty knot weights -> 0/0). This caps the grid
  ## so the cascade cannot commit sub-resolution scales (notably at small nT,
  ## where the greedy search would otherwise over-fit noise to tiny bands).
  ## The spacing is the per-time-point median nearest-neighbour distance,
  ## averaged over times: pooling all times into one set under-states the true
  ## resolution when the same site is recorded at slightly shifted coordinates
  ## across times (cm-scale cross-time near-duplicates), which would collapse
  ## the floor; the within-time median reflects the genuine snapshot spacing.
  per_time_nn <- vapply(seq_len(nT), function(t) {
    lt <- unique(pn$lk[pn$tk == t])
    if (length(lt) < 2L) return(NA_real_)
    stats::median(FNN::get.knn(pn$C[lt, , drop = FALSE], k = 1)$nn.dist)
  }, numeric(1))
  band_min  <- 0.5 * mean(per_time_nn, na.rm = TRUE)
  if (!is.finite(band_min)) band_min <- 0          # degenerate (single-time) panels
  Bands     <- Bands[Bands >= band_min]
  if (length(Bands) == 0L) Bands <- max_d * alpha   # safety for degenerate cases

  ## ---- global AR(1) (rho, Q): ML on a reference coarse band unless supplied
  if (is.null(rho) || is.null(Q)) {
    kn  <- .dglm_knots(unique(Cfit), Bands[1], seed = ifelse(is.null(seed), 4321, seed))
    Wf  <- .dglm_kmat(Cfit, kn, Bands[1], kernel)
    ag  <- .dglm_aggregate(Wf, Rfit, Wfit)         # NA-safe (irregular panels)
    ml  <- .dglm_ar1_ml(ag$Z, ag$Rmat)
    if (is.null(rho)) rho <- ml$rho
    if (is.null(Q))   Q   <- ml$Q
  }

  ## ---- greedy scale selection on EXACT validation deviance (patience = 5)
  ## cf_glm_hv-identical: for every candidate band, refit the constant coefs via a
  ## GLM with the (cumulative + candidate) field as offset, then accept the band
  ## iff it lowers the holdout family deviance (sum of dev.resids at validation
  ## observations) -- not the working-weighted SSE surrogate. On acceptance the
  ## working response/weights are recomputed at the refit linearization so the next
  ## scale sees the updated residual. Gaussian: numerically identical to the SSE rule.
  yv_obs <- y[obs_v]; off_v <- offset[obs_v]
  tvp_v  <- if (has_tv) tvp[obs_v] else rep(0, length(obs_v))
  mu_floor <- if (family$family == "poisson") 1e-8 else 0
  b_const <- beta[const_cols]                       # running constant-coef vector (incl. intercept)
  dev_of <- function(field_mat) {
    fld <- field_mat[cbind(vi_row, vi_col)]
    eta <- drop(xval %*% b_const) + off_v + tvp_v + fld
    mu  <- pmax(family$linkinv(.dglm_clip_l(eta, family)), mu_floor)
    sum(family$dev.resids(yv_obs, mu, 1), na.rm = TRUE)
  }
  message("--- Validation deviance: Basic GLM ---")
  pred_val <- matrix(0, length(vi), nT); Vval <- matrix(0, length(vi), nT); committed <- numeric(0)
  cumF_fit <- matrix(0, length(fi), nT)            # cumulative committed field at fit obs (link)
  xfc <- x[obs_f, const_cols, drop = FALSE]; offf <- offset[obs_f]
  tvpf <- if (has_tv) tvp[obs_f] else rep(0, length(obs_f))
  best <- dev_of(pred_val)
  message(format(best))
  message("--- Validation deviance: Learning multi-scale space-time process ---")
  count <- 0L; accept_num <- 5L; sk <- ifelse(is.null(seed), 4321, seed)
  Loss <- best; Loss_name <- "basic GLM"
  for (i in seq_along(Bands)) {
    b  <- Bands[i]
    sc <- .dglm_scale(Cfit, Rfit, Wfit, b, rho, Q, kernel, sk, Cpr = Cval)
    ## cf_glm_hv-IDENTICAL: for EVERY candidate band, refit the constant coefs via
    ## a GLM with (cumulative + candidate) field as offset, then judge the candidate
    ## by the resulting holdout deviance. Accept iff it lowers that deviance.
    cumF_fit_try <- cumF_fit + sc$Ftr
    Of_try <- offf + cumF_fit_try[cbind(fi_row, fi_col)] + tvpf
    g_try  <- stats::glm.fit(xfc, y[obs_f], offset = Of_try, family = family,
                             control = stats::glm.control(maxit = 25))
    b_try  <- g_try$coefficients; b_try[!is.finite(b_try)] <- 0
    cumF_val_try <- pred_val + sc$Fpr
    eta_v  <- drop(xval %*% b_try) + off_v + tvp_v + cumF_val_try[cbind(vi_row, vi_col)]
    mu_v   <- pmax(family$linkinv(.dglm_clip_l(eta_v, family)), mu_floor)
    trial  <- sum(family$dev.resids(yv_obs, mu_v, 1), na.rm = TRUE)
    if (trial < best - 1e-8) {
      pred_val <- cumF_val_try; Vval <- Vval + sc$Vpr; cumF_fit <- cumF_fit_try
      b_const  <- b_try; committed <- c(committed, b); count <- 0L; comment <- ""
      ## refresh the working response/weights at the refit linearization so the
      ## next scale fits the updated working residual (field absorbed in offset).
      eta_f <- drop(xfc %*% b_const) + Of_try
      zwf <- .dglm_work(family, eta_f, y[obs_f], offf)
      Rfit[] <- NA_real_; Wfit[] <- NA_real_
      Rfit[cbind(fi_row, fi_col)] <- zwf$z - (eta_f - offf)   # = (y-mu)/mu' (field in offset)
      Wfit[cbind(fi_row, fi_col)] <- zwf$w
      best <- trial
    } else { if (i > 10) count <- count + 1L; comment <- " no improvement" }
    Loss <- c(Loss, best); Loss_name <- c(Loss_name, paste0("scale ", i))
    message(paste0(formatC(best, digits = 7, format = "g"), " (Scale ", i, ")", comment))
    if (count == accept_num) break
  }

  ## ---- re-estimate the drift q on the spatially-cleaned residual (2nd pass):
  ## after scale selection Rfit holds the fit residual with the committed spatial
  ## scales (and constant coefs) peeled; adding the tv part back gives the
  ## spatial-removed, tv-present residual, on which q is no longer inflated by the
  ## (unremoved) spatial field as it was in the pre-selection estimate above.
  if (has_tv) {
    ## Only re-estimate when the drift was not fixed by the user: this value is
    ## what cf_dglm() reuses, so overwriting it here would silently ignore q_tvc.
    if (is.null(q_user)) {
      clean_fit <- Rfit[cbind(fi_row, fi_col)] + tvp[obs_f]
      q_tvc <- .dglm_dynreg(clean_fit, x[obs_f, tv_cols, drop = FALSE], zw$w[obs_f],
                            pn$tk[obs_f], nT, q = NULL)$q
    } else {
      q_tvc <- rep_len(q_user, length(tv_cols))
    }
  }

  ## ---- holdout deviance of the selected model (uses the re-linearized beta)
  loss_hv <- dev_of(pred_val)

  loss_hv_all <- data.frame(learning = Loss_name, loss_hv = Loss)

  ## ---- predictive-variance calibration scalar (holdout), regularized.
  ## The field-variance signal is weak relative to observation noise, so we
  ## isolate it by noise subtraction (a moment estimator): tau_raw =
  ## (holdout squared error - noise) / (holdout field variance). The noise
  ## sigma^2 is the working-weighted in-sample residual variance left after the
  ## committed cascade. This moment estimate is what correctly recovers the
  ## (small) field error at long panels, but it is noisy at short panels, so we
  ## shrink log(tau) toward 0 (tau -> 1, no inflation) by a reliability weight
  ## rel = num^2 / (num^2 + SE(num)^2): when the noise-removed signal num is
  ## within its sampling error (few/short holdout) tau -> 1; when num is clearly
  ## resolved (long panels) tau -> tau_raw. This preserves the long-panel
  ## calibration while stabilizing short panels.
  okf  <- !is.na(Rfit) & is.finite(Wfit) & (Wfit > 0)
  sig2 <- if (any(okf)) sum(Wfit[okf] * Rfit[okf]^2) / sum(Wfit[okf]) else 0
  ## re-linearized validation working residual (field absorbed in eta) for tau
  eta_v <- drop(xval %*% b_const) + off_v + tvp_v + pred_val[cbind(vi_row, vi_col)]
  zwv   <- .dglm_work(family, eta_v, yv_obs, off_v)
  Rval[] <- NA_real_; Wval[] <- NA_real_
  Rval[cbind(vi_row, vi_col)] <- zwv$z - (eta_v - off_v)
  Wval[cbind(vi_row, vi_col)] <- zwv$w
  resid_val <- Rval                              # already the noise part (field removed)
  okv  <- !is.na(resid_val) & is.finite(Wval) & (Wval > 0) & is.finite(Vval) & (Vval > 0)
  if (any(okv)) {
    Wv <- Wval[okv]; r2 <- resid_val[okv]^2; Vv <- Vval[okv]
    verr <- sum(Wv * r2) / sum(Wv)              # weighted holdout mean square
    vfld <- sum(Wv * Vv) / sum(Wv)              # weighted holdout field variance
    num  <- verr - sig2                         # noise-removed field error variance
    se   <- sqrt(2 / max(sum(okv), 1)) * verr   # approx. SE of verr (mean of squares)
    rel  <- if (num > 0 && is.finite(se) && se > 0) num^2 / (num^2 + se^2) else 0
    tau_raw <- if (vfld > 0) max(num, 1e-6) / vfld else 1
    tau  <- min(max(exp(log(tau_raw) * rel), 1e-2), 1e2)
  } else tau <- 1
  if (!is.finite(tau)) tau <- 1

  ## ---- genuine out-of-sample validation metrics: the selected model is trained
  ## on the training locations only and evaluated at the held-out (validation)
  ## locations. muv is E[y] on the response scale; pseudo-R2 is deviance-based
  ## against an intercept-only null fitted on the same holdout (Gaussian: ordinary
  ## R2 = 1 - SSE/SST). Unlike cf_dglm's e_summary -- which scores the FULL-sample
  ## refit at the holdout points (field estimated using them, optimistic) -- this
  ## is a true out-of-sample score.
  muv_obs <- family$linkinv(.dglm_clip_l(eta_v, family))
  okp <- is.finite(muv_obs) & is.finite(yv_obs)
  mu_lo <- if (family$family == "poisson") 1e-8 else 0
  rmse_hv <- sqrt(mean((yv_obs[okp] - muv_obs[okp])^2))
  mae_hv  <- mean(abs(yv_obs[okp] - muv_obs[okp]))
  mu0   <- suppressWarnings(stats::glm(yv_obs[okp] ~ 1, family = family)$fitted.values)
  dnull <- sum(family$dev.resids(yv_obs[okp], mu0, 1))
  dres  <- sum(family$dev.resids(yv_obs[okp], pmax(muv_obs[okp], mu_lo), 1))
  r2_hv <- if (is.finite(dnull) && dnull > 0) 1 - dres / dnull else NA_real_
  e_summary <- data.frame(stat = c("validation_Pseudo-R2", "validation_RMSE", "validation_MAE"),
                          value = c(r2_hv, rmse_hv, mae_hv))
  val_pred <- data.frame(loc = vi[vi_row], time = lev[vi_col],
                         y = yv_obs, pred = muv_obs)

  other <- list(bands = committed, bands_all = Bands, alpha = alpha,
                kernel = kernel, family = family, rho = rho, Q = Q,
                sigma = sqrt(max(sig2, 0)),            # data-noise SD (link/working scale)
                x_sel = x_sel, xname = xname, seed = seed,
                time_levels = lev, tau = tau, tv_cols = tv_cols, q_tvc = q_tvc)
  result <- list(loss_hv = loss_hv, loss_hv_all = loss_hv_all, e_summary = e_summary,
                 val_pred = val_pred, id_train = id_train, other = other, call = match.call())
  class(result) <- "cf_dglm_hv"
  result
}

#' @noRd
#' @export
print.cf_dglm_hv <- function(x, ...) {
  cat("Call:\n"); print(x$call)
  cat("\n---- Validation SSE trace over scales ----\n")
  print(x$loss_hv_all)
  cat(sprintf("\nSelected scales: %d  | AR(1): rho=%.3f, Q=%.3g\n",
              length(x$other$bands), x$other$rho, x$other$Q))
  cat("\n---- Out-of-sample validation metrics ----\n")
  print(x$e_summary, row.names = FALSE)
  invisible(x)
}
