## ---------------------------------------------------------------------------
## Observation (data-distribution) predictive with holdout calibration.
##
## Converts the SIGNAL predictive (mean uncertainty, as carried in pred_q) into
## the OBSERVATION predictive for a new data point, and calibrates it on the
## cf_*_hv holdout (out-of-fold) samples so that it is verifiable on data:
##   gaussian : N(mean, signal_var + sigma^2), split-conformal SD scaling
##   poisson  : NegBin count predictive (Poisson x lognormal-lambda),
##              holdout scaling of the mean-uncertainty component
##   binomial : Bernoulli(p_calibrated); p from holdout temperature scaling,
##              pred_sd = sqrt(p(1-p)) (interval coverage is degenerate for
##              binary, so calibration is on the probability)
##
## Works purely from (pred, pred_q) + the family + the holdout info in mod_hv,
## so it is shared unchanged by cf_lm / cf_glm / cf_dglm and does not touch
## their internal variance construction. Returns replacement pred_sd / pred_q
## (and, for binomial, a calibrated pred); the caller keeps the signal versions
## in separate fields so nothing is lost (non-destructive).
## ---------------------------------------------------------------------------
.spcf_obs_predict <- function(family, y, mod_hv,
                              pred_in, predq_in,
                              pred_out = NULL, predq_out = NULL) {
  fam <- family$family
  qs  <- c(0.005, 0.025, 0.05, seq(0.1, 0.9, 0.1), 0.95, 0.975, 0.995)
  znm <- paste0("q", qs)
  clp <- function(p) pmin(pmax(p, 1e-8), 1 - 1e-8)
  lk  <- switch(fam,
                gaussian = function(p) p,
                poisson  = function(p) log(pmax(p, 1e-8)),
                binomial = function(p) stats::qlogis(clp(p)))
  ## link-scale signal SD recovered from the signal quantiles
  slink <- function(q) {
    if (is.null(q) || is.null(q[["q0.975"]])) return(NULL)
    (lk(q[["q0.975"]]) - lk(q[["q0.025"]])) / (2 * 1.96)
  }
  s_in  <- slink(predq_in)
  s_out <- slink(predq_out)
  if (is.null(s_in)) return(NULL)                    # nothing to build from

  ## out-of-fold holdout predictions + observed y (from cf_*_hv)
  idt     <- mod_hv$id_train
  hvp     <- mod_hv$other$pred
  val     <- if (!is.null(idt)) setdiff(seq_along(y), idt) else integer(0)
  haveval <- length(val) >= 10 && !is.null(hvp) && all(is.finite(hvp[val]))

  out <- list(type = fam)

  if (fam == "gaussian") {
    sig2 <- if (haveval) mean((y[val] - hvp[val])^2) else mean((y - pred_in)^2)
    obs_in  <- sqrt(s_in^2 + sig2)
    obs_out <- if (!is.null(s_out)) sqrt(s_out^2 + sig2) else NULL
    cc <- 1
    if (haveval) {
      sv <- sqrt(s_in[val]^2 + sig2)
      cc <- as.numeric(stats::quantile(abs(y[val] - hvp[val]) / sv, 0.95,
                                       names = FALSE)) / 1.96
      if (!is.finite(cc) || cc <= 0) cc <- 1
    }
    mkq <- function(mu, s) {
      d <- as.data.frame(mu + outer(cc * s, stats::qnorm(qs)))
      names(d) <- znm; d
    }
    out$pred_sd <- cc * obs_in
    out$pred_q  <- mkq(pred_in, obs_in)
    if (!is.null(obs_out)) { out$pred0_sd <- cc * obs_out; out$pred0_q <- mkq(pred_out, obs_out) }
    out$calib <- list(type = "gaussian_conformal", sigma2 = sig2, scale = cc)

  } else if (fam == "poisson") {
    cc <- 1
    if (haveval) {
      sv <- s_in[val]; lamv <- pmax(hvp[val], 1e-8)
      cov_at <- function(c) {
        sz <- 1 / pmax((c * sv)^2, 1e-8)
        mean(y[val] >= stats::qnbinom(.025, size = sz, mu = lamv) &
             y[val] <= stats::qnbinom(.975, size = sz, mu = lamv))
      }
      grid <- seq(0.05, 1.5, 0.05)
      cc <- grid[which.min(abs(vapply(grid, cov_at, numeric(1)) - 0.95))]
    }
    nb_sd <- function(mu, s) sqrt(mu + (mu * cc * s)^2)
    mkq <- function(mu, s) {
      sz <- 1 / pmax((cc * s)^2, 1e-8)
      d <- as.data.frame(vapply(qs, function(a) stats::qnbinom(a, size = sz, mu = mu),
                                numeric(length(mu))))
      names(d) <- znm; d
    }
    out$pred_sd <- nb_sd(pred_in, s_in); out$pred_q <- mkq(pred_in, s_in)
    if (!is.null(s_out)) { out$pred0_sd <- nb_sd(pred_out, s_out); out$pred0_q <- mkq(pred_out, s_out) }
    out$calib <- list(type = "poisson_negbin", scale = cc)

  } else if (fam == "binomial") {
    Tt <- 1
    if (haveval) {
      mv <- stats::qlogis(clp(hvp[val]))
      Tt <- tryCatch(stats::optimize(function(T) {
        pc <- clp(stats::plogis(mv / T))
        -mean(y[val] * log(pc) + (1 - y[val]) * log(1 - pc))
      }, c(0.3, 3))$minimum, error = function(e) 1)
    }
    pcal <- function(mu) stats::plogis(stats::qlogis(clp(mu)) / Tt)
    p_in <- pcal(pred_in)
    out$pred <- p_in; out$pred_sd <- sqrt(p_in * (1 - p_in))
    if (!is.null(pred_out)) {
      p_out <- pcal(pred_out); out$pred0 <- p_out; out$pred0_sd <- sqrt(p_out * (1 - p_out))
    }
    out$calib <- list(type = "binomial_temperature", temperature = Tt)
    out$binary <- TRUE                               # interval coverage degenerate
  } else {
    return(NULL)
  }
  out
}

## Apply .spcf_obs_predict output onto a cf_* result's prediction fields,
## preserving the signal versions as *_signal (non-destructive).
.spcf_apply_obs <- function(res, ob) {
  if (is.null(ob)) return(res)
  res$pred_signal   <- res$pred
  res$pred_q_signal <- res$pred_q
  if (!is.null(res$pred0))   res$pred0_signal   <- res$pred0
  if (!is.null(res$pred0_q)) res$pred0_q_signal <- res$pred0_q
  ## in-sample
  if (!is.null(ob$pred))    res$pred$pred       <- ob$pred        # binomial: calibrated prob
  if (!is.null(ob$pred_sd)) res$pred$pred_sd    <- ob$pred_sd
  if (!is.null(ob$pred_q))  res$pred_q          <- ob$pred_q
  ## out-of-sample
  if (!is.null(res$pred0)) {
    if (!is.null(ob$pred0))    res$pred0$pred    <- ob$pred0
    if (!is.null(ob$pred0_sd)) res$pred0$pred_sd <- ob$pred0_sd
    if (!is.null(ob$pred0_q))  res$pred0_q       <- ob$pred0_q
  }
  res$other$se_type      <- "prediction"
  res$other$calibration  <- ob$calib
  res$other$binary_pred  <- isTRUE(ob$binary)
  res
}
