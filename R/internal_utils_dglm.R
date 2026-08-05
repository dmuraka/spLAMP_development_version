## Internal utilities for the coarse-to-fine *dynamic* spatial GLMM
## (separable cascade with full-train refit): cf_dglm() / cf_dglm_hv().
##
## The model is a separable space-time decomposition on the link scale:
##   g(mu_{i,t}) = x_{i,t}' beta + sum_k f_k(s_i, t) + offset_{i,t}
## where each scale-k field f_k is obtained by (i) a per-knot AR(1) Kalman
## smoother in time and (ii) an exponential/Gaussian-kernel kriging in space.
## Coefficients are fitted by outer IRLS so that any glm() family is handled
## generically through family$linkinv / family$mu.eta / family$variance.
##
## These helpers intentionally mirror the conventions of internal_utils_glm.R
## (kernel thresholds, the 1.5*area/band^2 knot rule, .spcf_clip_l clipping).
## When merged into the package they should reuse .spcf_clip_l, link_fun,
## inv_link_fun and response_se from internal_utils_glm.R rather than redefine.

#' @keywords internal
#' @noRd
.dglm_clip_l <- function(l, family = NULL, cap = getOption("spcf.l_pred_cap", 20)) {
  if (!is.null(family) && identical(family$link, "identity")) return(l)
  if (is.null(cap) || !is.finite(cap)) return(l)
  pmin(pmax(l, -cap), cap)
}

## Generic IRLS working response/weight for a glm() family object.
## Given the current linear predictor eta (INCLUDING offset) and response y,
## returns mu, the working response z for the (x'beta + cascade) part
## (i.e. with offset removed), and the working weight w.
#' @keywords internal
#' @noRd
.dglm_work <- function(family, eta, y, offset = 0) {
  eta <- .dglm_clip_l(eta, family)
  mu  <- family$linkinv(eta)
  me  <- family$mu.eta(eta)
  v   <- pmax(family$variance(mu), 1e-8)
  list(mu = mu, z = (eta - offset) + (y - mu) / me, w = pmax(me^2 / v, 1e-8))
}

## Build a balanced (location x time) panel from long-format vectors.
## Locations are the unique coordinate rows; time is mapped onto a 1..T grid
## using the supplied factor levels (so prediction reuses training levels).
#' @keywords internal
#' @noRd
.dglm_panel <- function(coords, time, vals = NULL, time_levels = NULL) {
  key  <- paste(coords[, 1], coords[, 2], sep = "\r")
  ul   <- unique(key)
  lk   <- match(key, ul)
  if (is.null(time_levels)) time_levels <- sort(unique(time))
  tk   <- match(time, time_levels)
  nL   <- length(ul); nT <- length(time_levels)
  C    <- matrix(0, nL, 2); fi <- which(!duplicated(lk)); C[lk[fi], ] <- as.matrix(coords)[fi, ]
  M    <- NULL
  if (!is.null(vals)) { M <- matrix(NA_real_, nL, nT); M[cbind(lk, tk)] <- vals }
  list(M = M, C = C, lk = lk, tk = tk, nL = nL, nT = nT, time_levels = time_levels)
}

## Kernel evaluation consistent with internal_utils_glm.R (lwr_glm).
#' @keywords internal
#' @noRd
.dglm_kfun <- function(d, band, kernel = "exp") {
  if (kernel == "gau") exp(-(d / band)^2) else exp(-d / band)
}
#' @keywords internal
#' @noRd
.dglm_threshold <- function(band, kernel = "exp") {
  if (kernel == "gau") sqrt(-log(0.05)) * band else -log(0.05) * band
}

## Neighbour-limited (sparse) kernel weight matrix between query sites and knots.
## The kernel exp(-d/band) (or the Gaussian) decays quickly, so only knots within
## a radius where the weight exceeds ~1e-3 are kept; the rest are exact zeros.
## This makes the kernel, the aggregation and the gPoE recombination O(n * m)
## (m = neighbours per site) instead of O(n * K), cutting both time and memory at
## fine bands without materially changing the result. Returns a dgCMatrix (n x K).
#' @keywords internal
#' @noRd
.dglm_kmat <- function(query, knots, band, kernel = "exp") {
  rad <- if (kernel == "gau") sqrt(-log(1e-3)) * band else -log(1e-3) * band
  nn  <- dbscan::frNN(x = knots, eps = rad, query = query, sort = FALSE)
  len <- lengths(nn$id)
  nq  <- nrow(query); nk <- nrow(knots)
  if (sum(len) == 0L)
    return(Matrix::sparseMatrix(i = integer(0), j = integer(0), x = numeric(0),
                                dims = c(nq, nk)))
  Matrix::sparseMatrix(
    i = rep.int(seq_len(nq), len),
    j = unlist(nn$id, use.names = FALSE),
    x = .dglm_kfun(unlist(nn$dist, use.names = FALSE), band, kernel),
    dims = c(nq, nk))
}

## CSR neighbour lists (0-based knot indices) + kernel weights for the fused
## C++ scale operator: for site i, knots idx[ptr[i]..ptr[i+1]-1] with weights w.
#' @keywords internal
#' @noRd
.dglm_nbr <- function(query, knots, band, kernel = "exp") {
  rad <- if (kernel == "gau") sqrt(-log(1e-3)) * band else -log(1e-3) * band
  nn  <- dbscan::frNN(x = knots, eps = rad, query = query, sort = FALSE)
  len <- lengths(nn$id)
  list(ptr = as.integer(c(0L, cumsum(len))),
       idx = as.integer(unlist(nn$id, use.names = FALSE) - 1L),
       w   = as.numeric(.dglm_kfun(unlist(nn$dist, use.names = FALSE), band, kernel)))
}

## The fused C++ scale operator (src/dglm_chunk.cpp) is compiled at package build
## time and registered via Rcpp; dglm_scale_chunk() is therefore available as a
## package-internal routine (see R/RcppExports.R) with no run-time compilation.

## Heteroscedastic per-knot AR(1) Kalman smoother. Z and Rmat are K x nT
## (knot-aggregated working residual and its observation variance per time).
#' @keywords internal
#' @noRd
.dglm_ksmooth <- function(Z, Rmat, rho, Q) {
  K <- nrow(Z); nT <- ncol(Z)
  af <- Pf <- ap <- Pp <- matrix(0, K, nT)
  a <- rep(0, K); P <- rep(Q / (1 - rho^2), K)
  for (t in 1:nT) {
    ap[, t] <- rho * a; Pp[, t] <- rho^2 * P + Q
    Kg <- Pp[, t] / (Pp[, t] + Rmat[, t])
    a <- ap[, t] + Kg * (Z[, t] - ap[, t]); P <- (1 - Kg) * Pp[, t]
    af[, t] <- a; Pf[, t] <- P
  }
  ms <- Ps <- matrix(0, K, nT); ms[, nT] <- af[, nT]; Ps[, nT] <- Pf[, nT]
  if (nT >= 2) for (t in (nT - 1):1) {
    G <- rho * Pf[, t] / pmax(Pp[, t + 1], 1e-12)
    ms[, t] <- af[, t] + G * (ms[, t + 1] - ap[, t + 1])
    Ps[, t] <- Pf[, t] + G^2 * (Ps[, t + 1] - Pp[, t + 1])
  }
  list(m = ms, P = pmax(Ps, 1e-8))
}

## ML estimate of a single global AR(1) (rho, Q) from a knot-aggregated
## working-residual series (K x nT) with per-time observation variance Rmat.
## The marginal (prediction-error) likelihood is summed over knots. Knot-times
## with no nearby observation are flagged by Rmat = Inf and are skipped in the
## likelihood (the Kalman recursion only predicts there, no update).
#' @keywords internal
#' @noRd
.dglm_ar1_ml <- function(Z, Rmat, rho0 = 0.7, Q0 = 1) {
  K <- nrow(Z); nT <- ncol(Z)
  nll <- function(par) {
    rho <- tanh(par[1]); Q <- exp(par[2])
    a <- rep(0, K); P <- rep(Q / (1 - rho^2), K); ll <- 0
    for (t in 1:nT) {
      ap <- rho * a; Pp <- rho^2 * P + Q
      ob <- is.finite(Rmat[, t])                # observed knot-times only
      S  <- Pp + Rmat[, t]; v <- Z[, t] - ap
      if (any(ob)) ll <- ll - 0.5 * sum((log(2 * pi * S) + v^2 / S)[ob])
      Kg <- ifelse(ob, Pp / S, 0); a <- ap + Kg * v; P <- ifelse(ob, (1 - Kg) * Pp, Pp)
    }
    if (!is.finite(ll)) 1e10 else -ll
  }
  opt <- nloptr::nloptr(c(atanh(rho0), log(Q0)), nll,
                        opts = list(algorithm = "NLOPT_LN_BOBYQA",
                                    maxeval = 80, xtol_rel = 1e-5))
  list(rho = max(min(tanh(opt$solution[1]), 0.999), -0.999), Q = exp(opt$solution[2]))
}

## Knot coordinates for a given bandwidth, following the lwr_glm() rule
## n_knot = round(1.5 * area / band^2) capped at the number of unique sites.
## kmeans centers for moderate K, random subsample for very fine scales.
#' @keywords internal
#' @noRd
.dglm_knots <- function(coords_uni, band, seed = 4321) {
  area   <- (max(coords_uni[, 1]) - min(coords_uni[, 1]))^2 +
            (max(coords_uni[, 2]) - min(coords_uni[, 2]))^2
  n_uni  <- nrow(coords_uni)
  ## cap before integer coercion: round(1.5*area/band^2) can exceed the integer
  ## range at very fine bandwidths, where as.integer() would return NA.
  n_knot <- max(8L, as.integer(min(round(1.5 * area / band^2), n_uni)))
  if (n_knot >= n_uni) return(coords_uni)
  if (n_knot > 1000) {
    withr::with_seed(seed, { sel <- sort(sample.int(n_uni, n_knot)) })
    coords_uni[sel, , drop = FALSE]
  } else {
    iter.max <- ifelse(n_uni > 5000, 5L, 10L)
    withr::with_seed(seed, {
      suppressWarnings(ck <- stats::kmeans(coords_uni, n_knot, iter.max = iter.max)$centers)
    })
    coords_uni[FNN::get.knnx(coords_uni, ck, 1)$nn.index, , drop = FALSE]
  }
}

## Kernel-weighted aggregation of a (possibly incomplete) working-residual
## panel onto knots. NA entries (locations not observed at a given time) are
## treated as weight 0, so each time uses only its observed locations. A
## knot-time with no nearby observation gets Z = 0, Rmat = Inf, which the
## Kalman recursion handles as a missing observation (predict, no update).
## Rtr, Wtr: nL x nT residual & working-weight panels; Wf: nL x K kernel matrix.
#' @keywords internal
#' @noRd
.dglm_aggregate <- function(Wf, Rtr, Wtr) {
  W0 <- Wtr; W0[is.na(W0)] <- 0
  R0 <- Rtr; R0[is.na(R0)] <- 0
  den  <- as.matrix(Matrix::crossprod(Wf, W0))                        # t(Wf) %*% W0 = K x nT
  Z    <- as.matrix(Matrix::crossprod(Wf, W0 * R0)) / den
  Rmat <- as.matrix(Matrix::crossprod(Wf * Wf, W0)) / den^2           # Wf*Wf: sparse-safe square
  miss <- !is.finite(den) | den < 1e-12
  Z[miss] <- 0; Rmat[miss] <- Inf
  list(Z = Z, Rmat = Rmat)
}

## One cascade scale: aggregate the (weighted) working residual to knots, run
## the per-knot AR(1) smoother, and recombine the knot Gaussians at any site by
## a generalized Product of Experts (gPoE) with the spatial kernel as exponent.
## Each knot j contributes a Gaussian expert N(m_jt, P_jt). Weighting its PDF by
## phi_j(s) := phi(s, knot_j) and taking the (normalized) product (generalized
## Product of Experts) gives, at site s and time t, the combination weight
##   b_j(s,t) = (phi_j/P_jt) / sum_k (phi_k/P_kt),
## and the field mean
##   f(s,t) = sum_j b_j(s,t) m_jt = (sum_j (phi_j/P_jt) m_jt) / sum_j (phi_j/P_jt).
## The weight phi_j/P_jt (kernel power 1) is reciprocal-consistent with the
## kernel-weighted local regression of .dglm_aggregate(). The product of the
## kernel-weighted Gaussian experts is itself Gaussian, giving the gPoE
## predictive variance
##   V(s,t) = 1 / sum_j (phi_j(s)/P_jt).
## Missing knot-times carry P_jt = +Inf (gain 0), so they drop out of the
## product. Rtr, Wtr: nL x nT residual & working-weight panels at Ctr
## (NA = unobserved).
#' @keywords internal
#' @noRd
## Knot placement and (train + optional prediction) neighbourhoods for one band.
## These depend only on the coordinates and bandwidth -- NOT on the residual --
## so they are identical across IRLS / backfitting iterations and can be built
## once and reused (see cf_dglm). Returns everything .dglm_scale_apply() needs.
#' @keywords internal
#' @noRd
.dglm_scale_setup <- function(Ctr, band, kernel, seed, Cpr = NULL) {
  knots <- .dglm_knots(unique(Ctr), band, seed)
  nb <- .dglm_nbr(Ctr, knots, band, kernel)
  if (!is.null(Cpr)) { pb <- .dglm_nbr(Cpr, knots, band, kernel); n0 <- nrow(Cpr) }
  else { pb <- list(ptr = c(0L, 0L), idx = integer(0), w = numeric(0)); n0 <- 0L }
  list(knots = knots, K = nrow(knots), nb = nb, pb = pb, n0 = n0)
}

## Residual-dependent part of one cascade scale: aggregate, per-knot AR(1)
## Kalman smoother and gPoE recombination, given a pre-built setup. predict =
## FALSE skips the prediction-site recombination (used during the IRLS sweep,
## where only the training field is needed); the grid is recombined once at the
## end with predict = TRUE.
#' @keywords internal
#' @noRd
.dglm_scale_apply <- function(su, Rtr, Wtr, rho, Q, predict = TRUE) {
  W0 <- Wtr; W0[is.na(W0)] <- 0
  R0 <- Rtr; R0[is.na(R0)] <- 0
  use_pr <- predict && su$n0 > 0
  pb <- if (use_pr) su$pb else list(ptr = c(0L, 0L), idx = integer(0), w = numeric(0))
  n0 <- if (use_pr) su$n0 else 0L
  ## panels passed time-major (t() -> nT x nL) so the C++ inner t-loop is contiguous
  res <- dglm_scale_chunk(su$nb$ptr, su$nb$idx, su$nb$w, t(W0), t(R0), su$K, rho, Q,
                          pb$ptr, pb$idx, pb$w, n0)
  out <- list(Ftr = res$Ftr, Vtr = res$Vtr, peeled = Rtr - res$Ftr, knots = su$knots)
  if (n0 > 0) { out$Fpr <- res$Fpr; out$Vpr <- res$Vpr }
  out
}

## Convenience wrapper: build the setup and apply it in one call (rebuilds the
## neighbourhoods every time). Used where a scale is computed only once, e.g.
## the single greedy pass in cf_dglm_hv.
#' @keywords internal
#' @noRd
.dglm_scale <- function(Ctr, Rtr, Wtr, band, rho, Q, kernel, seed, Cpr = NULL) {
  su <- .dglm_scale_setup(Ctr, band, kernel, seed, Cpr)
  .dglm_scale_apply(su, Rtr, Wtr, rho, Q, predict = !is.null(Cpr))
}

## Dynamic regression of a residual on (a few) covariates with time-varying
## coefficients. For each time t the cross-section is reduced to its weighted
## normal equations (info Lam_t = X_t' W_t X_t, xi_t = X_t' W_t r_t); the
## coefficient vector beta_t then follows a Gaussian random walk
## beta_t = beta_{t-1} + N(0, q I) observed through the per-time GLS estimate
## (b_t = Lam_t^{-1} xi_t, cov R_t = Lam_t^{-1}), and is recovered by a Kalman
## filter + RTS smoother. The drift q is ML-estimated (prediction-error
## likelihood) when not supplied. Returns the smoothed coefficients (nT x d), the
## per-time smoothed covariances (list of d x d), and q. Used by cf_dglm /
## cf_dglm_hv for the time-varying-coefficient option.
#' @keywords internal
#' @noRd
.dglm_dynreg <- function(r, Xtv, w, tk, nT, q = NULL, P0 = 1e4, rg = 1e-6) {
  d <- ncol(Xtv); Id <- diag(d)
  bt <- vector("list", nT); Rt <- vector("list", nT); has <- logical(nT)
  rss <- 0; dfr <- 0
  for (t in seq_len(nT)) {
    idx <- which(tk == t); if (!length(idx)) next
    Xt <- Xtv[idx, , drop = FALSE]; wt <- w[idx]
    L  <- crossprod(Xt, wt * Xt) + diag(rg, d)
    Ri <- solve(L); Rt[[t]] <- Ri
    bt[[t]] <- drop(Ri %*% crossprod(Xt, wt * r[idx])); has[t] <- TRUE
    e   <- r[idx] - drop(Xt %*% bt[[t]])            # per-time working residual
    rss <- rss + sum(wt * e^2); dfr <- dfr + length(idx) - d
  }
  ## Cov(b_t) = s2 * (X'WX)^{-1}: without the working-residual dispersion s2 the
  ## observation covariance implicitly assumes s2 = 1, which is right for the
  ## variance-weighted families but not for a Gaussian response of arbitrary
  ## scale. It then propagates into both the smoother gain and the reported
  ## beta_tv_sd, leaving the latter almost independent of the residual variance
  ## (too wide when s2 < 1, too narrow -- the unsafe direction -- when s2 > 1).
  s2 <- if (dfr > 0 && is.finite(rss) && rss > 0) rss / dfr else 1
  Rt <- lapply(Rt, function(R) if (is.null(R)) NULL else s2 * R)
  ## the random-walk innovation covariance is diag(q), so each time-varying
  ## coefficient carries its OWN drift variance. q is a length-d vector; a scalar
  ## is broadcast (for backward compatibility / a shared drift).
  mkQ <- function(qv) { qv <- rep_len(qv, d); Qm <- matrix(0, d, d); diag(Qm) <- qv; Qm }
  run <- function(qv, smooth = FALSE) {
    Qm <- mkQ(qv)
    a_p <- P_p <- a_f <- P_f <- vector("list", nT)
    a <- rep(0, d); P <- diag(P0, d); ll <- 0
    for (t in seq_len(nT)) {
      ap <- a; Pp <- P + Qm; a_p[[t]] <- ap; P_p[[t]] <- Pp
      if (has[t]) {
        S <- Pp + Rt[[t]]; Si <- solve(S); v <- bt[[t]] - ap
        ll <- ll - 0.5 * (as.numeric(determinant(S, logarithm = TRUE)$modulus) +
                          drop(crossprod(v, Si %*% v)))
        K <- Pp %*% Si; a <- ap + drop(K %*% v); P <- (Id - K) %*% Pp
      } else { a <- ap; P <- Pp }
      a_f[[t]] <- a; P_f[[t]] <- P
    }
    if (!smooth) return(ll)
    as <- a_f; Ps <- P_f
    if (nT >= 2) for (t in (nT - 1):1) {
      G <- P_f[[t]] %*% solve(P_p[[t + 1]])
      as[[t]] <- a_f[[t]] + drop(G %*% (as[[t + 1]] - a_p[[t + 1]]))
      Ps[[t]] <- P_f[[t]] + G %*% (Ps[[t + 1]] - P_p[[t + 1]]) %*% t(G)
    }
    list(a = as, P = Ps)
  }
  if (is.null(q)) {
    ## per-coefficient drift estimated by d-dimensional ML of the log-drift; if
    ## the multivariate optimiser fails, fall back to a single shared scalar drift.
    op <- tryCatch(stats::optim(rep(log(1e-3), d), function(lq) -run(exp(lq)),
                                method = "L-BFGS-B",
                                lower = rep(log(1e-8), d), upper = rep(log(1e3), d)),
                   error = function(e) NULL)
    q <- if (!is.null(op) && is.finite(op$value)) exp(op$par)
         else rep(exp(stats::optimize(function(lq) -run(rep(exp(lq), d)),
                                      c(log(1e-8), log(1e3)))$minimum), d)
  } else {
    q <- rep_len(q, d)                                 # scalar -> shared; vector -> per-coef
  }
  sm   <- run(q, smooth = TRUE)
  beta <- matrix(unlist(sm$a), nrow = nT, ncol = d, byrow = TRUE)
  list(beta = beta, V = sm$P, q = q)
}

## Spatial-block cluster-robust covariance for the constant coefficients.
## The naive GLM covariance treats the cascade field as a known offset, so it
## ignores that the residual is a spatially/temporally correlated random field;
## with smooth covariates this badly understates Var(beta-hat). Here the field is
## put back into the working residual (e = f + (y-mu)/mu') and a cluster-robust
## sandwich is taken over spatial blocks (all times of a location share a block),
## which captures the correlated-error inflation. Blocks are a Gx x Gy grid whose
## per-axis counts split each coordinate so a block side exceeds the field's
## correlation length (proxied by the MEDIAN committed bandwidth, c_guard = 1),
## clamped to [2, 8] per axis. Sizing each axis separately keeps blocks larger
## than the dependence range on both axes even for elongated regions, where a
## common count per axis would make the narrow axis's blocks too thin. The
## defaults were tuned (gaussian/Poisson/binomial, correlation ranges 0.06-0.40,
## aspect ratios up to 1:8). Reduces to OLS-with-field-error for gaussian.
## Returns the p x p covariance V and the number of blocks G.
#' @keywords internal
#' @noRd
.dglm_clusterSE <- function(y, Xg, beta, f_obs, tvpart, offset, family,
                            coords_obs, bands, c_guard = 1.0) {
  eta <- .dglm_clip_l(drop(Xg %*% beta) + f_obs + tvpart + offset, family)
  mu  <- family$linkinv(eta); mup <- family$mu.eta(eta)
  v   <- pmax(family$variance(mu), 1e-8)
  W   <- pmax(mup^2 / v, 1e-8)
  e   <- f_obs + (y - mu) / mup                   # working residual WITH the field
  ## per-axis block counts: each axis is split so a block side exceeds the field's
  ## correlation length (median committed bandwidth) independently in x and y, so
  ## blocks stay larger than the dependence range on BOTH axes even for elongated
  ## (anisotropic) regions. Counts clamped to [2, 8] per axis.
  rng <- as.numeric(stats::quantile(bands, 0.5))
  if (!is.finite(rng) || rng <= 0)
    rng <- mean(apply(coords_obs, 2, function(z) diff(range(z)))) / 8
  span <- apply(coords_obs, 2, function(z) diff(range(z)))
  Gxy  <- pmax(2L, pmin(8L, as.integer(floor(span / (c_guard * rng)))))
  qx <- stats::quantile(coords_obs[, 1], seq(0, 1, length.out = Gxy[1] + 1))
  qy <- stats::quantile(coords_obs[, 2], seq(0, 1, length.out = Gxy[2] + 1))
  blk <- interaction(cut(coords_obs[, 1], unique(qx), include.lowest = TRUE),
                     cut(coords_obs[, 2], unique(qy), include.lowest = TRUE),
                     drop = TRUE)
  G   <- nlevels(blk)
  XtWXi <- solve(crossprod(Xg, W * Xg))
  S   <- rowsum(Xg * (W * e), blk)                # G x p per-block score sums
  V   <- (G / (G - 1)) * XtWXi %*% crossprod(S) %*% XtWXi
  list(V = V, G = G)
}

## opt+field cluster-robust covariance for cf_dglm (default). See
## .spcf_optfield_SE (spatial GLM) for the rationale: the meat splits into a
## field-REMOVED noise part B_noise (block-clustered working residual) and a
## field part B_field that adds the calibrated per-point field variance s_f^2
## back with a within-block exp(-d / h) correlation, h = median committed
## bandwidth. eta / IRLS weights keep f_obs + tvpart so mu matches the fit; only
## the additive spatial field is removed from the residual. s_f is link-scale.
.dglm_optfield_SE <- function(y, Xg, beta, f_obs, s_f, tvpart, offset, family,
                              coords_obs, bands, c_guard = 1.0) {
  Xg <- as.matrix(Xg); beta <- as.numeric(beta); s_f <- as.numeric(s_f)
  eta <- .dglm_clip_l(drop(Xg %*% beta) + f_obs + tvpart + offset, family)
  mu  <- family$linkinv(eta); mup <- family$mu.eta(eta)
  v   <- pmax(family$variance(mu), 1e-8)
  W   <- pmax(mup^2 / v, 1e-8)
  r   <- (y - mu) / ifelse(abs(mup) < 1e-8, 1e-8, mup)   # field-removed residual
  rng <- as.numeric(stats::quantile(bands, 0.5, na.rm = TRUE))
  if (!is.finite(rng) || rng <= 0)
    rng <- mean(apply(coords_obs, 2, function(z) diff(range(z)))) / 8
  span <- apply(coords_obs, 2, function(z) diff(range(z)))
  Gxy  <- pmax(2L, pmin(8L, as.integer(floor(span / (c_guard * rng)))))
  qx <- stats::quantile(coords_obs[, 1], seq(0, 1, length.out = Gxy[1] + 1))
  qy <- stats::quantile(coords_obs[, 2], seq(0, 1, length.out = Gxy[2] + 1))
  blk <- interaction(cut(coords_obs[, 1], unique(qx), include.lowest = TRUE),
                     cut(coords_obs[, 2], unique(qy), include.lowest = TRUE),
                     drop = TRUE)
  G   <- nlevels(blk)
  Ai  <- solve(crossprod(Xg, W * Xg))
  S   <- rowsum(Xg * (W * r), blk)
  Bnoise <- (G / (G - 1)) * crossprod(S)
  U   <- Xg * (W * s_f)
  Bfield <- matrix(0, ncol(Xg), ncol(Xg))
  for (lv in levels(blk)) {
    ix <- which(blk == lv)
    if (length(ix) == 1L) { Bfield <- Bfield + tcrossprod(U[ix, ]); next }
    Dg <- as.matrix(stats::dist(coords_obs[ix, , drop = FALSE]))
    Rg <- exp(-Dg / rng)
    Ug <- U[ix, , drop = FALSE]
    Bfield <- Bfield + crossprod(Ug, Rg %*% Ug)
  }
  Vof <- Ai %*% (Bnoise + Bfield) %*% Ai
  ## leverage-LOO ceiling (self-calibrating); see .spcf_levloo_meat /
  ## .spcf_optfield_SE. Caps the count-family field over-shoot without touching
  ## already-calibrated families; correlation-preserving diagonal rescale.
  Bloo <- tryCatch(.spcf_levloo_meat(Xg, W, r, coords_obs, bands, blk, Ai),
                   error = function(e) NULL)
  if (!is.null(Bloo)) {
    Ve   <- Ai %*% Bnoise %*% Ai
    Vloo <- Ai %*% Bloo %*% Ai
    d    <- pmax(diag(Ve), pmin(diag(Vof), diag(Vloo)))
    sof  <- sqrt(pmax(diag(Vof), .Machine$double.eps))
    Rc   <- Vof / outer(sof, sof)
    sdn  <- sqrt(pmax(d, 0))
    Vof  <- Rc * outer(sdn, sdn)
  }
  list(V = Vof, G = G)
}
