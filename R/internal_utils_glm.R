## Soft-clip the log-scale linear predictor (`l_pred`) to `[-cap, cap]`,
## as a safety net against numerical blow-ups in glm refits when the
## spatial decomposition pushes a few points to extreme values
## (e.g., Poisson with >99% zeros, binomial with rare events).
##
## Applies only for non-identity link families (Poisson/Binomial/Gamma/...).
## For identity-link Gaussian models the linear predictor is on the response
## scale and a fixed cap would be inappropriate, so clipping is skipped.
##
## Cap is read from `getOption("spcf.l_pred_cap", 20)`. Users can tune via
## `options(spcf.l_pred_cap = <num>)` or disable via `NULL` / `Inf`.
#' @keywords internal
#' @noRd
.spcf_clip_l <- function(l, family = NULL, cap = getOption("spcf.l_pred_cap", 20)) {
  if (!is.null(family) && identical(family$link, "identity")) return(l)
  if (is.null(cap) || !is.finite(cap)) return(l)
  pmin(pmax(l, -cap), cap)
}

#' @keywords internal
#' @noRd
link_fun <- function(mu, family) {
  family$linkfun(mu)
}

#' @keywords internal
#' @noRd
inv_link_fun <- function(eta, family) {
  family$linkinv(eta)
}

#' @keywords internal
#' @noRd
response_se <- function(pred_lin, pred_lin_sd, family) {
  dmu_det   <- family$mu.eta(pred_lin)
  se_mu     <- abs(dmu_det) * pred_lin_sd
}

#' @keywords internal
#' @noRd
initial_fun_glm  <- function(x, y, coords, offset=NULL, x_sel=NULL,
                             train_rat, id_train=NULL, family, seed=NULL){

  id_uni         <- match(paste(coords[,1], coords[,2]),
                          unique(paste(coords[,1], coords[,2])))
  coords_uni     <- unique(coords)
  n_uni          <- nrow(coords_uni)
  if(is.null(id_train)){
    if(train_rat<1){
      do_split <- function(){
        if(n_uni > 30000){
          ## Beyond ~30k unique points the kmeans advantage over random
          ## shrinks (K grows with n and point density saturates), so fall
          ## back to random sampling to keep id_train selection cheap.
          sort(sample.int(n_uni, round(n_uni*train_rat)))
        } else {
          ## kmeans the smaller of training/validation and derive the other
          ## via set complement. Halves kmeans cost when train_rat > 0.5 and
          ## reduces id_train variance vs random sampling.
          K_train         <- round(n_uni*train_rat)
          K_val           <- n_uni - K_train
          pick_train      <- K_train <= K_val
          K               <- if(pick_train) K_train else K_val
          iter.max        <- if(n_uni > 20000) 3L else if(n_uni > 5000) 5L else 10L
          suppressWarnings(
            coords_uni_k_tmp <- kmeans(coords_uni, K, iter.max=iter.max)$centers
          )
          sel_uni         <- sort( get.knnx(coords_uni, coords_uni_k_tmp, 1)$nn.index )
          if(pick_train) sel_uni else setdiff(seq_len(n_uni), sel_uni)
        }
      }
      id_train_uni <- if(is.null(seed)) do_split() else withr::with_seed(seed, do_split())
    } else {
      id_train_uni      <- 1:n_uni
    }
    id_train            <- which( id_uni %in% id_train_uni )
  }

  n              <- length(y)
  xname          <- "Intercept"
  if( is.null(x) ){
    x            <- matrix(1,nrow=n,ncol=1)
  } else {
    x            <- as.matrix(x)
    if( is.null( x_sel ) ) x_sel <- apply(x,2,sd)!=0
    if( sum(x_sel) ==1 ){
      xname      <- c("Intercept", "x1")
    } else if( sum(x_sel) > 1 ){
      xname      <- c("Intercept", names(data.frame(x[,x_sel])))
    }
    x            <- cbind(1,x[,x_sel])
  }
  nx             <- ncol(x)

  coords         <- as.matrix(coords)
  if(is.null(offset)) offset<- rep(0,n)
  gmod           <- glm(y~0+x+offset(offset),family=family)
  pred           <- predict(gmod, type="link")
  resid          <- gmod$residuals            # (y - exp(m))/exp(m) in case of poisson
  beta_int       <- matrix( coefficients(gmod) )
  row.names(beta_int)<-xname

  beta_int_vcov  <- summary(gmod)$cov.scaled
  loss           <- sum(residuals(gmod, type="deviance")[-id_train]^2 )
  beta           <- matrix(beta_int  , nrow = n, ncol = nx, byrow = TRUE)
  beta_v         <- matrix(diag(beta_int_vcov), nrow = n, ncol = nx, byrow = TRUE)
  return(list(beta_int=beta_int, x=x, id_train=id_train, offset=offset,
              beta=beta, beta_v=beta_v, n=n, nx=nx, gmod=gmod,pred=pred,resid=resid,
              x_sel=x_sel,xname=xname,coords=coords))#xx_inv
}

## Internal utility for the coarse-to-fine GLM.
##
## The knot inner loop of lwr_glm() is delegated to the C++ routine
## lwr_chunk_glm_cpp (see src/lwr_chunk_glm.cpp). The chunked driver loop
## here caps peak neighbor-list memory at O(chunk_size * avg_neighbors).

#' @keywords internal
#' @noRd
lwr_glm        <- function(coords, coords_uni,resid, x, w=NULL, offset=NULL,
                           band, b_old, vc, ridge, coords_old=NULL, kernel,
                           id_train,y, coords0=NULL, x0=NULL,
                           sel_id=NULL, sse_hv0=NULL, l_pred, family, func){

  n            <- nrow(coords)
  if(is.null(w)){
    w          <- rep(1,n)
  }

  nx           <- ncol(x)
  if(kernel=="gau"){
    threshold  <- sqrt(-log(0.05))*band
    kernel_id  <- 2L
  } else if(kernel=="exp"){
    threshold  <- -log(0.05)*band
    kernel_id  <- 1L
  }

  if(is.null(sel_id)){ # cf_glm_hv
    area         <- (max(coords[,1])-min(coords[,1]))^2 + (max(coords[,2])-min(coords[,2]))^2
    n_knot       <- round(1.5*area/band^2)
    n_uni        <- nrow(coords_uni)
    if( n_knot < n_uni ){
      if( n_knot > 1000 ){
        ## Fine-scale knots: kmeans O(n_uni * n_knot * iter) becomes dominant.
        ## Random sampling preserves accuracy at these scales because the
        ## kernel support is small and uniform coverage is unnecessary.
        withr::with_seed(4321, {
          sel_id <- sort(sample.int(n_uni, n_knot))
        })
      } else {
        iter.max   <- ifelse( n_uni > 5000, 5, 10)
        withr::with_seed(4321,{
          suppressWarnings(coords_k_tmp<- kmeans(coords_uni,n_knot,iter.max=iter.max)$centers)
        })
        sel_id     <- get.knnx(coords_uni, coords_k_tmp, 1)$nn.index
      }
      coords_cent<- coords_uni[sel_id,]
      sel_list   <- 1:nrow(coords_cent)
    } else {
      n_knot     <- n_uni
      coords_cent<- coords_uni
      sel_list   <- 1:n_knot
      sel_id     <- NA
    }
  } else if(is.na(sel_id[1])){
    coords_cent   <- coords_uni
    n_knot        <- nrow(coords_cent)
    sel_list      <- 1:n_knot
  } else {  # cf_glm
    n_knot       <- length(sel_id)
    sel_list     <- 1:n_knot
    coords_cent  <- coords_uni[sel_id,]
  }

  ################# Prior coefficient variance
  B_var          <- matrix(Inf,nrow=n_knot,ncol=nx)
  if( !is.null(b_old) & ridge==TRUE ){
    for(i in 1:nx) B_var[,i]<- mean(b_old[,i]^2)
  }

  ################# accumulators via the fused nanoflann kernel
  n0 <- 0L
  if(!is.null(coords0)) n0 <- nrow(coords0)

  id_train_int            <- integer(n)
  id_train_int[id_train]  <- 1L
  vc_int                  <- as.integer(vc)

  ## Fused kernel (src/lwr_chunk_glm_fused.cpp): builds a kd-tree over `coords`
  ## (and `coords0`) ONCE and does radius-search -> local GLM -> scatter-add
  ## per knot inline, WITHOUT materialising the neighbour lists. This restores
  ## ~O(N log N) scaling (the earlier chunk-size problem) at much lower peak
  ## memory (no O(N * avg_neighbors) neighbour list) and is a bit faster.
  ## Numerically identical to the frNN + lwr_chunk_glm_cpp path (~1e-13, only
  ## from sqrt(squared distance) vs direct distance / summation order).
  fres <- lwr_glm_fused_cpp(
    coords       = coords,
    coords_cent  = matrix(coords_cent, ncol = 2),
    resid        = as.numeric(resid),
    w_obs        = as.numeric(w),
    x            = x,
    id_train     = id_train_int,
    B_var        = B_var,
    vc_cols      = vc_int,
    band         = band,
    kernel_id    = kernel_id,
    threshold    = threshold,
    is_lm        = 0L,
    coords0_sexp = if(!is.null(coords0)) as.matrix(coords0) else NULL,
    x0_sexp      = if(!is.null(coords0)) x0 else NULL)
  b_all        <- fres$b_all
  bv_inv_all   <- fres$bv_inv_all
  pv_inv_all   <- fres$pv_inv_all
  b_old        <- fres$b_old
  b_all0 <- bv_inv_all0 <- pv_inv_all0 <- NULL
  if (!is.null(coords0)) {
    b_all0      <- fres$b_all0
    bv_inv_all0 <- fres$bv_inv_all0
    pv_inv_all0 <- fres$pv_inv_all0
  }

  ################# selection of vc through CV
  run            <- FALSE
  if( func == "cf_glm_hv"|func=="cf_lm_hv" ){
    bv_all          <- bv_inv_all
    b_all[,vc]      <- b_all[,vc,drop=FALSE]/pv_inv_all[,vc,drop=FALSE]
    b_all[,-vc]         <- 0
    b_all[is.nan(b_all)]<- 0
    pred            <- rowSums(x*b_all)

    l_pred        <- l_pred  + pred
    l_pred_off    <- .spcf_clip_l(l_pred, family) + offset
    ## glm.fit direct (dglm-style): identical fitted values / deviance as
    ## glm(y ~ 0 + x + offset(l_pred_off)); avoids the per-band formula rebuild.
    gmod0         <- glm.fit(x, y, offset=l_pred_off, family=family)
    sse_hv        <- sum(family$dev.resids(y, gmod0$fitted.values, 1)[-id_train] )
    run           <- ifelse(sse_hv<sse_hv0, TRUE, FALSE)

    if(!run){
      sse_hv     <- sse_hv0
    }
  } else {
    bv_all          <- bv_inv_all
    b_all[,vc]      <- b_all[,vc]/pv_inv_all[,vc]
    b_all[,-vc]         <- 0
    b_all[is.nan(b_all)]<- 0
    pred            <- rowSums(x*b_all)

    sse_hv      <- NA
    run         <- TRUE
  }

  if( run ){
    #bv_all          <- bv_inv_all
    #b_all[,vc]      <- b_all[,vc]/pv_inv_all[,vc]
    #b_all[,-vc]         <- 0
    #b_all[is.nan(b_all)]<- 0

    bv_inv_all[, vc]<- bv_inv_all[, vc]/pv_inv_all[, vc]
    bv_all[, vc]    <- 1/bv_inv_all[, vc]
    bv_all[, -vc]   <- NA
    bv_all[is.nan(bv_all)]<-Inf

    ## Predictive variance per eq.(10): 1 / sum_k (w_k / pv_k). Grows away from
    ## data (link scale), unlike the coefficient variance bv_all.
    pv_all          <- 1/pv_inv_all
    pv_all[, -vc]   <- NA
    pv_all[is.nan(pv_all)] <- Inf

    #pred            <- rowSums(x*b_all)
    if( !is.null(coords0) ){
      bv_all0          <- bv_inv_all0
      b_all0[,vc]      <- b_all0[,vc]/pv_inv_all0[,vc]
      b_all0[,-vc]     <- 0
      b_all0[is.nan(b_all0)]<-0
      b_all0[is.na(b_all0)] <-0

      bv_inv_all0[, vc]<- bv_inv_all0[, vc]/pv_inv_all0[, vc]
      bv_all0[, vc]    <- 1/bv_inv_all0[, vc]
      bv_all0[, -vc]   <- NA
      bv_all0[is.nan(bv_all0)]<-Inf
      pv_all0          <- 1/pv_inv_all0
      pv_all0[, -vc]   <- NA
      pv_all0[is.nan(pv_all0)] <- Inf
      pred0       <- rowSums(x0*b_all0)

    } else {
      b_all0 <-bv_all0<-pv_all0<-pred0<-NULL
    }

    return(list(beta=b_all, beta_v=bv_all, beta_pv=pv_all, pred=pred, sel_id=sel_id,
                coords_cent=coords_cent,
                beta0=b_all0,beta0_v=bv_all0, beta0_pv=pv_all0, pred0=pred0, b_old=b_old,
                run=run,sse_hv=sse_hv,vc_sel=vc, sse_hv0=sse_hv0))
  } else {
    return(list(run=FALSE))
  }
}


## Spatial-block cluster-robust covariance for the (constant) coefficients of a
## coarse-to-fine spatial GLMM/LM. The model-based covariance treats the fitted
## spatial field as a known offset, so it ignores that the residual is a
## spatially correlated random field; with smooth covariates this badly
## understates Var(beta-hat). Here the field is put back into the working
## residual (e = field + (y - mu)/mu') and a cluster-robust sandwich is taken
## over spatial blocks (a GxG grid; G ~ n_loc^{1/3} clamped to [G_lo, 8], with a
## range guard widening blocks to at least c_guard x the MEDIAN committed
## bandwidth so blocks exceed the field's correlation length). The defaults
## c_guard = 1.0 and G_lo = 4 were tuned (gaussian/Poisson/binomial, correlation
## ranges 0.06-0.40) to remove the over-conservatism of stronger guards at
## short/moderate range while keeping coverage from collapsing at long range
## (worst-case coverage ~0.72 across families). Reduces to the field-in-error
## OLS sandwich for gaussian/identity. Returns the p x p covariance V and G.
#' @keywords internal
#' @noRd
.spcf_clusterSE <- function(y, X, beta, field, offset, family, coords, bands,
                            c_guard = 1.0) {
  X <- as.matrix(X); beta <- as.numeric(beta); field <- as.numeric(field)
  if (is.null(offset)) offset <- 0
  eta <- .spcf_clip_l(as.numeric(X %*% beta) + field + offset, family)
  mu  <- family$linkinv(eta); mup <- family$mu.eta(eta)
  v   <- pmax(family$variance(mu), 1e-8)
  W   <- pmax(mup^2 / v, 1e-8)
  e   <- field + (y - mu) / mup                       # working residual WITH the field
  coords <- as.matrix(coords)
  ## Per-axis block counts: each coordinate axis is split so that a block side
  ## exceeds the field's correlation length (proxied by the median committed
  ## bandwidth), independently in x and y. This keeps blocks larger than the
  ## dependence range on BOTH axes even when the study region is strongly
  ## anisotropic (elongated), where a common count per axis would make the narrow
  ## axis's blocks finer than the range and leave neighbouring blocks correlated.
  ## Counts are clamped to [2, 8] per axis. The defaults (median bandwidth,
  ## c_guard = 1) were tuned by checking coverage across response families,
  ## correlation ranges and aspect ratios.
  rng <- suppressWarnings(as.numeric(stats::quantile(bands, 0.5, na.rm = TRUE)))
  if (!length(rng) || !is.finite(rng) || rng <= 0)
    rng <- mean(apply(coords, 2, function(z) diff(range(z)))) / 8
  span <- apply(coords, 2, function(z) diff(range(z)))
  Gxy  <- pmax(2L, pmin(8L, as.integer(floor(span / (c_guard * rng)))))
  qx  <- stats::quantile(coords[, 1], seq(0, 1, length.out = Gxy[1] + 1))
  qy  <- stats::quantile(coords[, 2], seq(0, 1, length.out = Gxy[2] + 1))
  blk <- interaction(cut(coords[, 1], unique(qx), include.lowest = TRUE),
                     cut(coords[, 2], unique(qy), include.lowest = TRUE),
                     drop = TRUE)
  G   <- nlevels(blk)
  XtWXi <- solve(crossprod(X, W * X))
  S   <- rowsum(X * (W * e), blk)                     # G x p per-block score sums
  V   <- (G / (G - 1)) * XtWXi %*% crossprod(S) %*% XtWXi
  list(V = V, G = G)
}

## Analytic leave-one-out (LOO) sandwich meat -- a stable, refit-free ceiling for
## the opt+field field term. The cascade field is (locally) a linear smoother of
## the working response, so the LOO working residual is r_loo = r / (1 - h), with
## h the predictor leverage: h = h_X + h_field, where h_X = W * x' (X'WX)^{-1} x
## is the fixed-effect leverage and h_field aggregates the per-scale kernel
## self-weights (1 / sum_j exp(-d_ij / h_r)) across committed bandwidths h_r
## (combined as 1 - prod_r (1 - h_r)). The LOO residuals feed the same spatial
## block sandwich as B_noise. This estimates the TOTAL score variance (noise +
## field error) at full n without refitting; used only as an upper cap so it
## reins in the count-family (W = mu) field over-shoot without touching families
## where opt+field is already calibrated. O(N) cost (one knn + per-scale sums).
.spcf_levloo_meat <- function(X, W, r, coords, bands, blk, Ai) {
  X <- as.matrix(X); coords <- as.matrix(coords); np <- nrow(X)
  bands <- bands[is.finite(bands) & bands > 0]
  hX <- W * rowSums((X %*% Ai) * X)                     # fixed-effect leverage
  hfield <- rep(0, np)
  if (length(bands)) {
    k  <- min(np - 1L, 200L)
    kn <- FNN::get.knn(coords, k = k)$nn.dist
    loglev <- rep(0, np)
    for (hr in bands) {
      s_ir <- 1 + rowSums(exp(-kn / hr))               # kernel row sum (+ self)
      loglev <- loglev + log1p(-pmin(1 / s_ir, 0.999))
    }
    hfield <- 1 - exp(loglev)
  }
  h_ii  <- pmin(pmax(hX, 0) + hfield, 0.99)
  r_loo <- r / (1 - h_ii)
  G <- nlevels(blk)
  S <- rowsum(X * (W * r_loo), blk)
  (G / (G - 1)) * crossprod(S)
}

## opt+field cluster-robust coefficient covariance (default). The classic
## .spcf_clusterSE keeps the *realised* field inside the working residual
## (e = field + noise); because the field's local sign is treated as data, the
## clustered meat over-counts it and Var(beta) comes out conservative. Here the
## meat is split into two pieces:
##   (i)  B_noise -- the field-REMOVED working residual r = (y - mu)/mu.eta,
##        block-clustered (sandwich, small-sample correction G/(G-1)); this is
##        the pure observation-noise contribution.
##   (ii) B_field -- the field uncertainty added back through its *calibrated*
##        variance s_f^2 (the sill-capped, tau-scaled per-point field variance
##        that also drives pred_lin_sd), with an explicit within-block
##        exp(-d / h) correlation, h = median committed bandwidth. Modelling the
##        field as a smooth correlated error (rather than the raw realisation)
##        removes the classic conservatism while keeping near-nominal coverage.
## s_f is passed on the link scale (sqrt of the capped field variance). eta and
## the IRLS weights W still include the fitted field so mu / mu.eta match the
## fitted model; only the additive field term is taken out of the residual.
.spcf_optfield_SE <- function(y, X, beta, field, s_f, offset, family, coords,
                              bands, c_guard = 1.0) {
  X <- as.matrix(X); beta <- as.numeric(beta)
  field <- as.numeric(field); s_f <- as.numeric(s_f)
  if (is.null(offset)) offset <- 0
  eta <- .spcf_clip_l(as.numeric(X %*% beta) + field + offset, family)
  mu  <- family$linkinv(eta); mup <- family$mu.eta(eta)
  v   <- pmax(family$variance(mu), 1e-8)
  W   <- pmax(mup^2 / v, 1e-8)
  r   <- (y - mu) / ifelse(abs(mup) < 1e-8, 1e-8, mup)   # field-removed residual
  coords <- as.matrix(coords)
  rng <- suppressWarnings(as.numeric(stats::quantile(bands, 0.5, na.rm = TRUE)))
  if (!length(rng) || !is.finite(rng) || rng <= 0)
    rng <- mean(apply(coords, 2, function(z) diff(range(z)))) / 8
  span <- apply(coords, 2, function(z) diff(range(z)))
  Gxy  <- pmax(2L, pmin(8L, as.integer(floor(span / (c_guard * rng)))))
  qx  <- stats::quantile(coords[, 1], seq(0, 1, length.out = Gxy[1] + 1))
  qy  <- stats::quantile(coords[, 2], seq(0, 1, length.out = Gxy[2] + 1))
  blk <- interaction(cut(coords[, 1], unique(qx), include.lowest = TRUE),
                     cut(coords[, 2], unique(qy), include.lowest = TRUE),
                     drop = TRUE)
  G   <- nlevels(blk)
  Ai  <- solve(crossprod(X, W * X))
  S   <- rowsum(X * (W * r), blk)                       # noise meat
  Bnoise <- (G / (G - 1)) * crossprod(S)
  U   <- X * (W * s_f)                                  # field meat (per point)
  Bfield <- matrix(0, ncol(X), ncol(X))
  for (lv in levels(blk)) {
    ix <- which(blk == lv)
    if (length(ix) == 1L) { Bfield <- Bfield + tcrossprod(U[ix, ]); next }
    Dg <- as.matrix(stats::dist(coords[ix, , drop = FALSE]))
    Rg <- exp(-Dg / rng)                               # within-block field corr
    Ug <- U[ix, , drop = FALSE]
    Bfield <- Bfield + crossprod(Ug, Rg %*% Ug)
  }
  Vof <- Ai %*% (Bnoise + Bfield) %*% Ai
  ## leverage-LOO ceiling (self-calibrating). Cap each coefficient variance at
  ## the analytic LOO sandwich, floored at the noise-only variance, and rescale
  ## the opt+field covariance to the capped diagonal while preserving its
  ## correlation structure (keeps the matrix PSD). This removes the count-family
  ## over-shoot (Poisson) yet leaves Gaussian/binomial unchanged, where the LOO
  ## ceiling exceeds the opt+field variance so the cap does not bind.
  Bloo <- tryCatch(.spcf_levloo_meat(X, W, r, coords, bands, blk, Ai),
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
