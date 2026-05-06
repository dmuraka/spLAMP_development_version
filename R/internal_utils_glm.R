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

  ################# accumulators
  n0 <- 0L
  if(!is.null(coords0)) n0 <- nrow(coords0)

  id_train_flag           <- logical(n)
  id_train_flag[id_train] <- TRUE
  vc_int                  <- as.integer(vc)

  b_all        <- matrix(0, n, nx)
  bv_inv_all   <- matrix(0, n, nx)
  pv_inv_all   <- matrix(0, n, nx)
  b_old        <- matrix(0, length(sel_list), nx)
  b_all0 <- bv_inv_all0 <- pv_inv_all0 <- NULL
  if (!is.null(coords0)) {
    b_all0      <- matrix(0, n0, nx)
    bv_inv_all0 <- matrix(0, n0, nx)
    pv_inv_all0 <- matrix(0, n0, nx)
  }

  ## Process sel_list in chunks to cap peak neighbor-list memory at
  ## O(chunk_size * avg_neighbors). The C++ kernel accumulates in place.
  chunk_size   <- max(1L, min(length(sel_list),
                              as.integer(ceiling(1e8 / max(n, 1L)))))
  chunk_starts <- seq.int(1L, length(sel_list), by = chunk_size)

  for (cs in chunk_starts) {
    ce         <- min(cs + chunk_size - 1L, length(sel_list))
    sel_chunk  <- sel_list[cs:ce]
    query      <- coords_cent[sel_chunk, , drop = FALSE]
    dbnn       <- frNN(x = coords, query = query, eps = threshold, sort = FALSE)
    dbnn0      <- NULL
    if (!is.null(coords0)) {
      dbnn0 <- frNN(x = coords0, query = query, eps = threshold, sort = FALSE)
    }

    lwr_chunk_glm_cpp(
      nb_id            = dbnn$id,
      nb_dist          = dbnn$dist,
      nb_id0_sexp      = if(!is.null(coords0)) dbnn0$id   else NULL,
      nb_dist0_sexp    = if(!is.null(coords0)) dbnn0$dist else NULL,
      sel_chunk        = as.integer(sel_chunk),
      id_train_flag    = id_train_flag,
      resid            = as.numeric(resid),
      w_obs            = as.numeric(w),
      x                = x,
      x0_sexp          = if(!is.null(coords0)) x0 else NULL,
      B_var            = B_var,
      vc_cols          = vc_int,
      band             = band,
      kernel_id        = kernel_id,
      b_all            = b_all,
      bv_inv_all       = bv_inv_all,
      pv_inv_all       = pv_inv_all,
      b_all0_sexp      = if(!is.null(coords0)) b_all0      else NULL,
      bv_inv_all0_sexp = if(!is.null(coords0)) bv_inv_all0 else NULL,
      pv_inv_all0_sexp = if(!is.null(coords0)) pv_inv_all0 else NULL,
      b_old            = b_old
    )
    rm(dbnn); if (!is.null(coords0)) rm(dbnn0)
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
    gmod0         <- glm(y ~ 0 + x + offset(l_pred_off),family=family)
    sse_hv        <- sum(residuals(gmod0, type="deviance")[-id_train]^2 )
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
      pred0       <- rowSums(x0*b_all0)

    } else {
      b_all0 <-bv_all0<-pred0<-NULL
    }

    return(list(beta=b_all, beta_v=bv_all, pred=pred, sel_id=sel_id,
                coords_cent=coords_cent,
                beta0=b_all0,beta0_v=bv_all0, pred0=pred0, b_old=b_old,
                run=run,sse_hv=sse_hv,vc_sel=vc, sse_hv0=sse_hv0))
  } else {
    return(list(run=FALSE))
  }
}

