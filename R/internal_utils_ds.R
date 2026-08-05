## Internal utilities for the coarse-to-fine spatial downscaling model.
##
## The per-knot inner loop of lwr_ds() is delegated to the C++ routine
## lwr_ds_chunk_cpp (see src/lwr_ds_chunk.cpp). Per-area aggregation uses
## std::unordered_map to avoid the R aggregate() overhead.

#' Scale selection: validation SSE minimization.
#'
#' Pick the scale with the smallest validation SSE across all recorded
#' scales. The SSE vector includes the init value at index 1, so the
#' returned `opt_id` is in the bands index space (0 = use only init,
#' k >= 1 = use bands[1:k]).
#'
#' @keywords internal
#' @noRd
select_opt_id <- function(SSE_valid){
  if(length(SSE_valid) < 2 || all(!is.finite(SSE_valid))) return(0L)
  finite_idx <- which(is.finite(SSE_valid))
  if(length(finite_idx) == 0) return(0L)
  best <- finite_idx[which.min(SSE_valid[finite_idx])]
  max(best - 1L, 0L)
}

#' Multiplicative pycnophylactic finalize: scale per-area so that
#' \code{aggregate(a*pred, agg_id, sum) == Y} exactly. Areas where pred
#' is all-zero but Y != 0 are first filled uniformly with Y_i/sum(a_i).
#'
#' @keywords internal
#' @noRd
multiplicative_pycnophylactic <- function(pred, Y, agg_id, a){
  Pred      <- aggregate(a*pred, by = list(agg_id), sum)[, 2]
  zero_mask <- Pred == 0 & Y != 0
  if(any(zero_mask)){
    a_sum     <- aggregate(a, by = list(agg_id), sum)[, 2]
    area_vals <- sort(unique(agg_id))
    for(i_area in which(zero_mask)){
      in_area <- which(agg_id == area_vals[i_area])
      if(length(in_area) > 0 && a_sum[i_area] > 0){
        pred[in_area] <- Y[i_area] / a_sum[i_area]
      }
    }
    Pred <- aggregate(a*pred, by = list(agg_id), sum)[, 2]
  }
  ratio <- ifelse(Pred > 0, Y / Pred, 0)
  pred * ratio[agg_id]
}

#' Initial fit for the downscaling model.
#'
#' Sets up areal design X = aggregate(a*x, agg_id, sum), estimates an
#' auto-eps variance-component (nugget) model from the OLS residuals,
#' and refits beta with W_glob = 1/(sum(a^2) + eps).
#'
#' @keywords internal
#' @noRd
initial_ds_fun <- function(Y, Y_type, x, a, coords, train_rat, Id_train=NULL,
                           agg_id){

  N              <- length(Y)
  n              <- nrow(coords)

  ## Downstream code (lwr_ds spatial process, pred = x %*% beta + pred_sp)
  ## assumes x[, 1] is a constant intercept column. If the caller passed
  ## an x whose first column is not constant 1, prepend the intercept.
  if(is.null(x)){
    x            <- matrix(1, nrow = n, ncol = 1)
  } else {
    x            <- as.matrix(x)
    if(!all(x[, 1] == 1)){
      x          <- cbind(1, x)
    }
  }
  nx             <- ncol(x)
  xname          <- names(data.frame(x))

  if(is.null(a)) a <- rep(1, n)

  if(Y_type=="sum"){
    W            <- rep(1, N)
  } else if(Y_type=="mean"){
    A            <- aggregate(a, by=list(agg_id), sum)[,2]
    a            <- a / A[agg_id]
    W            <- rep(1, N)
  }

  Agg_id         <- sort(unique(agg_id))
  Coords         <- aggregate(coords, by=list(agg_id), mean)[,-1, drop=FALSE]
  X              <- as.matrix(aggregate(a*x, by=list(agg_id), sum)[,-1, drop=FALSE])
  Id_uni         <- match(paste(Coords[,1], Coords[,2]),
                          unique(paste(Coords[,1], Coords[,2])))
  Coords_uni     <- unique(Coords)
  coords_uni     <- unique(coords)
  N_uni          <- nrow(Coords_uni)
  n_uni          <- nrow(coords_uni)
  if(is.null(Id_train)){
    if(train_rat < 1){
      if(N_uni <= 1000){
        suppressWarnings(Coords_uni_k_tmp <- kmeans(coords_uni, round(N_uni*train_rat))$centers)
        Id_train_uni    <- sort( get.knnx(Coords_uni, Coords_uni_k_tmp, 1)$nn.index )
      } else {
        Id_train_uni    <- sort(sample(N_uni, round(N_uni*train_rat)))
      }
    } else {
      Id_train_uni      <- 1:N_uni
    }
    Id_train            <- which( Id_uni %in% Id_train_uni )
  }

  Coords         <- as.matrix(Coords)

  ## Area-level weight for the GLOBAL areal regressions, derived from an iid
  ## variance-component (nugget) model. Under independent cell-level noise,
  ## Var(Y_i) = sigma^2 * sum_{j in i} a_j^2 + tau^2. We estimate
  ## (sigma^2, tau^2) by regressing squared OLS residuals on
  ## s2_i = sum_{j in i} a_j^2 over training areas, then weight by
  ## W_glob_i = 1 / (s2_i + eps), eps = tau^2 / sigma^2. eps -> 0 recovers
  ## the pure 1/sum(a^2) GLS weight (independent cells); large eps drives
  ## W_glob toward a constant (area-level common shock dominates). The C++
  ## local regression keeps W_area = 1 because its V_i already carries
  ## sum a_j^2, so W_glob is applied only to the global areal regressions.
  s2_area        <- as.numeric(aggregate(a^2, by=list(agg_id), sum)[,2])
  Gmod_ols       <- lm(Y ~ 0 + as.matrix(X), weights=W)
  r0_area        <- as.numeric(Y - as.matrix(X) %*% matrix(coefficients(Gmod_ols)))
  vc_fit         <- lm(r0_area[Id_train]^2 ~ s2_area[Id_train])
  sig2_hat       <- as.numeric(coef(vc_fit)[2])
  tau2_hat       <- max(as.numeric(coef(vc_fit)[1]), 0)
  if(is.finite(sig2_hat) && sig2_hat > 0){
    eps_auto     <- tau2_hat / sig2_hat
    denom_floor  <- max(1e-6 * max(s2_area), .Machine$double.eps)
    W_glob       <- 1 / pmax(s2_area + eps_auto, denom_floor)
  } else {
    eps_auto     <- Inf
    W_glob       <- rep(1, N)
  }

  Gmod           <- lm(Y ~ 0 + as.matrix(X), weights=W_glob)
  beta_int       <- matrix( coefficients(Gmod) )
  row.names(beta_int) <- xname

  Pred           <- as.matrix(X) %*% beta_int
  Resid          <- Y - Pred
  pred           <- as.matrix(x) %*% beta_int

  beta           <- matrix(beta_int, nrow = n, ncol = nx, byrow = TRUE)

  return(list(beta_int=beta_int, beta=beta, coords_uni=coords_uni,
              Coords_uni=Coords_uni, pred=pred, Pred=Pred, Resid=Resid,
              X=X, W=W, W_glob=W_glob, eps_auto=eps_auto, sig2_hat=sig2_hat,
              s2_area=s2_area, Agg_id=Agg_id, x=x, a=a, xname=xname,
              n=n, nx=nx, N=N, Id_train=Id_train))
}

#' Per-knot areal weighted local regression for the downscaling model.
#'
#' @keywords internal
#' @noRd
lwr_ds <- function(coords, coords_uni, Resid, beta_int, Coords_uni,
                   Y, X, W, x, a, band, b_old, ridge=FALSE,
                   kernel, Id_train, sel_id=NULL, sse_hv0, pred_sp,
                   agg_id, Agg_id, func="cf_downscale_hv", c_shrink=0,
                   knots_train_only=FALSE){

  N            <- length(Y)
  n            <- nrow(coords)
  nx           <- ncol(x)
  if(kernel=="gau"){
    threshold  <- sqrt(-log(0.05))*band
  } else if(kernel=="exp"){
    threshold  <- -log(0.05)*band
  }

  if(is.null(sel_id)){
    ## Pool of candidate knot locations
    if(knots_train_only){
      pool_area_ids    <- Id_train
      pool_point_mask  <- agg_id %in% Id_train
    } else {
      pool_area_ids    <- Agg_id
      pool_point_mask  <- rep(TRUE, n)
    }
    pool_point_idx     <- which(pool_point_mask)
    n_pool_pts         <- length(pool_point_idx)
    n_pool_area        <- length(pool_area_ids)
    coords_pool_pts    <- coords[pool_point_idx, , drop=FALSE]

    area         <- (max(coords[,1])-min(coords[,1]))^2 + (max(coords[,2])-min(coords[,2]))^2
    n_knot       <- round(1.5*area/band^2)
    if(n_knot < n_pool_area){
      withr::with_seed(4321, {
        suppressWarnings(coords_cent <- kmeans(coords_pool_pts, n_knot)$centers)
      })
    } else if(n_knot == n_pool_area){
      withr::with_seed(4321, {
        agg_sel    <- sapply(pool_area_ids, function(aid){
                        sample(which(agg_id == aid), 1)
                      })
      })
      coords_cent <- coords[agg_sel, , drop=FALSE]
    } else if(n_pool_pts > n_knot & n_knot > n_pool_area){
      withr::with_seed(4321, {
        agg_sel_a  <- sapply(pool_area_ids, function(aid){
                        sample(which(agg_id == aid), 1)
                      })
        n_knot_add <- n_knot - n_pool_area
        agg_sel_b  <- sample(setdiff(pool_point_idx, agg_sel_a), n_knot_add)
      })
      agg_sel    <- c(agg_sel_a, agg_sel_b)
      coords_cent <- coords[agg_sel, , drop=FALSE]
    } else {
      coords_cent <- coords_pool_pts
    }
    n_knot       <- nrow(coords_cent)
    sel_list     <- 1:n_knot
  } else if(is.na(sel_id[1])){
    coords_cent  <- coords_uni
    n_knot       <- nrow(coords_cent)
    sel_list     <- 1:n_knot
  } else {
    n_knot       <- length(sel_id)
    sel_list     <- 1:n_knot
    coords_cent  <- coords_uni[sel_id,]
  }

  ## Prior coefficient variance (Inf = no ridge).
  B_var          <- Inf
  if( !is.null(b_old) & ridge==TRUE ){
    B_var        <- mean(b_old^2)
  }

  Id_train_flag           <- logical(N)
  Id_train_flag[Id_train] <- TRUE
  id_train_flag           <- agg_id %in% Id_train
  query                   <- coords_cent[sel_list, , drop=FALSE]

  local_bands  <- rep(band, nrow(query))
  if(kernel == "gau") local_thresh <- sqrt(-log(0.05))*local_bands
  else                local_thresh <- -log(0.05)*local_bands
  max_threshold<- max(local_thresh)
  dbnn         <- frNN(x = coords, query = query, eps = max_threshold, sort = FALSE)
  ## filter: per-knot local threshold
  keep_per_knot<- mapply(function(di, th) di <= th, dbnn$dist, local_thresh,
                         SIMPLIFY = FALSE)
  dbnn$dist    <- mapply(function(d, k) d[k], dbnn$dist, keep_per_knot,
                         SIMPLIFY = FALSE)
  dbnn$id      <- mapply(function(i, k) i[k], dbnn$id, keep_per_knot,
                         SIMPLIFY = FALSE)
  sel_list     <- sel_list[unlist(lapply(dbnn$id, length)) >= 3]
  if(length(sel_list) == 0){
    return(list(run = FALSE))
  }

  b_all        <- matrix(0, n, nx)
  bv_inv_all   <- matrix(0, n, nx)
  pv_inv_all   <- matrix(0, n, nx)

  b_old_mat    <- matrix(0, length(sel_list), nx)
  kernel_id    <- if(kernel == "gau") 2L else 1L
  nb_id_sub    <- dbnn$id[sel_list]
  nb_dist_sub  <- dbnn$dist[sel_list]
  B_var_col    <- rep(B_var, length(sel_list))

  lwr_ds_chunk_cpp(
    nb_id         = nb_id_sub,
    nb_dist       = nb_dist_sub,
    sel_chunk     = as.integer(seq_along(sel_list)),
    local_bands   = as.numeric(local_bands[sel_list]),
    kernel_id     = kernel_id,
    vc            = 1L,
    resid_area    = as.numeric(Resid),
    X_area        = as.numeric(X[, 1]),
    W_area        = as.numeric(W),
    a             = as.numeric(a),
    agg_id        = as.integer(agg_id),
    id_train_flag = as.logical(id_train_flag),
    x             = x,
    B_var_col     = B_var_col,
    c_shrink      = as.numeric(c_shrink),
    b_all         = b_all,
    bv_inv_all    = bv_inv_all,
    pv_inv_all    = pv_inv_all,
    b_old         = b_old_mat
  )
  b_old        <- b_old_mat[, 1]

  ## Selection through CV (if cf_downscale_hv) or accept (if cf_downscale).
  run          <- FALSE
  vc           <- 1
  if(func == "cf_downscale_hv"){
    bv_all                       <- bv_inv_all
    b_all[, vc]                  <- b_all[, vc, drop=FALSE] / pv_inv_all[, vc, drop=FALSE]
    b_all[, -vc]                 <- 0
    b_all[is.nan(b_all)]         <- 0
    pred_sp_add                  <- x[, vc] * b_all[, vc]
    pred_sp                      <- pred_sp + pred_sp_add
    pred_sp                      <- pred_sp - mean(pred_sp)
    Pred_sp                      <- aggregate(a*pred_sp, by=list(agg_id), sum)[,2]
    Gmod0                        <- lm(Y - Pred_sp ~ 0 + as.matrix(X), weights=W)
    sse_hv_valid                 <- sum(W[-Id_train] * residuals(Gmod0)[-Id_train]^2)
    sse_hv                       <- sse_hv_valid
    run                          <- ifelse(sse_hv < sse_hv0, TRUE, FALSE)
    if(!run) sse_hv              <- sse_hv0
  } else {
    bv_all                       <- bv_inv_all
    b_all[, vc]                  <- b_all[, vc] / pv_inv_all[, vc]
    b_all[, -vc]                 <- 0
    b_all[is.nan(b_all)]         <- 0
    pred_sp_add                  <- x[, vc] * b_all[, vc]
    pred_sp                      <- pred_sp + pred_sp_add
    Pred_sp                      <- aggregate(a*pred_sp, by=list(agg_id), sum)[,2]
    sse_hv                       <- NA
    run                          <- TRUE
  }

  if(run){
    bv_inv_all[, vc] <- bv_inv_all[, vc] / pv_inv_all[, vc]
    bv_all[, vc]     <- 1 / bv_inv_all[, vc]
    bv_all[, -vc]    <- NA
    bv_all[is.nan(bv_all)] <- Inf

    return(list(beta=b_all, beta_v=bv_all, sel_id=sel_id,
                coords_cent=coords_cent,
                pred_sp=pred_sp, Pred_sp=Pred_sp,
                b_old=b_old, run=run, sse_hv=sse_hv,
                vc_sel=vc, sse_hv0=sse_hv0))
  } else {
    return(list(run=FALSE))
  }
}
