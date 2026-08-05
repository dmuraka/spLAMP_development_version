## Internal utilities for the coarse-to-fine linear model.
##
## The knot inner loop of lwr() is delegated to the C++ routine
## lwr_chunk_cpp (see src/lwr_chunk.cpp). The chunked driver loop here
## caps peak neighbor-list memory at O(chunk_size * avg_neighbors).

#' @keywords internal
#' @noRd
add_mod <- function(add_learn="rf", train=TRUE, resid, x, coords, x0=NULL, coords0=NULL,
                    id_train=NULL, nx, xname, seed=123, sse_hv=NULL, a_par=NULL){
  a_data          <- data.frame(resid=resid, x[,-1], coords)
  a_xname         <- names(a_data)[-1] <- c(xname[-1],"px","py")
  if(add_learn=="rf"){
    if(!requireNamespace("ranger", quietly=TRUE)){
      stop("add_learn = \"rf\" requires the 'ranger' package. ",
           "Install it with install.packages(\"ranger\").", call. = FALSE)
    }
    if( train ){
      a_run       <- FALSE
      a_par       <- data.frame(mtry=NA, min.node.size=NA)
      a_pred_hv   <- NULL
      mtry_all    <- c( round((nx+1)/5), round((nx+1)/3), round((nx+1)/2))
      param_grid  <- expand.grid(mtry=unique(mtry_all), min.node.size=c(1,5,10))
      param_grid  <- rbind(data.frame(mtry=NA, min.node.size=NA), param_grid)
      for (i in 2:nrow(param_grid)){
        params    <- param_grid[i, ]
        rf_mod    <- ranger::ranger(formula=resid~., data=a_data[id_train,],
                            classification=FALSE, probability=FALSE,
                            verbose=FALSE, mtry=params$mtry, num.trees=500,
                            min.node.size=params$min.node.size)
        ## Validation prediction of the additional learner (already needed for
        ## the SSE); retained for the best candidate to feed cf_lm_hv's pred_hv.
        pred_rf   <- predict(rf_mod, data=a_data[-id_train,])$predictions
        resid_rf  <- resid[-id_train] - pred_rf
        sse_rf    <- sum(resid_rf^2)
        if(sse_rf < sse_hv){
          sse_hv  <- sse_rf
          a_par   <- params
          a_run   <- TRUE
          a_pred_hv <- pred_rf
        }
      }
      return(list(sse_hv=sse_hv, a_par=a_par, a_run=a_run, add_learn=add_learn,
                  a_pred_hv=a_pred_hv))
    } else {
      mod         <- ranger::ranger(formula=resid~., data=a_data, quantreg=TRUE,
                            classification=FALSE, probability=FALSE,
                            verbose=FALSE, mtry=a_par$mtry, num.trees=500,
                            min.node.size=a_par$min.node.size)
      pred        <- mod$predictions
      pred0       <- 0
      ## quantile predictions used downstream to derive the additional-learning
      ## predictive variance (see sample_from_qrf()).
      qlevels     <- seq(0, 1, length.out = 201)
      qmat        <- predict(mod, data=a_data, type="quantiles",
                             quantiles=qlevels)$predictions
      qmat0       <- NULL
      if(!is.null(coords0)){
        a_data0       <- data.frame(x0[,-1], coords0)
        names(a_data0)<- a_xname
        pred0         <- predict(mod, data=a_data0)$predictions
        qmat0         <- predict(mod, data=a_data0, type="quantiles",
                                 quantiles=qlevels)$predictions
      }
      return(list(mod=mod, pred=pred, pred0=pred0, qmat=qmat, qmat0=qmat0,
                  qlevels=qlevels, a_xname=a_xname, add_learn=add_learn))
    }

  } else if(add_learn=="lightgbm"){
    if(!requireNamespace("lightgbm", quietly=TRUE)){
      stop("add_learn = \"lightgbm\" requires the 'lightgbm' package. ",
           "Install it with install.packages(\"lightgbm\").", call. = FALSE)
    }
    X               <- as.matrix(a_data[, a_xname])
    yv              <- a_data$resid
    if( train ){
      ## Tune once on a point-prediction (L2) objective via validation SSE,
      ## mirroring the rf tuning. The chosen hyper-parameters (and the
      ## early-stopped number of rounds) are reused for the quantile models.
      Xtr         <- X[id_train, , drop=FALSE]; ytr <- yv[id_train]
      Xva         <- X[-id_train, , drop=FALSE]; yva <- yv[-id_train]
      dtrain      <- lightgbm::lgb.Dataset(Xtr, label=ytr)
      dvalid      <- lightgbm::lgb.Dataset.create.valid(dtrain, Xva, label=yva)
      grid        <- expand.grid(num_leaves=c(15,31,63), learning_rate=c(0.05,0.1))
      a_run       <- FALSE
      a_pred_hv   <- NULL
      a_par       <- list(num_leaves=31, learning_rate=0.1,
                          min_data_in_leaf=20, feature_fraction=0.9, nrounds=100,
                          train_frac=1)
      for(i in 1:nrow(grid)){
        params    <- list(objective="regression", num_leaves=grid$num_leaves[i],
                          learning_rate=grid$learning_rate[i], min_data_in_leaf=20,
                          feature_fraction=0.9, verbosity=-1, num_threads=0L)
        gbm       <- lightgbm::lgb.train(params, dtrain, nrounds=800,
                                         valids=list(valid=dvalid), eval="l2",
                                         early_stopping_rounds=30, verbose=-1L)
        ## Score on the explicit validation prediction (raw features) rather than
        ## gbm$best_score (computed on binned features), so the selected sse_hv
        ## matches the stored validation prediction exactly.
        pred_va   <- predict(gbm, Xva, num_iteration=gbm$best_iter)
        sse_gbm   <- sum((yva - pred_va)^2)
        if(sse_gbm < sse_hv){
          sse_hv  <- sse_gbm
          ## train_frac records how much of the data the early stopping saw, so
          ## the final all-data refit can scale the round budget accordingly.
          a_par   <- list(num_leaves=grid$num_leaves[i],
                          learning_rate=grid$learning_rate[i], min_data_in_leaf=20,
                          feature_fraction=0.9, nrounds=gbm$best_iter,
                          train_frac=length(id_train)/nrow(X))
          a_run   <- TRUE
          a_pred_hv <- pred_va
        }
      }
      return(list(sse_hv=sse_hv, a_par=a_par, a_run=a_run, add_learn=add_learn,
                  a_pred_hv=a_pred_hv))
    } else {
      base_par    <- list(num_leaves=a_par$num_leaves,
                          learning_rate=a_par$learning_rate,
                          min_data_in_leaf=a_par$min_data_in_leaf,
                          feature_fraction=a_par$feature_fraction,
                          verbosity=-1, num_threads=0L)
      ## The tuned nrounds was early-stopped on the training split only, while
      ## the final models below see all n rows. Scale the round budget by the
      ## same factor so the refit is not systematically under-trained
      ## (train_frac is absent in objects fitted by earlier versions: keep the
      ## stored budget then).
      tfrac       <- if(is.null(a_par$train_frac)) 1 else a_par$train_frac
      nrounds_all <- max(1L, as.integer(ceiling(a_par$nrounds / tfrac)))
      ## Point predictions: trained on all data, as ranger does.
      dall        <- lightgbm::lgb.Dataset(X, label=yv)
      pmod        <- lightgbm::lgb.train(c(base_par, list(objective="regression")),
                                         dall, nrounds=nrounds_all, verbose=-1L)
      pred        <- predict(pmod, X)
      X0          <- NULL; pred0 <- 0
      if(!is.null(coords0)){
        a_data0       <- data.frame(x0[,-1], coords0)
        names(a_data0)<- a_xname
        X0            <- as.matrix(a_data0[, a_xname])
        pred0         <- predict(pmod, X0)
      }
      ## Quantile predictions (raw, trained on all data like ranger). The
      ## combined predictive distribution is calibrated downstream by the total
      ## CQR step in cf_lm(); no component-level conformalization here.
      qlevels     <- c(.025, .05, .1, .25, .5, .75, .9, .95, .975)
      dq          <- lightgbm::lgb.Dataset(X, label=yv)
      qmods       <- lapply(qlevels, function(a)
        lightgbm::lgb.train(c(base_par, list(objective="quantile", alpha=a)),
                            dq, nrounds=nrounds_all, verbose=-1L))
      qpred       <- function(M) t(apply(sapply(qmods, predict, M), 1, sort))
      qmat        <- qpred(X)
      qmat0       <- NULL
      if(!is.null(coords0)) qmat0 <- qpred(X0)
      return(list(mod=qmods, pred=pred, pred0=pred0, qmat=qmat, qmat0=qmat0,
                  qlevels=qlevels, a_xname=a_xname, add_learn=add_learn))
    }

  } else if(add_learn=="none"){
    if( train ){
      return(list(sse_hv=sse_hv, a_par=NA, a_run=FALSE, add_learn=add_learn))
    } else {
      return(list(mod=NA, pred=0, pred0=0, add_learn=add_learn))
    }
  }
}

#' Split-conformal (CQR) offsets for symmetric quantile pairs.
#'
#' For each pair (q, 1-q) with q < 0.5, returns the additive half-width that
#' makes the calibrated interval reach its nominal coverage on the held-out
#' set. The median (q = 0.5) receives a zero offset.
#' @keywords internal
#' @noRd
cqr_offsets <- function(Qcal, ycal, qlevels){
  ncal   <- length(ycal)
  off    <- rep(0, length(qlevels))
  for(j in which(qlevels < 0.5)){
    k    <- which(abs(qlevels - (1 - qlevels[j])) < 1e-9)
    if(length(k) != 1) next
    nominal <- 1 - 2*qlevels[j]
    E    <- pmax(Qcal[, j] - ycal, ycal - Qcal[, k])
    lvl  <- min(ceiling((ncal + 1) * nominal) / ncal, 1)
    Q    <- max(as.numeric(quantile(E, lvl, type=1)), 0)
    off[j] <- Q; off[k] <- Q
  }
  off
}

#' Apply CQR offsets, widening lower quantiles down and upper quantiles up.
#' @keywords internal
#' @noRd
apply_cqr <- function(Q, qlevels, off){
  for(j in seq_along(qlevels)){
    if(qlevels[j] < 0.5)      Q[, j] <- Q[, j] - off[j]
    else if(qlevels[j] > 0.5) Q[, j] <- Q[, j] + off[j]
  }
  t(apply(Q, 1, sort))
}

#' @keywords internal
#' @noRd
sample_from_qrf <- function(rf_qmat, qs, n, n_draw = 100) {
  U      <- matrix(runif(n * n_draw), nrow = n, ncol = n_draw)
  draws  <- matrix(NA_real_, nrow = n, ncol = n_draw)
  for (i in 1:n) {
    qi   <- rf_qmat[i, ]
    if (is.unsorted(qi)) qi <- sort(qi)
    draws[i, ] <- approx(x = qs, y = qi, xout = U[i, ], ties = "ordered",
                         rule = 2)$y
  }
  draws
}

#' Total predictive quantiles: convolve the Gaussian core (mean \code{pred},
#' standard deviation \code{core_sd}) with the centered additional-learning
#' residual distribution (drawn from \code{qmat} at levels \code{qlevels_in}),
#' then take empirical quantiles at \code{qlevels_out}. \code{qmat}'s own mean
#' is removed because it is already included in \code{pred}.
#' @keywords internal
#' @noRd
total_qmat <- function(pred, core_sd, qmat, qlevels_in, qlevels_out, n_draw = 400){
  n   <- length(pred)
  A   <- sample_from_qrf(qmat, qlevels_in, n, n_draw)
  A   <- A - rowMeans(A)
  G   <- matrix(rnorm(n * n_draw), nrow = n, ncol = n_draw) * core_sd
  tot <- pred + G + A
  t(apply(tot, 1, quantile, probs = qlevels_out, type = 7, names = FALSE))
}

#' @keywords internal
#' @noRd
initial_fun <- function(x, y, coords, x_sel=NULL, train_rat, id_train=NULL, seed=NULL){
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
          K_train      <- round(n_uni*train_rat)
          K_val        <- n_uni - K_train
          pick_train   <- K_train <= K_val
          K            <- if(pick_train) K_train else K_val
          iter.max     <- if(n_uni > 20000) 3L else if(n_uni > 5000) 5L else 10L
          suppressWarnings(
            coords_uni_k_tmp <- kmeans(coords_uni, K, iter.max=iter.max)$centers
          )
          sel_uni      <- sort( get.knnx(coords_uni, coords_uni_k_tmp, 1)$nn.index )
          if(pick_train) sel_uni else setdiff(seq_len(n_uni), sel_uni)
        }
      }
      id_train_uni <- if(is.null(seed)) do_split() else withr::with_seed(seed, do_split())
    } else {
      id_train_uni   <- 1:n_uni
    }
    id_train <- which( id_uni %in% id_train_uni )
  }

  n              <- length(y)
  one            <- matrix(1, nrow=n, ncol=1)
  x_pre          <- cbind(one, x)
  x_sel          <- NULL
  if(dim(x_pre)[2] > 1){
    if(is.null(x_sel)) x_sel <- (apply(x_pre, 2, sd) != 0)[-1]
  }
  xname          <- "Intercept"
  if(sum(x_sel) == 1){
    xname        <- c(xname, "x")
  } else if(sum(x_sel) > 1){
    xname        <- c(xname, names(data.frame(x))[x_sel])
  }

  x              <- as.matrix(x_pre[, c(TRUE, x_sel)])
  coords         <- as.matrix(coords)
  nx             <- ncol(x)
  xx_inv         <- solve(t(x) %*% x)
  beta_int       <- xx_inv %*% t(x) %*% y
  row.names(beta_int) <- xname

  pred           <- x %*% beta_int
  resid          <- y - pred
  sig2           <- sum(resid^2) / (n - nx)
  beta_int_vcov  <- sig2 * xx_inv
  beta           <- matrix(beta_int, nrow=n, ncol=nx, byrow=TRUE)
  beta_v         <- matrix(diag(beta_int_vcov), nrow=n, ncol=nx, byrow=TRUE)
  return(list(xx_inv=xx_inv, beta_int=beta_int, x=x, id_train=id_train,
              beta=beta, beta_v=beta_v, pred=pred, resid=resid, n=n, nx=nx,
              x_sel=x_sel, xname=xname, coords=coords))
}

#' @keywords internal
#' @noRd
kfun <- function(dist, band, kernel){
  if(kernel=="gau"){
    wei <- exp(-dist^2/band^2)
  } else if(kernel=="exp"){
    wei <- exp(-dist/band)
  }
  return(wei)
}

#' @keywords internal
#' @noRd
bopt_core <- function(par, bands, Z, beta_int,
                      nx, x, y, n_bid, id_train=NULL) {
  xbeta    <- matrix(0, nrow = nrow(x), ncol = nx)
  w        <- exp(-par / bands)
  w        <- w / w[1]
  bbb      <- Reduce(`+`, lapply(1:n_bid, function(i) w[i] * Z[,i]))
  xbeta[,1]<- x[, 1] * (beta_int[1, 1] + bbb)
  if(nx > 2){
    for(j in 2:nx) xbeta[, j] <- x[, j] * (beta_int[j, 1])
  }

  resid      <- y - rowSums( xbeta[, -1, drop = FALSE] )
  xbeta_tt   <- xbeta[ , 1, drop = FALSE]
  vpar       <- solve(crossprod(xbeta_tt), crossprod(xbeta_tt, resid))
  if(!is.null(id_train)){
    xbeta_test <- xbeta[-id_train, 1, drop = FALSE]
    sse        <- sum((resid[-id_train] - as.vector(xbeta_test %*% vpar))^2)
  } else {
    sse        <- sum((resid - as.vector(xbeta_tt %*% vpar))^2)
  }
  return( list(sse = sse, vpar = vpar ) )
}

#' @keywords internal
#' @noRd
lwr <- function(coords, coords_uni, resid, x, band, b_old, vc,
                ridge, coords_old=NULL, kernel, id_train, y, beta=NULL,
                coords0, x0, sel_id=NULL, func="cf_lm"){

  n            <- nrow(coords)
  nx           <- ncol(x)
  if(kernel=="gau"){
    threshold  <- sqrt(-log(0.05))*band
    kernel_id  <- 2L
  } else if(kernel=="exp"){
    threshold  <- -log(0.05)*band
    kernel_id  <- 1L
  }

  if(is.null(sel_id)){ # cf_lm_hv
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
        iter.max <- ifelse( n_uni > 5000, 5, 10)
        withr::with_seed(4321,{
          suppressWarnings(coords_k_tmp<- kmeans(coords_uni, n_knot, iter.max=iter.max)$centers)
        })
        sel_id   <- get.knnx(coords_uni, coords_k_tmp, 1)$nn.index
      }
      coords_cent <- coords_uni[sel_id,]
      sel_list    <- 1:nrow(coords_cent)
    } else {
      n_knot      <- n_uni
      coords_cent <- coords_uni
      sel_list    <- 1:n_knot
      sel_id      <- NA
    }
  } else if(is.na(sel_id[1])){
    coords_cent <- coords_uni
    n_knot      <- nrow(coords_cent)
    sel_list    <- 1:n_knot
  } else {
    n_knot      <- length(sel_id)
    sel_list    <- 1:n_knot
    coords_cent <- coords_uni[sel_id,]
  }

  ################# Prior coefficient variance
  B_var          <- matrix(Inf, nrow=n_knot, ncol=nx)
  if( !is.null(b_old) & ridge==TRUE ){
    for(i in 1:nx) B_var[,i] <- mean(b_old[,i]^2)
  }

  ################# accumulators via the fused nanoflann kernel (LM: is_lm=1)
  n0 <- 0L
  if(!is.null(coords0)) n0 <- nrow(coords0)

  id_train_int            <- integer(n)
  id_train_int[id_train]  <- 1L
  vc_int                  <- as.integer(vc)

  ## Fused kernel (src/lwr_chunk_glm_fused.cpp, is_lm=1): builds a kd-tree over
  ## `coords` (and `coords0`) ONCE and does radius-search -> local LM ->
  ## scatter-add per knot inline, WITHOUT materialising the neighbour lists.
  ## With unit weights and is_lm=1 (variance over all neighbours / (m-1)) this
  ## is numerically identical to the frNN + lwr_chunk_cpp path (~1e-13), at much
  ## lower peak memory.
  fres <- lwr_glm_fused_cpp(
    coords       = coords,
    coords_cent  = matrix(coords_cent, ncol = 2),
    resid        = as.numeric(resid),
    w_obs        = rep(1, n),
    x            = x,
    id_train     = id_train_int,
    B_var        = B_var,
    vc_cols      = vc_int,
    band         = band,
    kernel_id    = kernel_id,
    threshold    = threshold,
    is_lm        = 1L,
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
  if( func == "cf_lm_hv" ){
    b_all_sel    <- b_all[-id_train, vc, drop=FALSE] / pv_inv_all[-id_train, vc, drop=FALSE]
    b_all_sel[is.nan(b_all_sel)] <- 0
    pred_hv      <- rowSums(x[-id_train, vc, drop=FALSE] * b_all_sel)
    resid_hv     <- resid[-id_train] - pred_hv
    sse_hv0      <- sum((resid[-id_train])^2)
    sse_hv       <- sum((resid_hv)^2)
    run          <- ifelse(sse_hv < sse_hv0, TRUE, FALSE)
    if(!run){
      sse_hv     <- sse_hv0
    }
  } else {
    sse_hv      <- NA
    run         <- TRUE
  }

  if( run ){
    bv_all          <- bv_inv_all
    b_all[,vc]      <- b_all[,vc]/pv_inv_all[,vc]
    b_all[,-vc]     <- 0
    b_all[is.nan(b_all)] <- 0

    bv_inv_all[, vc]<- bv_inv_all[, vc]/pv_inv_all[, vc]
    bv_all[, vc]    <- 1/bv_inv_all[, vc]
    bv_all[, -vc]   <- NA
    bv_all[is.nan(bv_all)] <- Inf

    ## Predictive variance per eq. (10): pred_se^2 = 1 / sum_k (w_k / pv_k),
    ## where pv_inv_all = sum_k (w_k / pv_k). Unlike the coefficient variance
    ## bv_all, this grows as the location moves away from data (all w_k -> 0).
    pv_all          <- 1/pv_inv_all
    pv_all[, -vc]   <- NA
    pv_all[is.nan(pv_all)] <- Inf

    pred            <- rowSums(x * b_all)
    if( !is.null(coords0) ){
      bv_all0         <- bv_inv_all0
      b_all0[,vc]     <- b_all0[,vc]/pv_inv_all0[,vc]
      b_all0[,-vc]    <- 0
      b_all0[is.nan(b_all0)] <- 0
      b_all0[is.na(b_all0)]  <- 0

      bv_inv_all0[, vc]<- bv_inv_all0[, vc]/pv_inv_all0[, vc]
      bv_all0[, vc]    <- 1/bv_inv_all0[, vc]
      bv_all0[, -vc]   <- NA
      bv_all0[is.nan(bv_all0)] <- Inf
      pv_all0          <- 1/pv_inv_all0
      pv_all0[, -vc]   <- NA
      pv_all0[is.nan(pv_all0)] <- Inf
      pred0            <- rowSums(x0 * b_all0)
    } else {
      b_all0 <- bv_all0 <- pv_all0 <- pred0 <- NULL
    }

    return(list(beta=b_all, beta_v=bv_all, beta_pv=pv_all, pred=pred, sel_id=sel_id,
                coords_cent=coords_cent, beta0=b_all0, beta0_v=bv_all0,
                beta0_pv=pv_all0, pred0=pred0, b_old=b_old, run=run,
                sse_hv=sse_hv, vc_sel=vc))
  } else {
    return(list(run=FALSE))
  }
}
