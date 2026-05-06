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
    if( train ){
      a_run       <- FALSE
      a_par       <- data.frame(mtry=NA, min.node.size=NA)
      mtry_all    <- c( round((nx+1)/5), round((nx+1)/3), round((nx+1)/2))
      param_grid  <- expand.grid(mtry=unique(mtry_all), min.node.size=c(1,5,10))
      param_grid  <- rbind(data.frame(mtry=NA, min.node.size=NA), param_grid)
      for (i in 2:nrow(param_grid)){
        params    <- param_grid[i, ]
        rf_mod    <- ranger(formula=resid~., data=a_data[id_train,],
                            classification=FALSE, probability=FALSE,
                            verbose=FALSE, mtry=params$mtry, num.trees=500,
                            min.node.size=params$min.node.size)
        resid_rf  <- resid[-id_train] - predict(rf_mod, data=a_data[-id_train,])$predictions
        sse_rf    <- sum(resid_rf^2)
        if(sse_rf < sse_hv){
          sse_hv  <- sse_rf
          a_par   <- params
          a_run   <- TRUE
        }
      }
      return(list(sse_hv=sse_hv, a_par=a_par, a_run=a_run, add_learn=add_learn))
    } else {
      mod         <- ranger(formula=resid~., data=a_data, quantreg=TRUE,
                            classification=FALSE, probability=FALSE,
                            verbose=FALSE, mtry=a_par$mtry, num.trees=500,
                            min.node.size=a_par$min.node.size)
      pred        <- mod$predictions
      pred0       <- 0
      if(!is.null(coords0)){
        a_data0       <- data.frame(x0[,-1], coords0)
        names(a_data0)<- a_xname
        pred0         <- predict(mod, data=a_data0)$predictions
      }
      return(list(mod=mod, pred=pred, pred0=pred0, a_xname=a_xname, add_learn=add_learn))
    }
  } else if(add_learn=="none"){
    if( train ){
      return(list(sse_hv=sse_hv, a_par=NA, a_run=FALSE, add_learn=add_learn))
    } else {
      return(list(mod=NA, pred=0, pred0=0, add_learn=add_learn))
    }
  }
}

#' @keywords internal
#' @noRd
sample_from_qrf <- function(rf_qmat, qs, n, n_draw = 100) {
  U      <- matrix(runif(n * n_draw), nrow = n, ncol = n_draw)
  draws  <- matrix(NA_real_, nrow = n, ncol = n_draw)
  for (i in 1:n) {
    qi   <- rf_qmat[i, ]
    if (is.unsorted(qi)) qi <- sort(qi)
    draws[i, ] <- approx(x = qs, y = qi, xout = U[i, ], ties = "ordered")$y
  }
  draws
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

    lwr_chunk_cpp(
      nb_id            = dbnn$id,
      nb_dist          = dbnn$dist,
      nb_id0_sexp      = if(!is.null(coords0)) dbnn0$id   else NULL,
      nb_dist0_sexp    = if(!is.null(coords0)) dbnn0$dist else NULL,
      sel_chunk        = as.integer(sel_chunk),
      id_train_flag    = id_train_flag,
      resid            = as.numeric(resid),
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
      pred0            <- rowSums(x0 * b_all0)
    } else {
      b_all0 <- bv_all0 <- pred0 <- NULL
    }

    return(list(beta=b_all, beta_v=bv_all, pred=pred, sel_id=sel_id,
                coords_cent=coords_cent, beta0=b_all0, beta0_v=bv_all0,
                pred0=pred0, b_old=b_old, run=run, sse_hv=sse_hv, vc_sel=vc))
  } else {
    return(list(run=FALSE))
  }
}
