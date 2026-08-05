#' Holdout validation for the coarse-to-fine spatial downscaling (CF-DS)
#'
#' Trains the CF-DS model and selects the number of spatial scales through
#' sequential holdout validation.
#'
#' @param Y Vector of aggregate-level response values (length \code{N}).
#' @param Y_type Aggregation type of \code{Y}: \code{"sum"} for extensive
#'   (count-like) data (e.g., population) or \code{"mean"} for intensive
#'   (density-like) data (e.g., population density, average temperature).
#' @param x Matrix of disaggregate-level covariates (\code{n x K}).
#' @param prop_weight Vector of disaggregate-level proportional allocation
#'   weights (length \code{n}) used to distribute the aggregate-level response
#'   across the disaggregate-level units. When \code{Y_type="mean"}, prop_weight
#'   should corresponding to the denominator of the intensive response variable.
#'   Examples include residential land area for population downscaling,
#'   population for morbidity downscaling, and \code{NULL} (= \code{rep(1, n)})
#'   for temperature downscaling.
#' @param coords Matrix of disaggregate-level coordinates (\code{n x 2}).
#' @param agg_id Area ID for each disaggregate-level unit (length \code{n}).
#' @param train_rat Ratio of the aggregate-level units used for model training
#'   (default 0.75) in the holdout validation.
#' @param id_train Optional. If specified, the corresponding aggregate-level
#'   units are used as training units. Otherwise, training units are chosen
#'   based on `train_rat`.
#' @param alpha Decay ratio of the kernel bandwidth in the coarse-to-fine
#'   training (default: 0.9). Values closer to one make the optimization
#'   more stringent but increase computation time.
#' @param kernel Kernel type for modeling spatial dependence. `"exp"` for the
#'   exponential kernel (default) and `"gau"` for the Gaussian kernel.
#' @param rel_tol Relative improvement threshold for validation SSE
#'   (default \code{1e-4}). At each scale, the spatial process is retained
#'   only if validation SSE improves by more than \code{rel_tol}; otherwise
#'   a stopping counter is incremented, and learning stops once 5 consecutive
#'   scales fail to improve. Larger values stop earlier, whereas smaller
#'   values allow finer scales to be selected.
#' @param seed Random seed used for the training/validation split when
#'   `id_train` is not supplied. Default is `123`. Set to `NULL` to allow
#'   a different split at each call (useful for assessing split sensitivity).
#'
#' @return A list with the following elements:
#' \describe{
#'   \item{sse_hv}{Final sum-of-squared error (SSE) for validation samples.}
#'   \item{sse_hv_all}{SSEs obtained at each learning step.}
#'   \item{id_train}{ID of training aggregate-level units.}
#'   \item{other}{Other internally used output objects.}
#' }
#'
#' @references
#' Murakami, D., Chun, Y., Yoshida, T., & Seya, H. (2026).
#' Scalable coarse-to-fine spatial downscaling. *ArXiv preprint*.
#'
#' @seealso \code{\link{cf_downscale}}, \code{\link{cf_lm_hv}}
#' @author Daisuke Murakami
#'
#' @importFrom dbscan frNN
#' @importFrom FNN get.knnx
#' @importFrom stats aggregate coefficients kmeans lm quantile residuals sd
#'
#' @export
cf_downscale_hv     <- function(Y, Y_type="sum", x=NULL, prop_weight=NULL,
                         coords, agg_id,
                         train_rat=0.75, id_train=NULL,
                         alpha=0.9, kernel="exp",
                         rel_tol=1e-4, seed=123){

  if(!is.character(Y_type) || length(Y_type) != 1L ||
     !(Y_type %in% c("sum","mean"))){
    .spcf_stop("'Y_type' must be either \"sum\" (areal totals) or \"mean\" (areal means); got ",
               paste(sQuote(Y_type), collapse=", "), ".")
  }
  dims            <- .spcf_check_downscale(Y, x, prop_weight, coords, agg_id)
  .spcf_check_hv_args(dims$N, train_rat, id_train, alpha, kernel)

  ## Internal code uses the paper notation `a` for the proportional
  ## allocation weight; bind the user-facing argument to it once here.
  a               <- prop_weight

  ## Hardcoded loop limits (match the cf_lm_hv convention):
  ##   stop_k    = consecutive-plateau count before terminating the loop
  ##   max_iter  = hard cap on the number of bandwidth scales considered
  stop_k          <- 5L
  max_iter        <- 100L

  ## Aggregation gate of the stopping rule: the loop may terminate only after
  ##   the agg_q-quantile of |Y_i - agg(a*pred)_i| / sd(Y) over multi-point
  ##   training areas has dropped to <= agg_tol at least once (sticky).
  agg_tol         <- 0.10
  agg_q           <- 0.95

  if(!is.null(seed) && is.null(id_train)){
    init <- withr::with_seed(seed,
              initial_ds_fun(Y=Y, Y_type=Y_type, x=x, a=a, coords=coords,
                             train_rat=train_rat, Id_train=id_train,
                             agg_id=agg_id))
  } else {
    init <- initial_ds_fun(Y=Y, Y_type=Y_type, x=x, a=a, coords=coords,
                           train_rat=train_rat, Id_train=id_train,
                           agg_id=agg_id)
  }
  beta_int     <- init$beta_int
  beta         <- as.numeric(beta_int)
  coords_uni   <- init$coords_uni
  Coords_uni   <- init$Coords_uni
  Pred         <- init$Pred
  Resid        <- init$Resid
  X            <- init$X
  W            <- init$W
  W_glob       <- init$W_glob
  x            <- init$x
  a            <- init$a
  xname        <- init$xname
  n            <- init$n
  nx           <- init$nx
  N            <- init$N
  Agg_id       <- init$Agg_id
  id_train     <- init$Id_train

  use_valid    <- length(id_train) < N

  ## Refit the initial beta on training-only data so that the per-scale
  ## training-only LM refits (which gain a sequential gamma_k coefficient)
  ## form a strictly nested chain. This guarantees SSE_train is
  ## monotonically non-increasing throughout the Phase 1 loop.
  Xmat_tr        <- as.matrix(X)[id_train, , drop=FALSE]
  Gmod_init_tr   <- lm(Y[id_train] ~ 0 + Xmat_tr, weights=W_glob[id_train])
  beta_int       <- matrix(coefficients(Gmod_init_tr))
  rownames(beta_int) <- xname
  beta           <- as.numeric(beta_int)
  Resid          <- as.numeric(Y - as.matrix(X) %*% beta_int)
  sse_tr0      <- sum(W_glob[id_train]  * Resid[id_train]^2)
  sse_va0      <- if(use_valid) sum(W_glob[-id_train] * Resid[-id_train]^2) else NA_real_

  Bands_max    <- max_iter
  max_d        <- sqrt(diff(range(coords[,1]))^2 + diff(range(coords[,2]))^2)/3
  Bands        <- max_d * alpha^(1:Bands_max)

  b_old        <- NULL
  bands        <- NULL
  bid          <- NULL
  pred_sp      <- 0
  SSE_train    <- sse_tr0
  SSE_valid    <- sse_va0
  count_plateau<- 0
  count_norun  <- 0

  pred_sp_add_list<- list()

  ## Phase 1 gate metric (multi-point training areas only).
  area_size       <- tabulate(agg_id, nbins = length(Y))
  Id_train_agg    <- intersect(id_train, which(area_size > 1))
  if(length(Id_train_agg) == 0) Id_train_agg <- id_train
  Y_sd_const      <- sd(Y)
  if(!is.finite(Y_sd_const) || Y_sd_const <= 0) Y_sd_const <- 1
  AGG_err_tr      <- unname(quantile(abs(Resid[Id_train_agg]) / Y_sd_const,
                                     probs = agg_q, names = FALSE))

  ## Match the cf_lm_hv message style.
  message("--- SSE: Linear regression ---")
  SSE_init <- if(use_valid) sse_va0 else sse_tr0
  message(formatC(SSE_init, digits = 7, format = "g"))

  SSE      <- SSE_init
  SSE_name <- "linear regression"

  message("--- SSE: Learning multi-scale spatial process ---")

  ## ---- agg_constrained main loop ----------------------------------------
  agg_satisfied <- FALSE
  plateau_va    <- 0L
  Pred_sp_areal <- numeric(N)
  gamma_list    <- numeric(0)
  beta_list     <- list()
  Xmat          <- as.matrix(X)
  for(i in seq_along(Bands)){
    band       <- Bands[i]
    lmod       <- lwr_ds(coords=coords, coords_uni=coords_uni,
                         beta_int=beta_int, Coords_uni=Coords_uni,
                         Resid=Resid, Y=Y, X=X, W=W, x=x, a=a, band=band,
                         b_old=b_old, ridge=FALSE, kernel=kernel,
                         sel_id=NULL, sse_hv0=NULL, pred_sp=pred_sp,
                         Id_train=id_train, agg_id=agg_id,
                         Agg_id=Agg_id, func="cf_downscale",
                         knots_train_only=TRUE, c_shrink=0)
    if(isFALSE(lmod$run)){
      count_norun <- count_norun + 1
      if(count_norun >= stop_k) break
      next
    }
    count_norun  <- 0

    ## Sequential gamma_k step.
    incr_raw     <- lmod$pred_sp - pred_sp
    incr_pt      <- incr_raw - mean(incr_raw)
    Pred_sp_add  <- as.numeric(aggregate(a*incr_pt, by=list(agg_id),
                                         sum)[,2])
    Y_lhs        <- Y - Pred_sp_areal
    design_train <- cbind(Xmat[id_train, , drop=FALSE],
                          Pred_sp_add[id_train])
    Gmod_cand    <- lm(Y_lhs[id_train] ~ 0 + design_train,
                       weights = W_glob[id_train])
    coef_full    <- as.numeric(coefficients(Gmod_cand))
    beta_cand    <- coef_full[seq_len(nx)]
    gamma_k      <- coef_full[nx + 1L]
    if(!is.finite(gamma_k)) gamma_k <- 0
    if(gamma_k > 1){
      gamma_k    <- 1
      Gmod2      <- lm(Y_lhs[id_train] ~ 0 + Xmat[id_train, , drop=FALSE] +
                       offset(Pred_sp_add[id_train]),
                       weights = W_glob[id_train])
      beta_cand  <- as.numeric(coefficients(Gmod2))
    } else if(gamma_k < 0){
      gamma_k    <- 0
      Gmod2      <- lm(Y_lhs[id_train] ~ 0 + Xmat[id_train, , drop=FALSE],
                       weights = W_glob[id_train])
      beta_cand  <- as.numeric(coefficients(Gmod2))
    }

    pred_sp_cand       <- pred_sp + gamma_k * incr_pt
    Pred_sp_areal_cand <- Pred_sp_areal + gamma_k * Pred_sp_add
    Resid_cand         <- as.numeric(Y - Xmat %*% beta_cand -
                                     Pred_sp_areal_cand)
    sse_tr_cand        <- sum(W_glob[id_train]  * Resid_cand[id_train]^2)
    sse_va_cand        <- if(use_valid)
                            sum(W_glob[-id_train] * Resid_cand[-id_train]^2)
                          else NA_real_
    agg_err_tr_cand    <- unname(quantile(
                            abs(Resid_cand[Id_train_agg]) / Y_sd_const,
                            probs = agg_q, names = FALSE))

    b_old        <- lmod$b_old
    pred_sp      <- pred_sp_cand
    Pred_sp_areal<- Pred_sp_areal_cand
    Resid        <- Resid_cand
    beta         <- beta_cand
    pred_sp_add_list[[length(pred_sp_add_list) + 1L]] <- incr_pt
    gamma_list   <- c(gamma_list, gamma_k)
    beta_list[[length(beta_list) + 1L]] <- beta_cand
    bands        <- c(bands, band)
    bid          <- c(bid, i)
    SSE_train    <- c(SSE_train, sse_tr_cand)
    SSE_valid    <- c(SSE_valid, sse_va_cand)
    AGG_err_tr   <- c(AGG_err_tr, agg_err_tr_cand)

    ## Sticky agg-satisfaction flag.
    if(!agg_satisfied && agg_err_tr_cand <= agg_tol) agg_satisfied <- TRUE

    ## Validation-SSE plateau counter (rel_tol slack).
    if(use_valid){
      prev_min <- suppressWarnings(min(SSE_valid[-length(SSE_valid)],
                                       na.rm = TRUE))
      threshold<- prev_min * (1 - rel_tol)
      improved <- is.finite(sse_va_cand) && sse_va_cand < threshold
    } else {
      prev_min <- suppressWarnings(min(SSE_train[-length(SSE_train)],
                                       na.rm = TRUE))
      threshold<- prev_min * (1 - rel_tol)
      improved <- is.finite(sse_tr_cand) && sse_tr_cand < threshold
    }
    plateau_va <- if(improved) 0L else plateau_va + 1L

    sse_show   <- if(use_valid) sse_va_cand else sse_tr_cand
    print_add  <- ifelse(i < 10, "  ", " ")
    comment    <- if(agg_satisfied) ""
                  else " agg constraint not yet satisfied"
    message(paste0(formatC(sse_show, digits = 7, format = "g"),
                 " (Scale", print_add, i, ")", comment))
    SSE        <- c(SSE, sse_show)
    SSE_name   <- c(SSE_name, paste0("scale ", i))

    if(agg_satisfied && plateau_va >= stop_k) break
  }

  ## Selected finest scale.
  K <- length(bands)
  if(K > 0){
    message("")
    message(paste0("-> Selected finest scale: ", K,
                 " (bandwidth: ",
                 formatC(bands[K], digits = 7, format = "g"), ")"))
    message("")
  } else {
    message("Warning: No residual spatial process was detected.")
  }

  ## Final SSE: use all committed scales (opt_id = K under the
  ## agg-constrained sequential-gamma_k formulation).
  sse_hv <- if(K > 0){
              if(use_valid) sum(W_glob[-id_train] * Resid[-id_train]^2)
              else          sum(W_glob[id_train]  * Resid[id_train]^2)
            } else {
              if(use_valid) sse_va0 else sse_tr0
            }

  ## Match the cf_lm_hv data.frame format.
  sse_hv_all   <- data.frame(learning = SSE_name, sse_hv = SSE)

  ## Final point-level pred for downstream cf_downscale use (clipped + Y_type
  ## conversion are deferred to cf_downscale).
  pred              <- as.matrix(x) %*% beta + pred_sp
  pred[pred < 0]    <- 0
  if(identical(Y_type, "sum")){
    pred            <- a * pred
  }

  ## Areal-level predictions from the training-only fit. On the validation
  ## areas these are genuine out-of-sample predictions (beta and gamma_k are
  ## estimated from training areas only), so cf_downscale uses them to report
  ## validation R2/RMSE. They are NOT subject to the pycnophylactic
  ## adjustment, which would otherwise force exact agreement with Y on every
  ## area and make the validation metrics trivially perfect.
  Pred_areal        <- as.numeric(Xmat %*% beta + Pred_sp_areal)

  other         <- list(bands=bands, bands_all=Bands, alpha=alpha,
                        x=x, X=X, xname=xname, kernel=kernel,
                        coords_uni=coords_uni, Coords_uni=Coords_uni, bid=bid,
                        pred=pred, Pred_areal=Pred_areal, a=a, Y_type=Y_type,
                        rel_tol=rel_tol,
                        agg_tol=agg_tol, agg_q=agg_q,
                        SSE_train=SSE_train, SSE_valid=SSE_valid,
                        AGG_err_tr=AGG_err_tr, gamma_list=gamma_list)
  result        <- list(sse_hv=sse_hv, sse_hv_all=sse_hv_all,
                        id_train=id_train, other=other, call=match.call())
  class(result) <- "cf_downscale_hv"
  return(result)
}

#' @noRd
#' @export
print.cf_downscale_hv <- function(x, ...){
  cat("Call:\n")
  print(x$call)
  cat("\n----Sum-of-squares errors for validation samples-----\n")
  print(x$sse_hv_all)
  invisible(x)
}
