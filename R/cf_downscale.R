#' Coarse-to-fine spatial downscaling (CF-DS)
#'
#' Scalable downscaling via CF-DS for predicting disaggregate-level responses
#' from aggregate-level response \code{Y}, while ensuring that predictions
#' aggregate exactly to the observed aggregate-level values.
#'
#' @param Y Vector of aggregate-level response variables (length \code{N}).
#' @param x Matrix of disaggregate-level covariates (\code{n x K}), assumed to
#'   match the \code{x} used in \code{\link{cf_downscale_hv}}.
#' @param prop_weight Vector of disaggregate-level proportional allocation
#'   weights (length \code{n}), assumed to match the \code{prop_weight} used in
#'   \code{\link{cf_downscale_hv}}. See \code{\link{cf_downscale_hv}} for the
#'   role and examples of choices.
#' @param coords Matrix of disaggregate-level coordinates (\code{n x 2}).
#' @param agg_id Area ID for each disaggregate-level unit (length \code{n}).
#' @param mod_hv Output object from \code{\link{cf_downscale_hv}}.
#' @param adj Logical (default \code{TRUE}). When \code{TRUE}, a per-area
#'   multiplicative adjustment is applied to satisfy the aggregation constraint
#'   so that the downscaled predictions aggregate exactly to the observed `Y`.
#'   When \code{FALSE}, the constraint is satisfied only approximately,
#'   which may be preferable when `Y` contains noise.
#' @param nonneg If \code{TRUE} (default), clip negative predictions to zero
#'   before the multiplicative adjustment.
#'
#' @return A list with the following elements:
#' \describe{
#'   \item{beta}{Regression coefficients, their standard errors, and the
#'     lower and upper limits of the 95 percent confidence intervals.}
#'   \item{sd_summary}{Standard deviation of the regression term (xb),
#'     spatial processes (spatial_scale1, spatial_scale2,...), and
#'     residuals.}
#'   \item{e_summary}{Aggregate-level holdout validation accuracy, evaluated on
#'     the validation units: R-squared (validation_R2), root mean squared error
#'     (validation_RMSE), and mean absolute error (validation_MAE).
#'     All are \code{NA} when no validation areas are available
#'     (e.g. \code{train_rat = 1}).}
#'   \item{pred}{Predictive mean (\code{pred}) and standard deviation
#'     (\code{pred_sd}) of the disaggregate-level response. The spatial-process
#'     contribution to \code{pred_sd} is rescaled by a holdout-calibrated
#'     factor (stored as \code{other$tau}) estimated on the validation areas.}
#'   \item{bands}{Bandwidth values for each accepted scale during the
#'     holdout validation in \code{\link{cf_downscale_hv}}.}
#'   \item{Z}{Predictive mean of each single-scale spatial process at the
#'     disaggregate-level (data.frame; one column per scale).}
#'   \item{Z_sd}{Predictive standard deviation of the single-scale process
#'     at the disaggregate-level units (data.frame).}
#'   \item{other}{Other internally used output objects.}
#' }
#'
#' @references
#' Murakami, D., Chun, Y., Yoshida, T., & Seya, H. (2026).
#' Scalable coarse-to-fine spatial downscaling. *ArXiv preprint*.
#'
#' @seealso \code{\link{cf_downscale_hv}}, \code{\link{cf_lm}}
#' @author Daisuke Murakami
#'
#' @examples
#' set.seed(123)
#' require(sf); require(CARBayesdata)
#' data(GGHB.IZ)
#' data(pollutionhealthdata)
#' d  <- pollutionhealthdata[pollutionhealthdata$year == 2010, ]
#' ar <- merge(GGHB.IZ, d, by = "IZ")
#'
#' ### Disaggregate-level data (271 units)
#' coords <- st_coordinates(suppressWarnings(st_centroid(ar)))
#' x      <- data.frame(pm10 = ar$pm10, jsa = ar$jsa, price = ar$price)
#' prop_weight <- as.numeric(ar$expected)
#'
#' ### Aggregate-level data (30 units).
#' agg_id <- as.integer(stats::kmeans(coords, centers = 30)$cluster)
#'
#' ### Two types of response variables are possible:
#' # Y_type = "sum"  : Y_I = sum(response variable for each aggregate unit)
#' # Y_type = "mean" : Y_I = mean(response variable for each aggregate unit)
#' Y_type <- "sum"   # change to "mean" for the density-type data
#' Y      <- as.numeric(stats::aggregate(ar$observed, by = list(agg_id),
#'                        FUN = if (Y_type == "sum") sum else mean)[, 2])
#'
#' ### Downscaling
#' mh <- cf_downscale_hv(Y = Y, Y_type = Y_type, x = x,
#'                       prop_weight = prop_weight,
#'                       coords = coords, agg_id = agg_id)
#' md <- cf_downscale(Y = Y, x = x, prop_weight = prop_weight,
#'                    coords = coords, agg_id = agg_id, mod_hv = mh)
#'
#' ### Mapping
#' ar$agg_id <- agg_id
#' agg_poly  <- stats::aggregate(ar["agg_id"], by = list(agg_id = agg_id),
#'                               FUN = function(z) z[1])
#' agg_poly$Y<- Y
#' ar$pred   <- md$pred$pred
#' plot(agg_poly["Y"], nbreaks = 20, main = "Aggregated data")
#' plot(ar["pred"], nbreaks = 20, main = "Downscaling result")
#'
#' @importFrom dbscan frNN
#' @importFrom FNN get.knnx
#' @importFrom stats aggregate coef coefficients kmeans lm sd cor var
#'
#' @export
cf_downscale <- function(Y, x=NULL, prop_weight=NULL, coords, agg_id, mod_hv,
                  adj=TRUE, nonneg=TRUE){

  .spcf_check_mod_hv(mod_hv, "cf_downscale_hv", "cf_downscale_hv")
  .spcf_check_downscale(Y, x, prop_weight, coords, agg_id)

  ## Internal code uses the paper notation `a` for the proportional
  ## allocation weight; bind the user-facing argument to it once here.
  a              <- prop_weight
  adj_method     <- if(isFALSE(adj)) "none" else "mult"

  bands          <- mod_hv$other$bands
  bands_all      <- mod_hv$other$bands_all
  alpha          <- mod_hv$other$alpha
  xname          <- mod_hv$other$xname
  kernel         <- mod_hv$other$kernel
  Y_type         <- mod_hv$other$Y_type
  id_train_hv    <- mod_hv$id_train

  N              <- length(Y)
  init           <- initial_ds_fun(Y=Y, Y_type=Y_type, x=x, a=a, coords=coords,
                                   train_rat=1, Id_train=NULL, agg_id=agg_id)
  beta_int       <- init$beta_int
  X              <- init$X
  Coords_uni     <- init$Coords_uni
  coords_uni     <- init$coords_uni
  W              <- init$W
  W_glob         <- init$W_glob
  x              <- init$x
  a              <- init$a
  Resid          <- init$Resid
  Agg_id         <- init$Agg_id
  Id_train       <- init$Id_train
  n              <- init$n
  nx             <- init$nx

  pred_sp        <- 0
  Bands          <- bands

  ## ---- Per-scale spatial increments (Z) and SD (Z_sd) ----
  Z    <- Z_sd   <- matrix(0, nrow=n, ncol=length(bands))

  message("--- Learning multi-scale spatial process ---")

  Xmat              <- as.matrix(X)
  Pred_sp_areal     <- numeric(N)
  b_old             <- NULL
  beta              <- as.numeric(beta_int)
  gamma_vec         <- numeric(length(bands))

  for(i in seq_along(Bands)){
    band         <- Bands[i]
    lmod         <- lwr_ds(coords=coords, coords_uni=coords_uni,
                           beta_int=beta_int,
                           Resid=Resid, Y=Y, X=X, W=W, x=x, a=a, band=band,
                           Coords_uni=Coords_uni,
                           b_old=b_old, ridge=FALSE, kernel=kernel,
                           sel_id=NULL, sse_hv0=NULL, pred_sp=pred_sp,
                           Id_train=Id_train, agg_id=agg_id, Agg_id=Agg_id,
                           func="cf_downscale", c_shrink=0,
                           knots_train_only=TRUE)
    b_old        <- lmod$b_old

    incr_raw     <- lmod$pred_sp - pred_sp
    incr_pt      <- incr_raw - mean(incr_raw)
    Pred_sp_add  <- as.numeric(aggregate(a*incr_pt, by=list(agg_id),
                                         sum)[,2])
    design       <- cbind(Xmat, Pred_sp_add)
    Gmod         <- lm((Y - Pred_sp_areal) ~ 0 + design, weights=W_glob)
    coef_full    <- as.numeric(coefficients(Gmod))
    beta         <- coef_full[seq_len(nx)]
    gamma_k      <- coef_full[nx + 1L]
    if(!is.finite(gamma_k)) gamma_k <- 0
    if(gamma_k > 1){
      gamma_k    <- 1
      G2         <- lm((Y - Pred_sp_areal) ~ 0 + Xmat +
                       offset(Pred_sp_add), weights = W_glob)
      beta       <- as.numeric(coefficients(G2))
    } else if(gamma_k < 0){
      gamma_k    <- 0
      G2         <- lm((Y - Pred_sp_areal) ~ 0 + Xmat, weights = W_glob)
      beta       <- as.numeric(coefficients(G2))
    }
    gamma_vec[i] <- gamma_k

    pred_sp      <- pred_sp + gamma_k * incr_pt
    Pred_sp_areal<- Pred_sp_areal + gamma_k * Pred_sp_add
    Resid        <- as.numeric(Y - Xmat %*% beta - Pred_sp_areal)

    ## Z: the committed spatial increment at this scale (gamma_k * incr_pt).
    ## Z_sd: predictive SD of the per-scale spatial process at each point.
    Z[, i]       <- gamma_k * incr_pt
    if(!is.null(lmod$beta_v)){
      bv         <- lmod$beta_v[, 1]
      bv[is.infinite(bv)] <- 0
      bv[bv < 0] <- 0
      Z_sd[, i]  <- abs(x[, 1]) * sqrt(bv) * gamma_k
    }

    print_add    <- ifelse(i < 10, "  ", " ")
    comment      <- ""
    message(paste0(" Scale", print_add, i,
                 " (bandwidth:", format(band, digits = 7), ")", comment))
  }

  ## ---- Raw point-level prediction ----
  pred           <- as.numeric(as.matrix(x) %*% beta + pred_sp)
  if(nonneg) pred[pred < 0] <- 0
  pred_naive     <- a * pred

  if(adj_method == "mult"){
    pred         <- multiplicative_pycnophylactic(pred=pred, Y=Y,
                                                  agg_id=agg_id, a=a)
  }

  ## Y_type conversion: for "sum", final pred is the local total (a*pred).
  if(identical(Y_type, "sum")){
    pred         <- a * pred
  }

  if(any(!is.finite(pred))){
    warning(sprintf("%d non-finite point predictions replaced with 0.",
                    sum(!is.finite(pred))))
    pred[!is.finite(pred)] <- 0
  }

  ## ---- Coefficient summary (W_glob-weighted GLS covariance) ----
  ## sigma2 = weighted RSS / (N - nx); Var(beta) = sigma2 * (X^T W_glob X)^{-1}
  XtWX           <- t(Xmat) %*% (W_glob * Xmat)
  XtWX_inv       <- tryCatch(solve(XtWX),
                             error = function(e) matrix(NA_real_, nx, nx))
  resid_areal    <- as.numeric(Y - Xmat %*% beta - Pred_sp_areal)
  sigma2_hat     <- sum(W_glob * resid_areal^2) / max(N - nx, 1)
  beta_vmat      <- sigma2_hat * XtWX_inv
  beta_se        <- sqrt(pmax(diag(beta_vmat), 0))
  beta_summ      <- data.frame(coef        = beta,
                               coef_se     = beta_se,
                               lower_95CI  = beta - 1.96 * beta_se,
                               upper_95CI  = beta + 1.96 * beta_se,
                               row.names   = xname)

  ## ---- Predictive SD at the disaggregate-level units ----
  ## Combines regression-coefficient uncertainty and per-scale spatial SDs.
  pred_sd_reg    <- sqrt(pmax(rowSums((x %*% beta_vmat) * x), 0))
  pred_sd_sp     <- sqrt(rowSums(Z_sd^2))

  ## ---- Holdout variance calibration (tau) for the spatial-process component.
  ## A single multiplicative scalar rescales the spatial-process variance so
  ## that the model's areal predictive field variance matches the noise-removed
  ## holdout error on the validation areas (moment estimator, shrunk toward
  ## tau = 1 when the holdout signal is weak). The areal field variance is
  ## aggregated under independence; since tau is a uniform scale on the single
  ## field's SD, the same scalar is consistent at the disaggregate level. The
  ## GLS coefficient variance is left as-is (calibrated separately by sigma2).
  Pred_areal_hv  <- mod_hv$other$Pred_areal
  tau            <- 1
  if(!is.null(id_train_hv) && length(id_train_hv) < N &&
     !is.null(Pred_areal_hv)){
    val          <- setdiff(seq_len(N), id_train_hv)
    if(length(val) >= 2){
      Vsp_area   <- as.numeric(aggregate((a * pred_sd_sp)^2,
                                          by = list(agg_id), sum)[, 2])
      e2         <- (Y[val] - Pred_areal_hv[val])^2          # holdout squared error
      Wv         <- W_glob[val]
      verr       <- sum(Wv * e2) / sum(Wv)                   # weighted holdout MSE
      vfld       <- sum(Wv * Vsp_area[val]) / sum(Wv)        # model areal field var
      tr         <- id_train_hv
      sig2       <- sum(W_glob[tr] * resid_areal[tr]^2) / sum(W_glob[tr])  # noise floor
      num        <- verr - sig2                              # noise-removed field error
      se         <- sqrt(2 / length(val)) * verr             # approx SE of verr
      rel        <- if(num > 0 && is.finite(se) && se > 0) num^2 / (num^2 + se^2) else 0
      tau_raw    <- if(vfld > 0) max(num, 1e-6) / vfld else 1
      tau        <- min(max(exp(log(tau_raw) * rel), 1e-2), 1e2)
      if(!is.finite(tau)) tau <- 1
    }
  }

  pred_sd_density<- sqrt(pred_sd_reg^2 + tau * pred_sd_sp^2)
  pred_sd        <- if(identical(Y_type, "sum")) a * pred_sd_density
                    else pred_sd_density
  pred_ms        <- data.frame(pred = pred, pred_sd = pred_sd)

  ## ---- sd_summary (match cf_lm structure) ----
  bands_scale    <- seq_along(bands)
  if(length(bands) > 0){
    Z              <- as.data.frame(Z)
    Z_sd           <- as.data.frame(Z_sd)
    names(Z) <- names(Z_sd) <- paste0("scale", bands_scale)
    elements       <- c("xb", paste0("spatial_scale", bands_scale),
                        "residuals")
    standard_deviation <- c(sd(as.numeric(x %*% beta)),
                            apply(Z, 2, sd),
                            sd(resid_areal))
  } else {
    Z <- Z_sd <- NULL
    elements           <- c("xb", "residuals")
    standard_deviation <- c(sd(as.numeric(x %*% beta)),
                            sd(resid_areal))
  }
  sd_summary     <- data.frame(elements, standard_deviation)
  row.names(sd_summary) <- NULL

  ## ---- Error statistics (areal holdout validation) ----
  ## Use the areal predictions from cf_downscale_hv (mod_hv), which are fitted
  ## on the training areas only and are therefore genuinely out-of-sample on
  ## the validation areas. The adjusted cell-level `pred` here aggregates to Y
  ## exactly on EVERY area (pycnophylactic constraint), so aggregating it back
  ## would force a perfect validation fit and cannot be used for assessment.
  Pred_areal_hv  <- mod_hv$other$Pred_areal
  if(!is.null(id_train_hv) && length(id_train_hv) < N &&
     !is.null(Pred_areal_hv)){
    val_idx    <- setdiff(seq_len(N), id_train_hv)
    if(length(val_idx) >= 2 && sd(Y[val_idx]) > 0 &&
       sd(Pred_areal_hv[val_idx]) > 0){
      r2_val <- cor(Y[val_idx], Pred_areal_hv[val_idx])^2
    } else {
      r2_val <- NA_real_
    }
    rmse_val <- sqrt(mean((Y[val_idx] - Pred_areal_hv[val_idx])^2))
    mae_val  <- mean(abs(Y[val_idx] - Pred_areal_hv[val_idx]))
  } else {
    r2_val   <- NA_real_
    rmse_val <- NA_real_
    mae_val  <- NA_real_
  }
  e_summary    <- data.frame(stat  = c("validation_R2", "validation_RMSE",
                                       "validation_MAE"),
                             value = c(r2_val, rmse_val, mae_val))

  ## ---- Output ----
  ## Aggregated final (adjusted) prediction; equals Y on every area when
  ## adj = TRUE. Kept for inspecting the pycnophylactic constraint.
  if(identical(Y_type, "sum")){
    Pred_agg   <- as.numeric(aggregate(pred, by=list(agg_id), sum)[, 2])
  } else {
    Pred_agg   <- as.numeric(aggregate(a*pred, by=list(agg_id), sum)[, 2])
  }
  other        <- list(Y=Y, x=x, a=a, agg_id=agg_id, X=X,
                       coords=coords,
                       beta_vmat=beta_vmat, sigma2_hat=sigma2_hat,
                       pred_naive=pred_naive, pred_sp=pred_sp,
                       Pred_agg=Pred_agg, Pred_areal_hv=Pred_areal_hv,
                       tau=tau,
                       gamma_list=gamma_vec, Y_type=Y_type,
                       sse_hv=mod_hv$sse_hv)
  result       <- list(beta=beta_summ, sd_summary=sd_summary,
                       e_summary=e_summary, pred=pred_ms,
                       bands=bands, Z=Z, Z_sd=Z_sd,
                       other=other, call=match.call())
  class(result) <- "cf_downscale"
  return(result)
}

#' @noRd
#' @export
print.cf_downscale <- function(x, ...){
  cat("Call:\n")
  print(x$call)
  cat("\n----Coefficients---------------------------------------\n")
  print(x$beta)
  cat("\n----Standard deviations (influential elements only)----\n")
  print(x$sd_summary)
  cat("\n----Error statistics ----------------------------------\n")
  print(x$e_summary)
  invisible(x)
}
