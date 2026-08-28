#' Coarse-to-fine spatial modeling (CFSM) for Gaussian response
#'
#' Scalable prediction, regression, and multiscale analysis via Gaussian CFSM.
#'
#' @param y Vector of response variables (N x 1).
#' @param x Matrix of covariates (N x K).
#' @param coords Matrix of 2-dimensional point coordinates (N x 2).
#' @param x0 Optional. Matrix of covariates at prediction sites (N0 x K).
#' @param coords0 Optional. Matrix of 2-dimensional point coordinates at
#'   prediction sites (N0 x 2).
#' @param mod_hv Output object of the \code{\link{cf_lm_hv}} function.
#' @param se_type Type of predictive uncertainty in \code{pred}/\code{pred_q}.
#'   \code{"prediction"} (default) returns the holdout-calibrated OBSERVATION
#'   predictive (mean uncertainty + residual variance, split-conformal SD
#'   scaling on the \code{cf_lm_hv} validation samples); the signal versions are
#'   kept in \code{pred_signal}/\code{pred_q_signal}. \code{"mean"} returns the
#'   signal (mean) uncertainty only (previous behaviour).
#' @param robust_se If \code{TRUE} (default), coefficient standard errors
#'   and predictive uncertainty are computed using a cluster-robust sandwich
#'   estimator accounting for local spatial correlation.
#'   Set \code{FALSE} to use naive SEs (not recommended).
#' @param se_method Cluster-robust coefficient-SE estimator (used when
#'   \code{robust_se = TRUE}). \code{"opt"} (default) splits the sandwich
#'   meat into a field-removed observation-noise part and a field part that adds
#'   the calibrated field variance back with a within-block \code{exp(-d/h)}
#'   correlation (\code{h} = median committed bandwidth); this is near-nominal. A refit-free leverage leave-one-out ceiling then caps the field term, preventing over-coverage for count (Poisson) responses while leaving already-calibrated families unchanged.
#'   \code{"classic"} keeps the realised field inside the working residual (the
#'   previous behaviour), which is valid but conservative.
#'
#' @return A list with the following elements:
#' \describe{
#'   \item{beta}{Regression coefficients, their standard errors, and the lower
#'   and upper limits of the 95 percent confidence intervals.}
#'   \item{sd_summary}{Standard deviation of the regression term (xb), spatial
#'   processes (spatial_scale1, spatial_scale2,...),
#'   additional learned components (effective if `cf_lm_hv/add_learn` is not
#'   `none`), and residuals.}
#'   \item{e_summary}{Holdout validation accuracy evaluated on the validation
#'   samples: R-squared (validation_R2), root mean squared error
#'   (validation_RMSE), and mean absolute error (validation_MAE).}
#'   \item{pred}{Predictive means and standard deviations (sample sites). When
#'   no additional learner is active, the spatial-process contribution to the
#'   predictive SD is rescaled by a holdout-calibrated factor (stored as
#'   \code{other$tau}) estimated on the validation samples.}
#'   \item{pred0}{Predictive means and standard deviations (prediction sites).}
#'   \item{pred_q}{Predictive quantiles at the sample sites (data.frame with
#'   columns \code{q0.005}, \code{q0.025}, ..., \code{q0.975}, \code{q0.995}).
#'   With \code{add_learn = "rf"}/\code{"lightgbm"} active, the combined
#'   predictive distribution is calibrated by total conformalized quantile
#'   regression (CQR) on the validation samples; otherwise the quantiles are
#'   Gaussian about the predictive mean using the (tau-calibrated)
#'   \code{pred_sd}. \code{pred_sd} is a Gaussian-equivalent summary of these
#'   quantiles.}
#'   \item{pred0_q}{Predictive quantiles at the prediction sites; identical
#'   column structure to \code{pred_q}. \code{NULL} when prediction sites are
#'   not supplied.}
#'   \item{bands}{Bandwidth values for each scale. The i-th bandwidth
#'   corresponding to the i-th column of the Z matrix.}
#'   \item{Z}{Predictive means of the single-scale processes at each scale,
#'   corresponding to each bandwidth value (sample sites; list).}
#'   \item{Z_sd}{Predictive standard deviation of the spatial processes
#'   at each scale (sample sites; list).}
#'   \item{Z0}{Predictive mean of the spatial process at each scale
#'   (prediction sites; list).}
#'   \item{Z0_sd}{Predictive standard deviation of the spatial process
#'   at each bandwidth (prediction sites; list).}
#'   \item{other}{Other internally used output objects.}
#' }
#'
#' @references
#' Murakami, D., Comber, A., Yoshida, T., Tsutsumida, N., Brunsdon, C.,
#' & Nakaya, T. (2026). Coarse-to-fine spatial modeling:
#' A scalable, machine-learning-compatible framework.
#' *Geographical Analysis*, 58(2), e70034.
#' https://onlinelibrary.wiley.com/doi/10.1111/gean.70034
#'
#' @seealso \code{\link{cf_glm}}, \code{\link{cf_lm_hv}}, \code{\link{sp_scalewise}}
#'
#' @examples
#' set.seed(123)
#' require(sp); require(sf)
#' data(meuse)
#' data(meuse.grid)
#'
#' ### Data
#' y        <- log(meuse[,"zinc"])
#' coords   <- meuse[,c("x","y")]
#' x        <- data.frame(dist   = meuse[,"dist"],
#'                        ffreq2 = as.integer(meuse$ffreq == 2),
#'                        ffreq3 = as.integer(meuse$ffreq == 3))
#'
#' ### Data at prediction sites
#' coords0  <- meuse.grid[,c("x","y")]
#' x0       <- data.frame(dist   = meuse.grid[,"dist"],
#'                        ffreq2 = as.integer(meuse.grid$ffreq == 2),
#'                        ffreq3 = as.integer(meuse.grid$ffreq == 3))
#'
#' ### Holdout validation optimizing the number of spatial scales
#' mod_hv   <- cf_lm_hv(y = y, x = x, coords = coords, add_learn = "none")
#'
#' ### Spatial modeling and prediction
#' mod      <- cf_lm(y = y, x = x, x0 = x0, coords = coords, coords0 = coords0,
#'                  mod_hv = mod_hv)
#' mod
#'
#' ### Mapping predictive mean and standard deviations (SD)
#' meuse.grid$pred   <- mod$pred0$pred
#' meuse.grid$pred_sd<- mod$pred0$pred_sd
#' meuse.grid_sf     <- st_as_sf(meuse.grid, coords = c("x","y"))
#' plot(meuse.grid_sf[,"pred"], pch = 15, cex = 0.5, nbreaks = 20)   # Predictive mean
#' plot(meuse.grid_sf[,"pred_sd"], pch = 15, cex = 0.5, nbreaks = 20)# Predictive SD
#'
#' ### Multiscale spatial pattern/feature extraction
#' mod_s1<- sp_scalewise(mod,bw_range=c(1000,Inf)) # Large scale (1000 <= bandwidth)
#' mod_s2<- sp_scalewise(mod,bw_range=c(500,1000)) # Middle scale (500 <= bandwidth <= 1000)
#' mod_s3<- sp_scalewise(mod,bw_range=c(0,500))    # Small scale (bandwidth <= 500)
#' z1    <- mod_s1$pred0$pred                      # Predictive mean
#' z2    <- mod_s2$pred0$pred
#' z3    <- mod_s3$pred0$pred
#' z1_sd <- mod_s1$pred0$pred_sd                   # Predictive SD
#' z2_sd <- mod_s2$pred0$pred_sd
#' z3_sd <- mod_s3$pred0$pred_sd
#' meuse.grid_sf3  <- cbind(meuse.grid_sf, z1, z2, z3, z1_sd, z2_sd, z3_sd)
#' plot(meuse.grid_sf3[,c("z1","z2","z3")], pch = 15,
#'      cex = 0.5, nbreaks = 20,key.pos=4,axes=TRUE) # Predictive means
#' plot(meuse.grid_sf3[,c("z1_sd","z2_sd","z3_sd")], pch = 15,
#'      cex = 0.5, nbreaks = 20,key.pos=4,axes=TRUE) # Predictive SD
#'
#' ### The same fit, explored interactively over a basemap
#' # spCFmap(mod, crs = 28992)   # crs = the system the coordinates are in
#'
#' @author Daisuke Murakami
#'
#' @importFrom dbscan frNN
#' @importFrom fields rdist
#' @importFrom FNN get.knnx
#' @importFrom nloptr nloptr
#' @importFrom utils capture.output
#' @importFrom stats approx kmeans predict quantile rnorm runif sd var cor glm as.formula vcov qnorm residuals coefficients gaussian
#'
#' @export
cf_lm        <- function(y, x=NULL, coords, x0=NULL, coords0=NULL, mod_hv,
                         robust_se=TRUE, se_type=c("prediction","mean"),
                         se_method=c("opt","classic")){
  se_type      <- match.arg(se_type)
  se_method    <- match.arg(se_method)

  .spcf_check_mod_hv(mod_hv, "cf_lm_hv", "cf_lm_hv")
  .spcf_check_data(y = y, x = x, coords = coords)
  .spcf_check_newdata(x = x, x0 = x0, coords0 = coords0)

  if(!is.null(coords0) && !is.null(x) && is.null(x0)){
    .spcf_stop("'x0' must be provided when 'x' is specified: the prediction sites need the same covariates.")
  }

  bands          <- mod_hv$other$bands
  bands_all      <- mod_hv$other$bands_all
  coords_uni     <- mod_hv$other$coords_uni
  vpar           <- mod_hv$other$vpar
  sel_id_list    <- mod_hv$other$sel_id_list
  alpha          <- mod_hv$other$alpha
  ridge          <- mod_hv$other$ridge
  vc             <- mod_hv$other$vc
  x_sel          <- mod_hv$other$x_sel
  VCmat          <- mod_hv$other$VCmat
  kernel         <- mod_hv$other$kernel
  a_par          <- mod_hv$other$a_mod0$a_par
  a_run          <- mod_hv$other$a_mod0$a_run
  add_learn      <- mod_hv$other$a_mod0$add_learn

  init           <- initial_fun(x=x,y=y,coords=coords,x_sel=x_sel,train_rat=1)
  xx_inv         <- init$xx_inv
  beta_int       <- init$beta_int
  beta           <- init$beta
  coords         <- init$coords
  pred           <- init$pred
  resid          <- init$resid
  x              <- init$x
  x_sel          <- init$x_sel
  xname          <- init$xname
  n              <- init$n
  nx             <- init$nx
  id_train       <- init$id_train

  if(!is.null(coords0)){
    n0           <- nrow(coords0)
    one0         <- matrix(1,nrow=n0,ncol=1)
    if(is.null(x_sel) || sum(x_sel)==0){
      x0         <- one0
    } else {
      x0         <- cbind(one0, as.matrix(x0)[,x_sel])
    }
    pred0        <- x0 %*% beta_int
    Z0 <- Z0_sd  <- matrix(0,nrow=n0,ncol=length(bands))
    Z0_pv        <- matrix(0,nrow=n0,ncol=length(bands))   # eq.(10) predictive var (diag)

  } else {
    n0   <- x0   <- NA
    pred0 <- Z0  <- Z0_sd <- Z0_pv <- NULL
  }

  ##################### main loop for feature extraction
  message("--- Learning multi-scale spatial process ---")

  bands_scale    <- which(mod_hv$other$VCmat[,1]==1)

  b_old          <- NULL
  Z    <- Z_sd   <- matrix(0,nrow=n ,ncol=length(bands))
  Z_pv           <- matrix(0,nrow=n ,ncol=length(bands))   # eq.(10) predictive var (diag)
  if(!is.null(bands)){
    for(i in 1:max(bands_scale)){
      vc           <- which(VCmat[i,]==1)
      lmod         <- lwr(coords=coords, coords_uni=coords_uni, resid=resid, x=x,
                          band=bands_all[i],b_old=b_old, vc=vc, id_train=id_train,
                          ridge=ridge,kernel=kernel,x0=x0, coords0=coords0,
                          sel_id=sel_id_list[[i]], func="cf_lm")
      b_old        <- lmod$b_old
      if(length(vc)>0){
        beta_add     <- lmod$beta
        beta_v_add   <- lmod$beta_v
        beta_v_add[is.infinite(beta_v_add)]<-0
        pred_add     <- lmod$pred
        pred         <- pred + pred_add
        resid        <- y - pred
        beta_int_add <- xx_inv %*% t(x)%*%resid
        pred_int_add <- x%*%beta_int_add
        pred         <- pred  + pred_int_add
        resid        <- resid - pred_int_add

        ii           <- which(bands_scale==i)
        beta_add_m   <- colMeans(beta_add)
        Z[,ii]        <- beta_add[,1]-beta_add_m[1]#sweep(beta_add, 2, beta_add_m, "-")
        Z_sd[,ii]     <- sqrt(beta_v_add[,1])
        bpv           <- lmod$beta_pv[,1]; bpv[!is.finite(bpv)] <- 0
        Z_pv[,ii]     <- sqrt(bpv)
        beta_int     <- beta_int + beta_int_add + beta_add_m
        if(!is.null(coords0)){
          beta0_add     <- lmod$beta0
          beta0_v_add   <- lmod$beta0_v
          beta0_v_add[is.infinite(beta0_v_add)]<-0#tentative
          pred0_add     <- lmod$pred0
          pred0         <- pred0 + pred0_add
          pred0_int_add <- x0 %*% beta_int_add
          pred0         <- pred0 + pred0_int_add

          Z0[,ii]       <- beta0_add[,1]-beta_add_m[1]#sweep(beta0_add, 2, beta_add_m, "-")
          Z0_sd[,ii]    <- sqrt(beta0_v_add[,1])
          b0pv          <- lmod$beta0_pv[,1]; b0pv[!is.finite(b0pv)] <- 0
          Z0_pv[,ii]    <- sqrt(b0pv)
        }
        comment         <- ""
      } else {
        comment         <- " no improvement (skipped)"
      }

      print_add   <- ifelse(i<10,"  "," ")
      message( paste0( " Scale",print_add,i,
                     " (bandwidth:",format(bands_all[i],digits=7),")", comment))
    }
  } else {
    message("Warning: No residual spatial process was modeled")
  }

  pred_pre       <- rowSums(x*beta)
  ######### coefficients
  sig_pre        <- sum( (y - pred_pre)^2)/(n-nx)
  v_diag         <- rowSums(Z_sd^2) + sig_pre
  beta_int_vmat  <- solve(crossprod(x, 1/v_diag * x))
  beta_int_se    <- sqrt(diag(beta_int_vmat))
  beta_int_summ  <- data.frame(coef=beta_int,coef_se=beta_int_se,
                               lower_95CI=beta_int-1.96*beta_int_se,
                               upper_95CI=beta_int+1.96*beta_int_se)

  ##################### tuning
  beta           <- matrix(beta_int[,1], nrow = n, ncol = nx, byrow = TRUE)
  if(!is.null(coords0)){
    beta0        <- matrix(beta_int[,1], nrow=n0,ncol=nx, byrow=TRUE)
  }

  n_bid          <- length(bands)
  if(n_bid>0){
    n_band_x       <- sum(VCmat[,1]==1)#apply(VCmat,2,function(x) sum(x==1))
    vpar_coef      <- bopt_core(vpar[2], bands=bands, Z=Z,
                                beta_int=beta_int, nx=nx,#, is_vc=ifelse(n_band_x>0,1,0)
                                x=x, y=y, n_bid=n_bid,id_train=NULL)$vpar[1]
    w_0        <- exp(-vpar[2]/bands)
    w          <- vpar_coef*w_0/w_0[1]#vpar[j]
    b          <- Z %*% w#Reduce("+", lapply(1:n_band_x, function(i) w[i]*BBB[,i]))
    beta[,1]   <- beta[,1] + b
    if(!is.null(coords0)){
      b0       <- Z0 %*% w#Reduce("+", lapply(1:n_band_x, function(i) w[i]*BBB0[,i]))
      beta0[,1]<- beta0[,1] + b0
    }
  }

  ## spatial-block cluster-robust coefficient covariance (default): the model SE
  ## above (diagonal field-variance GLS) ignores the spatial CORRELATION of the
  ## field, so it understates Var(beta). .spcf_clusterSE puts the field back into
  ## the residual and clusters over spatial blocks. Updates the reported SEs and
  ## the coefficient-uncertainty term of the predictive SE (beta_int_vmat).
  if(robust_se && n_bid>0 && exists("b")){
    cse <- tryCatch(.spcf_clusterSE(y=y, X=x, beta=beta_int, field=b,
                                    offset=NULL, family=gaussian(),
                                    coords=coords, bands=bands),
                    error=function(e) NULL)
    if(!is.null(cse)){
      beta_int_vmat <- cse$V
      beta_int_se   <- sqrt(diag(cse$V))
      beta_int_summ <- data.frame(coef=beta_int, coef_se=beta_int_se,
                                  lower_95CI=beta_int-1.96*beta_int_se,
                                  upper_95CI=beta_int+1.96*beta_int_se)
    }
  }

  ######### additional learning
  a_mod          <- list()
  a_mod$add_learn<- "none"
  a_pred <- a_pred0 <- 0
  if(a_run){
    a_mod        <- add_mod(add_learn=add_learn, train=FALSE, resid=resid,
                            x=x, coords=coords, x0=x0, coords0=coords0,
                            id_train=mod_hv$id_train, sse_hv=NULL, a_par=a_par,
                            nx=nx, xname=xname)
    a_pred       <- a_mod$pred
    a_pred0      <- a_mod$pred0
  }

  ######### prediction
  pred           <- rowSums(x*beta) + a_pred
  coef_var       <- rowSums((x %*% beta_int_vmat) * x)
  ## Spatial-process predictive variance from eq.(10) (pv-based: grows away from
  ## data, unlike the coefficient variance Z_sd). It is (i) level-calibrated by a
  ## holdout factor tau, and (ii) capped at the marginal field variance (sill)
  ## so it saturates rather than diverging far from data (kriging-like ceiling).
  field_var      <- rowSums(Z_pv^2)
  ## Ceiling = marginal variance of the TOTAL field. A per-scale cap (sum of
  ## var(Z[,k])) was tried but under-covers: the scales are positively
  ## correlated, so sum_k var(Z[,k]) << var(sum_k Z[,k]) and the ceiling becomes
  ## far too low. The total-field marginal variance is the correct ceiling.
  sill           <- as.numeric(var(rowSums(Z)))          # marginal field variance ceiling
  if(!is.finite(sill) || sill <= 0) sill <- Inf
  qlev_out       <- c(0.005, 0.025, 0.05, seq(0.1, 0.9, 0.1), 0.95, 0.975, 0.995)
  qn_hi          <- qnorm(qlev_out[length(qlev_out)])
  if(!is.null(coords0)){
    pred0        <- rowSums(x0*beta0) + a_pred0
    coef_var0    <- rowSums((x0 %*% beta_int_vmat) * x0)
    field_var0   <- rowSums(Z0_pv^2)
  }

  ## ---- Holdout tau calibration of the spatial-process variance level. A
  ## single scalar rescales the field variance so it matches the noise-removed
  ## holdout squared error on the validation samples (moment estimator, shrunk
  ## toward tau = 1 when the holdout signal is weak).
  tau            <- 1
  idt            <- mod_hv$id_train
  if(!is.null(idt) && length(idt) < n && !is.null(mod_hv$other$pred_hv)){
    val          <- setdiff(seq_len(n), idt)
    ph           <- mod_hv$other$pred_hv
    sig2         <- mean((y[idt] - pred[idt])^2)            # in-sample noise floor
    e2           <- (y[val] - ph[val])^2                    # holdout squared error
    fv           <- field_var[val]
    okv          <- is.finite(e2) & is.finite(fv) & fv > 0
    if(sum(okv) >= 2){
      verr       <- mean(e2[okv]); vfld <- mean(fv[okv])
      num        <- verr - sig2
      se         <- sqrt(2 / sum(okv)) * verr
      rel        <- if(num > 0 && is.finite(se) && se > 0) num^2 / (num^2 + se^2) else 0
      tau_raw    <- if(vfld > 0) max(num, 1e-6) / vfld else 1
      tau        <- min(max(exp(log(tau_raw) * rel), 1e-2), 1e2)
      if(!is.finite(tau)) tau <- 1
    }
  }
  ## calibrated, sill-capped spatial-process predictive variance (total ceiling)
  fv_cal         <- pmin(tau * field_var, sill)
  fv0_cal        <- if(!is.null(coords0)) pmin(tau * field_var0, sill) else NULL

  ## Reported per-scale Z_sd / Z0_sd (used by sp_scalewise) are the pv per-scale
  ## variances scaled proportionally so rowSums(Z_sd^2) == fv_cal, sharing the
  ## same pv/tau/sill footing as pred_sd. The bv-based Z_sd used earlier for the
  ## GLS coefficient covariance is untouched.
  sf_pt          <- sqrt(ifelse(field_var > 0, fv_cal / field_var, 1))
  Z_sd           <- Z_pv * sf_pt
  if(!is.null(coords0)){
    sf0_pt       <- sqrt(ifelse(field_var0 > 0, fv0_cal / field_var0, 1))
    Z0_sd        <- Z0_pv * sf0_pt
  }

  ## opt+field coefficient covariance (default se_method): recomputed here, once
  ## the calibrated per-point field SD s_f = sqrt(fv_cal) is available, replacing
  ## the classic field-retained cluster-robust covariance. Updates the reported
  ## SEs and the coefficient-uncertainty term coef_var of the predictive SE.
  if(robust_se && se_method=="opt" && n_bid>0 && exists("b")){
    ofse <- tryCatch(.spcf_optfield_SE(y=y, X=x, beta=beta_int, field=b,
                                       s_f=sqrt(fv_cal), offset=NULL,
                                       family=gaussian(), coords=coords, bands=bands),
                     error=function(e) NULL)
    if(!is.null(ofse) && all(is.finite(diag(ofse$V))) && all(diag(ofse$V) > 0)){
      beta_int_vmat <- ofse$V
      beta_int_se   <- sqrt(diag(ofse$V))
      beta_int_summ <- data.frame(coef=beta_int, coef_se=beta_int_se,
                                  lower_95CI=beta_int-1.96*beta_int_se,
                                  upper_95CI=beta_int+1.96*beta_int_se)
      coef_var      <- rowSums((x %*% beta_int_vmat) * x)
      if(!is.null(coords0)) coef_var0 <- rowSums((x0 %*% beta_int_vmat) * x0)
    }
  }

  pred_q <- pred0_q <- NULL
  if(a_run){
    ## ---- Total CQR (rf/lightgbm): calibrate the combined (core +
    ## additional-learning) predictive distribution. Total quantiles are
    ## simulated from the Gaussian core (with the calibrated, capped field
    ## variance) and the raw additional-learning residual draws; conformity
    ## scores use the holdout point (pred_hv) with the full-data quantile shape
    ## re-centered on pred_hv. pred_sd is a Gaussian-equivalent summary of pred_q.
    Qtot         <- total_qmat(pred, sqrt(coef_var + fv_cal),
                               a_mod$qmat, a_mod$qlevels, qlev_out)
    off          <- NULL
    if(!is.null(idt) && length(idt) < n && !is.null(mod_hv$other$pred_hv)){
      val        <- setdiff(seq_len(n), idt)
      Qcal       <- sweep(Qtot[val, , drop=FALSE], 1, pred[val], "-")
      Qcal       <- sweep(Qcal, 1, mod_hv$other$pred_hv[val], "+")
      off        <- cqr_offsets(Qcal, y[val], qlev_out)
      Qtot       <- apply_cqr(Qtot, qlev_out, off)
    }
    pred_q       <- as.data.frame(Qtot); names(pred_q) <- paste0("q", qlev_out)
    pred_sd      <- (Qtot[, length(qlev_out)] - Qtot[, 1]) / (2 * qn_hi)
    if(!is.null(coords0)){
      Qtot0      <- total_qmat(pred0, sqrt(coef_var0 + fv0_cal),
                               a_mod$qmat0, a_mod$qlevels, qlev_out)
      if(!is.null(off)) Qtot0 <- apply_cqr(Qtot0, qlev_out, off)
      pred0_q    <- as.data.frame(Qtot0); names(pred0_q) <- paste0("q", qlev_out)
      pred0_sd   <- (Qtot0[, length(qlev_out)] - Qtot0[, 1]) / (2 * qn_hi)
    }

  } else {
    pred_sd      <- sqrt(coef_var + fv_cal)
    pred_q       <- as.data.frame(pred + outer(pred_sd, qnorm(qlev_out)))
    names(pred_q)<- paste0("q", qlev_out)
    if(!is.null(coords0)){
      pred0_sd   <- sqrt(coef_var0 + fv0_cal)
      pred0_q    <- as.data.frame(pred0 + outer(pred0_sd, qnorm(qlev_out)))
      names(pred0_q) <- paste0("q", qlev_out)
    }
  }

  pred_ms        <- data.frame( pred, pred_sd )
  pred0_ms       <- NULL
  if(!is.null(coords0)){
    pred0_ms     <- data.frame( pred=pred0, pred_sd=pred0_sd )
  }

  ######### spatial process
  if(!is.null(bands)){
    Z            <- as.data.frame(Z)
    Z_sd         <- as.data.frame(Z_sd)
    names(Z)     <- names(Z_sd) <- paste0("scale",bands_scale)
    if(!is.null(coords0)){
      Z0         <- as.data.frame(Z0)
      Z0_sd      <- as.data.frame(Z0_sd)
      names(Z0) <-names(Z0_sd) <- paste0("scale",bands_scale)
    }
  }

  ######### standard deviations of model elements
  resid_sd       <- sd(y - pred)
  a_sd <- a_name <- NULL
  if(a_run){
    a_sd         <- sd(a_mod$pred)
    a_name       <- paste0("additional learning (",add_learn,")")
  }


  if(!is.null(bands)){
    elements       <- c("xb",paste0("spatial_scale",bands_scale),a_name,"residuals")
    standard_deviation<- c(sd(x %*% beta_int_summ$coef), apply(Z,2,sd), a_sd, resid_sd)
  } else {
    elements       <- c("xb",a_name,"residuals")
    standard_deviation<- c(sd(x %*% beta_int_summ$coef), a_sd, resid_sd)
  }
  sd_summary     <- data.frame(elements, standard_deviation)
  row.names(sd_summary)<-NULL

  ######### error statistics
  ## Evaluated on the holdout validation samples using cf_lm_hv's out-of-sample
  ## prediction (pred_hv), so validation_R2 is a genuine holdout metric
  ## consistent with the holdout SSE (sse_hv) used for validation_RMSE. All NA
  ## when no validation samples are available (e.g. train_rat = 1).
  ival           <- setdiff(seq_len(n), mod_hv$id_train)
  pred_hv        <- mod_hv$other$pred_hv
  r2 <- rmse <- mae <- NA_real_
  if(length(ival) >= 2 && !is.null(pred_hv)){
    r2           <- cor(y[ival], pred_hv[ival])^2
    rmse         <- sqrt(mod_hv$sse_hv/length(ival))
    mae          <- mean(abs(y[ival] - pred_hv[ival]))
  }
  e_summary      <- data.frame(stat=c("validation_R2", "validation_RMSE",
                                      "validation_MAE"),
                               value=c(r2, rmse, mae))

  ######### summary outputs
  other          <- list(n=n,n0=n0,nx=nx,y=y,x=x,x0=x0,VCmat=VCmat,
                         coords=coords,coords0=coords0,vpar=vpar,
                         vc=mod_hv$other$vc, xx_inv=xx_inv, a_mod=a_mod,
                         pred_pre=pred_pre, sse_hv=mod_hv$sse_hv, tau=tau,
                         Z_pv=Z_pv, Z0_pv=Z0_pv)
  result         <- list(beta=beta_int_summ, sd_summary=sd_summary,
                         e_summary=e_summary, pred=pred_ms,pred0=pred0_ms,
                         pred_q=pred_q, pred0_q=pred0_q, bands=bands,
                         Z=Z,Z_sd=Z_sd, Z0=Z0, Z0_sd=Z0_sd, other=other,
                         call = match.call() )
  if(identical(se_type,"prediction")){
    ob <- tryCatch(.spcf_obs_predict(family=stats::gaussian(), y=y, mod_hv=mod_hv,
                     pred_in=result$pred$pred, predq_in=result$pred_q,
                     pred_out=result$pred0$pred, predq_out=result$pred0_q),
                   error=function(e) NULL)
    result <- .spcf_apply_obs(result, ob)
  } else result$other$se_type <- "mean"
  class( result ) <- "cf_lm"
  return( result )
}

#' @noRd
#' @export
print.cf_lm <- function(x, ...)
  {
    ## print(), not message(format()): format() on a data.frame returns a
    ## data.frame, which message() flattens with as.character() -- printing each
    ## column deparsed as c("...", "...") instead of a table. A print method also
    ## belongs on stdout, where capture.output() and knitr can see it, rather
    ## than on stderr where suppressMessages() would silence it.
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
