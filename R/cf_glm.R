#' Coarse-to-fine spatial generalized linear mixed models (CF-GLMMs)
#'
#' Scalable prediction, regression, and multiscale analysis via CF-GLMMs.
#'
#' @param y Vector of response variables (N x 1), including continuous, count,
#'   and binary responses following an exponential family distribution.
#' @param x Matrix of covariates (N x K).
#' @param coords Matrix of 2-dimensional point coordinates (N x 2).
#' @param offset Optional. Vector of offset variables (N x 1) included
#'   in the linear predictor, consistent with \code{\link{glm}}.
#' @param x0 Optional. Matrix of covariates at prediction sites (N0 x K).
#' @param coords0 Optional. Matrix of 2-dimensional point coordinates at
#'   prediction sites (N0 x 2).
#' @param offset0 Optional. Vector of offset variables at prediction sites
#'  (N0 x 1)
#' @param mod_hv Output object of the \code{\link{cf_glm_hv}} function.
#' @param robust_se If \code{TRUE} (default), coefficient standard errors
#'   and predictive uncertainty are computed using a cluster-robust sandwich
#'   estimator accounting for local spatial correlation.
#'   Set \code{FALSE} to use naive SEs (not recommended).
#' @param se_type Type of predictive uncertainty in \code{pred}/\code{pred_q}.
#'   \code{"prediction"} (default) returns the OBSERVATION predictive for a new
#'   data point, holdout-calibrated on the \code{cf_glm_hv} validation samples
#'   (Gaussian: mean uncertainty + residual variance, split-conformal SD
#'   scaling; Poisson: negative-binomial count predictive; binomial: temperature
#'   -calibrated probability with \code{pred_sd = sqrt(p(1-p))}). The mean/signal
#'   versions are kept in \code{pred_signal}/\code{pred_q_signal}.
#'   \code{"mean"} returns the signal (mean) uncertainty only (previous
#'   behaviour). See \code{other$calibration} for the fitted calibration.
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
#'   process (spatial_scale1, spatial_scale2,...),
#'   additional learning, and residuals.}
#'   \item{e_summary}{Holdout validation accuracy evaluated on the validation
#'   samples: R-squared (validation_Pseudo-R2), root mean squared error
#'   (validation_RMSE), and mean absolute error (validation_MAE).}
#'   \item{pred}{Predictive means and standard deviations (sample sites). The
#'   spatial-process contribution to the predictive SD is rescaled by a
#'   holdout-calibrated factor (stored as \code{other$tau}) estimated on the
#'   validation samples.}
#'   \item{pred0}{Predictive means and standard deviations (prediction sites).}
#'   \item{pred_q}{Predictive quantiles on the response scale at the sample
#'   sites. A data frame whose columns \code{q0.005}, \code{q0.025},
#'   \code{q0.05}, \code{q0.1}, ..., \code{q0.9}, \code{q0.95}, \code{q0.975},
#'   \code{q0.995} give the corresponding quantile levels, obtained by
#'   Gaussian approximation on the link scale followed by inverse-link
#'   transformation.}
#'   \item{pred0_q}{Predictive quantiles on the response scale at the
#'   prediction sites. Column structure is identical to \code{pred_q}.
#'   \code{NULL} when prediction sites are not supplied.}
#'   \item{bands}{Bandwidth values for each scale. The i-th bandwidth
#'   corresponds to the i-th column of the Z matrix.}
#'   \item{Z}{Predictive mean of the spatial process at each scale
#'   (sample sites; list).}
#'   \item{Z_sd}{Predictive standard deviation of the spatial process at each
#'   scale (sample sites; list).}
#'   \item{Z0}{Predictive mean of the spatial process at each scale
#'   (prediction sites; list).}
#'   \item{Z0_sd}{Predictive standard deviation of the spatial process at each
#'   scale (prediction sites; list).}
#'   \item{other}{Other internally used output objects.}
#' }
#'
#' @references
#' Murakami, D., Comber, A., Yoshida, T., Tsutsumida, N., Brunsdon, C.,
#' & Nakaya, T. (2025).
#' Coarse-to-fine spatial GLMMs for scalable prediction and multiscale analysis.
#' *ArXiv preprint*, 2605.01157.
#' https://doi.org/10.48550/arXiv.2605.01157
#'
#' @seealso \code{\link{cf_glm_hv}}, \code{\link{sp_scalewise}}
#'
#' @examples
#' ################ Example 1: Count data modeling/Disease mapping/smoothing
#' set.seed(1234)
#' require( CARBayesdata )
#' require( sf )
#' data(pollutionhealthdata)
#' data(GGHB.IZ)
#'
#' ### Data
#' dat      <- pollutionhealthdata[pollutionhealthdata$year==2011,]
#' y        <- dat[,"observed"]             # count data
#' x        <- dat[,c("pm10","jsa","price")]
#' offset   <- log(dat[,"expected"])
#' coords   <- st_coordinates(st_centroid(GGHB.IZ))
#'
#' ### Holdout validation optimizing the number of spatial scales
#' mod_hv   <- cf_glm_hv(y = y, x = x, offset=offset, coords = coords, family=poisson())
#'
#' ### Spatial modeling and prediction
#' mod      <- cf_glm(y = y, x = x, coords = coords, mod_hv = mod_hv)
#' mod
#'
#' ### Mapping predictive mean and standard deviations (SD)
#' GGHB.IZ$y      <- y
#' GGHB.IZ$pred   <- mod$pred$pred
#' GGHB.IZ$pred_sd<- mod$pred$pred_sd
#' plot(GGHB.IZ[,c("pred")],lwd=0.2,axes=TRUE, key.pos=4,nbreaks=50)   # Predictive mean
#' plot(GGHB.IZ[,c("pred_sd")],lwd=0.2,axes=TRUE, key.pos=4,nbreaks=50)# Predictive SD
#'
#' ### Multiscale spatial pattern/feature extraction
#' mod_s1      <- sp_scalewise(mod,bw_range=c(4000,Inf)) # Large scale (4000 <= bandwidth)
#' mod_s2      <- sp_scalewise(mod,bw_range=c(0,4000))   # Small scale (bandwidth <= 4000)
#' GGHB.IZ$z1  <- mod_s1$pred$pred
#' GGHB.IZ$z2  <- mod_s2$pred$pred
#' plot(GGHB.IZ[,c("z1","z2")],lwd=0.2,axes=TRUE,key.pos=4, nbreaks=50)# Extracted features
#'
#' ### The same fit, explored interactively over a basemap
#' # spCFmap(mod, crs = 27700)   # crs = the system the coordinates are in
#'
#'
#' ################ Example 2: Binary data modeling/spatial prediction
#' set.seed(1234)
#' require(sp); require(sf)
#' data(meuse)
#' data(meuse.grid)
#'
#' ### Data
#' y        <- ifelse(meuse$ffreq==1, 1, 0 )# binary data
#' coords   <- meuse[,c("x","y")]
#' x        <- meuse[,"dist"]
#'
#' ### Data at prediction sites
#' coords0  <- meuse.grid[,c("x","y")]
#' x0       <- meuse.grid[,"dist"]
#'
#' ### Holdout validation optimizing the number of spatial scales
#' mod_hv   <- cf_glm_hv(y = y, x = x, coords = coords, family=binomial())
#'
#' ### Spatial modeling and prediction
#' mod      <- cf_glm(y = y, x=x, coords = coords, x0=x0, coords0 = coords0,
#'                    mod_hv = mod_hv)
#' mod
#'
#' ### Mapping predictive mean and standard deviations (SD)
#' meuse.grid$pred   <- mod$pred0$pred
#' meuse.grid$pred_sd<- mod$pred0$pred_sd
#' meuse.grid_sf     <- st_as_sf(meuse.grid, coords = c("x","y"))
#' plot(meuse.grid_sf[,"pred"], pch = 15, cex = 0.8, nbreaks = 20)   # Predictive mean
#' plot(meuse.grid_sf[,"pred_sd"], pch = 15, cex = 0.8, nbreaks = 20)# Predictive SD
#'
#' ### Multiscale spatial pattern/feature extraction
#' mod_s1<- sp_scalewise(mod,bw_range=c(1000,Inf)) # Large scale (1000 <= bandwidth)
#' mod_s2<- sp_scalewise(mod,bw_range=c(0,1000))   # Small scale (0 <= bandwidth <= 1000)
#' meuse.grid_sf$z1    <- mod_s1$pred0$pred
#' meuse.grid_sf$z2    <- mod_s2$pred0$pred
#' plot(meuse.grid_sf[,c("z1","z2")], pch = 15,
#'      cex = 0.5, nbreaks = 20,axes=TRUE) # Predictive means
#'
#' ### The same fit, explored interactively over a basemap
#' # spCFmap(mod, crs = 28992)   # crs = the system the coordinates are in
#'
#'
#' @author Daisuke Murakami
#'
#' @importFrom dbscan frNN
#' @importFrom fields rdist
#' @importFrom FNN get.knnx
#' @importFrom nloptr nloptr
#' @importFrom utils capture.output
#' @importFrom stats approx kmeans predict quantile rnorm runif sd var cor
#'
#' @export
cf_glm          <- function(y, x=NULL, coords, offset=NULL,
                            x0=NULL, coords0=NULL, offset0=NULL, mod_hv,
                            robust_se=TRUE, se_type=c("prediction","mean"),
                            se_method=c("opt","classic")){
  se_type        <- match.arg(se_type)
  se_method      <- match.arg(se_method)

  .spcf_check_mod_hv(mod_hv, "cf_glm_hv", "cf_glm_hv")
  .spcf_check_data(y = y, x = x, coords = coords, offset = offset)
  .spcf_check_newdata(x = x, x0 = x0, coords0 = coords0, offset0 = offset0)

  family         <- mod_hv$other$family
  bands          <- mod_hv$other$bands
  bands_all      <- mod_hv$other$bands_all
  coords_uni     <- mod_hv$other$coords_uni
  sel_id_list    <- mod_hv$other$sel_id_list
  alpha          <- mod_hv$other$alpha
  ridge          <- mod_hv$other$ridge
  vc             <- mod_hv$other$vc
  x_sel          <- mod_hv$other$x_sel
  VCmat          <- mod_hv$other$VCmat
  kernel         <- mod_hv$other$kernel
  #a_par          <- mod_hv$other$a_mod0$a_par
  #a_run          <- mod_hv$other$a_mod0$a_run
  #add_learn      <- mod_hv$other$a_mod0$add_learn

  if(!is.null(coords0)){
    if(!is.null(offset)&is.null(offset0)){
      .spcf_stop("'offset0' must be provided when 'offset' is specified: the prediction sites need their own offset.")
    }
    if(!is.null(x)&is.null(x0)){
      .spcf_stop("'x0' must be provided when 'x' is specified: the prediction sites need the same covariates.")
    }
  }

  init           <- initial_fun_glm(x=x,y=y,coords=coords,offset=offset,
                                    x_sel=x_sel,family=family,train_rat=1)
  gmod0          <- init$gmod
  beta_int       <- init$beta_int
  beta           <- init$beta
  coords         <- init$coords
  resid          <- init$resid
  x              <- init$x
  x_sel          <- init$x_sel
  xname          <- init$xname
  offset         <- init$offset
  w              <- init$gmod$weights
  n              <- init$n
  nx             <- init$nx
  id_train       <- init$id_train

  if( !is.null(coords0) ){
    n0           <- nrow(coords0)
    one0         <- matrix(1,nrow=n0,ncol=1)
    if(sum(x_sel)==0){
      x0         <- one0
    } else {
      x0         <- cbind(one0,as.matrix(x0)[,x_sel])
    }

    if( is.null(offset0) ) offset0<- rep(0,n0)
    beta0        <- matrix(beta_int,nrow=n0,ncol=nx,byrow=TRUE)
    l_pred0      <- 0
    Z0 <- Z0_sd  <- matrix(0,nrow=n0,ncol=length(bands))
    Z0_pv        <- matrix(0,nrow=n0,ncol=length(bands))   # eq.(10) predictive var (link)

  } else {
    n0  <- x0    <- NA
    l_pred0      <- Z0 <- Z0_sd  <- Z0_pv <- NULL
  }

  ##################### main loop for feature extraction
  message("--- Learning multi-scale spatial process ---")

  bands_scale    <- which(mod_hv$other$VCmat[,1]==1)

  b_old          <- NULL
  Z     <- Z_sd  <- matrix(0,nrow=n ,ncol=length(bands))
  Z_pv           <- matrix(0,nrow=n ,ncol=length(bands))   # eq.(10) predictive var (link)
  l_pred         <- 0
  if(!is.null(bands)){
    for(i in 1:max(bands_scale)){
      vc         <- which(VCmat[i,]==1)
      lmod       <- lwr_glm(coords=coords, coords_uni=coords_uni, resid=resid, y=y, x=x, w=w,
                            band=bands_all[i], b_old=b_old, vc=vc, id_train=id_train,
                            ridge=ridge,kernel=kernel, x0=x0, coords0=coords0,l_pred=l_pred,
                            sel_id=sel_id_list[[i]], family=family,func="cf_glm")

      b_old      <- lmod$b_old
      if(length(vc)>0){
        l_pred_add  <- lmod$pred
        l_pred      <- l_pred  + l_pred_add
        l_bias      <- mean(l_pred)
        l_pred      <- l_pred   - l_bias

        beta_add    <- lmod$beta
        beta_add[,1]<- beta_add[,1] - l_bias
        beta        <- beta    + beta_add

        beta_v_add  <- lmod$beta_v
        beta_v_add[is.infinite(beta_v_add)]<-0

        ii          <- which(bands_scale==i)
        Z[,ii]      <- beta_add[,1]
        Z_sd[,ii]   <- sqrt(beta_v_add[,1])
        bpv         <- lmod$beta_pv[,1]; bpv[!is.finite(bpv)] <- 0
        Z_pv[,ii]   <- sqrt(bpv)

        l_pred_off  <- .spcf_clip_l(l_pred, family) + offset
        gmod0       <- glm(y ~ 0 + x + offset(l_pred_off),family=family)
        resid       <- gmod0$residuals
        w           <- gmod0$weights
        beta_int_new<- matrix(gmod0$coefficients)
        for(jj in 1:nx){
          beta[,jj]     <- beta[,jj]-beta_int[jj,1]+beta_int_new[jj]
        }
        beta_int        <- beta_int_new
        if(!is.null(coords0)){
          l_pred0_add   <- lmod$pred0
          l_pred0       <- l_pred0 + l_pred0_add
          l_pred0       <- l_pred0 - l_bias

          beta0_add     <- lmod$beta0
          beta0_add[,1] <- beta0_add[,1]- l_bias
          beta0         <- beta0 + beta0_add

          beta0_v_add   <- lmod$beta0_v
          beta0_v_add[is.infinite(beta0_v_add)]<-0#tentative

          Z0[,ii]       <- beta0_add[,1]
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

  pred_pre       <- predict(gmod0,type="response")

  ######### coefficients
  beta_int_se    <- summary(gmod0)$coefficients[,2]
  beta_int_summ  <- data.frame(coef=beta_int,coef_se=beta_int_se,
                               lower_95CI=beta_int-1.96*beta_int_se,
                               upper_95CI=beta_int+1.96*beta_int_se)
  beta           <- matrix(beta_int[,1], nrow = n , ncol = nx, byrow = TRUE)
  if(!is.null(coords0)){
    beta0        <- matrix(beta_int[,1], nrow = n0, ncol = nx, byrow = TRUE)
  }

  n_band_x       <- sum(VCmat[,1]==1)
  n_bid          <- length(bands)
  b              <- rowSums( Z )
  beta[,1]       <- beta[,1] + b
  if(!is.null(coords0)){
    b0           <- rowSums( Z0 )
    beta0[,1]    <- beta0[,1] + b0
  }

  ######### prediction after adjustment
  gmod_dat        <- data.frame(y=y,l_pred=.spcf_clip_l(beta[,1], family),x[,-1],offset)
  names(gmod_dat) <- c("y","l_pred",xname[-1],"offset")
  if(length(xname[-1])==0){
    formula         <- as.formula("y ~ offset(l_pred+offset)")
  } else {
    formula         <- as.formula(paste0("y ~ offset(l_pred+offset)+", paste(xname[-1], collapse = "+")))
  }
  gmod            <- glm(formula=formula, data=gmod_dat,family=family)
  pred            <- predict(gmod,type="response")#gmod$fitted.values
  #pred_sd         <- sqrt( rowSums((x %*% beta_inv_vmat) * x) + rowSums(Z_sd^2))
  xbeta           <- predict(gmod,type="link")
  if(!is.null(coords0)){
    gmod0_dat        <- data.frame(y=NA,l_pred=.spcf_clip_l(beta0[,1], family),x0[,-1],offset0)
    names(gmod0_dat) <- names(gmod_dat)
    pred0            <- predict(gmod, newdata=gmod0_dat, type="response")
    #pred0_sd         <- sqrt( rowSums((x0 %*% beta_inv_vmat) * x0) + rowSums(Z0_sd^2))
    xbeta0           <- predict(gmod, newdata=gmod0_dat, type="link")
  }

  ######### under development
  #a_mod<- a_xname <- pred0_q <- NULL
  #if( add_learn=="lgb" & !is.na(a_par[1]) ){
  #  a_mod      <- add_mod(add_learn="lgb", train=FALSE, y=y, xbeta=xbeta, x=x,
  #                        coords=coords, xbeta0=xbeta0, x0=x0, coords0=coords0,
  #                        id_train=id_train, nx=nx, xname=xname, seed=123,
  #                        loss_hv=loss_hv, a_par=a_par, family=family)
  #  a_xname    <- a_mod$a_xname
  #  pred       <- a_mod$pred
  #  pred0      <- a_mod$pred0
  #  pred_sim   <- a_mod$pred_sim0$pred_sim
  #  pred_sd    <- a_mod$pred_sim0$pred_sd
  #  pred_q     <- a_mod$pred_sim0$pred_q
  #} else if(add_learn=="none"){
  pred0_q      <- NULL
  qs           <- c(0.005, 0.025, 0.05, seq(0.1, 0.9, 0.1), 0.95, 0.975, 0.995)
  beta_int_vmat<- vcov(gmod)
  ## spatial-block cluster-robust coefficient covariance (default): the naive
  ## vcov treats the cascade field as a known offset and understates Var(beta)
  ## because the residual is a correlated random field; .spcf_clusterSE restores
  ## near-nominal coverage. Updates both the reported coefficient SEs and the
  ## coefficient-uncertainty term of the predictive SE.
  if(robust_se && !is.null(bands) && length(bands)>0){
    cse <- tryCatch(.spcf_clusterSE(y=y, X=x, beta=beta_int, field=b,
                                    offset=offset, family=family,
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
  pred_lin     <- predict(gmod,type="link")

  ## ---- Holdout variance calibration (tau) for the spatial-process component.
  ## Uses the eq.(10) predictive variance (pv-based, link scale: grows away from
  ## data), level-calibrated by a single holdout scalar tau and capped at the
  ## marginal field variance (sill) so it saturates rather than diverging in
  ## extrapolation. The cluster-robust coefficient variance is left as-is.
  field_var    <- rowSums(Z_pv^2)
  sill         <- as.numeric(var(rowSums(Z)))            # marginal field var (link) ceiling
  if(!is.finite(sill) || sill <= 0) sill <- Inf
  ## Binomial: the logit-scale field signal is weak (binary data => small sill),
  ## so the cap binds too early and suppresses growth. Disable it for binomial
  ## (the response is already bounded in [0, 1]).
  if(identical(family$family, "binomial")) sill <- Inf
  tau          <- 1
  idt          <- mod_hv$id_train
  if(!is.null(idt) && length(idt) < n && !is.null(mod_hv$other$pred)){
    val        <- setdiff(seq_len(n), idt)
    ## in-sample working residual of the full model (field absorbed in eta) -> noise floor
    eta_f      <- .spcf_clip_l(pred_lin, family); me_f <- family$mu.eta(eta_f)
    v_f        <- pmax(family$variance(pred), 1e-8)
    r_f        <- (y - pred) / ifelse(abs(me_f) < 1e-8, 1e-8, me_f)
    w_f        <- me_f^2 / v_f
    self       <- okf <- is.finite(r_f) & is.finite(w_f) & w_f > 0 & (seq_len(n) %in% idt)
    sig2       <- if(any(self)) sum(w_f[self]*r_f[self]^2)/sum(w_f[self]) else 0
    ## holdout (out-of-sample) working residual at the validation samples
    mu_h       <- switch(family$family,
                         binomial = pmin(pmax(mod_hv$other$pred, 1e-6), 1-1e-6),
                         poisson  = pmax(mod_hv$other$pred, 1e-8), mod_hv$other$pred)
    eta_h      <- .spcf_clip_l(family$linkfun(mu_h), family); me_h <- family$mu.eta(eta_h)
    v_h        <- pmax(family$variance(mu_h), 1e-8)
    r_h        <- (y - mu_h) / ifelse(abs(me_h) < 1e-8, 1e-8, me_h)
    w_h        <- me_h^2 / v_h
    okv        <- (seq_len(n) %in% val) & is.finite(r_h) & is.finite(w_h) & w_h > 0 &
                  is.finite(field_var) & field_var > 0
    if(sum(okv) >= 2){
      Wv       <- w_h[okv]
      verr     <- sum(Wv * r_h[okv]^2) / sum(Wv)
      vfld     <- sum(Wv * field_var[okv]) / sum(Wv)
      num      <- verr - sig2
      se       <- sqrt(2 / sum(okv)) * verr
      rel      <- if(num > 0 && is.finite(se) && se > 0) num^2 / (num^2 + se^2) else 0
      tau_raw  <- if(vfld > 0) max(num, 1e-6) / vfld else 1
      tau      <- min(max(exp(log(tau_raw) * rel), 1e-2), 1e2)
      if(!is.finite(tau)) tau <- 1
    }
  }

  ## Report per-scale spatial SD (Z_sd / Z0_sd, used by sp_scalewise) on the same
  ## pv/tau/sill footing as pred_lin_sd, so they grow away from data and saturate
  ## at the sill (rowSums(Z_sd^2) == calibrated field variance, link scale).
  fv_cal       <- pmin(tau * field_var, sill)
  Z_sd         <- Z_pv * sqrt(ifelse(field_var > 0, fv_cal / field_var, 1))

  ## opt+field coefficient covariance (default se_method): recomputed here, once
  ## the calibrated per-point field SD s_f = sqrt(fv_cal) is available, replacing
  ## the classic field-retained cluster-robust covariance above. Overwrites the
  ## reported SEs and the coefficient-uncertainty term of the predictive SE.
  if(robust_se && se_method=="opt" && !is.null(bands) && length(bands)>0){
    ofse <- tryCatch(.spcf_optfield_SE(y=y, X=x, beta=beta_int, field=b,
                                       s_f=sqrt(fv_cal), offset=offset,
                                       family=family, coords=coords, bands=bands),
                     error=function(e) NULL)
    if(!is.null(ofse) && all(is.finite(diag(ofse$V))) && all(diag(ofse$V) > 0)){
      beta_int_vmat <- ofse$V
      beta_int_se   <- sqrt(diag(ofse$V))
      beta_int_summ <- data.frame(coef=beta_int, coef_se=beta_int_se,
                                  lower_95CI=beta_int-1.96*beta_int_se,
                                  upper_95CI=beta_int+1.96*beta_int_se)
    }
  }

  pred_lin_sd  <- sqrt( rowSums((x %*% beta_int_vmat) * x ) + fv_cal)
  pred_sd      <- response_se(pred_lin=pred_lin, pred_lin_sd=pred_lin_sd, family=family)

  pred_q       <- predict(gmod,type="link") + outer(pred_lin_sd, qnorm(qs), "*")
  pred_q       <- data.frame(inv_link_fun(pred_q,family=family))
  names(pred_q)<- paste("q", qs,sep="")
  #pred_sim     <- sample_from_qrf(pred_q, qs = qs, n=n, n_draw=100)

  if(!is.null(coords0)){
    pred0_lin   <- predict(gmod,type="link",newdata=gmod0_dat)
    field_var0  <- rowSums(Z0_pv^2)
    fv0_cal     <- pmin(tau * field_var0, sill)
    Z0_sd       <- Z0_pv * sqrt(ifelse(field_var0 > 0, fv0_cal / field_var0, 1))
    pred0_lin_sd<- sqrt( rowSums((x0 %*% beta_int_vmat)* x0) + fv0_cal)
    pred0_sd    <- response_se(pred_lin=pred0_lin, pred_lin_sd=pred0_lin_sd, family=family)

    pred0_q    <- predict(gmod,type="link",newdata=gmod0_dat) + outer(pred0_lin_sd, qnorm(qs), "*")
    pred0_q    <- data.frame( inv_link_fun(pred0_q,family=family) )
    names(pred0_q)<- paste("q", qs,sep="")
    #pred0_sim  <- sample_from_qrf(pred0_q, qs = qs, n=n0, n_draw=100)# Crossing?
    #pred0_sd   <- apply(pred0_sim, 1, sd)
  }
  #  a_mod      <- list(a_par=NA, a_run=FALSE, add_learn=add_learn)
  #}

  pred_ms      <- data.frame( pred, pred_sd )
  pred0_ms     <- NULL
  if(!is.null(coords0)){
    pred0_ms   <- data.frame( pred=pred0, pred_sd=pred0_sd )
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

  ######### standard deviations of model elements (transformed-scale)
  #resid_sd       <- sd(y - pred)
  #a_sd <- a_name <- NULL
  #if(a_run){
  #  a_sd         <- sd(a_mod$pred)
  #  a_name       <- paste0("additional learning (",add_learn,")")
  #}

  if(!is.null(bands)){
    elements       <- c("xb",paste0("spatial_scale",bands_scale))#,a_sd
    standard_deviation<- c(sd(x %*% beta_int_summ$coef), apply(Z,2,sd))#, a_name
  } else {
    elements       <- c("xb")#,a_sd
    standard_deviation<- c(sd(x %*% beta_int_summ$coef))#, a_name
  }
  sd_summary     <- data.frame(elements, standard_deviation)
  row.names(sd_summary)<-NULL

  ######### error statistics
  ## Evaluated on the holdout validation samples using cf_glm_hv's out-of-sample
  ## prediction (other$pred), so the pseudo-R2, RMSE, and MAE are genuine
  ## holdout metrics rather than in-sample fits at the validation indices. All NA
  ## when no validation samples are available (e.g. train_rat = 1).
  ival           <- setdiff(seq_len(n), mod_hv$id_train)
  r2 <- rmse <- mae <- NA_real_
  if(length(ival) >= 2 && !is.null(mod_hv$other$pred)){
    y_test       <- y[ival]
    y_pred       <- mod_hv$other$pred[ival]
    gmod_null    <- glm(y_test~1,family=family)
    y_pred_tr    <- link_fun(y_pred, family=family)
    gmod_fix     <- glm(y_test~0 + offset(y_pred_tr),family=family) ##################### log(y_pred)??????
    r2           <- 1 - gmod_fix$deviance / gmod_null$null.deviance# Deviance-based R2
    rmse         <- sqrt( mean( ( y_test - y_pred )^2 ) )
    mae          <- mean( abs( y_test - y_pred ) )
  }
  e_summary      <- data.frame(stat=c("validation_Pseudo-R2", "validation_RMSE","validation_MAE"),
                               value=c(r2, rmse, mae))

  ######### summary outputs
  other          <- list(n=n,n0=n0,nx=nx,y=y,x=x,x0=x0,VCmat=VCmat, #a_mod=a_mod,
                         coords=coords,coords0=coords0,vc=mod_hv$other$vc,
                         pred_pre=pred_pre, loss_hv=mod_hv$loss_hv, tau=tau)
  result         <- list(beta=beta_int_summ, sd_summary=sd_summary,
                         e_summary=e_summary, pred=pred_ms,pred0=pred0_ms,
                         pred_q=pred_q,pred0_q=pred0_q, bands=bands,
                         Z=Z,Z_sd=Z_sd, Z0=Z0, Z0_sd=Z0_sd, other=other,
                         call = match.call() )
  if(identical(se_type,"prediction")){
    ob <- tryCatch(.spcf_obs_predict(family=family, y=y, mod_hv=mod_hv,
                     pred_in=result$pred$pred, predq_in=result$pred_q,
                     pred_out=result$pred0$pred, predq_out=result$pred0_q),
                   error=function(e) NULL)
    result <- .spcf_apply_obs(result, ob)
  } else result$other$se_type <- "mean"
  class( result )<- "cf_glm"
  return( result )
}

#' @noRd
#' @export
print.cf_glm <- function(x, ...)
  {
    cat("Call:\n")
    message(format(x$call))
    cat("\n---- Coefficients -------------------------------------\n")
    message(format(x$beta))
    cat("\n---- Deviance losses (influential elements only) ------\n")
    message(format(x$sd_summary))
    cat("\n---- Error statistics ---------------------------------\n")
    message(format(x$e_summary))
    invisible(x)
  }
