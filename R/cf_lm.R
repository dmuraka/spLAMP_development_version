#' Coarse-to-fine spatial modeling (CFSM) for Gaussian response
#'
#' Prediction and regression via coarse-to-fine spatial modeling.
#'
#' @param y Vector of response variables (N x 1).
#' @param x Matrix of covariates (N x K).
#' @param coords Matrix of 2-dimensional point coordinates (N x 2).
#' @param x0 Optional. Matrix of covariates at prediction sites (N0 x K).
#' @param coords0 Optional. Matrix of 2-dimensional point coordinates at
#'   prediction sites (N0 x 2).
#' @param mod_hv Output object of the \code{\link{cf_lm_hv}} function.
#'
#' @return A list with the following elements:
#' \describe{
#'   \item{beta}{Regression coefficients, their standard errors, and the lower
#'   and upper limits of the 95 percent confidence intervals.}
#'   \item{sd_summary}{Standard deviation of the regression term (xb), spatial
#'   process (spatial_scale1, spatial_scale2,...),
#'   additionally learned components (effective if `cf_lm_hv/add_learn` is not
#'   `none`), and residuals.}
#'   \item{e_summary}{R-squared for the validation samples (validation_R2),
#'   root mean squared error for the validation samples (validation_RMSE),
#'   and the residual standard deviation (residual_SD).}
#'   \item{pred}{Predictive means and standard deviations (sample sites).}
#'   \item{pred0}{Predictive means and standard deviations (prediction sites).}
#'   \item{bands}{Bandwidth values for each scale. The i-th bandwidth is used
#'   to describe the spatial process corresponding to the i-th column of the Z matrix.}
#'   \item{Z}{Predictive means of the single-scale processes at each scale,
#'   corresponding to each bandwidth value (sample sites; list).}
#'   \item{Z_sd}{Predictive standard deviation of the spatial processes
#'   corresponding to in each bandwidth (sample sites; list).}
#'   \item{Z0}{Predictive mean of the spatial process corresponding to
#'   each bandwidth (prediction sites; list).}
#'   \item{Z0_sd}{Predictive standard deviation of the spatial process
#'   corresponding to in each bandwidth (prediction sites; list).}
#'   \item{Other}{Other internal output objects.}
#' }
#'
#' @references
#' Murakami, D., Comber, A., Yoshida, T., Tsutsumida, N., Brunsdon, C.,
#' & Nakaya, T. (2026).
#' Coarse-to-fine spatial modeling: A scalable, machine-learning-compatible
#' framework.
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
#' @author Daisuke Murakami
#'
#' @importFrom dbscan frNN
#' @importFrom fields rdist
#' @importFrom FNN get.knnx
#' @importFrom nloptr nloptr
#' @importFrom ranger ranger
#' @importFrom utils capture.output
#' @importFrom stats approx kmeans predict quantile rnorm runif sd var cor
#' glm as.formula vcov qnorm residuals coefficients gaussian
#'
#' @export
cf_lm        <- function(y, x=NULL, coords, x0=NULL, coords0=NULL, mod_hv){

  if(!is.null(coords0) && !is.null(x) && is.null(x0)){
    stop("Error: x0 must be provided when x is specified")
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

  } else {
    n0   <- x0   <- NA
    pred0 <- Z0  <- Z0_sd <- NULL
  }

  ##################### main loop for feature extraction
  print("--- Learning multi-scale spatial process ---", quote=FALSE)

  bands_scale    <- which(mod_hv$other$VCmat[,1]==1)

  b_old          <- NULL
  Z    <- Z_sd   <- matrix(0,nrow=n ,ncol=length(bands))
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
        }
        comment         <- ""
      } else {
        comment         <- " no improvement (skipped)"
      }

      print_add   <- ifelse(i<10,"  "," ")
      print( paste0( " Scale",print_add,i,
                     " (bandwidth:",format(bands_all[i],digits=7),")", comment),
             quote = FALSE )
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

  ######### additional learning
  a_mod          <- list()
  a_mod$add_learn<- "none"
  a_pred <- a_pred0 <- a_pred_v <- a_pred0_v <- 0
  if(a_run){
    a_mod        <- add_mod(add_learn=add_learn, train=FALSE, resid=resid,
                            x=x, coords=coords, x0=x0, coords0=coords0,
                            id_train=NULL, sse_hv=NULL, a_par=a_par,
                            nx=nx, xname=xname)
    a_pred       <- a_mod$pred
    a_pred0      <- a_mod$pred0

    ### Standard deviations
    qs           <- seq(0, 1, length.out = 201)
    a_dat        <- data.frame(x[,-1],coords)
    names(a_dat) <- a_mod$a_xname
    a_qmat       <- predict(a_mod$mod, data = a_dat, type = "quantiles", quantiles = qs)$predictions
    a_Pred_sim   <- sample_from_qrf(a_qmat, qs, n, n_draw = 200)
    a_pred_v     <- apply(a_Pred_sim ,1, var)
    if(!is.null(coords0)){
      a_dat0       <- data.frame(x0[,-1],coords0)
      names(a_dat0)<- a_mod$a_xname
      a_qmat0      <- predict(a_mod$mod, data = a_dat0, type = "quantiles", quantiles = qs)$predictions
      a_Pred0_sim  <- sample_from_qrf(a_qmat0, qs, n0, n_draw = 200)
      a_pred0_v    <- apply(a_Pred0_sim ,1, var)
    }
  }

  ######### prediction
  pred           <- rowSums(x*beta) + a_pred
  pred_sd        <- sqrt( rowSums((x %*% beta_int_vmat) * x) + rowSums(Z_sd^2) + a_pred_v)
  pred_ms        <- data.frame( pred, pred_sd )
  pred0_ms       <- NULL
  if(!is.null(coords0)){
    pred0        <- rowSums(x0*beta0)+ a_pred0
    pred0_sd     <- sqrt( rowSums((x0 %*% beta_int_vmat) * x0) + rowSums(Z0_sd^2) + a_pred0_v)
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
    elements       <- c("xb",paste0("spatial_scale",bands_scale),a_sd,"residuals")
    standard_deviation<- c(sd(x %*% beta_int_summ$coef), apply(Z,2,sd), a_name, resid_sd)
  } else {
    elements       <- c("xb",a_sd,"residuals")
    standard_deviation<- c(sd(x %*% beta_int_summ$coef), a_name, resid_sd)
  }
  sd_summary     <- data.frame(elements, standard_deviation)
  row.names(sd_summary)<-NULL

  ######### error statistics
  r2             <- cor(y[-mod_hv$id_train], pred[-mod_hv$id_train])^2
  rmse           <- sqrt(mod_hv$sse_hv/(n-length(mod_hv$id_train)))
  e_summary      <- data.frame(stat=c("validation_R2", "validation_RMSE",
                                      "residual_SD"),
                               value=c(r2, rmse, resid_sd))

  ######### summary outputs
  other          <- list(n=n,n0=n0,nx=nx,y=y,x=x,x0=x0,VCmat=VCmat,
                         coords=coords,coords0=coords0,vpar=vpar,
                         vc=mod_hv$other$vc, xx_inv=xx_inv, a_mod=a_mod,
                         pred_pre=pred_pre, sse_hv=mod_hv$sse_hv)
  result         <- list(beta=beta_int_summ, sd_summary=sd_summary,
                         e_summary=e_summary, pred=pred_ms,pred0=pred0_ms,bands=bands,
                         Z=Z,Z_sd=Z_sd, Z0=Z0, Z0_sd=Z0_sd, other=other,
                         call = match.call() )
  class( result ) <- "cf_lm"
  return( result )
}

#' @noRd
#' @export
print.cf_lm <- function(x, ...)
  {
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
