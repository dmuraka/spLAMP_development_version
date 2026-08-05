#' Holdout validation for the Gaussian coarse-to-fine spatial modeling (CFSM)
#'
#' Trains the CFSM-based Gaussian spatial regression and selects the number of
#' spatial scales through sequential holdout validation.
#'
#' @param y Vector of response variables (N x 1).
#' @param x Matrix of covariates (N x K).
#' @param coords Matrix of 2-dimensional point coordinates (N x 2).
#' @param train_rat Training sample ratio (default: 0.75). For small to
#'   moderate samples (N <= 30000), samples closest to the k-means centers
#'   are used for validation samples to stabilize training.
#'   For larger samples, training samples are drawn at random.
#' @param id_train Optional. ID indicating training samples. If specified,
#'   the corresponding samples are used as training samples.
#'   Otherwise, training samples are chosen based on `train_rat`.
#' @param alpha Decay ratio of the kernel bandwidth in the coarse-to-fine
#'   training (default: 0.9). Values closer to one make the optimization
#'   more stringent but increase computation time.
#' @param kernel Kernel type for modeling spatial dependence. `"exp"` for the
#'   exponential kernel (default) and `"gau"` for the Gaussian kernel.
#' @param add_learn Additional learner trained on the residuals to capture
#'   non-linear patterns and/or higher-order interactions. `"rf"` uses a
#'   random forest (\pkg{ranger}) and `"lightgbm"` uses LightGBM
#'   (\pkg{lightgbm}); both are tuned by minimizing validation SSE.
#'   For `"lightgbm"`, the predictive quantiles are conformalized on the
#'   validation split so that their uncertainty is calibrated.
#'   Both learners are optional: the corresponding package
#'   (\pkg{ranger} or \pkg{lightgbm}) must be installed, and an informative
#'   error is raised if it is not.
#'   Default is `"none"`, meaning no additional training.
#' @param seed Random seed used for the training/validation split when
#'   `id_train` is not supplied. Default is `123`. Set to `NULL` to allow
#'   a different split at each call (useful for assessing split sensitivity).
#'
#' @return A list with the following elements:
#' \describe{
#'   \item{sse_hv}{Final sum-of-squared error (SSE) for validation samples.}
#'   \item{sse_hv_all}{SSEs obtained at each learning step.}
#'   \item{id_train}{ID of training samples.}
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
#' @seealso \code{\link{cf_lm}}
#' @author Daisuke Murakami
#'
#' @export
cf_lm_hv     <- function(y, x=NULL, coords, train_rat=0.75, id_train=NULL,
                         alpha=0.9, kernel="exp", add_learn="none", seed=123){
  n_obs         <- .spcf_check_data(y = y, x = x, coords = coords)
  .spcf_check_hv_args(n_obs, train_rat, id_train, alpha, kernel, add_learn)

  init          <- initial_fun(y=y,x=x,coords=coords,train_rat=train_rat,
                               id_train=id_train, x_sel=NULL, seed=seed)
  xx_inv        <- init$xx_inv
  beta_int      <- init$beta_int
  beta          <- init$beta
  coords        <- init$coords
  coords_uni    <- unique(coords)
  pred          <- init$pred
  resid         <- init$resid
  x             <- init$x
  x_sel         <- init$x_sel
  xname         <- init$xname
  n             <- init$n
  nx            <- init$nx
  id_train      <- init$id_train
  vc            <- 1
  ridge         <- TRUE

  Z             <- matrix(0,nrow=n,ncol=100)#list(NULL)
  max_d         <- sqrt(diff(range(coords[,1]))^2+diff(range(coords[,2]))^2)/3
  Bands         <- max_d*alpha^(1:100)
  accept_num    <- 5

  ##################### main loop for feature extraction
  coords_old    <- NULL
  sel_id_list   <- list(NULL)
  b_old         <- NULL
  bands         <- NULL
  message("--- SSE: Linear regression ---")
  SSE           <- sum( resid[-id_train]^2 )
  SSE_name      <- "linear regression"
  message(format(SSE))

  message("--- SSE: Learning multi-scale spatial process ---")
  count         <- 0
  VCmat         <- NULL
  for(i in 1:length(Bands)){
    band        <- Bands[i]
    lmod        <- lwr(coords=coords, coords_uni=coords_uni, resid=resid, x=x,
                       band=band, coords_old=coords_old, b_old=b_old,vc=vc,
                       id_train=id_train,ridge=ridge, kernel=kernel,beta=beta,
                       y=y, coords0=NULL, x0=NULL, sel_id=NULL,func="cf_lm_hv")
    run         <- lmod$run
    if(run==TRUE){
      bands     <- c(bands, band)
      b_old     <- lmod$b_old
      coords_old<- lmod$coords_cent

      beta_add  <- lmod$beta
      pred_add  <- lmod$pred
      beta      <- beta + beta_add
      pred      <- pred + pred_add
      resid     <- y - pred
      SSE       <- c(SSE ,lmod$sse_hv)
      vc_sel    <- lmod$vc_sel
      vcmat     <- rep(0,nx);vcmat[vc_sel]<-1
      VCmat     <- rbind(VCmat,vcmat)

      beta_int_add  <- xx_inv %*% t(x)%*%resid
      pred0_add <- x%*%beta_int_add
      beta      <- sweep(beta, 2, beta_int_add, "+")
      pred      <- pred  + pred0_add
      resid     <- resid - pred0_add

      beta_add_m<- colMeans(beta_add)
      Z[,i]     <- beta_add[,1] - beta_add_m[1]#sweep(beta_add, 2, beta_add_m, "-") # centered process
      sel_id_list[[i]]<- lmod$sel_id
      beta_int  <- beta_int + beta_int_add + beta_add_m# de-centered coefficients
      count     <- 0
      comment   <- ""
    } else {
      if(i>10) count      <- count + 1
      if(count==accept_num) break

      VCmat     <-rbind(VCmat,rep(0,nx))
      SSE       <-c(SSE, SSE[length(SSE)])
      comment   <- " no improvement"
    }

    SSE_name    <- c(SSE_name, paste0("scale ",i))
    print_add   <- ifelse(i<10,"  "," ")
    message( paste0( formatC(SSE[length(SSE)], digits = 7, format = "g"),#, flag = "#"
                   " (Scale",print_add, i,")", comment))
  }

  nonzero_Z_sd    <- apply(Z,2,sd)>0
  if(sum(nonzero_Z_sd)>0){
    bid           <- which(nonzero_Z_sd)#which(sapply(BBB, length) > 0)
    max_bid       <- max(bid)
    Z             <- Z[,1:max_bid, drop=FALSE]
    n_bid         <- length(bid)

    message("")
    message(paste("-> Selected finest scale: ", max_bid, " (bandwidth: ",
                formatC(Bands[max_bid], digits = 7, format = "g"),")", sep=""))
    message("")
  } else {
    bid           <- NULL
    Z             <- NULL
    n_bid         <- 0
  }

  if(n_bid>1){
    message("--- SSE: After coefficient adjustment ---")
    ZZ          <- Z[,bid]
    bopt_obj    <- (function(bands, ZZ, beta_int, nx,#, is_vc
                             x, y, n_bid, id_train) {
      function(par) {
        out     <- try(bopt_core(par, bands = bands, Z = ZZ,
                                 beta_int = beta_int, nx = nx,#, is_vc = is_vc
                                 x = x, y = y, n_bid = n_bid, id_train=id_train),
                       silent = TRUE)
        if (inherits(out, "try-error") || !is.finite(out$sse)) {
          return(.Machine$double.xmax)
        }
        out$sse
      }
    })(bands, ZZ, beta_int, nx, x, y, n_bid, id_train)#, is_vc

    v_opt0      <- nloptr(x0 = 0,eval_f = bopt_obj,#rep(0, sum(is_vc))
                          opts = list(algorithm = "NLOPT_LN_BOBYQA", maxeval = 500))
    v_test      <- bopt_core(v_opt0$solution, bands=bands, Z=ZZ,
                             beta_int=beta_int, nx=nx,#, is_vc=is_vc
                             x=x, y=y, n_bid=n_bid,id_train=id_train)
    if(v_test$sse< SSE[length(SSE)]){
      vpar      <- c(v_test$vpar[1], v_opt0$solution)
    } else {
      vpar      <- c(1, 0)
    }

  } else {
    if(n_bid==1){
      vpar      <- c(1, 0)
    } else if(n_bid==0){
      vpar      <- c(NA,NA)
      message("Warning: No residual spatial process was detected.")
    }
  }

  xbeta         <- matrix(0,nrow=n,ncol=nx)
  for(j in 1:nx){
    xbeta[,j] <- x[,j] * beta_int[j,1]
  }

  if(!is.na(vpar[1])){
    w_0       <- exp(-vpar[2]/bands)
    w         <- vpar[1]* w_0/w_0[1]
    w[w<0]    <- 0
    b         <- Z[,bid,drop=FALSE]%*%w
    xbeta[,1] <- xbeta[,1] + x[,1]*b
  }

  pred          <- rowSums(xbeta)
  resid         <- y-pred
  sse_hv        <- sum( resid[-id_train]^2 )
  SSE           <- c(SSE,sse_hv)
  SSE_name      <- c(SSE_name, "coef. adjustment")

  if(n_bid>1){
    message(formatC(sse_hv, digits = 7))
  }

  if(add_learn=="rf" || add_learn=="lightgbm"){
    message("--- SSE: After additional learning ---")
    a_mod0      <- add_mod(add_learn=add_learn, train=TRUE, resid=resid, x=x,
                         coords=coords, x0=NULL, coords0=NULL,id_train=id_train,
                         nx=nx, xname=xname, sse_hv=sse_hv)
    sse_hv      <- a_mod0$sse_hv
    SSE         <- c(SSE,sse_hv)
    SSE_name    <- c(SSE_name, "additional learning")

    message(formatC(sse_hv, digits = 7))
  } else if(add_learn=="none"){
    a_mod0      <- list(a_par=NA, a_run=FALSE, add_learn=add_learn)
  }

  sse_hv_all    <- data.frame(learning=SSE_name, sse_hv=SSE)

  ##################### summary
  ## Holdout prediction of the selected model at all samples. On the validation
  ## samples (complement of id_train) this is out-of-sample, so cf_lm uses it to
  ## report a genuine holdout validation_R2 consistent with the holdout SSE
  ## (sse_hv). The additional learner's validation prediction (if any) is folded
  ## in at the validation samples so pred_hv matches sse_hv after add_learn.
  pred_hv       <- pred
  if(isTRUE(a_mod0$a_run) && !is.null(a_mod0$a_pred_hv)){
    pred_hv[-id_train] <- pred_hv[-id_train] + a_mod0$a_pred_hv
  }
  other         <- list(bands=bands,bands_all=Bands,vpar=vpar,alpha=alpha,ridge=ridge,
                        vc=vc,x_sel=x_sel, sel_id_list=sel_id_list,
                        coords_uni=coords_uni,VCmat=VCmat,kernel=kernel, a_mod0=a_mod0,
                        pred_hv=pred_hv)
  result        <- list(sse_hv=sse_hv, sse_hv_all=sse_hv_all,
                        id_train=id_train, other=other, call = match.call())
  class( result ) <- "cf_lm_hv"
  return( result )
}

#' @noRd
#' @export
print.cf_lm_hv <- function(x, ...)
{
  cat("Call:\n")
  print(x$call)
  cat("\n----Sum-of-squares errors for validation samples-----\n")
  print(x$sse_hv_all)
  invisible(x)
}
