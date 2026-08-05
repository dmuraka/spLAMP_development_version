#' Holdout validation for
#' coarse-to-fine spatial generalized linear mixed models (CF-GLMMs)
#'
#' Trains CF-GLMMs and selects the number of spatial scales through sequential
#' holdout validation.
#'
#' @param y Vector of response variables (N x 1) including continuous, count,
#'   and binary responses, following an exponential family distribution.
#' @param x Matrix of covariates (N x K).
#' @param coords Matrix of 2-dimensional point coordinates (N x 2).
#' @param offset Optional. Vector of offset variables (N x 1) included in the
#'   linear predictor, consistent with \code{\link{glm}}.
#' @param train_rat Training sample ratio (default: 0.75). For small to
#'   moderate samples (N <= 30000), samples closest to the k-means centers
#'   are used for validation samples to stabilize training.
#'   For larger samples, training samples are drawn at random.
#' @param id_train Optional. ID indicating training samples. If specified,
#'   the corresponding samples are used as training samples. Otherwise, training
#'   samples are chosen based on `train_rat`.
#' @param alpha Decay ratio of the kernel bandwidth in the coarse-to-fine
#'   training (default: 0.9). Values closer to one make the optimization
#'   more stringent but increase computation time.
#' @param kernel Kernel type for modeling spatial dependence. `"exp"` for
#'   the exponential kernel (default) and `"gau"` for the Gaussian kernel.
#' @param family Error distribution and link function specification,
#'   consistent with the 'family' argument of \code{\link{glm}}.
#' @param seed Random seed used for the training/validation split when
#'   `id_train` is not supplied. Default is `1234`. Set to `NULL` to allow
#'   a different split at each call (useful for assessing split sensitivity).
#'
#' @return A list with the following elements:
#' \describe{
#'   \item{loss_hv}{Final deviance loss for validation samples.}
#'   \item{loss_hv_all}{Deviance losses obtained at each learning step.}
#'   \item{id_train}{ID of training samples.}
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
#' @seealso \code{\link{cf_glm}}
#' @author Daisuke Murakami
#'
#' @export
cf_glm_hv  <- function(y, x=NULL, coords, offset=NULL, train_rat=0.75, id_train=NULL,
                       alpha=0.9, kernel="exp", family=gaussian(), seed=1234){

  n_obs          <- .spcf_check_data(y = y, x = x, coords = coords, offset = offset)
  .spcf_check_hv_args(n_obs, train_rat, id_train, alpha, kernel)

  init           <- initial_fun_glm(x=x,y=y,coords=coords,offset=offset,
                                    train_rat=train_rat,x_sel=NULL,family=family,
                                    id_train=id_train, seed=seed)
  beta_int       <- init$beta_int
  beta           <- init$beta
  coords         <- init$coords
  coords_uni     <- unique(coords)
  resid          <- init$resid
  x              <- init$x
  x_sel          <- init$x_sel
  xname          <- init$xname
  offset         <- init$offset
  n              <- init$n
  nx             <- init$nx
  id_train       <- init$id_train
  gmod0          <- init$gmod
  w              <- init$gmod$weights
  vc             <- 1
  ridge          <- TRUE
  Bands_max      <- 100
  Z              <- matrix(0,nrow=n,ncol=Bands_max)
  max_d          <- sqrt(diff(range(coords_uni[,1]))^2+diff(range(coords_uni[,2]))^2)/3
  Bands          <- max_d*alpha^(1:Bands_max)
  ## Floor the bandwidth grid at ~half the typical inter-point spacing (median
  ## nearest-neighbour distance of the unique locations), mirroring cf_dglm_hv:
  ## bands finer than the data resolution carry no information and make the
  ## kernel exp(-d/b) underflow to empty knot weights, so the greedy scan would
  ## only waste per-band frNN/kmeans/glm cost on them. Trims the (never-improving)
  ## fine tail of the grid; the accepted scales are unchanged.
  band_min       <- 0.5*stats::median(FNN::get.knn(coords_uni, k=1)$nn.dist)
  if(is.finite(band_min) && band_min>0){
    Bands        <- Bands[Bands >= band_min]
    if(length(Bands)==0) Bands <- max_d*alpha
  }
  accept_num     <- 5

  ##################### main loop for feature extraction
  coords_old     <- NULL
  sel_id_list    <- list(NULL)
  b_old          <- NULL
  bands          <- NULL
  message("--- Deviance: Basic GLM ---")
  Loss  <-sse_hv0<- sum( residuals(init$gmod, type="deviance")[-id_train]^2 )
  Loss_name      <- "basic GLM"
  message(format(Loss))

  message("--- Deviance: Learning multi-scale spatial process ---")
  l_pred         <- 0
  count          <- 0
  VCmat          <- NULL
  for(i in 1:length(Bands)){
    band         <- Bands[i]
    lmod         <- lwr_glm(coords=coords, coords_uni=coords_uni, resid=resid,
                            x=x, w=w, offset=offset, band=band, b_old=b_old,
                            coords_old=coords_old, vc=vc, id_train=id_train,
                            ridge=ridge,kernel=kernel,y=y,
                            coords0=NULL, x0=NULL, #offset0=NULL,
                            sel_id=NULL, sse_hv0=sse_hv0, l_pred=l_pred,
                            family=family,func="cf_glm_hv") #extras: w, sse_hv0
    run          <- lmod$run
    if(run==TRUE){
      lmod_final      <- lmod
      band_final      <- band

      bands           <- c(bands, band)
      b_old           <- lmod$b_old
      sse_hv0         <- lmod$sse_hv
      coords_old      <- lmod$coords_cent

      l_pred_add      <- lmod$pred
      l_pred          <- l_pred  + l_pred_add
      l_bias          <- mean(l_pred)         # mean of the linear predictor
      l_pred          <- l_pred   - l_bias    # centre it before the next scale

      beta_add        <- lmod$beta
      beta_add[,1]    <- beta_add[,1]- l_bias
      beta            <- beta    + beta_add
      Z[,i]           <- beta_add[,1]
      sel_id_list[[i]]<- lmod$sel_id

      l_pred_off      <- .spcf_clip_l(l_pred, family) + offset
      ## glm.fit direct (dglm-style): identical MLE / working residuals /
      ## weights as glm(y ~ 0 + x + offset(l_pred_off)), without the formula
      ## model.frame/terms rebuild each band.
      gmod0           <- glm.fit(x, y, offset=l_pred_off, family=family)
      resid           <- gmod0$residuals
      w               <- gmod0$weights
      beta_int_new    <- matrix(gmod0$coefficients)
      for(jj in 1:nx){
        beta[,jj]     <- beta[,jj] - beta_int[jj,1] + beta_int_new[jj]
      }
      beta_int        <- beta_int_new
      ## sum of squared deviance residuals == sum of per-obs deviance contribs
      loss_new        <- sum(family$dev.resids(y, gmod0$fitted.values, 1)[-id_train] )
      Loss            <- c(Loss ,loss_new)

      vc_sel          <- lmod$vc_sel
      vcmat           <- rep(0,nx);vcmat[vc_sel]<-1
      VCmat           <- rbind(VCmat,vcmat)
      count           <- 0
      comment   <- ""
    } else {
      if(i>10) count  <- count + 1
      if(count==accept_num) break

      VCmat           <- rbind(VCmat,rep(0,nx))
      Loss            <- c(Loss, Loss[length(Loss)])
      comment         <- " no improvement"
    }

    Loss_name     <- c(Loss_name, paste0("scale ",i))
    print_add<-ifelse(i<10,"  "," ")
    message( paste0( formatC(Loss[length(Loss)], digits = 7, format = "g"),#, flag = "#"
                   " (Scale",print_add, i,")", comment))
  }

  nonzero_Z_sd    <- apply(Z,2,sd)>0
  if(sum(nonzero_Z_sd)>0){
    bid           <- which(nonzero_Z_sd)
    max_bid       <- max(bid)
    Z             <- Z[,1:max_bid,drop=FALSE]
    n_bid         <- length(bid)
    z_pred        <- 0
    if(n_bid>0) z_pred  <- rowSums(Z[,bid,drop=FALSE])

    message("")
    message(paste("-> Selected finest scale: ", max_bid, " (bandwidth: ",
                formatC(Bands[max_bid], digits = 7, format = "g"),")", sep=""))
    message("")

  } else {
    bid           <- NULL#which(apply(Z,2,sd)>0)
    Z             <- NULL
    n_bid         <- 0#length(bid)
    z_pred        <- 0
  }

  xbeta                 <- 0
  for(j in 1:nx)  xbeta <- xbeta + x[,j] * beta_int[j,1]
  xbeta     <- xbeta + z_pred
  xbeta_off    <- .spcf_clip_l(xbeta, family) + offset
  gmod1        <- glm(y~0+offset(xbeta_off),family=family)         ####################delete together with out pred
  loss_hv      <- sum(residuals(gmod1, type="deviance")[-id_train]^2 )

  ### under development
  #a_par        <- data.frame(num_leaves=NA, min_data_in_leaf=NA,learning_rate=NA)
  #if( add_learn=="lgb" ){
  #  message("--- Loss: Additional learning ( LightGBM ) ---")
  #  a_mod0     <- add_mod(add_learn="lgb", train=TRUE, y=y, xbeta=xbeta, x=x,
  #                        coords=coords, xbeta0=NULL, x0=NULL, coords0=NULL,
  #                        id_train=id_train, nx=nx, xname=xname, seed=123,
  #                        loss_hv=loss_hv, family=family)
  #  a_par      <- a_mod0$a_par
  #  a_run      <- a_mod0$a_run
  #  loss_hv    <- a_mod0$loss_hv
  #  message(formatC(loss_hv, digits = 7))
  #} else if(add_learn=="none"){
  #  a_mod0     <- list(a_par=NA, a_run=FALSE, add_learn=add_learn)
  #}

  loss_hv_all  <- data.frame(learning=Loss_name, loss_hv=Loss)

  ##################### summary parameters
  other        <- list(bands=bands,bands_all=Bands, alpha=alpha,ridge=ridge,
                       vc=vc, x_sel=x_sel,sel_id_list=sel_id_list,Loss=Loss,
                       coords_uni=coords_uni,VCmat=VCmat,kernel=kernel, #a_mod0=a_mod0, a_par=a_par
                       pred=predict(gmod1,type="response"),
                       family=family)#,hetero=hetero,
  result       <- list(loss_hv=loss_hv, loss_hv_all=loss_hv_all,
                       id_train=id_train, other=other, call = match.call())
  class( result ) <- "cf_glm_hv"
  return( result )
}

#' @noRd
#' @export
print.cf_glm_hv <- function(x, ...)
{
  cat("Call:\n")
  print(x$call)
  cat("\n---- Deviance losses for validation samples -----\n")
  print(x$loss_hv_all)
  invisible(x)
}
