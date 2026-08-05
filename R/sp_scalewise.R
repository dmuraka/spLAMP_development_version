#' Extract scale-wise spatial processes
#'
#' Evaluate mean and standard deviation of the (multiscale) spatial process for
#' bandwidth values within a pre-specified range. For a spatio-temporal fit from
#' \code{\link{cf_dglm}}, the process can additionally be averaged over a
#' user-specified time range, returning the temporally averaged spatial process
#' at each (sample / prediction) location.
#'
#' @param mod Output object from the \code{\link{cf_lm}}, \code{\link{cf_glm}} or
#'   \code{\link{cf_dglm}} function.
#' @param bw_range Range of bandwidth values of the synthesized spatial
#'   processes, treated as the half-open interval [min, max). For example,
#'   bw_range = c(10, 20) synthesizes scales with bandwidth b such that
#'   10 <= b < 20. The half-open convention lets contiguous ranges (e.g.
#'   c(0, 10) and c(10, Inf)) partition the scales without double-counting a
#'   scale whose bandwidth equals the shared endpoint. The default c(0, Inf)
#'   synthesizes all scales.
#' @param time_range Range of time points over which the spatio-temporal process
#'   is averaged. Only used when \code{mod} is a \code{\link{cf_dglm}} fit (which
#'   carries a time index for every row of \code{Z}). For example, time_range =
#'   c(5, 10) averages the process over time points 5 to 10 at each location. The
#'   default c(-Inf, Inf) averages over all time points. Ignored (with a warning
#'   if set to a non-default value) for purely spatial fits.
#'
#' @return A list with the following elements:
#' \describe{
#'   \item{pred}{Means and standard deviations of the spatial process at the
#'   sample sites. For a \code{cf_dglm} fit, one row per (unique) location with
#'   the temporally averaged process, together with its coordinates and the
#'   number of averaged time points.}
#'   \item{pred0}{The same at the prediction sites. \code{NULL} when \code{mod}
#'   was fitted without prediction sites, and also (with a warning) when no
#'   prediction site falls inside \code{time_range}.}
#' }
#'
#' @seealso \code{\link{cf_lm}}, \code{\link{cf_glm}}, \code{\link{cf_dglm}}
#' @author Daisuke Murakami
#'
#' @export
sp_scalewise      <- function(mod, bw_range=c(0,Inf), time_range=c(-Inf,Inf)){

  ok_cls          <- c("cf_lm","cf_glm","cf_dglm")
  if( inherits(mod, "cf_downscale") ){
    .spcf_stop("sp_scalewise() supports cf_lm(), cf_glm() and cf_dglm() fits; for a cf_downscale() fit the scale-wise processes are already returned as mod$Z and mod$Z_sd.")
  }
  if( !inherits(mod, ok_cls) ){
    .spcf_stop(sprintf(
      "'mod' must be a fitted model from cf_lm(), cf_glm() or cf_dglm(), but an object of class %s was given.",
      paste(sQuote(class(mod)), collapse="/")))
  }
  ## bw_range is half-open [min, max), so an empty range is an error; time_range
  ## is closed, and c(t, t) legitimately selects a single time point.
  if( !is.numeric(bw_range) || length(bw_range) != 2L || anyNA(bw_range) ||
      bw_range[1] >= bw_range[2] ){
    .spcf_stop("'bw_range' must be a numeric vector of length 2 with bw_range[1] < bw_range[2]; the range is half-open, [min, max).")
  }
  if( !is.numeric(time_range) || length(time_range) != 2L || anyNA(time_range) ||
      time_range[1] > time_range[2] ){
    .spcf_stop("'time_range' must be a numeric vector of length 2 with time_range[1] <= time_range[2]; use c(t, t) for a single time point.")
  }
  if( is.null(mod$bands) || length(mod$bands) == 0 ){
    .spcf_stop("this fit committed no spatial scale (the holdout validation stopped before any scale improved the validation score), so there is no scale-wise process to extract.")
  }

  cols            <- which( mod$bands >= min(bw_range) & mod$bands < max(bw_range) )
  if( length(cols) == 0 ){
    .spcf_stop(sprintf(
      "no spatial scale falls inside bw_range = [%s, %s). The fitted bandwidths range from %s to %s.",
      format(min(bw_range)), format(max(bw_range)),
      format(min(mod$bands)), format(max(mod$bands))))
  }

  ## per-row time index (present only for cf_dglm fits); NULL otherwise
  has_time        <- !is.null(mod$other$time)
  if( !has_time && !identical(time_range, c(-Inf,Inf)) ){
    warning("time_range is ignored: 'mod' has no time dimension (not a cf_dglm fit)")
  }

  ## Collapse the scale-wise Z / Z_sd of one site set to a (pred, pred_sd) frame.
  ## Without a time index: one value per row (cf_lm / cf_glm; unchanged behaviour).
  ## With a time index   : average over time_range at each location, returning the
  ## temporally averaged process (variance of the mean assumes independence across
  ## time, so it is a lower bound when the AR(1) temporal correlation is strong).
  ## `required`: an empty time window at the sample sites leaves nothing to
  ## return and is an error, while prediction sites are optional output -- there
  ## the window is reported and NULL is returned, so a window chosen for the
  ## sample sites still works when the fit carries prediction sites at other
  ## time points.
  collapse        <- function(Z, Zsd, coords, time, where, required=TRUE){
    fld           <- rowSums(Z[, cols, drop=FALSE])          # multiscale field per row
    vr            <- rowSums(Zsd[, cols, drop=FALSE]^2)       # its variance (scales indep.)
    if( is.null(time) ){
      return(data.frame(pred=fld, pred_sd=sqrt(vr)))
    }
    keep          <- time >= min(time_range) & time <= max(time_range)
    if( !any(keep) ){
      msg         <- sprintf(
        "no %s fall inside time_range = [%s, %s]; their time points range from %s to %s.",
        where, format(min(time_range)), format(max(time_range)),
        format(min(time)), format(max(time)))
      if( required ) .spcf_stop(msg)
      warning(msg, " Returning NULL for them.", call.=FALSE)
      return(NULL)
    }
    fld <- fld[keep]; vr <- vr[keep]; co <- coords[keep, , drop=FALSE]
    g             <- match( paste(co[,1], co[,2]), unique(paste(co[,1], co[,2])) )
    nt            <- as.numeric( tapply(fld, g, length) )     # # averaged time points
    mn            <- as.numeric( tapply(fld, g, mean) )       # time-mean of the field
    vm            <- as.numeric( tapply(vr,  g, sum) ) / nt^2 # variance of the time-mean
    uc            <- co[!duplicated(g), , drop=FALSE]
    data.frame(px=uc[,1], py=uc[,2], pred=mn, pred_sd=sqrt(vm), n_time=nt)
  }

  z_ms            <- collapse(mod$Z, mod$Z_sd, mod$other$coords,
                              if(has_time) mod$other$time else NULL,
                              where="sample sites")

  z0_ms           <- NULL
  if( !is.na(mod$other$n0) ){
    z0_ms         <- collapse(mod$Z0, mod$Z0_sd, mod$other$coords0,
                              if(has_time) mod$other$time0 else NULL,
                              where="prediction sites", required=FALSE)
  }

  return(list(pred=z_ms, pred0=z0_ms))
}
