## Shared argument checks for the exported cf_* / sp_* functions.
##
## Purpose: fail early with a message that names the offending argument and
## says how to fix it, instead of letting a malformed input reach the linear
## algebra / C++ layer and surface as "logical subscript too long",
## "missing value where TRUE/FALSE needed" or "NA/NaN/Inf in foreign function
## call". Valid input must pass through completely unchanged.

.spcf_stop <- function(...) stop(..., call. = FALSE)

## Names a vector/matrix-like argument's row count without coercing it.
.spcf_nrow <- function(v) if (is.null(dim(v))) length(v) else nrow(v)

## y / x / coords / offset / time must be numeric and free of NA/NaN/Inf.
.spcf_check_finite <- function(v, nm) {
  if (is.data.frame(v)) {
    bad <- !vapply(v, function(z) is.numeric(z) || is.logical(z), logical(1))
    if (any(bad))
      .spcf_stop(sprintf(
        "'%s' must be numeric: column(s) %s are of class %s. Dummy-code factors/characters first (e.g. model.matrix).",
        nm, paste(sQuote(names(v)[bad]), collapse = ", "),
        paste(unique(vapply(v[bad], function(z) class(z)[1], "")), collapse = "/")))
    v <- as.matrix(v)
  }
  if (!is.numeric(v) && !is.logical(v))
    .spcf_stop(sprintf("'%s' must be numeric, but it is of class %s.",
                       nm, sQuote(class(v)[1])))
  nna <- sum(is.na(v))
  if (nna > 0)
    .spcf_stop(sprintf(
      "'%s' contains %d missing value(s) (NA/NaN). spCF does not handle missing data; remove or impute the affected observations first.",
      nm, nna))
  ninf <- sum(!is.finite(v))
  if (ninf > 0)
    .spcf_stop(sprintf("'%s' contains %d infinite value(s) (Inf/-Inf).", nm, ninf))
  invisible(TRUE)
}

## coords must be an n x 2 numeric object.
.spcf_check_coords <- function(coords, n, nm = "coords") {
  if (is.null(coords)) .spcf_stop(sprintf("'%s' must be supplied.", nm))
  if (is.null(dim(coords)) || ncol(coords) != 2)
    .spcf_stop(sprintf(
      "'%s' must have exactly 2 columns (x and y), but it has %s. Supply a two-column matrix or data.frame of projected coordinates.",
      nm, if (is.null(dim(coords))) "none (it is a plain vector)" else ncol(coords)))
  if (!is.null(n) && nrow(coords) != n)
    .spcf_stop(sprintf("'%s' has %d row(s) but %d observation(s) were given; they must match.",
                       nm, nrow(coords), n))
  .spcf_check_finite(coords, nm)
  invisible(TRUE)
}

## Core check shared by every cf_*_hv / cf_* fitting call.
##   y       response (length n)
##   x       covariates or NULL (n rows)
##   coords  n x 2
##   offset  optional, length n
##   time    optional, length n (cf_dglm)
.spcf_check_data <- function(y, x, coords, offset = NULL, time = NULL) {
  if (is.null(y)) .spcf_stop("'y' must be supplied.")
  if (!is.null(dim(y)) && length(dim(y)) == 2L && ncol(y) != 1L)
    .spcf_stop(sprintf("'y' must be a vector (or a one-column object), but it has %d columns.",
                       ncol(y)))
  .spcf_check_finite(y, "y")
  n <- length(as.vector(as.matrix(y)))
  if (n < 3L)
    .spcf_stop(sprintf("'y' has %d observation(s); too few to fit a model.", n))

  .spcf_check_coords(coords, n)

  if (!is.null(x)) {
    if (.spcf_nrow(x) != n)
      .spcf_stop(sprintf(
        "'x' has %d row(s) but 'y' has %d observation(s); they must match.",
        .spcf_nrow(x), n))
    .spcf_check_finite(x, "x")
  }
  if (!is.null(offset)) {
    if (.spcf_nrow(offset) != n)
      .spcf_stop(sprintf("'offset' has length %d but 'y' has %d observation(s); they must match.",
                         .spcf_nrow(offset), n))
    .spcf_check_finite(offset, "offset")
  }
  if (!is.null(time)) {
    if (.spcf_nrow(time) != n)
      .spcf_stop(sprintf("'time' has length %d but 'y' has %d observation(s); they must match.",
                         .spcf_nrow(time), n))
    .spcf_check_finite(time, "time")
  }
  invisible(n)
}

## Holdout-validation controls shared by the cf_*_hv functions.
.spcf_check_hv_args <- function(n, train_rat, id_train, alpha, kernel,
                                add_learn = NULL) {
  if (!is.numeric(train_rat) || length(train_rat) != 1L ||
      !is.finite(train_rat) || train_rat <= 0 || train_rat > 1)
    .spcf_stop("'train_rat' must be a single number in (0, 1]; got ",
               paste(format(train_rat), collapse = ", "), ".")
  if (!is.null(id_train)) {
    if (!is.numeric(id_train) || anyNA(id_train))
      .spcf_stop("'id_train' must be a numeric vector of row indices into 'y' (no NAs).")
    if (any(id_train < 1) || any(id_train > n) || any(id_train != round(id_train)))
      .spcf_stop(sprintf("'id_train' must contain whole numbers between 1 and %d (the number of observations).", n))
    if (anyDuplicated(id_train))
      .spcf_stop("'id_train' contains duplicated indices; the training rows must be unique.")
    if (length(id_train) < 2L || length(id_train) >= n)
      .spcf_stop(sprintf("'id_train' selects %d of %d observations; leave at least one row for validation.",
                         length(id_train), n))
  }
  if (!is.numeric(alpha) || length(alpha) != 1L || !is.finite(alpha) ||
      alpha <= 0 || alpha > 1)
    .spcf_stop("'alpha' must be a single number in (0, 1]; got ",
               paste(format(alpha), collapse = ", "), ".")
  if (!is.character(kernel) || length(kernel) != 1L || !(kernel %in% c("exp", "gau")))
    .spcf_stop("'kernel' must be either \"exp\" (exponential) or \"gau\" (Gaussian); got ",
               paste(sQuote(kernel), collapse = ", "), ".")
  if (!is.null(add_learn)) {
    if (!is.character(add_learn) || length(add_learn) != 1L ||
        !(add_learn %in% c("none", "rf", "lightgbm")))
      .spcf_stop("'add_learn' must be one of \"none\", \"rf\" or \"lightgbm\"; got ",
                 paste(sQuote(add_learn), collapse = ", "), ".")
  }
  invisible(TRUE)
}

## mod_hv must be the holdout-validation object produced by the matching
## cf_*_hv function.
.spcf_check_mod_hv <- function(mod_hv, want, fun) {
  if (missing(mod_hv) || is.null(mod_hv))
    .spcf_stop(sprintf("'mod_hv' must be supplied: run %s() first and pass its result.", want))
  if (!inherits(mod_hv, want))
    .spcf_stop(sprintf(
      "'mod_hv' must be a %s object created by %s(), but an object of class %s was given.",
      sQuote(want), want, paste(sQuote(class(mod_hv)), collapse = "/")))
  if (is.null(mod_hv$other))
    .spcf_stop(sprintf("'mod_hv' looks corrupted (no internal state); re-run %s().", want))
  invisible(TRUE)
}

## Prediction-site arguments: x0 must line up with x, and with coords0.
.spcf_check_newdata <- function(x, x0, coords0, time0 = NULL, offset0 = NULL) {
  if (is.null(coords0)) {
    if (!is.null(x0))
      .spcf_stop("'x0' was supplied without 'coords0'; prediction sites need coordinates.")
    return(invisible(TRUE))
  }
  .spcf_check_coords(coords0, NULL, "coords0")
  n0 <- nrow(coords0)
  if (!is.null(x0)) {
    if (.spcf_nrow(x0) != n0)
      .spcf_stop(sprintf("'x0' has %d row(s) but 'coords0' has %d prediction site(s); they must match.",
                         .spcf_nrow(x0), n0))
    if (!is.null(x)) {
      k <- if (is.null(dim(x))) 1L else ncol(x)
      k0 <- if (is.null(dim(x0))) 1L else ncol(x0)
      if (k != k0)
        .spcf_stop(sprintf(
          "'x0' has %d column(s) but 'x' has %d; the prediction sites must carry the same covariates, in the same order.",
          k0, k))
      nmx <- if (is.null(dim(x))) NULL else colnames(x)
      nmx0 <- if (is.null(dim(x0))) NULL else colnames(x0)
      if (!is.null(nmx) && !is.null(nmx0) && !identical(nmx, nmx0))
        .spcf_stop(sprintf(
          "'x0' columns (%s) do not match 'x' columns (%s); they must be the same covariates, in the same order.",
          paste(nmx0, collapse = ", "), paste(nmx, collapse = ", ")))
    }
    .spcf_check_finite(x0, "x0")
  }
  if (!is.null(time0)) {
    if (.spcf_nrow(time0) != n0)
      .spcf_stop(sprintf("'time0' has length %d but 'coords0' has %d prediction site(s); they must match.",
                         .spcf_nrow(time0), n0))
    .spcf_check_finite(time0, "time0")
  }
  if (!is.null(offset0)) {
    if (.spcf_nrow(offset0) != n0)
      .spcf_stop(sprintf("'offset0' has length %d but 'coords0' has %d prediction site(s); they must match.",
                         .spcf_nrow(offset0), n0))
    .spcf_check_finite(offset0, "offset0")
  }
  invisible(TRUE)
}

## Downscaling: Y is aggregate-level, coords/x/prop_weight disaggregate-level,
## and agg_id maps the latter to the former. The internal code indexes
## aggregate-level vectors *by position* with agg_id (A[agg_id]), so agg_id
## must be the integers 1..length(Y).
.spcf_check_downscale <- function(Y, x, prop_weight, coords, agg_id) {
  if (is.null(Y)) .spcf_stop("'Y' (aggregate-level response) must be supplied.")
  .spcf_check_finite(Y, "Y")
  N <- length(as.vector(as.matrix(Y)))
  if (is.null(agg_id)) .spcf_stop("'agg_id' must be supplied.")
  .spcf_check_finite(agg_id, "agg_id")
  n <- length(agg_id)
  .spcf_check_coords(coords, n)
  if (!is.null(x)) {
    if (.spcf_nrow(x) != n)
      .spcf_stop(sprintf("'x' has %d row(s) but 'agg_id' has length %d; both are disaggregate-level and must match.",
                         .spcf_nrow(x), n))
    .spcf_check_finite(x, "x")
  }
  if (!is.null(prop_weight)) {
    if (.spcf_nrow(prop_weight) != n)
      .spcf_stop(sprintf("'prop_weight' has length %d but 'agg_id' has length %d; they must match.",
                         .spcf_nrow(prop_weight), n))
    .spcf_check_finite(prop_weight, "prop_weight")
    if (any(prop_weight < 0))
      .spcf_stop("'prop_weight' must be non-negative.")
  }
  if (any(agg_id != round(agg_id)))
    .spcf_stop("'agg_id' must contain whole numbers indexing 'Y' (1, 2, ..., length(Y)).")
  ua <- sort(unique(agg_id))
  if (!identical(as.numeric(ua), as.numeric(seq_len(N))))
    .spcf_stop(sprintf(
      "'agg_id' must take the values 1, 2, ..., %d (= length(Y)), each aggregate unit matching the corresponding element of 'Y'. Found %d distinct value(s) ranging %s..%s. Recode with agg_id <- match(agg_id, <the labels ordered as in Y>).",
      N, length(ua), format(min(ua)), format(max(ua))))
  invisible(list(N = N, n = n))
}
