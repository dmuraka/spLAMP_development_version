## Small, deterministic data sets shared by the tests. Kept intentionally tiny
## so the whole suite stays well inside the CRAN check time budget.

sim_spatial <- function(n = 120, seed = 1) {
  set.seed(seed)
  coords <- data.frame(px = runif(n), py = runif(n))
  x      <- data.frame(v1 = rnorm(n), v2 = runif(n))
  field  <- 1.5 * sin(3 * coords$px) + cos(3 * coords$py)
  list(coords = coords, x = x, field = field,
       y = 0.5 * x$v1 + field + rnorm(n, sd = 0.3))
}

sim_spacetime <- function(ns = 60, nt = 5, seed = 2) {
  set.seed(seed)
  cs     <- data.frame(px = runif(ns), py = runif(ns))
  field  <- 1.5 * sin(3 * cs$px) + cos(3 * cs$py)
  tt     <- rep(seq_len(nt), each = ns)
  x      <- data.frame(v1 = rnorm(ns * nt))
  coords <- cs[rep(seq_len(ns), nt), ]
  rownames(coords) <- NULL
  list(coords_uni = cs, coords = coords, time = tt, x = x, field = field,
       y  = rep(field, nt) + 0.5 * x$v1 + 0.3 * tt + rnorm(ns * nt, sd = 0.3),
       ns = ns, nt = nt)
}

sim_areal <- function(n = 150, n_area = 15, seed = 3) {
  set.seed(seed)
  coords <- data.frame(px = runif(n), py = runif(n))
  x      <- data.frame(v1 = rnorm(n))
  agg_id <- rep(seq_len(n_area), length.out = n)
  fine   <- exp(0.5 + 0.3 * x$v1 + sin(3 * coords$px))
  list(coords = coords, x = x, agg_id = agg_id, fine = fine,
       Y = as.numeric(tapply(fine, agg_id, sum)))
}

## The cf_* functions report progress with message(); keep the test output quiet.
quiet <- function(expr) {
  invisible(utils::capture.output(
    out <- suppressMessages(suppressWarnings(force(expr)))))
  out
}
