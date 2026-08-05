## Time-varying coefficients (cf_dglm_hv(tvc = ), mod$beta_tv)

## Panel with a known coefficient path: y = 1 + beta_t * x1 + f(s) + e
sim_tvc <- function(seed = 1, ns = 40, nt = 20, varying = TRUE) {
  set.seed(seed)
  cs  <- data.frame(px = runif(ns), py = runif(ns))
  fld <- 1.5 * sin(3 * cs$px) + cos(3 * cs$py)
  tt  <- rep(seq_len(nt), each = ns)
  co  <- cs[rep(seq_len(ns), nt), ]; rownames(co) <- NULL
  x1  <- rnorm(ns * nt)
  bt  <- if (varying) 2 + sin(2 * pi * seq_len(nt) / nt) else rep(2, nt)
  list(y = 1 + bt[tt] * x1 + rep(fld, nt) + rnorm(ns * nt, sd = 0.4),
       x = data.frame(x1 = x1), coords = co, time = tt, beta_t = bt, nt = nt)
}

fit_tvc <- function(s, tvc = "x1", q_tvc = NULL) {
  hv <- quiet(cf_dglm_hv(y = s$y, x = s$x, coords = s$coords, time = s$time,
                         tvc = tvc, q_tvc = q_tvc))
  list(hv = hv,
       m = quiet(cf_dglm(y = s$y, x = s$x, coords = s$coords, time = s$time,
                         mod_hv = hv)))
}

test_that("beta_tv has one row per time point and excludes the intercept", {
  s <- sim_tvc()
  m <- fit_tvc(s)$m
  expect_s3_class(m$beta_tv, "data.frame")
  expect_equal(nrow(m$beta_tv), s$nt)
  expect_equal(names(m$beta_tv), c("x1", "time"))
  expect_equal(as.numeric(m$beta_tv$time), as.numeric(seq_len(s$nt)))
  expect_equal(dim(m$beta_tv_sd), dim(m$beta_tv))
  expect_true(all(is.finite(m$beta_tv$x1)), all(m$beta_tv_sd$x1 > 0))
  ## the constant coefficients still carry the intercept
  expect_true("Intercept" %in% rownames(m$beta))
  expect_false(any(grepl("Intercept", names(m$beta_tv))))
})

test_that("a time-varying coefficient is recovered, a constant one stays flat", {
  s <- sim_tvc(varying = TRUE)
  m <- fit_tvc(s)$m
  expect_gt(cor(m$beta_tv$x1, s$beta_t), 0.9)
  expect_lt(sqrt(mean((m$beta_tv$x1 - s$beta_t)^2)), 0.25)
  ## the estimated path has roughly the amplitude of the true one
  expect_gt(sd(m$beta_tv$x1), 0.5 * sd(s$beta_t))

  s0 <- sim_tvc(varying = FALSE)
  m0 <- fit_tvc(s0)$m
  expect_lt(sd(m0$beta_tv$x1), 0.2)            # essentially constant
  expect_lt(abs(mean(m0$beta_tv$x1) - 2), 0.2) # and centred on the truth
})

test_that("tvc accepts a column index and refuses the intercept", {
  s  <- sim_tvc()
  mi <- fit_tvc(s, tvc = 1L)$m
  expect_equal(names(mi$beta_tv), c("x1", "time"))
  expect_gt(cor(mi$beta_tv$x1, s$beta_t), 0.9)

  hv <- quiet(cf_dglm_hv(y = s$y, x = s$x, coords = s$coords, time = s$time,
                         tvc = "Intercept"))
  expect_length(hv$other$tv_cols, 0)
})

test_that("q_tvc fixes the drift instead of being re-estimated", {
  s <- sim_tvc()
  f_free  <- fit_tvc(s)
  f_stiff <- fit_tvc(s, q_tvc = 1e-4)
  f_loose <- fit_tvc(s, q_tvc = 1)

  ## the supplied value must survive into the fit cf_dglm() reuses
  expect_equal(f_stiff$hv$other$q_tvc[1], 1e-4)
  expect_equal(f_loose$hv$other$q_tvc[1], 1)
  expect_false(isTRUE(all.equal(f_free$hv$other$q_tvc[1], 1e-4)))

  ## and it must act as a smoothness prior: a small drift flattens the path
  expect_lt(sd(f_stiff$m$beta_tv$x1), sd(f_free$m$beta_tv$x1))
  expect_error(quiet(cf_dglm_hv(y = s$y, x = s$x, coords = s$coords,
                                time = s$time, tvc = "x1", q_tvc = -1)), "q_tvc")
})

test_that("beta_tv_sd scales with the residual variance", {
  ## The observation covariance carries the working-residual dispersion, so the
  ## reported SD must grow with the noise instead of staying put (it used to be
  ## almost independent of it: too wide for a quiet response, too narrow -- the
  ## unsafe direction -- for a noisy one).
  quiet_fit <- function(sd_e, seed = 3) {
    set.seed(seed); ns <- 40; nt <- 20
    cs  <- data.frame(px = runif(ns), py = runif(ns))
    fld <- 1.5 * sin(3 * cs$px) + cos(3 * cs$py)
    tt  <- rep(seq_len(nt), each = ns)
    co  <- cs[rep(seq_len(ns), nt), ]; rownames(co) <- NULL
    x1  <- rnorm(ns * nt); bt <- 2 + sin(2 * pi * seq_len(nt) / nt)
    y   <- 1 + bt[tt] * x1 + rep(fld, nt) + rnorm(ns * nt, sd = sd_e)
    hv  <- quiet(cf_dglm_hv(y = y, x = data.frame(x1 = x1), coords = co,
                            time = tt, tvc = "x1"))
    quiet(cf_dglm(y = y, x = data.frame(x1 = x1), coords = co, time = tt,
                  mod_hv = hv))
  }
  s_low  <- mean(quiet_fit(0.3)$beta_tv_sd$x1)
  s_high <- mean(quiet_fit(2.0)$beta_tv_sd$x1)
  expect_gt(s_high, 2 * s_low)
  expect_true(all(is.finite(c(s_low, s_high))))
})
