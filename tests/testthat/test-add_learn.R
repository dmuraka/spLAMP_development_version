## The additional learners are optional (Suggests). Each test skips when its
## package is absent, and the heavier lightgbm fit is skipped on CRAN.

test_that("add_learn = 'rf' trains and predicts", {
  skip_if_not_installed("ranger")
  d  <- sim_spatial(n = 120)
  hv <- quiet(cf_lm_hv(y = d$y, x = d$x, coords = d$coords, add_learn = "rf"))
  m  <- quiet(cf_lm(y = d$y, x = d$x, coords = d$coords,
                    x0 = d$x[1:20, ], coords0 = d$coords[1:20, ], mod_hv = hv))
  expect_true(all(is.finite(m$pred$pred)))
  expect_true(all(is.finite(m$pred0$pred)))
  expect_true(all(apply(as.matrix(m$pred_q), 1,
                        function(r) all(diff(r) >= -1e-8))))
})

test_that("add_learn = 'lightgbm' trains, predicts and labels sd_summary", {
  skip_on_cran()
  skip_if_not_installed("lightgbm")
  ## strongly non-linear covariate effect, so the extra learner is committed
  set.seed(2)
  n      <- 300
  coords <- data.frame(px = runif(n), py = runif(n))
  x      <- data.frame(v1 = runif(n, -3, 3), v2 = runif(n, -3, 3))
  y      <- 3 * sin(2 * x$v1) * cos(2 * x$v2) + rnorm(n, sd = 0.1)

  hv <- quiet(cf_lm_hv(y = y, x = x, coords = coords, add_learn = "lightgbm"))
  m  <- quiet(cf_lm(y = y, x = x, coords = coords, x0 = x[1:20, ],
                    coords0 = coords[1:20, ], mod_hv = hv))
  expect_true(all(is.finite(m$pred$pred)))
  expect_true(all(is.finite(m$pred0$pred)))
  expect_true(all(apply(as.matrix(m$pred0_q), 1,
                        function(r) all(diff(r) >= -1e-8))))
  ## when the learner is committed, its SD is a number in the SD column and its
  ## label a string in `elements` (regression test for a swapped-columns bug)
  if (isTRUE(hv$other$a_mod0$a_run)) {
    expect_true(any(grepl("additional learning", m$sd_summary$elements)))
    expect_type(m$sd_summary$standard_deviation, "double")
  }
})

test_that("a missing learner package is reported by name", {
  ## add_mod() guards both learners with requireNamespace(); simulate absence
  skip_if(requireNamespace("lightgbm", quietly = TRUE),
          "lightgbm is installed, so the guard cannot be exercised")
  d <- sim_spatial(n = 60)
  expect_error(quiet(cf_lm_hv(y = d$y, x = d$x, coords = d$coords,
                              add_learn = "lightgbm")), "lightgbm")
})
