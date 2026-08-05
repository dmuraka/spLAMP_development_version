s <- sim_spacetime(ns = 50, nt = 4)

test_that("cf_dglm fits a space-time model and predicts at new sites", {
  hv <- quiet(cf_dglm_hv(y = s$y, x = s$x, coords = s$coords, time = s$time))
  expect_s3_class(hv, "cf_dglm_hv")
  expect_output(print(hv))

  m <- quiet(cf_dglm(y = s$y, x = s$x, coords = s$coords, time = s$time,
                     x0 = s$x[seq_len(s$ns), , drop = FALSE],
                     coords0 = s$coords_uni, time0 = rep(s$nt, s$ns),
                     mod_hv = hv))
  expect_s3_class(m, "cf_dglm")
  expect_true(all(is.finite(as.matrix(m$beta))))
  expect_equal(nrow(m$pred), length(s$y))
  expect_equal(nrow(m$pred0), s$ns)
  expect_true(all(is.finite(m$pred0$pred)))
  expect_gt(cor(m$pred$pred, s$y), 0.5)
  expect_output(print(m))
})

test_that("coords0 without time0 is refused", {
  hv <- quiet(cf_dglm_hv(y = s$y, x = s$x, coords = s$coords, time = s$time))
  expect_error(quiet(cf_dglm(y = s$y, x = s$x, coords = s$coords, time = s$time,
                             x0 = s$x[seq_len(s$ns), , drop = FALSE],
                             coords0 = s$coords_uni, mod_hv = hv)),
               "time0")
})

test_that("Poisson space-time fits stay non-negative", {
  set.seed(21)
  y <- rpois(length(s$y), exp(pmin(rep(s$field, s$nt), 2)))
  hv <- quiet(cf_dglm_hv(y = y, x = s$x, coords = s$coords, time = s$time,
                         family = poisson()))
  m  <- quiet(cf_dglm(y = y, x = s$x, coords = s$coords, time = s$time,
                      mod_hv = hv))
  expect_true(all(m$pred$pred >= 0))
})

test_that("fixed rho/Q and time-varying coefficients are accepted", {
  hv_fixed <- quiet(cf_dglm_hv(y = s$y, x = s$x, coords = s$coords,
                               time = s$time, rho = 0.8, Q = 1))
  expect_s3_class(hv_fixed, "cf_dglm_hv")
  hv_tvc <- quiet(cf_dglm_hv(y = s$y, x = s$x, coords = s$coords,
                             time = s$time, tvc = "v1"))
  expect_s3_class(hv_tvc, "cf_dglm_hv")
})
