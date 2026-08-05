test_that("sp_scalewise splits a spatial fit into scales", {
  d  <- sim_spatial(n = 120)
  hv <- quiet(cf_lm_hv(y = d$y, x = d$x, coords = d$coords))
  m  <- quiet(cf_lm(y = d$y, x = d$x, coords = d$coords,
                    x0 = d$x[1:20, ], coords0 = d$coords[1:20, ], mod_hv = hv))

  all_sc <- sp_scalewise(m)
  expect_equal(nrow(all_sc$pred), length(d$y))
  expect_equal(nrow(all_sc$pred0), 20)
  expect_true(all(is.finite(all_sc$pred$pred)))
  expect_true(all(all_sc$pred$pred_sd >= 0))

  ## contiguous, non-overlapping bandwidth ranges partition the full process
  cut   <- stats::median(m$bands)
  lo    <- sp_scalewise(m, bw_range = c(0, cut))
  hi    <- sp_scalewise(m, bw_range = c(cut, Inf))
  expect_equal(lo$pred$pred + hi$pred$pred, all_sc$pred$pred, tolerance = 1e-8)

  ## a range containing no scale is an error naming the fitted range
  expect_error(sp_scalewise(m, bw_range = c(1e12, Inf)), "bw_range")
  ## bw_range is half-open, so an empty interval is rejected
  expect_error(sp_scalewise(m, bw_range = c(1, 1)), "bw_range\\[1\\] <")
})

test_that("time_range applies to cf_dglm fits and warns for spatial fits", {
  s  <- sim_spacetime(ns = 40, nt = 4)
  hv <- quiet(cf_dglm_hv(y = s$y, x = s$x, coords = s$coords, time = s$time))
  m  <- quiet(cf_dglm(y = s$y, x = s$x, coords = s$coords, time = s$time,
                      mod_hv = hv))

  sc <- sp_scalewise(m)
  expect_equal(nrow(sc$pred), s$ns)          # one row per unique location
  expect_true(all(is.finite(sc$pred$pred)))

  sc2 <- sp_scalewise(m, time_range = c(2, 3))
  expect_true(all(sc2$pred$n_time <= 2))

  ## a single time point is a closed range, c(t, t) -- used in ?cf_dglm
  sc1 <- sp_scalewise(m, time_range = c(2, 2))
  expect_true(all(sc1$pred$n_time == 1))
  expect_true(all(is.finite(sc1$pred$pred)))

  ## an empty time window names the site set it applies to
  expect_error(sp_scalewise(m, time_range = c(100, 200)), "sample sites")

  ## a window that keeps sample sites but no prediction site warns and drops
  ## pred0, instead of failing outright
  m0 <- quiet(cf_dglm(y = s$y, x = s$x, coords = s$coords, time = s$time,
                      x0 = s$x[seq_len(s$ns), , drop = FALSE],
                      coords0 = s$coords_uni, time0 = rep(s$nt, s$ns),
                      mod_hv = hv))
  expect_warning(sc3 <- sp_scalewise(m0, time_range = c(1, 2)),
                 "prediction sites")
  expect_null(sc3$pred0)
  expect_true(all(is.finite(sc3$pred$pred)))

  d   <- sim_spatial(n = 60)
  hvs <- quiet(cf_lm_hv(y = d$y, x = d$x, coords = d$coords))
  ms  <- quiet(cf_lm(y = d$y, x = d$x, coords = d$coords, mod_hv = hvs))
  expect_warning(sp_scalewise(ms, time_range = c(1, 5)), "time_range is ignored")
})
