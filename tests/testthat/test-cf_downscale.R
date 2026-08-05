a <- sim_areal(n = 150, n_area = 15)

test_that("cf_downscale reproduces the areal totals and stays non-negative", {
  hv <- quiet(cf_downscale_hv(Y = a$Y, x = a$x, coords = a$coords,
                              agg_id = a$agg_id))
  expect_s3_class(hv, "cf_downscale_hv")
  expect_output(print(hv))

  m <- quiet(cf_downscale(Y = a$Y, x = a$x, coords = a$coords,
                          agg_id = a$agg_id, mod_hv = hv))
  expect_s3_class(m, "cf_downscale")
  expect_equal(nrow(m$pred), nrow(a$coords))
  expect_true(all(is.finite(m$pred$pred)))
  expect_true(all(m$pred$pred >= 0))

  ## pycnophylactic property: predictions re-aggregate to the observed totals
  agg <- as.numeric(tapply(m$pred$pred, a$agg_id, sum))
  expect_equal(agg, a$Y, tolerance = 1e-6)

  ## the disaggregated surface must track the (unobserved) fine-scale truth
  expect_gt(cor(m$pred$pred, a$fine), 0.5)
  expect_output(print(m))
})

test_that("adj = FALSE drops the pycnophylactic adjustment", {
  hv <- quiet(cf_downscale_hv(Y = a$Y, x = a$x, coords = a$coords,
                              agg_id = a$agg_id))
  m  <- quiet(cf_downscale(Y = a$Y, x = a$x, coords = a$coords,
                           agg_id = a$agg_id, mod_hv = hv, adj = FALSE))
  expect_true(all(is.finite(m$pred$pred)))
})

test_that("Y_type = 'mean' and prop_weight are supported", {
  Ym <- as.numeric(tapply(a$fine, a$agg_id, mean))
  hv <- quiet(cf_downscale_hv(Y = Ym, Y_type = "mean", x = a$x,
                              coords = a$coords, agg_id = a$agg_id))
  m  <- quiet(cf_downscale(Y = Ym, x = a$x, coords = a$coords,
                           agg_id = a$agg_id, mod_hv = hv))
  expect_true(all(is.finite(m$pred$pred)))

  set.seed(31)
  w  <- runif(nrow(a$coords), 0.5, 2)
  hvw <- quiet(cf_downscale_hv(Y = a$Y, x = a$x, prop_weight = w,
                               coords = a$coords, agg_id = a$agg_id))
  expect_s3_class(hvw, "cf_downscale_hv")
})
