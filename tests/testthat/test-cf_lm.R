d  <- sim_spatial(n = 120)
hv <- quiet(cf_lm_hv(y = d$y, x = d$x, coords = d$coords))

test_that("cf_lm_hv returns a usable holdout object", {
  expect_s3_class(hv, "cf_lm_hv")
  expect_true(is.numeric(hv$id_train) && length(hv$id_train) > 0)
  expect_true(all(hv$id_train %in% seq_along(d$y)))
  expect_output(print(hv))
})

test_that("cf_lm fits, predicts and reports finite quantities", {
  n0 <- 30
  m  <- quiet(cf_lm(y = d$y, x = d$x, coords = d$coords,
                    x0 = d$x[1:n0, ], coords0 = d$coords[1:n0, ], mod_hv = hv))
  expect_s3_class(m, "cf_lm")
  expect_true(all(is.finite(as.matrix(m$beta))))
  expect_true(all(is.finite(m$pred$pred)), )
  expect_true(all(m$pred$pred_sd > 0))
  expect_equal(nrow(m$pred), length(d$y))
  expect_equal(nrow(m$pred0), n0)
  expect_equal(nrow(m$pred0_q), n0)
  expect_true(all(is.finite(m$pred0$pred)))
  ## predictive quantiles must be ordered within every row
  expect_true(all(apply(as.matrix(m$pred_q), 1,
                        function(r) all(diff(r) >= -1e-8))))
  ## the fit must track the response
  expect_gt(cor(m$pred$pred, d$y), 0.7)
  expect_equal(length(m$bands), length(m$Z))
  ## sd_summary carries labels in `elements` and numbers in the SD column
  expect_type(m$sd_summary$standard_deviation, "double")
  expect_true(all(c("xb", "residuals") %in% m$sd_summary$elements))
  expect_output(print(m))
})

test_that("every se_type / se_method combination yields usable SEs", {
  ## NOTE: "prediction" is not always wider than "mean" -- the split-conformal
  ## factor may shrink an over-dispersed signal SD -- so only the properties
  ## that must always hold are asserted here.
  for (st in c("prediction", "mean")) {
    for (sm in c("opt", "classic")) {
      m <- quiet(cf_lm(y = d$y, x = d$x, coords = d$coords, mod_hv = hv,
                       se_type = st, se_method = sm))
      expect_true(all(is.finite(m$pred$pred_sd)), label = paste(st, sm))
      expect_true(all(m$pred$pred_sd > 0), label = paste(st, sm))
      expect_true(all(is.finite(m$beta[, 2])) && all(m$beta[, 2] > 0),
                  label = paste(st, sm))
    }
  }
  ## the signal predictive is kept alongside the observation predictive
  m_pred <- quiet(cf_lm(y = d$y, x = d$x, coords = d$coords, mod_hv = hv,
                        se_type = "prediction"))
  expect_false(is.null(m_pred$pred_signal))

  m_nai <- quiet(cf_lm(y = d$y, x = d$x, coords = d$coords, mod_hv = hv,
                       robust_se = FALSE))
  expect_true(all(is.finite(m_nai$beta[, 2])))
})

test_that("cf_lm works without covariates and with both kernels", {
  hv0 <- quiet(cf_lm_hv(y = d$y, coords = d$coords))
  m0  <- quiet(cf_lm(y = d$y, coords = d$coords, mod_hv = hv0))
  expect_true(all(is.finite(m0$pred$pred)))

  hvg <- quiet(cf_lm_hv(y = d$y, x = d$x, coords = d$coords, kernel = "gau"))
  mg  <- quiet(cf_lm(y = d$y, x = d$x, coords = d$coords, mod_hv = hvg))
  expect_true(all(is.finite(mg$pred$pred)))
})

test_that("id_train is honoured and the split is reproducible", {
  idt <- sort(sample(length(d$y), 90))
  h1  <- quiet(cf_lm_hv(y = d$y, x = d$x, coords = d$coords, id_train = idt))
  expect_equal(sort(h1$id_train), idt)
  h2  <- quiet(cf_lm_hv(y = d$y, x = d$x, coords = d$coords, seed = 42))
  h3  <- quiet(cf_lm_hv(y = d$y, x = d$x, coords = d$coords, seed = 42))
  expect_equal(h2$id_train, h3$id_train)
})
