d <- sim_spatial(n = 120, seed = 5)

test_that("cf_glm supports the Gaussian family with an offset", {
  off <- rep(log(1.5), length(d$y))
  hv  <- quiet(cf_glm_hv(y = d$y, x = d$x, coords = d$coords, offset = off))
  expect_s3_class(hv, "cf_glm_hv")
  m   <- quiet(cf_glm(y = d$y, x = d$x, coords = d$coords, offset = off,
                      x0 = d$x[1:20, ], coords0 = d$coords[1:20, ],
                      offset0 = off[1:20], mod_hv = hv))
  expect_s3_class(m, "cf_glm")
  expect_true(all(is.finite(as.matrix(m$beta))))
  expect_equal(nrow(m$pred0), 20)
  expect_true(all(is.finite(m$pred0$pred)))
  expect_output(print(m))
})

test_that("Poisson fits stay on the response scale", {
  set.seed(11)
  y <- rpois(length(d$field), exp(pmin(d$field, 2)))
  hv <- quiet(cf_glm_hv(y = y, x = d$x, coords = d$coords, family = poisson()))
  m  <- quiet(cf_glm(y = y, x = d$x, coords = d$coords, mod_hv = hv))
  expect_true(all(m$pred$pred >= 0))
  expect_true(all(is.finite(m$pred$pred_sd)))
  expect_true(all(apply(as.matrix(m$pred_q), 1,
                        function(r) all(diff(r) >= -1e-8))))
})

test_that("binomial fits return probabilities", {
  set.seed(12)
  y <- rbinom(length(d$field), 1, plogis(d$field - 1))
  hv <- quiet(cf_glm_hv(y = y, x = d$x, coords = d$coords, family = binomial()))
  m  <- quiet(cf_glm(y = y, x = d$x, coords = d$coords, mod_hv = hv))
  expect_true(all(m$pred$pred >= 0 & m$pred$pred <= 1))
  expect_true(all(is.finite(m$pred$pred_sd)))
})

test_that("both SE methods run and the naive SE is the smallest", {
  hv  <- quiet(cf_glm_hv(y = d$y, x = d$x, coords = d$coords))
  mo  <- quiet(cf_glm(y = d$y, x = d$x, coords = d$coords, mod_hv = hv,
                      se_method = "opt"))
  mc  <- quiet(cf_glm(y = d$y, x = d$x, coords = d$coords, mod_hv = hv,
                      se_method = "classic"))
  mn  <- quiet(cf_glm(y = d$y, x = d$x, coords = d$coords, mod_hv = hv,
                      robust_se = FALSE))
  expect_true(all(is.finite(mo$beta[, 2])))
  expect_lte(mean(mn$beta[, 2]), mean(mc$beta[, 2]) + 1e-8)
})

test_that("an offset enters the linear predictor as documented", {
  ## Poisson with a log link: doubling the exposure must double the prediction.
  set.seed(9)
  d   <- sim_spatial(n = 120, seed = 9)
  y   <- rpois(length(d$field), exp(pmin(d$field, 2)))
  off <- rep(log(1), length(y))
  hv  <- quiet(cf_glm_hv(y = y, x = d$x, coords = d$coords, offset = off,
                         family = poisson()))
  m1  <- quiet(cf_glm(y = y, x = d$x, coords = d$coords, offset = off,
                      x0 = d$x, coords0 = d$coords, offset0 = off, mod_hv = hv))
  m2  <- quiet(cf_glm(y = y, x = d$x, coords = d$coords, offset = off,
                      x0 = d$x, coords0 = d$coords, offset0 = off + log(2),
                      mod_hv = hv))
  expect_equal(m2$pred0$pred, 2 * m1$pred0$pred, tolerance = 1e-6)
})

test_that("the Gaussian kernel works for every family", {
  d <- sim_spatial(n = 120, seed = 10)
  for (fam in list(gaussian(), poisson(), binomial())) {
    yy <- switch(fam$family,
                 gaussian = d$y,
                 poisson  = rpois(length(d$field), exp(pmin(d$field, 2))),
                 binomial = rbinom(length(d$field), 1, plogis(d$field)))
    hv <- quiet(cf_glm_hv(y = yy, x = d$x, coords = d$coords, kernel = "gau",
                          family = fam))
    m  <- quiet(cf_glm(y = yy, x = d$x, coords = d$coords, mod_hv = hv))
    expect_true(all(is.finite(m$pred$pred)), label = fam$family)
  }
})

test_that("train_rat = 1 leaves no validation samples but still fits", {
  d  <- sim_spatial(n = 100, seed = 11)
  hv <- quiet(cf_lm_hv(y = d$y, x = d$x, coords = d$coords, train_rat = 1))
  expect_s3_class(hv, "cf_lm_hv")
  expect_equal(length(hv$id_train), length(d$y))
  m  <- quiet(cf_lm(y = d$y, x = d$x, coords = d$coords, mod_hv = hv))
  expect_true(all(is.finite(m$pred$pred)))
})

test_that("repeated coordinates are handled", {
  d  <- sim_spatial(n = 120, seed = 12)
  co <- d$coords; co[1:40, ] <- co[rep(41:60, 2), ]      # a third of the sites repeat
  hv <- quiet(cf_lm_hv(y = d$y, x = d$x, coords = co))
  m  <- quiet(cf_lm(y = d$y, x = d$x, coords = co, mod_hv = hv))
  expect_true(all(is.finite(m$pred$pred)))
  expect_true(all(is.finite(as.matrix(m$beta))))
})
