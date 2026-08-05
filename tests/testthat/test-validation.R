## Argument checks: a malformed input must fail early with a message that names
## the offending argument, rather than surfacing from the linear algebra layer.

d <- sim_spatial(n = 60)

test_that("cf_lm_hv rejects mismatched lengths and non-finite values", {
  expect_error(cf_lm_hv(y = d$y[-1], x = d$x, coords = d$coords), "coords")
  expect_error(cf_lm_hv(y = replace(d$y, 1, NA), x = d$x, coords = d$coords),
               "missing value")
  expect_error(cf_lm_hv(y = replace(d$y, 1, Inf), x = d$x, coords = d$coords),
               "infinite")
  expect_error(cf_lm_hv(y = d$y, x = d$x[-1, ], coords = d$coords), "'x' has")
  expect_error(cf_lm_hv(y = d$y, x = d$x, coords = replace(d$coords, 1, NA)),
               "missing value")
  expect_error(cf_lm_hv(y = d$y, x = d$x, coords = d$coords[, 1, drop = FALSE]),
               "2 columns")
})

test_that("cf_lm_hv rejects out-of-range holdout controls", {
  expect_error(cf_lm_hv(y = d$y, x = d$x, coords = d$coords, train_rat = 1.5),
               "train_rat")
  expect_error(cf_lm_hv(y = d$y, x = d$x, coords = d$coords, alpha = 0),
               "alpha")
  expect_error(cf_lm_hv(y = d$y, x = d$x, coords = d$coords, kernel = "epan"),
               "kernel")
  expect_error(cf_lm_hv(y = d$y, x = d$x, coords = d$coords, add_learn = "xgboost"),
               "add_learn")
  expect_error(cf_lm_hv(y = d$y, x = d$x, coords = d$coords,
                        id_train = c(1, 1, 2)), "duplicated")
  expect_error(cf_lm_hv(y = d$y, x = d$x, coords = d$coords,
                        id_train = c(1, 999)), "between 1 and")
})

test_that("non-numeric covariates are reported by column", {
  xf <- d$x
  xf$grp <- factor(sample(letters[1:3], nrow(xf), replace = TRUE))
  expect_error(cf_lm_hv(y = d$y, x = xf, coords = d$coords), "grp")
})

test_that("cf_lm checks mod_hv and the prediction-site arguments", {
  hv <- quiet(cf_lm_hv(y = d$y, x = d$x, coords = d$coords))
  expect_error(cf_lm(y = d$y, x = d$x, coords = d$coords, mod_hv = list()),
               "cf_lm_hv")
  expect_error(cf_lm(y = d$y, x = d$x, coords = d$coords, mod_hv = hv,
                     x0 = d$x[1:10, 1, drop = FALSE], coords0 = d$coords[1:10, ]),
               "same covariates|column")
  expect_error(cf_lm(y = d$y, x = d$x, coords = d$coords, mod_hv = hv,
                     x0 = d$x[1:10, ], coords0 = d$coords[1:5, ]),
               "must match")
})

test_that("cf_dglm_hv checks the time argument", {
  s <- sim_spacetime(ns = 20, nt = 3)
  expect_error(cf_dglm_hv(y = s$y, x = s$x, coords = s$coords, time = s$time[-1]),
               "'time' has length")
  expect_error(cf_dglm_hv(y = s$y, x = s$x, coords = s$coords,
                          time = replace(s$time, 1, NA)), "missing value")
})

test_that("cf_downscale_hv enforces the agg_id contract", {
  a <- sim_areal(n = 60, n_area = 6)
  expect_error(cf_downscale_hv(Y = a$Y, x = a$x, coords = a$coords,
                               agg_id = a$agg_id + 100L), "agg_id")
  expect_error(cf_downscale_hv(Y = a$Y, x = a$x, coords = a$coords,
                               agg_id = a$agg_id[-1]), "row")
  expect_error(cf_downscale_hv(Y = a$Y, Y_type = "median", x = a$x,
                               coords = a$coords, agg_id = a$agg_id), "Y_type")
})

test_that("sp_scalewise rejects objects that are not spCF fits", {
  expect_error(sp_scalewise(list()), "cf_lm")
  expect_error(sp_scalewise(structure(list(), class = "cf_downscale")),
               "cf_downscale")
})
