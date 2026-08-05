## spCFmap() only builds the app object here (launch = FALSE); no browser is
## opened and no Shiny server is started.

test_that("spCFmap(mod, crs) builds an app for a fitted model", {
  skip_on_cran()
  skip_if_not_installed("shiny")
  skip_if_not_installed("leaflet")
  skip_if_not_installed("terra")
  skip_if_not_installed("sf")

  d  <- sim_spatial(n = 80)
  hv <- quiet(cf_lm_hv(y = d$y, x = d$x, coords = d$coords))
  m  <- quiet(cf_lm(y = d$y, x = d$x, coords = d$coords, mod_hv = hv))

  app <- spCFmap(m, crs = 4326, launch = FALSE)
  expect_s3_class(app, "shiny.appobj")
})

test_that("the full app object can be built", {
  skip_on_cran()
  skip_if_not_installed("shiny")
  skip_if_not_installed("bslib")
  skip_if_not_installed("leaflet")
  skip_if_not_installed("terra")
  skip_if_not_installed("sf")
  skip_if_not_installed("sp")

  expect_s3_class(spCFmap(launch = FALSE), "shiny.appobj")
})

test_that("the two modes validate their arguments", {
  skip_if_not_installed("shiny")
  d  <- sim_spatial(n = 60)
  hv <- quiet(cf_lm_hv(y = d$y, x = d$x, coords = d$coords))
  m  <- quiet(cf_lm(y = d$y, x = d$x, coords = d$coords, mod_hv = hv))

  expect_error(spCFmap(m, launch = FALSE), "crs")
  expect_warning(spCFmap(crs = 4326, launch = FALSE), "ignored")
})
