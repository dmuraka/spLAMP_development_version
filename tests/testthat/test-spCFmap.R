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

test_that("a terra that cannot resolve a CRS is reported up front", {
  skip_if_not_installed("shiny")
  skip_if_not_installed("terra")

  ## a healthy terra must not trip the guard (otherwise spCFmap() would refuse
  ## to open on a perfectly good installation)
  skip_if_not(is.null(spCF:::.sp_crs_failure()),
              "terra cannot resolve EPSG:4326 in this environment")
  expect_true(spCF:::.sp_check_crs())

  ## and a broken one stops with an actionable message rather than failing later
  ## inside a Shiny observer
  local_mocked_bindings(.sp_crs_failure = function() "[rast] empty srs")
  expect_error(spCF:::.sp_check_crs("spCFmap()"),
               "cannot create a coordinate reference system")
  expect_error(spCF:::.sp_check_crs("spCFmap()"), 'install.packages\\("terra"\\)')
  expect_error(spCF:::.sp_check_crs("spCFmap()"), "\\[rast\\] empty srs")

  d  <- sim_spatial(n = 60)
  hv <- quiet(cf_lm_hv(y = d$y, x = d$x, coords = d$coords))
  m  <- quiet(cf_lm(y = d$y, x = d$x, coords = d$coords, mod_hv = hv))
  expect_error(spCFmap(m, crs = 4326, launch = FALSE), "PROJ")
})
