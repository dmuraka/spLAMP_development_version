## The internal mapping engine behind spCFmap(): every model type must produce
## drawable layers, with or without prediction sites.

skip_if_no_map <- function() {
  skip_on_cran()
  for (p in c("sf", "terra", "shiny")) skip_if_not_installed(p)
}

layers_ok <- function(mod, crs) {
  vapply(c("pred", "pred_sd", "xb", "scale"), function(l) {
    d <- spCF:::.sp_xyz(mod, l)
    is.data.frame(d) && nrow(d) > 0 && all(is.finite(d$z))
  }, logical(1))
}

test_that("a cf_lm fit is mappable with and without prediction sites", {
  skip_if_no_map()
  d  <- sim_spatial(n = 100)
  hv <- quiet(cf_lm_hv(y = d$y, x = d$x, coords = d$coords))
  m1 <- quiet(cf_lm(y = d$y, x = d$x, coords = d$coords,
                    x0 = d$x[1:20, ], coords0 = d$coords[1:20, ], mod_hv = hv))
  m0 <- quiet(cf_lm(y = d$y, x = d$x, coords = d$coords, mod_hv = hv))

  expect_true(all(layers_ok(m1, 4326)))
  ## without coords0/x0 the map falls back to the sample sites
  expect_true(all(layers_ok(m0, 4326)))
  expect_equal(nrow(spCF:::.sp_xyz(m1, "pred")), 20)
  expect_equal(nrow(spCF:::.sp_xyz(m0, "pred")), length(d$y))

  ## the CSV export follows the same choice of sites
  expect_equal(nrow(spCF:::.sp_export(m1)), 20)
  expect_equal(nrow(spCF:::.sp_export(m0)), length(d$y))
  expect_true(all(c("xcoord", "ycoord", "pred", "pred_sd") %in%
                    names(spCF:::.sp_export(m0))))
})

test_that("cf_dglm and cf_downscale fits are mappable", {
  skip_if_no_map()
  s  <- sim_spacetime(ns = 40, nt = 4)
  hv <- quiet(cf_dglm_hv(y = s$y, x = s$x, coords = s$coords, time = s$time))
  md <- quiet(cf_dglm(y = s$y, x = s$x, coords = s$coords, time = s$time,
                      mod_hv = hv))
  expect_true(all(layers_ok(md, 4326)))
  ## one row per location after the time average
  expect_equal(nrow(spCF:::.sp_xyz(md, "pred")), s$ns)

  a   <- sim_areal(n = 120, n_area = 12)
  hvd <- quiet(cf_downscale_hv(Y = a$Y, x = a$x, coords = a$coords,
                               agg_id = a$agg_id))
  ms  <- quiet(cf_downscale(Y = a$Y, x = a$x, coords = a$coords,
                            agg_id = a$agg_id, mod_hv = hvd))
  expect_true(all(layers_ok(ms, 4326)))
  expect_equal(nrow(spCF:::.sp_xyz(ms, "pred")), nrow(a$coords))
})

test_that("the drawn layers rasterise and carry a CRS", {
  skip_if_no_map()
  skip_if_not_installed("leaflet")
  d  <- sim_spatial(n = 100)
  hv <- quiet(cf_lm_hv(y = d$y, x = d$x, coords = d$coords))
  m  <- quiet(cf_lm(y = d$y, x = d$x, coords = d$coords, mod_hv = hv))

  crs_str <- spCF:::.sp_crs(4326)
  expect_equal(crs_str, "EPSG:4326")
  expect_s4_class(spCF:::.sp_raster(m, "pred", crs_str), "SpatRaster")
  pts <- spCF:::.sp_points(m, crs_str)
  expect_s3_class(pts, "sf")
  expect_equal(sf::st_crs(pts)$epsg, 4326L)      # always returned in lon/lat
  bb <- spCF:::.sp_bbox4326(spCF:::.sp_view_coords(m), crs_str)
  expect_length(bb, 4)
  expect_true(all(is.finite(bb)))
})
