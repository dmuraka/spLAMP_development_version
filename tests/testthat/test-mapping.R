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

test_that("an irregular prediction grid rasterizes without dropping sites", {
  skip_if_no_map()
  ## clustered, non-lattice sites: terra::rast(type = "xyz") cannot take these,
  ## so .sp_raster() has to fall back on the nearest-neighbour fill
  set.seed(42)
  k   <- 12
  ctr <- cbind(stats::runif(k, 0, 10), stats::runif(k, 0, 10))
  j   <- sample(k, 1200, replace = TRUE)
  g0  <- cbind(ctr[j, 1] + stats::rnorm(1200, 0, 0.25),
               ctr[j, 2] + stats::rnorm(1200, 0, 0.25))
  expect_null(tryCatch(terra::rast(data.frame(x = g0[, 1], y = g0[, 2], z = 0),
                                   type = "xyz"), error = function(e) NULL))

  d  <- sim_spatial(n = 120)
  x0 <- matrix(stats::rnorm(nrow(g0)), nrow(g0), ncol(as.matrix(d$x)))
  colnames(x0) <- colnames(as.matrix(d$x))
  hv <- quiet(cf_lm_hv(y = d$y, x = d$x, coords = d$coords))
  m  <- quiet(cf_lm(y = d$y, x = d$x, x0 = x0, coords = d$coords,
                    coords0 = g0, mod_hv = hv))

  r <- spCF:::.sp_raster(m, "pred", "EPSG:4326")
  expect_s4_class(r, "SpatRaster")

  ## every site must land on a drawn (non-NA) cell -- the old rasterize(mean)
  ## fallback merged most of them away, and the marker rendering that replaced
  ## it piled thousands of circles on top of each other when zoomed out
  v  <- terra::values(r)[, 1]
  id <- terra::cellFromXY(r, g0)
  expect_false(any(is.na(id)))
  expect_true(all(!is.na(v[id])))

  ## the fill stops at the edge of the sampled region instead of flooding the
  ## whole bounding box, but still leaves a surface to look at (these 12 tight
  ## clusters sit in a mostly empty box, so most of it is legitimately blank)
  expect_gt(mean(is.na(v)), 0)
  expect_lt(mean(is.na(v)), 0.95)
})

test_that("no grid column is lost when leaflet reprojects the raster", {
  skip_if_no_map()
  ## A coarse lattice used to be handed to leaflet as-is. addRasterImage(project
  ## = TRUE) then reprojects it to Web Mercator with nearest-neighbour sampling
  ## at a resolution of its own choosing, which came out coarser than the source
  ## in x and dropped whole columns -- visible as a seam running the height of
  ## the mapped region.
  nx <- 25L; ny <- 21L
  g0 <- as.matrix(expand.grid(x = seq(6.0, 15.0, length.out = nx),
                              y = seq(47.0, 55.0, length.out = ny)))
  d  <- sim_spatial(n = 120)
  x0 <- matrix(stats::rnorm(nrow(g0)), nrow(g0), ncol(as.matrix(d$x)))
  colnames(x0) <- colnames(as.matrix(d$x))
  hv <- quiet(cf_lm_hv(y = d$y, x = d$x, coords = d$coords))
  m  <- quiet(cf_lm(y = d$y, x = d$x, x0 = x0, coords = d$coords,
                    coords0 = g0, mod_hv = hv))

  r <- spCF:::.sp_raster(m, "pred", "EPSG:4326")
  p <- terra::project(r, "EPSG:3857", method = "near")   # what leaflet does
  expect_equal(sum(is.na(terra::values(p))), 0)

  ## every column of the prediction grid must still be somewhere in the image
  src <- sort(unique(g0[, 1]))
  px  <- terra::xFromCol(p, seq_len(terra::ncol(p)))
  lon <- terra::project(cbind(px, terra::yFromRow(p, 1)),
                        "EPSG:3857", "EPSG:4326")[, 1]
  hit <- unique(FNN::get.knnx(matrix(src), matrix(lon), k = 1)$nn.index[, 1])
  expect_equal(sort(hit), seq_along(src))
})

test_that("a fit with no stored covariates refuses to draw a covariate effect", {
  skip_if_no_map()
  s  <- sim_spacetime(ns = 60, nt = 5)
  hv <- quiet(cf_dglm_hv(y = s$y, x = s$x, coords = s$coords, time = s$time))
  m  <- quiet(cf_dglm(y = s$y, x = s$x, coords = s$coords, time = s$time,
                      mod_hv = hv))
  expect_gt(nrow(m$beta), 1L)

  ## a real fit maps a covariate effect that actually varies
  expect_false(spCF:::.sp_xb_unavailable(m))
  d <- spCF:::.sp_xyz(m, "xb")
  expect_gt(diff(range(d$z)), 0)

  ## cf_dglm did not store x/x0 before spCF 0.2.1. Feeding a column of ones in
  ## their place used to return beta's row sum everywhere -- a flat surface that
  ## looks like a genuine covariate effect. It must come back empty instead.
  old <- m
  old$other$x <- NULL
  old$other$x0 <- NULL
  expect_true(spCF:::.sp_xb_unavailable(old))
  expect_null(spCF:::.sp_xyz(old, "xb"))
  expect_null(spCF:::.sp_raster(old, "xb", "EPSG:4326"))

  ## but a fit that never had covariates is intercept-only, not broken: it still
  ## maps, as the constant it is
  hv0 <- quiet(cf_dglm_hv(y = s$y, x = NULL, coords = s$coords, time = s$time))
  m0  <- quiet(cf_dglm(y = s$y, x = NULL, coords = s$coords, time = s$time,
                       mod_hv = hv0))
  expect_equal(nrow(m0$beta), 1L)
  expect_false(spCF:::.sp_xb_unavailable(m0))
  d0 <- spCF:::.sp_xyz(m0, "xb")
  expect_true(all(is.finite(d0$z)))
  expect_equal(diff(range(d0$z)), 0)
})
