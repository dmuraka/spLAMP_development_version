## sp_map_core - shared leaflet mapping core for fitted spCF models ----------
##
## One rendering engine used by BOTH:
##   * spCFmap(mod, crs)  -> standalone interactive map (call from RStudio)
##   * app.R              -> full app sources this and reuses the module
##
## All internal; the only exported entry point is spCFmap() (R/spCFmap.R).
##   sp_map_app(mod, crs)                   shiny.appobj for one fitted model
##   sp_map_controls(id)                    sidebar controls (Outputs + Display)
##   sp_map_view(id, height)                the leaflet output
##   sp_map_server(id, mod, crs, preview)   module server (mod/crs/preview reactives)
##
## Requires: shiny, leaflet, terra, sf (+ spCF for the "scale" layer).

.sp_titles <- c(pred = "Predictive mean", pred_sd = "Predictive SD",
                xb = "Covariate effect", scale = "Scale-wise component")

.sp_fmt <- function(x) formatC(x, format = "g", digits = 4)   # <=4 significant figures

## Basemap tiles as explicit URL templates, all of them free of an API key.
##
## Neither addProviderTiles("CartoDB.Positron") nor CARTO's own CDN works here
## any more: basemaps.cartocdn.com now serves every keyless request as a tile
## with a diagonal "API KEY REQUIRED" watermark stamped across it (a normal 200
## PNG, so nothing in the app can detect it). Esri's light/dark gray canvases
## are the closest keyless equivalents to Positron/DarkMatter, so they stand in
## as the light and dark basemaps. Check a tile by eye, not by status code,
## before adding a provider here.
.sp_basemaps <- list(
  light = list(url = paste0("https://server.arcgisonline.com/ArcGIS/rest/",
               "services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}"),
               attr = "Tiles \u00a9 Esri", sub = ""),
  dark  = list(url = paste0("https://server.arcgisonline.com/ArcGIS/rest/",
               "services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}"),
               attr = "Tiles \u00a9 Esri", sub = ""),
  osm   = list(url = "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
               attr = "\u00a9 OpenStreetMap contributors", sub = "abc"),
  sat   = list(url = paste0("https://server.arcgisonline.com/ArcGIS/rest/",
               "services/World_Imagery/MapServer/tile/{z}/{y}/{x}"),
               attr = "Tiles \u00a9 Esri", sub = ""),
  topo  = list(url = paste0("https://server.arcgisonline.com/ArcGIS/rest/",
               "services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}"),
               attr = "Tiles \u00a9 Esri", sub = ""))

## The basemap the map opens on. Used by BOTH the selector's initial value and
## the first render: the observer that redraws on a change is ignoreInit, so if
## these two drifted apart the map would open showing one basemap while the
## dropdown named another.
.sp_basemap_default <- "sat"

.sp_add_base <- function(map, key, group = "base") {
  b <- .sp_basemaps[[key]]; if (is.null(b)) b <- .sp_basemaps[["light"]]
  opt <- if (nzchar(b$sub)) leaflet::tileOptions(subdomains = b$sub)
         else leaflet::tileOptions()
  leaflet::addTiles(map, urlTemplate = b$url, attribution = b$attr,
                    group = group, options = opt)
}

.sp_crs <- function(crs) {
  if (is.null(crs)) return(NA_character_)
  if (is.numeric(crs) || grepl("^[0-9]+$", crs))
    paste0("EPSG:", gsub("\\D", "", crs)) else as.character(crs)
}

## Every map layer is built by giving terra a CRS (.sp_raster/.sp_sf_points), so
## a terra whose PROJ database (proj.db) is missing or unreadable cannot draw
## anything. terra then fails deep inside a Shiny observer, which tears the
## session down behind an opaque "An error has occurred" screen. Probe the one
## operation that matters once, up front, so the user gets an actionable message
## instead. Returns NULL when terra is healthy, else the reason as a string.
.sp_crs_failure <- function() {
  err <- NULL
  ok <- suppressWarnings(tryCatch({
    r <- terra::rast(nrows = 2, ncols = 2, xmin = 0, xmax = 1,
                     ymin = 0, ymax = 1, crs = "EPSG:4326")
    isTRUE(nzchar(terra::crs(r)))
  }, error = function(e) { err <<- conditionMessage(e); FALSE }))
  if (ok) return(NULL)
  if (is.null(err)) err <- "terra returned an empty coordinate reference system"
  err
}

## Stop with an actionable message when terra cannot resolve a CRS.
.sp_check_crs <- function(what = "spCFmap()") {
  err <- .sp_crs_failure()
  if (is.null(err)) return(invisible(TRUE))
  stop(what, " cannot create a coordinate reference system, so no map layer ",
       "could be drawn.\n  terra reported: ", err,
       "\n  This usually means the PROJ database (proj.db) bundled with terra ",
       "is missing or unreadable.\n  Reinstall terra with ",
       'install.packages("terra"), or point the PROJ_LIB environment variable ',
       "at a valid PROJ data directory.", call. = FALSE)
}

## The covariate-effect layer is X %*% beta, so it needs the covariates the model
## was fitted with. cf_dglm did not keep them before spCF 0.2.1, and a fit saved
## by an older version therefore carries none. Substituting a column of ones
## there does not fail -- it quietly returns the row sum of beta at every site, a
## perfectly flat surface that reads as a real covariate effect. Return NULL
## instead, so the layer comes up blank and the app can say why.
##
## A genuinely intercept-only fit (one coefficient, no covariates ever supplied)
## is a different case and still maps, as the constant it is.
.sp_xb <- function(xx, beta, n) {
  if (is.null(xx))
    return(if (nrow(beta) > 1L) NULL else rep(unname(beta[, "coef"])[1], n))
  X <- as.matrix(xx)
  if (ncol(X) != nrow(beta)) X <- cbind(1, X)
  if (ncol(X) != nrow(beta)) return(NULL)      # still mismatched: cannot form Xb
  as.numeric(X %*% beta[, "coef"])
}

## TRUE when a fit cannot show a covariate effect because the covariates were
## never stored -- i.e. a cf_dglm saved by spCF <= 0.2.0. Refitting is the fix.
.sp_xb_unavailable <- function(mod) {
  if (is.null(mod$beta) || nrow(mod$beta) <= 1L) return(FALSE)
  xx <- if (.sp_is_downscale(mod)) mod$other$x
        else if (.sp_use0(mod)) mod$other$x0 else mod$other$x
  is.null(xx)
}

.sp_is_dglm <- function(mod)
  inherits(mod, "cf_dglm") || !is.null(mod$other$time0) || !is.null(mod$other$time)

.sp_is_downscale <- function(mod) inherits(mod, "cf_downscale")

## cf_downscale layer values at the disaggregate-level units. Predictions live
## in mod$pred (not mod$pred0); scale increments are pre-computed in mod$Z.
.sp_values_ds <- function(mod, layer, bw_range = c(0, Inf)) {
  switch(layer,
    pred    = mod$pred$pred,
    pred_sd = mod$pred$pred_sd,
    xb      = .sp_xb(mod$other$x, mod$beta, length(mod$pred$pred)),
    scale   = {
      if (bw_range[1] >= bw_range[2]) return(NULL)
      Z <- mod$Z; if (is.null(Z) || !ncol(Z)) return(NULL)
      sel <- mod$bands >= bw_range[1] & mod$bands <= bw_range[2]
      if (!any(sel)) return(NULL)
      rowSums(as.matrix(Z[, sel, drop = FALSE]))   # sum increments in the band
    })
}

## Map the prediction sites when the fit has them, else the sample sites. Only
## cf_dglm used to fall back this way, which made a cf_lm/cf_glm fitted without
## coords0/x0 unmappable even though its sample-site predictions exist.
.sp_use0 <- function(mod) !is.null(mod$other$coords0) && !is.null(mod$pred0)

## static (cf_lm / cf_glm) layer values at the mapped sites
.sp_values <- function(mod, layer, bw_range = c(0, Inf)) {
  use0 <- .sp_use0(mod)
  cc   <- if (use0) mod$other$coords0 else mod$other$coords
  if (is.null(cc)) stop("Model carries no coordinates to map.")
  switch(layer,
    pred    = if (use0) mod$pred0$pred    else mod$pred$pred,
    pred_sd = if (use0) mod$pred0$pred_sd else mod$pred$pred_sd,
    xb      = .sp_xb(if (use0) mod$other$x0 else mod$other$x, mod$beta,
                     nrow(as.matrix(cc))),
    scale   = {
      if (bw_range[1] >= bw_range[2]) return(NULL)
      sw <- tryCatch(spCF::sp_scalewise(mod, bw_range = bw_range),
                     error = function(e) NULL)
      if (is.null(sw)) return(NULL)
      d <- if (use0 && !is.null(sw$pred0)) sw$pred0 else sw$pred
      d$pred
    })
}

## data.frame(x, y, z) to rasterize. For cf_dglm the space-time output is
## reduced to one value per location, averaged over the chosen time range.
.sp_xyz <- function(mod, layer, bw_range = c(0, Inf), time_range = c(-Inf, Inf)) {
  if (.sp_is_downscale(mod)) {
    z <- .sp_values_ds(mod, layer, bw_range)
    if (is.null(z)) return(NULL)
    xy <- as.matrix(mod$other$coords)
    return(data.frame(x = xy[, 1], y = xy[, 2], z = z))
  }
  if (!.sp_is_dglm(mod)) {
    z <- .sp_values(mod, layer, bw_range)
    if (is.null(z)) return(NULL)
    xy <- as.matrix(if (.sp_use0(mod)) mod$other$coords0 else mod$other$coords)
    return(data.frame(x = xy[, 1], y = xy[, 2], z = z))
  }
  ## --- cf_dglm ---
  use0 <- .sp_use0(mod)
  if (layer == "scale") {
    if (bw_range[1] >= bw_range[2]) return(NULL)
    sw <- tryCatch(spCF::sp_scalewise(mod, bw_range = bw_range,
                                      time_range = time_range),
                   error = function(e) NULL)
    if (is.null(sw)) return(NULL)
    d <- if (use0 && !is.null(sw$pred0)) sw$pred0 else sw$pred
    return(data.frame(x = d$px, y = d$py, z = d$pred))
  }
  if (use0) { xy <- as.matrix(mod$other$coords0); t0 <- mod$other$time0; p <- mod$pred0 }
  else      { xy <- as.matrix(mod$other$coords);  t0 <- mod$other$time;  p <- mod$pred }
  sel <- t0 >= time_range[1] & t0 <= time_range[2]
  if (!any(sel)) return(NULL)
  val <- tryCatch(switch(layer,
    pred    = p$pred,
    pred_sd = p$pred_sd,
    xb      = .sp_xb(if (use0) mod$other$x0 else mod$other$x, mod$beta,
                     length(t0))), error = function(e) NULL)
  if (is.null(val)) return(NULL)
  key <- paste(xy[sel, 1], xy[sel, 2], sep = "_")
  data.frame(x = as.numeric(tapply(xy[sel, 1], key, `[`, 1)),
             y = as.numeric(tapply(xy[sel, 2], key, `[`, 1)),
             z = as.numeric(tapply(val[sel], key, mean)))
}

.sp_raster <- function(mod, layer, crs_str, bw_range = c(0, Inf),
                       time_range = c(-Inf, Inf)) {
  df <- tryCatch(.sp_xyz(mod, layer, bw_range, time_range), error = function(e) NULL)
  if (is.null(df) || !nrow(df) || all(!is.finite(df$z))) return(NULL)
  r <- tryCatch(terra::rast(df, type = "xyz", crs = crs_str), error = function(e) NULL)
  if (is.null(r)) r <- .sp_rast_nn(df, crs_str)      # irregular sites
  .sp_rast_display(r)
}

## Upsample a coarse raster before it is handed to leaflet.
##
## leaflet::addRasterImage(project = TRUE) reprojects to Web Mercator with
## nearest-neighbour resampling and picks the output resolution itself. On a
## coarse grid that resolution lands finer than the source in y but COARSER in
## x, so whole columns are resampled away -- 6 of the 25 columns on the
## space-time demo grid -- and the survivors end up unevenly spaced, which reads
## on the map as a thin seam running the height of the region.
##
## Splitting each cell into a block of identical cells first makes every source
## cell many output pixels wide, so the resampling can no longer drop one. The
## values and the blocky look are unchanged; only the sampling grid gets finer.
.sp_rast_display <- function(r, target = 2e5) {
  if (is.null(r)) return(NULL)
  n <- tryCatch(terra::ncell(r), error = function(e) NA_real_)
  if (!isTRUE(is.finite(n)) || n <= 0 || n >= target) return(r)
  f <- as.integer(floor(sqrt(target / n)))
  if (f <= 1L) return(r)
  tryCatch(terra::disagg(r, fact = f, method = "near"), error = function(e) r)
}

## Rasterize prediction sites that do NOT form a complete regular lattice (e.g.
## Cho-Cho-Aza centroids), by giving every raster cell the value of its NEAREST
## site -- a Voronoi tessellation drawn as one image.
##
## The two obvious alternatives are both bad here. rasterize(fun = mean) snaps
## sites to a coarse grid, so most of them are merged away or dropped (~42% kept
## on a realistic clustered grid) and the map comes out blocky and full of holes.
## Drawing one marker per site instead keeps them all, but tens of thousands of
## fixed-pixel circles pile up into an unreadable blob as soon as the map is
## zoomed out, and the browser has to draw every one of them on each redraw.
## Nearest-neighbour fill keeps ~99% of the sites, has no holes at any zoom, and
## hands the browser a single PNG. The k-d tree query is the only added cost.
.sp_rast_nn <- function(df, crs_str) {
  xy <- cbind(df$x, df$y)
  xr <- range(xy[, 1]); yr <- range(xy[, 2])
  if (!all(is.finite(c(xr, yr))) || diff(xr) <= 0 || diff(yr) <= 0) return(NULL)
  ## ~16 cells per site: fine enough that sites are only merged where two of
  ## them fall inside one cell, capped so the PNG stays small and quick to draw
  ncell <- min(3e5, max(5e4, 16 * nrow(df)))
  asp   <- diff(xr) / diff(yr)
  nc    <- max(2L, as.integer(round(sqrt(ncell * asp))))
  nr    <- max(2L, as.integer(round(ncell / nc)))
  r  <- terra::rast(nrows = nr, ncols = nc, xmin = xr[1], xmax = xr[2],
                    ymin = yr[1], ymax = yr[2], crs = crs_str)
  nn <- FNN::get.knnx(xy, terra::xyFromCell(r, seq_len(terra::ncell(r))), k = 1)
  z  <- df$z[nn$nn.index[, 1]]
  ## Blank cells that sit farther from every site than the sites sit from each
  ## other, so the mapped area keeps the shape of the study region instead of
  ## flooding the whole bounding box (sea, land outside the sample, ...).
  gap <- tryCatch(stats::quantile(FNN::get.knn(xy, k = 1)$nn.dist[, 1], 0.99,
                                  names = FALSE), error = function(e) NA_real_)
  if (is.finite(gap)) z[nn$nn.dist[, 1] > 2 * gap] <- NA_real_
  terra::values(r) <- z
  names(r) <- "z"
  r
}

## layer values as sf points (lon/lat) for models whose prediction sites are
## irregular (cf_downscale): rendered as coloured markers, not a snapped raster.
.sp_sf_points <- function(mod, layer, crs_str, bw_range = c(0, Inf),
                          time_range = c(-Inf, Inf)) {
  df <- tryCatch(.sp_xyz(mod, layer, bw_range, time_range), error = function(e) NULL)
  if (is.null(df) || !nrow(df)) return(NULL)
  df <- df[is.finite(df$z), , drop = FALSE]
  if (!nrow(df)) return(NULL)
  sf::st_transform(sf::st_as_sf(df, coords = c("x", "y"), crs = crs_str), 4326)
}

## TRUE when the model's prediction sites are irregular points (map as markers)
.sp_point_render <- function(mod) .sp_is_downscale(mod)

## Values that fix a TIME-COMMON colour scale for cf_dglm: the layer's values
## across ALL time points (so the palette/legend stay the same as the time
## slider moves). Returns NULL for non-space-time models.
.sp_domain_vals <- function(mod, layer, bw_range = c(0, Inf)) {
  if (!.sp_is_dglm(mod)) return(NULL)
  use0 <- .sp_use0(mod)
  p <- if (use0) mod$pred0 else mod$pred
  switch(layer,
    pred    = p$pred,
    pred_sd = p$pred_sd,
    xb      = .sp_xb(if (use0) mod$other$x0 else mod$other$x, mod$beta,
                     length(p$pred)),
    scale   = {
      ## RAW scale-field values over ALL time points (per location-time), NOT the
      ## time-averaged process: the colour domain must span the widest range any
      ## single time slice can show, otherwise extreme values at some years fall
      ## outside the palette and render transparent. Same half-open band selection
      ## as sp_scalewise() so the domain matches the displayed values.
      Z <- if (use0) mod$Z0 else mod$Z
      sel <- mod$bands >= bw_range[1] & mod$bands < bw_range[2]
      if (is.null(Z) || !ncol(as.matrix(Z)) || !any(sel)) NULL
      else rowSums(as.matrix(Z)[, sel, drop = FALSE])
    })
}

## prediction table for CSV export: xcoord, ycoord, pred, pred_sd
## (cf_dglm values are averaged over the given time range, one row per location)
.sp_export <- function(mod, time_range = c(-Inf, Inf)) {
  if (.sp_is_downscale(mod)) {
    xy <- as.matrix(mod$other$coords)
    return(data.frame(xcoord = xy[, 1], ycoord = xy[, 2],
                      pred = mod$pred$pred, pred_sd = mod$pred$pred_sd))
  }
  if (!.sp_is_dglm(mod)) {
    use0 <- .sp_use0(mod)
    xy   <- as.matrix(if (use0) mod$other$coords0 else mod$other$coords)
    p    <- if (use0) mod$pred0 else mod$pred
    return(data.frame(xcoord = xy[, 1], ycoord = xy[, 2],
                      pred = p$pred, pred_sd = p$pred_sd))
  }
  use0 <- .sp_use0(mod)
  if (use0) { xy <- as.matrix(mod$other$coords0); t0 <- mod$other$time0; p <- mod$pred0 }
  else      { xy <- as.matrix(mod$other$coords);  t0 <- mod$other$time;  p <- mod$pred }
  sel <- t0 >= time_range[1] & t0 <= time_range[2]
  key <- paste(xy[sel, 1], xy[sel, 2], sep = "_")
  data.frame(
    xcoord  = as.numeric(tapply(xy[sel, 1], key, `[`, 1)),
    ycoord  = as.numeric(tapply(xy[sel, 2], key, `[`, 1)),
    pred    = as.numeric(tapply(p$pred[sel], key, mean)),
    pred_sd = as.numeric(tapply(p$pred_sd[sel], key, mean)))
}

.sp_points <- function(mod, crs_str) {
  cc <- mod$other$coords
  if (is.null(cc)) return(NULL)
  cc <- unique(as.data.frame(cc))          # dedupe (space-time panels repeat sites)
  sf::st_transform(sf::st_as_sf(cc, coords = 1:2, crs = crs_str), 4326)
}

## coords used for the initial view (prediction sites, else sample sites)
.sp_view_coords <- function(mod) {
  if (!is.null(mod$other$coords0)) mod$other$coords0 else mod$other$coords
}

.sp_bbox4326 <- function(coords0, crs_str) {
  sf::st_bbox(sf::st_transform(
    sf::st_as_sf(as.data.frame(coords0), coords = 1:2, crs = crs_str), 4326))
}

## ---- UI pieces (namespaced) ------------------------------------------------
sp_map_controls <- function(id) {
  ns <- shiny::NS(id)
  shiny::tagList(
    shiny::wellPanel(
      shiny::strong("Outputs"),
      shiny::radioButtons(ns("layer"), NULL,
        c("Predictive mean" = "pred", "Predictive SD" = "pred_sd",
          "Covariate effect (xb)" = "xb", "Scale-wise component" = "scale"), "pred"),
      shiny::conditionalPanel(
        sprintf("input['%s'] == 'scale'", ns("layer")),
        shiny::sliderInput(ns("bw"), "Bandwidth range", 0, 1500,
                           c(0, 1500), step = 10)),
      shiny::uiOutput(ns("time_ui"))),      # time-range slider for cf_dglm fits
    shiny::wellPanel(
      shiny::strong("Display"),
      shiny::selectInput(ns("class_method"), "Color classification",
        c("Continuous" = "continuous", "Equal interval" = "equal",
          "Quantile" = "quantile", "Manual breaks" = "manual"), "continuous"),
      shiny::conditionalPanel(
        sprintf("input['%s'] == 'equal' || input['%s'] == 'quantile'",
                ns("class_method"), ns("class_method")),
        shiny::sliderInput(ns("nclass"), "Number of classes", 2, 12, 6, 1)),
      shiny::conditionalPanel(
        sprintf("input['%s'] == 'manual'", ns("class_method")),
        shiny::textInput(ns("breaks"), "Break values (comma-separated)", ""),
        shiny::helpText("Interior breaks; min/max added automatically.")),
      ## Spectral reversed: red = high, blue = low, which is the way a
      ## predicted surface is normally read. Spectral runs blue -> red, so the
      ## reverse box starts ticked.
      shiny::selectInput(ns("palette"), "Palette",
        c("viridis", "magma", "plasma", "inferno", "cividis",
          "YlOrRd", "YlGnBu", "RdYlBu", "Spectral"), selected = "Spectral"),
      shiny::checkboxInput(ns("rev_pal"), "Reverse palette", TRUE),
      shiny::selectInput(ns("basemap"), "Basemap",
        c("Light (Esri)" = "light",
          "Dark (Esri)" = "dark",
          "OpenStreetMap" = "osm",
          "Satellite (Esri)" = "sat",
          "Topographic (Esri)" = "topo"), selected = .sp_basemap_default),
      shiny::sliderInput(ns("opacity"), "Layer opacity", 0.1, 1, 0.75, 0.05),
      shiny::conditionalPanel(                 # only when the result is points
        sprintf("output['%s'] == true", ns("use_pts")),
        shiny::sliderInput(ns("ptsize"), "Point size", 1, 12, 6, 1)),
      shiny::checkboxInput(ns("show_pts"), "Show observed data", FALSE),
      shiny::tags$hr(),
      shiny::downloadButton(ns("dl"), "Download predictions (CSV)",
                            class = "btn-sm w-100"),
      shiny::conditionalPanel(          # polygon inputs: export valued polygons
        sprintf("output['%s'] == true", ns("has_poly")),
        shiny::downloadButton(ns("dl_geo"), "Download predictions (GeoJSON)",
                              class = "btn-sm w-100 mt-1"))))
}

sp_map_view <- function(id, height = "78vh") {
  leaflet::leafletOutput(shiny::NS(id, "map"), height = height)
}

## Model summary panel: fixed height, scrollable (prints the model object).
## The bottom strip is a caption for the map, not the other way round: keep it
## to roughly a seventh of the window so the map gets the rest.
sp_map_summary <- function(id, height = "13vh") {
  shiny::tags$div(style = paste0("height:", height, ";overflow:auto"),
                  shiny::verbatimTextOutput(shiny::NS(id, "summary")))
}

## ---- module server ---------------------------------------------------------
## mod, crs, preview are REACTIVES (functions). mod() may be NULL before a fit;
## preview() (optional) returns sf points (lon/lat) to show for a CRS check.
## home: optional c(xmin, ymin, xmax, ymax) in lon/lat for the initial view.
## geom: optional reactive returning an sf of polygons (lon/lat) aligned to the
## model's prediction sites (cf_downscale). When present, the result is drawn as
## a polygon choropleth instead of point markers.
sp_map_server <- function(id, mod, crs, preview = NULL, home = NULL,
                          geom = NULL) {
  shiny::moduleServer(id, function(input, output, session) {

    output$summary <- shiny::renderPrint({
      m <- mod()
      if (is.null(m)) { cat("No model yet.\n"); return(invisible()) }
      # cf_dglm stores the estimated temporal AR(1) but print() omits it;
      # insert it just before the "Error statistics" block.
      if (is.null(m$other$rho) || is.null(m$other$Q)) { print(m); return(invisible()) }
      lines <- utils::capture.output(print(m))
      ar <- c("---- Temporal AR(1) -----------------------------------",
              sprintf("rho = %.3f (autocorrelation),  Q = %.3g (innovation var)",
                      m$other$rho, m$other$Q))
      idx <- grep("Error statistics", lines)[1]
      if (is.na(idx)) { cat(lines, "", ar, sep = "\n"); return(invisible()) }
      cut <- if (idx >= 2 && lines[idx - 1] == "") idx - 1 else idx
      cat(c(lines[seq_len(cut - 1)], "", ar, lines[cut:length(lines)]), sep = "\n")
    })

    crs_str <- shiny::reactive(.sp_crs(crs()))
    ## largest bandwidth in the model's own units (m, degrees, ... - CRS-dependent);
    ## rounded to 4 significant figures so the slider bound stays short.
    top <- shiny::reactive({
      m <- mod(); if (is.null(m)) return(1500)
      b <- m$bands[is.finite(m$bands)]
      if (!length(b)) 1500 else signif(max(b), 4)
    })

    output$map <- leaflet::renderLeaflet({
      ## preferCanvas: render markers on an HTML canvas so large point layers
      ## (irregular prediction grids, dense observed data) stay smooth.
      base <- leaflet::leaflet(
          options = leaflet::leafletOptions(preferCanvas = TRUE)) |>
        .sp_add_base(group = "base", key = .sp_basemap_default)
      if (!is.null(home))
        base |> leaflet::fitBounds(home[1], home[2], home[3], home[4])
      else base |> leaflet::setView(0, 20, 2)
    })

    ## Re-fit the home extent via a proxy once the layout has settled, so the
    ## initial view matches what plotting the data later shows (no zoom jump).
    if (!is.null(home)) session$onFlushed(function() {
      leaflet::leafletProxy("map") |>
        leaflet::fitBounds(home[1], home[2], home[3], home[4])
    }, once = TRUE)

    ## new model -> set slider range and recenter (clearing on NULL is handled
    ## by the preview observer, the single authority for the "no model" state)
    shiny::observeEvent(mod(), {
      m <- mod(); shiny::req(m)
      tp <- top()
      st <- signif(tp / 200, 1)                    # ~200 steps, unit-adaptive
      if (!is.finite(st) || st <= 0) st <- tp / 100
      shiny::updateSliderInput(session, "bw", min = 0, max = tp,
                               value = c(0, tp), step = st)
      bb <- .sp_bbox4326(.sp_view_coords(m), crs_str())
      leaflet::leafletProxy("map") |>
        leaflet::fitBounds(bb[["xmin"]], bb[["ymin"]], bb[["xmax"]], bb[["ymax"]])
    })

    ## time-range slider - only for cf_dglm fits
    tvals <- shiny::reactive({
      m <- mod(); if (is.null(m) || !.sp_is_dglm(m)) return(NULL)
      tt <- c(m$other$time, m$other$time0); tt <- tt[is.finite(tt)]
      if (!length(tt)) NULL else tt
    })
    output$time_ui <- shiny::renderUI({
      tt <- tvals(); if (is.null(tt)) return(NULL)
      rng <- range(tt)
      lv  <- sort(unique(tt))
      step <- if (all(lv == round(lv))) 1 else signif(diff(rng) / 100, 2)
      shiny::tagList(
        ## default to the LAST time point (a single slice) rather than the whole
        ## range, so the initial map shows one time point instead of the average
        ## over every time point (which is rarely what the user expects first).
        shiny::sliderInput(session$ns("trange"), "Time range", rng[1], rng[2],
                           value = c(rng[2], rng[2]), step = step),
        shiny::helpText("Process is averaged over the selected range ",
                        "(a single point shows that time slice)."))
    })
    time_range <- shiny::reactive({
      tt <- tvals(); if (is.null(tt) || is.null(input$trange)) return(c(-Inf, Inf))
      input$trange
    })

    ## basemap switch -> re-draw overlay on top
    refresh <- shiny::reactiveVal(0)
    shiny::observeEvent(input$basemap, {
      leaflet::leafletProxy("map") |> leaflet::clearGroup("base") |>
        .sp_add_base(key = input$basemap, group = "base")
      refresh(shiny::isolate(refresh()) + 1)
    }, ignoreInit = TRUE)

    ## optional preview before a model exists. If the sf carries a ".grp"
    ## column (e.g. downscale Area ID), colour points by it; otherwise a plain
    ## single-colour CRS-check overlay.
    if (!is.null(preview)) shiny::observe({
      if (!is.null(mod())) return()
      ## no model (initial, or after a task switch): wipe any stale overlay,
      ## then draw the preview if one is available.
      mp <- leaflet::leafletProxy("map") |> leaflet::clearImages() |>
        leaflet::clearControls() |> leaflet::clearMarkers() |>
        leaflet::clearShapes()
      pp <- preview(); if (is.null(pp)) return()
      bb <- sf::st_bbox(pp)
      poly <- grepl("POLYGON",
        as.character(sf::st_geometry_type(pp, by_geometry = FALSE))[1])
      draw <- function(m, fill) {          # data preview: polygons / markers
        if (poly)
          # thin light outline on the data preview only (result stays borderless)
          leaflet::addPolygons(m, data = pp, stroke = TRUE, weight = 0.8,
                               color = "#555", opacity = 0.9, fillColor = fill,
                               fillOpacity = if (".grp" %in% names(pp)) .8 else .4,
                               smoothFactor = 0.3)
        else {
          pc <- sf::st_coordinates(pp)
          leaflet::addCircleMarkers(m, lng = pc[, 1], lat = pc[, 2],
            radius = if (".grp" %in% names(pp)) 5 else 3, stroke = FALSE,
            fillColor = fill, fillOpacity = if (".grp" %in% names(pp)) .85 else .6)
        }
      }
      if (".grp" %in% names(pp)) {
        levs <- levels(as.factor(pp$.grp))
        cols <- grDevices::hcl.colors(max(length(levs), 2L), "Dark 3")
        pal  <- leaflet::colorFactor(cols, domain = levs)
        mp <- draw(mp, pal(pp$.grp))
        mp <- if (length(levs) <= 10)          # short legend only when few areas
          mp |> leaflet::addLegend("bottomright", pal = pal, values = levs,
                                   title = "Area ID", opacity = 1)
        else
          mp |> leaflet::addControl(position = "bottomright", html = sprintf(
            "<div style='font-size:11px'>Coloured by Area ID \u2014 %d areas</div>",
            length(levs)))
      } else {
        ## same black dots the fitted map draws for observed data, so the sites
        ## do not change colour the moment a model appears
        mp <- draw(mp, "black")
      }
      mp |> leaflet::fitBounds(bb[["xmin"]], bb[["ymin"]],
                               bb[["xmax"]], bb[["ymax"]])
    })

    bw_d <- shiny::debounce(shiny::reactive(input$bw), 300)
    tr_d <- shiny::debounce(time_range, 300)

    output$dl <- shiny::downloadHandler(
      filename = function() "spCF_predictions.csv",
      content  = function(file) {
        m <- mod(); shiny::req(m)
        utils::write.csv(.sp_export(m, tr_d()), file, row.names = FALSE)
      })

    ## polygon geometry aligned to the fit (cf_downscale from a polygon upload)
    poly_geom <- shiny::reactive({
      m <- mod(); if (is.null(m)) return(NULL)
      g <- if (!is.null(geom)) geom() else NULL
      if (!is.null(g) && nrow(g) == length(m$pred$pred)) g else NULL
    })
    output$has_poly <- shiny::reactive(!is.null(poly_geom()))
    shiny::outputOptions(output, "has_poly", suspendWhenHidden = FALSE)

    output$dl_geo <- shiny::downloadHandler(
      filename = function() "spCF_predictions.geojson",
      content  = function(file) {
        m <- mod(); g <- poly_geom(); shiny::req(m, g)
        ex <- .sp_export(m, tr_d())                 # pred / pred_sd, same order
        g$pred <- ex$pred; g$pred_sd <- ex$pred_sd
        sf::st_write(g, file, driver = "GeoJSON", delete_dsn = TRUE,
                     quiet = TRUE)
      })
    band_range <- shiny::reactive({
      if (input$layer != "scale") return(c(0, Inf))
      rng <- bw_d(); if (rng[1] >= rng[2]) return(NULL)
      c(rng[1], if (rng[2] >= top()) Inf else rng[2])
    })
    layer_raster <- shiny::reactive({
      m <- mod(); shiny::req(m)
      br <- band_range(); if (is.null(br)) return(NULL)  # empty scale range
      .sp_raster(m, input$layer, crs_str(), br, tr_d())
    })
    layer_points <- shiny::reactive({          # cf_downscale: irregular sites
      m <- mod(); shiny::req(m)
      br <- band_range(); if (is.null(br)) return(NULL)
      .sp_sf_points(m, input$layer, crs_str(), br, tr_d())
    })

    ## the result is drawn as point markers (so "Point size" applies) only for a
    ## cf_downscale fit whose input carried no polygon geometry. Every other fit
    ## is a raster image -- an irregular grid included, via .sp_rast_nn().
    output$use_pts <- shiny::reactive({
      m <- mod(); !is.null(m) && .sp_point_render(m) && is.null(poly_geom())
    })
    shiny::outputOptions(output, "use_pts", suspendWhenHidden = FALSE)

    build_pal <- function(vals) {
      p <- input$palette; rv <- isTRUE(input$rev_pal)
      switch(input$class_method,
        continuous = leaflet::colorNumeric(p, vals, na.color = "transparent",
                                           reverse = rv),
        equal = leaflet::colorBin(p, vals,
          bins = seq(min(vals), max(vals), length.out = input$nclass + 1),
          na.color = "transparent", reverse = rv),
        quantile = leaflet::colorQuantile(p, vals, n = input$nclass,
          na.color = "transparent", reverse = rv),
        manual = {
          rng <- range(vals)
          b <- suppressWarnings(as.numeric(strsplit(input$breaks, "[,\\s]+")[[1]]))
          b <- b[is.finite(b) & b > rng[1] & b < rng[2]]
          br <- sort(unique(c(rng[1], b, rng[2]))); if (length(br) < 2) br <- rng
          leaflet::colorBin(p, vals, bins = br, na.color = "transparent",
                            reverse = rv)
        })
    }

    ## descending legend (largest value on top), colours matched to labels.
    ## Built manually because leaflet's continuous legend cannot be reversed.
    add_legend <- function(map, pal, vals) {
      ttl <- .sp_titles[[input$layer]]
      brk <- switch(input$class_method,
        continuous = seq(min(vals), max(vals), length.out = 8),
        quantile   = unique(stats::quantile(vals,
                        probs = attr(pal, "colorArgs")$probs, names = FALSE)),
        attr(pal, "colorArgs")$bins)            # equal / manual (colorBin)
      if (length(brk) < 2)
        return(leaflet::addLegend(map, "bottomright", pal = pal, values = vals,
                                  title = ttl))
      lo   <- brk[-length(brk)]; hi <- brk[-1]
      cols <- pal((lo + hi) / 2)
      labs <- sprintf("%s \u2013 %s", .sp_fmt(lo), .sp_fmt(hi))
      leaflet::addLegend(map, "bottomright", colors = rev(cols),
                         labels = rev(labs), title = ttl)
    }

    ## wipe every overlay (used when the current layer has nothing to show,
    ## e.g. no spatial basis inside the chosen Scale bandwidth range).
    clear_overlay <- function()
      leaflet::leafletProxy("map") |> leaflet::clearImages() |>
        leaflet::clearControls() |> leaflet::clearMarkers() |>
        leaflet::clearShapes()

    shiny::observe({
      refresh()
      m <- mod(); shiny::req(m)

      ## A fit that never stored its covariates (cf_dglm from spCF <= 0.2.0)
      ## cannot show this layer. Say so on the map rather than leaving a blank
      ## panel -- or, worse than blank, the flat surface the old code drew.
      if (input$layer == "xb" && .sp_xb_unavailable(m)) {
        clear_overlay() |> leaflet::addControl(position = "topright",
          html = paste0("<div style='background:#fff;padding:6px 9px;",
            "border-radius:4px;font-size:12px;max-width:260px'>",
            "<b>Covariate effect unavailable</b><br>This fit was made with ",
            "spCF 0.2.0 or earlier, which did not keep the covariates. ",
            "Refit the model to map this layer.</div>"))
        return(invisible())
      }

      ## --- irregular prediction sites (cf_downscale): polygons if the input
      ## carried polygon geometry, otherwise one coloured marker per location ---
      if (.sp_point_render(m)) {
        br   <- band_range()
        zall <- if (is.null(br)) NULL else .sp_values_ds(m, input$layer, br)
        if (is.null(zall) || !any(is.finite(zall))) { clear_overlay(); return(invisible()) }
        fin  <- is.finite(zall)
        pal  <- build_pal(zall[fin])
        g    <- if (!is.null(geom)) geom() else NULL
        mp <- clear_overlay()
        if (!is.null(g) && nrow(g) == length(zall)) {         # polygon choropleth
          mp <- mp |> leaflet::addPolygons(data = g, stroke = FALSE,
            fillColor = pal(zall), fillOpacity = input$opacity,
            smoothFactor = 0.3)
        } else {                                              # point markers
          pts <- layer_points()
          if (is.null(pts)) { clear_overlay(); return(invisible()) }
          pc  <- sf::st_coordinates(pts)
          mp <- mp |> leaflet::addCircleMarkers(lng = pc[, 1], lat = pc[, 2],
            radius = input$ptsize, stroke = FALSE, fillColor = pal(pts$z),
            fillOpacity = input$opacity)
        }
        add_legend(mp, pal, zall[fin])
        return(invisible())
      }

      ## --- gridded fits (cf_lm / cf_glm / cf_dglm): raster image. A complete
      ## lattice rasterizes directly; an irregular grid is filled from its
      ## nearest site by .sp_rast_nn(), so both come out as one clean image.
      r <- layer_raster()
      vals <- if (is.null(r)) NULL else {
        v <- terra::values(r); v[is.finite(v)]
      }
      if (is.null(vals) || !length(vals)) { clear_overlay(); return(invisible()) }
      # space-time: fix the colour scale over all time points so the palette
      # and legend stay comparable as the time slider moves.
      dom <- .sp_domain_vals(m, input$layer, if (is.null(band_range())) c(0, Inf)
                             else band_range())
      pal_vals <- if (!is.null(dom)) dom[is.finite(dom)] else vals
      if (!length(pal_vals)) pal_vals <- vals

      pal  <- build_pal(pal_vals)
      ## clamp the raster into the colour domain so no in-range prediction pixel
      ## is left transparent by a floating-point edge (all sites are mapped).
      rng  <- range(pal_vals)
      r    <- tryCatch(terra::clamp(r, rng[1], rng[2], values = TRUE),
                       error = function(e) r)
      mp <- leaflet::leafletProxy("map") |>
        leaflet::clearImages() |> leaflet::clearControls() |>
        leaflet::clearMarkers() |> leaflet::clearShapes() |>
        leaflet::addRasterImage(r, colors = pal, opacity = input$opacity,
                                project = TRUE)
      mp <- add_legend(mp, pal, pal_vals)
      ## Observed data are a locator for the sample sites, not a second layer to
      ## read values off: the same fixed black dots on every layer, so they never
      ## compete with the raster's colour scale.
      if (isTRUE(input$show_pts)) {
        pts <- .sp_points(m, crs_str())
        if (!is.null(pts)) {
          pc <- sf::st_coordinates(pts)
          mp |> leaflet::addCircleMarkers(lng = pc[, 1], lat = pc[, 2],
            radius = 3, stroke = FALSE, fillColor = "black",   # fixed size
            fillOpacity = 0.6)
        }
      }
    })
  })
}

## ---- single-model app ------------------------------------------------------
## Builds (but does not run) the small app that maps one fitted model.
## Package availability is checked by the caller, spCFmap().
sp_map_app <- function(mod, crs) {
  ui <- shiny::fluidPage(
    shiny::tags$style("body{margin:0} .well{padding:10px}"),
    shiny::fluidRow(
      shiny::column(3, sp_map_controls("m")),
      shiny::column(9,
        sp_map_view("m", height = "78vh"),
        shiny::tags$h5("Model summary", style = "margin:8px 0 4px"),
        sp_map_summary("m", height = "13vh"))))
  server <- function(input, output, session)
    sp_map_server("m", mod = shiny::reactive(mod), crs = shiny::reactive(crs))
  shiny::shinyApp(ui, server)
}
