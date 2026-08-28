# spCFmap — Interactive coarse-to-fine spatial mapping
# CFSM (spCF) spatial prediction / covariate effect / scale-wise components,
# mapped over a basemap. Demo data (meuse) or user CSV upload.
# The map itself is drawn by the shared module in sp_map_core.R (same engine as
# spCFmap(mod, crs)).
# Run from RStudio:  shiny::runApp("app.R")

library(stats)     # glm family functions used by cf_glm internals
library(shiny)
library(bslib)
library(leaflet)
library(terra)
library(sf)
library(spCF)

# mapping engine lives in the spCF package (spCFmap() is the exported entry
# point; the Shiny module pieces used below are package internals)
sp_map_controls <- spCF:::sp_map_controls
sp_map_view     <- spCF:::sp_map_view
sp_map_summary  <- spCF:::sp_map_summary
sp_map_server   <- spCF:::sp_map_server
options(shiny.maxRequestSize = 100 * 1024^2)   # 100 MB uploads

# ---- Spatial upload helpers (GeoJSON) ---------------------------------------
# Read an uploaded GeoJSON into an sf object. `fileinfo` is input$f_* —
# a data.frame with $name and $datapath (one row per selected file).
.read_spatial_upload <- function(fileinfo) {
  exts <- tolower(tools::file_ext(fileinfo$name))
  gj <- fileinfo$datapath[exts %in% c("geojson", "json")]
  if (!length(gj)) stop("Unsupported file type. Use CSV or GeoJSON.")
  sf::st_read(gj[1], quiet = TRUE)
}

# sf -> plain data.frame with explicit coordinate columns + detected EPSG.
# Coordinates are the geometry CENTROIDS (used for computation); the original
# geometry is returned in `geom` so polygon inputs can be mapped as polygons.
.sf_to_xydf <- function(g, xcol = "x_coord", ycol = "y_coord") {
  gtype   <- as.character(sf::st_geometry_type(g, by_geometry = FALSE))[1]
  is_poly <- grepl("POLYGON", gtype)
  geom    <- sf::st_geometry(g)
  pts     <- if (grepl("POINT", gtype)) geom
             else suppressWarnings(sf::st_centroid(geom))
  xy      <- sf::st_coordinates(pts)
  df      <- sf::st_drop_geometry(g)
  df[[xcol]] <- xy[, 1]; df[[ycol]] <- xy[, 2]
  # polygon area as a ready-made proportional weight (spherical m^2 for lon/lat)
  if (is_poly)
    df[["areas_of_the_polygons"]] <-
      as.numeric(suppressWarnings(sf::st_area(geom)))
  epsg <- tryCatch(sf::st_crs(g)$epsg, error = function(e) NA_integer_)
  if (is.null(epsg)) epsg <- NA_integer_
  list(df = df, epsg = epsg, xcol = xcol, ycol = ycol,
       is_poly = is_poly, geom = sf::st_sf(geometry = geom))
}

# ---- Demo data (meuse) -------------------------------------------------------
utils::data("meuse", package = "sp")
utils::data("meuse.grid", package = "sp")
DEMO_CRS <- "EPSG:28992"

demo_sample <- data.frame(
  x = meuse$x, y = meuse$y, logzinc = log(meuse$zinc), dist = meuse$dist,
  ffreq2 = as.integer(meuse$ffreq == 2), ffreq3 = as.integer(meuse$ffreq == 3))
demo_grid <- data.frame(
  x = meuse.grid$x, y = meuse.grid$y, dist = meuse.grid$dist,
  ffreq2 = as.integer(meuse.grid$ffreq == 2),
  ffreq3 = as.integer(meuse.grid$ffreq == 3))

# initial map view (before any fit): the demo (meuse) area, in lon/lat
DEMO_HOME <- as.numeric(st_bbox(st_transform(
  st_as_sf(demo_sample[, c("x", "y")], coords = 1:2, crs = DEMO_CRS), 4326)))

# ---- Downscale demo (pollutionhealthdata, GeoJSON polygons) ------------------
# 271 fine units (Glasgow intermediate zones) grouped into 30 areas, as in the
# spCF CF-DS vignette. Read from the bundled GeoJSON so the demo behaves exactly
# like a polygon upload (mapped as polygons; computed on the centroids).
demo_ds_info <- c(.sf_to_xydf(sf::st_read("example_downscale.geojson",
                                          quiet = TRUE)),
                  list(spatial = TRUE, demo = TRUE))

# ---- Space-time demo (spacetime::air — monthly PM10, 2001-2005, lon/lat) -----
# ~63 German rural background stations x 60 months (time = 1..60). A point
# space-time set for cf_dglm. Bundled as a CSV so no runtime dependency.
demo_st <- utils::read.csv("example_spacetime_air.csv")

# ---- Worked examples offered next to each upload box -------------------------
# The package ships a filled-in file for every kind of upload; handing one to the
# user answers "what should my file look like?" better than any prose, and it can
# be uploaded straight back to try the whole flow. The directory is captured now
# because a downloadHandler's content() runs much later, when the working
# directory is no longer guaranteed to be the app's own.
EXAMPLE_DIR <- normalizePath(".", mustWork = FALSE)

# ---- UI ----------------------------------------------------------------------
ui <- page_sidebar(
  title = "spCFmap — Interactive coarse-to-fine spatial mapping",
  theme = bs_theme(version = 5, bootswatch = "flatly"),
  sidebar = sidebar(
    width = 340,
    # trim the empty space bslib reserves above the first control
    tags$style(HTML(".sidebar-content{padding-top:.4rem}")),
    radioButtons("mode", "Task",
                 c("Point prediction" = "point",
                   "Downscaling"      = "downscale"), "point"),

    # ---- POINT prediction: data (demo / CSV / GeoJSON) ----------------------
    conditionalPanel(
      condition = "input.mode == 'point'",
      card(
        card_header("Data"),
        radioButtons("src", NULL,
                     c("Demo (meuse, sp)" = "demo",
                       "Demo (air, space-time)" = "demo_st",
                       "Upload" = "upload"), "upload"),
        conditionalPanel(
          condition = "input.src == 'demo_st'",
          div(class = "small text-muted", style = "margin-top:-6px;margin-bottom:6px",
              "Monthly PM10, 2001–2005 (time = month).")),
        conditionalPanel(
          condition = "input.src == 'upload'",
          tags$style(HTML(
            "#f_sample_progress,#f_grid_progress{height:0;margin:0;min-height:0}")),
          fileInput("f_sample",
                    tagList("Sample points ", tags$span(
                      class = "text-muted", style = "font-size:.8em",
                      "(CSV / GeoJSON)")),
                    multiple = TRUE,
                    accept = c(".csv", ".geojson", ".json")),
          div(class = "small text-muted",
              style = "margin-top:-14px;margin-bottom:8px",
              "Example: ", downloadLink("dl_ex_samples", "CSV"), " \u00b7 ",
              downloadLink("dl_ex_points", "GeoJSON"), " \u00b7 ",
              downloadLink("dl_ex_pt_readme", "ReadMe")),
          fileInput("f_grid",
                    tagList("Prediction grid ", tags$span(
                      class = "text-muted", style = "font-size:.8em",
                      "(CSV / GeoJSON)")),
                    multiple = TRUE,
                    accept = c(".csv", ".geojson", ".json")),
          div(class = "small text-muted",
              style = "margin-top:-14px;margin-bottom:8px",
              "Example: ", downloadLink("dl_ex_grid", "CSV"), " \u00b7 ",
              downloadLink("dl_ex_grid_geo", "GeoJSON"), " \u00b7 ",
              downloadLink("dl_ex_pt_readme2", "ReadMe"), tags$br(),
              "No prediction grid \u2192 auto-build a grid.")
        ),
        uiOutput("col_map")
      )
    ),

    # ---- SPATIAL downscaling: data (disaggregate units) ----------------------
    conditionalPanel(
      condition = "input.mode == 'downscale'",
      card(
        card_header("Data"),
        radioButtons("src_ds", NULL,
                     c("Demo (pollutionhealthdata)" = "demo",
                       "Upload" = "upload"), "upload"),
        conditionalPanel(
          condition = "input.src_ds == 'upload'",
          tags$style(HTML("#f_ds_progress{height:0;margin:0;min-height:0}")),
          fileInput("f_ds",
                    tagList("Disaggregate units ", tags$span(
                      class = "text-muted", style = "font-size:.8em",
                      "(CSV / GeoJSON)")),
                    multiple = TRUE,
                    accept = c(".csv", ".geojson", ".json")),
          div(class = "small text-muted",
              style = "margin-top:-14px;margin-bottom:8px",
              "Example: ", downloadLink("dl_ex_ds", "CSV"), " \u00b7 ",
              downloadLink("dl_ex_ds_geo", "GeoJSON"), " \u00b7 ",
              downloadLink("dl_ex_ds_readme", "ReadMe"), tags$br(),
              "Disaggregate-level units; give an area ID + area-level response.")
        ),
        uiOutput("col_map_ds")
      )
    ),

    # ---- shared: coordinate system ------------------------------------------
    card(
      card_header("Coordinate system"),
      selectInput("crs_preset", NULL,
                  c("Longitude / latitude (WGS84)" = "4326",
                    "Web Mercator (web tiles)"     = "3857",
                    "Other — enter EPSG code"      = "other"),
                  selectize = FALSE),   # native select: dropdown not clipped by card
      conditionalPanel(
        condition = "input.crs_preset == 'other'",
        textInput("epsg", "EPSG code", "4326"),
        tags$a(href = "https://epsg.io", target = "_blank",
               class = "small", "Look up your EPSG code at epsg.io")),
      conditionalPanel(
        condition = "input.mode == 'downscale' || input.src == 'upload'",
        div(class = "small text-muted",
            textOutput("crs_note", inline = TRUE)))
    ),

    # ---- POINT prediction: model --------------------------------------------
    conditionalPanel(
      condition = "input.mode == 'point'",
      card(
        card_header("Model"),
        selectInput("family", "Distribution",
                    c("Gaussian"            = "gaussian",
                      "Poisson"             = "poisson",
                      "Binomial / logistic" = "binomial",
                      "Gamma"               = "Gamma",
                      "Inverse Gaussian"    = "inverse.gaussian",
                      "Quasipoisson"        = "quasipoisson",
                      "Quasibinomial"       = "quasibinomial"),
                    selectize = FALSE),
        actionButton("run", "Run spatial prediction", class = "btn-primary w-100")
      )
    ),

    # ---- SPATIAL downscaling: model -----------------------------------------
    conditionalPanel(
      condition = "input.mode == 'downscale'",
      card(
        card_header("Model"),
        selectInput("dytype", "Aggregate response type",
                    c("Sum (extensive: counts, totals)"     = "sum",
                      "Mean (intensive: rates, densities)"   = "mean"),
                    selectize = FALSE),
        actionButton("run_ds", "Run downscaling", class = "btn-primary w-100")
      )
    ),

    sp_map_controls("map")                     # Outputs + Display (shared module)
  ),
  layout_columns(
    col_widths = 12,
    card(full_screen = TRUE, sp_map_view("map", height = "78vh")),
    card(
      card_header(textOutput("bottom_title", inline = TRUE)),
      # before a fit: the input data; after a fit: the model summary
      conditionalPanel("output.has_fit == false",
        div(style = "height:13vh;min-height:110px;overflow:auto",
            dataTableOutput("data_tbl"))),
      conditionalPanel("output.has_fit == true",
        sp_map_summary("map", height = "13vh"))
    )
  ),
  tags$div(class = "text-muted small", style = "padding:2px 12px 8px",
    HTML(paste(
      "Demo data — <b>meuse</b>: R <i>sp</i> package (Pebesma &amp; Bivand).",
      "<b>air</b>: R <i>spacetime</i> package (Gräler, Pebesma &amp; Heuvelink).",
      "<b>pollutionhealthdata</b> &amp; <b>GGHB.IZ</b>:",
      "R <i>CARBayesdata</i> package (D. Lee).")))
)

# ---- Server ------------------------------------------------------------------
server <- function(input, output, session) {

  # -- bundled worked examples (see EXAMPLE_DIR above) -------------------------
  example_dl <- function(file) downloadHandler(
    filename = function() file,
    content  = function(con) {
      src <- file.path(EXAMPLE_DIR, file)
      if (!file.exists(src))
        stop("bundled example not found: ", src, call. = FALSE)
      file.copy(src, con, overwrite = TRUE)
    })
  output$dl_ex_samples  <- example_dl("example_samples.csv")
  output$dl_ex_points   <- example_dl("example_points.geojson")
  output$dl_ex_grid     <- example_dl("example_grid.csv")
  output$dl_ex_grid_geo <- example_dl("example_grid.geojson")
  output$dl_ex_ds       <- example_dl("example_downscale.csv")
  output$dl_ex_ds_geo   <- example_dl("example_downscale.geojson")
  # the point-prediction ReadMe covers both the samples and the grid, so it is
  # offered on both rows -- one file, two output ids
  output$dl_ex_pt_readme  <- example_dl("example_point_ReadMe.txt")
  output$dl_ex_pt_readme2 <- example_dl("example_point_ReadMe.txt")
  output$dl_ex_ds_readme  <- example_dl("example_downscale_ReadMe.txt")

  # sample input: CSV (plain) or GeoJSON (coords + EPSG auto-detected)
  sample_in <- reactive({
    if (input$src == "demo")
      return(list(df = demo_sample, epsg = 28992L,
                  xcol = "x", ycol = "y", spatial = FALSE))
    if (input$src == "demo_st")
      return(list(df = demo_st, epsg = 4326L,
                  xcol = "lon", ycol = "lat", spatial = FALSE))
    req(input$f_sample)
    fi <- input$f_sample; exts <- tolower(tools::file_ext(fi$name))
    if (all(exts == "csv"))
      return(list(df = utils::read.csv(fi$datapath[1], check.names = TRUE),
                  epsg = NA_integer_, xcol = NULL, ycol = NULL, spatial = FALSE))
    c(.sf_to_xydf(.read_spatial_upload(fi)), list(spatial = TRUE))
  })
  raw_sample <- reactive(sample_in()$df)

  raw_grid <- reactive({
    if (input$src == "demo") return(demo_grid)
    if (is.null(input$f_grid)) return(NULL)
    fi <- input$f_grid; exts <- tolower(tools::file_ext(fi$name))
    if (all(exts == "csv"))
      return(utils::read.csv(fi$datapath[1], check.names = TRUE))
    .sf_to_xydf(.read_spatial_upload(fi))$df       # same x_coord/y_coord names
  })

  # downscale input: one row per disaggregate unit (demo, or upload)
  ds_in <- reactive({
    if (input$src_ds == "demo") return(demo_ds_info)
    req(input$f_ds)
    fi <- input$f_ds; exts <- tolower(tools::file_ext(fi$name))
    if (all(exts == "csv"))
      return(list(df = utils::read.csv(fi$datapath[1], check.names = TRUE),
                  epsg = NA_integer_, xcol = NULL, ycol = NULL, spatial = FALSE))
    c(.sf_to_xydf(.read_spatial_upload(fi)), list(spatial = TRUE))
  })

  output$col_map <- renderUI({
    s <- sample_in(); df <- s$df; req(df)
    nm <- names(df)
    guess <- function(cands, pool = nm, default = pool[1]) {
      hit <- pool[tolower(pool) %in% cands]; if (length(hit)) hit[1] else default }
    time_sel <- ""
    if (input$src == "demo") {                     # pre-filled with demo variables
      resp_sel <- "logzinc"; cx_sel <- "x"; cy_sel <- "y"
      cov_sel  <- intersect(c("dist", "ffreq2", "ffreq3"), nm); off_sel <- ""
    } else if (input$src == "demo_st") {            # space-time demo (air, PM10)
      resp_sel <- "pm10"; cx_sel <- "lon"; cy_sel <- "lat"
      time_sel <- "time"; cov_sel <- character(0); off_sel <- ""
    } else if (isTRUE(s$spatial)) {                 # GeoJSON: coords fixed
      cx_sel <- s$xcol; cy_sel <- s$ycol
      attrs  <- setdiff(nm, c(cx_sel, cy_sel))
      resp_sel <- guess(c("y_resp", "response", "value", "z", "target"),
                        pool = attrs,
                        default = if (length(attrs)) attrs[1] else nm[1])
      cov_sel  <- character(0); off_sel <- ""
    } else {                                        # guessed from uploaded columns
      resp_sel <- guess(c("y_resp", "response", "value", "z", "target"))
      cx_sel   <- guess(c("x", "lon", "long", "longitude", "easting"))
      cy_sel   <- guess(c("lat", "latitude", "northing", "y"),
                        default = nm[min(2, length(nm))])
      cov_sel  <- character(0); off_sel <- ""
    }
    tagList(
      tags$style(HTML("#colmap .shiny-input-container{margin-bottom:2px}")),
      div(id = "colmap",
        selectInput("cx", "X-coordinate", nm, selected = cx_sel, selectize = FALSE),
        selectInput("cy", "Y-coordinate", nm, selected = cy_sel, selectize = FALSE),
        selectInput("time_coord", "Time coordinate — optional",
                    c("(none)" = "", nm), selected = time_sel, selectize = FALSE),
        selectInput("resp", "Response (y)", nm, selected = resp_sel, selectize = FALSE),
        selectizeInput("covs", "Covariates (x) — optional", nm, selected = cov_sel,
                       multiple = TRUE, options = list(dropdownParent = "body")),
        selectInput("offset", "Offset — optional", c("(none)" = "", nm),
                    selected = off_sel, selectize = FALSE)))
  })

  # downscale column mapping (coords auto for demo / spatial files)
  output$col_map_ds <- renderUI({
    s <- ds_in(); df <- s$df; req(df)
    nm <- names(df)
    guess <- function(cands, pool = nm, default = pool[1]) {
      hit <- pool[tolower(pool) %in% cands]; if (length(hit)) hit[1] else default }
    if (isTRUE(s$demo)) {                            # pre-filled demo columns
      cx_sel <- s$xcol; cy_sel <- s$ycol; agg_sel <- "agg_id"
      resp_sel <- "Y"; cov_sel <- c("pm10", "jsa", "price"); w_sel <- "expected"
    } else {
    if (isTRUE(s$spatial)) { cx_sel <- s$xcol; cy_sel <- s$ycol }
    else {
      cx_sel <- guess(c("x", "lon", "long", "longitude", "easting"))
      cy_sel <- guess(c("lat", "latitude", "northing", "y"),
                      default = nm[min(2, length(nm))])
    }
    attrs   <- setdiff(nm, c(cx_sel, cy_sel))
    agg_sel <- guess(c("agg_id", "area", "area_id", "region", "zone", "id"),
                     pool = attrs, default = if (length(attrs)) attrs[1] else nm[1])
    rpool   <- setdiff(attrs, agg_sel)
    resp_sel <- guess(c("y", "response", "value", "target", "count", "observed"),
                      pool = rpool, default = if (length(rpool)) rpool[1] else nm[1])
    cov_sel <- character(0)
    w_sel   <- guess(c("prop_weight", "weight", "expected", "pop", "population"),
                     pool = attrs,
                     default = if ("areas_of_the_polygons" %in% nm)
                       "areas_of_the_polygons" else "")
    }
    tagList(
      tags$style(HTML("#colmapds .shiny-input-container{margin-bottom:2px}")),
      div(id = "colmapds",
        selectInput("dx", "X-coordinate", nm, selected = cx_sel, selectize = FALSE),
        selectInput("dy", "Y-coordinate", nm, selected = cy_sel, selectize = FALSE),
        selectInput("dagg", "Area ID (aggregation unit)", nm,
                    selected = agg_sel, selectize = FALSE),
        selectInput("dresp", "Aggregate response (Y)", nm,
                    selected = resp_sel, selectize = FALSE),
        div(class = "small text-muted", style = "margin-top:-2px",
            "Aggregated value repeated for all subunits."),
        selectizeInput("dcovs", "Covariates (x) — optional", nm,
                       selected = cov_sel, multiple = TRUE,
                       options = list(dropdownParent = "body")),
        selectInput("dweight", "Proportional weight — optional",
                    c("(none)" = "", nm), selected = w_sel, selectize = FALSE)))
  })

  # set the shared CRS selector from a numeric EPSG (4326/3857 as presets)
  set_crs <- function(ep) {
    if (is.null(ep) || is.na(ep)) return()
    if (ep %in% c(4326L, 3857L)) {
      updateSelectInput(session, "crs_preset", selected = as.character(ep))
    } else {
      updateSelectInput(session, "crs_preset", selected = "other")
      updateTextInput(session, "epsg", value = as.character(ep))
    }
  }

  # keep the CRS selector in sync with the data source (point mode)
  observeEvent(input$src, {
    if (input$mode != "point") return()
    if (input$src == "demo") set_crs(28992L) else set_crs(4326L)
  })
  # ... and for the downscale data source
  observeEvent(input$src_ds, {
    if (input$mode != "downscale") return()
    if (input$src_ds == "demo") set_crs(demo_ds_info$epsg) else set_crs(4326L)
  })

  # switching task resets the CRS default and clears any previous fit/map
  observeEvent(input$mode, {
    if (input$mode == "downscale") {
      if (input$src_ds == "demo") set_crs(demo_ds_info$epsg) else set_crs(4326L)
    } else if (input$src == "demo") set_crs(28992L) else set_crs(4326L)
    fit(NULL)
  }, ignoreInit = TRUE)

  # auto-fill the CRS from a spatial file's own EPSG, when it carries one
  observeEvent(sample_in(), {
    s <- sample_in()
    if (input$mode == "point" && isTRUE(s$spatial)) set_crs(s$epsg)
  }, ignoreInit = TRUE)
  observeEvent(ds_in(), {
    s <- ds_in()
    if (input$mode == "downscale" && isTRUE(s$spatial)) set_crs(s$epsg)
  }, ignoreInit = TRUE)

  crs_code <- reactive({
    if (input$crs_preset == "other")
      return(paste0("EPSG:", gsub("\\D", "", input$epsg)))
    paste0("EPSG:", input$crs_preset)
  })
  output$crs_note <- renderText("Wrong spot on the map → enter EPSG code.")

  dataset <- reactive({
    df <- raw_sample(); req(df, input$resp, input$cx, input$cy)
    tcol <- input$time_coord
    time <- if (!is.null(tcol) && nzchar(tcol) && tcol %in% names(df))
      as.numeric(df[[tcol]]) else NULL
    is_dglm <- !is.null(time)
    covs <- input$covs; covs <- covs[covs %in% names(df)]
    covs <- setdiff(covs, c(input$resp, input$cx, input$cy, tcol))
    y <- as.numeric(df[[input$resp]]); coords <- as.matrix(df[, c(input$cx, input$cy)])
    x <- if (length(covs)) df[, covs, drop = FALSE] else NULL
    crs <- crs_code()
    g <- raw_grid()
    if (!is.null(g)) {
      coords0 <- as.matrix(g[, c(input$cx, input$cy)])
      x0 <- if (length(covs)) g[, covs, drop = FALSE] else NULL
    } else if (length(covs) == 0) {
      rx <- range(coords[, 1]); ry <- range(coords[, 2])
      rx <- rx + c(-1, 1) * diff(rx) * 0.05   # widen extent by 10% (5% each side)
      ry <- ry + c(-1, 1) * diff(ry) * 0.05
      dx <- diff(rx); dy <- diff(ry)
      # longer axis gets n_long points (<=38 for space-time, else 120); the
      # shorter axis count is scaled by the extent ratio so cells are square.
      n_long <- if (is_dglm)
        max(25L, min(38L, as.integer(floor(sqrt(18000 / length(unique(time)))))))
        else 120L
      if (dx >= dy) {
        nx <- n_long; ny <- max(2L, as.integer(round(1 + (n_long - 1) * dy / dx)))
      } else {
        ny <- n_long; nx <- max(2L, as.integer(round(1 + (n_long - 1) * dx / dy)))
      }
      coords0 <- as.matrix(expand.grid(
        x = seq(rx[1], rx[2], length.out = nx),
        y = seq(ry[1], ry[2], length.out = ny))); x0 <- NULL
    } else { coords0 <- NULL; x0 <- NULL }
    off_col <- input$offset
    has_off <- !is.null(off_col) && nzchar(off_col) && off_col %in% names(df)
    offset  <- if (has_off) as.numeric(df[[off_col]]) else NULL
    offset0 <- if (has_off && !is.null(g) && off_col %in% names(g))
      as.numeric(g[[off_col]]) else NULL
    # space-time: predict the grid at every training time point (grid x times)
    time0 <- NULL
    if (is_dglm && !is.null(coords0)) {
      tt  <- sort(unique(time)); n0 <- nrow(coords0)
      idx <- rep(seq_len(n0), times = length(tt))
      coords0 <- coords0[idx, , drop = FALSE]
      if (!is.null(x0))      x0      <- x0[idx, , drop = FALSE]
      if (!is.null(offset0)) offset0 <- offset0[idx]
      time0 <- rep(tt, each = n0)
    }
    list(y = y, coords = coords, x = x, coords0 = coords0, x0 = x0,
         offset = offset, offset0 = offset0, time = time, time0 = time0, crs = crs)
  })

  # -- fit: produce a plain spCF model object; the module maps it --------------
  fit <- reactiveVal(NULL)
  observeEvent(input$run, {
    d <- tryCatch(dataset(), error = function(e) {
      showNotification(paste("Data error:", conditionMessage(e)), type = "error"); NULL })
    req(d)
    is_dglm <- !is.null(d$time)
    if (!is_dglm && is.null(d$coords0)) {
      showNotification("Covariates selected but no prediction grid: upload a grid CSV with those columns.",
                       type = "error", duration = 8); return(invisible()) }
    if (!all(is.finite(d$y))) {
      showNotification("Response contains non-numeric / NA values.", type = "error"); return(invisible()) }
    fam_obj <- switch(input$family,
                      gaussian         = stats::gaussian(),
                      poisson          = stats::poisson(),
                      binomial         = stats::binomial(),
                      Gamma            = stats::Gamma("log"),
                      inverse.gaussian = stats::inverse.gaussian("log"),
                      quasipoisson     = stats::quasipoisson(),
                      quasibinomial    = stats::quasibinomial())
    mod <- tryCatch(
      withProgress(message = "Running CFSM", value = 0, {
        if (is_dglm) {                              # space-time -> cf_dglm
          incProgress(0.3, detail = "cf_dglm_hv")
          mh <- cf_dglm_hv(y = d$y, x = d$x, coords = d$coords, time = d$time,
                           offset = d$offset, family = fam_obj)
          incProgress(0.6, detail = "cf_dglm prediction")
          cf_dglm(y = d$y, x = d$x, coords = d$coords, time = d$time,
                  offset = d$offset, x0 = d$x0, coords0 = d$coords0,
                  time0 = d$time0, offset0 = d$offset0, mod_hv = mh)
        } else if (input$family == "gaussian") {    # Gaussian -> cf_lm (no offset)
          if (!is.null(d$offset))
            showNotification("Offset is ignored for Gaussian (cf_lm).",
                             type = "warning", duration = 5)
          incProgress(0.3, detail = "cf_lm_hv")
          mh <- cf_lm_hv(y = d$y, x = d$x, coords = d$coords, add_learn = "none")
          incProgress(0.6, detail = "cf_lm prediction")
          cf_lm(y = d$y, x = d$x, x0 = d$x0, coords = d$coords,
                coords0 = d$coords0, mod_hv = mh)
        } else {                                     # otherwise -> cf_glm
          fam <- switch(input$family,
                        poisson          = stats::poisson(),
                        binomial         = stats::binomial(),
                        Gamma            = stats::Gamma("log"),          # stable link
                        inverse.gaussian = stats::inverse.gaussian("log"),
                        quasipoisson     = stats::quasipoisson(),
                        quasibinomial    = stats::quasibinomial())
          incProgress(0.3, detail = "cf_glm_hv")
          mh <- cf_glm_hv(y = d$y, x = d$x, coords = d$coords,
                          offset = d$offset, family = fam)
          incProgress(0.6, detail = "cf_glm prediction")
          cf_glm(y = d$y, x = d$x, x0 = d$x0, coords = d$coords,
                 coords0 = d$coords0, offset = d$offset, offset0 = d$offset0,
                 mod_hv = mh)
        }
      }),
      error = function(e) {
        showNotification(paste("Model error:", conditionMessage(e)),
                         type = "error", duration = 8); NULL })
    req(mod)
    fit(list(mod = mod, crs = d$crs))
  })

  # -- fit: spatial downscaling (cf_downscale) ---------------------------------
  observeEvent(input$run_ds, {
    s <- tryCatch(ds_in(), error = function(e) {
      showNotification(paste("Data error:", conditionMessage(e)), type = "error"); NULL })
    req(s); df <- s$df
    req(df, input$dx, input$dy, input$dresp, input$dagg)
    coords <- as.matrix(df[, c(input$dx, input$dy)])
    agg_id <- df[[input$dagg]]
    Yv     <- as.numeric(df[[input$dresp]])
    if (!all(is.finite(Yv))) {
      showNotification("Aggregate response contains non-numeric / NA values.",
                       type = "error"); return(invisible()) }
    # the aggregate response must be constant within each area; warn otherwise
    # (likely a per-unit column was picked by mistake).
    if (any(tapply(Yv, agg_id, function(z) diff(range(z))) > 1e-9, na.rm = TRUE))
      showNotification(paste("Response varies within some areas; using each",
        "area's first value. Pick the area-level (constant) response column."),
        type = "warning", duration = 8)
    # area-level Y: one value per area, ordered by aggregate()'s grouping —
    # matching cf_downscale's internals.
    Yagg <- as.numeric(stats::aggregate(Yv, by = list(agg_id),
                                        FUN = function(z) z[1])[, 2])
    covs <- input$dcovs; covs <- covs[covs %in% names(df)]
    covs <- setdiff(covs, c(input$dx, input$dy, input$dresp, input$dagg, input$dweight))
    x  <- if (length(covs)) df[, covs, drop = FALSE] else NULL
    pw <- if (!is.null(input$dweight) && nzchar(input$dweight) &&
              input$dweight %in% names(df)) as.numeric(df[[input$dweight]]) else NULL
    crs <- crs_code()
    md <- tryCatch(
      withProgress(message = "Running CF-DS", value = 0, {
        incProgress(0.3, detail = "cf_downscale_hv")
        mh <- cf_downscale_hv(Y = Yagg, Y_type = input$dytype, x = x,
                              prop_weight = pw, coords = coords, agg_id = agg_id)
        incProgress(0.6, detail = "cf_downscale")
        cf_downscale(Y = Yagg, x = x, prop_weight = pw, coords = coords,
                     agg_id = agg_id, mod_hv = mh)
      }),
      error = function(e) {
        showNotification(paste("Model error:", conditionMessage(e)),
                         type = "error", duration = 8); NULL })
    req(md)
    # polygon geometry (lon/lat, upload order) to map the result as a choropleth
    geom4326 <- if (isTRUE(s$is_poly) && !is.null(s$geom))
      tryCatch(sf::st_transform(s$geom, 4326), error = function(e) NULL) else NULL
    fit(list(mod = md, crs = crs, geom = geom4326))
  })

  # -- data preview before fitting: polygons if the GeoJSON is polygon, else
  #    points (downscale is coloured by Area ID; other cases single colour) -----
  preview_pts <- reactive({
    if (!is.null(fit())) return(NULL)
    cc <- tryCatch(crs_code(), error = function(e) NA); req(!is.na(cc))
    to4326 <- function(x) tryCatch(st_transform(x, 4326), error = function(e) NULL)
    if (input$mode == "downscale") {
      s <- tryCatch(ds_in(), error = function(e) NULL); req(s)
      df <- s$df
      grp <- if (!is.null(input$dagg) && input$dagg %in% names(df))
               as.factor(df[[input$dagg]]) else NULL
      if (isTRUE(s$is_poly) && !is.null(s$geom)) {
        g <- to4326(s$geom); req(g); if (!is.null(grp)) g$.grp <- grp; return(g)
      }
      req(input$dx, input$dy)
      if (!is.null(grp)) df$.grp <- grp
      return(to4326(st_as_sf(df, coords = c(input$dx, input$dy), crs = cc)))
    }
    s <- tryCatch(sample_in(), error = function(e) NULL)
    df <- if (!is.null(s)) s$df else tryCatch(raw_sample(), error = function(e) NULL)
    req(df)
    if (!is.null(s) && isTRUE(s$is_poly) && !is.null(s$geom)) {
      g <- to4326(s$geom); req(g); return(g)
    }
    req(input$cx, input$cy)
    to4326(st_as_sf(df, coords = c(input$cx, input$cy), crs = cc))
  })

  # -- bottom panel: data table (before a fit) / model summary (after) ---------
  output$has_fit <- reactive(!is.null(fit()))
  outputOptions(output, "has_fit", suspendWhenHidden = FALSE)
  output$bottom_title <- renderText(
    if (is.null(fit())) "Data table" else "Model summary")

  active_data <- reactive({
    if (input$mode == "downscale") {
      s <- tryCatch(ds_in(), error = function(e) NULL)
      if (is.null(s)) NULL else s$df
    } else tryCatch(raw_sample(), error = function(e) NULL)
  })
  output$data_tbl <- renderDataTable({
    d <- active_data()
    validate(need(!is.null(d) && nrow(d) > 0, "Load data to preview."))
    # round numeric columns to 4 decimals for display, but leave columns that
    # are entirely tiny (max |x| < 1e-3, e.g. 1e-8) at full precision.
    num <- vapply(d, is.numeric, logical(1))
    d[num] <- lapply(d[num], function(v) {
      a <- abs(v[is.finite(v) & v != 0])
      if (length(a) && max(a) < 1e-3) v else round(v, 4)
    })
    d
  }, options = list(pageLength = 8, scrollX = TRUE, lengthChange = FALSE))

  # -- hand the fitted model to the shared mapping module ----------------------
  sp_map_server("map",
                mod     = reactive(fit()$mod),
                crs     = reactive(fit()$crs),
                preview = preview_pts,
                home    = DEMO_HOME,
                geom    = reactive(fit()$geom))
}

shinyApp(ui, server)
