#' Interactive mapping for coarse-to-fine spatial modelling
#'
#' Opens a Shiny web app for mapping CFSM results over a basemap. The function
#' has two modes:
#'
#' \describe{
#'   \item{\code{spCFmap()}}{Without \code{mod}, the full application is
#'   launched: models (\code{\link{cf_lm}}, \code{\link{cf_glm}},
#'   \code{\link{cf_dglm}}, \code{\link{cf_downscale}}) are fitted inside the app
#'   from demo data (meuse, a space-time air-quality set, and an areal
#'   downscaling set) or from user CSV / GeoJSON uploads, and predictions can be
#'   exported as CSV or GeoJSON.}
#'   \item{\code{spCFmap(mod, crs)}}{With a fitted model, a small app maps that
#'   model directly. The layer (predictive mean / SD, covariate effect, or a
#'   scale-wise spatial component), colour scaling, and - for space-time or
#'   downscaling fits - the time range or bandwidth range are chosen
#'   interactively.}
#' }
#'
#' @param mod Optional fitted model returned by \code{\link{cf_lm}},
#'   \code{\link{cf_glm}}, \code{\link{cf_dglm}} or \code{\link{cf_downscale}}.
#'   If omitted, the full data-upload and model-fitting application is launched.
#' @param crs Coordinate reference system of the coordinates that were passed
#'   to the model: an EPSG code (e.g. \code{4326} or \code{"EPSG:4326"}) or a
#'   proj/WKT string, used to place the results on the longitude/latitude
#'   basemap. Give \code{4326} when the model was fitted on raw
#'   longitude/latitude, and the code of the projected system (e.g.
#'   \code{28992}, \code{27700}, \code{25832}) when it was fitted on projected
#'   coordinates; an EPSG code assumes the coordinates are in the unit of that
#'   system, so rescaled coordinates (metres divided by 1000, say) need a
#'   proj string such as \code{"+proj=utm +zone=32 +datum=WGS84 +units=km"}.
#'   There is deliberately no default: coordinates carry no unit of their own,
#'   and guessing would silently place the map in the wrong part of the world,
#'   so \code{crs} must be supplied whenever \code{mod} is given. It is ignored
#'   when \code{mod} is omitted, since the application asks for the coordinate
#'   reference system interactively -- there, EPSG:4326 (longitude/latitude) is
#'   the default offered for uploaded files, while the bundled demo data sets
#'   preselect their own systems.
#' @param launch Logical; if \code{TRUE} (default) run the app, otherwise return
#'   the Shiny app object without launching.
#' @param ... Passed to \code{\link[shiny]{runApp}} (e.g. \code{port},
#'   \code{host}, \code{launch.browser}). Unused when \code{launch = FALSE}.
#'
#' @return If \code{launch = TRUE}, the value returned by
#'   \code{\link[shiny]{runApp}} (invisibly); otherwise a \code{shiny.appobj}.
#'
#' @seealso \code{\link{cf_lm}}, \code{\link{cf_glm}}, \code{\link{cf_dglm}},
#'   \code{\link{cf_downscale}}, \code{\link{sp_scalewise}}
#'
#' @examples
#' \dontrun{
#' spCFmap()                       # full app, opens in the browser
#' spCFmap(launch.browser = FALSE) # print the local URL instead
#'
#' m <- cf_lm(y = y, x = x, x0 = x0, coords = coords, coords0 = coords0,
#'            mod_hv = cf_lm_hv(y = y, x = x, coords = coords))
#' spCFmap(m, crs = 28992)         # map an already-fitted model
#' }
#'
#' @export
spCFmap <- function(mod = NULL, crs = NULL, launch = TRUE, ...) {
  need_pkgs <- function(need, what) {
    miss <- need[!vapply(need, requireNamespace, logical(1), quietly = TRUE)]
    if (length(miss))
      stop(what, " needs the package(s): ", paste(miss, collapse = ", "),
           ".\nInstall with install.packages(c(",
           paste(sprintf('"%s"', miss), collapse = ", "), ")).", call. = FALSE)
  }

  if (is.null(mod)) {
    if (!is.null(crs))
      warning("'crs' is ignored when 'mod' is not supplied; the full app asks ",
              "for the CRS interactively.", call. = FALSE)
    need_pkgs(c("shiny", "bslib", "leaflet", "terra", "sf", "sp"), "spCFmap()")

    app_dir <- system.file("shiny", "spCFmap", package = "spCF")
    if (!nzchar(app_dir) || !file.exists(file.path(app_dir, "app.R")))
      stop("spCFmap app files not found; please reinstall spCF.", call. = FALSE)
    app <- shiny::shinyAppDir(app_dir)

  } else {
    if (is.null(crs))
      stop("'crs' must be supplied when mapping a fitted model, e.g. ",
           "spCFmap(mod, crs = 4326).", call. = FALSE)
    need_pkgs(c("shiny", "leaflet", "terra", "sf"), "spCFmap(mod, crs)")
    app <- sp_map_app(mod, crs)
  }

  if (isTRUE(launch)) shiny::runApp(app, ...) else app
}
