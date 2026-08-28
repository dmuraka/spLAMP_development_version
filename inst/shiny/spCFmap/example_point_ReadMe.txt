spCFmap() -- example data for point prediction
==============================================


1. WHAT TO UPLOAD
-----------------

Sample points   (required)   CSV or GeoJSON
    One row per observed location, carrying
      - an X coordinate and a Y coordinate    (numeric)
      - the response to be modelled           (numeric)
      - optionally: covariates, a time coordinate, an offset

Prediction grid (optional)   CSV or GeoJSON
    One row per location you want predicted, carrying the same coordinate
    columns and the SAME covariate columns as the sample points. Leave it
    out and the app builds a regular grid over the sampled area instead.


Worth knowing
    * Column NAMES are free. After the upload you say which column plays
      which role -- "X-coordinate", "Response (y)", and so on -- in the
      sidebar. Nothing has to be called anything in particular.

    * The response belongs in the sample points only, never in the grid.

    * If you select covariates, the prediction grid must contain those same
      columns; there is nothing to predict from otherwise, and the app will
      say so.

    * Coordinates are read in the system chosen under "Coordinate system".
      EPSG:4326 (longitude/latitude) is the default; a projected system such
      as EPSG:28992 is equally fine. Give the code the coordinates are
      actually in -- nothing is guessed, and a wrong code silently puts the
      map in the wrong part of the world.

    * A GeoJSON may hold points or polygons. Polygons are computed on their
      centroids and drawn as polygons.

    * Supply a time coordinate to fit a space-time model (cf_dglm). The same
      locations should repeat at each time point.


2. THE EXAMPLE FILES
--------------------

Meuse heavy-metal data (R package 'sp'): 155 sampled sites on the river
Meuse floodplain, plus the prediction grid that ships with it.

example_samples.csv     155 rows
    px_epsg28992, py_epsg28992   coordinates in EPSG:28992 (metres)
    lon, lat                     the same sites in EPSG:4326
    logzinc                      response: log topsoil zinc, 4.73 .. 7.52
    dist                         covariate: distance to the river, 0 .. 0.88
    ffreq2, ffreq3               covariates: flood-frequency flags, 0 or 1

example_grid.csv       3103 rows
    the same coordinate and covariate columns, and no response column --
    which is exactly the shape a prediction grid should have.

example_points.geojson / example_grid.geojson
    the same two data sets as GeoJSON.


To reproduce the bundled demo from the uploads: choose EPSG:28992 as the
coordinate system, upload example_samples.csv and example_grid.csv, then map
px_epsg28992 / py_epsg28992 to the coordinates, logzinc to the response, and
dist / ffreq2 / ffreq3 to the covariates. Using lon / lat with EPSG:4326
instead works just as well.
