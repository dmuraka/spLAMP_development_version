spCFmap() -- example data for spatial downscaling
=================================================

Downscaling takes a response observed over COARSE areas and estimates it at
the FINE units inside them, using covariates that vary between those units.


1. WHAT TO UPLOAD
-----------------

Disaggregate units  (required)   CSV or GeoJSON
    One row per FINE unit -- the level you want results at -- carrying
      - an X coordinate and a Y coordinate    (numeric)
      - an area ID naming the coarse area the row belongs to
      - the response, observed at the COARSE level
      - optionally: covariates, a proportional weight

There is only this one upload: the coarse areas are not a separate file,
they are named by the area ID column.


Worth knowing
    * Column NAMES are free. After the upload you say which column plays
      which role -- "Area ID", "Aggregate response (Y)", and so on.

    * The response is an AREA-level value: every row sharing an area ID must
      repeat the same number. Picking a per-unit column here is the easiest
      mistake to make, and the app will tell you when it happens.

    * Covariates are the opposite -- they must VARY between the fine units
      of an area. That variation is what pulls the area's value apart.

    * A proportional weight (population, area, ...) sets how the area value
      is split across its units before the covariates are applied. Without
      one the split starts out even.

    * Areas need not hold the same number of units.

    * A polygon GeoJSON is computed on the centroids and drawn as polygons;
      a CSV of centroids is drawn as points.


2. THE EXAMPLE FILES
--------------------

example_downscale.csv    400 rows, 40 areas, sorted by area ID
    lon, lat      coordinates of the fine units (EPSG:4326)
    agg_id        area ID; areas hold between 3 and 18 units
    income, pop   covariates, varying between the units of an area
    Y             the response, constant within each agg_id

    The file is sorted by agg_id, so the rule above is visible at a glance:
    the rows come grouped by area, and Y simply repeats down each group
    while income and pop change from row to row.

example_downscale.geojson    271 polygons, 30 areas
    Glasgow intermediate zones (R package 'CARBayesdata') -- the data behind
    "Demo (pollutionhealthdata)". Fields: agg_id, pm10, jsa, price, expected,
    Y. Upload it to see the polygon path, where results are mapped as filled
    zones rather than points.


To try the CSV: choose EPSG:4326 as the coordinate system, upload
example_downscale.csv, then map lon / lat to the coordinates, agg_id to the
area ID, Y to the aggregate response, and income / pop to the covariates.
