# Deploying spCFmap

This directory is the deployment unit for the `spCFmap()` application. It holds
one file, `app.R`, which loads the spCF package and hands back its Shiny app
object. Deploy the directory, not the package.

## shinyapps.io

```r
install.packages("rsconnect")
rsconnect::setAccountInfo(name = "<account>", token = "<token>", secret = "<secret>")
rsconnect::deployApp("deploy", appName = "spCFmap")
```

The token comes from the shinyapps.io dashboard, under Account → Tokens.

## Posit Connect / Shiny Server

Copy this directory to the server and point the host at it. Shiny Server needs
`spCF` and the packages listed in `app.R` installed in the R library the server
uses.

## What the server needs

`spCF` comes from CRAN, and so do the rest. `terra` and `sf` additionally need
GDAL, PROJ and GEOS present on the machine. shinyapps.io provides them; a
hand-rolled Docker image usually does not, and a PROJ database that the R
packages cannot find is the most common way a container-hosted deployment
breaks. `spCFmap()` checks for that at startup and stops with an explicit
message rather than failing later inside the map.

## Why app.R repeats the library() calls

rsconnect decides what to install on the server by scanning the deployed
directory. It cannot see the `library()` calls inside the package's own app.R,
so `library(spCF)` alone yields a deployment with spCF but without
terra/sf/leaflet, which then fails at startup. Keep the list as it is.

## Resource limits

Users can upload their own CSV or GeoJSON; the app accepts files up to 100 MB
and fits the model server-side, so the host's memory is the real constraint.
For scale: a fit over 2,769 observations at 60 time points (the bundled air
demo) produces a ~21 MB model object, while a 267,907-observation space-time
fit over 15,220 sites produces one of ~831 MB and needs several GB while
running.

The shinyapps.io free tier gives 1 GB per instance, shared across everyone
connected at once, and disconnects idle sessions. Demo-sized data and uploads
of a few thousand observations are comfortable there; anything approaching the
larger figure above wants a paid plan, or is better mapped locally with
`spCFmap(mod, crs)` on an already-fitted model.
