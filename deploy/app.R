## spCFmap - deployment entry point
##
## Target: shinyapps.io, Posit Connect, or any Shiny Server. Deploy THIS
## directory; the application itself lives inside the spCF package and this file
## only launches it.
##
## The library() calls below are deliberate and must not be trimmed. rsconnect
## works out what to install on the server by scanning the deployed directory,
## and it cannot see the library() calls inside the package's own app.R. With
## only library(spCF) the deployment installs spCF but not terra/sf/leaflet, and
## the app dies at startup with "spCFmap() needs the package(s): ...".
library(spCF)
library(shiny)
library(bslib)
library(leaflet)
library(terra)
library(sf)
library(sp)

## Returns a shiny.appobj (the app is NOT run here) -- that object is what a
## Shiny host expects app.R to evaluate to.
spCFmap(launch = FALSE)
