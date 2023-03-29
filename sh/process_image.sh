#!/bin/bash

cp base_html/* $2
python src/gdal2tiles-leaflet/gdal2tiles.py -l -p raster -w none $1 -z 0-5 $2/tiles/
