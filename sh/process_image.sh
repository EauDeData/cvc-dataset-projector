#!/bin/bash

dir=${3:-.tmp/}

cp -r base_html/* $2
cp index.pkl $2 
cp output/index.ann $2
cd $2; cd "web"; ln -s "../../$dir" .; cd ../../
python src/gdal2tiles-leaflet/gdal2tiles.py -l -p raster -w none $1 -z 0-5 $2/web/tiles/
