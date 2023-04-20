sudo add-apt-repository ppa:ubuntugis/ppa
sudo apt-get update
sudo apt-get install gdal-bin

sudo apt install libpq-dev
sudo apt-get install libgdal-dev
export CPLUS_INCLUDE_PATH=/usr/include/gdal
export C_INCLUDE_PATH=/usr/include/gdal
conda install -c conda-forge gcc=12.1.0
pip install GDAL==3.5.1
pip install pygdal==3.5.1
