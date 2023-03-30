# cvc-dataset-projector

## How To Run

0. Install the ```requirements.txt```. Some months ago only god and I knew how GDAL was installed, now only god does! So pray your bests so it doesn't crash. 
  - Something I discovered is that if you first go like ```sudo apt-get install libgdal-dev; export CPLUS_INCLUDE_PATH=/usr/include/gdal; export C_INCLUDE_PATH=/usr/include/gdal``` and then install the requirements it usually works.
1. Generate the image by running ```python create_image.py``` under your collection compressed in a *.zip file.
2. Select the folder where the tiles will be processed and just run ```./sh/process_image.sh <generated_image> <target_folder>```

Voilà
