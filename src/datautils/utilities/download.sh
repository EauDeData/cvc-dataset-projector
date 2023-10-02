#!/bin/bash

# Create the download/ folder if it doesn't exist
mkdir -p download

# Read each link from the links.txt file and download it into download/
while read link; do
  echo "Downloading $link"
  wget -q --show-progress -P download/ $link || echo "Failed to download $link"
done < links.txt

