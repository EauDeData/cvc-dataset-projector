#!/bin/bash

mkdir $2; cd $2
while read p; do
  wget "$p"
done <$1
