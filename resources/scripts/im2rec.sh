#!/bin/bash
MXNET_HOME=~/personal/incubator-mxnet
PROJECT_ROOT=$(cd "$(dirname $0)/../.."; pwd)

data_path=$PROJECT_ROOT/data/

if [ ! -f "$data_path/data_train2.lst" ]; then

  # Cleaning up the images that are failing with OpenCV
  rm -f $data_path/Abyssinian/Abyssinian_34.jpg
  rm -f $data_path/Egyptian_Mau/Egyptian_Mau_139.jpg
  rm -f $data_path/Egyptian_Mau/Egyptian_Mau_145.jpg
  rm -f $data_path/Egyptian_Mau/Egyptian_Mau_167.jpg
  rm -f $data_path/Egyptian_Mau/Egyptian_Mau_177.jpg
  rm -f $data_path/Egyptian_Mau/Egyptian_Mau_191.jpg

  python $MXNET_HOME/tools/im2rec.py \
    --list \
    --train-ratio 0.8 \
    --recursive \
    $data_path/data $data_path

  python $MXNET_HOME/tools/im2rec.py \
    --resize 224 \
    --center-crop \
    --num-thread 4 \
    $data_path/data $data_path

fi