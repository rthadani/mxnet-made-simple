#!/bin/bash

set -evx

PROJECT_ROOT=$(cd "$(dirname $0)/../.."; pwd)

data_path=$PROJECT_ROOT/data/

if [ ! -d "$data_path" ]; then
    mkdir -p "$data_path"
fi


if [ ! -f "$data_path/saint_bernard/saint_bernard_33.jpg" ]; then

pushd $data_path

# Downloading the dataset
wget https://s3.amazonaws.com/fast-ai-imageclas/oxford-iiit-pet.tgz
tar zxvf oxford-iiit-pet.tgz
rm oxford-iiit-pet.tgz
mv oxford-iiit-pet/images/* .
rm -rf oxford-iiit-pet
rm *.mat

# Organizing images into folders
for image in *jpg ; do
  label=`echo $image | awk -F_ '{gsub($NF,"");sub(".$", "");print}'`
  mkdir -p $label
  mv $image $label/$image
done

popd

fi