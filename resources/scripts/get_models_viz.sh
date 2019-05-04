#!/bin/bash

set -evx

mkdir -p ../models
cd ../models

wget http://data.mxnet.io/models/imagenet/vgg/vgg16-symbol.json
wget http://data.mxnet.io/models/imagenet/vgg/vgg16-0000.params
wget http://data.mxnet.io/models/imagenet/resnet/18-layers/resnet-18-symbol.json
wget http://data.mxnet.io/models/imagenet/resnet/18-layers/resnet-18-0000.params

cd -

