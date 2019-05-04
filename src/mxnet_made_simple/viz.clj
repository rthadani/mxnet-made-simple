(ns mxnet-made-simple.viz
  (:require [org.apache.clojure-mxnet.visualization :as viz]
            [org.apache.clojure-mxnet.module :as m]))

(def model-dir "models")

(def vgg16-mod
  "VGG16 Module"
  (m/load-checkpoint {:prefix (str model-dir "/vgg16") :epoch 0}))

(def resnet18-mod
  "Resnet18 Module"
  (m/load-checkpoint {:prefix (str model-dir "/resnet-18") :epoch 0}))