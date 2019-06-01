(ns mxnet-made-simple.viz
  (:require [org.apache.clojure-mxnet.visualization :as viz]
            [org.apache.clojure-mxnet.module :as m]
            [org.apache.clojure-mxnet.context :as context]))

(def model-dir "models")
(def ctx  (context/gpu 0))

(defn render-model!
  "Render the `model-sym` and saves it as a pdf file in `path/model-name.pdf`"
  [{:keys [model-name model-sym input-data-shape path]}]
  (let [dot (viz/plot-network
             model-sym
             {"data" input-data-shape}
             {:title model-name
              :node-attrs {:shape "oval" :fixedsize "false"}})]
    (viz/render dot model-name path)))


(def vgg16-mod
  "VGG16 Module"
  (m/load-checkpoint {:prefix (str model-dir "/vgg16") :epoch 0 :contexts [ctx]} ))

(def resnet18-mod
  "Resnet18 Module"
  (m/load-checkpoint {:prefix (str model-dir "/resnet-18") :epoch 0 :contexts [ctx]}))

(def model-render-dir "model_render")

;; Rendering pretrained VGG16
(render-model! {:model-name "vgg16"
                :model-sym (m/symbol vgg16-mod)
                :input-data-shape [1 3 244 244]
                :path model-render-dir})

;; Rendering pretrained Resnet18
(render-model! {:model-name "resnet18"
                :model-sym (m/symbol resnet18-mod)
                :input-data-shape [1 3 244 244]
                :path model-render-dir})