(ns mxnet-made-simple.pretrained
  (:require [clojure.string :as string]
            [org.apache.clojure-mxnet.context :as ctx]
            [org.apache.clojure-mxnet.module :as m]
            [org.apache.clojure-mxnet.ndarray :as ndarray]
            [org.apache.clojure-mxnet.visualization :as viz]
            [opencv4.core :as cv]
            [opencv4.utils :as cvu]))

(def model-dir "models/")

(def h 224) ;; Image height
(def w 224) ;; Image width
(def c 3)   ;; Number of channels: Red, Green, Blue

(def ctx (ctx/gpu 0))

(defonce vgg-16-mod
  (-> {:prefix (str model-dir "vgg16") :epoch 0 :contexts [ctx]}
      (m/load-checkpoint)
      ;; Define the shape of input data and bind the name of the input layer
      ;; to "data"
      (m/bind {:for-training false
               :data-shapes [{:name "data" :shape [1 c h w]}]})))

(defonce inception-mod
  (-> {:prefix (str model-dir "Inception-BN") :epoch 0 :contexts [ctx]}
      (m/load-checkpoint)
      ;; Define the shape of input data and bind the name of the input layer
      ;; to "data"
      (m/bind {:for-training false
               :data-shapes [{:name "data" :shape [1 c h w]}]})))

#_ (m/params inception-mod)

(defn render-model!
  "Render the `model-sym` and saves it as a pdf file in `path/model-name.pdf`"
  [{:keys [model-name model-sym input-data-shape path]}]
  (let [dot (viz/plot-network
             model-sym
             {"data" input-data-shape}
             {:title model-name
              :node-attrs {:shape "oval" :fixedsize "false"}})]
    (viz/render dot model-name path)))


#_ (render-model! {:model-name "inception" 
                :model-sym  (m/symbol inception-mod)
                :input-data-shape [1 c h w] 
                :path model-dir})


(defonce image-net-labels
  (-> (str model-dir "/synset.txt")
      (slurp)
      (string/split #"\n")))



(defn preprocess-img-mat
  "Preprocessing steps on an `img-mat` from OpenCV to feed into the Model"
  [img-mat]
  (-> img-mat
      ;; Resize image to (w, h)
      (cv/resize! (cv/new-size w h))
      ;; Maps pixel values from [-128, 128] to [0, 127]
      (cv/convert-to! cv/CV_8SC3 0.5)
      ;; Subtract mean pixel values from ImageNet dataset
      (cv/add! (cv/new-scalar -103.939 -116.779 -123.68))
      ;; Flatten matrix
      (cvu/mat->flat-rgb-array)
      ;; Reshape to (1, c, h, w)
      (ndarray/array [1 c h w] {:ctx ctx})))

(defn- top-k
  "Return top `k` from prob-maps with :prob key"
  [k prob-maps]
  (->> prob-maps
       (sort-by :prob)
       (reverse)
       (take k)))

(defn predict
  "Predict with `model` the top `k` labels from `labels` of the ndarray `x`"
  ([model labels x]
   (predict model labels x 5))
  ([model labels x k]
   (let [probs (-> model
                   (m/forward {:data [x]})
                   (m/outputs)
                   (ffirst)
                   (ndarray/->vec))
         prob-maps (mapv (fn [p l] {:prob p :label l}) probs labels)]
     (top-k k prob-maps))))



#_ (->> "images/cat.jpg"
     (cv/imread)
     (preprocess-img-mat)
     (predict inception-mod image-net-labels))

#_(->> "images/dog.jpg"
     (cv/imread)
     (preprocess-img-mat)
     (predict inception-mod image-net-labels))