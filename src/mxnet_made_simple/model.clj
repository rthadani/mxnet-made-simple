(ns mxnet-made-simple.model
  (:require [org.apache.clojure-mxnet.context :as context]
            [org.apache.clojure-mxnet.ndarray :as ndarray]
            [org.apache.clojure-mxnet.shape :as mx-shape]
            [org.apache.clojure-mxnet.random :as random]
            [org.apache.clojure-mxnet.symbol :as sym] 
            [org.apache.clojure-mxnet.io :as mx-io]
            [org.apache.clojure-mxnet.module :as m]
            [org.apache.clojure-mxnet.initializer :as initializer]
            [org.apache.clojure-mxnet.optimizer :as optimizer]
                [org.apache.clojure-mxnet.eval-metric :as eval-metric]))

(def sample-size 1000)
(def train-size 800)
(def valid-size (- sample-size train-size))
(def feature-count 100)
(def category-count 10)
(def batch-size 10)

(def ctx (context/cpu))

(def X (random/uniform 0 1 [sample-size feature-count]))
(def Y
  (-> sample-size
      (repeatedly #(rand-int category-count))
      (ndarray/array [sample-size])))

#_ (ndarray/shape-vec X) ;[1000 100]
#_ (take 10 (ndarray/->vec X)) 

#_ (ndarray/shape-vec Y) ;[1000]
#_ (take 10 (ndarray/->vec Y)) 

(def X-train
    (ndarray/crop X
                  (mx-shape/->shape [0 0])
                  (mx-shape/->shape [train-size feature-count])))

(def X-valid
  (ndarray/crop X
                (mx-shape/->shape [train-size 0])
                (mx-shape/->shape [sample-size feature-count])))


(def Y-train
  (ndarray/crop Y
                (mx-shape/->shape [0])
                (mx-shape/->shape [train-size])))

(def Y-valid
  (ndarray/crop Y
                (mx-shape/->shape [train-size])
                (mx-shape/->shape [sample-size])))

#_(ndarray/shape-vec X-train) ;[800 100]
#_(take 10 (ndarray/->vec X-train)) ;(0.36371076 0.32504722 0.57019675 0.038425427 0.43860152 0.63427407 0.9883738 0.95894927 0.102044806 0.6527903)
#_(ndarray/shape-vec X-valid) ;[200 100]
#_(take 10 (ndarray/->vec X-valid)) ;(0.36371076 0.32504722 0.57019675 0.038425427 0.43860152 0.63427407 0.9883738 0.95894927 0.102044806 0.6527903)
#_(ndarray/shape-vec Y-train) ;[800]
#_(take 10 (ndarray/->vec Y-train)) ;(9.0 1.0 8.0 8.0 6.0 3.0 1.0 2.0 4.0 9.0)
#_(ndarray/shape-vec Y-valid) ;[200]
#_(take 10 (ndarray/->vec Y-valid))

(defn get-symbol []
  (as-> (sym/variable "data") data
    (sym/fully-connected "fc1" {:data data :num-hidden 128})
    (sym/activation "act1" {:data data :act-type "relu"})
    (sym/fully-connected "fc2" {:data data :num-hidden category-count})
    (sym/softmax-output "softmax" {:data data})))

(def train-iter
  (mx-io/ndarray-iter [X-train]
                      {:label-name "softmax_label"
                       :label [Y-train]
                       :data-batch-size batch-size}))

(def valid-iter
  (mx-io/ndarray-iter [X-valid]
                      {:label-name "softmax_label"
                       :label [Y-valid]
                       :data-batch-size batch-size}))

(def model-module (m/module (get-symbol) {:contexts [ctx]}))

;;; Training the Model
(defn train! [model-module]
  (-> model-module
      (m/bind {:data-shapes (mx-io/provide-data train-iter)
               :label-shapes (mx-io/provide-label train-iter)})
      ;; Initializing weights with Xavier
      (m/init-params {:initializer (initializer/xavier)})
      ;; Choosing Optimizer Algorithm: SGD with lr = 0.1
      (m/init-optimizer {:optimizer (optimizer/sgd {:learning-rate 0.1})})
      ;; Training for `num-epochs`
      (m/fit {:train-data train-iter :eval-data valid-iter :num-epoch 50})))

#_ (train! model-module)


;;; Validating the Model
#_(m/score model-module
         {:eval-data valid-iter
          :eval-metric (eval-metric/accuracy)}) ;["accuracy" 0.09]

(def save-prefix "my-model")

;;;Saving to disk
#_(m/save-checkpoint model-module
                   {:prefix save-prefix
                    :epoch 50
                    :save-opt-states true})


#_(def model-module-2
    (m/load-checkpoint {:prefix save-prefix
                        :epoch 50
                        :load-optimizer-states true}))