(ns mxnet-made-simple.autoencoder
  (:require [org.apache.clojure-mxnet.io :as mx-io]
            [org.apache.clojure-mxnet.symbol :as sym]
            [org.apache.clojure-mxnet.visualization :as viz]))

(def data-dir = "data")
(def train-data (mx-io/mnist-iter {:image (str data-dir "train-images-idx3-ubyte")
                                   :label (str data-dir "train-labels-idx1-ubyte")
                                   :input-shape [784]
                                   :flat true
                                   :batch-size batch-size
                                   :shuffle true}))

(def test-data (mx-io/mnist-iter {:image (str data-dir "t10k-images-idx3-ubyte")
                                  :label (str data-dir "t10k-labels-idx1-ubyte")
                                  :input-shape [784]
                                  :batch-size batch-size
                                  :flat true
                                  :shuffle true}))

(defn get-symbol []
  (as-> input data
    ;; encode
    (sym/fully-connected "encode1" {:data data :num-hidden 100})
    (sym/activation "sigmoid1" {:data data :act-type "sigmoid"})

    ;; encode
    (sym/fully-connected "encode2" {:data data :num-hidden 50})
    (sym/activation "sigmoid2" {:data data :act-type "sigmoid"})

    ;; decode
    (sym/fully-connected "decode1" {:data data :num-hidden 50})
    (sym/activation "sigmoid3" {:data data :act-type "sigmoid"})

    ;; decode
    (sym/fully-connected "decode2" {:data data :num-hidden 100})
    (sym/activation "sigmoid4" {:data data :act-type "sigmoid"})

    ;;output
    (sym/fully-connected "result" {:data data :num-hidden 784})
    (sym/activation "sigmoid5" {:data data :act-type "sigmoid"})

    (sym/linear-regression-output {:data data :label output})))

(def data-desc (first (mx-io/provide-data-desc train-data)))

(def model (-> (m/module (get-symbol) {:data-names ["input"] :label-names ["input_"]})
               (m/bind {:data-shapes [(assoc data-desc :name "input")]
                        :label-shapes [(assoc data-desc :name "input_")]})
               (m/init-params {:initializer  (initializer/uniform 1)})
               (m/init-optimizer {:optimizer (optimizer/adam {:learning-rage 0.001})})))


