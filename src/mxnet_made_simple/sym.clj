(ns mxnet-made-simple.sym
  (:require [org.apache.clojure-mxnet.dtype :as d]
            [org.apache.clojure-mxnet.ndarray :as ndarray]
            [org.apache.clojure-mxnet.symbol :as sym]
            [org.apache.clojure-mxnet.visualization :as viz]
            [org.apache.clojure-mxnet.executor :as executor]
            [org.apache.clojure-mxnet.context :as context]))

;; Define Input data as Variable
(def a (sym/variable "A"))
(def b (sym/variable "B"))
(def c (sym/variable "C"))
(def d (sym/variable "D"))

;; Define a Computation Graph: e = (a * b) + (c * d)
(def e
  (sym/+
   (sym/* a b)
   (sym/* c d)))

#_(sym/list-arguments e) ;["A" "B" "C" "D"]
#_ (sym/list-outputs e)
#_(sym/list-outputs (sym/get-internals e))

(defn render-computation-graph!
  "Render the `sym` and saves it as a pdf file in `path/sym-name.pdf`"
  [{:keys [sym-name sym input-data-shape path]}]
  (let [dot (viz/plot-network
             sym
             input-data-shape
             {:title sym-name
              :node-attrs {:shape "oval" :fixedsize "false"}})]
    (viz/render dot sym-name path)))

#_(render-computation-graph!
 {:sym-name "e"
  :sym e
  :input-data-shape {"A" [1] "B" [1] "C" [1] "D" [1]}
  :path "/tmp"})


(def ctx  (context/gpu 0))
(def data-binding
  {"A" (ndarray/array [1] [1] {:dtype d/INT32 :ctx ctx})
   "B" (ndarray/array [2] [1] {:dtype d/INT32 :ctx ctx})
   "C" (ndarray/array [3] [1] {:dtype d/INT32 :ctx ctx})
   "D" (ndarray/array [4] [1] {:dtype d/INT32 :ctx ctx})})

#_(-> e
    (sym/bind ctx data-binding)
    executor/forward
    executor/outputs
    first
    ndarray/->vec)

#_(let [symbol-filename "/tmp/symbol-e.json"]
  ;; Saving to disk symbol `e`
  (sym/save e symbol-filename)
  ;; Loading from disk symbol `e`
  (let [e2 (sym/load symbol-filename)]
    (println (= (sym/to-json e) (sym/to-json e2))) ;true
    ))