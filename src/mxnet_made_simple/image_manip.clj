(ns mxnet-made-simple.image-manip
  (:require [clojure.java.io :as io]
            [org.apache.clojure-mxnet.image :as mx-img]
            [org.apache.clojure-mxnet.ndarray :as ndarray]
            [org.apache.clojure-mxnet.shape :as mx-shape]
            
            [opencv4.colors.rgb :as rgb]
            [opencv4.mxnet :as mx-cv]
            [opencv4.core :as cv]
            [opencv4.utils :as cvu])
  (:import (org.opencv.core Mat)
           (java.awt.image DataBufferByte)))

(defn download!
  "Download `uri` and store it in `filename` on disk"
  [uri filename]
  (with-open [in (io/input-stream uri)
              out (io/output-stream filename)]
    (io/copy in out)))

#_ (download! "https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/doc/tutorials/python/predict_image/cat.jpg" "images/cat.jpg")
#_ (download! "https://github.com/dmlc/web-data/blob/master/mxnet/doc/tutorials/python/predict_image/dog.jpg?raw=true" "images/dog.jpg")

(defn preview!
  "Preview image from `filename` and display it on the screen in a new window
   Ex:
    (preview! \"images/cat.jpg\")
    (preview! \"images/cat.jpg\" :h 300 :w 200)"
  ([filename]
   (preview! filename {:h 400 :w 400}))
  ([filename {:keys [h w]}]
   (-> filename
       (cv/imread)
       (cv/resize! (cv/new-size h w))
       (cvu/imshow))))

#_ (preview! "images/cat.jpg")
#_ (preview! "images/cat.jpg" {:h 300 :w 200})

(defn preprocess-mat
  "Preprocessing steps on a `mat` from OpenCV.
   Example of commons preprocessing tasks"
  [mat]
  (-> mat
      ;; Subtract mean
      (cv/add! (cv/new-scalar 103.939 116.779 123.68))
      ;; Resize
      (cv/resize! (cv/new-size 400 400))
      ;; Maps pixel values from [-128, 127] to [0, 127]
      (cv/convert-to! cv/CV_8SC3 0.5)
      ;; Normalize pixel values into (0.0, 1.0)
      ;;(cv/normalize! 0.0 1.0 cv/NORM_MINMAX cv/CV_32FC1)
      ))

#_(-> "images/cat.jpg"
      (cv/imread)
      (preprocess-mat)
      (cvu/imshow))


#_(-> "images/dog.jpg"
    ;; Read image from disk
    (mx-img/read-image {:to-rgb false})
    ;; Resizing image
    (mx-img/resize-image 200 200)
    ;; Convert NDArray to Mat
    (mx-cv/ndarray-to-mat)
    ;; Mat to bytes
    (cv/<<)
    (cv/->bytes)
    (vec))

(defn mat->color-mat
  "Returns Mat of the selected channel.
   Assumes the Mat is in RGB format.

  `mat`: Mat object from OpenCV
  `color`: value in #{:red :green :blue}

  Ex
    (mat->color-mat m :green)
    (mat->color-mat m :blue)"
  [mat color]
  (let [color->selector #(get {:red first :green second :blue last} % first)]
    ((color->selector color) (cv/split! mat))))

#_ (-> "images/cat.jpg"
    ;; Read image from disk
    (mx-img/read-image {:to-rgb true})
    ;; Resize
    (mx-img/resize-image 200 200)
    ;; Convert NDArray to Mat
    (mx-cv/ndarray-to-mat)
    ;; Extract the different color channels
    ((juxt #(mat->color-mat % :blue)
           #(mat->color-mat % :green)
           #(mat->color-mat % :red)))
    ;; Display the images
    (#(map cvu/imshow %)))


;; Showing an image using `buffered-image-to-mat`
#_ (-> "images/dog.jpg"
    ;; Read image from disk
    (mx-img/read-image {:to-rgb true})
    ;; Convert to BufferedImage - Can be very slow...
    (mx-img/to-image)
    ;; Convert to Mat
    (cvu/buffered-image-to-mat)
    ;; Show Mat
    (cvu/imshow))



(defn mat->ndarray
  "Convert a `mat` from OpenCV to an MXNet `ndarray`"
  [mat]
  (let [h (.height mat)
        w (.width mat)
        c (.channels mat)]
    (-> mat
        cvu/mat->flat-rgb-array
        (ndarray/array [c h w]))))

(defn ndarray->mat
  "Convert a `ndarray` to an OpenCV `mat`"
  [ndarray]
  (let [shape (mx-shape/->vec ndarray)
        [h w _ _] (mx-shape/->vec (ndarray/shape ndarray))
        bytes (byte-array shape)
        mat (cv/new-mat h w cv/CV_8UC3)]
    (.put mat 0 0 bytes)
    mat))

(defn filename->ndarray!
  "Convert an image stored on disk `filename` into an `ndarray`

  `filename`: string representing the image on disk
  `shape-vec`: is the actual shape of the returned `ndarray`
   return: ndarray"
  [filename shape-vec]
  (-> filename
      (cv/imread)
      (mat->ndarray)))

(defn draw-bounding-box!
  "Draw bounding box on `img` given the `top-left` and `bottom-right` coordonates.
  Add `label` when provided.
  returns: nil"
  [img {:keys [label top-left bottom-right]}]
  (let [[x0 y0] top-left
        [x1 y1] bottom-right
        top-left-point (cv/new-point x0 y0)
        bottom-right-point (cv/new-point x1 y1)]
    (cv/rectangle img top-left-point bottom-right-point rgb/white 1)
    (when label
      (cv/put-text! img label top-left-point cv/FONT_HERSHEY_DUPLEX 1.0 rgb/white 1))))

(defn draw-predictions!
  "Draw all predictions on an `img` passing `results` which is a collection
     of bounding boxes data.
     returns: nil"
  [img results]
  (doseq [result results]
    (draw-bounding-box! img result)))

#_(let [img (cv/imread "images/dog.jpg")
      results [{:top-left [200 70] :bottom-right [830 430] :label "dog"}
               {:top-left [200 440] :bottom-right [350 525] :label "cookie"}]]
  (draw-predictions! img results)
  (cvu/imshow img))