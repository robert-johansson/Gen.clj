(ns gen.inference.importance
  (:require [clojure.math :as math]
            [gen.generative-function :as gf]))

;; This implementation comes from `fastmath.core`, ported here for cljc
;; purposes.

(defn- logsumexp
  "log(exp(x1)+...+exp(xn))."
  ^double [xs]
  (loop [[^double x & rst] xs
         r 0.0
         alpha ##-Inf]
    (if (<= x alpha)
      (let [nr (+ r (math/exp (- x alpha)))]
        (if-not (seq rst)
          (+ (math/log nr) alpha)
          (recur rst nr alpha)))
      (let [nr (inc (* r (math/exp (- alpha x))))]
        (if-not (seq rst)
          (+ (math/log nr) x)
          (recur rst nr (double x)))))))

(defn- neg-inf?
  [v]
  (= v ##-Inf))

(defn resampling [gf args observations n-samples]
  ;; https://github.com/probcomp/Gen.jl/blob/master/src/inference/importance.jl#L77...L95
  (let [first-result (gf/generate gf args observations)]
    (loop [i                1
           model-trace      (:trace first-result)
           log-total-weight (:weight first-result)]
      (if (< i n-samples)
        (let [candidate (gf/generate gf args observations)
              log-weight (:weight candidate)]
          (if (neg-inf? log-weight)
            (recur (inc i) model-trace log-total-weight)
            (let [new-log-total (logsumexp [log-weight log-total-weight])
                  accept-prob   (math/exp (- log-weight new-log-total))]
              (if (< (rand) accept-prob)
                (recur (inc i) (:trace candidate) new-log-total)
                (recur (inc i) model-trace new-log-total)))))
        (let [log-ml-estimate (- log-total-weight (math/log n-samples))]
          {:trace  model-trace
           :weight log-ml-estimate})))))
