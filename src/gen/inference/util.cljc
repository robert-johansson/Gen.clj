(ns gen.inference.util
  "Shared utilities for inference algorithms."
  (:require [clojure.math :as math]))

(defn logsumexp
  "log(exp(x1)+...+exp(xn)). Numerically stable."
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

(defn neg-inf?
  "Returns true if v is negative infinity."
  [v]
  (= v ##-Inf))
