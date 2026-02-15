(ns gen.inference.importance
  (:require [clojure.math :as math]
            [gen.choicemap :as choicemap]
            [gen.generative-function :as gf]
            [gen.inference.util :as util]))

(defn resampling [gf args observations n-samples]
  ;; https://github.com/probcomp/Gen.jl/blob/master/src/inference/importance.jl#L77...L95
  (let [first-result (gf/generate gf args observations)]
    (loop [i                1
           model-trace      (:trace first-result)
           log-total-weight (:weight first-result)]
      (if (< i n-samples)
        (let [candidate (gf/generate gf args observations)
              log-weight (:weight candidate)]
          (if (util/neg-inf? log-weight)
            (recur (inc i) model-trace log-total-weight)
            (let [new-log-total (util/logsumexp [log-weight log-total-weight])
                  accept-prob   (math/exp (- log-weight new-log-total))]
              (if (< (rand) accept-prob)
                (recur (inc i) (:trace candidate) new-log-total)
                (recur (inc i) model-trace new-log-total)))))
        (let [log-ml-estimate (- log-total-weight (math/log n-samples))]
          {:trace  model-trace
           :weight log-ml-estimate})))))

(defn custom-proposal-resampling
  "Importance resampling with a custom proposal distribution.

   Samples from `proposal` instead of the model prior:
   - proposal: generative function used for proposing choices
   - proposal-args: arguments to the proposal
   - weight = model-score - proposal-score

   Returns `{:trace :weight}` where :weight is the log-marginal-likelihood estimate."
  [model model-args observations proposal proposal-args n-samples]
  (letfn [(draw-sample []
            (let [{:keys [choices weight]}
                  (gf/propose proposal proposal-args)
                  merged (choicemap/merge observations choices)
                  {model-trace :trace
                   model-weight :weight}
                  (gf/generate model model-args merged)
                  importance-weight (- model-weight weight)]
              {:trace model-trace
               :weight importance-weight}))]
    (let [first-result (draw-sample)]
      (loop [i                1
             model-trace      (:trace first-result)
             log-total-weight (:weight first-result)]
        (if (< i n-samples)
          (let [candidate (draw-sample)
                log-weight (:weight candidate)]
            (if (util/neg-inf? log-weight)
              (recur (inc i) model-trace log-total-weight)
              (let [new-log-total (util/logsumexp [log-weight log-total-weight])
                    accept-prob   (math/exp (- log-weight new-log-total))]
                (if (< (rand) accept-prob)
                  (recur (inc i) (:trace candidate) new-log-total)
                  (recur (inc i) model-trace new-log-total)))))
          (let [log-ml-estimate (- log-total-weight (math/log n-samples))]
            {:trace  model-trace
             :weight log-ml-estimate}))))))
