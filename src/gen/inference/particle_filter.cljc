(ns gen.inference.particle-filter
  "Sequential Monte Carlo (particle filter) for online Bayesian inference.

   Core workflow:
   1. `initialize` — create initial particle population
   2. `step` — incorporate new observations via trace update
   3. `maybe-resample` — resample if ESS drops below threshold
   4. Query: `get-traces`, `get-log-weights`, `log-ml-estimate`"
  (:require [clojure.math :as math]
            [gen.generative-function :as gf]
            [gen.inference.util :as util]
            [gen.trace :as trace]))

;; ## State

(defrecord ParticleFilterState [traces log-weights log-ml-estimate n-particles])

;; ## Core Functions

(defn initialize
  "Create initial particle filter state by generating n-particles traces
   from `model` with `observations` as constraints.

   Returns a ParticleFilterState."
  [model args observations n-particles]
  (let [results (mapv (fn [_] (gf/generate model args observations))
                      (range n-particles))
        traces     (mapv :trace results)
        log-weights (mapv :weight results)
        log-total   (util/logsumexp log-weights)
        log-ml      (- log-total (math/log n-particles))]
    (->ParticleFilterState traces log-weights log-ml n-particles)))

(defn step
  "Advance the particle filter by one step.

   Updates each particle's trace with `new-args` and `new-observations`.
   `argdiffs` describes which arguments changed.

   Returns updated ParticleFilterState with accumulated log-ML estimate."
  [state new-args argdiffs new-observations]
  (let [{:keys [traces log-weights log-ml-estimate n-particles]} state
        results (mapv (fn [tr]
                        (trace/update tr new-args argdiffs new-observations))
                      traces)
        new-traces     (mapv :trace results)
        incremental-ws (mapv :weight results)
        new-log-weights (mapv + log-weights incremental-ws)
        log-total       (util/logsumexp new-log-weights)
        log-ml-increment (- log-total (math/log n-particles))
        new-log-ml      (+ log-ml-estimate log-ml-increment)]
    (->ParticleFilterState new-traces new-log-weights new-log-ml n-particles)))

(defn effective-sample-size
  "Compute the effective sample size (ESS) of the particle filter state.
   ESS = 1 / sum(w_i^2) where w_i are normalized weights.
   Returns a value in [1, n-particles]."
  ^double [state]
  (let [{:keys [log-weights]} state
        log-total (util/logsumexp log-weights)
        log-normalized (mapv #(- % log-total) log-weights)
        log-squared (mapv #(* 2.0 %) log-normalized)
        log-sum-squared (util/logsumexp log-squared)]
    (math/exp (- 0.0 log-sum-squared))))

(defn- multinomial-resample
  "Multinomial resampling: draw n-particles indices proportional to weights."
  [log-weights n-particles]
  (let [log-total  (util/logsumexp log-weights)
        ;; Compute cumulative weights in probability space
        probs      (mapv #(math/exp (- % log-total)) log-weights)
        cum-probs  (reductions + probs)]
    (mapv (fn [_]
            (let [u (rand)]
              (reduce (fn [idx cp]
                        (if (< u cp)
                          (reduced idx)
                          (inc idx)))
                      0
                      cum-probs)))
          (range n-particles))))

(defn resample
  "Resample particles: draw indices proportional to weights, clone traces,
   reset weights to uniform (log(1/n)).

   Returns updated ParticleFilterState."
  [state]
  (let [{:keys [traces log-weights log-ml-estimate n-particles]} state
        indices     (multinomial-resample log-weights n-particles)
        new-traces  (mapv #(nth traces %) indices)
        new-lws     (vec (repeat n-particles 0.0))]
    (->ParticleFilterState new-traces new-lws log-ml-estimate n-particles)))

(defn maybe-resample
  "Resample if ESS falls below `ess-threshold`.
   Typical threshold: n-particles / 2.

   Returns (possibly resampled) ParticleFilterState."
  [state ess-threshold]
  (if (< (effective-sample-size state) ess-threshold)
    (resample state)
    state))

;; ## Query Functions

(defn get-traces
  "Get the current particle traces."
  [state]
  (:traces state))

(defn get-log-weights
  "Get the current log importance weights."
  [state]
  (:log-weights state))

(defn log-ml-estimate
  "Get the log marginal likelihood estimate accumulated so far."
  [state]
  (:log-ml-estimate state))

(defn get-weighted-traces
  "Returns a sequence of [trace log-weight] pairs."
  [state]
  (map vector (:traces state) (:log-weights state)))
