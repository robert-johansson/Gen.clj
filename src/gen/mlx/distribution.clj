(ns gen.mlx.distribution
  "MLX-backed distributions for Gen.clj.

   Drop-in replacements for gen.distribution.kixi that compute logpdf
   via MLX operations. This enables automatic differentiation of logpdf
   w.r.t. distribution parameters — the key building block for gradient-based
   inference (HMC, MALA, MAP estimation).

   Usage mirrors kixi exactly:
     ;; Before: (require '[gen.distribution.kixi :as dist])
     ;; After:  (require '[gen.mlx.distribution :as dist])
     ;; Model code is unchanged."
  (:require [gen.distribution :as d]
            [gen.mlx.array :as arr]
            [gen.mlx.transforms :as xforms])
  (:import [java.util.concurrent ThreadLocalRandom]))

;; ---------------------------------------------------------------------------
;; Constants (as MLXArrays for use in computation graphs)
;; ---------------------------------------------------------------------------

(def ^:private log-2pi
  "log(2π) as an MLXArray scalar."
  (arr/scalar (Math/log (* 2.0 Math/PI))))

;; ---------------------------------------------------------------------------
;; Differentiable logpdf functions — return MLXArray
;;
;; These preserve the MLX computation graph, so they can be differentiated
;; via gen.mlx.transforms/grad. Use these directly when you need gradients.
;; The distribution records below call these and extract a JVM double
;; for Gen protocol compatibility.
;; ---------------------------------------------------------------------------

(defn gaussian-logpdf
  "Gaussian log-density using MLX ops. All args are MLXArrays or auto-coerced
   numbers. Returns MLXArray (differentiable).

   Formula: -0.5 * (log(2π) + 2·log(σ) + ((v-μ)/σ)²)"
  [mu sigma v]
  (let [z (arr/div (arr/sub v mu) sigma)]
    (arr/mul -0.5 (arr/add log-2pi
                           (arr/mul 2.0 (arr/log sigma))
                           (arr/square z)))))

;; ---------------------------------------------------------------------------
;; Compiled logpdf — traces the graph once, reuses on subsequent calls
;; ---------------------------------------------------------------------------

(def compiled-gaussian-logpdf
  "Compiled version of gaussian-logpdf. Traces the computation graph on
   first call, then reuses the optimized native kernel on subsequent calls.
   All args must be MLXArrays."
  (xforms/compile gaussian-logpdf))

;; ---------------------------------------------------------------------------
;; Distribution types — implement Gen's LogPDF + Sample protocols
;;
;; logpdf returns a JVM double (Gen compatibility).
;; sample uses JVM ThreadLocalRandom (sampling doesn't need MLX).
;; ---------------------------------------------------------------------------

(defrecord MLXNormal [^double mu ^double sigma mu-arr sigma-arr]
  d/LogPDF
  (logpdf [_ v]
    (arr/->double (compiled-gaussian-logpdf mu-arr sigma-arr (arr/scalar (double v)))))

  d/Sample
  (sample [_]
    (+ mu (* sigma (.nextGaussian (ThreadLocalRandom/current))))))

;; ---------------------------------------------------------------------------
;; Constructors + GenerativeFn wrappers
;; Same pattern as gen.distribution.kixi — constructor fn + GenerativeFn record.
;; ---------------------------------------------------------------------------

(defn normal-distribution
  "Create an MLX-backed Normal distribution."
  ([] (normal-distribution 0.0 1.0))
  ([mu sigma]
   (let [mu (double mu) sigma (double sigma)]
     (->MLXNormal mu sigma (arr/scalar mu) (arr/scalar sigma)))))

(def normal
  "Normal distribution as a Gen generative function.
   Usage: (gf/simulate normal [0.0 1.0])"
  (d/->GenerativeFn normal-distribution 2))
