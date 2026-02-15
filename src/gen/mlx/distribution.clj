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

;; ---------------------------------------------------------------------------
;; IMLXLogPDF protocol — enables vectorized multi-distribution score functions
;; ---------------------------------------------------------------------------

(defprotocol IMLXLogPDF
  (mlx-logpdf [dist v-arr]
    "Logpdf via MLX ops, returns MLXArray. `v-arr` may be scalar or vector.")
  (dist-tag [dist]
    "Keyword tag for vectorized grouping (e.g. :normal, :exponential)."))

(extend-type MLXNormal
  IMLXLogPDF
  (mlx-logpdf [this v-arr]
    (compiled-gaussian-logpdf (:mu-arr this) (:sigma-arr this) v-arr))
  (dist-tag [_] :normal))

;; ---------------------------------------------------------------------------
;; Exponential distribution
;;   logpdf(rate, v) = log(rate) - rate * v
;; ---------------------------------------------------------------------------

(def ^:private log-pi
  "log(π) as an MLXArray scalar."
  (arr/scalar (Math/log Math/PI)))

(defn exponential-logpdf
  "Exponential log-density using MLX ops. Returns MLXArray (differentiable).
   Formula: log(rate) - rate * v"
  [rate v]
  (arr/sub (arr/log rate) (arr/mul rate v)))

(def compiled-exponential-logpdf
  (xforms/compile exponential-logpdf))

(defrecord MLXExponential [^double rate rate-arr]
  d/LogPDF
  (logpdf [_ v]
    (arr/->double (compiled-exponential-logpdf rate-arr (arr/scalar (double v)))))

  d/Sample
  (sample [_]
    (/ (- (Math/log (- 1.0 (.nextDouble (ThreadLocalRandom/current)))))
       rate))

  IMLXLogPDF
  (mlx-logpdf [_ v-arr]
    (compiled-exponential-logpdf rate-arr v-arr))
  (dist-tag [_] :exponential))

(defn exponential-distribution
  "Create an MLX-backed Exponential distribution."
  [rate]
  (let [rate (double rate)]
    (->MLXExponential rate (arr/scalar rate))))

(def exponential
  "Exponential distribution as a Gen generative function."
  (d/->GenerativeFn exponential-distribution 1))

;; ---------------------------------------------------------------------------
;; Uniform distribution
;;   logpdf(lo, hi, v) = -log(hi - lo)   (for lo <= v <= hi)
;;
;;   Note: We compute the unconstrained logpdf (no boundary check in MLX
;;   graph since MLX autodiff needs smooth functions). Boundary checks
;;   happen in the JVM logpdf path.
;; ---------------------------------------------------------------------------

(defn uniform-logpdf
  "Uniform log-density using MLX ops. Returns MLXArray (differentiable).
   Formula: -log(hi - lo)"
  [lo hi _v]
  (arr/neg (arr/log (arr/sub hi lo))))

(def compiled-uniform-logpdf
  (xforms/compile uniform-logpdf))

(defrecord MLXUniform [^double lo ^double hi lo-arr hi-arr]
  d/LogPDF
  (logpdf [_ v]
    (let [v (double v)]
      (if (and (<= lo v) (<= v hi))
        (arr/->double (compiled-uniform-logpdf lo-arr hi-arr (arr/scalar v)))
        ##-Inf)))

  d/Sample
  (sample [_]
    (+ lo (* (- hi lo) (.nextDouble (ThreadLocalRandom/current)))))

  IMLXLogPDF
  (mlx-logpdf [_ v-arr]
    (compiled-uniform-logpdf lo-arr hi-arr v-arr))
  (dist-tag [_] :uniform))

(defn uniform-distribution
  "Create an MLX-backed Uniform distribution."
  ([] (uniform-distribution 0.0 1.0))
  ([lo hi]
   (let [lo (double lo) hi (double hi)]
     (->MLXUniform lo hi (arr/scalar lo) (arr/scalar hi)))))

(def uniform
  "Uniform distribution as a Gen generative function."
  (d/->GenerativeFn uniform-distribution 2))

;; ---------------------------------------------------------------------------
;; Laplace distribution
;;   logpdf(loc, scale, v) = -log(2*scale) - |v - loc| / scale
;; ---------------------------------------------------------------------------

(defn laplace-logpdf
  "Laplace log-density using MLX ops. Returns MLXArray (differentiable).
   Formula: -log(2*scale) - |v - loc| / scale"
  [loc scale v]
  (arr/sub (arr/neg (arr/log (arr/mul 2.0 scale)))
           (arr/div (arr/abs (arr/sub v loc)) scale)))

(def compiled-laplace-logpdf
  (xforms/compile laplace-logpdf))

(defrecord MLXLaplace [^double location ^double scale loc-arr scale-arr]
  d/LogPDF
  (logpdf [_ v]
    (arr/->double (compiled-laplace-logpdf loc-arr scale-arr (arr/scalar (double v)))))

  d/Sample
  (sample [_]
    (let [u (- (.nextDouble (ThreadLocalRandom/current)) 0.5)]
      (- location (* scale (Math/signum u) (Math/log (- 1.0 (* 2.0 (Math/abs u))))))))

  IMLXLogPDF
  (mlx-logpdf [_ v-arr]
    (compiled-laplace-logpdf loc-arr scale-arr v-arr))
  (dist-tag [_] :laplace))

(defn laplace-distribution
  "Create an MLX-backed Laplace distribution."
  ([location scale]
   (let [location (double location) scale (double scale)]
     (->MLXLaplace location scale (arr/scalar location) (arr/scalar scale)))))

(def laplace
  "Laplace distribution as a Gen generative function."
  (d/->GenerativeFn laplace-distribution 2))

;; ---------------------------------------------------------------------------
;; Cauchy distribution
;;   logpdf(loc, scale, v) = -log(π) - log(scale) - log(1 + ((v-loc)/scale)²)
;; ---------------------------------------------------------------------------

(defn cauchy-logpdf
  "Cauchy log-density using MLX ops. Returns MLXArray (differentiable).
   Formula: -log(π) - log(scale) - log(1 + ((v-loc)/scale)²)"
  [loc scale v]
  (let [z (arr/div (arr/sub v loc) scale)]
    (arr/sub (arr/sub (arr/neg log-pi)
                      (arr/log scale))
             (arr/log (arr/add 1.0 (arr/square z))))))

(def compiled-cauchy-logpdf
  (xforms/compile cauchy-logpdf))

(defrecord MLXCauchy [^double location ^double scale loc-arr scale-arr]
  d/LogPDF
  (logpdf [_ v]
    (arr/->double (compiled-cauchy-logpdf loc-arr scale-arr (arr/scalar (double v)))))

  d/Sample
  (sample [_]
    (+ location (* scale (Math/tan (* Math/PI (- (.nextDouble (ThreadLocalRandom/current)) 0.5))))))

  IMLXLogPDF
  (mlx-logpdf [_ v-arr]
    (compiled-cauchy-logpdf loc-arr scale-arr v-arr))
  (dist-tag [_] :cauchy))

(defn cauchy-distribution
  "Create an MLX-backed Cauchy distribution."
  [location scale]
  (let [location (double location) scale (double scale)]
    (->MLXCauchy location scale (arr/scalar location) (arr/scalar scale))))

(def cauchy
  "Cauchy distribution as a Gen generative function."
  (d/->GenerativeFn cauchy-distribution 2))
