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
            [gen.distribution.math.gamma :as gamma]
            [gen.mlx.array :as arr]
            [gen.mlx.transforms :as xforms])
  (:import [java.util.concurrent ThreadLocalRandom]
           [org.apache.commons.math3.distribution
            BetaDistribution GammaDistribution PoissonDistribution]))

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

;; ---------------------------------------------------------------------------
;; Beta distribution
;;   logpdf(α, β, v) = (α-1)*log(v) + (β-1)*log(1-v) + log-norm
;;   where log-norm = lgamma(α+β) - lgamma(α) - lgamma(β) (JVM constant)
;; ---------------------------------------------------------------------------

(defn beta-logpdf
  "Beta log-density using MLX ops. Returns MLXArray (differentiable w.r.t. v).
   `alpha-m1`, `beta-m1`, `log-norm` are pre-computed MLXArray constants.
   Formula: (α-1)*log(v) + (β-1)*log(1-v) + log-norm"
  [alpha-m1 beta-m1 log-norm v]
  (arr/add (arr/mul alpha-m1 (arr/log v))
           (arr/mul beta-m1 (arr/log (arr/sub 1.0 v)))
           log-norm))

(def compiled-beta-logpdf
  (xforms/compile beta-logpdf))

(defrecord MLXBeta [^double alpha ^double beta
                    alpha-m1-arr beta-m1-arr log-norm-arr]
  d/LogPDF
  (logpdf [_ v]
    (let [v (double v)]
      (if (< 0.0 v 1.0)
        (arr/->double (compiled-beta-logpdf alpha-m1-arr beta-m1-arr
                                            log-norm-arr (arr/scalar v)))
        ##-Inf)))

  d/Sample
  (sample [_]
    (.sample (BetaDistribution. alpha beta)))

  IMLXLogPDF
  (mlx-logpdf [_ v-arr]
    (compiled-beta-logpdf alpha-m1-arr beta-m1-arr log-norm-arr v-arr))
  (dist-tag [_] :beta))

(defn beta-distribution
  "Create an MLX-backed Beta distribution."
  [alpha beta]
  (let [alpha (double alpha)
        beta  (double beta)
        log-norm (- (gamma/log-gamma (+ alpha beta))
                    (gamma/log-gamma alpha)
                    (gamma/log-gamma beta))]
    (->MLXBeta alpha beta
               (arr/scalar (dec alpha))
               (arr/scalar (dec beta))
               (arr/scalar log-norm))))

(def beta-dist
  "Beta distribution as a Gen generative function."
  (d/->GenerativeFn beta-distribution 2))

;; ---------------------------------------------------------------------------
;; Gamma distribution
;;   logpdf(shape, scale, v) = (shape-1)*log(v) - v/scale + log-norm
;;   where log-norm = -lgamma(shape) - shape*log(scale) (JVM constant)
;; ---------------------------------------------------------------------------

(defn gamma-logpdf
  "Gamma log-density using MLX ops. Returns MLXArray (differentiable w.r.t. v).
   `shape-m1`, `inv-scale`, `log-norm` are pre-computed MLXArray constants.
   Formula: (shape-1)*log(v) - v/scale + log-norm"
  [shape-m1 inv-scale log-norm v]
  (arr/add (arr/mul shape-m1 (arr/log v))
           (arr/neg (arr/mul v inv-scale))
           log-norm))

(def compiled-gamma-logpdf
  (xforms/compile gamma-logpdf))

(defrecord MLXGamma [^double shape ^double scale
                     shape-m1-arr inv-scale-arr log-norm-arr]
  d/LogPDF
  (logpdf [_ v]
    (let [v (double v)]
      (if (pos? v)
        (arr/->double (compiled-gamma-logpdf shape-m1-arr inv-scale-arr
                                             log-norm-arr (arr/scalar v)))
        ##-Inf)))

  d/Sample
  (sample [_]
    (.sample (GammaDistribution. shape scale)))

  IMLXLogPDF
  (mlx-logpdf [_ v-arr]
    (compiled-gamma-logpdf shape-m1-arr inv-scale-arr log-norm-arr v-arr))
  (dist-tag [_] :gamma))

(defn gamma-distribution
  "Create an MLX-backed Gamma distribution."
  [shape scale]
  (let [shape (double shape)
        scale (double scale)
        log-norm (- (- (gamma/log-gamma shape))
                    (* shape (Math/log scale)))]
    (->MLXGamma shape scale
                (arr/scalar (dec shape))
                (arr/scalar (/ 1.0 scale))
                (arr/scalar log-norm))))

(def gamma-dist
  "Gamma distribution as a Gen generative function."
  (d/->GenerativeFn gamma-distribution 2))

;; ---------------------------------------------------------------------------
;; Poisson distribution
;;   logpdf(λ, k) = k*log(λ) - λ - lgamma(k+1)
;;   Discrete — not differentiable w.r.t. k. Included for count modeling.
;; ---------------------------------------------------------------------------

(defrecord MLXPoisson [^double lambda log-lambda-arr lambda-arr]
  d/LogPDF
  (logpdf [_ v]
    (let [k (long v)]
      (if (>= k 0)
        (- (* k (Math/log lambda))
           lambda
           (gamma/log-gamma (inc (double k))))
        ##-Inf)))

  d/Sample
  (sample [_]
    (.sample (PoissonDistribution. (int (Math/round lambda)))))

  IMLXLogPDF
  (mlx-logpdf [_ v-arr]
    ;; Compute k*log(λ) - λ in MLX, then subtract lgamma(k+1) per-element JVM-side
    ;; This is a hybrid: MLX for the batched part, JVM for lgamma
    (let [mlx-part (arr/sub (arr/mul v-arr log-lambda-arr) lambda-arr)
          ;; Extract values, compute lgamma(k+1) JVM-side
          ks (arr/->vec v-arr)
          lgamma-terms (arr/from-vec (mapv #(gamma/log-gamma (inc (double %))) ks))]
      (arr/sub mlx-part lgamma-terms)))
  (dist-tag [_] :poisson))

(defn poisson-distribution
  "Create an MLX-backed Poisson distribution."
  [lambda]
  (let [lambda (double lambda)]
    (->MLXPoisson lambda
                  (arr/scalar (Math/log lambda))
                  (arr/scalar lambda))))

(def poisson
  "Poisson distribution as a Gen generative function."
  (d/->GenerativeFn poisson-distribution 1))

;; ---------------------------------------------------------------------------
;; Multivariate Normal distribution
;;   logpdf(μ, Σ, x) = -0.5 * [k*log(2π) + 2*sum(log(diag(L))) + ||L⁻¹(x-μ)||²]
;;   where L = cholesky(Σ), pre-computed at construction time.
;;
;;   Does NOT implement IMLXLogPDF — vector-valued, uses :fallback path.
;; ---------------------------------------------------------------------------

(defn mvnormal-logpdf
  "Multivariate Normal log-density using MLX ops. Returns MLXArray.
   L is the lower Cholesky factor, log-det-term is pre-computed.
   Differentiable w.r.t. v (the value vector)."
  [L log-det-term mu v]
  (let [diff  (arr/sub v mu)
        ;; Solve L z = diff for z (L is lower-triangular)
        ;; diff needs to be a column vector for solve_triangular
        k     (long (first (:shape L)))
        diff-col (arr/reshape diff [k 1])
        z-col    (arr/solve-triangular L diff-col)
        z        (arr/reshape z-col [k])
        ;; ||z||² = sum(z²)
        mahal    (arr/sum (arr/square z))]
    (arr/sub log-det-term (arr/mul 0.5 mahal))))

(defrecord MLXMvNormal [^long k mu-vec cov-mat mu-arr L-arr log-det-term-arr]
  d/LogPDF
  (logpdf [_ v]
    (let [v-vec (if (vector? v) v (vec v))
          v-arr (arr/from-vec v-vec)]
      (arr/->double (mvnormal-logpdf L-arr log-det-term-arr mu-arr v-arr))))

  d/Sample
  (sample [_]
    ;; Sample z ~ N(0,I), then x = L @ z + μ
    (let [rng (ThreadLocalRandom/current)
          z   (float-array (repeatedly k #(.nextGaussian rng)))
          z-arr (arr/array (vec z) [k 1])
          ;; x = L @ z + μ
          x-col (arr/add (arr/matmul L-arr z-arr) (arr/reshape mu-arr [k 1]))
          x     (arr/reshape x-col [k])]
      (arr/->vec x))))

(defn mvnormal-distribution
  "Create an MLX-backed Multivariate Normal distribution.
   `mu` is a vector of means, `cov` is a vector-of-vectors covariance matrix."
  [mu cov]
  (let [mu-vec (vec (map double mu))
        k      (count mu-vec)
        cov-mat (vec (map #(vec (map double %)) cov))
        mu-arr (arr/from-vec mu-vec)
        cov-arr (arr/from-2d cov-mat)
        L-arr  (arr/cholesky cov-arr)
        ;; log-det-term = -0.5 * (k*log(2π) + 2*sum(log(diag(L))))
        ;; Compute on JVM side for the constant part
        _ (arr/eval! L-arr)
        L-diag (arr/diagonal L-arr)
        log-diag-sum (arr/sum (arr/log L-diag))
        _ (arr/eval! log-diag-sum)
        log-det-2 (* 2.0 (arr/->double log-diag-sum))
        log-det-term (* -0.5 (+ (* k (Math/log (* 2.0 Math/PI)))
                                log-det-2))]
    (->MLXMvNormal k mu-vec cov-mat mu-arr L-arr (arr/scalar log-det-term))))

;; ---------------------------------------------------------------------------
;; Dirichlet distribution
;;   logpdf(α, x) = sum_i[(α_i-1)*log(x_i)] + log-norm
;;   where log-norm = lgamma(sum(α)) - sum(lgamma(α_i)) (JVM constant)
;;
;;   Does NOT implement IMLXLogPDF — vector-valued, uses :fallback path.
;; ---------------------------------------------------------------------------

(defn dirichlet-logpdf
  "Dirichlet log-density using MLX ops. Returns MLXArray.
   alpha-m1 is (α-1) as an MLX vector, log-norm is pre-computed scalar."
  [alpha-m1 log-norm v]
  (arr/add (arr/sum (arr/mul alpha-m1 (arr/log v)))
           log-norm))

(defrecord MLXDirichlet [alpha-vec ^long k alpha-m1-arr log-norm-arr]
  d/LogPDF
  (logpdf [_ v]
    (let [v-vec (if (vector? v) v (vec v))
          v-sum (reduce + v-vec)]
      (if (and (every? pos? v-vec)
               (< (Math/abs (- v-sum 1.0)) 1e-6))
        (arr/->double (dirichlet-logpdf alpha-m1-arr log-norm-arr
                                        (arr/from-vec v-vec)))
        ##-Inf)))

  d/Sample
  (sample [_]
    ;; Sample k independent Gamma(α_i, 1) and normalize
    (let [raw (mapv (fn [a] (.sample (GammaDistribution. a 1.0))) alpha-vec)
          total (reduce + raw)]
      (mapv #(/ % total) raw))))

(defn dirichlet-distribution
  "Create an MLX-backed Dirichlet distribution.
   `alpha` is a vector of concentration parameters."
  [alpha]
  (let [alpha-vec (vec (map double alpha))
        k         (count alpha-vec)
        log-norm  (- (gamma/log-gamma (reduce + alpha-vec))
                     (reduce + (map gamma/log-gamma alpha-vec)))]
    (->MLXDirichlet alpha-vec k
                    (arr/from-vec (mapv #(dec (double %)) alpha-vec))
                    (arr/scalar log-norm))))
