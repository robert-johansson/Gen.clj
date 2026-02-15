(ns gen.mlx.distribution-new-test
  "Tests for new MLX distributions: Beta, Gamma, Poisson, MVN, Dirichlet."
  (:require [clojure.test :refer [deftest is testing]]
            [gen.distribution :as d]
            [gen.distribution.kixi :as kixi]
            [gen.distribution.math.gamma :as gamma]
            [gen.distribution.math.log-likelihood :as ll]
            [gen.mlx.array :as arr]
            [gen.mlx.distribution :as mlx-dist]
            [gen.mlx.dynamic :as mlx-dyn]
            [gen.mlx.transforms :as xforms]
            [gen.dynamic :as dynamic]
            [gen.generative-function :as gf]
            [gen.trace :as trace]))

(defn- close?
  "Check if two doubles are within tolerance."
  ([expected actual] (close? expected actual 1e-4))
  ([expected actual tol]
   (< (Math/abs (- (double expected) (double actual))) tol)))

;; ---------------------------------------------------------------------------
;; Beta distribution tests
;; ---------------------------------------------------------------------------

(deftest beta-logpdf-vs-kixi
  (testing "MLX Beta matches kixi for several (α,β,v) combos"
    (doseq [[alpha beta v] [[2.0 5.0 0.3]
                             [0.5 0.5 0.5]
                             [1.0 1.0 0.5]
                             [2.0 2.0 0.7]
                             [5.0 1.0 0.9]
                             [1.0 3.0 0.1]]]
      (let [kixi-val (d/logpdf (kixi/beta-distribution alpha beta) v)
            mlx-val  (d/logpdf (mlx-dist/beta-distribution alpha beta) v)]
        (is (close? kixi-val mlx-val 1e-3)
            (str "Beta(" alpha "," beta ") at v=" v
                 " kixi=" kixi-val " mlx=" mlx-val))))))

(deftest beta-logpdf-vs-ll
  (testing "MLX Beta matches log-likelihood module"
    (doseq [[alpha beta v] [[2.0 5.0 0.3]
                             [3.0 3.0 0.5]
                             [0.5 0.5 0.8]]]
      (let [ll-val  (ll/beta alpha beta v)
            mlx-val (d/logpdf (mlx-dist/beta-distribution alpha beta) v)]
        (is (close? ll-val mlx-val 1e-3)
            (str "Beta(" alpha "," beta ") at v=" v))))))

(deftest beta-out-of-bounds
  (testing "Returns ##-Inf for v <= 0 or v >= 1"
    (let [d (mlx-dist/beta-distribution 2.0 5.0)]
      (is (= ##-Inf (d/logpdf d 0.0)))
      (is (= ##-Inf (d/logpdf d 1.0)))
      (is (= ##-Inf (d/logpdf d -0.1)))
      (is (= ##-Inf (d/logpdf d 1.5))))))

(deftest beta-gradient
  (testing "d/dv matches analytical: (α-1)/v - (β-1)/(1-v)"
    (doseq [[alpha beta v] [[2.0 5.0 0.3]
                             [3.0 2.0 0.5]
                             [1.5 1.5 0.7]]]
      (let [expected (- (/ (dec alpha) v)
                        (/ (dec beta) (- 1.0 v)))
            d        (mlx-dist/beta-distribution alpha beta)
            grad-fn  (xforms/grad
                      (fn [v-arr]
                        (mlx-dist/beta-logpdf
                         (:alpha-m1-arr d) (:beta-m1-arr d)
                         (:log-norm-arr d) v-arr)))
            actual   (arr/->double (grad-fn (arr/scalar v)))]
        (is (close? expected actual 1e-2)
            (str "d/dv Beta(" alpha "," beta ") at v=" v))))))

(deftest beta-sampling
  (testing "Beta sampling produces values in (0,1)"
    (let [d (mlx-dist/beta-distribution 2.0 5.0)]
      (dotimes [_ 20]
        (let [v (d/sample d)]
          (is (< 0.0 v 1.0)))))))

;; ---------------------------------------------------------------------------
;; Gamma distribution tests
;; ---------------------------------------------------------------------------

(deftest gamma-logpdf-vs-kixi
  (testing "MLX Gamma matches kixi"
    (doseq [[shape scale v] [[2.0 1.0 1.0]
                               [1.0 1.0 0.5]
                               [3.0 2.0 4.0]
                               [0.5 1.0 0.1]
                               [5.0 0.5 2.0]]]
      (let [kixi-val (d/logpdf (kixi/gamma-distribution shape scale) v)
            mlx-val  (d/logpdf (mlx-dist/gamma-distribution shape scale) v)]
        (is (close? kixi-val mlx-val 1e-3)
            (str "Gamma(" shape "," scale ") at v=" v
                 " kixi=" kixi-val " mlx=" mlx-val))))))

(deftest gamma-logpdf-vs-ll
  (testing "MLX Gamma matches log-likelihood module"
    (doseq [[shape scale v] [[2.0 1.0 1.0]
                               [3.0 2.0 4.0]
                               [0.5 1.0 0.1]]]
      (let [ll-val  (ll/gamma shape scale v)
            mlx-val (d/logpdf (mlx-dist/gamma-distribution shape scale) v)]
        (is (close? ll-val mlx-val 1e-3)
            (str "Gamma(" shape "," scale ") at v=" v))))))

(deftest gamma-out-of-bounds
  (testing "Returns ##-Inf for v <= 0"
    (let [d (mlx-dist/gamma-distribution 2.0 1.0)]
      (is (= ##-Inf (d/logpdf d 0.0)))
      (is (= ##-Inf (d/logpdf d -1.0))))))

(deftest gamma-gradient
  (testing "d/dv matches analytical: (shape-1)/v - 1/scale"
    (doseq [[shape scale v] [[2.0 1.0 1.0]
                               [3.0 2.0 4.0]
                               [5.0 0.5 2.0]]]
      (let [expected (- (/ (dec shape) v) (/ 1.0 scale))
            d        (mlx-dist/gamma-distribution shape scale)
            grad-fn  (xforms/grad
                      (fn [v-arr]
                        (mlx-dist/gamma-logpdf
                         (:shape-m1-arr d) (:inv-scale-arr d)
                         (:log-norm-arr d) v-arr)))
            actual   (arr/->double (grad-fn (arr/scalar v)))]
        (is (close? expected actual 1e-2)
            (str "d/dv Gamma(" shape "," scale ") at v=" v))))))

(deftest gamma-sampling
  (testing "Gamma sampling produces positive values"
    (let [d (mlx-dist/gamma-distribution 2.0 1.0)]
      (dotimes [_ 20]
        (is (pos? (d/sample d)))))))

;; ---------------------------------------------------------------------------
;; Poisson distribution tests
;; ---------------------------------------------------------------------------

(deftest poisson-logpdf-manual
  (testing "Matches hand-computed k*log(λ) - λ - log(k!)"
    (doseq [[lambda k] [[3.0 0]
                          [3.0 1]
                          [3.0 3]
                          [5.0 5]
                          [1.0 2]]]
      (let [expected (- (* k (Math/log lambda))
                        lambda
                        (gamma/log-gamma (inc (double k))))
            actual   (d/logpdf (mlx-dist/poisson-distribution lambda) k)]
        (is (close? expected actual 1e-6)
            (str "Poisson(" lambda ") at k=" k))))))

(deftest poisson-out-of-bounds
  (testing "Returns ##-Inf for k < 0"
    (is (= ##-Inf (d/logpdf (mlx-dist/poisson-distribution 3.0) -1)))))

(deftest poisson-sampling
  (testing "Poisson sampling produces non-negative integers"
    (let [d (mlx-dist/poisson-distribution 5.0)]
      (dotimes [_ 20]
        (let [v (d/sample d)]
          (is (>= v 0))
          (is (integer? v)))))))

;; ---------------------------------------------------------------------------
;; Multivariate Normal tests
;; ---------------------------------------------------------------------------

(deftest mvnormal-logpdf-standard
  (testing "2-d standard MVN at origin = -log(2π)"
    (let [d (mlx-dist/mvnormal-distribution [0.0 0.0]
                                            [[1.0 0.0]
                                             [0.0 1.0]])
          expected (- (Math/log (* 2.0 Math/PI)))  ;; -log(2π)
          actual   (d/logpdf d [0.0 0.0])]
      (is (close? expected actual 1e-3)
          (str "MVN(0, I) at origin: expected=" expected " actual=" actual)))))

(deftest mvnormal-logpdf-correlated
  (testing "Correlated 2-d MVN matches manual calculation"
    (let [mu  [1.0 2.0]
          cov [[1.0 0.5]
               [0.5 2.0]]
          d   (mlx-dist/mvnormal-distribution mu cov)
          x   [1.5 2.5]
          ;; Manual: L = chol(cov), solve L z = x-mu, logp = -0.5*(2*log(2π) + 2*sum(log(diag(L))) + z'z)
          ;; det(cov) = 1*2 - 0.5*0.5 = 1.75
          ;; logpdf = -0.5 * (2*log(2π) + log(det(cov)) + (x-mu)' inv(cov) (x-mu))
          det-cov 1.75
          ;; inv(cov) = (1/det) * [[2, -0.5], [-0.5, 1]]
          dx [(- 1.5 1.0) (- 2.5 2.0)]  ;; [0.5, 0.5]
          ;; (x-mu)' inv(cov) (x-mu) = (1/1.75) * (2*0.25 + (-0.5)*0.25 + (-0.5)*0.25 + 1*0.25)
          ;; = (1/1.75) * (0.5 - 0.125 - 0.125 + 0.25) = (1/1.75) * 0.5 = 0.2857...
          mahal (/ (+ (* 2.0 0.25) (* -0.5 0.25) (* -0.5 0.25) (* 1.0 0.25))
                   det-cov)
          expected (* -0.5 (+ (* 2 (Math/log (* 2.0 Math/PI)))
                              (Math/log det-cov)
                              mahal))
          actual (d/logpdf d x)]
      (is (close? expected actual 1e-2)
          (str "MVN correlated: expected=" expected " actual=" actual)))))

(deftest mvnormal-sampling-stats
  (testing "2000 samples from standard 2-d MVN have reasonable mean/variance"
    (let [d (mlx-dist/mvnormal-distribution [0.0 0.0]
                                            [[1.0 0.0]
                                             [0.0 1.0]])
          samples (repeatedly 2000 #(d/sample d))
          dim0 (mapv first samples)
          dim1 (mapv second samples)
          mean0 (/ (reduce + dim0) (count dim0))
          mean1 (/ (reduce + dim1) (count dim1))
          var0  (/ (reduce + (map #(let [d (- % mean0)] (* d d)) dim0)) (count dim0))
          var1  (/ (reduce + (map #(let [d (- % mean1)] (* d d)) dim1)) (count dim1))]
      (is (close? 0.0 mean0 0.15) (str "mean dim0=" mean0))
      (is (close? 0.0 mean1 0.15) (str "mean dim1=" mean1))
      (is (close? 1.0 var0  0.2)  (str "var dim0=" var0))
      (is (close? 1.0 var1  0.2)  (str "var dim1=" var1)))))

;; ---------------------------------------------------------------------------
;; Dirichlet distribution tests
;; ---------------------------------------------------------------------------

(deftest dirichlet-logpdf-uniform
  (testing "Dir([1,1,1]) = log(2) everywhere on simplex"
    (let [d (mlx-dist/dirichlet-distribution [1.0 1.0 1.0])
          ;; Dir(1,1,1) = flat on simplex, density = (3-1)! = 2, logpdf = log(2)
          expected (Math/log 2.0)
          actual   (d/logpdf d [0.3 0.3 0.4])]
      (is (close? expected actual 1e-3)
          (str "Dir([1,1,1]) at [0.3,0.3,0.4]: expected=" expected " actual=" actual)))))

(deftest dirichlet-logpdf-nonuniform
  (testing "Dir([2,3,1]) matches manual calculation"
    (let [alpha [2.0 3.0 1.0]
          v     [0.3 0.5 0.2]
          ;; logpdf = sum((α_i-1)*log(x_i)) + lgamma(sum(α)) - sum(lgamma(α_i))
          log-norm (- (gamma/log-gamma 6.0)
                      (+ (gamma/log-gamma 2.0) (gamma/log-gamma 3.0) (gamma/log-gamma 1.0)))
          expected (+ (* 1.0 (Math/log 0.3))
                      (* 2.0 (Math/log 0.5))
                      (* 0.0 (Math/log 0.2))
                      log-norm)
          actual   (d/logpdf (mlx-dist/dirichlet-distribution alpha) v)]
      (is (close? expected actual 1e-3)
          (str "Dir([2,3,1]) at [0.3,0.5,0.2]: expected=" expected " actual=" actual)))))

(deftest dirichlet-sampling-on-simplex
  (testing "All samples sum to ~1 and are positive"
    (let [d (mlx-dist/dirichlet-distribution [2.0 3.0 1.0])]
      (dotimes [_ 20]
        (let [v (d/sample d)]
          (is (every? pos? v) "All components positive")
          (is (close? 1.0 (reduce + v) 1e-6) "Components sum to 1"))))))

(deftest dirichlet-out-of-bounds
  (testing "Returns ##-Inf for invalid simplex values"
    (let [d (mlx-dist/dirichlet-distribution [2.0 3.0 1.0])]
      (is (= ##-Inf (d/logpdf d [-0.1 0.6 0.5])) "Negative component")
      (is (= ##-Inf (d/logpdf d [0.5 0.5 0.5])) "Sum != 1"))))

;; ---------------------------------------------------------------------------
;; Mixed-model test: build-score-fn with new distributions
;; ---------------------------------------------------------------------------

(deftest mixed-model-with-new-dists
  (testing "Model mixing Beta + Gamma + Normal: build-score-fn works"
    (let [model (mlx-dyn/gen [x]
                  (let [p (dynamic/trace! :p mlx-dist/beta-dist 2.0 5.0)
                        r (dynamic/trace! :rate mlx-dist/gamma-dist 2.0 1.0)
                        y (dynamic/trace! :y mlx-dist/normal 0.0 1.0)]
                    (+ p r y)))
          tr    (gf/simulate model [1.0])]
      (is (number? (trace/get-score tr)))
      (is (some? (trace/get-choices tr))))))
