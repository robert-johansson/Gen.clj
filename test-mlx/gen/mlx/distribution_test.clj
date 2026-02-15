(ns gen.mlx.distribution-test
  "Tests for MLX-backed distributions.

   Four levels of verification:
   1. Direct logpdf comparison (MLX vs kixi)
   2. GenerativeFn generate — same constraints, same weights
   3. Gen model comparison — same model, both backends
   4. Gradient computation — the MLX advantage"
  (:require [clojure.test :refer [deftest is testing]]
            [gen.choicemap :as choicemap]
            [gen.distribution :as d]
            [gen.distribution.kixi :as kixi]
            [gen.dynamic :as dynamic]
            [gen.generative-function :as gf]
            [gen.mlx.array :as arr]
            [gen.mlx.distribution :as mlx-dist]
            [gen.mlx.transforms :as xforms]
            [gen.trace :as trace]))

(defn- close?
  "Check if two doubles are within tolerance."
  ([expected actual] (close? expected actual 1e-4))
  ([expected actual tol]
   (< (Math/abs (- (double expected) (double actual))) tol)))

;; ---------------------------------------------------------------------------
;; Level 1: Direct logpdf comparison (MLX vs kixi)
;;
;; Verify that MLX gaussian-logpdf matches kixi for spot-check values
;; from the existing distribution_test.cljc.
;; ---------------------------------------------------------------------------

(deftest gaussian-logpdf-spot-checks
  (testing "MLX matches kixi spot-check values"
    ;; Same values tested in gen.distribution-test/normal-tests
    (is (close? -1.0439385332046727
                (d/logpdf (mlx-dist/normal-distribution 0 1) 0.5)
                1e-4)
        "logpdf(Normal(0,1), 0.5)")

    (is (close? -1.643335713764618
                (d/logpdf (mlx-dist/normal-distribution 0 2) 0.5)
                1e-4)
        "logpdf(Normal(0,2), 0.5)")

    (is (close? -1.612085713764618
                (d/logpdf (mlx-dist/normal-distribution 0 2) 0)
                1e-4)
        "logpdf(Normal(0,2), 0)")))

(deftest gaussian-logpdf-vs-kixi
  (testing "MLX matches kixi for many parameter combinations"
    (doseq [[mu sigma v] [[0.0 1.0 0.0]
                           [0.0 1.0 1.0]
                           [0.0 1.0 -1.0]
                           [1.0 1.0 0.5]
                           [5.0 2.0 3.0]
                           [-3.0 0.5 -2.5]
                           [0.0 0.1 0.05]
                           [100.0 10.0 95.0]]]
      (let [kixi-val (d/logpdf (kixi/normal-distribution mu sigma) v)
            mlx-val  (d/logpdf (mlx-dist/normal-distribution mu sigma) v)]
        (is (close? kixi-val mlx-val 1e-3)
            (str "Normal(" mu "," sigma ") at v=" v
                 " kixi=" kixi-val " mlx=" mlx-val))))))

(deftest gaussian-logpdf-returns-mlxarray
  (testing "differentiable version returns MLXArray"
    (let [result (mlx-dist/gaussian-logpdf
                  (arr/scalar 0.0) (arr/scalar 1.0) (arr/scalar 0.5))]
      (is (instance? gen.mlx.array.MLXArray result))
      (is (close? -1.0439385332046727 (arr/->double result) 1e-4)))))

;; ---------------------------------------------------------------------------
;; Level 2: GenerativeFn generate — same constraints, same weights
;;
;; Verify that the GenerativeFn wrapper produces the same weight as kixi
;; when constrained to a specific value.
;; ---------------------------------------------------------------------------

(deftest generate-weight-comparison
  (testing "gf/generate with constraint produces identical weight"
    (doseq [[mu sigma v] [[0.0 1.0 0.5]
                           [1.0 2.0 -1.0]
                           [5.0 0.5 4.8]]]
      (let [kixi-result (gf/generate kixi/normal [mu sigma]
                                     (choicemap/choicemap v))
            mlx-result  (gf/generate mlx-dist/normal [mu sigma]
                                     (choicemap/choicemap v))]
        (is (close? (:weight kixi-result) (:weight mlx-result) 1e-3)
            (str "weight for Normal(" mu "," sigma ") constrained to " v))))))

(deftest generate-trace-score-comparison
  (testing "trace score matches between backends"
    (let [constraint (choicemap/choicemap 0.5)
          kixi-trace (:trace (gf/generate kixi/normal [0 1] constraint))
          mlx-trace  (:trace (gf/generate mlx-dist/normal [0 1] constraint))]
      (is (close? (trace/get-score kixi-trace)
                  (trace/get-score mlx-trace)
                  1e-3)))))

(deftest simulate-produces-valid-trace
  (testing "simulate produces a valid trace with score"
    (let [tr (gf/simulate mlx-dist/normal [0.0 1.0])]
      (is (= mlx-dist/normal (trace/get-gen-fn tr)))
      (is (= [0.0 1.0] (trace/get-args tr)))
      (is (number? (trace/get-retval tr)))
      (is (number? (trace/get-score tr)))
      (is (choicemap/has-value? (trace/get-choices tr))))))

;; ---------------------------------------------------------------------------
;; Level 3: Gen model comparison — same model, both backends
;;
;; Build identical models using kixi and MLX distributions,
;; constrain them identically, and verify matching weights/scores.
;; ---------------------------------------------------------------------------

(def model-kixi
  (dynamic/gen [x]
    (dynamic/trace! :y kixi/normal x 1.0)))

(def model-mlx
  (dynamic/gen [x]
    (dynamic/trace! :y mlx-dist/normal x 1.0)))

(deftest gen-model-comparison
  (testing "same model structure, same constraints → same weight"
    (let [constraints (choicemap/choicemap {:y 0.5})
          kixi-result (gf/generate model-kixi [0.0] constraints)
          mlx-result  (gf/generate model-mlx [0.0] constraints)]
      (is (close? (:weight kixi-result) (:weight mlx-result) 1e-3)
          "weights should match")
      (is (close? (trace/get-score (:trace kixi-result))
                  (trace/get-score (:trace mlx-result))
                  1e-3)
          "trace scores should match"))))

(deftest gen-model-different-args
  (testing "model comparison with different input args"
    (doseq [x [0.0 1.0 -2.0 5.0]]
      (let [constraints (choicemap/choicemap {:y 1.0})
            kixi-result (gf/generate model-kixi [x] constraints)
            mlx-result  (gf/generate model-mlx [x] constraints)]
        (is (close? (:weight kixi-result) (:weight mlx-result) 1e-3)
            (str "weights should match for x=" x))))))

;; ---------------------------------------------------------------------------
;; Level 4: Gradient computation — the MLX advantage
;;
;; Verify that we can differentiate logpdf w.r.t. parameters.
;; This is impossible with kixi — it's the whole reason MLX distributions exist.
;;
;; Analytical gradients of Gaussian logpdf:
;;   d/dmu[logpdf(mu, sigma, v)]    = (v - mu) / sigma^2
;;   d/dsigma[logpdf(mu, sigma, v)] = ((v-mu)^2 - sigma^2) / sigma^3
;; ---------------------------------------------------------------------------

(deftest grad-logpdf-wrt-mu
  (testing "d/dmu[logpdf(Normal(mu,1), 0.5)] at mu=0 should be 0.5"
    (let [grad-fn (xforms/grad
                   (fn [mu]
                     (mlx-dist/gaussian-logpdf mu (arr/scalar 1.0) (arr/scalar 0.5))))
          result (grad-fn (arr/scalar 0.0))]
      ;; d/dmu = (v - mu) / sigma^2 = (0.5 - 0) / 1 = 0.5
      (is (close? 0.5 (arr/->double result))))))

(deftest grad-logpdf-wrt-mu-various
  (testing "gradient w.r.t. mu for various parameter settings"
    (doseq [[mu sigma v] [[0.0 1.0 0.5]
                           [1.0 1.0 2.0]
                           [0.0 2.0 1.0]
                           [-1.0 0.5 0.0]]]
      (let [expected (/ (- v mu) (* sigma sigma))
            grad-fn  (xforms/grad
                      (fn [mu-arr]
                        (mlx-dist/gaussian-logpdf
                         mu-arr (arr/scalar sigma) (arr/scalar v))))
            actual   (arr/->double (grad-fn (arr/scalar mu)))]
        (is (close? expected actual 1e-3)
            (str "d/dmu at mu=" mu " sigma=" sigma " v=" v))))))

(deftest grad-logpdf-wrt-sigma
  (testing "gradient w.r.t. sigma"
    (doseq [[mu sigma v] [[0.0 1.0 0.5]
                           [1.0 2.0 3.0]
                           [0.0 0.5 0.0]]]
      ;; d/dsigma = ((v-mu)^2 - sigma^2) / sigma^3
      (let [diff     (- v mu)
            expected (/ (- (* diff diff) (* sigma sigma))
                        (* sigma sigma sigma))
            grad-fn  (xforms/grad
                      (fn [sigma-arr]
                        (mlx-dist/gaussian-logpdf
                         (arr/scalar mu) sigma-arr (arr/scalar v))))
            actual   (arr/->double (grad-fn (arr/scalar sigma)))]
        (is (close? expected actual 1e-3)
            (str "d/dsigma at mu=" mu " sigma=" sigma " v=" v))))))

(deftest value-and-grad-logpdf
  (testing "value-and-grad returns both logpdf value and gradient"
    (let [vag-fn (xforms/value-and-grad
                  (fn [mu]
                    (mlx-dist/gaussian-logpdf mu (arr/scalar 1.0) (arr/scalar 0.5))))
          result (vag-fn (arr/scalar 0.0))]
      ;; value = logpdf(Normal(0,1), 0.5)
      (is (close? -1.0439385332046727 (arr/->double (:value result)) 1e-3))
      ;; grad = (v - mu) / sigma^2 = 0.5
      (is (close? 0.5 (arr/->double (first (:grads result))))))))

(deftest grad-multiarg-logpdf
  (testing "gradient w.r.t. both mu and sigma simultaneously"
    (let [mu    0.0
          sigma 1.0
          v     0.5
          grad-fn (xforms/grad
                   (fn [mu-arr sigma-arr]
                     (mlx-dist/gaussian-logpdf mu-arr sigma-arr (arr/scalar v)))
                   {:argnums [0 1]})
          grads   (grad-fn (arr/scalar mu) (arr/scalar sigma))
          ;; d/dmu = (v - mu) / sigma^2 = 0.5
          ;; d/dsigma = ((v-mu)^2 - sigma^2) / sigma^3 = (0.25 - 1) / 1 = -0.75
          expected-dmu    0.5
          expected-dsigma -0.75]
      (is (vector? grads))
      (is (close? expected-dmu (arr/->double (first grads))))
      (is (close? expected-dsigma (arr/->double (second grads)))))))
