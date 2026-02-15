(ns gen.mlx.dynamic-test
  "Tests for the MLX dynamic DSL.

   1. Simulate basic — single choice, correct trace structure
   2. Simulate multi-choice — two choices, score is sum of logpdfs
   3. Generate constrained — fixed choice, weight = logpdf
   4. Generate matches gen.dynamic — same model + constraints → same weight
   5. Choice gradients (single) — analytical gradient check
   6. Choice gradients (multi) — per-choice gradient verification
   7. HMC from trace — Normal(3, 0.5) posterior
   8. HMC linear regression — slope posterior"
  (:require [clojure.test :refer [deftest is testing]]
            [gen.choicemap :as choicemap]
            [gen.distribution :as d]
            [gen.dynamic :as dynamic]
            [gen.generative-function :as gf]
            [gen.mlx.distribution :as mlx-dist]
            [gen.mlx.dynamic :as mlx-dyn]
            [gen.trace :as trace]))

(defn- close?
  ([expected actual] (close? expected actual 1e-3))
  ([expected actual tol]
   (< (Math/abs (- (double expected) (double actual))) tol)))

(defn- mean [xs]
  (/ (reduce + xs) (count xs)))

(defn- std [xs]
  (let [m (mean xs)
        variance (/ (reduce + (map #(let [d (- % m)] (* d d)) xs))
                    (count xs))]
    (Math/sqrt variance)))

;; ---------------------------------------------------------------------------
;; 1. Simulate basic — single choice
;; ---------------------------------------------------------------------------

(def single-model
  (mlx-dyn/gen [mu]
    (dynamic/trace! :x mlx-dist/normal mu 1.0)))

(deftest simulate-basic
  (testing "simulate produces correct trace structure"
    (let [tr (gf/simulate single-model [0.0])]
      (is (= single-model (trace/get-gen-fn tr)))
      (is (= [0.0] (trace/get-args tr)))
      (is (number? (trace/get-retval tr)))
      (is (number? (trace/get-score tr)))
      (let [cm (trace/get-choices tr)]
        (is (choicemap/has-value? cm :x))
        (is (number? (choicemap/get-value cm :x)))))))

;; ---------------------------------------------------------------------------
;; 2. Simulate multi-choice — two choices, score = sum of logpdfs
;; ---------------------------------------------------------------------------

(def two-choice-model
  (mlx-dyn/gen []
    (let [a (dynamic/trace! :a mlx-dist/normal 0.0 1.0)]
      (dynamic/trace! :b mlx-dist/normal a 1.0))))

(deftest simulate-multi-choice
  (testing "two choices both appear in choicemap, score is sum of logpdfs"
    (let [tr (gf/simulate two-choice-model [])]
      (let [cm (trace/get-choices tr)]
        (is (choicemap/has-value? cm :a))
        (is (choicemap/has-value? cm :b)))
      ;; Score should be sum of individual logpdfs
      (let [a-val (choicemap/get-value (trace/get-choices tr) :a)
            b-val (choicemap/get-value (trace/get-choices tr) :b)
            lp-a  (d/logpdf (mlx-dist/normal-distribution 0.0 1.0) a-val)
            lp-b  (d/logpdf (mlx-dist/normal-distribution a-val 1.0) b-val)]
        (is (close? (+ lp-a lp-b) (trace/get-score tr))
            "score should be sum of individual logpdfs")))))

;; ---------------------------------------------------------------------------
;; 3. Generate constrained — fixed choice, weight = logpdf
;; ---------------------------------------------------------------------------

(deftest generate-constrained
  (testing "constrained choice gives correct weight"
    (let [constraints (choicemap/choicemap {:x 0.5})
          result      (gf/generate single-model [0.0] constraints)
          expected-lp (d/logpdf (mlx-dist/normal-distribution 0.0 1.0) 0.5)]
      (is (close? expected-lp (:weight result))
          "weight should equal logpdf of constrained value")
      (is (close? expected-lp (trace/get-score (:trace result)))
          "trace score should equal logpdf")
      (is (= 0.5 (choicemap/get-value (trace/get-choices (:trace result)) :x))
          "choice value should be the constrained value"))))

;; ---------------------------------------------------------------------------
;; 4. Generate matches gen.dynamic — same weight with both DSLs
;; ---------------------------------------------------------------------------

(def dynamic-model
  (dynamic/gen [mu]
    (dynamic/trace! :x mlx-dist/normal mu 1.0)))

(deftest generate-matches-gen-dynamic
  (testing "same model + constraints -> same weight in both DSLs"
    (doseq [mu [0.0 1.0 -2.0 5.0]]
      (let [constraints (choicemap/choicemap {:x 0.5})
            dyn-result  (gf/generate dynamic-model [mu] constraints)
            mlx-result  (gf/generate single-model [mu] constraints)]
        (is (close? (:weight dyn-result) (:weight mlx-result))
            (str "weights should match for mu=" mu))))))

;; ---------------------------------------------------------------------------
;; 5. Choice gradients (single) — d/dx[logpdf(N(0,1), x)] at x=0.5
;; ---------------------------------------------------------------------------

(deftest choice-gradients-single
  (testing "gradient of logpdf N(0,1) w.r.t. x at x=0.5"
    (let [constraints (choicemap/choicemap {:x 0.5})
          result      (gf/generate single-model [0.0] constraints)
          grads       (trace/choice-gradients (:trace result) nil nil)
          ;; d/dx[logpdf(N(0,1), x)] = (0 - x) / 1^2 = -x = -0.5
          expected    -0.5
          actual      (choicemap/get-value
                       (choicemap/get-submap (:choice-grads grads) :x))]
      (is (close? expected actual)
          (str "gradient should be -0.5, got " actual)))))

;; ---------------------------------------------------------------------------
;; 6. Choice gradients (multi) — two independent choices
;; ---------------------------------------------------------------------------

(def independent-model
  (mlx-dyn/gen []
    (dynamic/trace! :a mlx-dist/normal 0.0 1.0)
    (dynamic/trace! :b mlx-dist/normal 3.0 2.0)))

(deftest choice-gradients-multi
  (testing "per-choice gradients for independent choices"
    (let [constraints (choicemap/choicemap {:a 0.5 :b 4.0})
          result      (gf/generate independent-model [] constraints)
          grads       (trace/choice-gradients (:trace result) nil nil)
          ;; d/da[logpdf(N(0,1), a)] = (0 - a)/1 = -0.5
          ;; d/db[logpdf(N(3,2), b)] = (3 - b)/4 = -0.25
          grad-a      (choicemap/get-value
                       (choicemap/get-submap (:choice-grads grads) :a))
          grad-b      (choicemap/get-value
                       (choicemap/get-submap (:choice-grads grads) :b))]
      (is (close? -0.5 grad-a)
          (str "d/da should be -0.5, got " grad-a))
      (is (close? -0.25 grad-b)
          (str "d/db should be -0.25, got " grad-b)))))

;; ---------------------------------------------------------------------------
;; 7. HMC from trace — Normal(3, 0.5) posterior
;; ---------------------------------------------------------------------------

(def normal-model
  (mlx-dyn/gen []
    (dynamic/trace! :x mlx-dist/normal 3.0 0.5)))

(deftest hmc-from-trace
  (testing "HMC samples from Normal(3, 0.5)"
    (let [constraints (choicemap/choicemap {:x 3.0})
          result      (gf/generate normal-model [] constraints)
          samples     (mlx-dyn/hmc-sample (:trace result) 500
                                          :L 10 :eps 0.05)
          ;; Burn first 100
          values      (mapv #(get (:choices %) :x) (drop 100 samples))
          m           (mean values)
          s           (std values)]
      (is (close? 3.0 m 0.5)
          (str "mean should be ~3.0, got " m))
      (is (close? 0.5 s 0.3)
          (str "std should be ~0.5, got " s)))))

;; ---------------------------------------------------------------------------
;; 8. HMC linear regression — slope ~ N(0,10), y ~ N(slope*x, 0.1)
;; ---------------------------------------------------------------------------

(def linreg-model
  (mlx-dyn/gen [x]
    (let [slope (dynamic/trace! :slope mlx-dist/normal 0.0 10.0)]
      (dynamic/trace! :y mlx-dist/normal (* slope x) 0.1))))

(deftest hmc-linear-regression
  (testing "HMC posterior for slope given y=2.1 at x=1"
    (let [constraints (choicemap/choicemap {:slope 2.0 :y 2.1})
          result      (gf/generate linreg-model [1.0] constraints)
          samples     (mlx-dyn/hmc-sample (:trace result) 500
                                          :L 10 :eps 0.005)
          slope-vals  (mapv #(get (:choices %) :slope) (drop 100 samples))
          m           (mean slope-vals)]
      (is (close? 2.1 m 0.5)
          (str "posterior slope mean should be ~2.1, got " m)))))
