(ns gen.mlx.hmc-test
  "Tests for MLX-based Hamiltonian Monte Carlo.

   1. Leapfrog reversibility — forward then backward recovers starting position
   2. 1D standard normal — mean ≈ 0, std ≈ 1
   3. 2D independent normals — same checks per dimension
   4. Acceptance rate — reasonable eps gives > 20%
   5. Integration with gaussian-logpdf"
  (:require [clojure.test :refer [deftest is testing]]
            [gen.mlx.array :as arr]
            [gen.mlx.distribution :as mlx-dist]
            [gen.mlx.hmc :as hmc]
            [gen.mlx.transforms :as xforms]))

(defn- close?
  "Check if two doubles are within tolerance."
  ([expected actual] (close? expected actual 1e-4))
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
;; 1. Leapfrog reversibility
;;
;; The leapfrog integrator is time-reversible: if we run L steps forward,
;; negate momentum, then run L steps forward again, we should recover
;; the original position (up to floating-point error).
;; ---------------------------------------------------------------------------

(deftest leapfrog-reversibility
  (testing "forward L steps, negate momentum, backward L steps ≈ start"
    (let [log-density (fn [x] (arr/mul -0.5 (arr/sum (arr/square x))))
          vag-fn (xforms/value-and-grad log-density)
          q0 (arr/from-vec [1.5 -0.7 0.3])
          p0 (arr/from-vec [0.5 -0.2 0.8])
          eps 0.05
          L 20
          ;; Forward integration
          fwd (hmc/leapfrog vag-fn q0 p0 eps L)
          ;; Negate momentum and integrate again
          p-neg (arr/neg (:momentum fwd))
          bwd (hmc/leapfrog vag-fn (:position fwd) p-neg eps L)
          ;; Should recover original position
          q-recovered (arr/->vec (:position bwd))
          q-original (arr/->vec q0)]
      (doseq [i (range 3)]
        (is (close? (nth q-original i) (nth q-recovered i) 1e-3)
            (str "dimension " i " should be recovered"))))))

;; ---------------------------------------------------------------------------
;; 2. 1D standard normal — log-density = -0.5 * x^2
;; ---------------------------------------------------------------------------

(deftest sample-1d-standard-normal
  (testing "HMC samples from 1D standard normal"
    (let [log-density (fn [x] (arr/mul -0.5 (arr/square (arr/sum x))))
          results (hmc/sample log-density (arr/from-vec [0.0]) 500
                              :L 10 :eps 0.1)
          ;; Burn first 100
          samples (mapv #(first (arr/->vec (:position %)))
                        (drop 100 results))
          m (mean samples)
          s (std samples)]
      (is (close? 0.0 m 0.3)
          (str "mean should be ≈ 0, got " m))
      (is (close? 1.0 s 0.3)
          (str "std should be ≈ 1, got " s)))))

;; ---------------------------------------------------------------------------
;; 3. 2D independent normals
;; ---------------------------------------------------------------------------

(deftest sample-2d-independent-normals
  (testing "HMC samples from 2D independent standard normals"
    (let [log-density (fn [x] (arr/mul -0.5 (arr/sum (arr/square x))))
          results (hmc/sample log-density (arr/from-vec [0.0 0.0]) 500
                              :L 10 :eps 0.1)
          positions (mapv #(arr/->vec (:position %)) (drop 100 results))
          dim0 (mapv first positions)
          dim1 (mapv second positions)]
      (is (close? 0.0 (mean dim0) 0.3)
          (str "dim0 mean should be ≈ 0, got " (mean dim0)))
      (is (close? 1.0 (std dim0) 0.3)
          (str "dim0 std should be ≈ 1, got " (std dim0)))
      (is (close? 0.0 (mean dim1) 0.3)
          (str "dim1 mean should be ≈ 0, got " (mean dim1)))
      (is (close? 1.0 (std dim1) 0.3)
          (str "dim1 std should be ≈ 1, got " (std dim1))))))

;; ---------------------------------------------------------------------------
;; 4. Acceptance rate
;; ---------------------------------------------------------------------------

(deftest acceptance-rate-reasonable
  (testing "acceptance rate > 20% with reasonable step size"
    (let [log-density (fn [x] (arr/mul -0.5 (arr/sum (arr/square x))))
          results (hmc/sample log-density (arr/from-vec [0.0]) 200
                              :L 10 :eps 0.1)
          n-accepted (count (filter :accepted? results))
          rate (/ (double n-accepted) (count results))]
      (is (> rate 0.2)
          (str "acceptance rate should be > 20%, got " (* 100 rate) "%")))))

;; ---------------------------------------------------------------------------
;; 5. Integration with gaussian-logpdf
;; ---------------------------------------------------------------------------

(deftest hmc-with-gaussian-logpdf
  (testing "HMC using gaussian-logpdf as log-density"
    (let [;; Target: Normal(3.0, 0.5)
          ;; log p(x) = gaussian-logpdf(3.0, 0.5, x)
          mu 3.0
          sigma 0.5
          log-density (fn [x]
                        (mlx-dist/gaussian-logpdf mu sigma (arr/sum x)))
          results (hmc/sample log-density (arr/from-vec [0.0]) 500
                              :L 10 :eps 0.05)
          samples (mapv #(first (arr/->vec (:position %)))
                        (drop 100 results))
          m (mean samples)
          s (std samples)]
      (is (close? mu m 0.3)
          (str "mean should be ≈ " mu ", got " m))
      (is (close? sigma s 0.3)
          (str "std should be ≈ " sigma ", got " s)))))

;; ---------------------------------------------------------------------------
;; 6. Single step API
;; ---------------------------------------------------------------------------

(deftest hmc-step-returns-correct-structure
  (testing "hmc-step returns map with required keys"
    (let [log-density (fn [x] (arr/mul -0.5 (arr/sum (arr/square x))))
          result (hmc/hmc-step log-density (arr/from-vec [0.0])
                               :L 5 :eps 0.1)]
      (is (contains? result :position))
      (is (contains? result :log-density))
      (is (contains? result :accepted?))
      (is (instance? gen.mlx.array.MLXArray (:position result)))
      (is (number? (:log-density result)))
      (is (boolean? (:accepted? result))))))
