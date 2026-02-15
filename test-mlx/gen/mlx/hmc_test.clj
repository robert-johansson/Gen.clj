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

;; ---------------------------------------------------------------------------
;; 7. MALA — single step (HMC with L=1)
;; ---------------------------------------------------------------------------

(deftest mala-step-returns-correct-structure
  (testing "mala-step returns map with required keys"
    (let [log-density (fn [x] (arr/mul -0.5 (arr/sum (arr/square x))))
          result (hmc/mala-step log-density (arr/from-vec [0.0])
                                :eps 0.1)]
      (is (contains? result :position))
      (is (contains? result :log-density))
      (is (contains? result :accepted?))
      (is (instance? gen.mlx.array.MLXArray (:position result))))))

;; ---------------------------------------------------------------------------
;; 8. MALA sampling — 1D standard normal
;; ---------------------------------------------------------------------------

(deftest mala-sample-1d-standard-normal
  (testing "MALA samples from 1D standard normal"
    (let [log-density (fn [x] (arr/mul -0.5 (arr/square (arr/sum x))))
          results (hmc/mala-sample log-density (arr/from-vec [0.0]) 800
                                   :eps 0.5)
          samples (mapv #(first (arr/->vec (:position %)))
                        (drop 200 results))
          m (mean samples)
          s (std samples)]
      (is (close? 0.0 m 0.4)
          (str "mean should be ≈ 0, got " m))
      (is (close? 1.0 s 0.5)
          (str "std should be ≈ 1, got " s)))))

;; ---------------------------------------------------------------------------
;; 9. MALA acceptance rate
;; ---------------------------------------------------------------------------

(deftest mala-acceptance-rate
  (testing "MALA acceptance rate is reasonable"
    (let [log-density (fn [x] (arr/mul -0.5 (arr/sum (arr/square x))))
          results (hmc/mala-sample log-density (arr/from-vec [0.0]) 300
                                   :eps 0.5)
          n-accepted (count (filter :accepted? results))
          rate (/ (double n-accepted) (count results))]
      (is (> rate 0.1)
          (str "acceptance rate should be > 10%, got " (* 100 rate) "%")))))

;; ---------------------------------------------------------------------------
;; 10. Lazy leapfrog
;; ---------------------------------------------------------------------------

(deftest leapfrog-lazy-returns-mlx-arrays
  (testing "leapfrog-lazy returns all MLXArrays (no eval)"
    (let [log-density (fn [x] (arr/mul -0.5 (arr/sum (arr/square x))))
          vag-fn (xforms/value-and-grad log-density)
          q (arr/from-vec [1.0 -0.5])
          p (arr/from-vec [0.5 0.3])
          eps-arr (arr/scalar 0.1)
          half-eps-arr (arr/scalar 0.05)
          result (hmc/leapfrog-lazy vag-fn q p eps-arr half-eps-arr 5)]
      (is (instance? gen.mlx.array.MLXArray (:position result)))
      (is (instance? gen.mlx.array.MLXArray (:momentum result)))
      (is (instance? gen.mlx.array.MLXArray (:log-density result)))
      ;; log-density should be a scalar MLXArray, not yet evaluated
      (is (= [] (:shape (:log-density result)))))))

;; ---------------------------------------------------------------------------
;; 11. Leapfrog one step
;; ---------------------------------------------------------------------------

(deftest leapfrog-one-step-test
  (testing "leapfrog-one-step returns valid result"
    (let [log-density (fn [x] (arr/mul -0.5 (arr/sum (arr/square x))))
          vag-fn (xforms/value-and-grad log-density)
          q (arr/from-vec [1.0])
          p (arr/from-vec [0.5])
          result (hmc/leapfrog-one-step vag-fn q p 0.1)]
      (is (contains? result :position))
      (is (contains? result :momentum))
      (is (contains? result :log-density))
      (is (instance? gen.mlx.array.MLXArray (:position result)))
      (is (instance? gen.mlx.array.MLXArray (:momentum result)))
      (is (number? (:log-density result))))))

;; ---------------------------------------------------------------------------
;; 12. Parallel chain HMC — structure
;; ---------------------------------------------------------------------------

(deftest parallel-sample-structure
  (testing "parallel-sample returns correct structure"
    (let [;; Single-sample score function: (D,) → scalar
          score-fn (fn [x] (arr/mul -0.5 (arr/sum (arr/square x))))
          ;; 4 chains, 2 dimensions
          init-pos (arr/array [0 0 0 0 0 0 0 0] [4 2])
          results (hmc/parallel-sample score-fn init-pos 5 :L 5 :eps 0.1)]
      (is (= 5 (count results)))
      (is (contains? (first results) :positions))
      (is (contains? (first results) :log-densities))
      (is (contains? (first results) :accepted))
      (is (= [4 2] (arr/shape (:positions (first results)))))
      (is (= 4 (count (:log-densities (first results)))))
      (is (= 4 (count (:accepted (first results))))))))

;; ---------------------------------------------------------------------------
;; 13. Parallel chains — posterior statistics
;; ---------------------------------------------------------------------------

(deftest parallel-sample-posterior
  (testing "parallel chains produce correct posterior statistics for 2D normal"
    (let [;; Single-sample score function: (D,) → scalar
          score-fn (fn [x] (arr/mul -0.5 (arr/sum (arr/square x))))
          ;; 4 chains, 2 dimensions
          init-pos (arr/array [0 0 0 0 0 0 0 0] [4 2])
          results (hmc/parallel-sample score-fn init-pos 300 :L 10 :eps 0.1)
          ;; Collect all samples from all chains (after burn-in)
          samples (mapcat (fn [step]
                            (let [flat (arr/->vec (:positions step))]
                              (partition 2 flat)))
                          (drop 50 results))
          dim0 (map first samples)
          dim1 (map second samples)]
      ;; With 4 chains x 250 post-burnin steps = 1000 samples
      (is (close? 0.0 (mean dim0) 0.3)
          (str "dim0 mean should be ≈ 0, got " (mean dim0)))
      (is (close? 1.0 (std dim0) 0.3)
          (str "dim0 std should be ≈ 1, got " (std dim0)))
      (is (close? 0.0 (mean dim1) 0.3)
          (str "dim1 mean should be ≈ 0, got " (mean dim1)))
      (is (close? 1.0 (std dim1) 0.3)
          (str "dim1 std should be ≈ 1, got " (std dim1))))))

;; ---------------------------------------------------------------------------
;; 14. Parallel chains — acceptance rate
;; ---------------------------------------------------------------------------

(deftest parallel-sample-acceptance-rate
  (testing "parallel chains have reasonable acceptance rate"
    (let [;; Single-sample score function: (D,) → scalar
          score-fn (fn [x] (arr/mul -0.5 (arr/sum (arr/square x))))
          init-pos (arr/array [0 0 0 0 0 0 0 0] [4 2])
          results (hmc/parallel-sample score-fn init-pos 100 :L 10 :eps 0.1)
          ;; Count total acceptances across all chains
          total-accepts (reduce + (map (fn [step]
                                         (count (filter true? (:accepted step))))
                                       results))
          total-proposals (* 4 100)
          rate (/ (double total-accepts) total-proposals)]
      (is (> rate 0.2)
          (str "acceptance rate should be > 20%, got " (* 100 rate) "%")))))
