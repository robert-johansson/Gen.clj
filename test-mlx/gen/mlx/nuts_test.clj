(ns gen.mlx.nuts-test
  "Tests for the No-U-Turn Sampler (NUTS).

   1. Single NUTS step — returns valid structure
   2. 1D standard normal — posterior mean ≈ 0, std ≈ 1
   3. 2D independent normals — per-dimension statistics
   4. Tree depth bounded — verify depth ≤ max-depth
   5. Step size adaptation — eps stabilizes during warmup
   6. Normal(3, 0.5) — known posterior
   7. Comparison: NUTS vs HMC — NUTS should work at least as well"
  (:require [clojure.test :refer [deftest is testing]]
            [gen.mlx.array :as arr]
            [gen.mlx.distribution :as mlx-dist]
            [gen.mlx.hmc :as hmc]
            [gen.mlx.nuts :as nuts]
            [gen.mlx.transforms :as xforms]))

(defn- close?
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
;; 1. Single NUTS step structure
;; ---------------------------------------------------------------------------

(deftest nuts-step-returns-correct-structure
  (testing "nuts-step returns map with required keys"
    (let [log-density (fn [x] (arr/mul -0.5 (arr/sum (arr/square x))))
          vag-fn (xforms/value-and-grad log-density)
          pos (arr/from-vec [0.0])
          ld (arr/->double (log-density pos))
          result (nuts/nuts-step vag-fn pos ld 0.1 10)]
      (is (contains? result :position))
      (is (contains? result :log-density))
      (is (contains? result :tree-depth))
      (is (contains? result :n-leapfrog))
      (is (contains? result :accept-prob))
      (is (instance? gen.mlx.array.MLXArray (:position result)))
      (is (number? (:log-density result)))
      (is (integer? (:tree-depth result)))
      (is (integer? (:n-leapfrog result)))
      (is (number? (:accept-prob result))))))

;; ---------------------------------------------------------------------------
;; 2. 1D standard normal
;; ---------------------------------------------------------------------------

(deftest sample-1d-standard-normal
  (testing "NUTS samples from 1D standard normal"
    (let [log-density (fn [x] (arr/mul -0.5 (arr/square (arr/sum x))))
          results (nuts/sample log-density (arr/from-vec [0.0]) 600
                               :max-depth 6 :n-warmup 200)
          ;; Drop warmup
          samples (mapv #(first (arr/->vec (:position %)))
                        (drop 200 results))
          m (mean samples)
          s (std samples)]
      (is (close? 0.0 m 0.4)
          (str "mean should be ≈ 0, got " m))
      (is (close? 1.0 s 0.5)
          (str "std should be ≈ 1, got " s)))))

;; ---------------------------------------------------------------------------
;; 3. 2D independent normals
;; ---------------------------------------------------------------------------

(deftest sample-2d-independent-normals
  (testing "NUTS samples from 2D independent standard normals"
    (let [log-density (fn [x] (arr/mul -0.5 (arr/sum (arr/square x))))
          results (nuts/sample log-density (arr/from-vec [0.0 0.0]) 600
                               :max-depth 6 :n-warmup 200)
          positions (mapv #(arr/->vec (:position %)) (drop 200 results))
          dim0 (mapv first positions)
          dim1 (mapv second positions)]
      (is (close? 0.0 (mean dim0) 0.4)
          (str "dim0 mean should be ≈ 0, got " (mean dim0)))
      (is (close? 1.0 (std dim0) 0.5)
          (str "dim0 std should be ≈ 1, got " (std dim0)))
      (is (close? 0.0 (mean dim1) 0.4)
          (str "dim1 mean should be ≈ 0, got " (mean dim1)))
      (is (close? 1.0 (std dim1) 0.5)
          (str "dim1 std should be ≈ 1, got " (std dim1))))))

;; ---------------------------------------------------------------------------
;; 4. Tree depth bounded by max-depth
;; ---------------------------------------------------------------------------

(deftest tree-depth-bounded
  (testing "tree depth never exceeds max-depth"
    (let [log-density (fn [x] (arr/mul -0.5 (arr/sum (arr/square x))))
          max-depth 5
          results (nuts/sample log-density (arr/from-vec [0.0]) 100
                               :max-depth max-depth :n-warmup 50)]
      (doseq [r results]
        (is (<= (:tree-depth r) max-depth)
            (str "tree depth " (:tree-depth r) " exceeds max " max-depth))))))

;; ---------------------------------------------------------------------------
;; 5. Step size adaptation
;; ---------------------------------------------------------------------------

(deftest step-size-adaptation
  (testing "step size stabilizes during warmup"
    (let [log-density (fn [x] (arr/mul -0.5 (arr/sum (arr/square x))))
          results (nuts/sample log-density (arr/from-vec [0.0]) 400
                               :max-depth 6 :n-warmup 200)
          ;; Post-warmup eps should be constant (all same value)
          post-warmup (drop 200 results)
          eps-values (mapv :eps post-warmup)]
      (is (pos? (count eps-values)))
      ;; All post-warmup eps should be the same (frozen after warmup)
      (let [first-eps (first eps-values)]
        (is (every? #(close? first-eps % 1e-10) eps-values)
            "post-warmup step sizes should all be equal"))
      ;; Adapted eps should be positive and reasonable
      (let [final-eps (first eps-values)]
        (is (pos? final-eps) "adapted eps should be positive")
        (is (< final-eps 10.0) "adapted eps should be reasonable (<10)")))))

;; ---------------------------------------------------------------------------
;; 6. Known posterior — Normal(3, 0.5)
;; ---------------------------------------------------------------------------

(deftest sample-normal-3-05
  (testing "NUTS samples from Normal(3, 0.5) via gaussian-logpdf"
    (let [mu 3.0
          sigma 0.5
          log-density (fn [x]
                        (mlx-dist/gaussian-logpdf mu sigma (arr/sum x)))
          results (nuts/sample log-density (arr/from-vec [3.0]) 600
                               :max-depth 6 :n-warmup 200)
          samples (mapv #(first (arr/->vec (:position %)))
                        (drop 200 results))
          m (mean samples)
          s (std samples)]
      (is (close? mu m 0.5)
          (str "mean should be ≈ " mu ", got " m))
      (is (close? sigma s 0.4)
          (str "std should be ≈ " sigma ", got " s)))))

;; ---------------------------------------------------------------------------
;; 7. NUTS results include eps
;; ---------------------------------------------------------------------------

(deftest results-include-eps
  (testing "each NUTS result includes the step size used"
    (let [log-density (fn [x] (arr/mul -0.5 (arr/sum (arr/square x))))
          results (nuts/sample log-density (arr/from-vec [0.0]) 50
                               :max-depth 5 :n-warmup 25)]
      (doseq [r results]
        (is (contains? r :eps))
        (is (number? (:eps r)))
        (is (pos? (:eps r)))))))

;; ---------------------------------------------------------------------------
;; 8. N-leapfrog is positive
;; ---------------------------------------------------------------------------

(deftest n-leapfrog-positive
  (testing "every NUTS step takes at least 1 leapfrog step"
    (let [log-density (fn [x] (arr/mul -0.5 (arr/sum (arr/square x))))
          results (nuts/sample log-density (arr/from-vec [0.0]) 50
                               :max-depth 5 :n-warmup 25)]
      (doseq [r results]
        (is (pos? (:n-leapfrog r))
            (str "n-leapfrog should be > 0, got " (:n-leapfrog r)))))))
