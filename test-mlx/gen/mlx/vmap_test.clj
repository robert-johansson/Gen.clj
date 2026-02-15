(ns gen.mlx.vmap-test
  "Tests for vmap and manual batching patterns — parallel computation over
   batch dimensions."
  (:require [clojure.test :refer [deftest is testing]]
            [gen.mlx.array :as arr]
            [gen.mlx.transforms :as xforms]))

(defn- close?
  ([expected actual] (close? expected actual 1e-4))
  ([expected actual tol]
   (< (Math/abs (- (double expected) (double actual))) tol)))

;; ---------------------------------------------------------------------------
;; 1. Manual batching — element-wise operations broadcast over batch dim
;; ---------------------------------------------------------------------------

(deftest batched-arithmetic
  (testing "element-wise ops broadcast over (N, D) arrays"
    (let [;; 3 samples, each 2D
          a (arr/array [1 2 3 4 5 6] [3 2])
          b (arr/array [10 20 30 40 50 60] [3 2])
          result (arr/add a b)]
      (is (= [3 2] (arr/shape result)))
      (is (= [11.0 22.0 33.0 44.0 55.0 66.0] (arr/->vec result))))))

;; ---------------------------------------------------------------------------
;; 2. Batched score function using sum-axis
;; ---------------------------------------------------------------------------

(deftest batched-score-function
  (testing "batched score computes per-sample log-densities via sum-axis"
    (let [;; Score function for standard normal: f(x) = -0.5 * sum(x^2)
          ;; For (N, D) input, sum over axis 1 gives (N,) per-sample scores
          positions (arr/array [1 0   0 1   1 1] [3 2])
          scores (arr/mul -0.5 (arr/sum-axis (arr/square positions) 1))]
      (is (= [3] (arr/shape scores)))
      ;; [1,0] -> -0.5*1 = -0.5
      ;; [0,1] -> -0.5*1 = -0.5
      ;; [1,1] -> -0.5*2 = -1.0
      (is (= [-0.5 -0.5 -1.0] (arr/->vec scores))))))

;; ---------------------------------------------------------------------------
;; 3. Per-sample gradients via value-and-grad of sum-reduced batched score
;; ---------------------------------------------------------------------------

(deftest batched-gradients
  (testing "value-and-grad of sum-reduced score gives per-sample gradients"
    (let [;; Batched score: sum_i f(x_i) where f(x) = -0.5 * sum(x^2)
          ;; grad of sum_i f(x_i) w.r.t. X = [-x_1, -x_2, ...] = -X
          batched-score (fn [X]
                          (arr/sum (arr/mul -0.5 (arr/sum-axis (arr/square X) 1))))
          vag (xforms/value-and-grad batched-score)
          ;; 3 samples, each 2D
          positions (arr/array [1 2 3 4 5 6] [3 2])
          {:keys [value grads]} (vag positions)]
      ;; Value = sum of per-sample scores = -0.5*(1+4) + -0.5*(9+16) + -0.5*(25+36) = -45.5
      (is (close? -45.5 (arr/->double value)))
      ;; Gradient = -X (per-sample gradients stacked)
      (is (= [3 2] (arr/shape (first grads))))
      (is (= [-1.0 -2.0 -3.0 -4.0 -5.0 -6.0] (arr/->vec (first grads)))))))

;; ---------------------------------------------------------------------------
;; 4. Stack / unstack for parallel chain initialization
;; ---------------------------------------------------------------------------

(deftest stack-unstack-chains
  (testing "stack individual chain positions into batched array"
    (let [chain0 (arr/from-vec [0.0 0.0])
          chain1 (arr/from-vec [1.0 1.0])
          chain2 (arr/from-vec [2.0 2.0])
          batched (arr/stack [chain0 chain1 chain2])]
      (is (= [3 2] (arr/shape batched)))
      (is (= [0.0 0.0 1.0 1.0 2.0 2.0] (arr/->vec batched)))))
  (testing "extract individual chain via take-axis"
    (let [batched (arr/array [0 0 1 1 2 2] [3 2])
          chain1 (arr/squeeze (arr/take-axis batched (arr/from-ints [1]) 0) 0)]
      (is (= [2] (arr/shape chain1)))
      (is (= [1.0 1.0] (arr/->vec chain1))))))

;; ---------------------------------------------------------------------------
;; 5. Batched momentum sampling and kinetic energy
;; ---------------------------------------------------------------------------

(deftest batched-momentum
  (testing "random-normal with (N, D) shape produces correct dimensions"
    (let [momentum (arr/random-normal [4 3])]
      (is (= [4 3] (arr/shape momentum)))))
  (testing "per-chain kinetic energy via sum-axis"
    (let [momentum (arr/array [1 0  0 2  1 1] [3 2])
          ke (arr/mul 0.5 (arr/sum-axis (arr/square momentum) 1))]
      (is (= [3] (arr/shape ke)))
      ;; [1,0] -> 0.5*1 = 0.5
      ;; [0,2] -> 0.5*4 = 2.0
      ;; [1,1] -> 0.5*2 = 1.0
      (is (= [0.5 2.0 1.0] (arr/->vec ke))))))

;; ---------------------------------------------------------------------------
;; 6. Branchless per-chain MH accept/reject
;; ---------------------------------------------------------------------------

(deftest batched-mh-accept-reject
  (testing "where selects per-chain based on boolean mask"
    (let [;; 3 chains: accept chain 0 and 2, reject chain 1
          accepted (arr/less (arr/from-vec [0.1 0.9 0.2]) (arr/from-vec [0.5 0.5 0.5]))
          proposed (arr/array [10 20  30 40  50 60] [3 2])
          current  (arr/array [1 2  3 4  5 6] [3 2])
          ;; Need to expand accepted from (3,) to (3,1) for broadcasting with (3,2)
          accepted-2d (arr/expand-dims accepted 1)
          result (arr/where accepted-2d proposed current)]
      (is (= [3 2] (arr/shape result)))
      ;; chain 0: accepted → [10, 20]
      ;; chain 1: rejected → [3, 4]
      ;; chain 2: accepted → [50, 60]
      (is (= [10.0 20.0 3.0 4.0 50.0 60.0] (arr/->vec result))))))

;; ---------------------------------------------------------------------------
;; 7. vmap — element-wise square
;; ---------------------------------------------------------------------------

(deftest vmap-square
  (testing "vmap applies element-wise function over batch dimension"
    (let [f  (fn [x] (arr/mul x x))
          vf (xforms/vmap f)
          result (vf (arr/from-vec [1 2 3 4]))]
      (is (= [4] (arr/shape result)))
      (is (= [1.0 4.0 9.0 16.0] (arr/->vec result))))))

;; ---------------------------------------------------------------------------
;; 8. vmap — two inputs
;; ---------------------------------------------------------------------------

(deftest vmap-two-inputs
  (testing "vmap with two batched inputs"
    (let [f  (fn [x y] (arr/add x y))
          vf (xforms/vmap f :in-axes [0 0])
          result (vf (arr/from-vec [1 2 3]) (arr/from-vec [10 20 30]))]
      (is (= [3] (arr/shape result)))
      (is (= [11.0 22.0 33.0] (arr/->vec result))))))

;; ---------------------------------------------------------------------------
;; 9. vmap — multi-op function
;; ---------------------------------------------------------------------------

(deftest vmap-multi-op
  (testing "vmap with a function that chains multiple ops"
    (let [f  (fn [x] (arr/add (arr/mul x x) x))  ;; x^2 + x
          vf (xforms/vmap f)
          result (vf (arr/from-vec [0 1 2 3]))]
      (is (= [4] (arr/shape result)))
      (is (= [0.0 2.0 6.0 12.0] (arr/->vec result))))))

;; ---------------------------------------------------------------------------
;; 10. vmap + value-and-grad composition — validates parallel chain pattern
;; ---------------------------------------------------------------------------

(deftest vmap-with-value-and-grad
  (testing "vmap composes with value-and-grad through the closure bridge"
    (let [;; Single-sample score: (D,) → scalar
          score-fn (fn [x] (arr/mul -0.5 (arr/sum (arr/square x))))
          ;; vmap maps score over rows: (N, D) → (N,)
          vmapped-score (xforms/vmap score-fn)
          ;; value-and-grad of score-fn applied to (N, D) broadcasts naturally
          vag-fn (xforms/value-and-grad score-fn)
          ;; Test data: 3 samples, each 2D
          positions (arr/array [1 0  0 2  1 1] [3 2])]
      ;; vmap per-chain scores
      (let [per-chain (vmapped-score positions)]
        (is (= [3] (arr/shape per-chain)))
        ;; [1,0] → -0.5*1 = -0.5
        ;; [0,2] → -0.5*4 = -2.0
        ;; [1,1] → -0.5*2 = -1.0
        (is (= [-0.5 -2.0 -1.0] (arr/->vec per-chain))))
      ;; value-and-grad on batched input — total score + (N, D) gradient
      (let [{:keys [value grads]} (vag-fn positions)]
        ;; Total = -0.5 - 2.0 - 1.0 = -3.5
        (is (close? -3.5 (arr/->double value)))
        ;; Gradient = -X
        (is (= [3 2] (arr/shape (first grads))))
        (is (= [-1.0 0.0 0.0 -2.0 -1.0 -1.0] (arr/->vec (first grads))))))))
