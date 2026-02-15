(ns gen.mlx.map-optimize-test
  "Tests for MAP optimization via Adam gradient ascent.

   1. Adam state update — verify momentum/variance accumulation
   2. Quadratic optimization — maximize -(x-3)² → x converges to 3.0
   3. 2D optimization — bivariate normal → converges to mode
   4. Convergence history — log-density generally increases
   5. Early stopping — convergence tolerance terminates early"
  (:require [clojure.test :refer [deftest is testing]]
            [gen.mlx.array :as arr]
            [gen.mlx.map-optimize :as map-opt]
            [gen.mlx.transforms :as xforms]))

(defn- close?
  ([expected actual] (close? expected actual 1e-2))
  ([expected actual tol]
   (< (Math/abs (- (double expected) (double actual))) tol)))

;; ---------------------------------------------------------------------------
;; 1. Adam state initialization and basic step
;; ---------------------------------------------------------------------------

(deftest adam-state-initialization
  (testing "AdamState has correct initial structure"
    (let [pos (arr/from-vec [1.0 2.0])
          state (#'gen.mlx.map-optimize/make-adam-state pos)]
      (is (= 0 (.t state)))
      (is (= [0.0 0.0] (arr/->vec (.m state))))
      (is (= [0.0 0.0] (arr/->vec (.v state)))))))

(deftest adam-step-updates-position
  (testing "Adam step moves position toward gradient direction"
    (let [pos (arr/from-vec [0.0])
          state (#'gen.mlx.map-optimize/make-adam-state pos)
          ;; Positive gradient → position should increase (ascent)
          grad (arr/from-vec [1.0])
          new-state (map-opt/adam-step grad state {:lr 0.01 :beta1 0.9
                                                   :beta2 0.999 :epsilon 1e-8})]
      (is (= 1 (.t new-state)))
      (is (> (first (arr/->vec (.position new-state))) 0.0)
          "position should move in positive direction for positive gradient"))))

;; ---------------------------------------------------------------------------
;; 2. Quadratic optimization — maximize -(x-3)²
;; ---------------------------------------------------------------------------

(deftest optimize-quadratic-1d
  (testing "MAP finds mode of -(x-3)² ≈ x=3.0"
    (let [;; log-density = -(x-3)², mode at x=3
          log-density (fn [x] (arr/neg (arr/square (arr/sub (arr/sum x) 3.0))))
          result (map-opt/map-optimize log-density (arr/from-vec [0.0]) 1000
                                       :lr 0.05)]
      (is (close? 3.0 (first (arr/->vec (:position result))) 0.1)
          (str "should converge to x≈3.0, got "
               (first (arr/->vec (:position result)))))
      (is (close? 0.0 (:log-density result) 0.5)
          (str "log-density at mode should be ≈0, got " (:log-density result))))))

;; ---------------------------------------------------------------------------
;; 3. 2D optimization — bivariate normal → mode at (2, -1)
;; ---------------------------------------------------------------------------

(deftest optimize-2d-mode
  (testing "MAP finds mode of 2D quadratic"
    (let [;; log-density = -0.5 * sum((x - target)²), mode at (2, -1)
          target (arr/from-vec [2.0 -1.0])
          log-density (fn [x]
                        (arr/mul -0.5 (arr/sum (arr/square (arr/sub x target)))))
          result (map-opt/map-optimize log-density (arr/from-vec [0.0 0.0]) 2000
                                       :lr 0.05)
          pos (arr/->vec (:position result))]
      (is (close? 2.0 (first pos) 0.3)
          (str "x0 should be ≈2.0, got " (first pos)))
      (is (close? -1.0 (second pos) 0.3)
          (str "x1 should be ≈-1.0, got " (second pos))))))

;; ---------------------------------------------------------------------------
;; 4. Convergence history
;; ---------------------------------------------------------------------------

(deftest convergence-history-recorded
  (testing "history contains step and log-density entries"
    (let [log-density (fn [x] (arr/neg (arr/square (arr/sub (arr/sum x) 3.0))))
          result (map-opt/map-optimize log-density (arr/from-vec [0.0]) 100
                                       :lr 0.05)]
      (is (= 100 (count (:history result))))
      (is (every? #(contains? % :step) (:history result)))
      (is (every? #(contains? % :log-density) (:history result)))
      ;; log-density should generally increase (last > first for convex problem)
      (let [first-ld (:log-density (first (:history result)))
            last-ld  (:log-density (last (:history result)))]
        (is (> last-ld first-ld)
            (str "log-density should increase: first=" first-ld " last=" last-ld))))))

;; ---------------------------------------------------------------------------
;; 5. Early stopping with tolerance
;; ---------------------------------------------------------------------------

(deftest early-stopping
  (testing "optimization stops early when converged"
    (let [log-density (fn [x] (arr/neg (arr/square (arr/sub (arr/sum x) 3.0))))
          result (map-opt/map-optimize log-density (arr/from-vec [2.9]) 10000
                                       :lr 0.01 :tol 1e-8 :patience 10)]
      ;; Should stop well before 10000 steps
      (is (< (count (:history result)) 10000)
          (str "should stop early, took " (count (:history result)) " steps"))
      ;; Should still find the mode
      (is (close? 3.0 (first (arr/->vec (:position result))) 0.1)
          (str "should still find mode, got "
               (first (arr/->vec (:position result))))))))

;; ---------------------------------------------------------------------------
;; 6. map-step for custom loops
;; ---------------------------------------------------------------------------

(deftest map-step-single
  (testing "map-step performs a single optimization step"
    (let [log-density (fn [x] (arr/neg (arr/square (arr/sub (arr/sum x) 3.0))))
          vag-fn (xforms/value-and-grad log-density)
          state (#'gen.mlx.map-optimize/make-adam-state (arr/from-vec [0.0]))
          result (map-opt/map-step vag-fn state {:lr 0.01 :beta1 0.9
                                                  :beta2 0.999 :epsilon 1e-8})]
      (is (contains? result :state))
      (is (contains? result :log-density))
      (is (number? (:log-density result)))
      (is (instance? gen.mlx.map_optimize.AdamState (:state result))))))
