(ns gen.inference.particle-filter-test
  (:require [clojure.test :refer [deftest is testing]]
            [gen.choicemap :as choicemap]
            [gen.combinator.unfold :as unfold-combinator]
            [gen.distribution.kixi :as dist]
            [gen.dynamic :as dynamic :refer [gen]]
            [gen.inference.particle-filter :as pf]
            [gen.trace :as trace]))

(def simple-model
  (gen [y-obs]
    (let [x (dynamic/trace! :x dist/normal 0.0 5.0)]
      (dynamic/trace! :y dist/normal x 1.0)
      x)))

(deftest initialize-test
  (testing "initialize creates correct number of particles"
    (let [n 10
          state (pf/initialize simple-model [0.0]
                               (choicemap/choicemap {:y 3.0})
                               n)]
      (is (= n (:n-particles state)))
      (is (= n (count (pf/get-traces state))))
      (is (= n (count (pf/get-log-weights state))))
      (is (number? (pf/log-ml-estimate state)))
      ;; All traces should satisfy constraint
      (doseq [tr (pf/get-traces state)]
        (is (= 3.0 (choicemap/get-value (trace/get-choices tr) :y))
            "each particle should satisfy observation constraint")))))

(deftest ess-test
  (testing "ESS is in valid range"
    (let [state (pf/initialize simple-model [0.0]
                               (choicemap/choicemap {:y 3.0})
                               10)
          ess (pf/effective-sample-size state)]
      (is (<= 1.0 ess 10.0)
          (str "ESS should be in [1, n-particles], got " ess)))))

(deftest resample-test
  (testing "resample produces correct number of particles"
    (let [state (pf/initialize simple-model [0.0]
                               (choicemap/choicemap {:y 3.0})
                               10)
          resampled (pf/resample state)]
      (is (= 10 (count (pf/get-traces resampled))))
      (is (= 10 (count (pf/get-log-weights resampled))))
      ;; After resampling, weights should be uniform (all 0.0)
      (doseq [lw (pf/get-log-weights resampled)]
        (is (< (#?(:clj Math/abs :cljs js/Math.abs) lw) 1e-10)
            "weights should be reset to 0.0 after resampling")))))

(deftest maybe-resample-test
  (testing "maybe-resample respects threshold"
    (let [state (pf/initialize simple-model [0.0]
                               (choicemap/choicemap {:y 3.0})
                               10)]
      ;; With threshold 0, should never resample
      (let [result (pf/maybe-resample state 0.0)]
        (is (= (pf/get-log-weights result)
               (pf/get-log-weights state))
            "should not resample when threshold is 0"))
      ;; With very high threshold, should always resample
      (let [result (pf/maybe-resample state 100.0)]
        (doseq [lw (pf/get-log-weights result)]
          (is (< (#?(:clj Math/abs :cljs js/Math.abs) lw) 1e-10)
              "should have resampled with high threshold"))))))

;; --- Unfold integration for sequential tracking ---

(def tracking-kernel
  (gen [t state]
    (let [new-x (dynamic/trace! :x dist/normal state 0.5)]
      (dynamic/trace! :y dist/normal new-x 1.0)
      new-x)))

(def tracking-model (unfold-combinator/unfold-gen-fn tracking-kernel))

(deftest unfold-particle-filter-integration
  (testing "particle filter tracks a random walk via Unfold"
    (let [;; True states: 0, 1, 2, 3, 4, 5
          true-states [1.0 2.0 3.0 4.0 5.0]
          ;; Noisy observations
          observations (mapv #(+ % (* 0.5 (- (rand) 0.5))) true-states)
          n-particles 50

          ;; Initialize with first observation
          init-obs (choicemap/choicemap {0 {:y (nth observations 0)}})
          state (pf/initialize tracking-model [1 0.0] init-obs n-particles)]

      ;; Verify initialization
      (is (= n-particles (count (pf/get-traces state))))

      ;; Run several steps
      (let [final-state
            (reduce
             (fn [st t]
               (let [new-obs (choicemap/choicemap {t {:y (nth observations t)}})
                     stepped (pf/step st
                                      [(inc t) 0.0]
                                      [:no-change :no-change]
                                      new-obs)]
                 (pf/maybe-resample stepped (/ n-particles 2.0))))
             state
             (range 1 (count observations)))]

        ;; Should still have correct number of particles
        (is (= n-particles (count (pf/get-traces final-state))))

        ;; Log-ML estimate should be finite
        (let [log-ml (pf/log-ml-estimate final-state)]
          (is (not (#?(:clj Double/isNaN :cljs js/isNaN) log-ml))
              "log-ML estimate should not be NaN")
          (is (not (#?(:clj Double/isInfinite :cljs #(or (= % js/Infinity) (= % js/-Infinity)))
                    log-ml))
              "log-ML estimate should be finite"))))))

(deftest log-ml-estimate-test
  (testing "log-ML estimate is reasonable for simple model"
    (let [state (pf/initialize simple-model [0.0]
                               (choicemap/choicemap {:y 0.0})
                               100)
          log-ml (pf/log-ml-estimate state)]
      (is (number? log-ml))
      (is (not (#?(:clj Double/isNaN :cljs js/isNaN) log-ml))))))

(deftest non-unfold-model
  (testing "particle filter works with plain dynamic DSL model"
    (let [state (pf/initialize simple-model [0.0]
                               (choicemap/choicemap {:y 2.0})
                               5)]
      (is (= 5 (count (pf/get-traces state))))
      (is (number? (pf/log-ml-estimate state))))))
