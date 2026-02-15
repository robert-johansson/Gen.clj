(ns gen.combinator.unfold-test
  (:require [clojure.test :refer [deftest is testing]]
            [gen.choicemap :as choicemap]
            [gen.combinator.unfold :as unfold-combinator]
            [gen.distribution.kixi :as dist]
            [gen.dynamic :as dynamic :refer [gen]]
            [gen.generative-function :as gf]
            [gen.selection :as selection]
            [gen.trace :as trace]))

(defn- get-nested-value
  "Get a value at a nested path in a choicemap: cm[k1][k2]"
  [cm k1 k2]
  (-> cm (choicemap/get-submap k1) (choicemap/get-value k2)))

;; Random walk kernel: new_state = Normal(old_state, 1.0)
(def random-walk-kernel
  (gen [t state]
    (dynamic/trace! :x dist/normal state 1.0)))

(def unfold-rw (unfold-combinator/unfold-gen-fn random-walk-kernel))

(deftest simulate-basic
  (testing "simulate: random walk for 5 steps"
    (let [tr     (gf/simulate unfold-rw [5 0.0])
          states (trace/get-retval tr)
          choices (trace/get-choices tr)]
      (is (= 6 (count states))
          "states should have n+1 entries (init + 5 steps)")
      (is (= 0.0 (first states))
          "first state should be the initial state")
      (is (= 5 (count (choicemap/get-submaps-shallow choices)))
          "choices should have 5 sub-maps")
      (is (number? (trace/get-score tr))
          "score should be a number"))))

(deftest generate-with-constraint
  (testing "generate with constraint on step 2"
    (let [constraints (choicemap/choicemap {2 {:x 42.0}})
          {:keys [trace weight]} (gf/generate unfold-rw [5 0.0] constraints)]
      (is (= 42.0 (get-nested-value (trace/get-choices trace) 2 :x))
          "step 2's :x should be constrained to 42.0")
      (is (number? weight)
          "weight should be a number"))))

(deftest extend-by-one-step
  (testing "extend by one step (particle filter critical path)"
    (let [;; Start with 3 steps
          {:keys [trace]} (gf/generate unfold-rw [3 0.0]
                                       (choicemap/choicemap {0 {:x 1.0}
                                                             1 {:x 2.0}
                                                             2 {:x 3.0}}))
          ;; Extend to 4 steps, constraining step 3
          {:keys [trace weight]}
          (trace/update trace
                        [4 0.0]
                        [:no-change :no-change]
                        (choicemap/choicemap {3 {:x 4.0}}))
          states (trace/get-retval trace)]
      (is (= 5 (count states))
          "states should have 5 entries after extending to 4 steps")
      ;; Steps 0-2 should be unchanged
      (is (= 1.0 (get-nested-value (trace/get-choices trace) 0 :x))
          "step 0 should be unchanged")
      (is (= 2.0 (get-nested-value (trace/get-choices trace) 1 :x))
          "step 1 should be unchanged")
      (is (= 3.0 (get-nested-value (trace/get-choices trace) 2 :x))
          "step 2 should be unchanged")
      ;; Step 3 should be constrained
      (is (= 4.0 (get-nested-value (trace/get-choices trace) 3 :x))
          "step 3 should be constrained to 4.0")
      (is (number? weight)
          "weight should be a number"))))

(deftest shrink-steps
  (testing "shrink: n=5 to n=3"
    (let [tr (gf/simulate unfold-rw [5 0.0])
          {:keys [trace]}
          (trace/update tr [3 0.0]
                        [:no-change :no-change]
                        (choicemap/choicemap))
          states (trace/get-retval trace)]
      (is (= 4 (count states))
          "states should have 4 entries (init + 3 steps)")
      (is (= 3 (count (choicemap/get-submaps-shallow (trace/get-choices trace))))
          "choices should have 3 sub-maps"))))

(deftest state-threading
  (testing "state threading correctness"
    (let [;; Deterministic kernel: new_state = state + 1
          det-kernel (gen [t state]
                       (let [new-state (+ state 1.0)]
                         (dynamic/trace! :x dist/normal new-state 0.001)
                         new-state))
          unfold-det (unfold-combinator/unfold-gen-fn det-kernel)
          tr         (gf/simulate unfold-det [5 0.0])
          states     (trace/get-retval tr)]
      ;; States should be approximately [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
      (is (= 6 (count states)))
      (is (= 0.0 (nth states 0)))
      ;; Each state should be approximately n (deterministic with tiny noise)
      (doseq [i (range 1 6)]
        (is (< (#?(:clj Math/abs :cljs js/Math.abs)
                (- (double i) (double (nth states i)))) 0.1)
            (str "state " i " should be approximately " i))))))

(deftest project-test
  (testing "project on step 2"
    (let [{:keys [trace]} (gf/generate unfold-rw [5 0.0]
                                       (choicemap/choicemap {0 {:x 1.0}
                                                             1 {:x 2.0}
                                                             2 {:x 3.0}
                                                             3 {:x 4.0}
                                                             4 {:x 5.0}}))
          sel (selection/->HierarchicalSelection {2 selection/ALL})
          projected (trace/project trace sel)]
      (is (number? projected)
          "project should return a number")
      (let [s2 (trace/get-score (nth (:sub-traces trace) 2))]
        (is (< (#?(:clj Math/abs :cljs js/Math.abs) (- projected s2)) 1e-10)
            "projected score should equal step 2's score")))))

(deftest zero-steps
  (testing "zero steps"
    (let [tr (gf/simulate unfold-rw [0 5.0])]
      (is (= [5.0] (trace/get-retval tr))
          "should return just the initial state")
      (is (= 0.0 (trace/get-score tr))
          "score should be 0 with no steps")
      (is (choicemap/empty? (trace/get-choices tr))
          "no choices with zero steps"))))
