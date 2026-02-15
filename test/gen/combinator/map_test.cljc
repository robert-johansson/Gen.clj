(ns gen.combinator.map-test
  (:require [clojure.test :refer [deftest is testing]]
            [gen.choicemap :as choicemap]
            [gen.combinator.map :as map-combinator]
            [gen.distribution.kixi :as dist]
            [gen.dynamic :as dynamic :refer [gen]]
            [gen.generative-function :as gf]
            [gen.inference.mh :as mh]
            [gen.selection :as selection]
            [gen.trace :as trace]))

(defn- get-nested-value
  "Get a value at a nested path in a choicemap: cm[k1][k2]"
  [cm k1 k2]
  (-> cm (choicemap/get-submap k1) (choicemap/get-value k2)))

(def kernel
  (gen [x]
    (dynamic/trace! :z dist/normal x 1.0)))

(def mapped (map-combinator/map-gen-fn kernel))

(deftest simulate-basic
  (testing "simulate maps kernel over 5 elements"
    (let [args  [[1.0 2.0 3.0 4.0 5.0]]
          tr    (gf/simulate mapped args)
          choices (trace/get-choices tr)]
      (is (= 5 (count (trace/get-retval tr)))
          "retval should be a 5-element vector")
      (is (= 5 (count (choicemap/get-submaps-shallow choices)))
          "choices should have 5 sub-maps (indices 0-4)")
      (is (every? #(choicemap/has-submap? choices %) (range 5))
          "each index should have a submap")
      (is (number? (trace/get-score tr))
          "score should be a number")
      (is (= mapped (trace/get-gen-fn tr))
          "gen-fn round-trips through trace"))))

(deftest generate-with-constraints
  (testing "generate with constraints on indices 0 and 2"
    (let [args        [[0.0 0.0 0.0]]
          constraints (choicemap/choicemap {0 {:z 42.0}
                                            2 {:z 99.0}})
          {:keys [trace weight]} (gf/generate mapped args constraints)]
      (is (= 42.0 (get-nested-value (trace/get-choices trace) 0 :z))
          "index 0 should be constrained to 42.0")
      (is (= 99.0 (get-nested-value (trace/get-choices trace) 2 :z))
          "index 2 should be constrained to 99.0")
      (is (not= 0.0 weight)
          "weight should be non-zero for constrained choices"))))

(deftest update-constraint
  (testing "update: change constraint on index 1"
    (let [args [[0.0 0.0 0.0]]
          {:keys [trace]} (gf/generate mapped args
                                       (choicemap/choicemap {0 {:z 1.0}
                                                             1 {:z 2.0}
                                                             2 {:z 3.0}}))
          {:keys [trace weight]}
          (trace/update trace (choicemap/choicemap {1 {:z 20.0}}))]
      (is (= 20.0 (get-nested-value (trace/get-choices trace) 1 :z))
          "index 1 should be updated to 20.0")
      (is (= 1.0 (get-nested-value (trace/get-choices trace) 0 :z))
          "index 0 should be unchanged")
      (is (number? weight)
          "weight should be returned"))))

(deftest length-change-extend
  (testing "length change: simulate n=3, update to n=5"
    (let [args3 [[1.0 2.0 3.0]]
          tr3   (gf/simulate mapped args3)
          ;; Update to n=5 by providing longer arg vectors
          new-args [[1.0 2.0 3.0 4.0 5.0]]
          {:keys [trace]} (trace/update tr3 new-args
                                        (repeat (count new-args) :no-change)
                                        (choicemap/choicemap))]
      (is (= 5 (count (trace/get-retval trace)))
          "retval should have 5 elements after extension")
      (is (= 5 (count (choicemap/get-submaps-shallow (trace/get-choices trace))))
          "choices should have 5 sub-maps"))))

(deftest project-test
  (testing "project with hierarchical selection on indices 0 and 2"
    (let [args [[0.0 0.0 0.0]]
          {:keys [trace]} (gf/generate mapped args
                                       (choicemap/choicemap {0 {:z 1.0}
                                                             1 {:z 2.0}
                                                             2 {:z 3.0}}))
          sel (selection/->HierarchicalSelection
               {0 selection/ALL
                2 selection/ALL})
          projected (trace/project trace sel)]
      (is (number? projected)
          "project should return a number")
      ;; projected score should be sum of scores for indices 0 and 2
      (let [s0 (trace/get-score (nth (:sub-traces trace) 0))
            s2 (trace/get-score (nth (:sub-traces trace) 2))]
        (is (< (#?(:clj Math/abs :cljs js/Math.abs) (- projected (+ s0 s2))) 1e-10)
            "projected score should equal sum of selected sub-trace scores")))))

(deftest regenerate-test
  (testing "regenerate with selection on index 1"
    (let [args [[0.0 0.0 0.0]]
          {:keys [trace]} (gf/generate mapped args
                                       (choicemap/choicemap {0 {:z 1.0}
                                                             1 {:z 2.0}
                                                             2 {:z 3.0}}))
          sel (selection/->HierarchicalSelection {1 selection/ALL})
          {:keys [trace]} (gf/regenerate trace sel)]
      ;; Index 0 and 2 should keep their values
      (is (= 1.0 (get-nested-value (trace/get-choices trace) 0 :z))
          "index 0 should be unchanged")
      (is (= 3.0 (get-nested-value (trace/get-choices trace) 2 :z))
          "index 2 should be unchanged")
      ;; Index 1 should be re-sampled (probably different from 2.0)
      (is (number? (get-nested-value (trace/get-choices trace) 1 :z))
          "index 1 should have a numeric value"))))

(deftest empty-map
  (testing "empty map (n=0)"
    (let [tr (gf/simulate mapped [[]])]
      (is (= 0.0 (trace/get-score tr))
          "score of empty map should be 0")
      (is (= [] (trace/get-retval tr))
          "retval should be empty vector")
      (is (choicemap/empty? (trace/get-choices tr))
          "choices should be empty"))))

(deftest mh-integration
  (testing "MH works with mapped traces"
    (let [args  [[0.0 0.0 0.0]]
          {:keys [trace]} (gf/generate mapped args
                                       (choicemap/choicemap {0 {:z 1.0}
                                                             1 {:z 2.0}
                                                             2 {:z 3.0}}))
          sel (selection/->HierarchicalSelection {1 selection/ALL})
          result (mh/mh trace sel)]
      (is (contains? result :trace))
      (is (contains? result :accepted?)))))
