(ns gen.combinator.switch-test
  (:require [clojure.test :refer [deftest is testing]]
            [gen.choicemap :as choicemap]
            [gen.combinator.switch :as switch-combinator]
            [gen.distribution.kixi :as dist]
            [gen.dynamic :as dynamic :refer [gen]]
            [gen.generative-function :as gf]
            [gen.selection :as selection]
            [gen.trace :as trace]))

(defn- get-nested-value
  "Get a value at a nested path in a choicemap: cm[k1][k2]"
  [cm k1 k2]
  (-> cm (choicemap/get-submap k1) (choicemap/get-value k2)))

(def branch-a
  (gen [x]
    (dynamic/trace! :a-val dist/normal x 1.0)))

(def branch-b
  (gen [x]
    (dynamic/trace! :b-val dist/normal x 0.5)))

(def switched (switch-combinator/switch-gen-fn [branch-a branch-b]))

(deftest simulate-each-branch
  (testing "simulate branch 0"
    (let [tr (gf/simulate switched [0 3.0])
          choices (trace/get-choices tr)]
      (is (choicemap/has-submap? choices 0)
          "choices should be namespaced under branch index 0")
      (is (number? (get-nested-value choices 0 :a-val))
          "should have :a-val under branch 0")))

  (testing "simulate branch 1"
    (let [tr (gf/simulate switched [1 3.0])
          choices (trace/get-choices tr)]
      (is (choicemap/has-submap? choices 1)
          "choices should be namespaced under branch index 1")
      (is (number? (get-nested-value choices 1 :b-val))
          "should have :b-val under branch 1"))))

(deftest generate-with-constraints
  (testing "generate with branch-specific constraints"
    (let [constraints (choicemap/choicemap {0 {:a-val 42.0}})
          {:keys [trace weight]} (gf/generate switched [0 0.0] constraints)]
      (is (= 42.0 (get-nested-value (trace/get-choices trace) 0 :a-val))
          "branch 0 value should be constrained")
      (is (number? weight)))))

(deftest update-same-branch
  (testing "update within the same branch"
    (let [{:keys [trace]} (gf/generate switched [0 0.0]
                                       (choicemap/choicemap {0 {:a-val 1.0}}))
          {:keys [trace]}
          (trace/update trace (choicemap/choicemap {0 {:a-val 5.0}}))]
      (is (= 5.0 (get-nested-value (trace/get-choices trace) 0 :a-val))
          "value should be updated to 5.0"))))

(deftest update-branch-switch
  (testing "update with branch switch"
    (let [{:keys [trace]} (gf/generate switched [0 0.0]
                                       (choicemap/choicemap {0 {:a-val 1.0}}))
          ;; Switch from branch 0 to branch 1
          {:keys [trace discard]}
          (trace/update trace [1 0.0]
                        [:no-change :no-change]
                        (choicemap/choicemap {1 {:b-val 10.0}}))]
      ;; New trace should have branch 1 choices
      (is (choicemap/has-submap? (trace/get-choices trace) 1)
          "new choices should be under branch 1")
      (is (= 10.0 (get-nested-value (trace/get-choices trace) 1 :b-val))
          "branch 1 value should be constrained")
      ;; Discard should contain old branch 0 choices
      (is (choicemap/has-submap? discard 0)
          "discard should contain old branch 0 choices"))))

(deftest project-test
  (testing "project on branch-specific addresses"
    (let [{:keys [trace]} (gf/generate switched [0 0.0]
                                       (choicemap/choicemap {0 {:a-val 1.0}}))
          ;; Select everything under branch 0
          sel (selection/->HierarchicalSelection {0 selection/ALL})
          projected (trace/project trace sel)]
      (is (number? projected))
      (is (= (trace/get-score trace) projected)
          "projecting all of a single-branch trace should equal its score"))))

(deftest mixture-model-integration
  (testing "mixture model: branch selection via Bernoulli"
    (let [mixture (gen [x]
                    (let [branch (if (dynamic/trace! :branch dist/bernoulli 0.5) 0 1)]
                      (dynamic/trace! :obs switched branch x)))
          {:keys [trace]} (gf/generate mixture [3.0]
                                       (choicemap/choicemap {:branch true
                                                             :obs {0 {:a-val 3.5}}}))]
      (is (satisfies? trace/ITrace trace)
          "should produce a valid trace"))))
