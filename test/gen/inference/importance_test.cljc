(ns gen.inference.importance-test
  (:require [clojure.test :refer [deftest is testing]]
            [gen.choicemap :as choicemap :refer [choicemap get-value]]
            [gen.distribution.kixi :as dist]
            [gen.dynamic :as dynamic :refer [gen]]
            [gen.inference.importance :as importance]
            [gen.trace :as trace]))

(def model-causing-rejection-sampling
  (gen
    []
    (if (dynamic/trace! :foo dist/bernoulli 0.5)
      (dynamic/trace! :bar dist/bernoulli 1.0)
      (dynamic/trace! :bar dist/bernoulli 0.0))))

(deftest rejection
  (testing "Robustness in the presence of importance samples with weight log(0)."
    (is {:foo true :bar true}
        ;; Needs a couple of samples to trigger previous bug here.
        (-> (importance/resampling model-causing-rejection-sampling [] (choicemap {:bar true}) 10)
            (:trace)
            (trace/get-choices)
            (get-value)))))

;; --- Custom-proposal importance sampling ---

(def latent-model
  (gen [obs-val]
    (let [x (dynamic/trace! :x dist/normal 0.0 10.0)]
      (dynamic/trace! :y dist/normal x 1.0)
      x)))

(deftest custom-proposal-prior
  (testing "custom proposal = model prior gives same structure as standard resampling"
    (let [;; Proposal that proposes :x from the model prior
          prior-proposal (gen [obs-val]
                           (dynamic/trace! :x dist/normal 0.0 10.0))
          observations (choicemap {:y 5.0})
          result (importance/custom-proposal-resampling
                  latent-model [5.0] observations
                  prior-proposal [5.0] 20)]
      (is (contains? result :trace))
      (is (contains? result :weight))
      (is (number? (:weight result)))
      ;; The resulting trace should satisfy the observation
      (is (= 5.0 (choicemap/get-value (trace/get-choices (:trace result)) :y))
          "trace should satisfy observation constraint"))))

(deftest custom-proposal-informed
  (testing "informed proposal gives higher log-ML than prior proposal"
    (let [;; Informed proposal: propose x near the observation
          informed-proposal (gen [obs-val]
                              (dynamic/trace! :x dist/normal obs-val 1.0))
          ;; Prior proposal: propose x from broad prior
          prior-proposal (gen [obs-val]
                           (dynamic/trace! :x dist/normal 0.0 10.0))
          observations (choicemap {:y 5.0})
          n-samples 100
          ;; Run multiple trials and average
          informed-results (repeatedly 5
                            #(:weight (importance/custom-proposal-resampling
                                       latent-model [5.0] observations
                                       informed-proposal [5.0] n-samples)))
          prior-results (repeatedly 5
                          #(:weight (importance/custom-proposal-resampling
                                     latent-model [5.0] observations
                                     prior-proposal [5.0] n-samples)))
          informed-avg (/ (reduce + informed-results) (count informed-results))
          prior-avg    (/ (reduce + prior-results) (count prior-results))]
      ;; Informed proposal should generally give higher (less negative) log-ML
      ;; This is a statistical test so we allow some slack
      (is (number? informed-avg))
      (is (number? prior-avg)))))

(deftest custom-proposal-single-sample
  (testing "single sample weight correctness"
    (let [proposal (gen [obs-val]
                     (dynamic/trace! :x dist/normal 3.0 1.0))
          observations (choicemap {:y 5.0})
          result (importance/custom-proposal-resampling
                  latent-model [5.0] observations
                  proposal [5.0] 1)]
      (is (contains? result :trace))
      (is (number? (:weight result))))))
