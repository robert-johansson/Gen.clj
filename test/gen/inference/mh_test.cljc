(ns gen.inference.mh-test
  (:require [clojure.test :refer [deftest is testing]]
            [gen.choicemap :as choicemap]
            [gen.distribution.kixi :as dist]
            [gen.dynamic :as dynamic :refer [gen]]
            [gen.generative-function :as gf]
            [gen.inference.mh :as mh]
            [gen.trace :as trace]))

(defn- close?
  ([expected actual] (close? expected actual 0.5))
  ([expected actual tol]
   (< (abs (- (double expected) (double actual))) tol)))

(defn- mean [xs]
  (/ (reduce + xs) (count xs)))

;; Simple model: x ~ Normal(3, 0.5)
(def simple-model
  (gen []
    (dynamic/trace! :x dist/normal 3.0 0.5)))

(deftest mh-basic
  (testing "mh returns trace and accepted? flag"
    (let [tr     (gf/simulate simple-model [])
          result (mh/mh tr #{:x})]
      (is (contains? result :trace))
      (is (contains? result :accepted?))
      (is (boolean? (:accepted? result)))
      (is (satisfies? trace/ITrace (:trace result))))))

(deftest mh-step-composition
  (testing "mh-step returns a kernel function"
    (let [tr   (gf/simulate simple-model [])
          step (mh/mh-step #{:x})
          new  (step tr)]
      (is (satisfies? trace/ITrace new)))))

(deftest chain-test
  (testing "chain produces lazy infinite sequence"
    (let [tr    (gf/simulate simple-model [])
          step  (mh/mh-step #{:x})
          ch    (mh/chain tr step)
          first-10 (take 10 ch)]
      (is (= 10 (count first-10)))
      (is (every? #(satisfies? trace/ITrace %) first-10)))))

(deftest cycle-kernels-test
  (testing "cycle-kernels composes multiple kernels"
    (let [model (gen []
                  (dynamic/trace! :a dist/normal 0 1)
                  (dynamic/trace! :b dist/normal 0 1))
          tr    (gf/simulate model [])
          sweep (mh/cycle-kernels
                 (mh/mh-step #{:a})
                 (mh/mh-step #{:b}))
          new   (sweep tr)]
      (is (satisfies? trace/ITrace new)))))

(deftest mh-convergence
  (testing "MH chain converges to correct posterior mean"
    (let [;; Model: x ~ Normal(3, 0.5)
          ;; After many MH steps, mean of x should be close to 3.0
          tr     (gf/simulate simple-model [])
          step   (mh/mh-step #{:x})
          traces (take 2000 (mh/chain tr step))
          ;; Drop burn-in
          values (mapv #(choicemap/get-value (trace/get-choices %) :x)
                       (drop 500 traces))
          m      (mean values)]
      (is (close? 3.0 m 0.5)
          (str "mean should be ~3.0, got " m)))))
