(ns gen.selection-test
  (:require [clojure.test :refer [deftest is testing]]
            [gen.selection :as sel]))

(deftest all-selection-tests
  (testing "ALL includes everything"
    (is (sel/includes? sel/ALL :x))
    (is (sel/includes? sel/ALL :any-address))
    (is (sel/includes? sel/ALL 42)))

  (testing "ALL subselection is ALL"
    (is (= sel/ALL (sel/get-subselection sel/ALL :x)))
    (is (= sel/ALL (sel/get-subselection sel/ALL :nested)))))

(deftest empty-selection-tests
  (testing "NONE includes nothing"
    (is (not (sel/includes? sel/NONE :x)))
    (is (not (sel/includes? sel/NONE :any-address))))

  (testing "NONE subselection is NONE"
    (is (= sel/NONE (sel/get-subselection sel/NONE :x)))))

(deftest set-selection-tests
  (testing "SetSelection includes elements in the set"
    (let [s (sel/select :slope :noise)]
      (is (sel/includes? s :slope))
      (is (sel/includes? s :noise))
      (is (not (sel/includes? s :other)))))

  (testing "SetSelection subselection returns ALL/NONE"
    (let [s (sel/select :slope :noise)]
      (is (= sel/ALL (sel/get-subselection s :slope)))
      (is (= sel/NONE (sel/get-subselection s :other))))))

(deftest clojure-set-extension-tests
  (testing "Clojure sets work as selections"
    (is (sel/includes? #{:slope :noise} :slope))
    (is (not (sel/includes? #{:slope :noise} :other)))
    (is (= sel/ALL (sel/get-subselection #{:slope} :slope)))
    (is (= sel/NONE (sel/get-subselection #{:slope} :other)))))

(deftest hierarchical-selection-tests
  (testing "nested address paths"
    (let [s (sel/select [:data 1 :y])]
      (is (sel/includes? s :data))
      (is (not (sel/includes? s :slope)))
      (let [sub (sel/get-subselection s :data)]
        (is (sel/includes? sub 1))
        (is (not (sel/includes? sub 0)))
        (let [sub2 (sel/get-subselection sub 1)]
          (is (sel/includes? sub2 :y))
          (is (= sel/ALL (sel/get-subselection sub2 :y)))))))

  (testing "mixed flat and nested"
    (let [s (sel/select :slope [:data 1 :y])]
      (is (sel/includes? s :slope))
      (is (sel/includes? s :data))
      (is (not (sel/includes? s :other)))
      (is (= sel/ALL (sel/get-subselection s :slope))))))

(deftest complement-selection-tests
  (testing "complement inverts includes?"
    (let [s (sel/complement #{:slope})]
      (is (not (sel/includes? s :slope)))
      (is (sel/includes? s :noise))
      (is (sel/includes? s :anything)))))

(deftest union-selection-tests
  (testing "union includes either"
    (let [s (sel/union #{:a} #{:b})]
      (is (sel/includes? s :a))
      (is (sel/includes? s :b))
      (is (not (sel/includes? s :c))))))

(deftest intersection-selection-tests
  (testing "intersection includes both"
    (let [s (sel/intersection #{:a :b} #{:b :c})]
      (is (not (sel/includes? s :a)))
      (is (sel/includes? s :b))
      (is (not (sel/includes? s :c))))))

(deftest selection-predicate-tests
  (testing "selection? works"
    (is (sel/selection? sel/ALL))
    (is (sel/selection? sel/NONE))
    (is (sel/selection? #{:a}))
    (is (sel/selection? (sel/select :a :b)))
    (is (not (sel/selection? 42)))
    (is (not (sel/selection? "hello")))))
