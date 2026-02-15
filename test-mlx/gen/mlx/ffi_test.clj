(ns gen.mlx.ffi-test
  "Smoke tests for raw FFI bindings to mlx-c."
  (:require [clojure.test :refer [deftest is testing]]
            [gen.mlx.ffi :as ffi]))

(deftest default-stream-test
  (testing "default CPU stream is valid (non-null ctx)"
    (is (some? (:ctx ffi/default-stream)))))

(deftest scalar-array-roundtrip-test
  (testing "create float32 scalar and read it back"
    (let [arr (ffi/array-new-float (float 42.0))]
      (is (some? (:ctx arr)))
      (ffi/array-eval arr)
      (is (== 42.0 (ffi/array-item-float32 arr)))
      (is (== 0 (ffi/array-ndim arr)))
      (is (= [] (ffi/array-shape arr))))))

(deftest array-from-data-test
  (testing "create 1-d array from float buffer"
    (let [data (float-array [1.0 2.0 3.0])
          arr (ffi/array-new-data data [3])]
      (is (some? (:ctx arr)))
      (ffi/array-eval arr)
      (is (== 1 (ffi/array-ndim arr)))
      (is (= [3] (ffi/array-shape arr)))
      (is (== 3 (ffi/array-size arr)))
      (is (= [1.0 2.0 3.0]
             (mapv double (ffi/array-data-float32 arr)))))))

(deftest add-two-scalars-test
  (testing "add two scalar arrays via mlx_add"
    (let [a (ffi/array-new-float (float 2.0))
          b (ffi/array-new-float (float 3.0))
          c (ffi/mlx-add a b)]
      (ffi/array-eval c)
      (is (== 5.0 (ffi/array-item-float32 c))))))

(deftest vector-array-test
  (testing "vector<array> operations"
    (let [a (ffi/array-new-float (float 10.0))
          b (ffi/array-new-float (float 20.0))
          va (ffi/arrays->vector [a b])]
      (is (== 2 (ffi/vector-array-size va)))
      (let [first-elem (ffi/vector-array-get va 0)]
        (ffi/array-eval first-elem)
        (is (== 10.0 (ffi/array-item-float32 first-elem))))
      (let [second-elem (ffi/vector-array-get va 1)]
        (ffi/array-eval second-elem)
        (is (== 20.0 (ffi/array-item-float32 second-elem)))))))
