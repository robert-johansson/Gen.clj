(ns gen.mlx.array-test
  "Tests for Clojure-idiomatic MLX array operations."
  (:require [clojure.test :refer [deftest is testing]]
            [gen.mlx.array :as arr]))

;; ---------------------------------------------------------------------------
;; Basic construction & extraction
;; ---------------------------------------------------------------------------

(deftest scalar-test
  (testing "scalar creation and extraction"
    (is (== 42.0 (arr/->double (arr/scalar 42.0))))
    (is (== 0.0 (arr/->double (arr/scalar 0.0))))
    (is (== -1.5 (arr/->double (arr/scalar -1.5))))))

(deftest from-vec-roundtrip-test
  (testing "from-vec creates 1-d array, ->vec extracts it"
    (is (= [1.0 2.0 3.0] (arr/->vec (arr/from-vec [1 2 3]))))
    (is (= [0.0] (arr/->vec (arr/from-vec [0]))))
    (is (= [] (arr/->vec (arr/from-vec []))))))

(deftest shape-test
  (testing "shape of scalar"
    (is (= [] (arr/shape (arr/scalar 1.0)))))
  (testing "shape of 1-d array"
    (is (= [3] (arr/shape (arr/from-vec [1 2 3])))))
  (testing "shape of 2-d array"
    (is (= [2 3] (arr/shape (arr/array [1 2 3 4 5 6] [2 3]))))))

;; ---------------------------------------------------------------------------
;; IDeref — @x forces eval and returns Clojure value
;; ---------------------------------------------------------------------------

(deftest deref-scalar-test
  (testing "@scalar returns a double"
    (is (== 42.0 @(arr/scalar 42.0)))
    (is (== -3.0 @(arr/scalar -3.0)))))

(deftest deref-vector-test
  (testing "@vector returns a Clojure vector of doubles"
    (is (= [1.0 2.0 3.0] @(arr/from-vec [1 2 3])))))

(deftest deref-lazy-test
  (testing "@lazy-result forces the whole computation graph"
    (let [result (arr/mul (arr/add (arr/scalar 2.0) (arr/scalar 3.0))
                          (arr/scalar 4.0))]
      (is (== 20.0 @result)))))

;; ---------------------------------------------------------------------------
;; Counted — (count x)
;; ---------------------------------------------------------------------------

(deftest count-test
  (testing "count returns total number of elements"
    (is (= 1 (count (arr/scalar 5.0))))
    (is (= 3 (count (arr/from-vec [1 2 3]))))
    (is (= 6 (count (arr/array [1 2 3 4 5 6] [2 3]))))
    (is (= 0 (count (arr/from-vec []))))))

;; ---------------------------------------------------------------------------
;; Seqable — (seq x)
;; ---------------------------------------------------------------------------

(deftest seq-test
  (testing "seq returns elements as Clojure seq"
    (is (= '(1.0 2.0 3.0) (seq (arr/from-vec [1 2 3]))))
    (is (nil? (seq (arr/from-vec [])))))
  (testing "seq works with for/doseq"
    (is (= [2.0 4.0 6.0]
           (vec (for [x (arr/from-vec [1 2 3])] (* x 2))))))
  (testing "seq on lazy computation forces eval"
    (let [result (arr/add (arr/from-vec [10 20 30])
                          (arr/from-vec [1 2 3]))]
      (is (= '(11.0 22.0 33.0) (seq result))))))

;; ---------------------------------------------------------------------------
;; Indexed — (nth x i)
;; ---------------------------------------------------------------------------

(deftest nth-test
  (testing "nth on 1-d array returns element as double"
    (let [a (arr/from-vec [10 20 30])]
      (is (== 10.0 (nth a 0)))
      (is (== 20.0 (nth a 1)))
      (is (== 30.0 (nth a 2)))))
  (testing "nth on scalar"
    (is (== 5.0 (nth (arr/scalar 5.0) 0))))
  (testing "nth on 2-d array (flat index)"
    (let [a (arr/array [1 2 3 4 5 6] [2 3])]
      (is (== 1.0 (nth a 0)))
      (is (== 4.0 (nth a 3)))
      (is (== 6.0 (nth a 5)))))
  (testing "nth out of bounds throws"
    (is (thrown? IndexOutOfBoundsException
                (nth (arr/from-vec [1 2 3]) 5))))
  (testing "nth with not-found returns default for out of bounds"
    (is (= :nope (nth (arr/from-vec [1 2 3]) 99 :nope))))
  (testing "nth on lazy computation forces eval"
    (let [result (arr/add (arr/from-vec [10 20 30])
                          (arr/from-vec [1 2 3]))]
      (is (== 11.0 (nth result 0)))
      (is (== 33.0 (nth result 2))))))

;; ---------------------------------------------------------------------------
;; IReduce — (reduce f x)
;; ---------------------------------------------------------------------------

(deftest reduce-test
  (testing "reduce without init"
    (is (== 6.0 (reduce + (arr/from-vec [1 2 3]))))
    (is (== 5.0 (reduce + (arr/from-vec [5])))))
  (testing "reduce with init"
    (is (== 16.0 (reduce + 10.0 (arr/from-vec [1 2 3]))))
    (is (== 10.0 (reduce + 10.0 (arr/from-vec [])))))
  (testing "reduce with early termination"
    (is (== 3.0 (reduce (fn [acc x]
                          (if (> acc 2.0)
                            (reduced acc)
                            (+ acc x)))
                        (arr/from-vec [1 2 3 4 5])))))
  (testing "into works via reduce"
    (is (= [1.0 2.0 3.0] (into [] (arr/from-vec [1 2 3])))))
  (testing "reduce on lazy computation"
    (let [result (arr/add (arr/from-vec [10 20 30])
                          (arr/from-vec [1 2 3]))]
      (is (== 66.0 (reduce + result))))))

;; ---------------------------------------------------------------------------
;; IMeta / IObj — metadata support
;; ---------------------------------------------------------------------------

(deftest meta-test
  (testing "new arrays have nil metadata"
    (is (nil? (meta (arr/scalar 3.0)))))
  (testing "with-meta attaches metadata"
    (let [a (arr/scalar 3.0)
          b (with-meta a {:label "weight"})]
      (is (= {:label "weight"} (meta b)))
      (is (nil? (meta a)))))
  (testing "metadata does not affect value"
    (let [a (arr/scalar 3.0)
          b (with-meta a {:label "weight"})]
      (is (== 3.0 @b))
      (is (= [3] (:shape (arr/from-vec [1 2 3]))))))
  (testing "metadata does not affect equality"
    (let [a (arr/scalar 3.0)
          b (with-meta (arr/scalar 3.0) {:label "x"})]
      (is (= a b))))
  (testing "metadata does not affect hashCode"
    (let [a (arr/scalar 3.0)
          b (with-meta (arr/scalar 3.0) {:label "x"})]
      (is (= (.hashCode a) (.hashCode b)))))
  (testing "vary-meta works"
    (let [a (with-meta (arr/scalar 1.0) {:n 1})
          b (vary-meta a assoc :m 2)]
      (is (= {:n 1 :m 2} (meta b))))))

;; ---------------------------------------------------------------------------
;; ILookup — keyword access for metadata
;; ---------------------------------------------------------------------------

(deftest lookup-test
  (testing "(:shape x) returns shape vector"
    (is (= [3] (:shape (arr/from-vec [1 2 3]))))
    (is (= [2 3] (:shape (arr/array [1 2 3 4 5 6] [2 3]))))
    (is (= [] (:shape (arr/scalar 1.0)))))
  (testing "(:dtype x) returns dtype keyword"
    (is (= :float32 (:dtype (arr/scalar 1.0)))))
  (testing "(:ndim x) returns number of dimensions"
    (is (= 0 (:ndim (arr/scalar 1.0))))
    (is (= 1 (:ndim (arr/from-vec [1 2 3]))))
    (is (= 2 (:ndim (arr/array [1 2 3 4 5 6] [2 3])))))
  (testing "(:size x) returns total element count"
    (is (= 6 (:size (arr/array [1 2 3 4 5 6] [2 3])))))
  (testing "missing key returns nil"
    (is (nil? (:missing (arr/scalar 1.0)))))
  (testing "missing key with default"
    (is (= :default (get (arr/scalar 1.0) :missing :default)))))

;; ---------------------------------------------------------------------------
;; IFn — (x :shape) works like (:shape x)
;; ---------------------------------------------------------------------------

(deftest ifn-test
  (testing "array as function for metadata access"
    (let [a (arr/from-vec [1 2 3])]
      (is (= [3] (a :shape)))
      (is (= :float32 (a :dtype)))
      (is (= :default (a :missing :default))))))

;; ---------------------------------------------------------------------------
;; Arithmetic ops
;; ---------------------------------------------------------------------------

(deftest addition-test
  (testing "scalar addition"
    (is (== 5.0 @(arr/add (arr/scalar 2.0) (arr/scalar 3.0)))))
  (testing "vector addition"
    (is (= [3.0 5.0 7.0]
           @(arr/add (arr/from-vec [1 2 3])
                     (arr/from-vec [2 3 4]))))))

(deftest subtraction-test
  (testing "scalar subtraction"
    (is (== 1.0 @(arr/sub (arr/scalar 3.0) (arr/scalar 2.0))))))

(deftest multiplication-test
  (testing "scalar multiplication"
    (is (== 20.0 @(arr/mul (arr/scalar 4.0) (arr/scalar 5.0))))))

(deftest division-test
  (testing "scalar division"
    (is (== 2.5 @(arr/div (arr/scalar 5.0) (arr/scalar 2.0))))))

(deftest exp-test
  (testing "exp(0) = 1"
    (is (< (Math/abs (- 1.0 @(arr/exp (arr/scalar 0.0)))) 1e-6)))
  (testing "exp(1) = e"
    (is (< (Math/abs (- Math/E @(arr/exp (arr/scalar 1.0)))) 1e-5))))

(deftest log-test
  (testing "log(1) = 0"
    (is (< (Math/abs (double @(arr/log (arr/scalar 1.0)))) 1e-6)))
  (testing "log(e) = 1"
    (is (< (Math/abs (- 1.0 @(arr/log (arr/scalar (float Math/E))))) 1e-5))))

(deftest square-test
  (testing "square(3) = 9"
    (is (== 9.0 @(arr/square (arr/scalar 3.0))))))

(deftest sqrt-test
  (testing "sqrt(4) = 2"
    (is (== 2.0 @(arr/sqrt (arr/scalar 4.0))))))

(deftest neg-test
  (testing "neg(5) = -5"
    (is (== -5.0 @(arr/neg (arr/scalar 5.0))))))

(deftest sum-test
  (testing "sum of vector"
    (is (== 6.0 @(arr/sum (arr/from-vec [1 2 3]))))))

;; ---------------------------------------------------------------------------
;; Auto-coercion — plain numbers accepted in arithmetic
;; ---------------------------------------------------------------------------

(deftest auto-coerce-test
  (testing "add with number on right"
    (is (== 5.0 @(arr/add (arr/scalar 2.0) 3.0))))
  (testing "add with number on left"
    (is (== 5.0 @(arr/add 2.0 (arr/scalar 3.0)))))
  (testing "add two numbers"
    (is (== 5.0 @(arr/add 2 3))))
  (testing "mul with number"
    (is (== 6.0 @(arr/mul (arr/scalar 3.0) 2))))
  (testing "sub with number"
    (is (== 1.0 @(arr/sub (arr/scalar 3.0) 2))))
  (testing "div with number"
    (is (== 2.0 @(arr/div (arr/scalar 6.0) 3))))
  (testing "unary ops with numbers"
    (is (== -3.0 @(arr/neg 3)))
    (is (== 9.0 @(arr/square 3)))
    (is (== 2.0 @(arr/sqrt 4)))))

;; ---------------------------------------------------------------------------
;; Variadic arithmetic — (add x y z ...)
;; ---------------------------------------------------------------------------

(deftest variadic-test
  (testing "add is variadic"
    (is (== 10.0 @(arr/add (arr/scalar 1.0) (arr/scalar 2.0)
                            (arr/scalar 3.0) (arr/scalar 4.0)))))
  (testing "mul is variadic"
    (is (== 24.0 @(arr/mul (arr/scalar 1.0) (arr/scalar 2.0)
                            (arr/scalar 3.0) (arr/scalar 4.0)))))
  (testing "sub is variadic: (sub 10 1 2 3) = 4"
    (is (== 4.0 @(arr/sub 10 1 2 3))))
  (testing "div is variadic: (div 24 2 3) = 4"
    (is (== 4.0 @(arr/div 24 2 3))))
  (testing "unary add is identity"
    (is (== 5.0 @(arr/add (arr/scalar 5.0)))))
  (testing "unary sub is negation"
    (is (== -5.0 @(arr/sub (arr/scalar 5.0)))))
  (testing "unary mul is identity"
    (is (== 5.0 @(arr/mul (arr/scalar 5.0)))))
  (testing "unary div is reciprocal"
    (is (== 0.5 @(arr/div (arr/scalar 2.0)))))
  (testing "variadic with auto-coercion"
    (is (== 15.0 @(arr/add (arr/scalar 1.0) 2 3 4 5)))))

;; ---------------------------------------------------------------------------
;; Value equality — (= x y) compares contents
;; ---------------------------------------------------------------------------

(deftest equality-test
  (testing "identical arrays are equal"
    (let [a (arr/scalar 3.0)]
      (is (= a a))))
  (testing "same-valued scalars are equal"
    (is (= (arr/scalar 3.0) (arr/scalar 3.0))))
  (testing "different-valued scalars are not equal"
    (is (not= (arr/scalar 3.0) (arr/scalar 4.0))))
  (testing "same-valued vectors are equal"
    (is (= (arr/from-vec [1 2 3]) (arr/from-vec [1 2 3]))))
  (testing "different-valued vectors are not equal"
    (is (not= (arr/from-vec [1 2 3]) (arr/from-vec [1 2 4]))))
  (testing "different-shaped arrays are not equal"
    (is (not= (arr/array [1 2 3 4 5 6] [2 3])
              (arr/array [1 2 3 4 5 6] [3 2]))))
  (testing "empty arrays are equal"
    (is (= (arr/from-vec []) (arr/from-vec []))))
  (testing "computed results equal to literal"
    (is (= (arr/scalar 5.0)
           (arr/add (arr/scalar 2.0) (arr/scalar 3.0))))))

;; ---------------------------------------------------------------------------
;; hashCode — consistent with equals
;; ---------------------------------------------------------------------------

(deftest hashcode-test
  (testing "equal arrays have equal hash codes"
    (is (= (.hashCode (arr/scalar 3.0))
           (.hashCode (arr/scalar 3.0)))))
  (testing "equal vectors have equal hash codes"
    (is (= (.hashCode (arr/from-vec [1 2 3]))
           (.hashCode (arr/from-vec [1 2 3])))))
  (testing "works as hash map key"
    (let [a (arr/scalar 3.0)
          b (arr/scalar 3.0)
          m {a :found}]
      (is (= :found (get m b))))))

;; ---------------------------------------------------------------------------
;; toString / print — never forces evaluation
;; ---------------------------------------------------------------------------

(deftest tostring-test
  (testing "toString includes shape, dtype, and size"
    (let [s (str (arr/from-vec [1 2 3]))]
      (is (clojure.string/includes? s ":shape [3]"))
      (is (clojure.string/includes? s ":float32"))
      (is (clojure.string/includes? s ":size 3"))))
  (testing "toString on scalar"
    (let [s (str (arr/scalar 5.0))]
      (is (clojure.string/includes? s ":shape []"))
      (is (clojure.string/includes? s ":size 1"))))
  (testing "toString on lazy result does not force eval"
    (let [result (arr/add (arr/scalar 2.0) (arr/scalar 3.0))
          s (str result)]
      (is (clojure.string/includes? s "#mlx/array"))
      (is (clojure.string/includes? s ":shape []")))))

;; ---------------------------------------------------------------------------
;; Lazy evaluation
;; ---------------------------------------------------------------------------

(deftest lazy-evaluation-test
  (testing "arithmetic is lazy — building a computation graph"
    (let [a (arr/scalar 2.0)
          b (arr/scalar 3.0)
          c (arr/add a b)                  ; lazy
          d (arr/mul c (arr/scalar 4.0))]  ; lazy
      ;; d = (2+3)*4 = 20, only computed on deref
      (is (== 20.0 @d)))))

(deftest chained-operations-test
  (testing "chained arithmetic"
    ;; (3*4 + 2) / 2 - 1 = 6
    (let [result (arr/sub (arr/div (arr/add (arr/mul (arr/scalar 3.0)
                                                     (arr/scalar 4.0))
                                            (arr/scalar 2.0))
                                   (arr/scalar 2.0))
                          (arr/scalar 1.0))]
      (is (== 6.0 @result)))))

;; ---------------------------------------------------------------------------
;; Memory management — GC handles cleanup automatically
;; ---------------------------------------------------------------------------

;; ---------------------------------------------------------------------------
;; Comparison & conditional ops
;; ---------------------------------------------------------------------------

(deftest less-test
  (testing "element-wise less-than"
    (let [a (arr/from-vec [1 3 5])
          b (arr/from-vec [2 2 6])
          result (arr/less a b)
          ;; less returns bool dtype; convert to float via where for extraction
          as-float (arr/where result 1.0 0.0)]
      (is (= [1.0 0.0 1.0] (arr/->vec as-float)))))
  (testing "less with scalar auto-coercion"
    (let [a (arr/from-vec [1 2 3])
          result (arr/less a 2)
          as-float (arr/where result 1.0 0.0)]
      (is (= [1.0 0.0 0.0] (arr/->vec as-float))))))

(deftest greater-test
  (testing "element-wise greater-than"
    (let [a (arr/from-vec [1 3 5])
          b (arr/from-vec [2 2 6])
          result (arr/greater a b)
          as-float (arr/where result 1.0 0.0)]
      (is (= [0.0 1.0 0.0] (arr/->vec as-float))))))

(deftest where-test
  (testing "branchless select with boolean condition"
    (let [cond-arr (arr/less (arr/from-vec [1 3 1]) (arr/from-vec [2 2 2]))
          x (arr/from-vec [10 20 30])
          y (arr/from-vec [40 50 60])
          result (arr/where cond-arr x y)]
      ;; cond is [true false true] -> select [10 50 30]
      (is (= [10.0 50.0 30.0] (arr/->vec result)))))
  (testing "where with scalar values"
    (let [cond-arr (arr/greater (arr/scalar 5.0) (arr/scalar 3.0))
          result (arr/where cond-arr 100 200)]
      (is (== 100.0 (arr/->double result))))))

(deftest logaddexp-test
  (testing "logaddexp is numerically stable log(exp(a) + exp(b))"
    (let [a (arr/scalar 1.0)
          b (arr/scalar 2.0)
          result (arr/->double (arr/logaddexp a b))
          expected (Math/log (+ (Math/exp 1.0) (Math/exp 2.0)))]
      (is (< (Math/abs (- expected result)) 1e-5)
          (str "expected " expected ", got " result))))
  (testing "logaddexp with large values doesn't overflow"
    (let [result (arr/->double (arr/logaddexp (arr/scalar 1000.0) (arr/scalar 1001.0)))
          expected (+ 1001.0 (Math/log (+ (Math/exp -1.0) 1.0)))]
      (is (< (Math/abs (- expected result)) 1e-2)))))

(deftest stop-gradient-test
  (testing "stop-gradient returns same value"
    (is (== 5.0 @(arr/stop-gradient (arr/scalar 5.0))))))

(deftest random-uniform-test
  (testing "random-uniform produces values in [0, 1)"
    (let [samples (arr/->vec (arr/random-uniform [100]))]
      (is (= 100 (count samples)))
      (is (every? #(and (>= % 0.0) (< % 1.0)) samples))))
  (testing "random-uniform scalar shape"
    (let [v (arr/->double (arr/random-uniform []))]
      (is (and (>= v 0.0) (< v 1.0))))))

;; ---------------------------------------------------------------------------
;; Shape manipulation ops
;; ---------------------------------------------------------------------------

(deftest sum-axis-test
  (testing "sum-axis along axis 0 of 2D array"
    (let [a (arr/array [1 2 3 4 5 6] [2 3])
          result (arr/sum-axis a 0)]
      (is (= [3] (arr/shape result)))
      (is (= [5.0 7.0 9.0] (arr/->vec result)))))
  (testing "sum-axis along axis 1 of 2D array"
    (let [a (arr/array [1 2 3 4 5 6] [2 3])
          result (arr/sum-axis a 1)]
      (is (= [2] (arr/shape result)))
      (is (= [6.0 15.0] (arr/->vec result)))))
  (testing "sum-axis with keepdims"
    (let [a (arr/array [1 2 3 4 5 6] [2 3])
          result (arr/sum-axis a 1 true)]
      (is (= [2 1] (arr/shape result)))
      (is (= [6.0 15.0] (arr/->vec result))))))

(deftest stack-test
  (testing "stack 1D arrays into 2D"
    (let [a (arr/from-vec [1 2 3])
          b (arr/from-vec [4 5 6])
          result (arr/stack [a b])]
      (is (= [2 3] (arr/shape result)))
      (is (= [1.0 2.0 3.0 4.0 5.0 6.0] (arr/->vec result)))))
  (testing "stack along axis 1"
    (let [a (arr/from-vec [1 2 3])
          b (arr/from-vec [4 5 6])
          result (arr/stack [a b] 1)]
      (is (= [3 2] (arr/shape result)))
      (is (= [1.0 4.0 2.0 5.0 3.0 6.0] (arr/->vec result))))))

(deftest expand-dims-test
  (testing "expand-dims on 1D array"
    (let [a (arr/from-vec [1 2 3])
          result (arr/expand-dims a 0)]
      (is (= [1 3] (arr/shape result)))
      (is (= [1.0 2.0 3.0] (arr/->vec result)))))
  (testing "expand-dims at end"
    (let [a (arr/from-vec [1 2 3])
          result (arr/expand-dims a 1)]
      (is (= [3 1] (arr/shape result))))))

(deftest squeeze-test
  (testing "squeeze removes length-1 axis"
    (let [a (arr/array [1 2 3] [1 3])
          result (arr/squeeze a 0)]
      (is (= [3] (arr/shape result)))
      (is (= [1.0 2.0 3.0] (arr/->vec result))))))

(deftest take-axis-test
  (testing "take along axis 0"
    (let [a (arr/array [1 2 3 4 5 6] [3 2])
          idx (arr/from-ints [0 2])
          result (arr/take-axis a idx 0)]
      (is (= [2 2] (arr/shape result)))
      (is (= [1.0 2.0 5.0 6.0] (arr/->vec result))))))

;; ---------------------------------------------------------------------------
;; Memory management — GC handles cleanup automatically
;; ---------------------------------------------------------------------------

(deftest free-idempotent-test
  (testing "free! is safe to call multiple times"
    (let [a (arr/scalar 42.0)]
      (is (== 42.0 @a))
      (arr/free! a)
      (arr/free! a)))  ; second call is a no-op

  (testing "arrays work without explicit free"
    ;; Just create and use — GC handles cleanup
    (let [results (mapv (fn [i]
                          @(arr/mul (arr/scalar (double i))
                                    (arr/scalar 2.0)))
                        (range 10))]
      (is (= [0.0 2.0 4.0 6.0 8.0 10.0 12.0 14.0 16.0 18.0]
             results)))))
