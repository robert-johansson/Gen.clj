(ns gen.mlx.transforms-test
  "Tests for MLX function transforms — the key proof-of-concept.
   Proves that Clojure can compute gradients via MLX's autodiff."
  (:require [clojure.test :refer [deftest is testing]]
            [gen.mlx.array :as arr]
            [gen.mlx.transforms :as xforms]))

(defn- close?
  "Check if two doubles are within tolerance."
  ([expected actual] (close? expected actual 1e-4))
  ([expected actual tol]
   (< (Math/abs (- (double expected) (double actual))) tol)))

;; ---------------------------------------------------------------------------
;; value-and-grad — returns {:value :grads}
;; ---------------------------------------------------------------------------

(deftest value-and-grad-quadratic
  (testing "f(x) = x^2, f'(3) = 6"
    (let [f      (fn [x] (arr/mul x x))
          vag-f  (xforms/value-and-grad f)
          result (vag-f (arr/scalar 3.0))]
      (is (close? 9.0 (arr/->double (:value result))))
      (is (close? 6.0 (arr/->double (first (:grads result))))))))

(deftest value-and-grad-sum-of-squares
  (testing "f(x) = x^2 + x^2 = 2x^2, f'(3) = 12"
    (let [f      (fn [x] (arr/add (arr/mul x x) (arr/mul x x)))
          vag-f  (xforms/value-and-grad f)
          result (vag-f (arr/scalar 3.0))]
      (is (close? 18.0 (arr/->double (:value result))))
      (is (close? 12.0 (arr/->double (first (:grads result))))))))

(deftest value-and-grad-cubic
  (testing "f(x) = x^3 = x * x * x, f'(2) = 12"
    (let [f      (fn [x] (arr/mul x (arr/mul x x)))
          vag-f  (xforms/value-and-grad f)
          result (vag-f (arr/scalar 2.0))]
      (is (close? 8.0 (arr/->double (:value result))))
      (is (close? 12.0 (arr/->double (first (:grads result))))))))

(deftest value-and-grad-exp
  (testing "f(x) = exp(x), f'(1) = e"
    (let [f      (fn [x] (arr/exp x))
          vag-f  (xforms/value-and-grad f)
          result (vag-f (arr/scalar 1.0))]
      (is (close? Math/E (arr/->double (:value result))))
      (is (close? Math/E (arr/->double (first (:grads result))))))))

(deftest value-and-grad-log
  (testing "f(x) = log(x), f'(2) = 0.5"
    (let [f      (fn [x] (arr/log x))
          vag-f  (xforms/value-and-grad f)
          result (vag-f (arr/scalar 2.0))]
      (is (close? (Math/log 2.0) (arr/->double (:value result))))
      (is (close? 0.5 (arr/->double (first (:grads result))))))))

(deftest value-and-grad-softplus
  (testing "f(x) = log(exp(x) + 1), softplus derivative = sigmoid"
    (let [f      (fn [x] (arr/log (arr/add (arr/exp x) (arr/scalar 1.0))))
          vag-f  (xforms/value-and-grad f)
          result (vag-f (arr/scalar 1.0))
          ;; sigmoid(1) = e/(1+e)
          expected-grad (/ Math/E (+ 1.0 Math/E))]
      (is (close? (Math/log (+ (Math/exp 1.0) 1.0))
                  (arr/->double (:value result))))
      (is (close? expected-grad
                  (arr/->double (first (:grads result))))))))

(deftest value-and-grad-sqrt
  (testing "f(x) = sqrt(x), f'(4) = 0.25"
    (let [f      (fn [x] (arr/sqrt x))
          vag-f  (xforms/value-and-grad f)
          result (vag-f (arr/scalar 4.0))]
      (is (close? 2.0 (arr/->double (:value result))))
      (is (close? 0.25 (arr/->double (first (:grads result))))))))

(deftest value-and-grad-multiarg
  (testing "f(x,y) = x*y, df/dx at (3,4) = 4, df/dy at (3,4) = 3"
    (let [f      (fn [x y] (arr/mul x y))
          vag-f  (xforms/value-and-grad f {:argnums [0 1]})
          result (vag-f (arr/scalar 3.0) (arr/scalar 4.0))]
      (is (close? 12.0 (arr/->double (:value result))))
      (is (close? 4.0 (arr/->double (first (:grads result)))))
      (is (close? 3.0 (arr/->double (second (:grads result))))))))

;; ---------------------------------------------------------------------------
;; grad — returns gradient only (standard AD convention)
;; ---------------------------------------------------------------------------

(deftest grad-quadratic
  (testing "grad returns just the gradient, not {:value :grads}"
    (let [f      (fn [x] (arr/mul x x))
          grad-f (xforms/grad f)]
      (is (close? 6.0 @(grad-f (arr/scalar 3.0))))
      (is (close? 10.0 @(grad-f (arr/scalar 5.0)))))))

(deftest grad-exp
  (testing "grad of exp(x) at x=0 is 1.0"
    (let [grad-f (xforms/grad (fn [x] (arr/exp x)))]
      (is (close? 1.0 @(grad-f (arr/scalar 0.0)))))))

(deftest grad-multiarg
  (testing "grad with multiple argnums returns vector of gradients"
    (let [f      (fn [x y] (arr/mul x y))
          grad-f (xforms/grad f {:argnums [0 1]})
          grads  (grad-f (arr/scalar 3.0) (arr/scalar 4.0))]
      ;; With multiple argnums, grad returns a vector
      (is (vector? grads))
      (is (close? 4.0 @(first grads)))
      (is (close? 3.0 @(second grads))))))

(deftest grad-composition
  (testing "grad of composed function: f(x) = exp(x^2)"
    (let [f (fn [x] (arr/exp (arr/mul x x)))
          ;; f'(x) = 2x * exp(x^2)
          grad-f (xforms/grad f)
          x 1.0
          expected (* 2.0 x (Math/exp (* x x)))]
      (is (close? expected @(grad-f (arr/scalar x)))))))

;; ---------------------------------------------------------------------------
;; vjp — reverse-mode AD
;; ---------------------------------------------------------------------------

(deftest vjp-basic
  (testing "VJP of x^2 at x=3 with cotangent=1"
    (let [f (fn [x] (arr/mul x x))
          result (xforms/vjp f
                             [(arr/scalar 3.0)]
                             [(arr/scalar 1.0)])]
      (is (close? 9.0 (arr/->double (first (:values result)))))
      (is (close? 6.0 (arr/->double (first (:vjps result))))))))

;; ---------------------------------------------------------------------------
;; jvp — forward-mode AD
;; ---------------------------------------------------------------------------

(deftest jvp-basic
  (testing "JVP of x^2 at x=3 with tangent=1"
    (let [f (fn [x] (arr/mul x x))
          result (xforms/jvp f
                             [(arr/scalar 3.0)]
                             [(arr/scalar 1.0)])]
      (is (close? 9.0 (arr/->double (first (:values result)))))
      (is (close? 6.0 (arr/->double (first (:jvps result))))))))
