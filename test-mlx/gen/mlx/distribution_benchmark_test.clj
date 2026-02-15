(ns gen.mlx.distribution-benchmark-test
  "Speed benchmarks comparing kixi (pure JVM) vs MLX (native FFI) distributions.

   Four scenarios:
   1. Scalar logpdf — shows per-call FFI overhead and compile speedup
   2. Batched logpdf — shows where MLX vectorization wins
   3. Compiled batched — shows compile eliminates graph-building overhead
   4. Gradient computation — MLX-only capability

   Run with: bb test:mlx"
  (:require [clojure.string]
            [clojure.test :refer [deftest testing]]
            [gen.distribution :as d]
            [gen.distribution.kixi :as kixi]
            [gen.distribution.math.log-likelihood :as ll]
            [gen.mlx.array :as arr]
            [gen.mlx.distribution :as mlx-dist]
            [gen.mlx.transforms :as xforms]))

;; ---------------------------------------------------------------------------
;; Timing helpers
;; ---------------------------------------------------------------------------

(defn- bench
  "Run `f` with warmup, then measure `trial-n` iterations.
   Returns mean time in nanoseconds."
  [warmup-n trial-n f]
  (dotimes [_ warmup-n] (f))
  (let [start (System/nanoTime)]
    (dotimes [_ trial-n] (f))
    (let [elapsed (- (System/nanoTime) start)]
      (/ (double elapsed) trial-n))))

(defn- format-ns
  "Format nanoseconds with commas for readability."
  [ns]
  (let [n (long ns)]
    (if (< n 1000)
      (str n " ns")
      (let [s (str n)
            groups (loop [s s acc []]
                     (if (<= (count s) 3)
                       (cons s acc)
                       (recur (subs s 0 (- (count s) 3))
                              (cons (subs s (- (count s) 3)) acc))))]
        (str (clojure.string/join "," groups) " ns")))))

(defn- print-row
  "Print a benchmark row: label and timing."
  [label ns-per-call]
  (printf "  %-24s %s%n" label (format-ns ns-per-call)))

;; ---------------------------------------------------------------------------
;; Scenario 1: Scalar logpdf (single value)
;;
;; kixi: pure JVM Math/log + arithmetic (~100-500ns)
;; MLX compiled: FFI call to pre-compiled kernel
;; MLX uncompiled: FFI overhead for ~7 native calls to build graph + eval
;; ---------------------------------------------------------------------------

(deftest ^:benchmark scalar-logpdf-comparison
  (testing "Scalar logpdf: kixi vs MLX (compiled vs uncompiled)"
    (let [mu 0.0 sigma 1.0 v 0.5
          kixi-dist   (kixi/normal-distribution mu sigma)
          mlx-d       (mlx-dist/normal-distribution mu sigma)
          ;; Uncompiled: build graph every call
          mlx-uncomp  (mlx-dist/normal-distribution mu sigma)
          mu-arr      (arr/scalar mu)
          sigma-arr   (arr/scalar sigma)
          kixi-raw-ns (bench 1000 100000 #(ll/gaussian mu sigma v))
          kixi-ns     (bench 1000 100000 #(d/logpdf kixi-dist v))
          ;; Record uses compiled-gaussian-logpdf internally
          mlx-ns      (bench 1000 10000  #(d/logpdf mlx-d v))
          ;; Direct uncompiled for comparison
          mlx-uncomp-ns (bench 1000 10000
                               #(arr/->double
                                 (mlx-dist/gaussian-logpdf mu-arr sigma-arr
                                                           (arr/scalar v))))]
      (println)
      (println "--- Scalar logpdf (1 value) ---")
      (print-row "kixi (raw math):" kixi-raw-ns)
      (print-row "kixi (protocol):" kixi-ns)
      (print-row "mlx (compiled):" mlx-ns)
      (print-row "mlx (uncompiled):" mlx-uncomp-ns)
      (printf "  compiled/uncompiled:     %.1fx faster%n" (/ mlx-uncomp-ns mlx-ns))
      (println))))

;; ---------------------------------------------------------------------------
;; Scenario 2: Batched logpdf (N values at once)
;;
;; kixi: loop N times, each doing JVM math. O(N) with low constant.
;; MLX uncompiled: builds graph each call, but vectorized compute.
;; MLX compiled: skips graph build, just runs vectorized kernel.
;; ---------------------------------------------------------------------------

(deftest ^:benchmark batched-logpdf-comparison
  (testing "Batched logpdf: kixi loop vs MLX vectorized (compiled vs uncompiled)"
    (let [mu 0.0 sigma 1.0
          mu-arr    (arr/scalar mu)
          sigma-arr (arr/scalar sigma)
          compiled-fn mlx-dist/compiled-gaussian-logpdf
          sizes [100 1000 10000 100000]]
      (println "--- Batched logpdf (N values, mu=0, sigma=1) ---")
      (printf "  %-10s %15s %15s %15s %10s%n" "N" "kixi" "mlx-compiled" "mlx-uncomp" "speedup")
      (printf "  %-10s %15s %15s %15s %10s%n" "---" "---" "---" "---" "---")
      (doseq [n sizes]
        (let [values    (vec (repeatedly n #(- (* 2.0 (Math/random)) 1.0)))
              kixi-dist (kixi/normal-distribution mu sigma)
              mlx-arr   (arr/from-vec values)
              trials    (max 1 (quot 10000 n))
              ;; kixi: loop over all values
              kixi-ns   (bench 100 trials
                               #(reduce + (mapv (fn [v] (d/logpdf kixi-dist v))
                                                values)))
              ;; MLX compiled: pre-compiled kernel, no graph building
              compiled-ns (bench 100 trials
                                 #(do (arr/eval!
                                       (compiled-fn mu-arr sigma-arr mlx-arr))
                                      nil))
              ;; MLX uncompiled: builds graph each call
              uncomp-ns (bench 100 trials
                               #(do (arr/eval!
                                     (mlx-dist/gaussian-logpdf mu sigma mlx-arr))
                                    nil))
              speedup   (/ kixi-ns compiled-ns)]
          (printf "  %-10s %15s %15s %15s %9.1fx%n"
                  (str "N=" n)
                  (format-ns kixi-ns)
                  (format-ns compiled-ns)
                  (format-ns uncomp-ns)
                  speedup)))
      (println))))

;; ---------------------------------------------------------------------------
;; Scenario 3: logpdf + gradient (MLX only)
;;
;; kixi cannot compute gradients at all — this is the unique MLX capability.
;; Show cost of value-and-grad for scalar and batched cases.
;; ---------------------------------------------------------------------------

(deftest ^:benchmark gradient-benchmark
  (testing "Gradient computation (MLX only)"
    (println "--- Gradient: value-and-grad of logpdf ---")

    ;; Scalar gradient
    (let [sigma-arr (arr/scalar 1.0)
          v-arr     (arr/scalar 0.5)
          vag-fn    (xforms/value-and-grad
                     (fn [mu]
                       (mlx-dist/gaussian-logpdf mu sigma-arr v-arr)))
          mu-arr    (arr/scalar 0.0)
          scalar-ns (bench 500 5000 #(let [{:keys [value grads]} (vag-fn mu-arr)]
                                       (arr/eval! value)
                                       (arr/eval! (first grads))))]
      (print-row "scalar grad:" scalar-ns))

    ;; Scalar gradient w.r.t. both mu and sigma
    (let [v-arr       (arr/scalar 0.5)
          vag-fn      (xforms/value-and-grad
                       (fn [mu sigma]
                         (mlx-dist/gaussian-logpdf mu sigma v-arr))
                       {:argnums [0 1]})
          mu-arr      (arr/scalar 0.0)
          sigma-arr   (arr/scalar 1.0)
          multi-ns    (bench 500 5000 #(let [{:keys [value grads]} (vag-fn mu-arr sigma-arr)]
                                         (arr/eval! value)
                                         (doseq [g grads] (arr/eval! g))))]
      (print-row "scalar grad (2 args):" multi-ns))

    ;; Batched gradient — sum of logpdfs, grad w.r.t. mu
    (doseq [n [1000 10000]]
      (let [values    (vec (repeatedly n #(- (* 2.0 (Math/random)) 1.0)))
            v-arr     (arr/from-vec values)
            sigma-arr (arr/scalar 1.0)
            vag-fn    (xforms/value-and-grad
                       (fn [mu]
                         (arr/sum (mlx-dist/gaussian-logpdf mu sigma-arr v-arr))))
            mu-arr    (arr/scalar 0.0)
            batch-ns  (bench 200 1000 #(let [{:keys [value grads]} (vag-fn mu-arr)]
                                         (arr/eval! value)
                                         (arr/eval! (first grads))))]
        (print-row (str "batch grad N=" n ":") batch-ns)))
    (println)))
