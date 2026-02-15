(ns gen.mlx.benchmark-inference-test
  "Inference benchmarks comparing CPU-only (MH) vs MLX gradient-based inference.

   Three benchmarks:
   1. Normal posterior: MH vs HMC vs NUTS
   2. Linear regression scaling: MH vs HMC vs NUTS at varying N
   3. MAP convergence speed: gradient vs random search

   Run with: bb test:mlx (tagged ^:benchmark for filtering)"
  (:require [clojure.string :as str]
            [clojure.test :refer [deftest testing]]
            [gen.choicemap :as choicemap]
            [gen.distribution :as d]
            [gen.distribution.kixi :as kixi]
            [gen.dynamic :as dynamic]
            [gen.generative-function :as gf]
            [gen.inference.mh :as mh]
            [gen.mlx.array :as arr]
            [gen.mlx.distribution :as mlx-dist]
            [gen.mlx.dynamic :as mlx-dyn]
            [gen.trace :as trace]))

;; ---------------------------------------------------------------------------
;; Helpers
;; ---------------------------------------------------------------------------

(defn- bench
  "Run `f` with warmup, then measure `trial-n` iterations.
   Returns mean time in nanoseconds."
  [warmup-n trial-n f]
  (dotimes [_ warmup-n] (f))
  (let [start (System/nanoTime)]
    (dotimes [_ trial-n] (f))
    (/ (double (- (System/nanoTime) start)) trial-n)))

(defn- format-time
  "Format nanoseconds as human-readable."
  [ns]
  (cond
    (< ns 1000) (format "%.0f ns" ns)
    (< ns 1e6)  (format "%.1f μs" (/ ns 1e3))
    (< ns 1e9)  (format "%.1f ms" (/ ns 1e6))
    :else        (format "%.2f s" (/ ns 1e9))))

(defn- mean [xs]
  (/ (reduce + (map double xs)) (count xs)))

(defn- choice-val
  "Extract raw value from a trace's choicemap at an address."
  [tr addr]
  (choicemap/get-value (choicemap/get-submap (trace/get-choices tr) addr)))

(defn- ess
  "Effective sample size via lag-1 autocorrelation."
  [samples]
  (let [n (count samples)
        m (mean samples)
        cov1 (/ (reduce + (map (fn [[a b]] (* (- a m) (- b m)))
                               (map vector samples (rest samples))))
                (dec n))
        var0 (/ (reduce + (map #(let [d (- % m)] (* d d)) samples))
                (dec n))
        rho1 (if (pos? var0) (/ cov1 var0) 0.0)]
    (/ (double n) (max 1.0 (+ 1.0 (* 2.0 rho1))))))

;; ---------------------------------------------------------------------------
;; Benchmark 1: Normal posterior — MH vs HMC vs NUTS
;;
;; Target: Normal(3, 0.5) posterior
;; ---------------------------------------------------------------------------

(def ^:private mh-model
  "CPU model for MH baseline."
  (dynamic/gen []
    (dynamic/trace! :x kixi/normal 0.0 10.0)))

(def ^:private mlx-model
  "MLX model for HMC/NUTS."
  (mlx-dyn/gen []
    (dynamic/trace! :x mlx-dist/normal 0.0 10.0)))

(deftest ^:benchmark normal-posterior-mh-vs-hmc-vs-nuts
  (testing "Normal posterior: MH vs HMC vs NUTS (1000 samples)"
    (let [n-samples  1000
          target-mu  3.0
          target-sig 0.5
          obs-constraint (choicemap/choicemap {:x target-mu})

          ;; MH baseline
          mh-start  (System/nanoTime)
          mh-trace0 (:trace (gf/generate mh-model [] obs-constraint))
          mh-traces (take n-samples (mh/chain mh-trace0 (mh/mh-step #{:x})))
          mh-vals   (mapv #(choice-val % :x) mh-traces)
          _         (doall mh-vals)
          mh-time   (- (System/nanoTime) mh-start)
          mh-ess    (ess mh-vals)
          mh-mean   (mean mh-vals)

          ;; HMC
          hmc-trace0 (:trace (gf/generate mlx-model [] obs-constraint))
          hmc-start  (System/nanoTime)
          hmc-results (mlx-dyn/hmc-sample hmc-trace0 n-samples :L 10 :eps 0.1)
          hmc-vals   (mapv #(get (:choices %) :x) hmc-results)
          hmc-time   (- (System/nanoTime) hmc-start)
          hmc-ess    (ess hmc-vals)
          hmc-mean   (mean hmc-vals)

          ;; NUTS
          nuts-trace0 (:trace (gf/generate mlx-model [] obs-constraint))
          nuts-start  (System/nanoTime)
          nuts-results (mlx-dyn/nuts-sample nuts-trace0 n-samples
                                           :max-depth 5 :target-accept 0.8)
          nuts-vals   (mapv #(get (:choices %) :x) nuts-results)
          nuts-time   (- (System/nanoTime) nuts-start)
          nuts-ess    (ess nuts-vals)
          nuts-mean   (mean nuts-vals)]

      (println)
      (println "=== Benchmark 1: Normal Posterior (target μ=3.0) ===")
      (printf "  %-10s %15s %10s %10s%n" "Method" "Wall-clock" "ESS" "Post.Mean")
      (printf "  %-10s %15s %10s %10s%n" "------" "----------" "---" "---------")
      (printf "  %-10s %15s %10.0f %10.3f%n" "MH"   (format-time mh-time)   mh-ess   mh-mean)
      (printf "  %-10s %15s %10.0f %10.3f%n" "HMC"  (format-time hmc-time)  hmc-ess  hmc-mean)
      (printf "  %-10s %15s %10.0f %10.3f%n" "NUTS" (format-time nuts-time) nuts-ess nuts-mean)
      (println))))

;; ---------------------------------------------------------------------------
;; Benchmark 2: Linear regression scaling
;;
;; Model: y_i = slope * x_i + intercept + noise
;; ---------------------------------------------------------------------------

(defn- make-mh-linreg [xs ys]
  (let [model (dynamic/gen [xs ys]
                (let [slope     (dynamic/trace! :slope kixi/normal 0.0 10.0)
                      intercept (dynamic/trace! :intercept kixi/normal 0.0 10.0)]
                  (doseq [i (range (count xs))]
                    (dynamic/trace! (keyword (str "y" i)) kixi/normal
                                   (+ (* slope (nth xs i)) intercept) 0.5))
                  [slope intercept]))]
    model))

(defn- make-mlx-linreg [xs ys]
  (let [model (mlx-dyn/gen [xs ys]
                (let [slope     (dynamic/trace! :slope mlx-dist/normal 0.0 10.0)
                      intercept (dynamic/trace! :intercept mlx-dist/normal 0.0 10.0)]
                  (doseq [i (range (count xs))]
                    (dynamic/trace! (keyword (str "y" i)) mlx-dist/normal
                                   (+ (* slope (nth xs i)) intercept) 0.5))
                  [slope intercept]))]
    model))

(deftest ^:benchmark linear-regression-scaling
  (testing "Linear regression: MH vs HMC vs NUTS at varying N"
    (let [true-slope 2.0
          true-inter 1.0
          n-samples  200]

      (println)
      (println "=== Benchmark 2: Linear Regression Scaling ===")
      (printf "  %-6s %15s %15s %15s %10s %10s %10s%n"
              "N" "MH time" "HMC time" "NUTS time" "MH ESS" "HMC ESS" "NUTS ESS")
      (printf "  %-6s %15s %15s %15s %10s %10s %10s%n"
              "---" "-------" "--------" "---------" "------" "-------" "--------")

      (doseq [n [10 50 100]]
        (let [xs (vec (map #(/ (double %) n) (range n)))
              ys (vec (map #(+ (* true-slope %) true-inter
                               (* 0.5 (- (rand) 0.5))) xs))
              y-constraints (reduce (fn [m i]
                                      (assoc m (keyword (str "y" i)) (nth ys i)))
                                    {} (range n))
              constraints (choicemap/choicemap y-constraints)

              ;; MH
              mh-model (make-mh-linreg xs ys)
              mh-tr    (:trace (gf/generate mh-model [xs ys] constraints))
              mh-start (System/nanoTime)
              mh-sweep (mh/cycle-kernels (mh/mh-step #{:slope})
                                         (mh/mh-step #{:intercept}))
              mh-traces (doall (take n-samples (mh/chain mh-tr mh-sweep)))
              mh-slopes (mapv #(choice-val % :slope) mh-traces)
              mh-time   (- (System/nanoTime) mh-start)
              mh-ess    (ess mh-slopes)

              ;; HMC (with selection — only slope + intercept)
              sel #{:slope :intercept}
              mlx-model (make-mlx-linreg xs ys)
              mlx-tr    (:trace (gf/generate mlx-model [xs ys] constraints))
              hmc-start (System/nanoTime)
              hmc-results (mlx-dyn/hmc-sample mlx-tr n-samples :L 10 :eps 0.01
                                              :selection sel)
              hmc-slopes  (mapv #(get (:choices %) :slope) hmc-results)
              hmc-time    (- (System/nanoTime) hmc-start)
              hmc-ess     (ess hmc-slopes)

              ;; NUTS (with selection — only slope + intercept)
              mlx-tr2     (:trace (gf/generate mlx-model [xs ys] constraints))
              nuts-start  (System/nanoTime)
              nuts-results (mlx-dyn/nuts-sample mlx-tr2 n-samples
                                               :max-depth 5 :target-accept 0.8
                                               :selection sel)
              nuts-slopes  (mapv #(get (:choices %) :slope) nuts-results)
              nuts-time    (- (System/nanoTime) nuts-start)
              nuts-ess     (ess nuts-slopes)]

          (printf "  %-6d %15s %15s %15s %10.0f %10.0f %10.0f%n"
                  n
                  (format-time mh-time)
                  (format-time hmc-time)
                  (format-time nuts-time)
                  mh-ess hmc-ess nuts-ess)))
      (println))))

;; ---------------------------------------------------------------------------
;; Benchmark 3: MAP convergence speed
;;
;; 2-d target with known mode at [3, -2]
;; Compare MAP (gradient ascent) vs MH (random walk to best)
;; ---------------------------------------------------------------------------

(def ^:private map-model-mh
  "CPU model: 2-d Gaussian with mode at [3, -2]."
  (dynamic/gen []
    (let [x (dynamic/trace! :x kixi/normal 3.0 0.5)
          y (dynamic/trace! :y kixi/normal -2.0 0.5)]
      [x y])))

(def ^:private map-model-mlx
  "MLX model: 2-d Gaussian with mode at [3, -2]."
  (mlx-dyn/gen []
    (let [x (dynamic/trace! :x mlx-dist/normal 3.0 0.5)
          y (dynamic/trace! :y mlx-dist/normal -2.0 0.5)]
      [x y])))

(deftest ^:benchmark map-convergence-speed
  (testing "MAP convergence: gradient ascent vs MH best-sample"
    (let [true-mode  [3.0 -2.0]
          epsilon    0.1
          n-steps    500

          ;; MAP (gradient ascent)
          map-tr    (gf/simulate map-model-mlx [])
          map-start (System/nanoTime)
          map-result (mlx-dyn/map-optimize map-tr n-steps :lr 0.05 :tol 1e-6)
          map-time   (- (System/nanoTime) map-start)
          map-x      (get (:choices map-result) :x)
          map-y      (get (:choices map-result) :y)
          map-dist   (Math/sqrt (+ (Math/pow (- map-x 3.0) 2)
                                   (Math/pow (- map-y -2.0) 2)))

          ;; MH: run many steps, track best
          mh-tr     (gf/simulate map-model-mh [])
          mh-start  (System/nanoTime)
          mh-sweep  (mh/cycle-kernels (mh/mh-step #{:x})
                                       (mh/mh-step #{:y}))
          mh-chain  (take n-steps (mh/chain mh-tr mh-sweep))
          best-mh   (reduce (fn [best tr]
                              (if (> (trace/get-score tr)
                                     (trace/get-score best))
                                tr best))
                            mh-chain)
          mh-time   (- (System/nanoTime) mh-start)
          mh-x       (choice-val best-mh :x)
          mh-y       (choice-val best-mh :y)
          mh-dist    (Math/sqrt (+ (Math/pow (- mh-x 3.0) 2)
                                   (Math/pow (- mh-y -2.0) 2)))]

      (println)
      (println "=== Benchmark 3: MAP Convergence (target mode=[3,-2]) ===")
      (printf "  %-10s %15s %15s %15s%n"
              "Method" "Wall-clock" "Dist to mode" "Log-density")
      (printf "  %-10s %15s %15s %15s%n"
              "------" "----------" "------------" "-----------")
      (printf "  %-10s %15s %15.4f %15.3f%n"
              "MAP" (format-time map-time) map-dist (:log-density map-result))
      (printf "  %-10s %15s %15.4f %15.3f%n"
              "MH-best" (format-time mh-time) mh-dist (trace/get-score best-mh))
      (println))))
