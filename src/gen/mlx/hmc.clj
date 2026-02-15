(ns gen.mlx.hmc
  "Standalone Hamiltonian Monte Carlo on MLX arrays.

   Provides HMC sampling for any differentiable log-density function
   built from gen.mlx.array operations. Uses MLX autodiff for gradient
   computation and the leapfrog integrator for Hamiltonian dynamics.

   Usage:
     (def log-density (fn [x] (arr/mul -0.5 (arr/square x))))
     (hmc-step log-density (arr/from-vec [0.0]))
     ;; => {:position MLXArray, :log-density double, :accepted? boolean}

     (sample log-density (arr/from-vec [0.0]) 500)
     ;; => vector of {:position :log-density :accepted?}"
  (:require [gen.mlx.array :as arr]
            [gen.mlx.transforms :as xforms])
  (:import [java.util.concurrent ThreadLocalRandom]))

;; ---------------------------------------------------------------------------
;; Momentum sampling — JVM random, no need for MLX AD
;; ---------------------------------------------------------------------------

(defn- sample-momentum
  "Sample momentum from N(0,I) of dimension n.
   Uses JVM ThreadLocalRandom (fast, no MLX overhead)."
  [n]
  (let [rng (ThreadLocalRandom/current)]
    (arr/from-vec (mapv (fn [_] (.nextGaussian rng)) (range n)))))

;; ---------------------------------------------------------------------------
;; Kinetic energy — 0.5 * p^T p
;; ---------------------------------------------------------------------------

(defn- kinetic-energy
  "Kinetic energy: 0.5 * sum(p^2). Returns a double."
  [momentum]
  (* 0.5 (arr/->double (arr/sum (arr/square momentum)))))

;; ---------------------------------------------------------------------------
;; Leapfrog integrator
;; ---------------------------------------------------------------------------

(defn leapfrog
  "Leapfrog integration for Hamiltonian dynamics.

   Takes a value-and-grad function, current position and momentum,
   step size eps, and number of steps L.

   The standard leapfrog scheme:
     1. Half-step momentum
     2. L full position steps interleaved with (L-1) full momentum steps
     3. Half-step momentum

   Returns {:position :momentum :log-density}."
  [vag-fn position momentum eps L]
  (let [;; Initial half-step for momentum
        {:keys [grads]} (vag-fn position)
        p (arr/add momentum (arr/mul (/ eps 2.0) (first grads)))]
    (loop [i 0
           q position
           p p]
      (if (>= i L)
        ;; Compute final value and gradient for half-step
        (let [{:keys [value grads]} (vag-fn q)
              p-final (arr/add p (arr/mul (/ eps 2.0) (first grads)))]
          {:position q
           :momentum p-final
           :log-density (arr/->double value)})
        ;; Full position step
        (let [q-new (arr/add q (arr/mul eps p))]
          (if (< i (dec L))
            ;; Full momentum step
            (let [{:keys [grads]} (vag-fn q-new)]
              (recur (inc i) q-new (arr/add p (arr/mul eps (first grads)))))
            ;; Last iteration — skip momentum (final half-step done above)
            (recur (inc i) q-new p)))))))

;; ---------------------------------------------------------------------------
;; Internal HMC step — takes pre-built vag-fn to avoid recreating it
;; ---------------------------------------------------------------------------

(defn- hmc-step*
  "Internal HMC step. Takes a pre-built value-and-grad function."
  [log-density-fn vag-fn position current-log-density n eps L]
  (let [;; Sample momentum
        momentum (sample-momentum n)
        current-ke (kinetic-energy momentum)
        current-H (- current-ke current-log-density)
        ;; Leapfrog integration
        proposal (leapfrog vag-fn position momentum eps L)
        proposed-log-density (:log-density proposal)
        proposed-ke (kinetic-energy (:momentum proposal))
        proposed-H (- proposed-ke proposed-log-density)
        ;; Metropolis-Hastings acceptance
        log-accept (- current-H proposed-H)
        accepted? (< (Math/log (Math/random)) log-accept)]
    (if accepted?
      {:position (:position proposal)
       :log-density proposed-log-density
       :accepted? true}
      {:position position
       :log-density current-log-density
       :accepted? false})))

;; ---------------------------------------------------------------------------
;; HMC step — public API for a single step
;; ---------------------------------------------------------------------------

(defn hmc-step
  "Perform a single HMC step with Metropolis-Hastings correction.

   `log-density-fn` takes an MLXArray position and returns a scalar MLXArray
   (the log-density, built from differentiable gen.mlx.array ops).

   `position` is an MLXArray (1-d vector of parameters).

   Options:
     :L   — number of leapfrog steps (default 10)
     :eps — leapfrog step size (default 0.01)

   Returns {:position MLXArray, :log-density double, :accepted? boolean}."
  [log-density-fn position & {:keys [L eps] :or {L 10 eps 0.01}}]
  (let [vag-fn (xforms/value-and-grad log-density-fn)
        n (arr/size position)
        current-log-density (arr/->double (log-density-fn position))]
    (hmc-step* log-density-fn vag-fn position current-log-density n eps L)))

;; ---------------------------------------------------------------------------
;; Sampling — collect multiple HMC steps
;; ---------------------------------------------------------------------------

(defn sample
  "Run HMC for `n-steps` iterations, collecting results.

   `log-density-fn` takes an MLXArray position and returns a scalar MLXArray.
   `initial-position` is an MLXArray (1-d vector).

   Options:
     :L   — number of leapfrog steps per HMC step (default 10)
     :eps — leapfrog step size (default 0.01)

   Returns a vector of n-steps maps, each {:position :log-density :accepted?}."
  [log-density-fn initial-position n-steps & {:keys [L eps] :or {L 10 eps 0.01}}]
  (let [vag-fn (xforms/value-and-grad log-density-fn)
        n (arr/size initial-position)
        init-log-density (arr/->double (log-density-fn initial-position))]
    (loop [i 0
           pos initial-position
           ld init-log-density
           results []]
      (if (>= i n-steps)
        results
        (let [step (hmc-step* log-density-fn vag-fn pos ld n eps L)]
          (recur (inc i)
                 (:position step)
                 (:log-density step)
                 (conj results step)))))))
