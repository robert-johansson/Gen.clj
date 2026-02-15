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
            [gen.mlx.ffi :as ffi]
            [gen.mlx.transforms :as xforms]))

;; ---------------------------------------------------------------------------
;; Momentum sampling — MLX native random
;; ---------------------------------------------------------------------------

(defn- sample-momentum
  "Sample momentum from N(0,I) of dimension n.
   Uses MLX native random (avoids JVM->MLX arr/from-vec round-trip)."
  [n]
  (arr/random-normal [n]))

;; ---------------------------------------------------------------------------
;; Kinetic energy — 0.5 * p^T p
;; ---------------------------------------------------------------------------

(defn- kinetic-energy
  "Kinetic energy: 0.5 * sum(p^2). Returns a double."
  [momentum]
  (* 0.5 (arr/->double (arr/sum (arr/square momentum)))))

(defn- kinetic-energy-arr
  "Kinetic energy: 0.5 * sum(p^2). Returns an MLXArray (stays lazy)."
  [momentum]
  (arr/mul 0.5 (arr/sum (arr/square momentum))))

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
;; Lazy leapfrog — returns all MLXArrays (no arr/->double graph breaks)
;; ---------------------------------------------------------------------------

(defn leapfrog-lazy
  "Leapfrog integration that returns all MLXArrays (no eval points).
   Same algorithm as `leapfrog` but keeps the entire computation as a lazy
   MLX graph. Pre-allocated scalar arrays eliminate repeated ensure-array calls.

   When the C shim is available, the entire loop runs in native code (1 FFI call).
   Falls back to Clojure loop otherwise.

   Returns {:position :momentum :log-density} where :log-density is MLXArray."
  [vag-fn position momentum eps-arr half-eps-arr L
   & {:keys [vag-ctx] :or {vag-ctx nil}}]
  (if (and ffi/fast-leapfrog-handle vag-ctx)
    ;; Fast path: entire leapfrog in C — 1 FFI call for L+1 vag applications
    (let [arena (java.lang.foreign.Arena/ofAuto)
          pos-out (.allocate arena 8 8)
          mom-out (.allocate arena 8 8)
          val-out (.allocate arena 8 8)
          eps-double (arr/->double eps-arr)
          status (int (.invokeWithArguments
                       ^java.lang.invoke.MethodHandle ffi/fast-leapfrog-handle
                       (object-array [pos-out mom-out val-out
                                      ^java.lang.foreign.MemorySegment vag-ctx
                                      ^java.lang.foreign.MemorySegment
                                      (:ctx (arr/handle position))
                                      ^java.lang.foreign.MemorySegment
                                      (:ctx (arr/handle momentum))
                                      (float eps-double)
                                      (int L)])))]
      (when-not (zero? status)
        (throw (ex-info "gen_mlx_leapfrog failed" {:status status})))
      {:position (arr/wrap-handle
                  {:ctx (.get ^java.lang.foreign.MemorySegment pos-out
                              java.lang.foreign.ValueLayout/ADDRESS (long 0))})
       :momentum (arr/wrap-handle
                  {:ctx (.get ^java.lang.foreign.MemorySegment mom-out
                              java.lang.foreign.ValueLayout/ADDRESS (long 0))})
       :log-density (arr/wrap-handle
                     {:ctx (.get ^java.lang.foreign.MemorySegment val-out
                                 java.lang.foreign.ValueLayout/ADDRESS (long 0))})})
    ;; Fallback: Clojure loop
    (let [;; Initial half-step for momentum
          {:keys [grads]} (vag-fn position)
          p (arr/add momentum (arr/mul half-eps-arr (first grads)))]
      (loop [i 0
             q position
             p p]
        (if (>= i L)
          ;; Compute final value and gradient for half-step
          (let [{:keys [value grads]} (vag-fn q)
                p-final (arr/add p (arr/mul half-eps-arr (first grads)))]
            {:position q
             :momentum p-final
             :log-density value}) ;; MLXArray, not double
          ;; Full position step
          (let [q-new (arr/add q (arr/mul eps-arr p))]
            (if (< i (dec L))
              ;; Full momentum step
              (let [{:keys [grads]} (vag-fn q-new)]
                (recur (inc i) q-new (arr/add p (arr/mul eps-arr (first grads)))))
              ;; Last iteration — skip momentum (final half-step done above)
              (recur (inc i) q-new p))))))))

;; ---------------------------------------------------------------------------
;; Lazy HMC step — branchless MH accept/reject, zero eval points
;; ---------------------------------------------------------------------------

(defn- hmc-step-lazy
  "HMC step that stays entirely in the MLX graph. Uses arr/where for
   branchless MH accept/reject. No arr/->double calls.

   Returns {:position :log-density :accepted?} where all values are MLXArrays."
  [vag-fn position current-ld-arr n eps-arr half-eps-arr L vag-ctx]
  (let [;; Sample momentum
        momentum (sample-momentum n)
        current-ke (kinetic-energy-arr momentum)
        current-H (arr/sub current-ke current-ld-arr)
        ;; Leapfrog integration (all lazy, uses C shim when available)
        proposal (leapfrog-lazy vag-fn position momentum eps-arr half-eps-arr L
                                :vag-ctx vag-ctx)
        proposed-ld (:log-density proposal)
        proposed-ke (kinetic-energy-arr (:momentum proposal))
        proposed-H (arr/sub proposed-ke proposed-ld)
        ;; Metropolis-Hastings acceptance — branchless via arr/where
        log-accept (arr/sub current-H proposed-H)
        log-u (arr/log (arr/random-uniform []))
        accepted (arr/less log-u log-accept)]
    {:position    (arr/where accepted (:position proposal) position)
     :log-density (arr/where accepted proposed-ld current-ld-arr)
     :accepted?   accepted}))

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
;; Single leapfrog step — used by NUTS for tree building
;; ---------------------------------------------------------------------------

(defn leapfrog-one-step
  "Single leapfrog integration step. Returns {:position :momentum :log-density}.
   Used by NUTS for tree building where each step extends the trajectory."
  [vag-fn position momentum eps]
  (leapfrog vag-fn position momentum eps 1))

;; ---------------------------------------------------------------------------
;; Sampling — collect multiple HMC steps
;; ---------------------------------------------------------------------------

(defn sample
  "Run HMC for `n-steps` iterations, collecting results.

   `log-density-fn` takes an MLXArray position and returns a scalar MLXArray.
   `initial-position` is an MLXArray (1-d vector).

   Uses lazy HMC: the entire leapfrog + MH accept/reject is built as a single
   MLX computation graph per step, evaluated once at the outermost level.

   Options:
     :L   — number of leapfrog steps per HMC step (default 10)
     :eps — leapfrog step size (default 0.01)

   Returns a vector of n-steps maps, each {:position :log-density :accepted?}."
  [log-density-fn initial-position n-steps & {:keys [L eps] :or {L 10 eps 0.01}}]
  (let [vag-fn (xforms/value-and-grad log-density-fn)
        vag-ctx (::xforms/vag-ctx (meta vag-fn))
        n (arr/size initial-position)
        ;; Pre-allocate scalar arrays for step size
        eps-arr (arr/scalar eps)
        half-eps-arr (arr/scalar (/ eps 2.0))
        ;; Initial log-density as MLXArray (for lazy path)
        init-ld-arr (log-density-fn initial-position)]
    (loop [i 0
           pos initial-position
           ld-arr init-ld-arr
           results []]
      (if (>= i n-steps)
        results
        ;; Lazy HMC step — builds entire graph without eval
        (let [step (hmc-step-lazy vag-fn pos ld-arr n eps-arr half-eps-arr L vag-ctx)
              ;; Single eval point: extract log-density and accepted? for result
              ld-double (arr/->double (:log-density step))
              ;; Convert bool to float before extraction (bool dtype can't use item-float32)
              accepted-float (arr/where (:accepted? step) 1.0 0.0)
              accepted? (> (arr/->double accepted-float) 0.5)]
          (recur (inc i)
                 (:position step)      ;; pass MLXArray to next iteration
                 (:log-density step)   ;; pass MLXArray to next iteration
                 (conj results {:position (:position step)
                                :log-density ld-double
                                :accepted? accepted?})))))))

;; ---------------------------------------------------------------------------
;; MALA — Metropolis-Adjusted Langevin Algorithm (HMC with L=1)
;; ---------------------------------------------------------------------------

(defn mala-step
  "Single MALA step (Metropolis-Adjusted Langevin Algorithm).
   Equivalent to HMC with a single leapfrog step (L=1).

   Returns {:position MLXArray, :log-density double, :accepted? boolean}."
  [log-density-fn position & {:keys [eps] :or {eps 0.01}}]
  (hmc-step log-density-fn position :L 1 :eps eps))

(defn mala-sample
  "Run MALA for n-steps iterations.

   MALA is HMC with L=1 — a single gradient-informed proposal per step.
   Simpler than full HMC but can be effective for well-conditioned targets.

   Returns a vector of n-steps maps, each {:position :log-density :accepted?}."
  [log-density-fn initial-position n-steps & {:keys [eps] :or {eps 0.01}}]
  (sample log-density-fn initial-position n-steps :L 1 :eps eps))

