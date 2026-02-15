(ns gen.inference.mh
  "Metropolis-Hastings kernels for MCMC inference.

   Core function `mh` performs a single MH step using `regenerate` as the
   proposal (prior proposal). Composable helpers let you build MCMC chains
   and multi-kernel sweeps."
  (:require [clojure.math :as math]
            [gen.generative-function :as gf]))

(defn mh
  "Perform one Metropolis-Hastings step on `trace` using `selection` to
   identify which addresses to repropose from the prior.

   Returns `{:trace <new-or-old-trace> :accepted? <boolean>}`."
  [trace selection]
  (let [{new-trace :trace
         weight    :weight} (gf/regenerate trace selection)
        accept-prob (math/exp (min 0.0 (double weight)))
        accepted?   (< (rand) accept-prob)]
    (if accepted?
      {:trace new-trace :accepted? true}
      {:trace trace     :accepted? false})))

(defn mh-step
  "Returns a kernel function `(fn [trace] -> trace)` that applies one MH
   step using `selection`.

   Usage:
     (def step (mh-step #{:slope}))
     (step trace) ;=> updated trace"
  [selection]
  (fn [trace]
    (:trace (mh trace selection))))

(defn chain
  "Returns a lazy infinite sequence of traces by repeatedly applying
   `kernel` (a fn from trace to trace).

   Usage:
     (take 1000 (chain initial-trace (mh-step #{:slope})))"
  [trace kernel]
  (iterate kernel trace))

(defn cycle-kernels
  "Compose multiple kernel functions into a single sweep that applies
   each kernel in sequence.

   Usage:
     (def sweep (cycle-kernels
                  (mh-step #{:slope})
                  (mh-step #{:intercept})))
     (take 1000 (chain trace sweep))"
  [& kernels]
  (fn [trace]
    (reduce (fn [t k] (k t)) trace kernels)))
