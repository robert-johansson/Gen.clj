(ns gen.mlx.map-optimize
  "Maximum a posteriori (MAP) optimization via gradient ascent.

   Uses Adam optimizer with MLX autodiff for gradients. Maximizes
   log p(θ|data) directly — Adam update *adds* gradient (ascent)
   rather than subtracting (descent).

   Usage:
     (def log-density (fn [x] (arr/neg (arr/sum (arr/square (arr/sub x 3.0))))))
     (map-optimize log-density (arr/from-vec [0.0]) 500)
     ;; => {:position MLXArray near [3.0], :log-density ~0.0, :history [...]}"
  (:require [gen.mlx.array :as arr]
            [gen.mlx.transforms :as xforms]))

;; ---------------------------------------------------------------------------
;; Adam state
;; ---------------------------------------------------------------------------

(defrecord AdamState [position m v t])

(defn- make-adam-state
  "Create initial Adam state with zero moments."
  [position]
  (let [n (arr/size position)]
    (->AdamState position
                 (arr/from-vec (vec (repeat n 0.0)))
                 (arr/from-vec (vec (repeat n 0.0)))
                 0)))

;; ---------------------------------------------------------------------------
;; Adam step — gradient ASCENT (maximizing log-density)
;; ---------------------------------------------------------------------------

(defn adam-step
  "Single Adam optimizer step for gradient ascent.

   Updates position in the direction of the gradient (ascent, not descent).
   Returns updated AdamState."
  [grad-arr ^AdamState state {:keys [lr beta1 beta2 epsilon]
                               :or {lr 0.01 beta1 0.9 beta2 0.999 epsilon 1e-8}}]
  (let [t-new    (inc (.t state))
        ;; Update biased first moment estimate
        m-new    (arr/add (arr/mul beta1 (.m state))
                          (arr/mul (- 1.0 beta1) grad-arr))
        ;; Update biased second raw moment estimate
        v-new    (arr/add (arr/mul beta2 (.v state))
                          (arr/mul (- 1.0 beta2) (arr/square grad-arr)))
        ;; Bias-corrected moment estimates
        m-hat    (arr/div m-new (- 1.0 (Math/pow beta1 t-new)))
        v-hat    (arr/div v-new (- 1.0 (Math/pow beta2 t-new)))
        ;; Ascent: position += lr * m_hat / (sqrt(v_hat) + eps)
        pos-new  (arr/add (.position state)
                          (arr/mul lr (arr/div m-hat
                                               (arr/add (arr/sqrt v-hat) epsilon))))]
    (->AdamState pos-new m-new v-new t-new)))

;; ---------------------------------------------------------------------------
;; Single MAP step — for custom optimization loops
;; ---------------------------------------------------------------------------

(defn map-step
  "Single MAP optimization step. Takes a pre-built value-and-grad function,
   current AdamState, and optimizer options.

   Returns {:state AdamState, :log-density double}."
  [vag-fn state opts]
  (let [{:keys [value grads]} (vag-fn (.position state))
        new-state (adam-step (first grads) state opts)]
    {:state new-state
     :log-density (arr/->double value)}))

;; ---------------------------------------------------------------------------
;; MAP optimization — full run
;; ---------------------------------------------------------------------------

(defn map-optimize
  "Find MAP estimate by gradient ascent with Adam optimizer.

   `log-density-fn` takes an MLXArray position and returns a scalar MLXArray.
   `initial-position` is an MLXArray (1-d vector of parameters).

   Options:
     :lr       — learning rate (default 0.01)
     :beta1    — first moment decay (default 0.9)
     :beta2    — second moment decay (default 0.999)
     :epsilon  — numerical stability (default 1e-8)
     :tol      — convergence tolerance (default nil, no early stopping)
     :patience — consecutive steps within tol before stopping (default 5)

   Returns {:position MLXArray, :log-density double, :history [{:step :log-density}]}."
  [log-density-fn initial-position n-steps
   & {:keys [lr beta1 beta2 epsilon tol patience]
      :or {lr 0.01 beta1 0.9 beta2 0.999 epsilon 1e-8 patience 5}}]
  (let [vag-fn (xforms/value-and-grad log-density-fn)
        opts   {:lr lr :beta1 beta1 :beta2 beta2 :epsilon epsilon}
        init   (make-adam-state initial-position)]
    (loop [i          0
           state      init
           best-pos   initial-position
           best-ld    Double/NEGATIVE_INFINITY
           history    []
           converged  0
           prev-ld    Double/NEGATIVE_INFINITY]
      (if (>= i n-steps)
        {:position best-pos :log-density best-ld :history history}
        (let [{:keys [state log-density]} (map-step vag-fn state opts)
              entry    {:step i :log-density log-density}
              ;; Track best position
              better?  (> log-density best-ld)
              new-best-pos (if better? (.position state) best-pos)
              new-best-ld  (if better? log-density best-ld)
              ;; Convergence check
              new-converged (if (and tol
                                     (< (Math/abs (- log-density prev-ld)) (double tol)))
                              (inc converged)
                              0)]
          (if (and tol (>= new-converged patience))
            {:position new-best-pos :log-density new-best-ld
             :history (conj history entry)}
            (recur (inc i) state new-best-pos new-best-ld
                   (conj history entry) new-converged log-density)))))))
