(ns gen.mlx.nuts
  "No-U-Turn Sampler (NUTS) — adaptive HMC that automatically tunes
   trajectory length and step size.

   Implements Algorithm 6 from Hoffman & Gelman (2014) with multinomial
   sampling from the trajectory and dual averaging for step size adaptation.

   All tree-building logic runs on the JVM. MLX is only invoked for
   leapfrog steps (position/momentum updates + gradient computation).

   Usage:
     (def log-density (fn [x] (arr/mul -0.5 (arr/sum (arr/square x)))))
     (sample log-density (arr/from-vec [0.0]) 1000)
     ;; => vector of {:position :log-density :tree-depth :n-leapfrog :eps}"
  (:require [gen.mlx.array :as arr]
            [gen.mlx.hmc :as hmc]
            [gen.mlx.transforms :as xforms])
  (:import [java.util.concurrent ThreadLocalRandom]))

;; ---------------------------------------------------------------------------
;; Helpers
;; ---------------------------------------------------------------------------

(defn- sample-momentum
  "Sample momentum from N(0,I) of dimension n."
  [n]
  (arr/random-normal [n]))

(defn- kinetic-energy
  "Kinetic energy: 0.5 * sum(p^2). Returns a double."
  [momentum]
  (* 0.5 (arr/->double (arr/sum (arr/square momentum)))))

(defn- joint-log-prob
  "Joint log-probability: log-density - kinetic-energy."
  [log-density momentum]
  (- log-density (kinetic-energy momentum)))

(defn- log-sum-exp
  "Numerically stable log(exp(a) + exp(b))."
  [a b]
  (let [mx (max a b)]
    (if (Double/isInfinite mx)
      mx
      (+ mx (Math/log (+ (Math/exp (- a mx)) (Math/exp (- b mx))))))))

;; ---------------------------------------------------------------------------
;; U-turn detection
;; ---------------------------------------------------------------------------

(defn- u-turn?
  "Check the generalized U-turn criterion (NUTS).
   Returns true if the trajectory should stop extending."
  [theta-minus theta-plus r-minus r-plus]
  (let [delta (arr/sub theta-plus theta-minus)
        fwd (arr/->double (arr/sum (arr/mul delta r-plus)))
        bwd (arr/->double (arr/sum (arr/mul delta r-minus)))]
    (or (neg? fwd) (neg? bwd))))

;; ---------------------------------------------------------------------------
;; Tree building (Algorithm 6 from Hoffman & Gelman 2014)
;; ---------------------------------------------------------------------------

(defn- build-tree
  "Recursively build a balanced binary tree of leapfrog steps.

   Returns a map:
     :theta-minus, :theta-plus  — endpoints of the trajectory
     :r-minus, :r-plus          — momenta at endpoints
     :theta-prime               — proposed position (sampled from trajectory)
     :log-weight                — log of sum of weights for multinomial sampling
     :n-valid                   — count of valid states in the tree
     :n-leapfrog                — number of leapfrog steps taken
     :diverging?                — whether a divergence was detected
     :turning?                  — whether a U-turn was detected
     :sum-accept-prob           — sum of acceptance probabilities (for adaptation)
     :n-accept                  — count for acceptance probability averaging"
  [vag-fn theta r log-u direction depth eps joint0]
  (let [rng (ThreadLocalRandom/current)]
    (if (zero? depth)
      ;; Base case: single leapfrog step
      (let [{:keys [position momentum log-density]}
            (hmc/leapfrog vag-fn theta r (* (double direction) eps) 1)
            joint (joint-log-prob log-density momentum)
            valid? (>= joint (- log-u 1000.0))
            diverging? (> (- joint0 joint) 1000.0)
            ;; Metropolis acceptance probability (clamped to 1)
            accept-prob (min 1.0 (Math/exp (- joint joint0)))]
        {:theta-minus position :theta-plus position
         :r-minus momentum :r-plus momentum
         :theta-prime position :log-weight joint
         :n-valid (if valid? 1 0)
         :n-leapfrog 1
         :diverging? diverging? :turning? false
         :sum-accept-prob accept-prob :n-accept 1})
      ;; Recursive case: build two half-trees
      (let [left (build-tree vag-fn theta r log-u direction (dec depth) eps joint0)]
        (if (or (:diverging? left) (:turning? left))
          left
          (let [;; Continue from the appropriate end
                [cont-theta cont-r]
                (if (pos? direction)
                  [(:theta-plus left) (:r-plus left)]
                  [(:theta-minus left) (:r-minus left)])
                right (build-tree vag-fn cont-theta cont-r log-u direction (dec depth) eps joint0)]
            (if (or (:diverging? right) (:turning? right))
              ;; Right subtree terminated — merge what we can
              (assoc right
                     :theta-minus (if (pos? direction) (:theta-minus left) (:theta-minus right))
                     :theta-plus  (if (pos? direction) (:theta-plus right) (:theta-plus left))
                     :r-minus     (if (pos? direction) (:r-minus left) (:r-minus right))
                     :r-plus      (if (pos? direction) (:r-plus right) (:r-plus left))
                     :n-leapfrog  (+ (:n-leapfrog left) (:n-leapfrog right))
                     :sum-accept-prob (+ (:sum-accept-prob left) (:sum-accept-prob right))
                     :n-accept (+ (:n-accept left) (:n-accept right)))
              ;; Both subtrees OK — combine
              (let [log-weight-combined (log-sum-exp (:log-weight left) (:log-weight right))
                    ;; Multinomial sampling: accept right proposal with probability
                    ;; exp(log-weight-right - log-weight-combined)
                    accept-right? (< (Math/log (.nextDouble rng))
                                     (- (:log-weight right) log-weight-combined))
                    theta-prime (if accept-right?
                                  (:theta-prime right)
                                  (:theta-prime left))
                    ;; Merge endpoints
                    theta-minus (if (pos? direction) (:theta-minus left) (:theta-minus right))
                    theta-plus  (if (pos? direction) (:theta-plus right) (:theta-plus left))
                    r-minus     (if (pos? direction) (:r-minus left) (:r-minus right))
                    r-plus      (if (pos? direction) (:r-plus right) (:r-plus left))
                    ;; Check U-turn on the combined tree
                    turning? (u-turn? theta-minus theta-plus r-minus r-plus)]
                {:theta-minus theta-minus :theta-plus theta-plus
                 :r-minus r-minus :r-plus r-plus
                 :theta-prime theta-prime
                 :log-weight log-weight-combined
                 :n-valid (+ (:n-valid left) (:n-valid right))
                 :n-leapfrog (+ (:n-leapfrog left) (:n-leapfrog right))
                 :diverging? false :turning? turning?
                 :sum-accept-prob (+ (:sum-accept-prob left) (:sum-accept-prob right))
                 :n-accept (+ (:n-accept left) (:n-accept right))}))))))))

;; ---------------------------------------------------------------------------
;; Single NUTS step
;; ---------------------------------------------------------------------------

(defn nuts-step
  "Perform a single NUTS step with given step size.

   `vag-fn` is a pre-built value-and-grad function.
   `position` is an MLXArray.
   `eps` is the step size.
   `max-depth` is the maximum tree depth.

   Returns {:position :log-density :tree-depth :n-leapfrog :accept-prob}."
  [vag-fn position current-log-density eps max-depth]
  (let [n (arr/size position)
        rng (ThreadLocalRandom/current)
        ;; Sample momentum
        momentum (sample-momentum n)
        joint0 (joint-log-prob current-log-density momentum)
        ;; Slice variable
        log-u (- joint0 (- (Math/log (.nextDouble rng))))]
    (loop [depth 0
           theta-minus position
           theta-plus position
           r-minus momentum
           r-plus momentum
           theta-prime position
           log-weight joint0
           n-leapfrog 0
           sum-accept-prob 0.0
           n-accept 0]
      (if (>= depth max-depth)
        ;; Reached max depth
        {:position theta-prime
         :log-density (arr/->double ((comp :value vag-fn) theta-prime))
         :tree-depth depth
         :n-leapfrog n-leapfrog
         :accept-prob (if (pos? n-accept) (/ sum-accept-prob n-accept) 0.0)}
        ;; Randomly choose direction
        (let [direction (if (.nextBoolean rng) 1 -1)
              ;; Build tree in chosen direction
              [cont-theta cont-r]
              (if (pos? direction)
                [theta-plus r-plus]
                [theta-minus r-minus])
              tree (build-tree vag-fn cont-theta cont-r log-u direction depth eps joint0)]
          (if (or (:diverging? tree) (:turning? tree))
            ;; Tree terminated
            (let [;; Still consider proposal from this tree via multinomial
                  accept-new? (and (not (:diverging? tree))
                                   (< (Math/log (.nextDouble rng))
                                      (- (:log-weight tree) log-weight)))
                  final-theta (if accept-new? (:theta-prime tree) theta-prime)]
              {:position final-theta
               :log-density (arr/->double ((comp :value vag-fn) final-theta))
               :tree-depth depth
               :n-leapfrog (+ n-leapfrog (:n-leapfrog tree))
               :accept-prob (let [total-n (+ n-accept (:n-accept tree))]
                              (if (pos? total-n)
                                (/ (+ sum-accept-prob (:sum-accept-prob tree)) total-n)
                                0.0))})
            ;; Tree OK — combine and continue
            (let [log-weight-new (log-sum-exp log-weight (:log-weight tree))
                  accept-new? (< (Math/log (.nextDouble rng))
                                 (- (:log-weight tree) log-weight-new))
                  new-theta-prime (if accept-new? (:theta-prime tree) theta-prime)
                  ;; Update endpoints
                  new-theta-minus (if (pos? direction) theta-minus (:theta-minus tree))
                  new-theta-plus  (if (pos? direction) (:theta-plus tree) theta-plus)
                  new-r-minus     (if (pos? direction) r-minus (:r-minus tree))
                  new-r-plus      (if (pos? direction) (:r-plus tree) r-plus)
                  ;; Check U-turn on full trajectory
                  turning? (u-turn? new-theta-minus new-theta-plus
                                    new-r-minus new-r-plus)]
              (if turning?
                {:position new-theta-prime
                 :log-density (arr/->double ((comp :value vag-fn) new-theta-prime))
                 :tree-depth (inc depth)
                 :n-leapfrog (+ n-leapfrog (:n-leapfrog tree))
                 :accept-prob (let [total-n (+ n-accept (:n-accept tree))]
                                (if (pos? total-n)
                                  (/ (+ sum-accept-prob (:sum-accept-prob tree)) total-n)
                                  0.0))}
                (recur (inc depth)
                       new-theta-minus new-theta-plus
                       new-r-minus new-r-plus
                       new-theta-prime log-weight-new
                       (+ n-leapfrog (:n-leapfrog tree))
                       (+ sum-accept-prob (:sum-accept-prob tree))
                       (+ n-accept (:n-accept tree)))))))))))

;; ---------------------------------------------------------------------------
;; Dual averaging for step size adaptation (Nesterov 2009)
;; ---------------------------------------------------------------------------

(defrecord DualAverageState [log-eps log-eps-bar H-bar mu t gamma t0 kappa])

(defn- make-dual-average-state
  "Initialize dual averaging state."
  [initial-eps]
  (->DualAverageState (Math/log initial-eps)  ;; log-eps
                      0.0                      ;; log-eps-bar
                      0.0                      ;; H-bar
                      (Math/log (* 10.0 initial-eps))  ;; mu
                      0                        ;; t
                      0.05                     ;; gamma
                      10                       ;; t0
                      0.75))                   ;; kappa

(defn- dual-average-step
  "Update dual averaging state given observed acceptance probability."
  [^DualAverageState state accept-prob target-accept]
  (let [t-new (inc (.t state))
        ;; Running average of accept-prob error
        eta (/ 1.0 (+ t-new (.t0 state)))
        H-new (+ (* (- 1.0 eta) (.H-bar state))
                  (* eta (- target-accept accept-prob)))
        ;; Update log step size
        log-eps-new (- (.mu state)
                       (* (/ (Math/sqrt (double t-new)) (.gamma state)) H-new))
        ;; Smoothed step size (for final value)
        m-eta (Math/pow (double t-new) (- (.kappa state)))
        log-eps-bar-new (+ (* m-eta log-eps-new)
                           (* (- 1.0 m-eta) (.log-eps-bar state)))]
    (assoc state
           :log-eps log-eps-new
           :log-eps-bar log-eps-bar-new
           :H-bar H-new
           :t t-new)))

(defn- find-reasonable-eps
  "Heuristic to find a reasonable initial step size.
   Starts at 1.0 and doubles or halves until acceptance probability ≈ 0.5."
  [vag-fn position]
  (let [n (arr/size position)
        p (sample-momentum n)
        {:keys [value]} (vag-fn position)
        log-density (arr/->double value)
        joint0 (joint-log-prob log-density p)]
    (loop [eps 1.0
           i 0]
      (if (>= i 50)
        eps ;; give up, use current
        (let [{:keys [position momentum log-density]}
              (hmc/leapfrog vag-fn position p eps 1)
              joint1 (joint-log-prob log-density momentum)
              log-ratio (- joint1 joint0)]
          (if (Double/isNaN log-ratio)
            (recur (* eps 0.5) (inc i))
            (let [;; We want acceptance prob ≈ 0.5, i.e., log-ratio ≈ -0.693
                  too-big? (> log-ratio (Math/log 0.5))]
              (if (< (Math/abs (+ log-ratio (Math/log 2.0))) 1.0)
                eps ;; close enough
                (if too-big?
                  (recur (* eps 2.0) (inc i))
                  (recur (* eps 0.5) (inc i)))))))))))

;; ---------------------------------------------------------------------------
;; Full NUTS sampling with warmup adaptation
;; ---------------------------------------------------------------------------

(defn sample
  "Run NUTS for n-steps iterations with dual averaging step size adaptation.

   `log-density-fn` takes an MLXArray position and returns a scalar MLXArray.
   `initial-position` is an MLXArray (1-d vector).

   Options:
     :max-depth     — maximum tree depth (default 10, limits to 1024 leapfrog steps)
     :target-accept — target acceptance probability (default 0.8)
     :n-warmup      — number of warmup steps for adaptation (default n-steps/2)
     :initial-eps   — initial step size (default: auto-detected)

   Returns a vector of n-steps maps, each containing:
     :position    — MLXArray
     :log-density — double
     :tree-depth  — int
     :n-leapfrog  — int
     :eps         — step size used
     :accept-prob — acceptance probability"
  [log-density-fn initial-position n-steps
   & {:keys [max-depth target-accept n-warmup initial-eps]
      :or {max-depth 10 target-accept 0.8}}]
  (let [vag-fn    (xforms/value-and-grad log-density-fn)
        n-warmup  (or n-warmup (quot n-steps 2))
        ;; Find reasonable initial step size
        eps0      (or initial-eps (find-reasonable-eps vag-fn initial-position))
        da-state  (make-dual-average-state eps0)
        init-ld   (arr/->double (log-density-fn initial-position))]
    (loop [i         0
           pos       initial-position
           ld        init-ld
           da        da-state
           results   []]
      (if (>= i n-steps)
        results
        (let [;; Current step size
              eps (if (< i n-warmup)
                    (Math/exp (:log-eps da))
                    (Math/exp (:log-eps-bar da)))
              ;; NUTS step
              step (nuts-step vag-fn pos ld eps max-depth)
              ;; Update dual averaging during warmup
              new-da (if (< i n-warmup)
                       (dual-average-step da (:accept-prob step) target-accept)
                       da)
              result (assoc step :eps eps)]
          (recur (inc i)
                 (:position step)
                 (:log-density step)
                 new-da
                 (conj results result)))))))
