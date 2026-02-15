(ns gen.mlx.dynamic
  "MLX-backed dynamic DSL for Gen.clj.

   Replaces atoms with volatile! for single-threaded semantics and records
   addr-info (distribution + params at each address) so the model's score
   computation can be replayed as a pure MLX function for autodiff.

   Implements IChoiceGradients via value-and-grad on the replayed score,
   and provides an HMC bridge for gradient-based inference.

   The score function uses vectorized MLX operations: choice values are
   packed into a single MLXArray vector, and all logpdfs are computed in
   one element-wise gaussian-logpdf call + arr/sum. This batches ~6N
   FFI round-trips down to ~6, with value-and-grad differentiating
   1 vector arg instead of N scalars."
  (:require [clojure.walk :as walk]
            [gen.choicemap :as choicemap]
            [gen.distribution :as d]
            [gen.dynamic :as dynamic]
            [gen.generative-function :as gf]
            [gen.mlx.array :as arr]
            [gen.mlx.distribution :as mlx-dist]
            [gen.mlx.hmc :as hmc]
            [gen.mlx.map-optimize :as map-opt]
            [gen.mlx.nuts :as nuts]
            [gen.mlx.transforms :as xforms]
            [gen.trace :as trace]))

;; ---------------------------------------------------------------------------
;; MLXTrace — forward declaration for use in defrecord
;; ---------------------------------------------------------------------------

(declare ->MLXTrace)

;; ---------------------------------------------------------------------------
;; MLXDynamicDSLFunction + apply-inner (defined before build-score-fn)
;; ---------------------------------------------------------------------------

(declare apply-inner)

(defrecord MLXDynamicDSLFunction [clojure-fn]
  gf/IGenerativeFunction
  (has-argument-grads [_] [])
  (accepts-output-grad? [_] false)
  (get-params [_] ())
  (simulate [gf args]
    (let [!choices (volatile! {})
          !score   (volatile! 0.0)
          !addrs   (volatile! [])]
      (binding [dynamic/*trace*
                (fn
                  ([inner-gf inner-args]
                   (apply-inner inner-gf inner-args))
                  ([k dist-gf dist-args]
                   (let [sub-trace (gf/simulate dist-gf dist-args)
                         val       (trace/get-retval sub-trace)]
                     (vswap! !choices assoc k val)
                     (vswap! !score + (trace/get-score sub-trace))
                     (vswap! !addrs conj {:addr k :dist-gf dist-gf
                                          :dist-args (vec dist-args)})
                     val)))]
        (let [retval (apply clojure-fn args)]
          (->MLXTrace gf @!choices @!score args retval @!addrs))))))

(defn- apply-inner [^MLXDynamicDSLFunction gf args]
  (apply (.-clojure-fn gf) args))

(extend-type MLXDynamicDSLFunction
  gf/IGenerate
  (-generate [gf args constraints]
    (let [!choices (volatile! {})
          !score   (volatile! 0.0)
          !weight  (volatile! 0.0)
          !addrs   (volatile! [])]
      (binding [dynamic/*trace*
                (fn
                  ([inner-gf inner-args]
                   (apply-inner inner-gf inner-args))
                  ([k dist-gf dist-args]
                   (let [k-constraint (choicemap/get-submap constraints k)]
                     (if (choicemap/has-value? k-constraint)
                       ;; Constrained: use provided value
                       (let [val    (choicemap/get-value k-constraint)
                             dist   (apply (:ctor dist-gf) dist-args)
                             lp     (d/logpdf dist val)]
                         (vswap! !choices assoc k val)
                         (vswap! !score + lp)
                         (vswap! !weight + lp)
                         (vswap! !addrs conj {:addr k :dist-gf dist-gf
                                              :dist-args (vec dist-args)})
                         val)
                       ;; Unconstrained: sample
                       (let [sub-trace (gf/simulate dist-gf dist-args)
                             val       (trace/get-retval sub-trace)]
                         (vswap! !choices assoc k val)
                         (vswap! !score + (trace/get-score sub-trace))
                         (vswap! !addrs conj {:addr k :dist-gf dist-gf
                                              :dist-args (vec dist-args)})
                         val)))))]
        (let [retval (apply (.-clojure-fn gf) args)]
          {:trace  (->MLXTrace gf @!choices @!score args retval @!addrs)
           :weight @!weight})))))

;; ---------------------------------------------------------------------------
;; MLXTrace
;; ---------------------------------------------------------------------------

(deftype MLXTrace [gen-fn choices score args retval addr-info]
  trace/ITrace
  (get-args [_] args)
  (get-retval [_] retval)
  (get-gen-fn [_] gen-fn)
  (get-choices [_]
    (choicemap/map->choicemap
     (persistent!
      (reduce-kv (fn [acc k v]
                   (assoc! acc k (choicemap/->Choice v)))
                 (transient {})
                 choices))))
  (get-score [_] score))

;; ---------------------------------------------------------------------------
;; build-score-fn — vectorized two-phase replay
;;
;; Phase 1 (JVM): replay model with JVM doubles extracted from input vector
;;   to collect distribution parameters + distribution type at each address.
;; Phase 2 (MLX): group by distribution type, vectorized logpdf per group.
;;
;; O(D) groups of FFI calls where D = number of distinct distribution types.
;; Fast path: when all addresses use same distribution, skip masking.
;; ---------------------------------------------------------------------------

(defn- build-dist-instance
  "Build a distribution instance from a dist-gf and args, used for mlx-logpdf."
  [dist-gf dist-args]
  (apply (:ctor dist-gf) dist-args))

(defn build-score-fn
  "Build a pure function (MLXArray -> MLXArray) that computes the total
   log-probability of all choices. Takes a single MLXArray vector of
   choice values (ordered by sorted-addrs) and returns a scalar MLXArray.

   Phase 1 replays the model in JVM to collect dist params and types.
   Phase 2 groups by distribution type and computes vectorized logpdfs."
  [gf model-args sorted-addrs]
  (fn [values-arr]
    ;; Phase 1: extract JVM doubles, replay model to collect dist params + types
    (let [jvm-vals  (arr/->vec values-arr)
          addr->jvm (zipmap sorted-addrs jvm-vals)
          !entries  (volatile! [])]
      (binding [dynamic/*trace*
                (fn
                  ([inner-gf inner-args]
                   (apply (.-clojure-fn ^MLXDynamicDSLFunction inner-gf)
                          inner-args))
                  ([k dist-gf dist-args]
                   (vswap! !entries conj {:dist-gf dist-gf
                                          :dist-args (vec dist-args)})
                   (get addr->jvm k)))]
        (apply (.-clojure-fn ^MLXDynamicDSLFunction gf) model-args))
      ;; Phase 2: group by distribution type, vectorized logpdf per group
      (let [entries @!entries
            n       (count entries)
            tags    (mapv (fn [{:keys [dist-gf dist-args]}]
                            (let [dist (build-dist-instance dist-gf dist-args)]
                              (if (satisfies? mlx-dist/IMLXLogPDF dist)
                                (mlx-dist/dist-tag dist)
                                :fallback)))
                          entries)
            distinct-tags (distinct tags)]
        (if (and (= 1 (count distinct-tags))
                 (not= :fallback (first distinct-tags)))
          ;; Fast path: all same distribution type, no masking needed
          (let [tag       (first distinct-tags)
                dist0     (build-dist-instance (:dist-gf (first entries))
                                               (:dist-args (first entries)))
                _         (assert (satisfies? mlx-dist/IMLXLogPDF dist0))
                ;; For same-type distributions, build param vectors and
                ;; call the vectorized logpdf from the first instance's type
                ;; We need to build dist instances for each entry to get their params
                dists     (mapv (fn [{:keys [dist-gf dist-args]}]
                                 (build-dist-instance dist-gf dist-args))
                               entries)]
            ;; Dispatch by tag — each type knows how to vectorize itself
            (case tag
              :normal
              (let [mus    (arr/from-vec (mapv #(double (:mu %)) dists))
                    sigmas (arr/from-vec (mapv #(double (:sigma %)) dists))]
                (arr/sum (mlx-dist/gaussian-logpdf mus sigmas values-arr)))

              :exponential
              (let [rates (arr/from-vec (mapv #(double (:rate %)) dists))]
                (arr/sum (mlx-dist/exponential-logpdf rates values-arr)))

              :uniform
              (let [los (arr/from-vec (mapv #(double (:lo %)) dists))
                    his (arr/from-vec (mapv #(double (:hi %)) dists))]
                (arr/sum (mlx-dist/uniform-logpdf los his values-arr)))

              :laplace
              (let [locs   (arr/from-vec (mapv #(double (:location %)) dists))
                    scales (arr/from-vec (mapv #(double (:scale %)) dists))]
                (arr/sum (mlx-dist/laplace-logpdf locs scales values-arr)))

              :cauchy
              (let [locs   (arr/from-vec (mapv #(double (:location %)) dists))
                    scales (arr/from-vec (mapv #(double (:scale %)) dists))]
                (arr/sum (mlx-dist/cauchy-logpdf locs scales values-arr)))))

          ;; General path: group by type, mask per group, sum
          (reduce
           (fn [total tag]
             (let [mask-vec   (mapv (fn [t] (if (= t tag) 1.0 0.0)) tags)
                   mask-arr   (arr/from-vec mask-vec)
                   group-idxs (keep-indexed (fn [i t] (when (= t tag) i)) tags)]
               (if (= tag :fallback)
                 ;; Fallback: compute scalar logpdfs and sum
                 (let [fallback-sum
                       (reduce
                        (fn [acc i]
                          (let [{:keys [dist-gf dist-args]} (nth entries i)
                                dist (build-dist-instance dist-gf dist-args)
                                v    (nth jvm-vals i)]
                            (+ acc (double (d/logpdf dist v)))))
                        0.0
                        group-idxs)]
                   (arr/add total (arr/scalar fallback-sum)))
                 ;; Vectorized path for this distribution type
                 (let [dists (mapv (fn [i]
                                    (let [{:keys [dist-gf dist-args]} (nth entries i)]
                                      (build-dist-instance dist-gf dist-args)))
                                  group-idxs)
                       ;; Build full-size param vectors with safe defaults at non-group positions
                       sample-dist (first dists)]
                   (case tag
                     :normal
                     (let [mus    (arr/from-vec (mapv (fn [i t]
                                                       (if (= t tag)
                                                         (double (:mu (build-dist-instance
                                                                       (:dist-gf (nth entries i))
                                                                       (:dist-args (nth entries i)))))
                                                         0.0))
                                                     (range n) tags))
                           sigmas (arr/from-vec (mapv (fn [i t]
                                                       (if (= t tag)
                                                         (double (:sigma (build-dist-instance
                                                                          (:dist-gf (nth entries i))
                                                                          (:dist-args (nth entries i)))))
                                                         1.0))
                                                     (range n) tags))]
                       (arr/add total (arr/sum (arr/mul mask-arr
                                                       (mlx-dist/gaussian-logpdf mus sigmas values-arr)))))

                     :exponential
                     (let [rates (arr/from-vec (mapv (fn [i t]
                                                      (if (= t tag)
                                                        (double (:rate (build-dist-instance
                                                                        (:dist-gf (nth entries i))
                                                                        (:dist-args (nth entries i)))))
                                                        1.0))
                                                    (range n) tags))]
                       (arr/add total (arr/sum (arr/mul mask-arr
                                                       (mlx-dist/exponential-logpdf rates values-arr)))))

                     :uniform
                     (let [los (arr/from-vec (mapv (fn [i t]
                                                    (if (= t tag)
                                                      (double (:lo (build-dist-instance
                                                                     (:dist-gf (nth entries i))
                                                                     (:dist-args (nth entries i)))))
                                                      0.0))
                                                  (range n) tags))
                           his (arr/from-vec (mapv (fn [i t]
                                                    (if (= t tag)
                                                      (double (:hi (build-dist-instance
                                                                     (:dist-gf (nth entries i))
                                                                     (:dist-args (nth entries i)))))
                                                      1.0))
                                                  (range n) tags))]
                       (arr/add total (arr/sum (arr/mul mask-arr
                                                       (mlx-dist/uniform-logpdf los his values-arr)))))

                     :laplace
                     (let [locs (arr/from-vec (mapv (fn [i t]
                                                     (if (= t tag)
                                                       (double (:location (build-dist-instance
                                                                            (:dist-gf (nth entries i))
                                                                            (:dist-args (nth entries i)))))
                                                       0.0))
                                                   (range n) tags))
                           scales (arr/from-vec (mapv (fn [i t]
                                                       (if (= t tag)
                                                         (double (:scale (build-dist-instance
                                                                           (:dist-gf (nth entries i))
                                                                           (:dist-args (nth entries i)))))
                                                         1.0))
                                                     (range n) tags))]
                       (arr/add total (arr/sum (arr/mul mask-arr
                                                       (mlx-dist/laplace-logpdf locs scales values-arr)))))

                     :cauchy
                     (let [locs (arr/from-vec (mapv (fn [i t]
                                                     (if (= t tag)
                                                       (double (:location (build-dist-instance
                                                                            (:dist-gf (nth entries i))
                                                                            (:dist-args (nth entries i)))))
                                                       0.0))
                                                   (range n) tags))
                           scales (arr/from-vec (mapv (fn [i t]
                                                       (if (= t tag)
                                                         (double (:scale (build-dist-instance
                                                                           (:dist-gf (nth entries i))
                                                                           (:dist-args (nth entries i)))))
                                                         1.0))
                                                     (range n) tags))]
                       (arr/add total (arr/sum (arr/mul mask-arr
                                                       (mlx-dist/cauchy-logpdf locs scales values-arr))))))))))
           (arr/scalar 0.0)
           distinct-tags))))))

;; ---------------------------------------------------------------------------
;; IChoiceGradients
;; ---------------------------------------------------------------------------

(extend-type MLXTrace
  trace/IChoiceGradients
  (choice-gradients [trace _selection _retgrad]
    (let [gf         (.-gen-fn trace)
          model-args (.-args trace)
          info       (.-addr-info trace)
          choices    (.-choices trace)
          sorted-addrs (mapv :addr info)
          score-fn   (build-score-fn gf model-args sorted-addrs)
          vag-fn     (xforms/value-and-grad score-fn)
          values-arr (arr/from-vec (mapv #(double (get choices %)) sorted-addrs))
          result     (vag-fn values-arr)
          grad-vec   (arr/->vec (first (:grads result)))]
      {:arg-grads     nil
       :choice-values (choicemap/map->choicemap
                       (zipmap sorted-addrs
                               (mapv #(choicemap/->Choice (get choices %))
                                     sorted-addrs)))
       :choice-grads  (choicemap/map->choicemap
                       (zipmap sorted-addrs
                               (mapv choicemap/->Choice grad-vec)))})))

;; ---------------------------------------------------------------------------
;; gen macro
;; ---------------------------------------------------------------------------

(defn ^:no-doc gen-body [& xs]
  (let [name (when (simple-symbol? (first xs))
               (first xs))
        [params & body] (if name (rest xs) xs)]
    `(-> (fn ~@(when name [name])
           ~params
           ~@(walk/postwalk
              (fn [form]
                (cond (dynamic/trace-form? form)
                      (let [[addr gf & xs] (rest form)]
                        `((dynamic/active-trace) ~addr ~gf [~@xs]))

                      (dynamic/splice-form? form)
                      (let [[gf & xs] (rest form)]
                        `((dynamic/active-trace) ~gf [~@xs]))

                      :else form))
              body))
         (->MLXDynamicDSLFunction))))

(defmacro gen
  "Defines an MLX-backed generative function."
  [& args]
  {:clj-kondo/lint-as 'clojure.core/fn}
  (apply gen-body args))

;; ---------------------------------------------------------------------------
;; HMC bridge — delegates to standalone gen.mlx.hmc
;; ---------------------------------------------------------------------------

(defn hmc-sample
  "Run HMC on the choices of an MLXTrace.

   Builds a vectorized score function from the trace and delegates to
   gen.mlx.hmc/sample for leapfrog integration and MH accept/reject.

   Returns a vector of n-steps result maps, each containing:
     :position    - MLXArray vector
     :log-density - double
     :accepted?   - boolean
     :trace       - new MLXTrace with updated choices
     :choices     - {addr -> double}"
  [^MLXTrace trace n-steps & {:keys [L eps] :or {L 10 eps 0.01}}]
  (let [gf           (.-gen-fn trace)
        model-args   (.-args trace)
        info         (.-addr-info trace)
        choices      (.-choices trace)
        sorted-addrs (mapv :addr info)
        score-fn     (build-score-fn gf model-args sorted-addrs)
        init-pos     (arr/from-vec (mapv #(double (get choices %)) sorted-addrs))
        raw-samples  (hmc/sample score-fn init-pos n-steps :L L :eps eps)]
    (mapv (fn [sample]
            (let [pos-vec     (arr/->vec (:position sample))
                  new-choices (zipmap sorted-addrs pos-vec)]
              (assoc sample
                     :choices new-choices
                     :trace   (->MLXTrace gf new-choices (:log-density sample)
                                          model-args (.-retval trace) info))))
          raw-samples)))

;; ---------------------------------------------------------------------------
;; MALA bridge — HMC with L=1
;; ---------------------------------------------------------------------------

(defn mala-sample
  "Run MALA on the choices of an MLXTrace.

   MALA is HMC with a single leapfrog step (L=1). Simpler than HMC
   but can be effective for well-conditioned targets.

   Returns a vector of n-steps result maps, each containing:
     :position    - MLXArray vector
     :log-density - double
     :accepted?   - boolean
     :trace       - new MLXTrace with updated choices
     :choices     - {addr -> double}"
  [^MLXTrace trace n-steps & {:keys [eps] :or {eps 0.01}}]
  (hmc-sample trace n-steps :L 1 :eps eps))

;; ---------------------------------------------------------------------------
;; MAP optimization bridge
;; ---------------------------------------------------------------------------

(defn map-optimize
  "Find MAP estimate for the choices of an MLXTrace.

   Uses Adam optimizer with MLX autodiff for gradients.

   Options:
     :lr       — learning rate (default 0.01)
     :beta1    — first moment decay (default 0.9)
     :beta2    — second moment decay (default 0.999)
     :epsilon  — numerical stability (default 1e-8)
     :tol      — convergence tolerance (default nil)
     :patience — consecutive steps within tol before stopping (default 5)

   Returns {:position MLXArray, :log-density double,
            :choices {addr -> double}, :trace MLXTrace,
            :history [{:step :log-density}]}."
  [^MLXTrace trace n-steps & {:keys [lr beta1 beta2 epsilon tol patience]
                               :or {lr 0.01 beta1 0.9 beta2 0.999
                                    epsilon 1e-8 patience 5}
                               :as opts}]
  (let [gf           (.-gen-fn trace)
        model-args   (.-args trace)
        info         (.-addr-info trace)
        choices      (.-choices trace)
        sorted-addrs (mapv :addr info)
        score-fn     (build-score-fn gf model-args sorted-addrs)
        init-pos     (arr/from-vec (mapv #(double (get choices %)) sorted-addrs))
        result       (map-opt/map-optimize score-fn init-pos n-steps
                                           :lr lr :beta1 beta1 :beta2 beta2
                                           :epsilon epsilon :tol tol
                                           :patience patience)
        pos-vec      (arr/->vec (:position result))
        new-choices  (zipmap sorted-addrs pos-vec)]
    (assoc result
           :choices new-choices
           :trace   (->MLXTrace gf new-choices (:log-density result)
                                model-args (.-retval trace) info))))

;; ---------------------------------------------------------------------------
;; NUTS bridge
;; ---------------------------------------------------------------------------

(defn nuts-sample
  "Run NUTS on the choices of an MLXTrace.

   Automatically tunes step size during warmup via dual averaging.

   Options:
     :max-depth     — maximum tree depth (default 10)
     :target-accept — target acceptance probability (default 0.8)
     :n-warmup      — warmup steps for adaptation (default n-steps/2)
     :initial-eps   — initial step size (default: auto-detected)

   Returns a vector of n-steps result maps, each containing:
     :position    — MLXArray
     :log-density — double
     :tree-depth  — int
     :n-leapfrog  — int
     :eps         — step size used
     :accept-prob — acceptance probability
     :trace       — new MLXTrace with updated choices
     :choices     — {addr -> double}"
  [^MLXTrace trace n-steps & {:keys [max-depth target-accept n-warmup initial-eps]
                               :or {max-depth 10 target-accept 0.8}}]
  (let [gf           (.-gen-fn trace)
        model-args   (.-args trace)
        info         (.-addr-info trace)
        choices      (.-choices trace)
        sorted-addrs (mapv :addr info)
        score-fn     (build-score-fn gf model-args sorted-addrs)
        init-pos     (arr/from-vec (mapv #(double (get choices %)) sorted-addrs))
        raw-samples  (nuts/sample score-fn init-pos n-steps
                                  :max-depth max-depth
                                  :target-accept target-accept
                                  :n-warmup n-warmup
                                  :initial-eps initial-eps)]
    (mapv (fn [sample]
            (let [pos-vec     (arr/->vec (:position sample))
                  new-choices (zipmap sorted-addrs pos-vec)]
              (assoc sample
                     :choices new-choices
                     :trace   (->MLXTrace gf new-choices (:log-density sample)
                                          model-args (.-retval trace) info))))
          raw-samples)))
