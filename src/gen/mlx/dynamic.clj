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
            [gen.distribution.math.gamma :as gamma]
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
;; Score function construction — vectorized logpdf computation
;;
;; Three layers:
;;   1. replay-model-once — run model in JVM to discover distribution structure
;;   2. precompute-param-arrays — build all MLX parameter arrays from structure
;;   3. compute-score-from-cache — pure MLX scoring using cached arrays
;;
;; build-score-fn: original per-call replay (retained for correctness baseline)
;; build-compiled-score-fn: one-time replay + cached arrays (fast path)
;; ---------------------------------------------------------------------------

(defn- build-dist-instance
  "Build a distribution instance from a dist-gf and args, used for mlx-logpdf."
  [dist-gf dist-args]
  (apply (:ctor dist-gf) dist-args))

(defn- replay-model-once
  "Replay the model once in JVM to discover distribution structure.
   Returns a vector of entries, each {:dist-gf :dist-args :tag :dist}."
  [gf model-args sorted-addrs initial-choices]
  (let [addr->jvm (if (map? initial-choices)
                    initial-choices
                    (zipmap sorted-addrs (repeat 0.0)))
        !entries  (volatile! [])]
    (binding [dynamic/*trace*
              (fn
                ([inner-gf inner-args]
                 (apply (.-clojure-fn ^MLXDynamicDSLFunction inner-gf)
                        inner-args))
                ([k dist-gf dist-args]
                 (let [dist (build-dist-instance dist-gf (vec dist-args))
                       tag  (if (satisfies? mlx-dist/IMLXLogPDF dist)
                              (mlx-dist/dist-tag dist)
                              :fallback)]
                   (vswap! !entries conj {:dist-gf   dist-gf
                                          :dist-args (vec dist-args)
                                          :tag       tag
                                          :dist      dist}))
                 (get addr->jvm k)))]
      (apply (.-clojure-fn ^MLXDynamicDSLFunction gf) model-args))
    @!entries))

(defn- extract-params-for-tag
  "Extract parameter arrays for a specific distribution tag from entries.
   Returns a map of param-name -> MLXArray.
   For the fast path (all same tag), entries are used directly.
   For the general path, builds full-size vectors with safe defaults at non-group positions."
  [tag entries tags n]
  (let [fast-path? (nil? tags) ;; when tags is nil, all entries are the same tag
        dists (if fast-path?
                (mapv :dist entries)
                (mapv (fn [i t]
                        (when (= t tag) (:dist (nth entries i))))
                      (range n) tags))]
    (case tag
      :normal
      (if fast-path?
        {:mus    (arr/from-vec (mapv #(double (:mu %)) dists))
         :sigmas (arr/from-vec (mapv #(double (:sigma %)) dists))}
        {:mus    (arr/from-vec (mapv (fn [d t] (if (= t tag) (double (:mu d)) 0.0)) dists tags))
         :sigmas (arr/from-vec (mapv (fn [d t] (if (= t tag) (double (:sigma d)) 1.0)) dists tags))
         :mask   (arr/from-vec (mapv (fn [t] (if (= t tag) 1.0 0.0)) tags))})

      :exponential
      (if fast-path?
        {:rates (arr/from-vec (mapv #(double (:rate %)) dists))}
        {:rates (arr/from-vec (mapv (fn [d t] (if (= t tag) (double (:rate d)) 1.0)) dists tags))
         :mask  (arr/from-vec (mapv (fn [t] (if (= t tag) 1.0 0.0)) tags))})

      :uniform
      (if fast-path?
        {:los (arr/from-vec (mapv #(double (:lo %)) dists))
         :his (arr/from-vec (mapv #(double (:hi %)) dists))}
        {:los  (arr/from-vec (mapv (fn [d t] (if (= t tag) (double (:lo d)) 0.0)) dists tags))
         :his  (arr/from-vec (mapv (fn [d t] (if (= t tag) (double (:hi d)) 1.0)) dists tags))
         :mask (arr/from-vec (mapv (fn [t] (if (= t tag) 1.0 0.0)) tags))})

      :laplace
      (if fast-path?
        {:locs   (arr/from-vec (mapv #(double (:location %)) dists))
         :scales (arr/from-vec (mapv #(double (:scale %)) dists))}
        {:locs   (arr/from-vec (mapv (fn [d t] (if (= t tag) (double (:location d)) 0.0)) dists tags))
         :scales (arr/from-vec (mapv (fn [d t] (if (= t tag) (double (:scale d)) 1.0)) dists tags))
         :mask   (arr/from-vec (mapv (fn [t] (if (= t tag) 1.0 0.0)) tags))})

      :cauchy
      (if fast-path?
        {:locs   (arr/from-vec (mapv #(double (:location %)) dists))
         :scales (arr/from-vec (mapv #(double (:scale %)) dists))}
        {:locs   (arr/from-vec (mapv (fn [d t] (if (= t tag) (double (:location d)) 0.0)) dists tags))
         :scales (arr/from-vec (mapv (fn [d t] (if (= t tag) (double (:scale d)) 1.0)) dists tags))
         :mask   (arr/from-vec (mapv (fn [t] (if (= t tag) 1.0 0.0)) tags))})

      :beta
      (if fast-path?
        {:alpha-m1s (arr/from-vec (mapv #(double (arr/->double (:alpha-m1-arr %))) dists))
         :beta-m1s  (arr/from-vec (mapv #(double (arr/->double (:beta-m1-arr %))) dists))
         :log-norms (arr/from-vec (mapv #(double (arr/->double (:log-norm-arr %))) dists))}
        {:alpha-m1s (arr/from-vec (mapv (fn [d t] (if (= t tag) (double (dec (:alpha d))) 0.0)) dists tags))
         :beta-m1s  (arr/from-vec (mapv (fn [d t] (if (= t tag) (double (dec (:beta d))) 0.0)) dists tags))
         :log-norms (arr/from-vec (mapv (fn [d t] (if (= t tag) (double (arr/->double (:log-norm-arr d))) 0.0)) dists tags))
         :mask      (arr/from-vec (mapv (fn [t] (if (= t tag) 1.0 0.0)) tags))})

      :gamma
      (if fast-path?
        {:shape-m1s  (arr/from-vec (mapv #(double (arr/->double (:shape-m1-arr %))) dists))
         :inv-scales (arr/from-vec (mapv #(double (arr/->double (:inv-scale-arr %))) dists))
         :log-norms  (arr/from-vec (mapv #(double (arr/->double (:log-norm-arr %))) dists))}
        {:shape-m1s  (arr/from-vec (mapv (fn [d t] (if (= t tag) (double (dec (:shape d))) 0.0)) dists tags))
         :inv-scales (arr/from-vec (mapv (fn [d t] (if (= t tag) (/ 1.0 (double (:scale d))) 1.0)) dists tags))
         :log-norms  (arr/from-vec (mapv (fn [d t] (if (= t tag) (double (arr/->double (:log-norm-arr d))) 0.0)) dists tags))
         :mask       (arr/from-vec (mapv (fn [t] (if (= t tag) 1.0 0.0)) tags))})

      :poisson
      (if fast-path?
        {:log-lambdas (arr/from-vec (mapv #(double (arr/->double (:log-lambda-arr %))) dists))
         :lambdas     (arr/from-vec (mapv #(double (:lambda %)) dists))}
        {:log-lambdas (arr/from-vec (mapv (fn [d t] (if (= t tag) (Math/log (double (:lambda d))) 0.0)) dists tags))
         :lambdas     (arr/from-vec (mapv (fn [d t] (if (= t tag) (double (:lambda d)) 0.0)) dists tags))
         :mask        (arr/from-vec (mapv (fn [t] (if (= t tag) 1.0 0.0)) tags))})

      ;; Fallback: no cached params, will be computed JVM-side
      nil)))

(defn- precompute-param-arrays
  "Build all parameter MLX arrays from discovered model structure.
   Returns {:tags :distinct-tags :fast-path? :param-cache :fallback-entries}."
  [entries]
  (let [n     (count entries)
        tags  (mapv :tag entries)
        distinct-tags (distinct tags)
        fast-path? (and (= 1 (count distinct-tags))
                        (not= :fallback (first distinct-tags)))]
    {:n             n
     :tags          tags
     :distinct-tags distinct-tags
     :fast-path?    fast-path?
     :param-cache   (into {}
                          (map (fn [tag]
                                 [tag (if fast-path?
                                        (extract-params-for-tag tag entries nil n)
                                        (extract-params-for-tag tag entries tags n))]))
                          distinct-tags)
     ;; For fallback entries, store the dist instances for JVM-side logpdf
     :fallback-entries (when (some #(= :fallback %) tags)
                         (vec (keep-indexed
                               (fn [i entry]
                                 (when (= :fallback (:tag entry))
                                   {:idx i :dist (:dist entry)}))
                               entries)))}))

(defn- score-tag-contribution
  "Compute the score contribution for a single distribution tag using cached params.
   Returns an MLXArray scalar (or (N,) when sum-fn is sum-axis for batched mode).
   Optional sum-fn overrides the reduction (default: arr/sum for scalar output,
   pass #(arr/sum-axis % 1) for per-chain (N,) output)."
  ([tag params values-arr fast-path?]
   (score-tag-contribution tag params values-arr fast-path? arr/sum))
  ([tag params values-arr fast-path? sum-fn]
   (case tag
     :normal
     (let [{:keys [mus sigmas mask]} params
           lps (mlx-dist/gaussian-logpdf mus sigmas values-arr)]
       (if fast-path? (sum-fn lps) (sum-fn (arr/mul mask lps))))

     :exponential
     (let [{:keys [rates mask]} params
           lps (mlx-dist/exponential-logpdf rates values-arr)]
       (if fast-path? (sum-fn lps) (sum-fn (arr/mul mask lps))))

     :uniform
     (let [{:keys [los his mask]} params
           lps (mlx-dist/uniform-logpdf los his values-arr)]
       (if fast-path? (sum-fn lps) (sum-fn (arr/mul mask lps))))

     :laplace
     (let [{:keys [locs scales mask]} params
           lps (mlx-dist/laplace-logpdf locs scales values-arr)]
       (if fast-path? (sum-fn lps) (sum-fn (arr/mul mask lps))))

     :cauchy
     (let [{:keys [locs scales mask]} params
           lps (mlx-dist/cauchy-logpdf locs scales values-arr)]
       (if fast-path? (sum-fn lps) (sum-fn (arr/mul mask lps))))

     :beta
     (let [{:keys [alpha-m1s beta-m1s log-norms mask]} params
           lps (mlx-dist/beta-logpdf alpha-m1s beta-m1s log-norms values-arr)]
       (if fast-path? (sum-fn lps) (sum-fn (arr/mul mask lps))))

     :gamma
     (let [{:keys [shape-m1s inv-scales log-norms mask]} params
           lps (mlx-dist/gamma-logpdf shape-m1s inv-scales log-norms values-arr)]
       (if fast-path? (sum-fn lps) (sum-fn (arr/mul mask lps))))

     :poisson
     (let [{:keys [log-lambdas lambdas mask]} params
           mlx-part (arr/sub (arr/mul values-arr log-lambdas) lambdas)
           ks (arr/->vec values-arr)
           lgamma-terms (arr/from-vec (mapv #(gamma/log-gamma (inc (double %))) ks))
           lps (arr/sub mlx-part lgamma-terms)]
       (if fast-path? (arr/sum lps) (arr/sum (arr/mul mask lps)))))))

(defn- compute-score-from-cache
  "Compute total log-probability using pre-cached parameter arrays.
   Pure MLX computation — no model replay, no arr/->vec extraction."
  [{:keys [distinct-tags fast-path? param-cache fallback-entries]} values-arr jvm-vals]
  (reduce
   (fn [total tag]
     (if (= tag :fallback)
       ;; Fallback: compute scalar logpdfs JVM-side
       (let [fallback-sum (reduce
                           (fn [acc {:keys [idx dist]}]
                             (+ acc (double (d/logpdf dist (nth jvm-vals idx)))))
                           0.0
                           fallback-entries)]
         (arr/add total (arr/scalar fallback-sum)))
       ;; Vectorized path using cached params
       (arr/add total (score-tag-contribution tag (get param-cache tag) values-arr fast-path?))))
   (arr/scalar 0.0)
   distinct-tags))

(defn build-score-fn
  "Build a function (MLXArray -> MLXArray) that computes total log-probability.
   Replays the model on EVERY call to collect distribution parameters.
   Use build-compiled-score-fn for the fast path that caches parameters."
  [gf model-args sorted-addrs]
  (fn [values-arr]
    ;; Phase 1: extract JVM doubles, replay model to collect dist params + types
    (let [jvm-vals  (arr/->vec values-arr)
          addr->jvm (zipmap sorted-addrs jvm-vals)
          entries   (replay-model-once gf model-args sorted-addrs addr->jvm)
          ;; Phase 2: build params and compute score
          cache     (precompute-param-arrays entries)]
      (compute-score-from-cache cache values-arr jvm-vals))))

(defn build-compiled-score-fn
  "Build a fast function (MLXArray -> MLXArray) that computes total log-probability.
   Replays the model ONCE at construction time to discover structure and cache
   all parameter arrays. Subsequent calls are pure MLX — no model replay,
   no arr/->vec extraction, no arr/from-vec rebuilding.

   IMPORTANT: Only valid when distribution parameters are constants (don't depend
   on other choice values). For models where parameters depend on choices
   (e.g. y ~ Normal(slope*x, 0.5) where slope is a choice), use build-score-fn
   or build-free-score-fn with selection."
  [gf model-args sorted-addrs initial-choices]
  (let [entries (replay-model-once gf model-args sorted-addrs initial-choices)
        cache   (precompute-param-arrays entries)
        has-fallback? (some #(= :fallback (:tag %)) entries)]
    (if has-fallback?
      ;; If there are fallback distributions, we need JVM values for those
      (fn [values-arr]
        (let [jvm-vals (arr/->vec values-arr)]
          (compute-score-from-cache cache values-arr jvm-vals)))
      ;; Pure MLX path — no JVM extraction needed
      (fn [values-arr]
        (compute-score-from-cache cache values-arr nil)))))

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
          score-fn   (build-compiled-score-fn gf model-args sorted-addrs choices)
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

(defn- params-depend-on-choices?
  "Check if any distribution parameters in the model depend on choice values.
   Replays model twice with different choice values and checks if parameters change.
   Returns true if the model has data-dependent parameters."
  [gf model-args sorted-addrs choices]
  (let [entries1 (replay-model-once gf model-args sorted-addrs choices)
        ;; Perturb all choice values slightly
        perturbed (reduce (fn [m addr] (assoc m addr (+ (double (get choices addr)) 0.1)))
                          {} sorted-addrs)
        entries2 (replay-model-once gf model-args sorted-addrs perturbed)]
    (not (every? true?
                 (map (fn [e1 e2] (= (:dist-args e1) (:dist-args e2)))
                      entries1 entries2)))))

(defn- build-free-score-fn
  "Build a score function for selection-aware inference.
   Only free-addrs are optimized; fixed-addrs contribute constant logpdf.

   When distribution parameters depend on free choices (data-dependent params),
   falls back to per-call model replay for the free portion.
   When parameters are constant, caches everything for pure MLX evaluation.

   Options:
     :force-constant — if true, always use cached constant params (no model replay).
                       Produces a pure-MLX function suitable for vmap. Used for
                       per-chain scoring in parallel HMC when the gradient-accurate
                       score function handles data-dependent params separately."
  [gf model-args free-addrs fixed-addrs choices
   & {:keys [force-constant] :or {force-constant false}}]
  (let [all-addrs (into (vec free-addrs) (mapv :addr fixed-addrs))
        ;; Compute constant logpdf contribution from fixed addresses
        fixed-score (when (seq fixed-addrs)
                      (let [entries (replay-model-once gf model-args all-addrs choices)
                            fixed-addr-set (set (mapv :addr fixed-addrs))]
                        (reduce
                         (fn [acc [entry addr]]
                           (let [dist (:dist entry)
                                 v    (get choices addr)]
                             (+ acc (double (d/logpdf dist v)))))
                         0.0
                         (filter (fn [[_ addr]] (contains? fixed-addr-set addr))
                                 (map vector entries all-addrs)))))
        fixed-score-arr (arr/scalar (or fixed-score 0.0))
        ;; Check if free portion has data-dependent params
        has-deps? (and (not force-constant)
                       (params-depend-on-choices?
                        gf model-args all-addrs choices))]
    (if has-deps?
      ;; Data-dependent params: must replay model each call, but only over free addrs
      (fn [values-arr]
        (let [jvm-vals  (arr/->vec values-arr)
              ;; Build choices map with free values from position, fixed from original
              addr->jvm (merge (reduce (fn [m {:keys [addr]}]
                                         (assoc m addr (get choices addr)))
                                       {} fixed-addrs)
                               (zipmap free-addrs jvm-vals))
              entries   (replay-model-once gf model-args all-addrs addr->jvm)
              ;; Only score the free addresses
              free-set  (set free-addrs)
              free-entries (vec (filter (fn [[_ addr]] (contains? free-set addr))
                                       (map vector entries all-addrs)))
              free-entry-vec (mapv first free-entries)
              cache     (precompute-param-arrays free-entry-vec)]
          (arr/add fixed-score-arr
                   (compute-score-from-cache cache values-arr jvm-vals))))
      ;; Constant params: cache everything at construction time
      (let [entries (replay-model-once gf model-args all-addrs choices)
            free-set (set free-addrs)
            free-entries (vec (keep (fn [[entry addr]]
                                     (when (contains? free-set addr) entry))
                                   (map vector entries all-addrs)))
            cache (precompute-param-arrays free-entries)
            has-fallback? (some #(= :fallback (:tag %)) free-entries)]
        (if has-fallback?
          (fn [values-arr]
            (let [jvm-vals (arr/->vec values-arr)]
              (arr/add fixed-score-arr
                       (compute-score-from-cache cache values-arr jvm-vals))))
          (fn [values-arr]
            (arr/add fixed-score-arr
                     (compute-score-from-cache cache values-arr nil))))))))

(defn hmc-sample
  "Run HMC on the choices of an MLXTrace.

   Builds a vectorized score function from the trace and delegates to
   gen.mlx.hmc/sample for leapfrog integration and MH accept/reject.

   Options:
     :L         — number of leapfrog steps (default 10)
     :eps       — leapfrog step size (default 0.01)
     :selection — set of addresses to sample (others held fixed).
                  When nil, samples all addresses.

   Returns a vector of n-steps result maps, each containing:
     :position    - MLXArray vector
     :log-density - double
     :accepted?   - boolean
     :trace       - new MLXTrace with updated choices
     :choices     - {addr -> double}"
  [^MLXTrace trace n-steps & {:keys [L eps selection] :or {L 10 eps 0.01}}]
  (let [gf           (.-gen-fn trace)
        model-args   (.-args trace)
        info         (.-addr-info trace)
        choices      (.-choices trace)
        all-addrs    (mapv :addr info)]
    (if selection
      ;; Selection-aware: only sample selected addresses
      (let [sel-set      (set selection)
            free-addrs   (filterv #(contains? sel-set %) all-addrs)
            fixed-info   (filterv #(not (contains? sel-set (:addr %))) info)
            score-fn     (build-free-score-fn gf model-args free-addrs fixed-info choices)
            init-pos     (arr/from-vec (mapv #(double (get choices %)) free-addrs))
            raw-samples  (hmc/sample score-fn init-pos n-steps :L L :eps eps)]
        (mapv (fn [sample]
                (let [pos-vec     (arr/->vec (:position sample))
                      free-choices (zipmap free-addrs pos-vec)
                      new-choices  (merge choices free-choices)]
                  (assoc sample
                         :choices new-choices
                         :trace   (->MLXTrace gf new-choices (:log-density sample)
                                              model-args (.-retval trace) info))))
              raw-samples))
      ;; No selection: sample all addresses with compiled score fn
      (let [score-fn     (build-compiled-score-fn gf model-args all-addrs choices)
            init-pos     (arr/from-vec (mapv #(double (get choices %)) all-addrs))
            raw-samples  (hmc/sample score-fn init-pos n-steps :L L :eps eps)]
        (mapv (fn [sample]
                (let [pos-vec     (arr/->vec (:position sample))
                      new-choices (zipmap all-addrs pos-vec)]
                  (assoc sample
                         :choices new-choices
                         :trace   (->MLXTrace gf new-choices (:log-density sample)
                                              model-args (.-retval trace) info))))
              raw-samples)))))

;; ---------------------------------------------------------------------------
;; MALA bridge — HMC with L=1
;; ---------------------------------------------------------------------------

(defn mala-sample
  "Run MALA on the choices of an MLXTrace.

   MALA is HMC with a single leapfrog step (L=1). Simpler than HMC
   but can be effective for well-conditioned targets.

   Options:
     :eps       — step size (default 0.01)
     :selection — set of addresses to sample (others held fixed)

   Returns a vector of n-steps result maps, each containing:
     :position    - MLXArray vector
     :log-density - double
     :accepted?   - boolean
     :trace       - new MLXTrace with updated choices
     :choices     - {addr -> double}"
  [^MLXTrace trace n-steps & {:keys [eps selection] :or {eps 0.01}}]
  (hmc-sample trace n-steps :L 1 :eps eps :selection selection))

;; ---------------------------------------------------------------------------
;; MAP optimization bridge
;; ---------------------------------------------------------------------------

(defn map-optimize
  "Find MAP estimate for the choices of an MLXTrace.

   Uses Adam optimizer with MLX autodiff for gradients.

   Options:
     :lr        — learning rate (default 0.01)
     :beta1     — first moment decay (default 0.9)
     :beta2     — second moment decay (default 0.999)
     :epsilon   — numerical stability (default 1e-8)
     :tol       — convergence tolerance (default nil)
     :patience  — consecutive steps within tol before stopping (default 5)
     :selection — set of addresses to optimize (others held fixed)

   Returns {:position MLXArray, :log-density double,
            :choices {addr -> double}, :trace MLXTrace,
            :history [{:step :log-density}]}."
  [^MLXTrace trace n-steps & {:keys [lr beta1 beta2 epsilon tol patience selection]
                               :or {lr 0.01 beta1 0.9 beta2 0.999
                                    epsilon 1e-8 patience 5}
                               :as opts}]
  (let [gf           (.-gen-fn trace)
        model-args   (.-args trace)
        info         (.-addr-info trace)
        choices      (.-choices trace)
        all-addrs    (mapv :addr info)]
    (if selection
      ;; Selection-aware: only optimize selected addresses
      (let [sel-set      (set selection)
            free-addrs   (filterv #(contains? sel-set %) all-addrs)
            fixed-info   (filterv #(not (contains? sel-set (:addr %))) info)
            score-fn     (build-free-score-fn gf model-args free-addrs fixed-info choices)
            init-pos     (arr/from-vec (mapv #(double (get choices %)) free-addrs))
            result       (map-opt/map-optimize score-fn init-pos n-steps
                                               :lr lr :beta1 beta1 :beta2 beta2
                                               :epsilon epsilon :tol tol
                                               :patience patience)
            pos-vec      (arr/->vec (:position result))
            free-choices (zipmap free-addrs pos-vec)
            new-choices  (merge choices free-choices)]
        (assoc result
               :choices new-choices
               :trace   (->MLXTrace gf new-choices (:log-density result)
                                    model-args (.-retval trace) info)))
      ;; No selection: optimize all addresses with compiled score fn
      (let [score-fn     (build-compiled-score-fn gf model-args all-addrs choices)
            init-pos     (arr/from-vec (mapv #(double (get choices %)) all-addrs))
            result       (map-opt/map-optimize score-fn init-pos n-steps
                                               :lr lr :beta1 beta1 :beta2 beta2
                                               :epsilon epsilon :tol tol
                                               :patience patience)
            pos-vec      (arr/->vec (:position result))
            new-choices  (zipmap all-addrs pos-vec)]
        (assoc result
               :choices new-choices
               :trace   (->MLXTrace gf new-choices (:log-density result)
                                    model-args (.-retval trace) info))))))

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
     :selection     — set of addresses to sample (others held fixed)

   Returns a vector of n-steps result maps, each containing:
     :position    — MLXArray
     :log-density — double
     :tree-depth  — int
     :n-leapfrog  — int
     :eps         — step size used
     :accept-prob — acceptance probability
     :trace       — new MLXTrace with updated choices
     :choices     — {addr -> double}"
  [^MLXTrace trace n-steps & {:keys [max-depth target-accept n-warmup initial-eps selection]
                               :or {max-depth 10 target-accept 0.8}}]
  (let [gf           (.-gen-fn trace)
        model-args   (.-args trace)
        info         (.-addr-info trace)
        choices      (.-choices trace)
        all-addrs    (mapv :addr info)]
    (if selection
      ;; Selection-aware: only sample selected addresses
      (let [sel-set      (set selection)
            free-addrs   (filterv #(contains? sel-set %) all-addrs)
            fixed-info   (filterv #(not (contains? sel-set (:addr %))) info)
            score-fn     (build-free-score-fn gf model-args free-addrs fixed-info choices)
            init-pos     (arr/from-vec (mapv #(double (get choices %)) free-addrs))
            raw-samples  (nuts/sample score-fn init-pos n-steps
                                      :max-depth max-depth
                                      :target-accept target-accept
                                      :n-warmup n-warmup
                                      :initial-eps initial-eps)]
        (mapv (fn [sample]
                (let [pos-vec     (arr/->vec (:position sample))
                      free-choices (zipmap free-addrs pos-vec)
                      new-choices  (merge choices free-choices)]
                  (assoc sample
                         :choices new-choices
                         :trace   (->MLXTrace gf new-choices (:log-density sample)
                                              model-args (.-retval trace) info))))
              raw-samples))
      ;; No selection: sample all addresses with compiled score fn
      (let [score-fn     (build-compiled-score-fn gf model-args all-addrs choices)
            init-pos     (arr/from-vec (mapv #(double (get choices %)) all-addrs))
            raw-samples  (nuts/sample score-fn init-pos n-steps
                                      :max-depth max-depth
                                      :target-accept target-accept
                                      :n-warmup n-warmup
                                      :initial-eps initial-eps)]
        (mapv (fn [sample]
                (let [pos-vec     (arr/->vec (:position sample))
                      new-choices (zipmap all-addrs pos-vec)]
                  (assoc sample
                         :choices new-choices
                         :trace   (->MLXTrace gf new-choices (:log-density sample)
                                              model-args (.-retval trace) info))))
              raw-samples)))))

;; ---------------------------------------------------------------------------
;; Parallel HMC bridge — N chains simultaneously
;; ---------------------------------------------------------------------------

(defn parallel-hmc-sample
  "Run N parallel HMC chains on the choices of an MLXTrace.

   Uses manual batching: all N chains start from the same initial position
   and diverge through independent momentum sampling. Operations on (N x D)
   arrays amortize FFI overhead across chains.

   Options:
     :L         — leapfrog steps per HMC step (default 10)
     :eps       — leapfrog step size (default 0.01)
     :selection — set of addresses to sample (others held fixed)

   Returns a vector of N chains, where each chain is a vector of n-steps
   result maps:
     {:position    — MLXArray (1-D)
      :log-density — double
      :accepted?   — boolean
      :choices     — {addr -> double}
      :trace       — MLXTrace}"
  [^MLXTrace trace n-chains n-steps
   & {:keys [L eps selection] :or {L 10 eps 0.01}}]
  (let [gf           (.-gen-fn trace)
        model-args   (.-args trace)
        info         (.-addr-info trace)
        choices      (.-choices trace)
        all-addrs    (mapv :addr info)]
    (if selection
      ;; Selection-aware: only sample selected addresses
      (let [sel-set      (set selection)
            free-addrs   (filterv #(contains? sel-set %) all-addrs)
            fixed-info   (filterv #(not (contains? sel-set (:addr %))) info)
            score-fn     (build-free-score-fn gf model-args free-addrs fixed-info choices)
            ;; Data-dependent score fns call arr/->vec (eval), which is
            ;; incompatible with vmap tracing. Detect this and build a
            ;; pure-MLX per-chain function using cached constant params.
            has-deps?    (params-depend-on-choices? gf model-args all-addrs choices)
            per-chain-fn (when has-deps?
                           (build-free-score-fn gf model-args free-addrs fixed-info choices
                                               :force-constant true))
            init-pos-1d    (arr/from-vec (mapv #(double (get choices %)) free-addrs))
            initial-positions (arr/stack (repeat n-chains init-pos-1d))
            raw-steps (hmc/parallel-sample score-fn initial-positions n-steps
                                           :L L :eps eps
                                           :per-chain-score-fn
                                           (when per-chain-fn
                                             (xforms/vmap per-chain-fn)))
            D (count free-addrs)]
        ;; Transpose: step-oriented → chain-oriented
        (mapv (fn [chain-idx]
                (mapv (fn [step]
                        (let [flat (arr/->vec (:positions step))
                              chain-vals (subvec (vec flat)
                                                 (* chain-idx D)
                                                 (* (inc chain-idx) D))
                              ld (nth (:log-densities step) chain-idx)
                              acc (nth (:accepted step) chain-idx)
                              free-choices (zipmap free-addrs chain-vals)
                              new-choices (merge choices free-choices)]
                          {:position (arr/from-vec chain-vals)
                           :log-density ld
                           :accepted? acc
                           :choices new-choices
                           :trace (->MLXTrace gf new-choices ld
                                              model-args (.-retval trace) info)}))
                      raw-steps))
              (range n-chains)))
      ;; No selection: sample all addresses
      (let [score-fn     (build-compiled-score-fn gf model-args all-addrs choices)
            init-pos-1d    (arr/from-vec (mapv #(double (get choices %)) all-addrs))
            initial-positions (arr/stack (repeat n-chains init-pos-1d))
            raw-steps (hmc/parallel-sample score-fn initial-positions n-steps
                                           :L L :eps eps)
            D (count all-addrs)]
        ;; Transpose: step-oriented → chain-oriented
        (mapv (fn [chain-idx]
                (mapv (fn [step]
                        (let [flat (arr/->vec (:positions step))
                              chain-vals (subvec (vec flat)
                                                 (* chain-idx D)
                                                 (* (inc chain-idx) D))
                              ld (nth (:log-densities step) chain-idx)
                              acc (nth (:accepted step) chain-idx)
                              new-choices (zipmap all-addrs chain-vals)]
                          {:position (arr/from-vec chain-vals)
                           :log-density ld
                           :accepted? acc
                           :choices new-choices
                           :trace (->MLXTrace gf new-choices ld
                                              model-args (.-retval trace) info)}))
                      raw-steps))
              (range n-chains))))))
