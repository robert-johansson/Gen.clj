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
;;   to collect distribution parameters at each address.
;; Phase 2 (MLX): one vectorized gaussian-logpdf + arr/sum.
;;
;; This batches ~6N FFI ops down to ~6 regardless of N choices.
;; ---------------------------------------------------------------------------

(defn build-score-fn
  "Build a pure function (MLXArray -> MLXArray) that computes the total
   log-probability of all choices. Takes a single MLXArray vector of
   choice values (ordered by sorted-addrs) and returns a scalar MLXArray.

   Phase 1 replays the model in JVM to collect dist params (which may
   depend on earlier choices). Phase 2 computes all logpdfs in one
   vectorized MLX call."
  [gf model-args sorted-addrs]
  (fn [values-arr]
    ;; Phase 1: extract JVM doubles, replay model to collect dist params
    (let [jvm-vals  (arr/->vec values-arr)
          addr->jvm (zipmap sorted-addrs jvm-vals)
          !params   (volatile! [])]
      (binding [dynamic/*trace*
                (fn
                  ([inner-gf inner-args]
                   (apply (.-clojure-fn ^MLXDynamicDSLFunction inner-gf)
                          inner-args))
                  ([k _dist-gf dist-args]
                   (vswap! !params conj (vec dist-args))
                   (get addr->jvm k)))]
        (apply (.-clojure-fn ^MLXDynamicDSLFunction gf) model-args))
      ;; Phase 2: vectorized logpdf
      (let [params @!params
            mus    (arr/from-vec (mapv #(double (first %)) params))
            sigmas (arr/from-vec (mapv #(double (second %)) params))]
        (arr/sum (mlx-dist/gaussian-logpdf mus sigmas values-arr))))))

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
