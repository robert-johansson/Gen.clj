(ns gen.mlx.dynamic
  "MLX-backed dynamic DSL for Gen.clj.

   Replaces atoms with volatile! for single-threaded semantics and records
   addr-info (distribution + params at each address) so the model's score
   computation can be replayed as a pure MLX function for autodiff.

   Implements IChoiceGradients via value-and-grad on the replayed score,
   and provides an HMC bridge for gradient-based inference."
  (:require [clojure.walk :as walk]
            [gen.choicemap :as choicemap]
            [gen.distribution :as d]
            [gen.dynamic :as dynamic]
            [gen.generative-function :as gf]
            [gen.mlx.array :as arr]
            [gen.mlx.distribution :as mlx-dist]
            [gen.mlx.transforms :as xforms]
            [gen.trace :as trace])
  (:import [java.util.concurrent ThreadLocalRandom]))

;; ---------------------------------------------------------------------------
;; MLX logpdf dispatch
;; ---------------------------------------------------------------------------

(defn- mlx-logpdf
  "Compute differentiable logpdf for a distribution's GenerativeFn.
   Prototype: only supports normal distribution."
  [dist-gf dist-args mlx-value]
  (when-not (= dist-gf mlx-dist/normal)
    (throw (ex-info "mlx-logpdf only supports mlx-dist/normal"
                    {:dist-gf dist-gf})))
  (mlx-dist/gaussian-logpdf (first dist-args) (second dist-args) mlx-value))

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
;; build-score-fn — replay model to construct differentiable score
;; ---------------------------------------------------------------------------

(defn build-score-fn
  "Build a pure function (MLXArray... -> MLXArray) that replays the model,
   substituting MLXArray values for each choice and computing score via
   differentiable logpdf. Distribution params that depend on other choices
   are computed in JVM land (constants from MLX's perspective)."
  [gf model-args addr-info sorted-addrs]
  (fn [& mlx-vals]
    (let [addr->mlx (zipmap sorted-addrs mlx-vals)
          !score    (volatile! (arr/scalar 0.0))]
      (binding [dynamic/*trace*
                (fn
                  ([inner-gf inner-args]
                   (apply (.-clojure-fn ^MLXDynamicDSLFunction inner-gf)
                          inner-args))
                  ([k dist-gf dist-args]
                   (let [mlx-v (get addr->mlx k)
                         lp    (mlx-logpdf dist-gf dist-args mlx-v)]
                     (vswap! !score #(arr/add % lp))
                     (arr/->double mlx-v))))]
        (apply (.-clojure-fn ^MLXDynamicDSLFunction gf) model-args)
        @!score))))

;; ---------------------------------------------------------------------------
;; IChoiceGradients
;; ---------------------------------------------------------------------------

(extend-type MLXTrace
  trace/IChoiceGradients
  (choice-gradients [trace _selection _retgrad]
    (let [gf        (.-gen-fn trace)
          model-args (.-args trace)
          info      (.-addr-info trace)
          choices   (.-choices trace)
          sorted-addrs (mapv :addr info)
          n         (count sorted-addrs)
          score-fn  (build-score-fn gf model-args info sorted-addrs)
          vag-fn    (xforms/value-and-grad score-fn {:argnums (vec (range n))})
          mlx-vals  (mapv #(arr/scalar (double (get choices %))) sorted-addrs)
          result    (apply vag-fn mlx-vals)
          grads     (:grads result)]
      {:arg-grads     nil
       :choice-values (choicemap/map->choicemap
                       (zipmap sorted-addrs
                               (mapv #(choicemap/->Choice (get choices %))
                                     sorted-addrs)))
       :choice-grads  (choicemap/map->choicemap
                       (zipmap sorted-addrs
                               (mapv #(choicemap/->Choice (arr/->double %))
                                     grads)))})))

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
;; HMC bridge
;; ---------------------------------------------------------------------------

(defn- sample-momentum [n]
  (let [rng (ThreadLocalRandom/current)]
    (mapv (fn [_] (.nextGaussian rng)) (range n))))

(defn- kinetic-energy [p]
  (* 0.5 (reduce + (map #(* % %) p))))

(defn hmc-sample
  "Run HMC on the choices of an MLXTrace.

   Internally uses JVM-double vectors for leapfrog arithmetic and
   calls the MLX value-and-grad function for gradient evaluation.

   Returns a vector of n-steps result maps, each containing:
     :position    - vector of doubles
     :log-density - double
     :accepted?   - boolean
     :trace       - new MLXTrace with updated choices
     :choices     - {addr -> double}"
  [^MLXTrace trace n-steps & {:keys [L eps] :or {L 10 eps 0.01}}]
  (let [gf         (.-gen-fn trace)
        model-args (.-args trace)
        info       (.-addr-info trace)
        choices    (.-choices trace)
        sorted-addrs (mapv :addr info)
        n          (count sorted-addrs)
        score-fn   (build-score-fn gf model-args info sorted-addrs)
        vag-fn     (xforms/value-and-grad score-fn {:argnums (vec (range n))})
        eval-vag   (fn [position]
                     (let [mlx-args (mapv #(arr/scalar (double %)) position)
                           result   (apply vag-fn mlx-args)]
                       {:val  (arr/->double (:value result))
                        :grad (mapv arr/->double (:grads result))}))
        leapfrog   (fn [q p]
                     (let [{:keys [grad]} (eval-vag q)
                           p (mapv (fn [pi gi] (+ pi (* (/ eps 2.0) gi))) p grad)]
                       (loop [i 0 q q p p]
                         (if (>= i L)
                           (let [{:keys [val grad]} (eval-vag q)
                                 p-final (mapv (fn [pi gi] (+ pi (* (/ eps 2.0) gi)))
                                               p grad)]
                             {:position q :momentum p-final :log-density val})
                           (let [q-new (mapv (fn [qi pi] (+ qi (* eps pi))) q p)]
                             (if (< i (dec L))
                               (let [{:keys [grad]} (eval-vag q-new)
                                     p-new (mapv (fn [pi gi] (+ pi (* eps gi))) p grad)]
                                 (recur (inc i) q-new p-new))
                               (recur (inc i) q-new p)))))))
        init-pos   (mapv #(double (get choices %)) sorted-addrs)
        init-ld    (:val (eval-vag init-pos))]
    (loop [i 0
           pos init-pos
           ld init-ld
           results []]
      (if (>= i n-steps)
        results
        (let [p         (sample-momentum n)
              current-ke (kinetic-energy p)
              current-H  (- current-ke ld)
              proposal   (leapfrog pos p)
              proposed-ld (:log-density proposal)
              proposed-ke (kinetic-energy (:momentum proposal))
              proposed-H  (- proposed-ke proposed-ld)
              log-accept  (- current-H proposed-H)
              accepted?   (< (Math/log (Math/random)) log-accept)
              new-pos     (if accepted? (:position proposal) pos)
              new-ld      (if accepted? proposed-ld ld)
              new-choices (zipmap sorted-addrs new-pos)
              new-trace   (->MLXTrace gf new-choices new-ld model-args
                                      (.-retval trace) info)]
          (recur (inc i)
                 new-pos
                 new-ld
                 (conj results {:position    new-pos
                                :log-density new-ld
                                :accepted?   accepted?
                                :trace       new-trace
                                :choices     new-choices})))))))
