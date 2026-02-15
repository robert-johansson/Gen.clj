(ns gen.combinator.map
  "Map combinator: applies a kernel generative function independently to each
   element of vectorized arguments. Produces VectorChoiceMap-indexed traces."
  (:require [gen.choicemap :as choicemap]
            [gen.diff :as diff]
            [gen.generative-function :as gf]
            [gen.selection :as selection]
            [gen.trace :as trace]))

;; ## Helpers

(defn- element-args
  "Extract the i-th element from each argument vector."
  [args i]
  (mapv #(nth % i) args))

(defn- n-elements
  "Number of elements — length of first arg vector."
  [args]
  (if (seq args) (count (first args)) 0))

;; ## MapTrace

(defrecord MapTrace [gen-fn args sub-traces retval score]
  trace/ITrace
  (get-args [_] args)
  (get-retval [_] retval)
  (get-gen-fn [_] gen-fn)
  (get-score [_] score)
  (get-choices [_]
    (if (seq sub-traces)
      (choicemap/->VectorChoiceMap
       (mapv trace/get-choices sub-traces))
      choicemap/EMPTY))

  trace/IUpdate
  (-update [_ new-args _argdiffs constraints]
    (let [new-n  (n-elements new-args)
          old-n  (count sub-traces)
          kernel (:kernel gen-fn)]
      (loop [i       0
             traces  (transient [])
             weight  0.0
             discard (transient {})]
        (if (< i new-n)
          (let [el-args       (element-args new-args i)
                el-constraints (choicemap/get-submap constraints i)]
            (if (< i old-n)
              ;; Existing index: update sub-trace
              (let [{new-trace :trace
                     w         :weight
                     d         :discard}
                    (trace/update (nth sub-traces i) el-constraints)]
                (recur (inc i)
                       (conj! traces new-trace)
                       (+ weight w)
                       (if (choicemap/empty? d)
                         discard
                         (assoc! discard i d))))
              ;; New index: generate fresh
              (let [{new-trace :trace
                     w         :weight}
                    (gf/generate kernel el-args el-constraints)]
                (recur (inc i)
                       (conj! traces new-trace)
                       (+ weight w)
                       discard))))
          ;; Done with new indices, handle removals
          (let [traces-vec (persistent! traces)
                discard-map (persistent! discard)
                ;; Removed indices go to discard, subtract their scores
                {:keys [removed-discard removed-score]}
                (loop [j old-n, rm-d discard-map, rm-s 0.0]
                  (if (> j new-n)
                    (let [jj (dec j)]
                      (recur jj
                             (assoc rm-d jj (trace/get-choices (nth sub-traces jj)))
                             (+ rm-s (trace/get-score (nth sub-traces jj)))))
                    {:removed-discard rm-d :removed-score rm-s}))
                new-score (reduce + 0.0 (map trace/get-score traces-vec))
                new-retval (mapv trace/get-retval traces-vec)]
            {:trace   (->MapTrace gen-fn new-args traces-vec new-retval new-score)
             :weight  (- weight removed-score)
             :change  diff/unknown-change
             :discard (if (seq removed-discard)
                        (choicemap/choicemap removed-discard)
                        (choicemap/choicemap discard-map))})))))

  trace/IProject
  (project [_ sel]
    (reduce-kv
     (fn [acc i sub-trace]
       (let [sub-sel (selection/get-subselection sel i)]
         (if (instance? gen.selection.EmptySelection sub-sel)
           acc
           (+ acc (trace/project sub-trace sub-sel)))))
     0.0
     (vec sub-traces))))

;; ## MapGenerativeFn

(defrecord MapGenerativeFn [kernel]
  gf/IGenerativeFunction
  (has-argument-grads [_] (gf/has-argument-grads kernel))
  (accepts-output-grad? [_] (gf/accepts-output-grad? kernel))
  (get-params [_] (gf/get-params kernel))

  (simulate [this args]
    (let [n (n-elements args)]
      (loop [i      0
             traces (transient [])
             score  0.0]
        (if (< i n)
          (let [sub-trace (gf/simulate kernel (element-args args i))]
            (recur (inc i)
                   (conj! traces sub-trace)
                   (+ score (trace/get-score sub-trace))))
          (let [traces-vec (persistent! traces)
                retval     (mapv trace/get-retval traces-vec)]
            (->MapTrace this args traces-vec retval score))))))

  gf/IGenerate
  (-generate [this args constraints]
    (let [n (n-elements args)]
      (loop [i      0
             traces (transient [])
             score  0.0
             weight 0.0]
        (if (< i n)
          (let [el-constraints (choicemap/get-submap constraints i)
                result (if (choicemap/empty? el-constraints)
                         (gf/generate kernel (element-args args i))
                         (gf/generate kernel (element-args args i) el-constraints))
                w (:weight result)]
            (recur (inc i)
                   (conj! traces (:trace result))
                   (+ score (trace/get-score (:trace result)))
                   (+ weight w)))
          (let [traces-vec (persistent! traces)
                retval     (mapv trace/get-retval traces-vec)]
            {:trace  (->MapTrace this args traces-vec retval score)
             :weight weight})))))

  gf/IRegenerate
  (-regenerate [this old-trace new-args _argdiffs sel]
    (let [old-sub-traces (:sub-traces old-trace)
          n              (n-elements new-args)]
      (loop [i      0
             traces (transient [])
             score  0.0]
        (if (< i n)
          (let [sub-sel (selection/get-subselection sel i)]
            (if (and (< i (count old-sub-traces))
                     (instance? gen.selection.EmptySelection sub-sel))
              ;; Not selected + exists: keep old sub-trace
              (let [old-st (nth old-sub-traces i)]
                (recur (inc i)
                       (conj! traces old-st)
                       (+ score (trace/get-score old-st))))
              ;; Selected or new: regenerate/simulate
              (if (< i (count old-sub-traces))
                (let [{new-st :trace}
                      (gf/regenerate (nth old-sub-traces i)
                                     (element-args new-args i)
                                     (repeat (count new-args) diff/no-change)
                                     sub-sel)]
                  (recur (inc i)
                         (conj! traces new-st)
                         (+ score (trace/get-score new-st))))
                (let [new-st (gf/simulate kernel (element-args new-args i))]
                  (recur (inc i)
                         (conj! traces new-st)
                         (+ score (trace/get-score new-st)))))))
          (let [traces-vec (persistent! traces)
                retval     (mapv trace/get-retval traces-vec)
                new-trace  (->MapTrace this new-args traces-vec retval score)
                old-score  (trace/get-score old-trace)
                new-sel-score (trace/project new-trace sel)
                old-sel-score (trace/project old-trace sel)
                weight (- (- score new-sel-score)
                          (- old-score old-sel-score))]
            {:trace  new-trace
             :weight weight
             :change diff/unknown-change}))))))

;; ## Public API

(defn map-gen-fn
  "Create a Map combinator that applies `kernel` to each element of vectorized args."
  [kernel]
  (->MapGenerativeFn kernel))
