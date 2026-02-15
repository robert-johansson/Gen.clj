(ns gen.combinator.unfold
  "Unfold combinator: runs a kernel generative function sequentially, threading
   state through each step. The kernel signature is (step-index, state, extra-args...)
   and returns the new state. Used for time-series models."
  (:require [gen.choicemap :as choicemap]
            [gen.diff :as diff]
            [gen.generative-function :as gf]
            [gen.selection :as selection]
            [gen.trace :as trace]))

;; ## UnfoldTrace

(defrecord UnfoldTrace [gen-fn args sub-traces states score]
  ;; args = [n-steps init-state & extra-args]
  ;; sub-traces = vector of kernel traces, length n-steps
  ;; states = [init-state s0 s1 ... s_{n-1}], length n-steps + 1

  trace/ITrace
  (get-args [_] args)
  (get-retval [_] states)
  (get-gen-fn [_] gen-fn)
  (get-score [_] score)
  (get-choices [_]
    (if (seq sub-traces)
      (choicemap/->VectorChoiceMap
       (mapv trace/get-choices sub-traces))
      choicemap/EMPTY))

  trace/IUpdate
  (-update [_ new-args _argdiffs constraints]
    (let [[new-n new-init-state & new-extra] new-args
          kernel (:kernel gen-fn)
          old-n-steps (count sub-traces)]
      ;; General case: re-thread state through all steps
      (loop [i          0
             state      new-init-state
             traces     (transient [])
             all-states [new-init-state]
             weight     0.0
             discard    (transient {})]
        (if (< i (int new-n))
          (let [kernel-args     (into [i state] new-extra)
                el-constraints  (choicemap/get-submap constraints i)]
            (if (< i old-n-steps)
              ;; Existing step: update
              (let [{new-trace :trace
                     w         :weight
                     d         :discard}
                    (trace/update (nth sub-traces i) el-constraints)
                    new-state (trace/get-retval new-trace)]
                (recur (inc i)
                       new-state
                       (conj! traces new-trace)
                       (conj all-states new-state)
                       (+ weight w)
                       (if (choicemap/empty? d)
                         discard
                         (assoc! discard i d))))
              ;; New step: generate fresh
              (let [{new-trace :trace
                     w         :weight}
                    (gf/generate kernel kernel-args el-constraints)
                    new-state (trace/get-retval new-trace)]
                (recur (inc i)
                       new-state
                       (conj! traces new-trace)
                       (conj all-states new-state)
                       (+ weight w)
                       discard))))
          ;; Done — handle removed steps
          (let [traces-vec  (persistent! traces)
                discard-map (persistent! discard)
                ;; Removed steps: subtract scores, add to discard
                {:keys [removed-discard removed-score]}
                (loop [j old-n-steps, rm-d discard-map, rm-s 0.0]
                  (if (> j (int new-n))
                    (let [jj (dec j)]
                      (recur jj
                             (assoc rm-d jj (trace/get-choices (nth sub-traces jj)))
                             (+ rm-s (trace/get-score (nth sub-traces jj)))))
                    {:removed-discard rm-d :removed-score rm-s}))
                new-score (reduce + 0.0 (map trace/get-score traces-vec))]
            {:trace   (->UnfoldTrace gen-fn new-args traces-vec all-states new-score)
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

;; ## UnfoldGenerativeFn

(defrecord UnfoldGenerativeFn [kernel]
  gf/IGenerativeFunction
  (has-argument-grads [_] (gf/has-argument-grads kernel))
  (accepts-output-grad? [_] false)
  (get-params [_] (gf/get-params kernel))

  (simulate [this args]
    (let [[n-steps init-state & extra] args]
      (loop [i      0
             state  init-state
             traces (transient [])
             states [init-state]
             score  0.0]
        (if (< i (int n-steps))
          (let [kernel-args (into [i state] extra)
                sub-trace   (gf/simulate kernel kernel-args)
                new-state   (trace/get-retval sub-trace)]
            (recur (inc i)
                   new-state
                   (conj! traces sub-trace)
                   (conj states new-state)
                   (+ score (trace/get-score sub-trace))))
          (->UnfoldTrace this args (persistent! traces) states score)))))

  gf/IGenerate
  (-generate [this args constraints]
    (let [[n-steps init-state & extra] args]
      (loop [i      0
             state  init-state
             traces (transient [])
             states [init-state]
             score  0.0
             weight 0.0]
        (if (< i (int n-steps))
          (let [kernel-args    (into [i state] extra)
                el-constraints (choicemap/get-submap constraints i)
                result         (if (choicemap/empty? el-constraints)
                                 (gf/generate kernel kernel-args)
                                 (gf/generate kernel kernel-args el-constraints))
                sub-trace      (:trace result)
                new-state      (trace/get-retval sub-trace)]
            (recur (inc i)
                   new-state
                   (conj! traces sub-trace)
                   (conj states new-state)
                   (+ score (trace/get-score sub-trace))
                   (+ weight (:weight result))))
          {:trace  (->UnfoldTrace this args (persistent! traces) states score)
           :weight weight}))))

  gf/IRegenerate
  (-regenerate [this old-trace new-args _argdiffs sel]
    (let [[new-n new-init-state & new-extra] new-args
          old-sub-traces (:sub-traces old-trace)]
      (loop [i      0
             state  new-init-state
             traces (transient [])
             states [new-init-state]
             score  0.0]
        (if (< i (int new-n))
          (let [sub-sel (selection/get-subselection sel i)]
            (if (and (< i (count old-sub-traces))
                     (instance? gen.selection.EmptySelection sub-sel))
              ;; Not selected: keep old sub-trace, but re-thread state
              (let [old-st    (nth old-sub-traces i)
                    new-state (trace/get-retval old-st)]
                (recur (inc i)
                       new-state
                       (conj! traces old-st)
                       (conj states new-state)
                       (+ score (trace/get-score old-st))))
              ;; Selected or new: regenerate or simulate
              (if (< i (count old-sub-traces))
                (let [{new-st :trace}
                      (gf/regenerate (nth old-sub-traces i)
                                     (into [i state] new-extra)
                                     (repeat (count new-args) diff/no-change)
                                     sub-sel)
                      new-state (trace/get-retval new-st)]
                  (recur (inc i)
                         new-state
                         (conj! traces new-st)
                         (conj states new-state)
                         (+ score (trace/get-score new-st))))
                (let [new-st    (gf/simulate kernel (into [i state] new-extra))
                      new-state (trace/get-retval new-st)]
                  (recur (inc i)
                         new-state
                         (conj! traces new-st)
                         (conj states new-state)
                         (+ score (trace/get-score new-st)))))))
          (let [traces-vec (persistent! traces)
                new-trace  (->UnfoldTrace this new-args traces-vec states score)
                old-score  (trace/get-score old-trace)
                new-sel-score (trace/project new-trace sel)
                old-sel-score (trace/project old-trace sel)
                weight (- (- score new-sel-score)
                          (- old-score old-sel-score))]
            {:trace  new-trace
             :weight weight
             :change diff/unknown-change}))))))

;; ## Public API

(defn unfold-gen-fn
  "Create an Unfold combinator that runs `kernel` sequentially, threading state.
   The kernel signature is (step-index, state, extra-args...) -> new-state.
   Args to the Unfold gen-fn are [n-steps init-state & extra-args]."
  [kernel]
  (->UnfoldGenerativeFn kernel))
