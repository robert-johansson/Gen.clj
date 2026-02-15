(ns gen.combinator.switch
  "Switch combinator: selects one of several branch generative functions based
   on a branch index. Used for mixture models and conditional branching."
  (:require [gen.choicemap :as choicemap]
            [gen.diff :as diff]
            [gen.generative-function :as gf]
            [gen.selection :as selection]
            [gen.trace :as trace]))

;; ## SwitchTrace

(defrecord SwitchTrace [gen-fn args branch-idx sub-trace score]
  trace/ITrace
  (get-args [_] args)
  (get-retval [_] (trace/get-retval sub-trace))
  (get-gen-fn [_] gen-fn)
  (get-score [_] score)
  (get-choices [_]
    ;; Namespace sub-trace choices under the branch index
    (choicemap/choicemap {branch-idx (trace/get-choices sub-trace)}))

  trace/IUpdate
  (-update [_ new-args _argdiffs constraints]
    (let [[new-branch-idx & new-branch-args] new-args
          new-branch (nth (:branches gen-fn) new-branch-idx)
          branch-constraints (choicemap/get-submap constraints new-branch-idx)]
      (if (= new-branch-idx branch-idx)
        ;; Same branch: update sub-trace
        (let [{new-sub-trace :trace
               w             :weight
               d             :discard}
              (trace/update sub-trace branch-constraints)
              new-score (trace/get-score new-sub-trace)
              new-discard (if (choicemap/empty? d)
                            choicemap/EMPTY
                            (choicemap/choicemap {branch-idx d}))]
          {:trace   (->SwitchTrace gen-fn new-args new-branch-idx new-sub-trace new-score)
           :weight  w
           :change  diff/unknown-change
           :discard new-discard})
        ;; Branch switch: discard old, generate new
        (let [{new-sub-trace :trace
               gen-weight    :weight}
              (if (choicemap/empty? branch-constraints)
                (gf/generate new-branch (vec new-branch-args))
                (gf/generate new-branch (vec new-branch-args) branch-constraints))
              new-score    (trace/get-score new-sub-trace)
              old-discard  (choicemap/choicemap {branch-idx (trace/get-choices sub-trace)})
              weight       (- gen-weight score)]
          {:trace   (->SwitchTrace gen-fn new-args new-branch-idx new-sub-trace new-score)
           :weight  weight
           :change  diff/unknown-change
           :discard old-discard}))))

  trace/IProject
  (project [_ sel]
    (let [sub-sel (selection/get-subselection sel branch-idx)]
      (if (instance? gen.selection.EmptySelection sub-sel)
        0.0
        (trace/project sub-trace sub-sel)))))

;; ## SwitchGenerativeFn

(defrecord SwitchGenerativeFn [branches]
  gf/IGenerativeFunction
  (has-argument-grads [_] [false])
  (accepts-output-grad? [_] false)
  (get-params [_] ())

  (simulate [this args]
    (let [[branch-idx & branch-args] args
          branch    (nth branches branch-idx)
          sub-trace (gf/simulate branch (vec branch-args))
          score     (trace/get-score sub-trace)]
      (->SwitchTrace this args branch-idx sub-trace score)))

  gf/IGenerate
  (-generate [this args constraints]
    (let [[branch-idx & branch-args] args
          branch           (nth branches branch-idx)
          branch-constraints (choicemap/get-submap constraints branch-idx)
          {:keys [trace weight]}
          (if (choicemap/empty? branch-constraints)
            (gf/generate branch (vec branch-args))
            (gf/generate branch (vec branch-args) branch-constraints))
          score (trace/get-score trace)]
      {:trace  (->SwitchTrace this args branch-idx trace score)
       :weight weight}))

  gf/IRegenerate
  (-regenerate [this old-trace new-args _argdiffs sel]
    (let [[new-branch-idx & new-branch-args] new-args
          old-branch-idx (:branch-idx old-trace)
          old-sub-trace  (:sub-trace old-trace)]
      (if (= new-branch-idx old-branch-idx)
        ;; Same branch: delegate regenerate
        (let [sub-sel (selection/get-subselection sel new-branch-idx)
              {new-sub-trace :trace
               w             :weight}
              (gf/regenerate old-sub-trace
                             (vec new-branch-args)
                             (repeat (count new-branch-args) diff/no-change)
                             sub-sel)
              new-score (trace/get-score new-sub-trace)]
          {:trace  (->SwitchTrace this new-args new-branch-idx new-sub-trace new-score)
           :weight w
           :change diff/unknown-change})
        ;; Branch changed: simulate new, weight = 0 - old projected score
        (let [new-branch    (nth branches new-branch-idx)
              new-sub-trace (gf/simulate new-branch (vec new-branch-args))
              new-score     (trace/get-score new-sub-trace)
              old-projected (trace/project old-sub-trace
                                           (selection/get-subselection sel old-branch-idx))
              weight        (- 0.0 old-projected)]
          {:trace  (->SwitchTrace this new-args new-branch-idx new-sub-trace new-score)
           :weight weight
           :change diff/unknown-change})))))

;; ## Public API

(defn switch-gen-fn
  "Create a Switch combinator from a vector of branch generative functions.
   First arg when called is the branch index (0-based), remaining args passed to branch."
  [branches]
  (->SwitchGenerativeFn (vec branches)))
