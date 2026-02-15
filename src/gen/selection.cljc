(ns gen.selection
  "Selection protocol and types for specifying subsets of random choices
   in a trace. Used by `project`, `regenerate`, and MH kernels."
  (:refer-clojure :exclude [complement]))

;; ## Protocol

(defprotocol ISelection
  (includes? [sel addr]
    "Returns true if `addr` is directly included in this selection.")
  (get-subselection [sel addr]
    "Returns the sub-selection for nested addresses under `addr`."))

;; ## Types

(deftype AllSelection []
  ISelection
  (includes? [_ _] true)
  (get-subselection [this _] this)

  #?@(:clj
      [Object
       (equals [_ o] (instance? AllSelection o))
       (hashCode [_] 1)
       (toString [_] "#gen/selection :all")]

      :cljs
      [Object
       (equiv [_ o] (instance? AllSelection o))]))

(deftype EmptySelection []
  ISelection
  (includes? [_ _] false)
  (get-subselection [this _] this)

  #?@(:clj
      [Object
       (equals [_ o] (instance? EmptySelection o))
       (hashCode [_] 0)
       (toString [_] "#gen/selection :none")]

      :cljs
      [Object
       (equiv [_ o] (instance? EmptySelection o))]))

(def ALL
  "Selection that includes everything."
  (->AllSelection))

(def NONE
  "Selection that includes nothing."
  (->EmptySelection))

(deftype SetSelection [s]
  ISelection
  (includes? [_ addr] (contains? s addr))
  (get-subselection [_ addr]
    (if (contains? s addr) ALL NONE))

  #?@(:clj
      [Object
       (equals [_ o]
               (and (instance? SetSelection o)
                    (= s (.-s ^SetSelection o))))
       (hashCode [_] (.hashCode s))
       (toString [_] (str "#gen/selection " (pr-str s)))]

      :cljs
      [Object
       (equiv [_ o]
              (and (instance? SetSelection o)
                   (= s (.-s ^SetSelection o))))]))

(deftype HierarchicalSelection [m]
  ISelection
  (includes? [_ addr] (contains? m addr))
  (get-subselection [_ addr]
    (get m addr NONE))

  #?@(:clj
      [Object
       (equals [_ o]
               (and (instance? HierarchicalSelection o)
                    (= m (.-m ^HierarchicalSelection o))))
       (hashCode [_] (.hashCode m))
       (toString [_] (str "#gen/selection " (pr-str m)))]

      :cljs
      [Object
       (equiv [_ o]
              (and (instance? HierarchicalSelection o)
                   (= m (.-m ^HierarchicalSelection o))))]))

(deftype ComplementSelection [sel]
  ISelection
  (includes? [_ addr]
    (not (includes? sel addr)))
  (get-subselection [_ addr]
    (ComplementSelection. (get-subselection sel addr)))

  #?@(:clj
      [Object
       (equals [_ o]
               (and (instance? ComplementSelection o)
                    (= sel (.-sel ^ComplementSelection o))))
       (hashCode [_] (bit-not (.hashCode sel)))
       (toString [_] (str "(complement " sel ")"))]

      :cljs
      [Object
       (equiv [_ o]
              (and (instance? ComplementSelection o)
                   (= sel (.-sel ^ComplementSelection o))))]))

(deftype UnionSelection [a b]
  ISelection
  (includes? [_ addr]
    (or (includes? a addr)
        (includes? b addr)))
  (get-subselection [_ addr]
    (UnionSelection. (get-subselection a addr)
                     (get-subselection b addr)))

  #?@(:clj
      [Object
       (equals [_ o]
               (and (instance? UnionSelection o)
                    (= a (.-a ^UnionSelection o))
                    (= b (.-b ^UnionSelection o))))
       (hashCode [_] (unchecked-add (.hashCode a) (.hashCode b)))
       (toString [_] (str "(union " a " " b ")"))]

      :cljs
      [Object
       (equiv [_ o]
              (and (instance? UnionSelection o)
                   (= a (.-a ^UnionSelection o))
                   (= b (.-b ^UnionSelection o))))]))

(deftype IntersectionSelection [a b]
  ISelection
  (includes? [_ addr]
    (and (includes? a addr)
         (includes? b addr)))
  (get-subselection [_ addr]
    (IntersectionSelection. (get-subselection a addr)
                            (get-subselection b addr)))

  #?@(:clj
      [Object
       (equals [_ o]
               (and (instance? IntersectionSelection o)
                    (= a (.-a ^IntersectionSelection o))
                    (= b (.-b ^IntersectionSelection o))))
       (hashCode [_] (bit-and (.hashCode a) (.hashCode b)))
       (toString [_] (str "(intersection " a " " b ")"))]

      :cljs
      [Object
       (equiv [_ o]
              (and (instance? IntersectionSelection o)
                   (= a (.-a ^IntersectionSelection o))
                   (= b (.-b ^IntersectionSelection o))))]))

;; ## Extend Clojure sets to act as selections

#?(:clj
   (extend-type clojure.lang.IPersistentSet
     ISelection
     (includes? [s addr] (contains? s addr))
     (get-subselection [s addr]
       (if (contains? s addr) ALL NONE)))

   :cljs
   (extend-type cljs.core/PersistentHashSet
     ISelection
     (includes? [s addr] (contains? s addr))
     (get-subselection [s addr]
       (if (contains? s addr) ALL NONE))))

;; ## Constructors

(defn- addr->hierarchical
  "Convert a vector address path into a nested HierarchicalSelection."
  [[k & ks]]
  (if ks
    (->HierarchicalSelection {k (addr->hierarchical ks)})
    (->HierarchicalSelection {k ALL})))

(defn select
  "Create a selection from addresses. Flat keywords create a SetSelection.
   Vector addresses create nested HierarchicalSelections.

   Examples:
     (select :slope :noise)          => flat set selection
     (select [:data 1 :y])           => nested hierarchical selection
     (select :slope [:data 1 :y])    => mixed hierarchical selection"
  [& addrs]
  (if (every? #(not (vector? %)) addrs)
    (->SetSelection (set addrs))
    (let [merged (reduce
                  (fn [m addr]
                    (if (vector? addr)
                      (let [sel (addr->hierarchical addr)]
                        (clojure.core/merge m (.-m ^HierarchicalSelection sel)))
                      (assoc m addr ALL)))
                  {}
                  addrs)]
      (->HierarchicalSelection merged))))

(defn complement
  "Returns a selection that includes everything not in `sel`."
  [sel]
  (->ComplementSelection sel))

(defn union
  "Returns a selection that includes anything in either `a` or `b`."
  [a b]
  (->UnionSelection a b))

(defn intersection
  "Returns a selection that includes only what's in both `a` and `b`."
  [a b]
  (->IntersectionSelection a b))

(defn selection?
  "Returns true if `x` implements ISelection."
  [x]
  (satisfies? ISelection x))
