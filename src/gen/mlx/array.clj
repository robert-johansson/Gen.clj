(ns gen.mlx.array
  "Clojure-idiomatic wrapper around MLX arrays.

   MLXArray implements core Clojure abstractions:
   - `deref`/`@`  — forces evaluation, returns Clojure value (like delay/future)
   - `count`      — total number of elements
   - `seq`        — elements as a lazy Clojure seq (forces eval)
   - `nth`        — indexed element access (forces eval)
   - `reduce`     — efficient fold over elements (forces eval)
   - `meta`/`with-meta` — metadata support
   - keyword access — `(:shape x)`, `(:dtype x)`, `(:ndim x)`

   All arithmetic returns unevaluated (lazy) arrays — the computation
   graph is only materialized when `deref`, `seq`, or extraction fns are called.
   This mirrors Clojure's philosophy of deferred computation (cf. lazy seqs,
   delays, transducers).

   Arithmetic ops accept plain numbers (auto-coerced to scalar arrays)
   and support variadic application, following Clojure's numeric conventions:
     (add x 3.0)       — number coerced to scalar
     (add x y z)       — variadic via reduce
     (sub x)           — unary negation, like (- x)"
  (:refer-clojure :exclude [abs])
  (:require [gen.mlx.ffi :as ffi])
  (:import [clojure.lang Counted IDeref IFn ILookup IMeta Indexed IObj
            IReduce IReduceInit Seqable]
           [java.util.concurrent.atomic AtomicBoolean]))

;; ---------------------------------------------------------------------------
;; dtype name mapping
;; ---------------------------------------------------------------------------

(def ^:private dtype-names
  {ffi/dtype-bool    :bool
   ffi/dtype-uint8   :uint8
   ffi/dtype-uint16  :uint16
   ffi/dtype-uint32  :uint32
   ffi/dtype-uint64  :uint64
   ffi/dtype-int8    :int8
   ffi/dtype-int16   :int16
   ffi/dtype-int32   :int32
   ffi/dtype-int64   :int64
   ffi/dtype-float16 :float16
   ffi/dtype-float32 :float32
   ffi/dtype-float64 :float64
   ffi/dtype-bfloat16 :bfloat16
   ffi/dtype-complex64 :complex64})

;; ---------------------------------------------------------------------------
;; Forward declarations for helpers used in deftype
;; ---------------------------------------------------------------------------

(declare ->double ->vec eval! wrap-handle ensure-array neg)

;; ---------------------------------------------------------------------------
;; MLXArray — wraps an mlx_array handle
;;
;; Implements Clojure's core abstractions so that MLX arrays feel native:
;;   @x          — like deref on a delay: forces computation, returns value
;;   (count x)   — number of elements
;;   (seq x)     — Clojure seq of elements (forces eval)
;;   (nth x i)   — element at index (forces eval)
;;   (reduce f x)— fold over elements (forces eval)
;;   (:shape x)  — property lookup (shape, dtype, ndim, size)
;;   (x :shape)  — same, via IFn
;;   (meta x)    — metadata
;;
;; Fields:
;;   handle — FFI handle map {:ctx <MemorySegment>}
;;   freed  — AtomicBoolean shared with Cleaner to prevent double-free
;;   _meta  — Clojure metadata (nil for no metadata)
;;   _ref   — strong ref to root MLXArray for withMeta copies (prevents GC)
;; ---------------------------------------------------------------------------

(deftype MLXArray [handle ^AtomicBoolean freed _meta _ref]
  IDeref
  ;; Like delay/future: @x forces evaluation and returns a Clojure value.
  ;; - 0-d (scalar) → returns a double
  ;; - 1-d+ → returns a Clojure vector of doubles
  (deref [this]
    (if (zero? (ffi/array-ndim handle))
      (->double this)
      (->vec this)))

  Counted
  ;; (count x) returns the total number of elements.
  (count [_]
    (int (ffi/array-size handle)))

  Seqable
  ;; (seq x) returns a Clojure seq of the array elements.
  ;; Forces evaluation, like (seq (range ...)) forces lazy seq realization.
  (seq [_]
    (let [n (ffi/array-size handle)]
      (when (pos? n)
        (ffi/array-eval handle)
        (let [data (ffi/array-data-float32 handle)]
          (clojure.core/seq (mapv double data))))))

  Indexed
  ;; (nth x i) returns the ith element (flat index) as a double.
  ;; Forces evaluation, like nth on a lazy seq realizes up to that point.
  (nth [_ i]
    (let [n (ffi/array-size handle)]
      (if (and (>= i 0) (< i n))
        (do (ffi/array-eval handle)
            (double (ffi/array-item-at handle i)))
        (throw (IndexOutOfBoundsException.
                (str "Index " i " out of bounds for array of size " n))))))
  (nth [_ i not-found]
    (let [n (ffi/array-size handle)]
      (if (and (>= i 0) (< i n))
        (do (ffi/array-eval handle)
            (double (ffi/array-item-at handle i)))
        not-found)))

  IReduceInit
  ;; (reduce f init x) — fold with initial value.
  (reduce [_ f start]
    (let [n (ffi/array-size handle)]
      (if (zero? n)
        start
        (do (ffi/array-eval handle)
            (loop [i 0 ret start]
              (if (>= i n)
                ret
                (let [ret (f ret (double (ffi/array-item-at handle i)))]
                  (if (reduced? ret)
                    @ret
                    (recur (inc i) ret)))))))))

  IReduce
  ;; (reduce f x) — fold without initial value.
  ;; Empty: calls (f). Single element: returns it. Otherwise: left fold.
  (reduce [_ f]
    (let [n (ffi/array-size handle)]
      (if (zero? n)
        (f)
        (do (ffi/array-eval handle)
            (loop [i 1
                   ret (double (ffi/array-item-at handle 0))]
              (if (>= i n)
                ret
                (let [ret (f ret (double (ffi/array-item-at handle i)))]
                  (if (reduced? ret)
                    @ret
                    (recur (inc i) ret)))))))))

  IMeta
  (meta [_] _meta)

  IObj
  ;; (with-meta x m) — returns a new MLXArray sharing the same handle.
  ;; The copy holds a strong ref (_ref) to the root array, preventing
  ;; GC of the Cleaner-registered original while the copy is alive.
  (withMeta [this m]
    (MLXArray. handle freed m (or _ref this)))

  ILookup
  ;; Keyword access for array metadata:
  ;;   (:shape x) → [3 4]
  ;;   (:dtype x) → :float32
  ;;   (:ndim x)  → 2
  ;;   (:size x)  → 12
  (valAt [this k]
    (.valAt this k nil))
  (valAt [_ k not-found]
    (case k
      :shape (ffi/array-shape handle)
      :dtype (get dtype-names (ffi/array-dtype handle) :unknown)
      :ndim  (ffi/array-ndim handle)
      :size  (ffi/array-size handle)
      not-found))

  IFn
  ;; (x :shape) — same as (:shape x), for symmetry with Clojure maps.
  (invoke [this k]
    (.valAt this k nil))
  (invoke [this k not-found]
    (.valAt this k not-found))

  Object
  ;; toString never forces evaluation — like how Clojure's delay shows
  ;; :pending without forcing. Shape, dtype, and size are metadata,
  ;; always available without triggering the computation graph.
  (toString [_]
    (let [sh (ffi/array-shape handle)
          dt (get dtype-names (ffi/array-dtype handle) :unknown)
          n  (ffi/array-size handle)]
      (str "#mlx/array {:shape " sh " :dtype " dt " :size " n "}")))
  ;; Value-based equality — two arrays are equal if they have the same
  ;; shape, dtype, and data. Forces evaluation (asking "are these equal?"
  ;; inherently requires knowing the values).
  ;; Metadata does NOT affect equality, following Clojure convention.
  (equals [_ other]
    (and (instance? MLXArray other)
         (let [^MLXArray o other
               h2 (.-handle o)]
           (or (identical? handle h2)
               (and (= (ffi/array-ndim handle) (ffi/array-ndim h2))
                    (= (ffi/array-dtype handle) (ffi/array-dtype h2))
                    (= (ffi/array-shape handle) (ffi/array-shape h2))
                    (let [n (ffi/array-size handle)]
                      (or (zero? n)
                          (do (ffi/array-eval handle)
                              (ffi/array-eval h2)
                              (= (ffi/array-data-float32 handle)
                                 (ffi/array-data-float32 h2))))))))))
  ;; hashCode consistent with value-based equals.
  ;; Metadata does NOT affect hash, following Clojure convention.
  (hashCode [_]
    (ffi/array-eval handle)
    (let [sh   (ffi/array-shape handle)
          dt   (ffi/array-dtype handle)
          data (ffi/array-data-float32 handle)]
      (unchecked-int
       (-> (hash sh)
           (bit-xor (hash dt))
           (bit-xor (hash data)))))))

(defmethod print-method MLXArray [^MLXArray a ^java.io.Writer w]
  (.write w (.toString a)))

;; ---------------------------------------------------------------------------
;; GC-managed memory — arrays are freed automatically when unreachable.
;; Like any Clojure value: allocate, use, forget. The GC handles the rest.
;; ---------------------------------------------------------------------------

(defonce ^:private ^java.lang.ref.Cleaner cleaner
  (java.lang.ref.Cleaner/create))

(defn wrap-handle
  "Wrap a raw FFI array handle as a GC-managed MLXArray.
   Registers a clean-up action so that `mlx_array_free` is called
   automatically when the MLXArray becomes unreachable — no manual
   memory management required."
  [handle]
  (let [freed (AtomicBoolean. false)
        arr   (->MLXArray handle freed nil nil)]
    ;; The cleaning action captures `handle` and `freed`, NOT `arr`.
    ;; This avoids preventing GC of the MLXArray itself.
    (.register cleaner arr
               (reify Runnable
                 (run [_]
                   (when (.compareAndSet freed false true)
                     (ffi/array-free handle)))))
    arr))

(defn mlx-array? [x]
  (instance? MLXArray x))

(defn handle
  "Get the raw FFI handle from an MLXArray."
  [^MLXArray a]
  (.-handle a))

;; ---------------------------------------------------------------------------
;; Constructors
;; ---------------------------------------------------------------------------

(defn scalar
  "Create a 0-d (scalar) MLXArray from a number.
   Defaults to float32."
  [v]
  (wrap-handle (ffi/array-new-float (float v))))

(defn from-vec
  "Create a 1-d MLXArray from a Clojure vector of numbers."
  [v]
  (let [fa (float-array (map float v))]
    (wrap-handle (ffi/array-new-data fa [(count v)]))))

(defn array
  "Create an MLXArray from flat float data and a shape vector.
   `data` is a sequence of numbers, `shape` is a vector of ints."
  [data shape]
  (let [fa (float-array (map float data))]
    (wrap-handle (ffi/array-new-data fa shape))))

;; ---------------------------------------------------------------------------
;; Auto-coercion — numbers become scalar arrays automatically
;; ---------------------------------------------------------------------------

(defn- ensure-array
  "Coerce x to an MLXArray if it isn't one already.
   Numbers are wrapped as scalar float32 arrays.
   Follows Clojure's convention of auto-promoting numeric types."
  [x]
  (if (mlx-array? x)
    x
    (scalar (double x))))

;; ---------------------------------------------------------------------------
;; Properties (also available via ILookup: (:shape x), (:dtype x), etc.)
;; ---------------------------------------------------------------------------

(defn shape
  "Get the shape of an MLXArray as a Clojure vector."
  [^MLXArray a]
  (ffi/array-shape (handle a)))

(defn ndim
  "Get the number of dimensions."
  [^MLXArray a]
  (ffi/array-ndim (handle a)))

(defn size
  "Get the total number of elements."
  [^MLXArray a]
  (ffi/array-size (handle a)))

;; ---------------------------------------------------------------------------
;; Evaluation & extraction
;; ---------------------------------------------------------------------------

(defn eval!
  "Force evaluation of the MLX computation graph for this array.
   Returns the array (for chaining).
   Analogous to `force` on a delay — the value is already determined,
   this just ensures it is materialized."
  [^MLXArray a]
  (ffi/array-eval (handle a))
  a)

(defn ->double
  "Extract a scalar value as a JVM double. Forces evaluation."
  [^MLXArray a]
  (eval! a)
  (double (ffi/array-item-float32 (handle a))))

(defn ->vec
  "Extract all elements as a Clojure vector of doubles. Forces evaluation."
  [^MLXArray a]
  (eval! a)
  (mapv double (ffi/array-data-float32 (handle a))))

;; ---------------------------------------------------------------------------
;; Memory management
;;
;; Automatic: the Cleaner frees native memory when the MLXArray is GC'd.
;; Optional: free! provides eager release for performance-sensitive code.
;; ---------------------------------------------------------------------------

(defn free!
  "Eagerly free the underlying MLX array.
   Optional — the GC will free it automatically.
   Safe to call multiple times (idempotent).
   After this call, the MLXArray is invalid."
  [^MLXArray a]
  (when (.compareAndSet (.-freed a) false true)
    (ffi/array-free (handle a))))

;; ---------------------------------------------------------------------------
;; Arithmetic — all return lazy (unevaluated) MLXArrays
;;
;; These build a computation graph without performing any work.
;; The graph is only evaluated when a value is demanded via deref,
;; seq, ->double, or ->vec — mirroring how Clojure lazy seqs
;; defer computation until consumption.
;;
;; All ops accept plain numbers (auto-coerced to scalar arrays)
;; and support variadic application, following Clojure conventions:
;;   (add x 3.0)       — number coerced to scalar
;;   (add x y z)       — variadic via reduce
;;   (sub x)           — unary negation, like (- x)
;; ---------------------------------------------------------------------------

(defn add
  "Element-wise addition. Accepts MLXArrays and/or numbers.
   Variadic: (add x y z) = (add (add x y) z)."
  ([a] (ensure-array a))
  ([a b]
   (let [a (ensure-array a) b (ensure-array b)]
     (wrap-handle (ffi/mlx-add (handle a) (handle b)))))
  ([a b & more]
   (reduce add (add a b) more)))

(defn sub
  "Element-wise subtraction. Accepts MLXArrays and/or numbers.
   Unary: (sub x) = negation, like Clojure's (- x).
   Variadic: (sub x y z) = (sub (sub x y) z)."
  ([a] (neg (ensure-array a)))
  ([a b]
   (let [a (ensure-array a) b (ensure-array b)]
     (wrap-handle (ffi/mlx-subtract (handle a) (handle b)))))
  ([a b & more]
   (reduce sub (sub a b) more)))

(defn mul
  "Element-wise multiplication. Accepts MLXArrays and/or numbers.
   Variadic: (mul x y z) = (mul (mul x y) z)."
  ([a] (ensure-array a))
  ([a b]
   (let [a (ensure-array a) b (ensure-array b)]
     (wrap-handle (ffi/mlx-multiply (handle a) (handle b)))))
  ([a b & more]
   (reduce mul (mul a b) more)))

(defn div
  "Element-wise division. Accepts MLXArrays and/or numbers.
   Unary: (div x) = reciprocal (1/x), like Clojure's (/ x).
   Variadic: (div x y z) = (div (div x y) z)."
  ([a]
   (let [a (ensure-array a)]
     (wrap-handle (ffi/mlx-divide (handle (scalar 1.0)) (handle a)))))
  ([a b]
   (let [a (ensure-array a) b (ensure-array b)]
     (wrap-handle (ffi/mlx-divide (handle a) (handle b)))))
  ([a b & more]
   (reduce div (div a b) more)))

(defn neg
  "Element-wise negation."
  [a]
  (let [a (ensure-array a)]
    (wrap-handle (ffi/mlx-negative (handle a)))))

(defn exp
  "Element-wise exponential."
  [a]
  (let [a (ensure-array a)]
    (wrap-handle (ffi/mlx-exp (handle a)))))

(defn log
  "Element-wise natural logarithm."
  [a]
  (let [a (ensure-array a)]
    (wrap-handle (ffi/mlx-log (handle a)))))

(defn square
  "Element-wise square."
  [a]
  (let [a (ensure-array a)]
    (wrap-handle (ffi/mlx-square (handle a)))))

(defn sqrt
  "Element-wise square root."
  [a]
  (let [a (ensure-array a)]
    (wrap-handle (ffi/mlx-sqrt (handle a)))))

(defn abs
  "Element-wise absolute value."
  [a]
  (let [a (ensure-array a)]
    (wrap-handle (ffi/mlx-abs (handle a)))))

(defn sum
  "Reduce-sum over all axes."
  [a]
  (let [a (ensure-array a)]
    (wrap-handle (ffi/mlx-sum (handle a)))))
