(ns gen.mlx.ffi
  "Raw Coffi bindings to the mlx-c library (C API for MLX).
   All mlx-c types (mlx_array, mlx_stream, mlx_closure, etc.) are
   structs containing a single `void* ctx` field, passed by value."
  (:require [coffi.ffi :as ffi]
            [coffi.mem :as mem]
            [clojure.java.io :as io]))

;; ---------------------------------------------------------------------------
;; Library loading
;; ---------------------------------------------------------------------------

(defn find-mlx-lib
  "Locate the mlx-c shared library. Searches:
   1. MLX_LIB_PATH env var
   2. vendor/mlx-c/build/lib/ relative to project root"
  []
  (or (System/getenv "MLX_LIB_PATH")
      (let [candidates ["vendor/mlx-c/build/lib/libmlxc.dylib"
                        "vendor/mlx-c/build/lib/libmlx.dylib"
                        "vendor/mlx-c/build/libmlxc.dylib"
                        "vendor/mlx-c/build/libmlx.dylib"]]
        (some (fn [path]
                (let [f (io/file path)]
                  (when (.exists f)
                    (.getAbsolutePath f))))
              candidates))
      (throw (ex-info "Cannot find mlx-c shared library. Set MLX_LIB_PATH or build vendor/mlx-c."
                      {}))))

(defonce ^:private _load-lib
  (ffi/load-library (find-mlx-lib)))

;; ---------------------------------------------------------------------------
;; Type aliases — all mlx-c handle types are struct { void* ctx; }
;; ---------------------------------------------------------------------------

(mem/defalias ::array
  [::mem/struct [[:ctx ::mem/pointer]]])

(mem/defalias ::stream
  [::mem/struct [[:ctx ::mem/pointer]]])

(mem/defalias ::vector-array
  [::mem/struct [[:ctx ::mem/pointer]]])

(mem/defalias ::closure
  [::mem/struct [[:ctx ::mem/pointer]]])

(mem/defalias ::closure-value-and-grad
  [::mem/struct [[:ctx ::mem/pointer]]])

(mem/defalias ::string
  [::mem/struct [[:ctx ::mem/pointer]]])

;; ---------------------------------------------------------------------------
;; dtype enum (mirrors mlx_dtype)
;; ---------------------------------------------------------------------------

(def dtype-bool     0)
(def dtype-uint8    1)
(def dtype-uint16   2)
(def dtype-uint32   3)
(def dtype-uint64   4)
(def dtype-int8     5)
(def dtype-int16    6)
(def dtype-int32    7)
(def dtype-int64    8)
(def dtype-float16  9)
(def dtype-float32 10)
(def dtype-float64 11)
(def dtype-bfloat16 12)
(def dtype-complex64 13)

;; ---------------------------------------------------------------------------
;; Helper: call-with-output — allocates output struct, calls native fn,
;; checks return code, deserializes result.
;; ---------------------------------------------------------------------------

(defn- check-status!
  "Throw if mlx-c returns non-zero status."
  [status fn-name]
  (when-not (zero? status)
    (throw (ex-info (str fn-name " failed with status " status)
                    {:status status :fn fn-name}))))

;; ---------------------------------------------------------------------------
;; Stream
;; ---------------------------------------------------------------------------

(def ^:private raw-default-cpu-stream
  (ffi/cfn "mlx_default_cpu_stream_new" [] ::stream))

(def ^:private raw-default-gpu-stream
  (ffi/cfn "mlx_default_gpu_stream_new" [] ::stream))

(def ^:private raw-stream-free
  (ffi/cfn "mlx_stream_free" [::stream] ::mem/int))

(defonce default-stream
  (raw-default-cpu-stream))

(defn default-gpu-stream []
  (raw-default-gpu-stream))

;; ---------------------------------------------------------------------------
;; Array lifecycle
;; ---------------------------------------------------------------------------

(def array-new-float
  "Create a scalar float32 array. Returns ::array map."
  (ffi/cfn "mlx_array_new_float32" [::mem/float] ::array))

(def array-new-double
  "Create a scalar float64 array. Returns ::array map."
  (ffi/cfn "mlx_array_new_float64" [::mem/double] ::array))

(def array-new-int
  "Create a scalar int32 array. Returns ::array map."
  (ffi/cfn "mlx_array_new_int" [::mem/int] ::array))

(def ^:private raw-array-new-data
  "mlx_array mlx_array_new_data(const void* data, const int* shape, int dim, mlx_dtype dtype)"
  (ffi/cfn "mlx_array_new_data"
           [::mem/pointer ::mem/pointer ::mem/int ::mem/int]
           ::array))

(def array-eval
  "Force evaluation of an array. Returns status int."
  (ffi/cfn "mlx_array_eval" [::array] ::mem/int))

(def array-free
  "Free an array. Returns status int."
  (ffi/cfn "mlx_array_free" [::array] ::mem/int))

(def array-ndim
  "Get number of dimensions."
  (ffi/cfn "mlx_array_ndim" [::array] ::mem/long))

(def array-size
  "Get total number of elements."
  (ffi/cfn "mlx_array_size" [::array] ::mem/long))

(def array-dtype
  "Get element dtype as int."
  (ffi/cfn "mlx_array_dtype" [::array] ::mem/int))

(def ^:private raw-array-shape
  "Returns pointer to shape array (int*)."
  (ffi/cfn "mlx_array_shape" [::array] ::mem/pointer))

(def ^:private raw-array-item-float32
  "int mlx_array_item_float32(float* res, const mlx_array arr)"
  (ffi/cfn "mlx_array_item_float32" [::mem/pointer ::array] ::mem/int))

(def ^:private raw-array-item-float64
  "int mlx_array_item_float64(double* res, const mlx_array arr)"
  (ffi/cfn "mlx_array_item_float64" [::mem/pointer ::array] ::mem/int))

(def ^:private raw-array-data-float32
  "const float* mlx_array_data_float32(const mlx_array arr)"
  (ffi/cfn "mlx_array_data_float32" [::array] ::mem/pointer))

;; ---------------------------------------------------------------------------
;; Array high-level helpers
;; ---------------------------------------------------------------------------

(defn array-new-data
  "Create an array from a float array buffer.
   `data` is a float array, `shape` is a vector of ints."
  [^floats data shape]
  (let [arena (mem/auto-arena)
        ;; Write float data
        data-seg (mem/alloc (* 4 (alength data)) arena)
        _ (dotimes [i (alength data)]
            (.set (.reinterpret data-seg (* 4 (alength data)))
                  java.lang.foreign.ValueLayout/JAVA_FLOAT
                  (* i 4)
                  (aget data i)))
        ;; Write shape as int array
        ndim (count shape)
        shape-seg (mem/alloc (* 4 ndim) arena)
        _ (dotimes [i ndim]
            (.set (.reinterpret shape-seg (* 4 ndim))
                  java.lang.foreign.ValueLayout/JAVA_INT
                  (* i 4)
                  (int (nth shape i))))]
    (raw-array-new-data data-seg shape-seg (int ndim) dtype-float32)))

(defn array-new-data-int
  "Create an array from an int array buffer.
   `data` is an int array, `shape` is a vector of ints."
  [^ints data shape]
  (let [arena (mem/auto-arena)
        data-seg (mem/alloc (* 4 (alength data)) arena)
        _ (dotimes [i (alength data)]
            (.set (.reinterpret data-seg (* 4 (alength data)))
                  java.lang.foreign.ValueLayout/JAVA_INT
                  (* i 4)
                  (aget data i)))
        ndim (count shape)
        shape-seg (mem/alloc (* 4 ndim) arena)
        _ (dotimes [i ndim]
            (.set (.reinterpret shape-seg (* 4 ndim))
                  java.lang.foreign.ValueLayout/JAVA_INT
                  (* i 4)
                  (int (nth shape i))))]
    (raw-array-new-data data-seg shape-seg (int ndim) dtype-int32)))

(defn array-item-float32
  "Extract scalar value as float from an array."
  [arr]
  (let [arena (mem/auto-arena)
        res-seg (mem/alloc 4 arena)
        status (raw-array-item-float32 res-seg arr)]
    (check-status! status "mlx_array_item_float32")
    (.get (.reinterpret res-seg 4)
          java.lang.foreign.ValueLayout/JAVA_FLOAT
          0)))

(defn array-item-float64
  "Extract scalar value as double from an array."
  [arr]
  (let [arena (mem/auto-arena)
        res-seg (mem/alloc 8 arena)
        status (raw-array-item-float64 res-seg arr)]
    (check-status! status "mlx_array_item_float64")
    (.get (.reinterpret res-seg 8)
          java.lang.foreign.ValueLayout/JAVA_DOUBLE
          0)))

(defn array-shape
  "Get the shape of an array as a Clojure vector of ints."
  [arr]
  (let [ndim (array-ndim arr)]
    (if (zero? ndim)
      []
      (let [shape-ptr (raw-array-shape arr)
            shape-seg (.reinterpret shape-ptr (* 4 ndim))]
        (mapv (fn [i]
                (.get shape-seg java.lang.foreign.ValueLayout/JAVA_INT (* i 4)))
              (range ndim))))))

(defn array-data-float32
  "Get the raw float data as a Clojure vector. Array must be evaluated."
  [arr]
  (let [size (array-size arr)]
    (if (zero? size)
      []
      (let [ptr (raw-array-data-float32 arr)]
        (when (.equals ptr java.lang.foreign.MemorySegment/NULL)
          (throw (ex-info "Array not evaluated — call array-eval first" {})))
        (let [seg (.reinterpret ptr (* 4 size))]
          (mapv (fn [i]
                  (.get seg java.lang.foreign.ValueLayout/JAVA_FLOAT (* i 4)))
                (range size)))))))

(defn array-item-at
  "Get the float32 value at a flat index. Array must be evaluated first."
  [arr idx]
  (let [ptr (raw-array-data-float32 arr)]
    (when (.equals ptr java.lang.foreign.MemorySegment/NULL)
      (throw (ex-info "Array not evaluated — call array-eval first" {})))
    (let [seg (.reinterpret ptr (* 4 (inc (long idx))))]
      (.get seg java.lang.foreign.ValueLayout/JAVA_FLOAT (* (int idx) 4)))))

;; ---------------------------------------------------------------------------
;; Arithmetic ops — all follow pattern:
;;   int mlx_op(mlx_array* res, mlx_array a, mlx_array b, mlx_stream s)
;; ---------------------------------------------------------------------------

(defn- defop-binary
  "Create a binary op function from a C symbol name."
  [sym-name]
  (let [raw-fn (ffi/cfn sym-name
                        [::mem/pointer ::array ::array ::stream]
                        ::mem/int)]
    (fn [a b]
      (let [arena (mem/auto-arena)
            res (mem/alloc-instance ::array arena)
            status (raw-fn res a b default-stream)]
        (check-status! status sym-name)
        (mem/deserialize-from res ::array)))))

(defn- defop-unary
  "Create a unary op function from a C symbol name."
  [sym-name]
  (let [raw-fn (ffi/cfn sym-name
                        [::mem/pointer ::array ::stream]
                        ::mem/int)]
    (fn [a]
      (let [arena (mem/auto-arena)
            res (mem/alloc-instance ::array arena)
            status (raw-fn res a default-stream)]
        (check-status! status sym-name)
        (mem/deserialize-from res ::array)))))

(def mlx-add      (defop-binary "mlx_add"))
(def mlx-subtract (defop-binary "mlx_subtract"))
(def mlx-multiply (defop-binary "mlx_multiply"))
(def mlx-divide   (defop-binary "mlx_divide"))

(def mlx-negative      (defop-unary "mlx_negative"))
(def mlx-exp           (defop-unary "mlx_exp"))
(def mlx-log           (defop-unary "mlx_log"))
(def mlx-square        (defop-unary "mlx_square"))
(def mlx-sqrt          (defop-unary "mlx_sqrt"))
(def mlx-abs           (defop-unary "mlx_abs"))
(def mlx-stop-gradient (defop-unary "mlx_stop_gradient"))

(def mlx-less      (defop-binary "mlx_less"))
(def mlx-greater   (defop-binary "mlx_greater"))
(def mlx-logaddexp (defop-binary "mlx_logaddexp"))

;; where: int mlx_where(mlx_array* res, mlx_array cond, mlx_array x, mlx_array y, mlx_stream s)
(def ^:private raw-mlx-where
  (ffi/cfn "mlx_where"
           [::mem/pointer ::array ::array ::array ::stream]
           ::mem/int))

(defn mlx-where
  "Ternary conditional: where(cond, x, y) — x if cond else y, element-wise."
  [condition x y]
  (let [arena (mem/auto-arena)
        res (mem/alloc-instance ::array arena)
        status (raw-mlx-where res condition x y default-stream)]
    (check-status! status "mlx_where")
    (mem/deserialize-from res ::array)))

;; sum: int mlx_sum(mlx_array* res, mlx_array a, bool keepdims, mlx_stream s)
(def ^:private raw-mlx-sum
  (ffi/cfn "mlx_sum"
           [::mem/pointer ::array ::mem/byte ::stream]
           ::mem/int))

(defn mlx-sum
  "Reduce-sum over all axes."
  [a]
  (let [arena (mem/auto-arena)
        res (mem/alloc-instance ::array arena)
        status (raw-mlx-sum res a (byte 0) default-stream)]
    (check-status! status "mlx_sum")
    (mem/deserialize-from res ::array)))

;; sum_axis: int mlx_sum_axis(mlx_array* res, mlx_array a, int axis, bool keepdims, mlx_stream s)
(def ^:private raw-mlx-sum-axis
  (ffi/cfn "mlx_sum_axis"
           [::mem/pointer ::array ::mem/int ::mem/byte ::stream]
           ::mem/int))

(defn mlx-sum-axis
  "Reduce-sum along a single axis."
  [a axis keepdims]
  (let [arena (mem/auto-arena)
        res (mem/alloc-instance ::array arena)
        status (raw-mlx-sum-axis res a (int axis)
                                 (if keepdims (byte 1) (byte 0))
                                 default-stream)]
    (check-status! status "mlx_sum_axis")
    (mem/deserialize-from res ::array)))

;; ---------------------------------------------------------------------------
;; Shape manipulation ops
;; ---------------------------------------------------------------------------

;; stack_axis: int mlx_stack_axis(mlx_array* res, const mlx_vector_array arrays, int axis, mlx_stream s)
(def ^:private raw-mlx-stack-axis
  (ffi/cfn "mlx_stack_axis"
           [::mem/pointer ::vector-array ::mem/int ::stream]
           ::mem/int))

(defn mlx-stack-axis
  "Stack arrays along a new axis."
  [arrays-va axis]
  (let [arena (mem/auto-arena)
        res (mem/alloc-instance ::array arena)
        status (raw-mlx-stack-axis res arrays-va (int axis) default-stream)]
    (check-status! status "mlx_stack_axis")
    (mem/deserialize-from res ::array)))

;; expand_dims: int mlx_expand_dims(mlx_array* res, mlx_array a, int axis, mlx_stream s)
(def ^:private raw-mlx-expand-dims
  (ffi/cfn "mlx_expand_dims"
           [::mem/pointer ::array ::mem/int ::stream]
           ::mem/int))

(defn mlx-expand-dims
  "Insert a new axis at the given position."
  [a axis]
  (let [arena (mem/auto-arena)
        res (mem/alloc-instance ::array arena)
        status (raw-mlx-expand-dims res a (int axis) default-stream)]
    (check-status! status "mlx_expand_dims")
    (mem/deserialize-from res ::array)))

;; squeeze_axis: int mlx_squeeze_axis(mlx_array* res, mlx_array a, int axis, mlx_stream s)
(def ^:private raw-mlx-squeeze-axis
  (ffi/cfn "mlx_squeeze_axis"
           [::mem/pointer ::array ::mem/int ::stream]
           ::mem/int))

(defn mlx-squeeze-axis
  "Remove a length-1 axis at the given position."
  [a axis]
  (let [arena (mem/auto-arena)
        res (mem/alloc-instance ::array arena)
        status (raw-mlx-squeeze-axis res a (int axis) default-stream)]
    (check-status! status "mlx_squeeze_axis")
    (mem/deserialize-from res ::array)))

;; take_axis: int mlx_take_axis(mlx_array* res, mlx_array a, mlx_array indices, int axis, mlx_stream s)
(def ^:private raw-mlx-take-axis
  (ffi/cfn "mlx_take_axis"
           [::mem/pointer ::array ::array ::mem/int ::stream]
           ::mem/int))

(defn mlx-take-axis
  "Take elements along an axis by index."
  [a indices axis]
  (let [arena (mem/auto-arena)
        res (mem/alloc-instance ::array arena)
        status (raw-mlx-take-axis res a indices (int axis) default-stream)]
    (check-status! status "mlx_take_axis")
    (mem/deserialize-from res ::array)))

;; ---------------------------------------------------------------------------
;; Additional arithmetic/matrix ops
;; ---------------------------------------------------------------------------

(def mlx-matmul  (defop-binary "mlx_matmul"))
(def mlx-power   (defop-binary "mlx_power"))
(def mlx-floor   (defop-unary "mlx_floor"))

;; transpose: int mlx_transpose(mlx_array* res, mlx_array a, mlx_stream s)
(def mlx-transpose (defop-unary "mlx_transpose"))

;; reshape: int mlx_reshape(mlx_array* res, mlx_array a, const int* shape,
;;                          size_t shape_num, mlx_stream s)
(def ^:private raw-mlx-reshape
  (ffi/cfn "mlx_reshape"
           [::mem/pointer ::array ::mem/pointer ::mem/long ::stream]
           ::mem/int))

(defn mlx-reshape
  "Reshape an array to a new shape."
  [a shape-vec]
  (let [arena (mem/auto-arena)
        n (count shape-vec)
        shape-seg (mem/alloc (* 4 n) arena)
        _ (dotimes [i n]
            (.set (.reinterpret shape-seg (* 4 n))
                  java.lang.foreign.ValueLayout/JAVA_INT
                  (* i 4)
                  (int (nth shape-vec i))))
        res (mem/alloc-instance ::array arena)
        status (raw-mlx-reshape res a shape-seg (long n) default-stream)]
    (check-status! status "mlx_reshape")
    (mem/deserialize-from res ::array)))

;; diagonal: int mlx_diagonal(mlx_array* res, mlx_array a, int offset,
;;                            int axis1, int axis2, mlx_stream s)
(def ^:private raw-mlx-diagonal
  (ffi/cfn "mlx_diagonal"
           [::mem/pointer ::array ::mem/int ::mem/int ::mem/int ::stream]
           ::mem/int))

(defn mlx-diagonal
  "Extract diagonal from a 2-d array."
  ([a] (mlx-diagonal a 0 0 1))
  ([a offset axis1 axis2]
   (let [arena (mem/auto-arena)
         res (mem/alloc-instance ::array arena)
         status (raw-mlx-diagonal res a (int offset) (int axis1) (int axis2) default-stream)]
     (check-status! status "mlx_diagonal")
     (mem/deserialize-from res ::array))))

;; ---------------------------------------------------------------------------
;; Linear algebra ops
;; ---------------------------------------------------------------------------

;; cholesky: int mlx_linalg_cholesky(mlx_array* res, mlx_array a,
;;                                   bool upper, mlx_stream s)
(def ^:private raw-mlx-linalg-cholesky
  (ffi/cfn "mlx_linalg_cholesky"
           [::mem/pointer ::array ::mem/byte ::stream]
           ::mem/int))

(defn mlx-linalg-cholesky
  "Compute Cholesky decomposition. Returns lower-triangular L by default."
  ([a] (mlx-linalg-cholesky a false))
  ([a upper]
   (let [arena (mem/auto-arena)
         res (mem/alloc-instance ::array arena)
         status (raw-mlx-linalg-cholesky res a (if upper (byte 1) (byte 0)) default-stream)]
     (check-status! status "mlx_linalg_cholesky")
     (mem/deserialize-from res ::array))))

;; solve_triangular: int mlx_linalg_solve_triangular(mlx_array* res,
;;   mlx_array a, mlx_array b, bool upper, mlx_stream s)
(def ^:private raw-mlx-linalg-solve-triangular
  (ffi/cfn "mlx_linalg_solve_triangular"
           [::mem/pointer ::array ::array ::mem/byte ::stream]
           ::mem/int))

(defn mlx-linalg-solve-triangular
  "Solve triangular system A x = b. Default: lower-triangular."
  ([a b] (mlx-linalg-solve-triangular a b false))
  ([a b upper]
   (let [arena (mem/auto-arena)
         res (mem/alloc-instance ::array arena)
         status (raw-mlx-linalg-solve-triangular res a b
                                                  (if upper (byte 1) (byte 0))
                                                  default-stream)]
     (check-status! status "mlx_linalg_solve_triangular")
     (mem/deserialize-from res ::array))))

;; ---------------------------------------------------------------------------
;; Vector array operations
;; ---------------------------------------------------------------------------

(def vector-array-new
  "Create empty vector<array>."
  (ffi/cfn "mlx_vector_array_new" [] ::vector-array))

(def vector-array-new-value
  "Create vector<array> with a single element."
  (ffi/cfn "mlx_vector_array_new_value" [::array] ::vector-array))

(def vector-array-append-value
  "Append an array to a vector<array>."
  (ffi/cfn "mlx_vector_array_append_value" [::vector-array ::array] ::mem/int))

(def vector-array-size
  "Get size of a vector<array>."
  (ffi/cfn "mlx_vector_array_size" [::vector-array] ::mem/long))

(def ^:private raw-vector-array-get
  (ffi/cfn "mlx_vector_array_get"
           [::mem/pointer ::vector-array ::mem/long]
           ::mem/int))

(defn vector-array-get
  "Get array at index from vector<array>."
  [vec idx]
  (let [arena (mem/auto-arena)
        res (mem/alloc-instance ::array arena)
        status (raw-vector-array-get res vec (long idx))]
    (check-status! status "mlx_vector_array_get")
    (mem/deserialize-from res ::array)))

(def vector-array-free
  "Free a vector<array>."
  (ffi/cfn "mlx_vector_array_free" [::vector-array] ::mem/int))

(defn arrays->vector
  "Convert a sequence of ::array maps to a ::vector-array."
  [arrays]
  (let [va (vector-array-new)]
    (doseq [a arrays]
      (let [status (vector-array-append-value va a)]
        (check-status! status "mlx_vector_array_append_value")))
    va))

(defn vector->arrays
  "Convert a ::vector-array to a Clojure vector of ::array maps."
  [va]
  (let [n (vector-array-size va)]
    (mapv (fn [i] (vector-array-get va i)) (range n))))

;; ---------------------------------------------------------------------------
;; Closures — wrapping Clojure fns as mlx_closure via C callbacks
;;
;; Uses the Java Panama FFM API directly for the upcall stub,
;; because Coffi's [::ffi/fn ...] with struct-by-value args can
;; crash on GraalVM JDK 22 / aarch64.
;; ---------------------------------------------------------------------------

;; Strong references to prevent GC of callback functions and upcall stubs.
;; Keyed by closure ctx pointer for O(1) release when a closure is freed.
(defonce ^:private callback-refs (atom {}))

(def ^:private raw-closure-free
  (ffi/cfn "mlx_closure_free" [::closure] ::mem/int))

(defn closure-free
  "Free a closure and release its callback references."
  [cls]
  (swap! callback-refs dissoc (:ctx cls))
  (raw-closure-free cls))

(def ^:private raw-closure-apply
  "int mlx_closure_apply(mlx_vector_array* res, mlx_closure cls, const mlx_vector_array input)"
  (ffi/cfn "mlx_closure_apply"
           [::mem/pointer ::closure ::vector-array]
           ::mem/int))

(defn closure-apply
  "Apply an mlx_closure to a vector<array> input, returning vector<array> output."
  [cls input-va]
  (let [arena (mem/auto-arena)
        res (mem/alloc-instance ::vector-array arena)
        status (raw-closure-apply res cls input-va)]
    (check-status! status "mlx_closure_apply")
    (mem/deserialize-from res ::vector-array)))

;; The C callback signature for mlx_closure_new_func is:
;;   int (*fun)(mlx_vector_array*, const mlx_vector_array)
;; where mlx_vector_array is struct { void* ctx; } — an 8-byte value.
;; On aarch64, this single-member struct is passed in a register
;; identically to a plain pointer. We create the upcall stub treating
;; the struct as ADDRESS to avoid struct-value ABI issues in the JVM.

(def ^:private callback-fn-descriptor
  "FunctionDescriptor for: int callback(mlx_vector_array*, mlx_vector_array)"
  (java.lang.foreign.FunctionDescriptor/of
   java.lang.foreign.ValueLayout/JAVA_INT      ;; return: int
   (into-array java.lang.foreign.MemoryLayout
               [java.lang.foreign.ValueLayout/ADDRESS        ;; arg0: mlx_vector_array* (output pointer)
                java.lang.foreign.ValueLayout/ADDRESS])))    ;; arg1: mlx_vector_array (struct as pointer)

(def ^:private closure-new-func-descriptor
  "FunctionDescriptor for: mlx_closure mlx_closure_new_func(int (*fn)(...))"
  (java.lang.foreign.FunctionDescriptor/of
   (java.lang.foreign.MemoryLayout/structLayout
    (into-array java.lang.foreign.MemoryLayout
                [(.withName java.lang.foreign.ValueLayout/ADDRESS "ctx")]))  ;; return: mlx_closure
   (into-array java.lang.foreign.MemoryLayout
               [java.lang.foreign.ValueLayout/ADDRESS])))

(defn- make-closure-new-func-downcall
  "Create a downcall MethodHandle for mlx_closure_new_func."
  []
  (let [linker (java.lang.foreign.Linker/nativeLinker)
        sym (ffi/find-symbol "mlx_closure_new_func")]
    (when (nil? sym)
      (throw (ex-info "Symbol mlx_closure_new_func not found" {})))
    (.downcallHandle linker
                     ^java.lang.foreign.MemorySegment sym
                     ^java.lang.foreign.FunctionDescriptor closure-new-func-descriptor
                     (into-array java.lang.foreign.Linker$Option []))))

(def ^:private closure-new-func-handle
  (make-closure-new-func-downcall))

(defn closure-new
  "Wrap a Clojure function (vec<array> -> vec<array>) as an mlx_closure.
   The function receives a ::vector-array and must return a ::vector-array.
   A strong reference to the callback is held to prevent GC."
  [f]
  (let [;; Create the Java method that will be called from C
        callback-impl
        (fn [res-ptr input-va-ptr]
          (try
            (let [;; Wrap the raw pointer as a ::vector-array map
                  input-va {:ctx input-va-ptr}
                  ;; Call the Clojure function
                  result-va (f input-va)
                  ;; Write result ctx pointer into the output struct
                  res-seg (.reinterpret ^java.lang.foreign.MemorySegment res-ptr 8)]
              (.set res-seg java.lang.foreign.ValueLayout/ADDRESS 0
                    ^java.lang.foreign.MemorySegment (:ctx result-va))
              (int 0))
            (catch Throwable t
              (binding [*out* *err*]
                (println "MLX closure callback error:" (.getMessage t))
                (.printStackTrace t))
              (int 1))))

        ;; Create a MethodHandle that calls callback-impl and returns int
        ;; IFn.invoke(Object, Object) -> Object, then cast to int
        target-type (java.lang.invoke.MethodType/methodType
                     Integer/TYPE
                     (into-array Class [java.lang.foreign.MemorySegment
                                        java.lang.foreign.MemorySegment]))
        ifn-invoke-type (java.lang.invoke.MethodType/methodType
                         Object
                         (into-array Class [Object Object]))
        raw-handle (-> (java.lang.invoke.MethodHandles/lookup)
                       (.findVirtual clojure.lang.IFn "invoke" ifn-invoke-type)
                       (.bindTo callback-impl))
        ;; Adapt: MemorySegment args -> Object args, Object return -> int return
        fn-handle (java.lang.invoke.MethodHandles/explicitCastArguments
                   raw-handle target-type)

        ;; Create the upcall stub
        arena (java.lang.foreign.Arena/ofAuto)
        linker (java.lang.foreign.Linker/nativeLinker)
        upcall-stub (.upcallStub linker
                                 ^java.lang.invoke.MethodHandle fn-handle
                                 ^java.lang.foreign.FunctionDescriptor callback-fn-descriptor
                                 ^java.lang.foreign.Arena arena
                                 (into-array java.lang.foreign.Linker$Option []))

        ;; Call mlx_closure_new_func with our upcall stub
        ;; The downcall handle for struct returns prepends a SegmentAllocator arg
        result-struct (.invokeWithArguments
                       closure-new-func-handle
                       (object-array [^java.lang.foreign.SegmentAllocator arena
                                      upcall-stub]))]
    (let [ctx-ptr (.get ^java.lang.foreign.MemorySegment result-struct
                        java.lang.foreign.ValueLayout/ADDRESS 0)]
      ;; Hold strong refs keyed by ctx pointer for targeted release
      (swap! callback-refs assoc ctx-ptr {:fn callback-impl :stub upcall-stub :arena arena})
      ;; Return as ::closure map
      {:ctx ctx-ptr})))

;; ---------------------------------------------------------------------------
;; Transforms — value_and_grad, vjp, jvp
;; ---------------------------------------------------------------------------

;; mlx_value_and_grad: returns a closure_value_and_grad
(def ^:private raw-value-and-grad
  "int mlx_value_and_grad(mlx_closure_value_and_grad* res, const mlx_closure fun,
                          const int* argnums, size_t argnums_num)"
  (ffi/cfn "mlx_value_and_grad"
           [::mem/pointer ::closure ::mem/pointer ::mem/long]
           ::mem/int))

(def ^:private raw-closure-vag-apply
  "int mlx_closure_value_and_grad_apply(
     mlx_vector_array* res_0, mlx_vector_array* res_1,
     mlx_closure_value_and_grad cls, const mlx_vector_array input)"
  (ffi/cfn "mlx_closure_value_and_grad_apply"
           [::mem/pointer ::mem/pointer ::closure-value-and-grad ::vector-array]
           ::mem/int))

(def closure-value-and-grad-free
  (ffi/cfn "mlx_closure_value_and_grad_free"
           [::closure-value-and-grad] ::mem/int))

(defn value-and-grad
  "Create a value-and-grad closure from an mlx_closure.
   `argnums` is a vector of argument indices to differentiate w.r.t."
  [cls argnums]
  (let [arena (mem/auto-arena)
        ;; Write argnums as int array
        n (count argnums)
        argnums-seg (mem/alloc (* 4 n) arena)
        _ (dotimes [i n]
            (.set (.reinterpret argnums-seg (* 4 n))
                  java.lang.foreign.ValueLayout/JAVA_INT
                  (* i 4)
                  (int (nth argnums i))))
        res (mem/alloc-instance ::closure-value-and-grad arena)
        status (raw-value-and-grad res cls argnums-seg (long n))]
    (check-status! status "mlx_value_and_grad")
    (mem/deserialize-from res ::closure-value-and-grad)))

(defn closure-value-and-grad-apply
  "Apply a value-and-grad closure. Returns {:values va, :grads va}."
  [vag-cls input-va]
  (let [arena (mem/auto-arena)
        res-vals (mem/alloc-instance ::vector-array arena)
        res-grads (mem/alloc-instance ::vector-array arena)
        status (raw-closure-vag-apply res-vals res-grads vag-cls input-va)]
    (check-status! status "mlx_closure_value_and_grad_apply")
    {:values (mem/deserialize-from res-vals ::vector-array)
     :grads  (mem/deserialize-from res-grads ::vector-array)}))

;; VJP
(def ^:private raw-vjp
  "int mlx_vjp(mlx_vector_array* res_0, mlx_vector_array* res_1,
               const mlx_closure fun, const mlx_vector_array primals,
               const mlx_vector_array cotangents)"
  (ffi/cfn "mlx_vjp"
           [::mem/pointer ::mem/pointer ::closure ::vector-array ::vector-array]
           ::mem/int))

(defn mlx-vjp
  "Compute VJP (reverse-mode AD). Returns {:values va, :vjps va}."
  [cls primals cotangents]
  (let [arena (mem/auto-arena)
        res-vals (mem/alloc-instance ::vector-array arena)
        res-vjps (mem/alloc-instance ::vector-array arena)
        status (raw-vjp res-vals res-vjps cls primals cotangents)]
    (check-status! status "mlx_vjp")
    {:values (mem/deserialize-from res-vals ::vector-array)
     :vjps   (mem/deserialize-from res-vjps ::vector-array)}))

;; JVP
(def ^:private raw-jvp
  "int mlx_jvp(mlx_vector_array* res_0, mlx_vector_array* res_1,
               const mlx_closure fun, const mlx_vector_array primals,
               const mlx_vector_array tangents)"
  (ffi/cfn "mlx_jvp"
           [::mem/pointer ::mem/pointer ::closure ::vector-array ::vector-array]
           ::mem/int))

(defn mlx-jvp
  "Compute JVP (forward-mode AD). Returns {:values va, :jvps va}."
  [cls primals tangents]
  (let [arena (mem/auto-arena)
        res-vals (mem/alloc-instance ::vector-array arena)
        res-jvps (mem/alloc-instance ::vector-array arena)
        status (raw-jvp res-vals res-jvps cls primals tangents)]
    (check-status! status "mlx_jvp")
    {:values (mem/deserialize-from res-vals ::vector-array)
     :jvps   (mem/deserialize-from res-jvps ::vector-array)}))

;; ---------------------------------------------------------------------------
;; vmap — vectorized map (detail API)
;; ---------------------------------------------------------------------------

;; ---------------------------------------------------------------------------
;; vmap — direct Panama MethodHandles (bypass coffi for struct-by-value args)
;;
;; mlx_detail_vmap_trace and mlx_detail_vmap_replace take struct-by-value
;; args (mlx_closure, mlx_vector_array). On aarch64, these single-pointer
;; structs are passed identically to bare pointers. We use direct Panama
;; MethodHandles to avoid any coffi serialization ambiguity.
;; ---------------------------------------------------------------------------

(def ^:private vmap-trace-descriptor
  "FunctionDescriptor for: int mlx_detail_vmap_trace(
     mlx_vector_array* res_0, mlx_vector_array* res_1,
     mlx_closure fun, mlx_vector_array inputs,
     const int* in_axes, size_t in_axes_num)"
  (java.lang.foreign.FunctionDescriptor/of
   java.lang.foreign.ValueLayout/JAVA_INT
   (into-array java.lang.foreign.MemoryLayout
               [java.lang.foreign.ValueLayout/ADDRESS    ;; res_0 (out ptr)
                java.lang.foreign.ValueLayout/ADDRESS    ;; res_1 (out ptr)
                java.lang.foreign.ValueLayout/ADDRESS    ;; fun ctx (struct-as-ptr)
                java.lang.foreign.ValueLayout/ADDRESS    ;; inputs ctx (struct-as-ptr)
                java.lang.foreign.ValueLayout/ADDRESS    ;; in_axes (int*)
                java.lang.foreign.ValueLayout/JAVA_LONG]))) ;; in_axes_num

(def ^:private vmap-trace-handle
  (let [sym (ffi/find-symbol "mlx_detail_vmap_trace")]
    (when sym
      (.downcallHandle (java.lang.foreign.Linker/nativeLinker)
                       ^java.lang.foreign.MemorySegment sym
                       ^java.lang.foreign.FunctionDescriptor vmap-trace-descriptor
                       (into-array java.lang.foreign.Linker$Option [])))))

(def ^:private vmap-replace-descriptor
  "FunctionDescriptor for: int mlx_detail_vmap_replace(
     mlx_vector_array* res,
     mlx_vector_array inputs, mlx_vector_array s_inputs, mlx_vector_array s_outputs,
     const int* in_axes, size_t in_axes_num,
     const int* out_axes, size_t out_axes_num)"
  (java.lang.foreign.FunctionDescriptor/of
   java.lang.foreign.ValueLayout/JAVA_INT
   (into-array java.lang.foreign.MemoryLayout
               [java.lang.foreign.ValueLayout/ADDRESS    ;; res (out ptr)
                java.lang.foreign.ValueLayout/ADDRESS    ;; inputs ctx
                java.lang.foreign.ValueLayout/ADDRESS    ;; s_inputs ctx
                java.lang.foreign.ValueLayout/ADDRESS    ;; s_outputs ctx
                java.lang.foreign.ValueLayout/ADDRESS    ;; in_axes
                java.lang.foreign.ValueLayout/JAVA_LONG  ;; in_axes_num
                java.lang.foreign.ValueLayout/ADDRESS    ;; out_axes
                java.lang.foreign.ValueLayout/JAVA_LONG]))) ;; out_axes_num

(def ^:private vmap-replace-handle
  (let [sym (ffi/find-symbol "mlx_detail_vmap_replace")]
    (when sym
      (.downcallHandle (java.lang.foreign.Linker/nativeLinker)
                       ^java.lang.foreign.MemorySegment sym
                       ^java.lang.foreign.FunctionDescriptor vmap-replace-descriptor
                       (into-array java.lang.foreign.Linker$Option [])))))

(defn vmap-trace
  "Trace a closure for vmap. Returns {:outputs va-handle, :inputs va-handle}
   where each handle is a raw ctx MemorySegment (not a coffi map)."
  [cls-ctx input-va-ctx in-axes]
  (let [arena (java.lang.foreign.Arena/ofAuto)
        n (count in-axes)
        axes-seg (.allocate arena (* 4 n) 4)
        _ (dotimes [i n]
            (.set axes-seg java.lang.foreign.ValueLayout/JAVA_INT
                  (long (* i 4)) (int (nth in-axes i))))
        ;; Output pointers (zero-initialized by Arena)
        out0-seg (.allocate arena 8 8)
        out1-seg (.allocate arena 8 8)
        status (int (.invokeWithArguments
                     ^java.lang.invoke.MethodHandle vmap-trace-handle
                     (object-array [out0-seg out1-seg
                                    ^java.lang.foreign.MemorySegment cls-ctx
                                    ^java.lang.foreign.MemorySegment input-va-ctx
                                    axes-seg (long n)])))]
    (check-status! status "mlx_detail_vmap_trace")
    {:outputs (.get ^java.lang.foreign.MemorySegment out0-seg
                    java.lang.foreign.ValueLayout/ADDRESS (long 0))
     :inputs  (.get ^java.lang.foreign.MemorySegment out1-seg
                    java.lang.foreign.ValueLayout/ADDRESS (long 0))}))

(defn vmap-replace
  "Produce batched outputs from a vmap trace. Returns ctx MemorySegment for result vector-array."
  [input-va-ctx s-inputs-ctx s-outputs-ctx in-axes out-axes]
  (let [arena (java.lang.foreign.Arena/ofAuto)
        n-in (count in-axes)
        n-out (count out-axes)
        in-axes-seg (.allocate arena (* 4 n-in) 4)
        _ (dotimes [i n-in]
            (.set in-axes-seg java.lang.foreign.ValueLayout/JAVA_INT
                  (long (* i 4)) (int (nth in-axes i))))
        out-axes-seg (.allocate arena (* 4 n-out) 4)
        _ (dotimes [i n-out]
            (.set out-axes-seg java.lang.foreign.ValueLayout/JAVA_INT
                  (long (* i 4)) (int (nth out-axes i))))
        res-seg (.allocate arena 8 8)
        status (int (.invokeWithArguments
                     ^java.lang.invoke.MethodHandle vmap-replace-handle
                     (object-array [res-seg
                                    ^java.lang.foreign.MemorySegment input-va-ctx
                                    ^java.lang.foreign.MemorySegment s-inputs-ctx
                                    ^java.lang.foreign.MemorySegment s-outputs-ctx
                                    in-axes-seg (long n-in)
                                    out-axes-seg (long n-out)])))]
    (check-status! status "mlx_detail_vmap_replace")
    (.get ^java.lang.foreign.MemorySegment res-seg
          java.lang.foreign.ValueLayout/ADDRESS (long 0))))

;; ---------------------------------------------------------------------------
;; Compile — trace once, reuse compiled graph
;; ---------------------------------------------------------------------------

(def ^:private raw-mlx-compile
  "int mlx_compile(mlx_closure* res, const mlx_closure fun, bool shapeless)"
  (ffi/cfn "mlx_compile"
           [::mem/pointer ::closure ::mem/byte]
           ::mem/int))

(defn mlx-compile
  "Compile a closure for optimized execution. Returns a new compiled closure."
  [cls shapeless]
  (let [arena (mem/auto-arena)
        res (mem/alloc-instance ::closure arena)
        status (raw-mlx-compile res cls (if shapeless (byte 1) (byte 0)))]
    (check-status! status "mlx_compile")
    (mem/deserialize-from res ::closure)))

;; ---------------------------------------------------------------------------
;; Random number generation
;; ---------------------------------------------------------------------------

(def ^:private raw-random-key
  "int mlx_random_key(mlx_array* res, uint64_t seed)"
  (ffi/cfn "mlx_random_key"
           [::mem/pointer ::mem/long]
           ::mem/int))

(def ^:private raw-random-normal
  "int mlx_random_normal(mlx_array* res, const int* shape, size_t shape_num,
                          mlx_dtype dtype, float loc, float scale,
                          const mlx_array key, const mlx_stream s)"
  (ffi/cfn "mlx_random_normal"
           [::mem/pointer ::mem/pointer ::mem/long ::mem/int
            ::mem/float ::mem/float ::array ::stream]
           ::mem/int))

(defn random-key
  "Create an MLX random key from a seed. Returns ::array handle."
  [seed]
  (let [arena (mem/auto-arena)
        res (mem/alloc-instance ::array arena)
        status (raw-random-key res (long seed))]
    (check-status! status "mlx_random_key")
    (mem/deserialize-from res ::array)))

(defn random-normal
  "Sample from N(0,I) with given shape. Returns ::array handle.
   Uses MLX's native random number generator."
  [shape-vec key-arr]
  (let [arena (mem/auto-arena)
        ndim (count shape-vec)
        shape-seg (mem/alloc (* 4 ndim) arena)
        _ (dotimes [i ndim]
            (.set (.reinterpret shape-seg (* 4 ndim))
                  java.lang.foreign.ValueLayout/JAVA_INT
                  (* i 4)
                  (int (nth shape-vec i))))
        res (mem/alloc-instance ::array arena)
        status (raw-random-normal res shape-seg (long ndim) dtype-float32
                                  (float 0.0) (float 1.0)
                                  key-arr default-stream)]
    (check-status! status "mlx_random_normal")
    (mem/deserialize-from res ::array)))

;; random_uniform: int mlx_random_uniform(mlx_array* res, mlx_array low, mlx_array high,
;;                   const int* shape, size_t shape_num, mlx_dtype dtype,
;;                   mlx_array key, mlx_stream s)
(def ^:private raw-random-uniform
  (ffi/cfn "mlx_random_uniform"
           [::mem/pointer ::array ::array ::mem/pointer ::mem/long
            ::mem/int ::array ::stream]
           ::mem/int))

(defn random-uniform
  "Sample from Uniform(0,1) with given shape. Returns ::array handle."
  [shape-vec key-arr]
  (let [arena (mem/auto-arena)
        ndim (count shape-vec)
        shape-seg (if (zero? ndim)
                    (mem/alloc 4 arena) ;; dummy for scalar
                    (let [seg (mem/alloc (* 4 ndim) arena)]
                      (dotimes [i ndim]
                        (.set (.reinterpret seg (* 4 ndim))
                              java.lang.foreign.ValueLayout/JAVA_INT
                              (* i 4)
                              (int (nth shape-vec i))))
                      seg))
        low-arr  (array-new-float (float 0.0))
        high-arr (array-new-float (float 1.0))
        res (mem/alloc-instance ::array arena)
        status (raw-random-uniform res low-arr high-arr
                                    shape-seg (long ndim) dtype-float32
                                    key-arr default-stream)]
    (check-status! status "mlx_random_uniform")
    (mem/deserialize-from res ::array)))

;; ---------------------------------------------------------------------------
;; Fast-path shim — single FFI call for compiled closure application
;;
;; The C shim (native/gen_mlx_shim.c) fuses vector-array construction,
;; closure application, result extraction, and cleanup into one native call.
;; Combined with direct Panama MethodHandle (bypassing coffi serialization),
;; this reduces per-call overhead from ~8 FFI calls + 8 arenas to 1 FFI call.
;; ---------------------------------------------------------------------------

(defn- find-shim-lib []
  (let [candidates ["native/libgen_mlx_shim.dylib"]]
    (some (fn [path]
            (let [f (io/file path)]
              (when (.exists f)
                (.getAbsolutePath f))))
          candidates)))

(defonce ^:private shim-lookup
  (when-let [path (find-shim-lib)]
    (java.lang.foreign.SymbolLookup/libraryLookup
     (java.nio.file.Path/of path (into-array String []))
     (java.lang.foreign.Arena/ofAuto))))

(def ^:private fast-apply-descriptor
  "FunctionDescriptor for: int gen_mlx_fast_apply(void*, void*, void*, int)"
  (java.lang.foreign.FunctionDescriptor/of
   java.lang.foreign.ValueLayout/JAVA_INT
   (into-array java.lang.foreign.MemoryLayout
               [java.lang.foreign.ValueLayout/ADDRESS
                java.lang.foreign.ValueLayout/ADDRESS
                java.lang.foreign.ValueLayout/ADDRESS
                java.lang.foreign.ValueLayout/JAVA_INT])))

(def fast-apply-handle
  "Direct Panama MethodHandle for the fast-apply shim.
   nil if shim library is not available (falls back to coffi path)."
  (when shim-lookup
    (let [opt (.find ^java.lang.foreign.SymbolLookup shim-lookup
                     "gen_mlx_fast_apply")]
      (when (.isPresent opt)
        (.downcallHandle (java.lang.foreign.Linker/nativeLinker)
                         ^java.lang.foreign.MemorySegment (.get opt)
                         ^java.lang.foreign.FunctionDescriptor fast-apply-descriptor
                         (into-array java.lang.foreign.Linker$Option []))))))

;; ---------------------------------------------------------------------------
;; Fast-path value-and-grad shim — single FFI call for vag apply
;; ---------------------------------------------------------------------------

(def ^:private fast-vag-descriptor
  "FunctionDescriptor for: int gen_mlx_fast_vag_apply(void*, void*, void*, void*, int)"
  (java.lang.foreign.FunctionDescriptor/of
   java.lang.foreign.ValueLayout/JAVA_INT
   (into-array java.lang.foreign.MemoryLayout
               [java.lang.foreign.ValueLayout/ADDRESS    ;; value_ctx_out
                java.lang.foreign.ValueLayout/ADDRESS    ;; grad_ctx_out
                java.lang.foreign.ValueLayout/ADDRESS    ;; vag_closure_ctx
                java.lang.foreign.ValueLayout/ADDRESS    ;; input_ctxs
                java.lang.foreign.ValueLayout/JAVA_INT]))) ;; n_inputs

(def fast-vag-handle
  "Direct Panama MethodHandle for the fast-vag-apply shim.
   nil if shim library is not available."
  (when shim-lookup
    (let [opt (.find ^java.lang.foreign.SymbolLookup shim-lookup
                     "gen_mlx_fast_vag_apply")]
      (when (.isPresent opt)
        (.downcallHandle (java.lang.foreign.Linker/nativeLinker)
                         ^java.lang.foreign.MemorySegment (.get opt)
                         ^java.lang.foreign.FunctionDescriptor fast-vag-descriptor
                         (into-array java.lang.foreign.Linker$Option []))))))

;; ---------------------------------------------------------------------------
;; Fast-path C leapfrog shim — entire leapfrog loop in native code
;; ---------------------------------------------------------------------------

(def ^:private fast-leapfrog-descriptor
  "FunctionDescriptor for: int gen_mlx_leapfrog(void*, void*, void*, void*, void*, void*, float, int)"
  (java.lang.foreign.FunctionDescriptor/of
   java.lang.foreign.ValueLayout/JAVA_INT
   (into-array java.lang.foreign.MemoryLayout
               [java.lang.foreign.ValueLayout/ADDRESS    ;; final_pos_out
                java.lang.foreign.ValueLayout/ADDRESS    ;; final_mom_out
                java.lang.foreign.ValueLayout/ADDRESS    ;; final_val_out
                java.lang.foreign.ValueLayout/ADDRESS    ;; vag_ctx
                java.lang.foreign.ValueLayout/ADDRESS    ;; position_ctx
                java.lang.foreign.ValueLayout/ADDRESS    ;; momentum_ctx
                java.lang.foreign.ValueLayout/JAVA_FLOAT ;; eps
                java.lang.foreign.ValueLayout/JAVA_INT]))) ;; L

(def fast-leapfrog-handle
  "Direct Panama MethodHandle for the leapfrog shim.
   nil if shim library is not available."
  (when shim-lookup
    (let [opt (.find ^java.lang.foreign.SymbolLookup shim-lookup
                     "gen_mlx_leapfrog")]
      (when (.isPresent opt)
        (.downcallHandle (java.lang.foreign.Linker/nativeLinker)
                         ^java.lang.foreign.MemorySegment (.get opt)
                         ^java.lang.foreign.FunctionDescriptor fast-leapfrog-descriptor
                         (into-array java.lang.foreign.Linker$Option []))))))

;; ---------------------------------------------------------------------------
;; Fast-path vmap shim — trace + replace in a single native call
;; ---------------------------------------------------------------------------

(def ^:private fast-vmap-descriptor
  "FunctionDescriptor for: int gen_mlx_vmap_apply(
     void** result_ctxs_out, int* n_outputs_out,
     void* closure_ctx,
     const void* input_ctxs, int n_inputs,
     const int* in_axes, int n_in_axes,
     const int* out_axes, int n_out_axes)"
  (java.lang.foreign.FunctionDescriptor/of
   java.lang.foreign.ValueLayout/JAVA_INT
   (into-array java.lang.foreign.MemoryLayout
               [java.lang.foreign.ValueLayout/ADDRESS    ;; result_ctxs_out
                java.lang.foreign.ValueLayout/ADDRESS    ;; n_outputs_out
                java.lang.foreign.ValueLayout/ADDRESS    ;; closure_ctx
                java.lang.foreign.ValueLayout/ADDRESS    ;; input_ctxs
                java.lang.foreign.ValueLayout/JAVA_INT   ;; n_inputs
                java.lang.foreign.ValueLayout/ADDRESS    ;; in_axes
                java.lang.foreign.ValueLayout/JAVA_INT   ;; n_in_axes
                java.lang.foreign.ValueLayout/ADDRESS    ;; out_axes
                java.lang.foreign.ValueLayout/JAVA_INT]))) ;; n_out_axes

(def fast-vmap-handle
  "Direct Panama MethodHandle for the vmap shim.
   nil if shim library is not available."
  (when shim-lookup
    (let [opt (.find ^java.lang.foreign.SymbolLookup shim-lookup
                     "gen_mlx_vmap_apply")]
      (when (.isPresent opt)
        (.downcallHandle (java.lang.foreign.Linker/nativeLinker)
                         ^java.lang.foreign.MemorySegment (.get opt)
                         ^java.lang.foreign.FunctionDescriptor fast-vmap-descriptor
                         (into-array java.lang.foreign.Linker$Option []))))))
