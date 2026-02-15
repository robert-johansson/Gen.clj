(ns gen.mlx.transforms
  "Function transforms: grad, value-and-grad, vjp, jvp, compile.
   Wraps Clojure functions as mlx_closures and applies MLX's
   automatic differentiation transforms.

   Follows standard AD conventions:
     (grad f)           — returns fn computing gradient only
     (value-and-grad f) — returns fn computing {:value :grads}
     (vjp f primals cotangents) — reverse-mode AD
     (jvp f primals tangents)   — forward-mode AD
     (compile f)        — trace once, reuse compiled graph (like JAX's jit)"
  (:refer-clojure :exclude [compile])
  (:require [gen.mlx.ffi :as ffi]
            [gen.mlx.array :as arr]))

;; ---------------------------------------------------------------------------
;; Closure bridge: Clojure fn -> mlx_closure
;; ---------------------------------------------------------------------------

(defn ->closure
  "Wrap a Clojure function (MLXArray... -> MLXArray) as an mlx_closure.
   The function takes one or more MLXArray args and returns a single MLXArray.
   MLX will call this function during grad/vjp/jvp to trace the computation."
  [f]
  (ffi/closure-new
   (fn [input-va]
     ;; input-va is a ::vector-array handle (map with :ctx)
     ;; Convert inputs to MLXArrays
     (let [n (ffi/vector-array-size input-va)
           args (mapv (fn [i]
                        (arr/wrap-handle (ffi/vector-array-get input-va i)))
                      (range n))
           ;; Call the Clojure function
           result (apply f args)]
       ;; Convert result back to vector<array>
       (ffi/vector-array-new-value (arr/handle result))))))

;; ---------------------------------------------------------------------------
;; Internal helpers
;; ---------------------------------------------------------------------------

(defn- vector-array->mlx-arrays
  "Extract all arrays from a vector-array as GC-managed MLXArrays,
   then free the vector-array handle."
  [va]
  (let [result (mapv (fn [i]
                       (arr/wrap-handle (ffi/vector-array-get va i)))
                     (range (ffi/vector-array-size va)))]
    (ffi/vector-array-free va)
    result))

;; ---------------------------------------------------------------------------
;; value-and-grad — returns {:value :grads}
;; ---------------------------------------------------------------------------

(defn value-and-grad
  "Returns a function that computes the value and gradients of `f`.
   `f` takes one or more MLXArray args and returns a scalar MLXArray.

   Options:
     :argnums — vector of argument indices to differentiate w.r.t.
                Defaults to [0].

   The returned function takes the same args as `f` and returns:
     {:value  MLXArray   — the function value
      :grads  [MLXArray] — gradient w.r.t. each argnum}

   Example:
     (let [f      (fn [x] (arr/mul x x))
           vag-f  (value-and-grad f)]
       (vag-f (arr/scalar 3.0)))
     ;; => {:value <9.0>, :grads [<6.0>]}"
  ([f] (value-and-grad f {:argnums [0]}))
  ([f {:keys [argnums] :or {argnums [0]}}]
   (let [cls (->closure f)
         vag (ffi/value-and-grad cls argnums)
         vag-ctx (:ctx vag)]
     (if (and ffi/fast-vag-handle (= argnums [0]))
       ;; Fast path: single FFI call via C shim for the common single-argnum case.
       ;; Pre-allocates reusable segments to eliminate per-call allocation.
       (let [arena (java.lang.foreign.Arena/ofAuto)
             value-out-seg (.allocate arena 8 8)
             grad-out-seg (.allocate arena 8 8)
             inputs-seg (.allocate arena (* 8 8) 8)] ;; up to 8 args
         (with-meta
           (fn [& args]
             (let [n (count args)]
               ;; Write input ctx pointers
               (dotimes [i n]
                 (.set ^java.lang.foreign.MemorySegment inputs-seg
                       java.lang.foreign.ValueLayout/ADDRESS
                       (long (* i 8))
                       ^java.lang.foreign.MemorySegment
                       (:ctx (arr/handle (nth args i)))))
               ;; Single FFI call
               (let [status (int (.invokeWithArguments
                                  ^java.lang.invoke.MethodHandle ffi/fast-vag-handle
                                  (object-array [value-out-seg grad-out-seg
                                                 vag-ctx inputs-seg (int n)])))]
                 (when-not (zero? status)
                   (throw (ex-info "gen_mlx_fast_vag_apply failed" {:status status})))
                 {:value (arr/wrap-handle
                          {:ctx (.get ^java.lang.foreign.MemorySegment value-out-seg
                                      java.lang.foreign.ValueLayout/ADDRESS (long 0))})
                  :grads [(arr/wrap-handle
                           {:ctx (.get ^java.lang.foreign.MemorySegment grad-out-seg
                                       java.lang.foreign.ValueLayout/ADDRESS (long 0))})]})))
           {::vag-ctx vag-ctx}))
       ;; Fallback: coffi-based path
       (with-meta
         (fn [& args]
           (let [input-va (ffi/arrays->vector (map arr/handle args))]
             (try
               (let [{:keys [values grads]} (ffi/closure-value-and-grad-apply vag input-va)
                     value (arr/wrap-handle (ffi/vector-array-get values 0))
                     grad-arrays (vector-array->mlx-arrays grads)]
                 (ffi/vector-array-free values)
                 {:value value :grads grad-arrays})
               (finally
                 (ffi/vector-array-free input-va)))))
         {::vag-ctx vag-ctx})))))

;; ---------------------------------------------------------------------------
;; grad — returns gradient only (standard AD convention)
;; ---------------------------------------------------------------------------

(defn grad
  "Returns a function that computes the gradient of `f`.
   Follows the standard AD convention (like JAX's jax.grad):
   the returned function has the same signature as `f` but returns
   the gradient instead of the value.

   Options:
     :argnums — vector of argument indices to differentiate w.r.t.
                Defaults to [0].

   For single argnum: returns a single MLXArray (the gradient).
   For multiple argnums: returns a vector of MLXArray gradients.

   Example:
     (let [f      (fn [x] (arr/mul x x))
           grad-f (grad f)]
       @(grad-f (arr/scalar 3.0)))
     ;; => 6.0"
  ([f] (grad f {:argnums [0]}))
  ([f {:keys [argnums] :or {argnums [0]}}]
   (let [vag-fn (value-and-grad f {:argnums argnums})]
     (fn [& args]
       (let [{:keys [grads]} (apply vag-fn args)]
         (if (= 1 (count argnums))
           (first grads)
           grads))))))

;; ---------------------------------------------------------------------------
;; vjp — reverse-mode AD (more fine-grained than grad)
;; ---------------------------------------------------------------------------

(defn vjp
  "Compute the Vector-Jacobian Product (reverse-mode AD).
   `f` takes MLXArray(s) and returns MLXArray.
   `primals` is a vector of MLXArray inputs.
   `cotangents` is a vector of MLXArray upstream gradients.

   Returns {:values [MLXArray], :vjps [MLXArray]}.
   Cleans up all native intermediate resources."
  [f primals cotangents]
  (let [cls (->closure f)
        primals-va (ffi/arrays->vector (map arr/handle primals))
        cotangents-va (ffi/arrays->vector (map arr/handle cotangents))]
    (try
      (let [{:keys [values vjps]} (ffi/mlx-vjp cls primals-va cotangents-va)]
        {:values (vector-array->mlx-arrays values)
         :vjps   (vector-array->mlx-arrays vjps)})
      (finally
        (ffi/vector-array-free primals-va)
        (ffi/vector-array-free cotangents-va)
        (ffi/closure-free cls)))))

;; ---------------------------------------------------------------------------
;; jvp — forward-mode AD
;; ---------------------------------------------------------------------------

(defn jvp
  "Compute the Jacobian-Vector Product (forward-mode AD).
   `f` takes MLXArray(s) and returns MLXArray.
   `primals` is a vector of MLXArray inputs.
   `tangents` is a vector of MLXArray tangent vectors (dx).

   Returns {:values [MLXArray], :jvps [MLXArray]}.
   Cleans up all native intermediate resources."
  [f primals tangents]
  (let [cls (->closure f)
        primals-va (ffi/arrays->vector (map arr/handle primals))
        tangents-va (ffi/arrays->vector (map arr/handle tangents))]
    (try
      (let [{:keys [values jvps]} (ffi/mlx-jvp cls primals-va tangents-va)]
        {:values (vector-array->mlx-arrays values)
         :jvps   (vector-array->mlx-arrays jvps)})
      (finally
        (ffi/vector-array-free primals-va)
        (ffi/vector-array-free tangents-va)
        (ffi/closure-free cls)))))

;; ---------------------------------------------------------------------------
;; compile — trace once, reuse compiled graph (like JAX's jit)
;; ---------------------------------------------------------------------------

(defn compile
  "Returns a compiled version of `f` that traces the computation graph
   on first call and caches it. Subsequent calls with same-shaped inputs
   skip graph construction and execute the optimized native kernel directly.

   `f` takes one or more MLXArray args and returns a single MLXArray.

   When the native fast-path shim is available (native/libgen_mlx_shim.dylib),
   uses a single FFI call via direct Panama MethodHandle, bypassing coffi
   serialization entirely. Falls back to the coffi-based path otherwise.

   Options:
     :shapeless — if true, handles varying input shapes without
                  recompilation. Default false.

   Example:
     (let [f       (fn [x] (arr/mul x x))
           fast-f  (compile f)]
       @(fast-f (arr/scalar 3.0)))
     ;; => 9.0  (first call traces + compiles, subsequent calls are fast)"
  ([f] (compile f {}))
  ([f {:keys [shapeless] :or {shapeless false}}]
   (let [cls (->closure f)
         compiled-cls (ffi/mlx-compile cls shapeless)]
     (if ffi/fast-apply-handle
       ;; Fast path: single FFI call via C shim + direct Panama MethodHandle.
       ;; Pre-allocates reusable segments to eliminate per-call allocation.
       (let [compiled-ctx (:ctx compiled-cls)
             arena (java.lang.foreign.Arena/ofAuto)
             out-seg (.allocate arena 8 8)
             inputs-seg (.allocate arena (* 8 8) 8)] ;; up to 8 args
         (fn [& args]
           (let [n (count args)]
             ;; Write input ctx pointers into contiguous buffer
             (dotimes [i n]
               (.set ^java.lang.foreign.MemorySegment inputs-seg
                     java.lang.foreign.ValueLayout/ADDRESS
                     (long (* i 8))
                     ^java.lang.foreign.MemorySegment
                     (:ctx (arr/handle (nth args i)))))
             ;; Single FFI call: build vector + apply + extract + cleanup
             (let [status (int (.invokeWithArguments
                                ^java.lang.invoke.MethodHandle ffi/fast-apply-handle
                                (object-array [out-seg compiled-ctx
                                               inputs-seg (int n)])))]
               (when-not (zero? status)
                 (throw (ex-info "gen_mlx_fast_apply failed" {:status status})))
               ;; Read result ctx pointer, wrap as GC-managed MLXArray
               (arr/wrap-handle
                {:ctx (.get ^java.lang.foreign.MemorySegment out-seg
                            java.lang.foreign.ValueLayout/ADDRESS
                            (long 0))})))))
       ;; Fallback: coffi-based path (no shim available)
       (fn [& args]
         (let [input-va (ffi/arrays->vector (map arr/handle args))]
           (try
             (let [output-va (ffi/closure-apply compiled-cls input-va)
                   result (arr/wrap-handle (ffi/vector-array-get output-va 0))]
               (ffi/vector-array-free output-va)
               result)
             (finally
               (ffi/vector-array-free input-va)))))))))

;; ---------------------------------------------------------------------------
;; vmap — vectorize a function over a batch dimension
;; ---------------------------------------------------------------------------

(defn vmap
  "Vectorize `f` over a batch dimension, like JAX's vmap.

   `f` takes one or more MLXArray args and returns a single MLXArray.
   The returned function maps `f` over the specified axes of the inputs,
   producing batched outputs.

   Options:
     :in-axes  — vector of axis indices to map over per input (default [0])
     :out-axes — vector of axis indices for output batch dim (default [0])

   Uses the C shim to fuse vmap_trace + vmap_replace in a single native call.

   Example:
     (let [f     (fn [x] (arr/mul x x))
           vf    (vmap f)]
       @(vf (arr/from-vec [1 2 3 4])))
     ;; => [1.0 4.0 9.0 16.0]"
  [f & {:keys [in-axes out-axes] :or {in-axes [0] out-axes [0]}}]
  (let [cls (->closure f)]
    (fn [& args]
      (let [arena (java.lang.foreign.Arena/ofAuto)
            n (count args)
            ;; Write input ctx pointers
            inputs-seg (.allocate arena (* 8 n) 8)
            _ (dotimes [i n]
                (.set ^java.lang.foreign.MemorySegment inputs-seg
                      java.lang.foreign.ValueLayout/ADDRESS
                      (long (* i 8))
                      ^java.lang.foreign.MemorySegment
                      (:ctx (arr/handle (nth args i)))))
            ;; Prepare axes as native int arrays (MemorySegments, not Java int[])
            in-axes-vec (vec (take n (concat in-axes (repeat (last in-axes)))))
            in-axes-seg (.allocate arena (* 4 (count in-axes-vec)) 4)
            _ (dotimes [i (count in-axes-vec)]
                (.set ^java.lang.foreign.MemorySegment in-axes-seg
                      java.lang.foreign.ValueLayout/JAVA_INT
                      (long (* i 4))
                      (int (nth in-axes-vec i))))
            out-axes-seg (.allocate arena (* 4 (count out-axes)) 4)
            _ (dotimes [i (count out-axes)]
                (.set ^java.lang.foreign.MemorySegment out-axes-seg
                      java.lang.foreign.ValueLayout/JAVA_INT
                      (long (* i 4))
                      (int (nth out-axes i))))
            ;; Output buffer — up to 8 outputs
            result-ptrs (.allocate arena (* 8 8) 8)
            n-outputs-seg (.allocate arena 4 4)
            status (int (.invokeWithArguments
                         ^java.lang.invoke.MethodHandle ffi/fast-vmap-handle
                         (object-array [result-ptrs n-outputs-seg
                                        (:ctx cls)
                                        inputs-seg (int n)
                                        in-axes-seg (int (count in-axes-vec))
                                        out-axes-seg (int (count out-axes))])))]
        (when-not (zero? status)
          (throw (ex-info "gen_mlx_vmap_apply failed" {:status status})))
        (let [n-out (.get ^java.lang.foreign.MemorySegment n-outputs-seg
                          java.lang.foreign.ValueLayout/JAVA_INT (long 0))]
          (if (= 1 n-out)
            (arr/wrap-handle
             {:ctx (.get ^java.lang.foreign.MemorySegment result-ptrs
                         java.lang.foreign.ValueLayout/ADDRESS (long 0))})
            (mapv (fn [i]
                    (arr/wrap-handle
                     {:ctx (.get ^java.lang.foreign.MemorySegment result-ptrs
                                 java.lang.foreign.ValueLayout/ADDRESS (long (* i 8)))}))
                  (range n-out))))))))

