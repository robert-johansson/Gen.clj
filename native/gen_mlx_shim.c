/*
 * Gen.clj MLX fast-path shim.
 *
 * Fuses vector-array construction, closure application, result extraction,
 * and cleanup into a single native call. Combined with direct Panama
 * MethodHandle on the Java side, this reduces compiled-closure invocation
 * from ~8 FFI round-trips to 1.
 *
 * gen_mlx_fast_vag_apply: same for value-and-grad closures — fuses
 * input vector construction, vag apply, value/grad extraction, and cleanup.
 *
 * gen_mlx_leapfrog: full leapfrog integration in C. L+1 vag calls stay
 * in native land (C-to-C), JVM makes 1 FFI call for the entire integration.
 */

#include "mlx/c/array.h"
#include "mlx/c/closure.h"
#include "mlx/c/ops.h"
#include "mlx/c/stream.h"
#include "mlx/c/vector.h"
#include "mlx/c/transforms_impl.h"

int gen_mlx_fast_apply(void* result_ctx_out, void* closure_ctx,
                       const void* input_ctxs, int n_inputs) {
    mlx_closure cls;
    cls.ctx = closure_ctx;

    /* Build input vector<array> from raw ctx pointers. */
    mlx_vector_array input_va = mlx_vector_array_new();
    void** ptrs = (void**)input_ctxs;
    for (int i = 0; i < n_inputs; i++) {
        mlx_array a;
        a.ctx = ptrs[i];
        mlx_vector_array_append_value(input_va, a);
    }

    /* Apply the compiled closure.
     * CRITICAL: output_va must be zero-initialized — mlx-c's internal
     * mlx_vector_array_set_ checks if (d.ctx) and dereferences it as a
     * C++ vector when non-null. Uninitialized stack garbage crashes. */
    mlx_vector_array output_va = {NULL};
    int status = mlx_closure_apply(&output_va, cls, input_va);

    if (status == 0) {
        /* Extract first result array. Same zero-init requirement. */
        mlx_array result = {NULL};
        status = mlx_vector_array_get(&result, output_va, 0);
        if (status == 0) {
            *(void**)result_ctx_out = result.ctx;
        }
        mlx_vector_array_free(output_va);
    }
    mlx_vector_array_free(input_va);
    return status;
}

/*
 * Fast value-and-grad apply.
 *
 * Fuses: build input vector-array + apply vag closure + extract value[0]
 * and grad[0] + free intermediates. Replaces ~7 FFI calls with 1.
 *
 * Returns: 0 on success, non-zero on failure.
 * On success, value_ctx_out and grad_ctx_out contain the array handles
 * for the scalar value and the gradient array respectively.
 */
int gen_mlx_fast_vag_apply(void* value_ctx_out, void* grad_ctx_out,
                            void* vag_closure_ctx,
                            const void* input_ctxs, int n_inputs) {
    mlx_closure_value_and_grad vag;
    vag.ctx = vag_closure_ctx;

    /* Build input vector<array> from raw ctx pointers. */
    mlx_vector_array input_va = mlx_vector_array_new();
    void** ptrs = (void**)input_ctxs;
    for (int i = 0; i < n_inputs; i++) {
        mlx_array a;
        a.ctx = ptrs[i];
        mlx_vector_array_append_value(input_va, a);
    }

    /* Apply vag closure. Zero-init both output vectors. */
    mlx_vector_array values_va = {NULL};
    mlx_vector_array grads_va = {NULL};
    int status = mlx_closure_value_and_grad_apply(&values_va, &grads_va,
                                                   vag, input_va);

    if (status == 0) {
        /* Extract value[0] */
        mlx_array value = {NULL};
        status = mlx_vector_array_get(&value, values_va, 0);
        if (status == 0) {
            *(void**)value_ctx_out = value.ctx;
        }

        /* Extract grad[0] */
        if (status == 0) {
            mlx_array grad = {NULL};
            status = mlx_vector_array_get(&grad, grads_va, 0);
            if (status == 0) {
                *(void**)grad_ctx_out = grad.ctx;
            }
        }

        mlx_vector_array_free(values_va);
        mlx_vector_array_free(grads_va);
    }
    mlx_vector_array_free(input_va);
    return status;
}

/*
 * Helper: call vag closure on a single position array.
 * Extracts value[0] and grad[0], frees intermediate vector-arrays.
 * Returns 0 on success.
 */
static int vag_apply_single(mlx_array* value_out, mlx_array* grad_out,
                             mlx_closure_value_and_grad vag, mlx_array position,
                             mlx_stream s) {
    mlx_vector_array input_va = mlx_vector_array_new();
    mlx_vector_array_append_value(input_va, position);

    mlx_vector_array values_va = {NULL};
    mlx_vector_array grads_va = {NULL};
    int status = mlx_closure_value_and_grad_apply(&values_va, &grads_va,
                                                   vag, input_va);
    if (status == 0) {
        mlx_array v = {NULL};
        status = mlx_vector_array_get(&v, values_va, 0);
        if (status == 0 && value_out) *value_out = v;

        mlx_array g = {NULL};
        if (status == 0) {
            status = mlx_vector_array_get(&g, grads_va, 0);
            if (status == 0) *grad_out = g;
        }
        mlx_vector_array_free(values_va);
        mlx_vector_array_free(grads_va);
    }
    mlx_vector_array_free(input_va);
    return status;
}

/*
 * Full leapfrog integration in C.
 *
 * All L+1 value-and-grad calls happen C-to-C — no JVM boundary crossings.
 * The entire integration builds a lazy MLX graph; no mlx_array_eval inside.
 *
 * Implements standard leapfrog:
 *   1. Half-step momentum: p = p + (eps/2) * grad(q)
 *   2. For i in 0..L-1:
 *      a. Full position step: q = q + eps * p
 *      b. If not last: full momentum step: p = p + eps * grad(q)
 *   3. Half-step momentum: p = p + (eps/2) * grad(q)
 *
 * IMPORTANT: Each intermediate mlx_array handle owns a reference. When we
 * reassign q or p, we must free the OLD handle to release its reference.
 * The underlying data survives because newer graph nodes reference it internally.
 * Input position/momentum handles are NOT freed (owned by caller).
 * Grad/scaled temporaries are freed immediately after use.
 *
 * Outputs: final position, momentum, and log-density (value from last vag call).
 */
int gen_mlx_leapfrog(void* final_pos_out, void* final_mom_out,
                      void* final_val_out,
                      void* vag_ctx, void* position_ctx, void* momentum_ctx,
                      float eps, int L) {
    mlx_stream s = mlx_default_cpu_stream_new();
    mlx_closure_value_and_grad vag;
    vag.ctx = vag_ctx;

    /* q and p start as caller-owned handles — do NOT free these originals */
    mlx_array q;   q.ctx = position_ctx;
    mlx_array p;   p.ctx = momentum_ctx;
    int q_owned = 0;  /* 1 if we created q (need to free on reassign) */
    int p_owned = 0;  /* 1 if we created p (need to free on reassign) */

    /* Pre-allocate step-size scalars */
    mlx_array eps_arr      = mlx_array_new_float32(eps);
    mlx_array half_eps_arr = mlx_array_new_float32(eps / 2.0f);

    int status = 0;

    /* Initial half-step for momentum: p = p + (eps/2) * grad(q) */
    mlx_array grad = {NULL};
    mlx_array val = {NULL};
    status = vag_apply_single(NULL, &grad, vag, q, s);
    if (status != 0) goto cleanup;

    {
        mlx_array scaled = {NULL};
        status = mlx_multiply(&scaled, half_eps_arr, grad, s);
        mlx_array_free(grad);
        if (status != 0) goto cleanup;
        mlx_array p_new = {NULL};
        status = mlx_add(&p_new, p, scaled, s);
        mlx_array_free(scaled);
        if (status != 0) goto cleanup;
        if (p_owned) mlx_array_free(p);
        p = p_new;
        p_owned = 1;
    }

    /* L full steps */
    for (int i = 0; i < L; i++) {
        /* Full position step: q = q + eps * p */
        mlx_array scaled_p = {NULL};
        status = mlx_multiply(&scaled_p, eps_arr, p, s);
        if (status != 0) goto cleanup;
        mlx_array q_new = {NULL};
        status = mlx_add(&q_new, q, scaled_p, s);
        mlx_array_free(scaled_p);
        if (status != 0) goto cleanup;
        if (q_owned) mlx_array_free(q);
        q = q_new;
        q_owned = 1;

        if (i < L - 1) {
            /* Full momentum step: p = p + eps * grad(q_new) */
            mlx_array grad_i = {NULL};
            status = vag_apply_single(NULL, &grad_i, vag, q, s);
            if (status != 0) goto cleanup;
            mlx_array scaled_g = {NULL};
            status = mlx_multiply(&scaled_g, eps_arr, grad_i, s);
            mlx_array_free(grad_i);
            if (status != 0) goto cleanup;
            mlx_array p_new2 = {NULL};
            status = mlx_add(&p_new2, p, scaled_g, s);
            mlx_array_free(scaled_g);
            if (status != 0) goto cleanup;
            if (p_owned) mlx_array_free(p);
            p = p_new2;
            p_owned = 1;
        }
    }

    /* Final half-step: get value and gradient */
    {
        mlx_array final_grad = {NULL};
        status = vag_apply_single(&val, &final_grad, vag, q, s);
        if (status != 0) goto cleanup;
        mlx_array scaled = {NULL};
        status = mlx_multiply(&scaled, half_eps_arr, final_grad, s);
        mlx_array_free(final_grad);
        if (status != 0) goto cleanup;
        mlx_array p_final = {NULL};
        status = mlx_add(&p_final, p, scaled, s);
        mlx_array_free(scaled);
        if (status != 0) goto cleanup;
        if (p_owned) mlx_array_free(p);
        p = p_final;
        p_owned = 1;
    }

    /* Write outputs — caller takes ownership */
    *(void**)final_pos_out = q.ctx;
    *(void**)final_mom_out = p.ctx;
    *(void**)final_val_out = val.ctx;

cleanup:
    mlx_array_free(eps_arr);
    mlx_array_free(half_eps_arr);
    mlx_stream_free(s);
    return status;
}

/*
 * vmap: trace + replace in a single native call.
 *
 * Performs vmap_trace on the closure with the given inputs and in_axes,
 * then immediately calls vmap_replace to produce real batched outputs.
 * This keeps all traced arrays alive in the same C scope.
 *
 * n_outputs is written with the number of output arrays.
 * result_ctxs_out must point to space for at least max_outputs void* pointers.
 */
int gen_mlx_vmap_apply(void** result_ctxs_out, int* n_outputs_out,
                        void* closure_ctx,
                        const void* input_ctxs, int n_inputs,
                        const int* in_axes, int n_in_axes,
                        const int* out_axes, int n_out_axes) {
    mlx_closure cls;
    cls.ctx = closure_ctx;

    /* Build input vector<array> from raw ctx pointers. */
    mlx_vector_array input_va = mlx_vector_array_new();
    void** ptrs = (void**)input_ctxs;
    for (int i = 0; i < n_inputs; i++) {
        mlx_array a;
        a.ctx = ptrs[i];
        mlx_vector_array_append_value(input_va, a);
    }

    /* Phase 1: trace
     * mlx_detail_vmap_trace(res_0, res_1, ...):
     *   res_0 = traced inputs (placeholders with vmap axis removed)
     *   res_1 = traced outputs (result of fun(traced_inputs), has primitives)
     */
    mlx_vector_array trace_inputs = {NULL};
    mlx_vector_array trace_outputs = {NULL};
    int status = mlx_detail_vmap_trace(&trace_inputs, &trace_outputs,
                                        cls, input_va,
                                        in_axes, (size_t)n_in_axes);
    if (status != 0) {
        mlx_vector_array_free(input_va);
        return status;
    }

    /* Phase 2: replace */
    mlx_vector_array result_va = {NULL};
    status = mlx_detail_vmap_replace(&result_va, input_va,
                                      trace_inputs, trace_outputs,
                                      in_axes, (size_t)n_in_axes,
                                      out_axes, (size_t)n_out_axes);

    if (status == 0) {
        /* Extract result arrays */
        int n_out = (int)mlx_vector_array_size(result_va);
        *n_outputs_out = n_out;
        for (int i = 0; i < n_out; i++) {
            mlx_array r = {NULL};
            status = mlx_vector_array_get(&r, result_va, i);
            if (status != 0) break;
            result_ctxs_out[i] = r.ctx;
        }
        mlx_vector_array_free(result_va);
    }

    /* Cleanup traced arrays and input vector */
    mlx_vector_array_free(trace_outputs);
    mlx_vector_array_free(trace_inputs);
    mlx_vector_array_free(input_va);
    return status;
}
