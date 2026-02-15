/*
 * Gen.clj MLX fast-path shim.
 *
 * Fuses vector-array construction, closure application, result extraction,
 * and cleanup into a single native call. Combined with direct Panama
 * MethodHandle on the Java side, this reduces compiled-closure invocation
 * from ~8 FFI round-trips to 1.
 */

#include "mlx/c/array.h"
#include "mlx/c/closure.h"
#include "mlx/c/vector.h"

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
