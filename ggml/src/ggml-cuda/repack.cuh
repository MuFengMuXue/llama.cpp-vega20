// Repacked-weight matmul path for gfx906: Q8_0, Q4_0, IQ4_NL.
#pragma once

#include "common.cuh"

#if defined(GGML_USE_MUSA)

inline bool ggml_backend_buft_is_cuda_repack(ggml_backend_buffer_type_t) {
    return false;
}

inline ggml_backend_buffer_type_t ggml_backend_cuda_repack_buffer_type(int) {
    return nullptr;
}

inline bool ggml_cuda_repack_tensor_supported(const ggml_tensor *) {
    return false;
}

inline bool ggml_cuda_repack_mul_mat_should_fire(const ggml_tensor *) {
    return false;
}

inline void ggml_cuda_repack_set_tensor_async(
        int, cudaStream_t, ggml_tensor *, const void *, size_t, size_t) {
    GGML_ABORT("repacked weights are unavailable on MUSA");
}

inline void ggml_cuda_repack_async_release(int) {
}

inline bool ggml_cuda_repack_try_release_scratch(int) {
    return false;
}

inline void ggml_cuda_mul_mat_repacked(
        ggml_backend_cuda_context &, const ggml_tensor *, const ggml_tensor *, ggml_tensor *) {
    GGML_ABORT("repacked weights are unavailable on MUSA");
}

inline void ggml_cuda_mul_mat_id_repacked(
        ggml_backend_cuda_context &, const ggml_tensor *, const ggml_tensor *, const ggml_tensor *, ggml_tensor *) {
    GGML_ABORT("repacked weights are unavailable on MUSA");
}

inline bool ggml_cuda_repack_mul_mat_id_fused_supported(
        const ggml_tensor *, const ggml_tensor *, const ggml_tensor *) {
    return false;
}

inline void ggml_cuda_mul_mat_id_repacked_fused(
        ggml_backend_cuda_context &, const ggml_tensor *, const ggml_tensor *, const ggml_tensor *,
        const ggml_tensor *, ggml_tensor *, ggml_glu_op) {
    GGML_ABORT("repacked weights are unavailable on MUSA");
}

inline bool ggml_cuda_repack_mul_mat_fused_supported(
        const ggml_tensor *, const ggml_tensor *, const ggml_tensor *) {
    return false;
}

inline void ggml_cuda_mul_mat_repacked_fused(
        ggml_backend_cuda_context &, const ggml_tensor *, const ggml_tensor *,
        ggml_tensor *, const ggml_tensor *, ggml_glu_op) {
    GGML_ABORT("repacked weights are unavailable on MUSA");
}

#else

bool ggml_backend_buft_is_cuda_repack(ggml_backend_buffer_type_t buft);

ggml_backend_buffer_type_t ggml_backend_cuda_repack_buffer_type(int device);

bool ggml_cuda_repack_tensor_supported(const ggml_tensor * t);

bool ggml_cuda_repack_mul_mat_should_fire(const ggml_tensor * src0);

// Async-upload path: canonical chunks stage into per-device scratch and the
// device-side repack kernel runs when the tensor completes. The scratch is
// released on the first graph compute after a load.
void ggml_cuda_repack_set_tensor_async(int device, cudaStream_t stream,
    ggml_tensor * tensor, const void * data, size_t offset, size_t size);
void ggml_cuda_repack_async_release(int device);
// Frees the device scratch early when no tensor upload is in progress, and returns whether any VRAM was released.
bool ggml_cuda_repack_try_release_scratch(int device);

void ggml_cuda_mul_mat_repacked(ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);

void ggml_cuda_mul_mat_id_repacked(ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids,
    ggml_tensor * dst);

// Fused up/gate/GLU for repacked MoE weights: mat-vec below
// MMQ_RP_Q8_MOE_MMV_MAX_TOKENS, the gated tile GEMM above it.
bool ggml_cuda_repack_mul_mat_id_fused_supported(
    const ggml_tensor * src0, const ggml_tensor * gate, const ggml_tensor * ids);

void ggml_cuda_mul_mat_id_repacked_fused(ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids,
    const ggml_tensor * gate, ggml_tensor * dst, ggml_glu_op glu_op);

// Fused up/gate/GLU for repacked dense weights, single-token only.
bool ggml_cuda_repack_mul_mat_fused_supported(
    const ggml_tensor * src0, const ggml_tensor * gate, const ggml_tensor * src1);

void ggml_cuda_mul_mat_repacked_fused(ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst,
    const ggml_tensor * gate, ggml_glu_op glu_op);

#endif
