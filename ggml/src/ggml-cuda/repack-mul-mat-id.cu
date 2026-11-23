// MoE entry point (mul_mat_id) for repacked Q8_0 / Q4_0 / IQ4_NL weights.
#include "repack.cuh"
#include "repack-common.cuh"
#include "repack-kernels.cuh"
#include "quantize.cuh"
#include "mmq.cuh"
#include "mmid.cuh"

// Upstream's ids helper sorts the assignments by expert but does not return the
// per-expert counts the GEMM tiles need, so build the prefix sum here. The counts
// live in LDS: a global scan is one dependent load per expert, ~60 us at 256 experts.
static __global__ void repack_expert_bounds(const int32_t * __restrict__ ids,
        int32_t * __restrict__ bounds, const int n_expert, const int n_tokens,
        const int n_expert_used, const int si1) {
    extern __shared__ int cnt[];
    for (int e = threadIdx.x; e < n_expert; e += blockDim.x) {
        cnt[e] = 0;
    }
    __syncthreads();
    const int n_assign = n_tokens * n_expert_used;
    for (int i = threadIdx.x; i < n_assign; i += blockDim.x) {
        const int e = ids[(i / n_expert_used) * si1 + (i % n_expert_used)];
        atomicAdd(&cnt[e], 1);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        int acc = 0;
        bounds[0] = 0;
        for (int e = 0; e < n_expert; e++) {
            acc += cnt[e];
            bounds[e + 1] = acc;
        }
    }
}

// MoE GEMM launch for one tile width. CW < 64 uses the narrow kernel; CW = 64
// uses the shared 64-wide kernel (BN = 64 * TN).
template <bool HAS_IDS, int CW, int TN, int NRL, bool HAS_GATE, ggml_type WT>
static void repack_gemm_launch_t(cudaStream_t stream,
        const uint8_t * w, const uint8_t * w_gate, const block_q8_1 * xq, float * y,
        const uint32_t ne0, const uint32_t ne1, const uint32_t n_tok,
        const int32_t * ids_src1, const int32_t * ids_dst,
        const int32_t * expert_bounds, const int32_t * tile_off,
        const repack_tile_meta * tile_meta,
        const uint32_t n_expert, const size_t expert_stride, const uint32_t dst_s1,
        const int64_t max_tiles, const ggml_glu_op glu_op) {
    const dim3 grid((unsigned) ((ne1 + MMQ_RP_Q8_BM - 1) / MMQ_RP_Q8_BM), (unsigned) max_tiles, 1);
    const dim3 block(CW, NRL, 1);
    if constexpr (CW < 64) {
        if constexpr (HAS_GATE) {
            mmq_gemm_repacked_narrow_gated<HAS_IDS, CW, TN, NRL, WT><<<grid, block, 0, stream>>>(
                w, w_gate, xq, y, ne0, ne1, n_tok, ids_src1, ids_dst, expert_bounds, tile_off, tile_meta,
                n_expert, expert_stride, dst_s1, glu_op);
        } else {
            mmq_gemm_repacked_narrow<HAS_IDS, CW, TN, NRL, WT><<<grid, block, 0, stream>>>(
                w, xq, y, ne0, ne1, n_tok, ids_src1, ids_dst, expert_bounds, tile_off, tile_meta,
                n_expert, expert_stride, dst_s1);
        }
    } else {
        static_assert(!HAS_GATE, "the wide tile has no fused epilogue");
        mmq_gemm_repacked<HAS_IDS, TN, NRL, WT><<<grid, block, 0, stream>>>(
            w, xq, y, ne0, ne1, n_tok, ids_src1, ids_dst, expert_bounds, tile_off, tile_meta,
            n_expert, expert_stride, dst_s1);
    }
}

template <bool HAS_IDS, int CW, int TN, int NRL, bool HAS_GATE = false>
static void repack_gemm_launch(const ggml_type type, cudaStream_t stream,
        const uint8_t * w, const uint8_t * w_gate, const block_q8_1 * xq, float * y,
        const uint32_t ne0, const uint32_t ne1, const uint32_t n_tok,
        const int32_t * ids_src1, const int32_t * ids_dst,
        const int32_t * expert_bounds, const int32_t * tile_off,
        const repack_tile_meta * tile_meta,
        const uint32_t n_expert, const size_t expert_stride, const uint32_t dst_s1,
        const int64_t max_tiles, const ggml_glu_op glu_op = GGML_GLU_OP_REGLU) {
    switch (type) {
        case GGML_TYPE_Q8_0:
            repack_gemm_launch_t<HAS_IDS, CW, TN, NRL, HAS_GATE, GGML_TYPE_Q8_0>(stream, w, w_gate, xq, y, ne0, ne1, n_tok,
                ids_src1, ids_dst, expert_bounds, tile_off, tile_meta, n_expert, expert_stride, dst_s1, max_tiles, glu_op);
            break;
        case GGML_TYPE_Q4_0:
            repack_gemm_launch_t<HAS_IDS, CW, TN, NRL, HAS_GATE, GGML_TYPE_Q4_0>(stream, w, w_gate, xq, y, ne0, ne1, n_tok,
                ids_src1, ids_dst, expert_bounds, tile_off, tile_meta, n_expert, expert_stride, dst_s1, max_tiles, glu_op);
            break;
        default:
            repack_gemm_launch_t<HAS_IDS, CW, TN, NRL, HAS_GATE, GGML_TYPE_IQ4_NL>(stream, w, w_gate, xq, y, ne0, ne1, n_tok,
                ids_src1, ids_dst, expert_bounds, tile_off, tile_meta, n_expert, expert_stride, dst_s1, max_tiles, glu_op);
            break;
    }
}

static void mul_mat_id_repacked_impl(ggml_backend_cuda_context & ctx,
        const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids,
        ggml_tensor * dst, const ggml_tensor * gate, ggml_glu_op glu_op) {
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(ids->type  == GGML_TYPE_I32);
    GGML_ASSERT(src1->nb[0] == sizeof(float));
    GGML_ASSERT(ids->nb[0]  == sizeof(int32_t));
    GGML_ASSERT(src1->ne[3] == 1 && dst->ne[3] == 1);
    GGML_ASSERT(src1->nb[2] == src1->nb[1] * src1->ne[1]);
    GGML_ASSERT(dst->nb[2]  == dst->nb[1]  * dst->ne[1]);
    GGML_ASSERT(dst->nb[1]  == (size_t) dst->ne[0] * sizeof(float));

    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];
    const int64_t ne02 = src0->ne[2];
    const int64_t ne10 = src1->ne[0];
    GGML_ASSERT(ne10 == ne00);
    const int64_t n_expert_used = ids->ne[0];
    const int64_t n_tokens      = ids->ne[1];
    const int64_t n_assign      = n_expert_used * n_tokens;

    cudaStream_t stream = ctx.stream();

    const uint8_t * w;
    if (src0->view_src != nullptr && ggml_cuda_repack_tensor_supported(src0->view_src)) {
        w = repack_view_get_cached(src0, src0->view_src, stream);
    } else {
        w = (const uint8_t *) src0->data;
    }
    const uint8_t * w_gate = nullptr;
    if (gate != nullptr) {
        GGML_ASSERT(gate->type == src0->type);
        GGML_ASSERT(gate->ne[0] == src0->ne[0] && gate->ne[1] == src0->ne[1]);
        GGML_ASSERT(dst->ne[0] == ne01);
        if (gate->view_src != nullptr && ggml_cuda_repack_tensor_supported(gate->view_src)) {
            w_gate = repack_view_get_cached(gate, gate->view_src, stream);
        } else {
            w_gate = (const uint8_t *) gate->data;
        }
    }
    float * dst_d = (float *) dst->data;
    const size_t expert_stride = repack_gcn_nbytes(src0->type, ne00, ne01);
    const uint32_t dst_s1 = dst->nb[1] / sizeof(float);

    const int si1         = ids->nb[1] / sizeof(int32_t);
    const int sis1        = src1->nb[2] / src1->nb[1];
    const int nchannels_y = src1->ne[1];

    const int64_t ne10_padded = GGML_PAD(ne10, MATRIX_ROW_PADDING);
    const int64_t x_stride    = ne10_padded / QK8_1;
    const int64_t n_cols      = src1->ne[1] * src1->ne[2];
    const int64_t s11         = src1->nb[1] / sizeof(float);

    if (n_tokens <= MMQ_RP_Q8_MOE_MMV_MAX_TOKENS) {
        ggml_cuda_pool_alloc<block_q8_1> xq_own;
        block_q8_1 * xq = repack_quantize_src1_q8_1(ctx, src0, src1, ne10,
            ne10_padded, s11, s11 * n_cols, s11 * n_cols, n_cols, 1, 1,
            (size_t) (n_cols * x_stride), xq_own, stream);

        // Raw routing: the assignment index is (token, slot), so the counting
        // sort and the expert bounds are only needed by the tiled GEMM below.
        if (gate != nullptr) {
            // Fused up+gate+GLU: both weights stream through one launch and one
            // activation read. Geometry follows the traced MoE fusion point.
            repack_mmv_ids_launch<8, 4, 1, 32, true>(src0->type, w, xq, dst_d,
                (uint32_t) ne00, (uint32_t) ne01, (const int32_t *) ids->data,
                (uint32_t) n_expert_used, (uint32_t) si1, (uint32_t) sis1, (uint32_t) nchannels_y,
                expert_stride, (uint32_t) x_stride, dst_s1, n_assign, stream, w_gate, glu_op);
        } else {
            const uint32_t n_blocks = (uint32_t) ne00 >> 5;
            if (n_blocks <= 8) {
                repack_mmv_ids_launch<16, 2, 1, 8>(src0->type, w, xq, dst_d,
                    (uint32_t) ne00, (uint32_t) ne01, (const int32_t *) ids->data,
                    (uint32_t) n_expert_used, (uint32_t) si1, (uint32_t) sis1, (uint32_t) nchannels_y,
                    expert_stride, (uint32_t) x_stride, dst_s1, n_assign, stream);
            } else if (n_blocks <= 16) {
                repack_mmv_ids_launch<8, 2, 1, 16>(src0->type, w, xq, dst_d,
                    (uint32_t) ne00, (uint32_t) ne01, (const int32_t *) ids->data,
                    (uint32_t) n_expert_used, (uint32_t) si1, (uint32_t) sis1, (uint32_t) nchannels_y,
                    expert_stride, (uint32_t) x_stride, dst_s1, n_assign, stream);
            } else {
                repack_mmv_ids_launch<4, 2, 2, 64>(src0->type, w, xq, dst_d,
                    (uint32_t) ne00, (uint32_t) ne01, (const int32_t *) ids->data,
                    (uint32_t) n_expert_used, (uint32_t) si1, (uint32_t) sis1, (uint32_t) nchannels_y,
                    expert_stride, (uint32_t) x_stride, dst_s1, n_assign, stream);
            }
        }
        return;
    }

    ggml_cuda_pool_alloc<int32_t> ids_src1(ctx.pool(), n_assign);
    ggml_cuda_pool_alloc<int32_t> ids_dst (ctx.pool(), n_assign);
    ggml_cuda_pool_alloc<int32_t> expert_bounds(ctx.pool(), ne02 + 1);
    // upstream's helper always writes its tile spec; the repack GEMM builds its
    // own tiles, so the buffer is only here to keep the helper happy.
    ggml_cuda_pool_alloc<ggml_cuda_mm_id_tile_spec> tile_spec(ctx.pool(), (size_t) n_assign + ne02);
    ggml_cuda_launch_mm_ids_helper((const int32_t *) ids->data,
        ids_src1.get(), ids_dst.get(), tile_spec.get(),
        ne02, n_tokens, n_expert_used, nchannels_y, /*tile_width =*/ 1 << 20, si1, sis1, false, stream);
    repack_expert_bounds<<<1, 256, ne02 * (int) sizeof(int), stream>>>(
        (const int32_t *) ids->data, expert_bounds.get(), ne02, n_tokens, n_expert_used, si1);
    CUDA_CHECK(cudaGetLastError());

    const uint64_t n_groups = (uint64_t) ne10_padded / (4 * QK8_1);
    ggml_cuda_pool_alloc<block_q8_1_mmq_h> xq_buf;
    const ggml_type x_type = repack_mmq_x_type(src0->type);
    const block_q8_1 * xq = repack_quantize_mmq_q8_1(ctx, src1, x_type,
        1 + (int) mmq_get_q8_1_ds_layout(x_type), ne10, ne10_padded,
        s11, s11 * n_cols, s11 * n_cols, n_cols, 1, 1,
        (size_t) n_cols * n_groups, xq_buf, stream);

    // Tile width follows the average expert load: a narrow tile at P=512
    // (n_expert_used * n_tokens / n_expert ~= 16) does not spend half its MACs on
    // empty columns, a wider one at long prompts does not read the expert weights twice.
    // Fixed 32-wide tile. Measured on Qwen3.6-35B-A3B IQ4_NL, 2x MI50,
    // -sm tensor: pp at ub 2048 2854 t/s and at pp4096 2749, against 2578 and
    // 2480 for a 64-wide tile and 2522 for 128 at pp4096. The load-based
    // 32/64/128 selection this replaces was only ever exercised at ub 512
    // (avg 16 -> 32); its 64-wide instantiation is the narrow CW=32/TN=2 tile,
    // which is 2.5x slower than 32 at ub 2048.
    constexpr int BN = 32;

    const int64_t max_tiles = n_assign / BN + ne02;
    ggml_cuda_pool_alloc<int32_t>          tile_off (ctx.pool(), ne02 + 1);
    ggml_cuda_pool_alloc<repack_tile_meta> tile_meta(ctx.pool(), max_tiles);
    repack_tile_off<BN><<<1, 1, 0, stream>>>(expert_bounds.get(), tile_off.get(), tile_meta.get(), ne02);

    switch (src0->type) {
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_IQ4_NL: {
            if (gate == nullptr) {
                repack_gemm_launch<true, 32, 1, MMQ_RP_Q8_NROW_LANES * 2>(src0->type, stream, w, nullptr, xq, dst_d,
                    (uint32_t) ne00, (uint32_t) ne01, (uint32_t) n_cols, ids_src1.get(), ids_dst.get(),
                    expert_bounds.get(), tile_off.get(), tile_meta.get(), (uint32_t) ne02, expert_stride, dst_s1, max_tiles);
            } else {
                repack_gemm_launch<true, 32, 1, MMQ_RP_Q8_NROW_LANES * 2, true>(src0->type, stream, w, w_gate, xq, dst_d,
                    (uint32_t) ne00, (uint32_t) ne01, (uint32_t) n_cols, ids_src1.get(), ids_dst.get(),
                    expert_bounds.get(), tile_off.get(), tile_meta.get(), (uint32_t) ne02, expert_stride, dst_s1, max_tiles, glu_op);
            }
        } break;
        default:
            GGML_ABORT("unsupported repack type");
    }
}

void ggml_cuda_mul_mat_id_repacked(ggml_backend_cuda_context & ctx,
        const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids,
        ggml_tensor * dst) {
    mul_mat_id_repacked_impl(ctx, src0, src1, ids, dst, nullptr, GGML_GLU_OP_REGLU);
}

void ggml_cuda_mul_mat_id_repacked_fused(ggml_backend_cuda_context & ctx,
        const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids,
        const ggml_tensor * gate, ggml_tensor * dst, ggml_glu_op glu_op) {
    mul_mat_id_repacked_impl(ctx, src0, src1, ids, dst, gate, glu_op);
}
