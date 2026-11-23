// Dense (non-MoE) entry point for repacked Q8_0 weights: quantize src1, then dispatch
// the mat-vec (single token) or tiled GEMM (multi-token) path per 2D slice.
#include "repack.cuh"
#include "repack-common.cuh"
#include "repack-kernels.cuh"
#include "quantize.cuh"
#include "mmq.cuh"

static void ggml_cuda_mul_mat_repacked_slice(ggml_backend_cuda_context & ctx,
        const ggml_tensor * src0, const uint8_t * w, const block_q8_1 * xq,
        float * dst_d, int64_t ne00, int64_t ne01, int64_t ne11,
        cudaStream_t stream);
template <ggml_type WT>
static void ggml_cuda_mul_mat_repacked_nc_t(
        const uint8_t * w, const block_q8_1 * xq, float * dst_d,
        const int64_t ne00, const int64_t ne01, const int64_t ne11,
        const uint32_t xs, const uint32_t ys, cudaStream_t stream);

// Dense multi-token GEMM for one token-tile width: 32 uses the narrow kernel,
// 64 and 128 the 64-wide one at TN 1 and 2 (BN = 64 * TN).
template <ggml_type WT>
static void repack_dense_gemm_launch(cudaStream_t stream,
        const uint8_t * w, const block_q8_1 * xq, float * y,
        const uint32_t ne0, const uint32_t ne1, const uint32_t n_tok,
        const int bn, const dim3 & grid) {
    constexpr int nrl = MMQ_RP_Q8_NROW_LANES;
    if (bn <= 32) {
        mmq_gemm_repacked_narrow<false, 32, 1, nrl * 2, WT><<<grid, dim3(32, nrl * 2), 0, stream>>>(
            w, xq, y, ne0, ne1, n_tok, nullptr, nullptr, nullptr, nullptr, nullptr, 0, 0, ne1);
    } else if (bn <= 64) {
        mmq_gemm_repacked<false, 1, nrl, WT><<<grid, dim3(64, nrl), 0, stream>>>(
            w, xq, y, ne0, ne1, n_tok, nullptr, nullptr, nullptr, nullptr, nullptr, 0, 0, ne1);
    } else {
        mmq_gemm_repacked<false, MMQ_RP_Q8_TN, nrl, WT><<<grid, dim3(64, nrl), 0, stream>>>(
            w, xq, y, ne0, ne1, n_tok, nullptr, nullptr, nullptr, nullptr, nullptr, 0, 0, ne1);
    }
}

void ggml_cuda_mul_mat_repacked(ggml_backend_cuda_context & ctx,
        const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(src1->nb[0] == sizeof(float));
    GGML_ASSERT(dst->nb[1]  == (size_t) dst->ne[0] * sizeof(float));

    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];
    const int64_t ne10 = src1->ne[0];
    const int64_t ne11 = src1->ne[1];
    const int64_t ne12 = src1->ne[2];
    const int64_t ne13 = src1->ne[3];
    GGML_ASSERT(ne10 == ne00);

    cudaStream_t stream = ctx.stream();

    // Views are re-packed into a persistent cache buffer (pool allocs are invalid
    // inside CUDA graph capture).
    const uint8_t * w;
    if (src0->view_src != nullptr && ggml_cuda_repack_tensor_supported(src0->view_src)) {
        w = repack_view_get_cached(src0, src0->view_src, stream);
    } else {
        w = (const uint8_t *) src0->data;
    }

    const int64_t ne10_padded = GGML_PAD(ne10, MATRIX_ROW_PADDING);
    const int64_t x_stride    = ne10_padded / QK8_1;

    if (ne11 >= 1 && ne11 <= MMQ_RP_Q8_MMV_MAX_TOKENS) {
        // One token, or a narrow batch: plain Q8_1 rows, one weight pass. The
        // tiled path below computes a full 32-wide tile whatever the batch is.
        ggml_cuda_pool_alloc<block_q8_1> src1_q8_1_own;
        block_q8_1 * src1_q8_1_d;
        {
            const int64_t s11 = src1->nb[1] / sizeof(float);
            const int64_t s12 = src1->nb[2] / sizeof(float);
            const int64_t s13 = src1->nb[3] / sizeof(float);
            src1_q8_1_d = repack_quantize_src1_q8_1(ctx, src0, src1, ne10, ne10_padded,
                s11, s12, s13, ne11, ne12, ne13,
                (size_t) (ne13 * ne12 * ne11 * x_stride), src1_q8_1_own, stream);
        }
        const uint32_t dst_s1 = dst->nb[1] / sizeof(float);
        for (int64_t i3 = 0; i3 < ne13; i3++) {
        for (int64_t i2 = 0; i2 < ne12; i2++) {
            const block_q8_1 * xq = src1_q8_1_d
                              + (i3 * ne12 + i2) * ne11 * x_stride;
            float * dst_d = (float *)((char *) dst->data + i3 * dst->nb[3] + i2 * dst->nb[2]);
            if (ne11 == 1) {
                ggml_cuda_mul_mat_repacked_slice(ctx, src0, w, xq, dst_d,
                    ne00, ne01, ne11, stream);
            } else {
                switch (src0->type) {
                    case GGML_TYPE_Q8_0:
                        ggml_cuda_mul_mat_repacked_nc_t<GGML_TYPE_Q8_0>(w, xq, dst_d,
                            ne00, ne01, ne11, (uint32_t) x_stride, dst_s1, stream);
                        break;
                    case GGML_TYPE_IQ4_NL:
                        ggml_cuda_mul_mat_repacked_nc_t<GGML_TYPE_IQ4_NL>(w, xq, dst_d,
                            ne00, ne01, ne11, (uint32_t) x_stride, dst_s1, stream);
                        break;
                    case GGML_TYPE_Q4_0:
                        ggml_cuda_mul_mat_repacked_nc_t<GGML_TYPE_Q4_0>(w, xq, dst_d,
                            ne00, ne01, ne11, (uint32_t) x_stride, dst_s1, stream);
                        break;
                    default: GGML_ABORT("unsupported repack type");
                }
            }
        }
        }
        return;
    }

    // Multi-token: grouped MMQ Q8_1 layout for the tiled GEMM.
    const uint64_t n_groups   = (uint64_t) ne10_padded / (4 * QK8_1);

    const int64_t chunk_ne11 = ne11;
    ggml_cuda_pool_alloc<block_q8_1_mmq_h> src1_q8_1;
    const ggml_type x_type = repack_mmq_x_type(src0->type);
    const int x_variant  = 1 + (int) mmq_get_q8_1_ds_layout(x_type);
    block_q8_1 * src1_q8_1_d = repack_quantize_mmq_q8_1(ctx, src1, x_type, x_variant,
        ne10, ne10_padded, src1->nb[1] / sizeof(float), src1->nb[2] / sizeof(float), src1->nb[3] / sizeof(float),
        ne11, ne12, ne13, (size_t) ne13 * ne12 * ne11 * n_groups, src1_q8_1, stream);

    const int64_t s11 = src1->nb[1] / sizeof(float);
    const int64_t s12 = src1->nb[2] / sizeof(float);
    const int64_t s13 = src1->nb[3] / sizeof(float);
    const uint32_t dst_s1 = dst->nb[1] / sizeof(float);
    for (int64_t col = 0; col < ne11; col += chunk_ne11) {
        const int64_t iter_ne11 = std::min(chunk_ne11, ne11 - col);

        for (int64_t i3 = 0; i3 < ne13; i3++) {
        for (int64_t i2 = 0; i2 < ne12; i2++) {
            const block_q8_1_mmq_h * slice_base = reinterpret_cast<const block_q8_1_mmq_h *>(src1_q8_1_d)
                                  + (i3 * ne12 + i2) * iter_ne11 * n_groups;
            const block_q8_1 * xq = reinterpret_cast<const block_q8_1 *>(slice_base);
            float * dst_d = (float *)((char *) dst->data + i3 * dst->nb[3] + i2 * dst->nb[2])
                           + col*dst_s1;
            ggml_cuda_mul_mat_repacked_slice(ctx, src0, w, xq, dst_d,
                ne00, ne01, iter_ne11, stream);
        }
        }
    }
}

// Dense FFN fusion: up and gate share the single activation row, the GLU
// epilogue writes the result directly.
void ggml_cuda_mul_mat_repacked_fused(ggml_backend_cuda_context & ctx,
        const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst,
        const ggml_tensor * gate, ggml_glu_op glu_op) {
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(src1->nb[0] == sizeof(float));
    GGML_ASSERT(dst->nb[1]  == (size_t) dst->ne[0] * sizeof(float));
    GGML_ASSERT(src1->ne[1] == 1 && src1->ne[2] == 1 && src1->ne[3] == 1);
    GGML_ASSERT(gate != nullptr && gate->type == src0->type);

    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];
    const int64_t ne10 = src1->ne[0];
    GGML_ASSERT(ne10 == ne00);
    GGML_ASSERT(dst->ne[0] == ne01);

    cudaStream_t stream = ctx.stream();

    const uint8_t * w;
    if (src0->view_src != nullptr && ggml_cuda_repack_tensor_supported(src0->view_src)) {
        w = repack_view_get_cached(src0, src0->view_src, stream);
    } else {
        w = (const uint8_t *) src0->data;
    }
    const uint8_t * w_gate;
    if (gate->view_src != nullptr && ggml_cuda_repack_tensor_supported(gate->view_src)) {
        w_gate = repack_view_get_cached(gate, gate->view_src, stream);
    } else {
        w_gate = (const uint8_t *) gate->data;
    }

    const int64_t ne10_padded = GGML_PAD(ne10, MATRIX_ROW_PADDING);
    const int64_t x_stride    = ne10_padded / QK8_1;
    const int64_t s11 = src1->nb[1] / sizeof(float);
    const int64_t s12 = src1->nb[2] / sizeof(float);
    const int64_t s13 = src1->nb[3] / sizeof(float);

    ggml_cuda_pool_alloc<block_q8_1> xq_own;
    block_q8_1 * xq = repack_quantize_src1_q8_1(ctx, src0, src1, ne10, ne10_padded,
        s11, s12, s13, 1, 1, 1, (size_t) x_stride, xq_own, stream);

    float * dst_d = (float *) dst->data;
    const dim3 grid((ne01 + 15) / 16, 1, 1);
    switch (src0->type) {
        case GGML_TYPE_Q8_0:
            mul_mat_vec_q8_0_repacked<16, 16, false, 64, true><<<grid, 1024, 0, stream>>>(
                w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01,
                nullptr, nullptr, nullptr, 0, 1, 0, 0, 0,
                w_gate, nullptr, nullptr, glu_op);
            break;
        case GGML_TYPE_IQ4_NL:
            mul_mat_vec_rp<GGML_TYPE_IQ4_NL, 16, 16, false, 64, true><<<grid, 1024, 0, stream>>>(
                w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01,
                nullptr, 0, 0, 0, 1, w_gate, nullptr, nullptr, glu_op);
            break;
        case GGML_TYPE_Q4_0:
            mul_mat_vec_rp<GGML_TYPE_Q4_0, 16, 16, false, 64, true><<<grid, 1024, 0, stream>>>(
                w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01,
                nullptr, 0, 0, 0, 1, w_gate, nullptr, nullptr, glu_op);
            break;
        default: GGML_ABORT("unsupported repack type");
    }
}

// Narrow batch (2..8 tokens): one multi-column mat-vec pass per slice.
template <ggml_type WT>
static void ggml_cuda_mul_mat_repacked_nc_t(
        const uint8_t * w, const block_q8_1 * xq, float * dst_d,
        const int64_t ne00, const int64_t ne01, const int64_t ne11,
        const uint32_t xs, const uint32_t ys, cudaStream_t stream) {
    // Geometry per width, measured on Qwen3.8-27B (pp512 at that ubatch, 4x
    // MI50 -sm tensor). The row count per lane is small once the tensor is
    // split, so scheduling granularity beats per-group reuse and 64-thread
    // workgroups spread over the CUs win. Two rows per lane reuse the
    // activation block across rows and pay from 3 tokens up; at 2 there is
    // too little to amortize, and 4 rows spills.
    // Lane width per row, chosen from the ROW LENGTH and the batch width.
    //
    // LANES is the number of lanes cooperating on one row. The strided loop
    // runs ne0/32/LANES iterations and is followed by a log2(LANES)-step DPP
    // reduction, so the useful knob is how much work a lane gets before it has
    // to reduce. Narrowing the lane group raises that, and raises occupancy,
    // but it also cuts the block count that hides memory latency.
    //
    // Which effect wins is decided by the row length, and two models disagree
    // about the batch width alone, so that is the wrong axis. Measured pp512
    // at the ubatch, 2x MI50 -sm layer, narrow-lane against the 64-lane shape:
    //
    //                       ne0    K-blocks   width 2   width 3
    //   Qwen3.6-27B        5120       160      +13.8%    +2.1%
    //   Qwen3.6-35B-A3B    2048        64       -9.5%    -0.7%
    //
    // A 2048-wide row is 64 K-blocks, so splitting it over 16 lanes leaves 4
    // iterations against a 4-step reduction while cutting the grid by 16x -
    // the row is simply too short to pay for the lost parallelism. Gate on the
    // block count so the choice follows the shape and not the model.
    // Narrowing the lane group raises the work a lane does before it reduces,
    // and multiplies the rows a block covers, which shrinks the grid. Both
    // have a floor and BOTH must hold.
    //
    // A lane accumulates n_blocks/LANES times, then pays a log2(LANES)-step
    // reduction, so require at least four accumulations at the widest group we
    // would narrow to. Measured on two models: 160 and 128 blocks win, 64
    // loses at every lane width, so the break is between them.
    //
    // The grid condition is what the row length alone cannot see - a 48-row
    // tensor is a 2-block grid however long its rows are.
    constexpr int RP_NC_MIN_ACCUM   = 4;
    constexpr int RP_NC_LANES_WIDE  = 32;   // widest group we ever narrow to
    const int     nsm       = ggml_cuda_info().devices[ggml_cuda_get_device()].nsm;
    const int64_t n_blocks  = ne00 / 32;
    const int64_t min_grid  = 2LL * nsm;
    const bool    work_ok   = n_blocks >= RP_NC_MIN_ACCUM * RP_NC_LANES_WIDE;
    const bool    lanes16_ok = work_ok && (ne01 + 31) / 32 >= min_grid;
    const bool    lanes32_ok = work_ok && (ne01 +  7) /  8 >= min_grid;
    switch (ne11) {
        case 2: {
            if (lanes16_ok) {
                // 16 lanes to a row, 32 rows per block, one row per lane.
                const dim3 grid((ne01 + 31) / 32, 1, 1);
                mul_mat_vec_repacked_nc<32, 8, 2, 1, 16, WT><<<grid, 512, 0, stream>>>(
                    w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01, xs, ys);
            } else {
                const dim3 grid((ne01 + 1) / 2, 1, 1);
                mul_mat_vec_repacked_nc<2, 2, 2, 1, 64, WT><<<grid, 128, 0, stream>>>(
                    w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01, xs, ys);
            }
        } break;
        case 3: {
            if (lanes32_ok) {
                // 32 lanes to a row, 8 rows per block, one row per lane.
                const dim3 grid((ne01 + 7) / 8, 1, 1);
                mul_mat_vec_repacked_nc<8, 4, 3, 1, 32, WT><<<grid, 256, 0, stream>>>(
                    w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01, xs, ys);
            } else {
                const dim3 grid((ne01 + 1) / 2, 1, 1);
                mul_mat_vec_repacked_nc<2, 1, 3, 2, 64, WT><<<grid, 64, 0, stream>>>(
                    w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01, xs, ys);
            }
        } break;
        case 4: {
            const dim3 grid((ne01 + 1) / 2, 1, 1);
            mul_mat_vec_repacked_nc<2, 1, 4, 2, 64, WT><<<grid, 64, 0, stream>>>(
                w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01, xs, ys);
        } break;
        case 5: {
            const dim3 grid((ne01 + 1) / 2, 1, 1);
            mul_mat_vec_repacked_nc<2, 1, 5, 2, 64, WT><<<grid, 64, 0, stream>>>(
                w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01, xs, ys);
        } break;
        case 6: {
            const dim3 grid((ne01 + 1) / 2, 1, 1);
            mul_mat_vec_repacked_nc<2, 1, 6, 2, 64, WT><<<grid, 64, 0, stream>>>(
                w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01, xs, ys);
        } break;
        case 7: {
            const dim3 grid((ne01 + 1) / 2, 1, 1);
            mul_mat_vec_repacked_nc<2, 1, 7, 2, 64, WT><<<grid, 64, 0, stream>>>(
                w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01, xs, ys);
        } break;
        case 8: {
            const dim3 grid((ne01 + 1) / 2, 1, 1);
            mul_mat_vec_repacked_nc<2, 1, 8, 2, 64, WT><<<grid, 64, 0, stream>>>(
                w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01, xs, ys);
        } break;
        default: GGML_ABORT("nc mat-vec: unsupported batch width");
    }
}

// Dispatch one 2D slice: mat-vec vs tiled GEMM.
static void ggml_cuda_mul_mat_repacked_slice(ggml_backend_cuda_context & ctx,
        const ggml_tensor * src0, const uint8_t * w, const block_q8_1 * xq,
        float * dst_d, const int64_t ne00, const int64_t ne01, const int64_t ne11,
        cudaStream_t stream) {
    if (ne11 == 1) {
        switch (src0->type) {
            case GGML_TYPE_Q8_0: {
                const dim3 grid((ne01 + 15) / 16, 1, 1);
                // Measured A/B hook: route Q8_0 through the shared generic
                // mat-vec. If it holds parity the bespoke kernel retires.
                static const bool generic_mmv = getenv("GGML_RP_GENERIC_MMV") != nullptr;
                if (generic_mmv) {
                    mul_mat_vec_rp<GGML_TYPE_Q8_0, 16, 16, false><<<grid, 1024, 0, stream>>>(
                        w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01,
                        nullptr, 1, 0, 0, 0);
                    break;
                }
                mul_mat_vec_q8_0_repacked<16, 16, false><<<grid, 1024, 0, stream>>>(
                    w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01,
                    nullptr, nullptr, nullptr, 0, 1, 0, 0, 0,
                    nullptr, nullptr, nullptr, GGML_GLU_OP_REGLU);
            } break;
            case GGML_TYPE_IQ4_NL: {
                const dim3 grid((ne01 + 15) / 16, 1, 1);
                mul_mat_vec_rp<GGML_TYPE_IQ4_NL, 16, 16, false><<<grid, 1024, 0, stream>>>(
                    w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01,
                    nullptr, 0, 0, 0, 1, 0, 0, 0);
            } break;
            case GGML_TYPE_Q4_0: {
                const dim3 grid((ne01 + 15) / 16, 1, 1);
                mul_mat_vec_rp<GGML_TYPE_Q4_0, 16, 16, false><<<grid, 1024, 0, stream>>>(
                    w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01,
                    nullptr, 0, 0, 0, 1, 0, 0, 0);
            } break;
            default: GGML_ABORT("unsupported repack type");
        }
        return;
    }

    // Multi-token. The token tile picks between parallelism and weight reuse:
    // a 128-wide tile on a short batch of a small matrix leaves the CUs idle,
    // while a 32-wide tile on a long batch re-reads the weights four times.
    // Measured on Qwen3.6-35B-A3B (M 2048/512/4096, ub 512) the narrow tile is
    // +4%; on the 9B (M >= 4096) and at ub >= 1024 the wide tile wins, and the
    // 9B loses 24% if the narrow one is forced. GGML_RP_DENSE_BN overrides it.
    const int nrl = MMQ_RP_Q8_NROW_LANES;
    const int mmq_bm = MMQ_RP_Q8_BM;
    static const int bn_env = [] {
        const char * e = getenv("GGML_RP_DENSE_BN");
        return e ? atoi(e) : 0;
    }();
    const int row_tiles = (int) ((ne01 + mmq_bm - 1) / mmq_bm);
    // Below 128 tokens a 128-wide token tile spends most of its columns on
    // padding, whatever the matrix size.
    const bool starved = ne11 < 128 || (ne11 <= 512 && row_tiles <= 32);
    const int bn = bn_env ? bn_env : (starved ? 32 : 128);
    const dim3 grid((ne01 + mmq_bm - 1) / mmq_bm, (ne11 + bn - 1) / bn, 1);
    switch (src0->type) {
        case GGML_TYPE_Q8_0:
            repack_dense_gemm_launch<GGML_TYPE_Q8_0>(stream, w, xq, dst_d,
                (uint32_t) ne00, (uint32_t) ne01, (uint32_t) ne11, bn, grid);
            break;
        case GGML_TYPE_IQ4_NL:
            repack_dense_gemm_launch<GGML_TYPE_IQ4_NL>(stream, w, xq, dst_d,
                (uint32_t) ne00, (uint32_t) ne01, (uint32_t) ne11, bn, grid);
            break;
        case GGML_TYPE_Q4_0:
            repack_dense_gemm_launch<GGML_TYPE_Q4_0>(stream, w, xq, dst_d,
                (uint32_t) ne00, (uint32_t) ne01, (uint32_t) ne11, bn, grid);
            break;
        default: GGML_ABORT("unsupported repack type");
    }
    GGML_UNUSED(ctx);
}

// Fused MMV entry for repacked weights (dense): single-token only, fuses the up and
// gate dot products (shared quantized input) plus optional biases and the GLU op.
// Dispatch: ggml_cuda_try_fuse `{op,op,GLU}` and `{op,ADD}` repack branches.
