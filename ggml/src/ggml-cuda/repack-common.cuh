// Shared layout helpers and device-side data types for the Q8_0/Q4_0/IQ4_NL repacked-weight path.
#pragma once

#include "common.cuh"
#include "mmq.cuh"
#include "quantize.cuh"
#include "vecdotq.cuh"   // get_int_from_table_16, for the IQ4_NL nibble expand

#include <cstddef>

#define MMQ_RP_Q8_BK 4
#define MMQ_RP_Q8_TN 2
#define MMQ_RP_Q8_BM 64
#define MMQ_RP_Q8_NROW_LANES 4
// Widths at or below this take one multi-column mat-vec pass instead of the
// tiled GEMM. Measured on Qwen3.8-27B pp512, 4x MI50 -sm tensor: the mat-vec
// leads from 2 tokens (77.2 against the tile's 14.3) through 8 (159.6 against
// 98.9), and the tile takes back over by 12, so the crossover is 8 to 12.
#define MMQ_RP_Q8_MMV_MAX_TOKENS 8
// MoE narrow batch: at or below this many tokens, run one mat-vec per
// assignment instead of the tiled GEMM. Expert token counts are per expert, so
// at these widths nearly every active expert holds a single assignment.
#define MMQ_RP_Q8_MOE_MMV_MAX_TOKENS 8

// qs plane row stride in BYTES: ne0 (one byte per quant), bumped by 16 whenever
// it lands on a multiple of 128 so row starts do not alias HBM channels. 16
// keeps uint4 alignment and costs half of the padding sub-block it replaces,
// which also only fired when n_sub was a power of two and so missed shapes like
// ne0=5120. Scales keep their own plane: the narrow mat-vec reads several rows
// per lane and wants them adjacent.
template <typename T>
static __host__ __device__ inline T repack_qs_stride(const T ne0) {
    T rs = ne0;
    if (rs % 128 == 0) {
        rs += 16;
    }
    return rs;
}

// qs row stride and scale row bytes: Q8_0 rows carry ne0 payload bytes and
// 2 B f16 scales per sub-block; Q4_0 / IQ4_NL rows carry packed nibbles
// (ne0/2 B) and 2 B f16 scales. Both share the de-alias bump.
template <typename T>
static __host__ __device__ inline T repack_qs_row_stride(const ggml_type type, const T ne0) {
    return repack_qs_stride((type == GGML_TYPE_Q4_0 || type == GGML_TYPE_IQ4_NL) ? ne0 / 2 : ne0);
}
template <typename T>
static __host__ __device__ inline T repack_scale_row_bytes(const ggml_type type, const T ne0) {
    GGML_UNUSED(type);
    return (ne0 / 32) * 2;
}

static inline size_t repack_gcn_nbytes(const ggml_type type, const int64_t ne0, const int64_t ne1) {
    GGML_ASSERT(ne0 % 32 == 0);
    switch (type) {
        case GGML_TYPE_Q8_0:
            return (size_t) ne1 * ((size_t) repack_qs_stride(ne0) + (size_t)(ne0 / 32) * 2);
        case GGML_TYPE_IQ4_NL:
        case GGML_TYPE_Q4_0:
            return (size_t) ne1 * ((size_t) repack_qs_stride(ne0 / 2) + (size_t)(ne0 / 32) * 2);
        default:             GGML_ABORT("unsupported repack type");
    }
}

// Bytes per sub-block in the qs plane, and uint4 loads needed to fetch one.
static __host__ __device__ inline int repack_qs_bytes(const ggml_type type) {
    return (type == GGML_TYPE_Q4_0 || type == GGML_TYPE_IQ4_NL) ? 16 : 32;
}

#if defined(GGML_USE_HIP) && defined(__gfx906__)
// IQ4_NL sub-block: 16 payload bytes coding 32 values as nibbles, looked up in
// kvalues_iq4nl. get_int_from_table_16 returns v.x for the four LOW nibbles and
// v.y for the four HIGH nibbles, i.e. v.x is values 0..15 and v.y is 16..31,
// the lo/hi split dp4a consumes.
static __device__ __forceinline__ void rp_iq4nl_expand(
        const uint4 nib, uint4 & lo, uint4 & hi) {
    const int2 v0 = get_int_from_table_16((int) nib.x, kvalues_iq4nl);
    const int2 v1 = get_int_from_table_16((int) nib.y, kvalues_iq4nl);
    const int2 v2 = get_int_from_table_16((int) nib.z, kvalues_iq4nl);
    const int2 v3 = get_int_from_table_16((int) nib.w, kvalues_iq4nl);
    lo = make_uint4((uint32_t) v0.x, (uint32_t) v1.x, (uint32_t) v2.x, (uint32_t) v3.x);
    hi = make_uint4((uint32_t) v0.y, (uint32_t) v1.y, (uint32_t) v2.y, (uint32_t) v3.y);
}

// Q4_0 sub-block: same nibble layout and lo/hi split as IQ4_NL, but the map is
// q - 8 instead of a table. Bytewise subtract without inter-byte borrow: set
// bit 7, subtract, clear bit 7. Gives the signed int8 the dp4a path wants.
static __device__ __forceinline__ uint32_t rp_sub8(const uint32_t v) {
    return ((v | 0x80808080u) - 0x08080808u) ^ 0x80808080u;
}

static __device__ __forceinline__ void rp_q4_0_expand(
        const uint4 nib, uint4 & lo, uint4 & hi) {
    const uint32_t M = 0x0F0F0F0Fu;
    lo = make_uint4(rp_sub8(nib.x & M), rp_sub8(nib.y & M),
                    rp_sub8(nib.z & M), rp_sub8(nib.w & M));
    hi = make_uint4(rp_sub8((nib.x >> 4) & M), rp_sub8((nib.y >> 4) & M),
                    rp_sub8((nib.z >> 4) & M), rp_sub8((nib.w >> 4) & M));
}

#endif

template <int CW>
static __device__ __forceinline__ int sX_swizzle(int lr) {
    if constexpr (CW == 64) {
        const int n  = lr >> 6;
        int       tx = lr & 63;
        tx ^= (tx >> 5) << 4;
        return (n << 6) | tx;
    } else {
        const int n  = lr >> 5;
        int       tx = lr & 31;
        tx ^= (tx >> 4) << 3;
        return (n << 5) | tx;
    }
}

struct rp_x_sub {
    uint4 q0, q1;
    float d;
};

struct block_q8_1_mmq_h {
    float  d4[4];
    int8_t qs[QK8_1_MMQ];
};

static_assert(sizeof(block_q8_1_mmq_h) == sizeof(block_q8_1_mmq),
              "Unexpected block_q8_1_mmq_h size");
static_assert(offsetof(block_q8_1_mmq_h, d4) == offsetof(block_q8_1_mmq, d4),
              "block_q8_1_mmq_h d4 offset mismatch");
static_assert(offsetof(block_q8_1_mmq_h, qs) == offsetof(block_q8_1_mmq, qs),
              "block_q8_1_mmq_h qs offset mismatch");

// DS4 view of the same 16-byte header: affine quants need the activation SUM per 32-block for their -dmin*m term, and quantize_mmq_q8_1_cuda emits half2 {d, s} slots for them (MMQ_Q8_1_DS_LAYOUT_DS4, keyed off the src0 type it is passed).
struct block_q8_1_mmq_ds_h {
    half2  ds4[4];
    int8_t qs[QK8_1_MMQ];
};

static_assert(sizeof(block_q8_1_mmq_ds_h) == sizeof(block_q8_1_mmq),
              "Unexpected block_q8_1_mmq_ds_h size");
static_assert(offsetof(block_q8_1_mmq_ds_h, qs) == offsetof(block_q8_1_mmq, qs),
              "block_q8_1_mmq_ds_h qs offset mismatch");

// The generic mmq table routes Q4_0 to the DS4 activation layout because the
// canonical kernel needs the block sum. The repacked kernels fold the -8 into
// the weights, so they read D4 only. Quantize as Q8_0 (same 32-value blocks,
// D4 layout) for repacked Q4_0.
static inline ggml_type repack_mmq_x_type(const ggml_type wt) {
    return wt == GGML_TYPE_Q4_0 ? GGML_TYPE_Q8_0 : wt;
}

struct sXq_row_q8 {
    uint4 q[MMQ_RP_Q8_BK][2];
    uint4 pad;
};

static_assert(sizeof(sXq_row_q8) == (2 * MMQ_RP_Q8_BK + 1) * 16,
              "unexpected sXq row size");

static __device__ __forceinline__ uint4 rp_ldcs_u4(const uint4 * __restrict__ p) {
    return *p;
}

#if defined(GGML_USE_HIP) && defined(__gfx906__)

#define RP_DPP_ADD(name, nop, dpp_ctrl)                                      \
    static __device__ __forceinline__ float name(const float x) {            \
        float r;                                                             \
        asm volatile(                                                        \
            nop                                                              \
            "v_add_f32_dpp %0, %1, %1 " dpp_ctrl " row_mask:0xf bank_mask:0xf" \
            : "=v"(r) : "v"(x) : "memory");                                  \
        return r;                                                            \
    }

RP_DPP_ADD(rp_dpp_add_xor1, "s_nop 4\n", "quad_perm:[1,0,3,2]")
RP_DPP_ADD(rp_dpp_add_xor2, "s_nop 1\n", "quad_perm:[2,3,0,1]")
RP_DPP_ADD(rp_dpp_add_xor8, "s_nop 1\n", "row_ror:8")

#undef RP_DPP_ADD

static __device__ __forceinline__ float rp_dpp_xfer_xor4(const float x) {
    int d;
    asm volatile("v_mov_b32 %0, %1\n"
                 "s_nop 1\n"
                 "v_mov_b32_dpp %0, %1 row_shl:4 row_mask:0xf bank_mask:0x5\n"
                 "v_mov_b32_dpp %0, %1 row_shr:4 row_mask:0xf bank_mask:0xa\n"
                 : "=v"(d) : "v"(__float_as_int(x)) : "memory");
    return __int_as_float(d);
}

static __device__ __forceinline__ float rp_dpp_xfer_xor16(const float x) {
    int d;
    asm volatile("ds_swizzle_b32 %0, %1 offset:swizzle(SWAP,16)\n"
                 "s_waitcnt lgkmcnt(0)\n"
                 : "=v"(d) : "v"(__float_as_int(x)) : "memory");
    return __int_as_float(d);
}

template <int width>
static __device__ __forceinline__ float rp_warp_reduce_sum(const float x) {
    static_assert(width >= 1 && (width & (width - 1)) == 0);
    float r = x;
    if constexpr (width >=  2) { r = rp_dpp_add_xor1(r); }
    if constexpr (width >=  4) { r = rp_dpp_add_xor2(r); }
    if constexpr (width >=  8) { r += rp_dpp_xfer_xor4(r); }
    if constexpr (width >= 16) { r = rp_dpp_add_xor8(r); }
    if constexpr (width >= 32) { r += rp_dpp_xfer_xor16(r); }
    if constexpr (width >= 64) { r += __shfl_xor(r, 32, 64); }
    return r;
}

#else

template <int width>
static __device__ __forceinline__ float rp_warp_reduce_sum(const float x) {
    return warp_reduce_sum<width>(x);
}

#endif

__device__ __forceinline__ rp_x_sub rp_x_sub_from_mmq_group(
        const block_q8_1_mmq_h * __restrict__ group, const uint32_t col, const uint32_t lk) {
    const block_q8_1_mmq_h & m = group[col];
    const uint4           * mq = reinterpret_cast<const uint4 *>(m.qs + lk * QK8_1);
    rp_x_sub out;
    out.q0 = mq[0];
    out.q1 = mq[1];
    out.d  = m.d4[lk];
    return out;
}

// DS4 variant: same qs fetch, scale AND sum decoded from the half2 slot.
__device__ __forceinline__ rp_x_sub rp_x_sub_from_mmq_group_ds(
        const block_q8_1_mmq_h * __restrict__ group, const uint32_t col, const uint32_t lk,
        float & s) {
    const block_q8_1_mmq_ds_h & m = *reinterpret_cast<const block_q8_1_mmq_ds_h *>(group + col);
    const uint4               * mq = reinterpret_cast<const uint4 *>(m.qs + lk * QK8_1);
    rp_x_sub out;
    out.q0 = mq[0];
    out.q1 = mq[1];
    const float2 ds = __half22float2(m.ds4[lk]);
    out.d = ds.x;
    s     = ds.y;
    return out;
}

// Zero a raw scale slot for a dead row (the nc kernels clamp the row index and kill the term through the slot).
// Integral slots and the uint2 affine slot.
template <typename T>
__device__ __forceinline__ T rp_slot_keep(const T v, const bool keep) {
    return keep ? v : (T) 0;
}
template <>
__device__ __forceinline__ uint2 rp_slot_keep<uint2>(const uint2 v, const bool keep) {
    return keep ? v : make_uint2(0u, 0u);
}

// ---- per-type traits ---------------------------------------------------------
// Everything the shared kernels need to know about a repacked quant type.
// Adding a type = one specialization here + one line in RP_FOREACH_TYPE below
// (plus host/device extract functions until those join the traits too).
//   geom    - per-tensor constants, built once per kernel launch
//   load_w  - fetch sub-block (wrow, sb): 32 int8 values as a lo/hi uint4 pair
//             plus the raw scale slot. THE hot path.
//   scale   - decode a raw scale slot to float.
template <ggml_type T> struct rp_traits;

template <> struct rp_traits<GGML_TYPE_Q8_0> {
    using slot_t = uint16_t;   // f16 scale
    static constexpr bool split_scales = false;
    static constexpr bool affine       = false;
    static constexpr bool raw_lds = false;
    // de-aliased qs rows [ne1 x rs], then an f16 scale plane [ne1 x n_sub].
    struct geom {
        uint32_t rs, rs_u4, n_sub;
        size_t   dplane_off;
        __host__ __device__ geom(uint32_t ne0, uint32_t ne1)
            : rs(repack_qs_row_stride(GGML_TYPE_Q8_0, ne0)), rs_u4(rs >> 4),
              n_sub(ne0 >> 5), dplane_off((size_t) ne1 * rs) {}
    };
#if defined(GGML_USE_HIP) && defined(__gfx906__)
    static __device__ __forceinline__ void load_w(const uint8_t * __restrict__ wbase,
            const geom & g, uint32_t wrow, uint32_t sb, uint4 & lo, uint4 & hi, uint16_t & d) {
        const uint4 * qsp = reinterpret_cast<const uint4 *>(wbase);
        const size_t  q4  = (size_t) wrow * g.rs_u4 + 2 * sb;
        lo = rp_ldcs_u4(qsp + q4);
        hi = rp_ldcs_u4(qsp + q4 + 1);
        d  = reinterpret_cast<const uint16_t *>(wbase + g.dplane_off)[(size_t) wrow * g.n_sub + sb];
    }
    static __device__ __forceinline__ float scale(const uint16_t s) {
        return __half2float(*reinterpret_cast<const __half *>(&s));
    }
#endif
};


template <> struct rp_traits<GGML_TYPE_IQ4_NL> {
    using slot_t = uint16_t;   // f16 scale
    static constexpr bool split_scales = false;
    static constexpr bool affine       = false;
    // De-aliased nibble rows [ne1 x rs], then an f16 scale plane [ne1 x n_sub],
    // values from kvalues_iq4nl, one scale per 32 weights. Raw nibbles are
    // staged in LDS and expanded when the register tile is filled.
    struct geom {
        uint32_t rs, rs_u4, n_sub;
        size_t   dplane_off;
        __host__ __device__ geom(uint32_t ne0, uint32_t ne1)
            : rs(repack_qs_row_stride(GGML_TYPE_IQ4_NL, ne0)), rs_u4(rs >> 4),
              n_sub(ne0 >> 5), dplane_off((size_t) ne1 * rs) {}
    };
    static constexpr bool raw_lds = true;
#if defined(GGML_USE_HIP) && defined(__gfx906__)
    static __device__ __forceinline__ void load_w(const uint8_t * __restrict__ wbase,
            const geom & g, uint32_t wrow, uint32_t sb, uint4 & lo, uint4 & hi, uint16_t & d) {
        const uint4 * qsp = reinterpret_cast<const uint4 *>(wbase);
        rp_iq4nl_expand(rp_ldcs_u4(qsp + (size_t) wrow * g.rs_u4 + sb), lo, hi);
        d = reinterpret_cast<const uint16_t *>(wbase + g.dplane_off)[(size_t) wrow * g.n_sub + sb];
    }
    static __device__ __forceinline__ void load_w_raw(const uint8_t * __restrict__ wbase,
            const geom & g, uint32_t wrow, uint32_t sb, uint4 & raw, uint16_t & d) {
        const uint4 * qsp = reinterpret_cast<const uint4 *>(wbase);
        raw = rp_ldcs_u4(qsp + (size_t) wrow * g.rs_u4 + sb);
        d   = reinterpret_cast<const uint16_t *>(wbase + g.dplane_off)[(size_t) wrow * g.n_sub + sb];
    }
    static __device__ __forceinline__ float scale(const uint16_t s) {
        return __half2float(*reinterpret_cast<const __half *>(&s));
    }
    static __device__ __forceinline__ void expand_raw(const uint4 raw, uint4 & lo, uint4 & hi) {
        rp_iq4nl_expand(raw, lo, hi);
    }
#endif
};

template <> struct rp_traits<GGML_TYPE_Q4_0> {
    using slot_t = uint16_t;   // f16 scale
    static constexpr bool split_scales = false;
    static constexpr bool affine       = false;
    // IQ4_NL's twin: de-aliased nibble rows [ne1 x rs], then an f16 scale plane
    // [ne1 x n_sub]. The nibbles are q and the kernel gets q - 8, folded at
    // expand time, so every epilogue is the Q8_0 one.
    struct geom {
        uint32_t rs, rs_u4, n_sub;
        size_t   dplane_off;
        __host__ __device__ geom(uint32_t ne0, uint32_t ne1)
            : rs(repack_qs_row_stride(GGML_TYPE_Q4_0, ne0)), rs_u4(rs >> 4),
              n_sub(ne0 >> 5), dplane_off((size_t) ne1 * rs) {}
    };
    static constexpr bool raw_lds = true;
#if defined(GGML_USE_HIP) && defined(__gfx906__)
    static __device__ __forceinline__ void load_w(const uint8_t * __restrict__ wbase,
            const geom & g, uint32_t wrow, uint32_t sb, uint4 & lo, uint4 & hi, uint16_t & d) {
        const uint4 * qsp = reinterpret_cast<const uint4 *>(wbase);
        rp_q4_0_expand(rp_ldcs_u4(qsp + (size_t) wrow * g.rs_u4 + sb), lo, hi);
        d = reinterpret_cast<const uint16_t *>(wbase + g.dplane_off)[(size_t) wrow * g.n_sub + sb];
    }
    static __device__ __forceinline__ void load_w_raw(const uint8_t * __restrict__ wbase,
            const geom & g, uint32_t wrow, uint32_t sb, uint4 & raw, uint16_t & d) {
        const uint4 * qsp = reinterpret_cast<const uint4 *>(wbase);
        raw = rp_ldcs_u4(qsp + (size_t) wrow * g.rs_u4 + sb);
        d   = reinterpret_cast<const uint16_t *>(wbase + g.dplane_off)[(size_t) wrow * g.n_sub + sb];
    }
    static __device__ __forceinline__ float scale(const uint16_t s) {
        return __half2float(*reinterpret_cast<const __half *>(&s));
    }
    static __device__ __forceinline__ void expand_raw(const uint4 raw, uint4 & lo, uint4 & hi) {
        rp_q4_0_expand(raw, lo, hi);
    }
#endif
};





// One entry per supported repack type - generates the dispatch switches.
#define RP_FOREACH_TYPE(X)     X(GGML_TYPE_Q8_0)          X(GGML_TYPE_Q4_0)          X(GGML_TYPE_IQ4_NL)
void repack_q8_0_host(const block_q8_0 * blocks, uint8_t * dst, const int64_t ne0, const int64_t ne1);
void repack_q4_0_host(const block_q4_0 * blocks, uint8_t * dst, const int64_t ne0, const int64_t ne1);
void repack_iq4nl_host(const block_iq4_nl * blocks, uint8_t * dst, const int64_t ne0, const int64_t ne1);
void repack_host(ggml_type type, const void * blocks, uint8_t * dst, const int64_t ne0, const int64_t ne1);

const uint8_t * repack_view_get_cached(
        const ggml_tensor * view, const ggml_tensor * base, cudaStream_t stream);

// Drop cached re-packed views that live inside [base, base+size) - called when the
// owning buffer is freed so the allocator cannot hand the same addresses to a
// different model and hit a stale entry.
void repack_view_cache_purge(int device, const void * base, size_t size);

// Quantize src1 into Q8_1 rows, reusing the graph-wide activation cache when the
// same activation was already quantized to this exact layout. One activation
// feeds several matmuls back to back - q/k/v off one attention norm, router and
// routed gate/up off one ffn norm - and each re-quantized its own copy, so the
// launch count tracked the matmul count exactly. quantize_row_q8_1_cuda
// always writes the plain row layout, so variant 0 is the same
// layout the canonical mat-vec caches under and the two share entries. The MoE
// sites pass the rows flattened as (n_cols, 1, 1) where the dense sites pass
// (ne11, ne12, ne13). That is still the same bytes: the kernel writes
// i_cont = ((i3*ne2 + i2)*ne1 + i1)*ne0 + i0, which is contiguous either way,
// and it can only collide in the cache when the strides and the size already
// match, which is the contiguous case where both walk src1 identically.
// Fusing the quantize into the mat-vec instead is the wrong shape: it is a
// shared producer, and making every block re-quantize the row cost 42 percent.
static inline block_q8_1 * repack_quantize_src1_q8_1(
        ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
        int64_t ne10, int64_t ne10_padded, int64_t s11, int64_t s12, int64_t s13,
        int64_t nrows, int64_t ne2, int64_t ne3, size_t nblocks,
        ggml_cuda_pool_alloc<block_q8_1> & fallback, cudaStream_t stream) {
    GGML_UNUSED(src0);
    bool hit = false;
    char * buf = ggml_cuda_q8_1_cache_acquire(ctx, src1, /*variant =*/ 0, ne10_padded,
        s11, s12, s13, nblocks * sizeof(block_q8_1), hit);
    if (buf == nullptr) {
        fallback.alloc(ctx.pool(), nblocks);
        buf = (char *) fallback.get();
    }
    block_q8_1 * dstq = (block_q8_1 *) buf;
    if (!hit) {
        quantize_row_q8_1_cuda((const float *) src1->data, dstq,
            ne10, s11, s12, s13, ne10_padded, nrows, ne2, ne3, stream);
    }
    return dstq;
}

// Same for the grouped layout the tiled GEMM reads. `variant` separates the
// plain q8_1 groups (1) from the affine DS4 groups (2): the two are not
// interchangeable, and the canonical MMQ path uses the same numbering.
static inline block_q8_1 * repack_quantize_mmq_q8_1(
        ggml_backend_cuda_context & ctx, const ggml_tensor * src1, ggml_type x_type, int variant,
        int64_t ne10, int64_t ne10_padded, int64_t s11, int64_t s12, int64_t s13,
        int64_t nrows, int64_t ne2, int64_t ne3, size_t nblocks,
        ggml_cuda_pool_alloc<block_q8_1_mmq_h> & fallback, cudaStream_t stream) {
    bool hit = false;
    char * buf = ggml_cuda_q8_1_cache_acquire(ctx, src1, variant, ne10_padded,
        s11, s12, s13, nblocks * sizeof(block_q8_1_mmq_h), hit);
    if (buf == nullptr) {
        fallback.alloc(ctx.pool(), nblocks);
        buf = (char *) fallback.get();
    }
    if (!hit) {
        quantize_mmq_q8_1_cuda((const float *) src1->data, nullptr, buf,
            x_type, ne10, s11, s12, s13, ne10_padded, nrows, ne2, ne3, stream);
    }
    return (block_q8_1 *) buf;
}
