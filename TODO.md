# TODO

Work items after the repack decode/prefill changes.

## Baseline

2x MI50 (gfx906), Qwen3.6-35B-A3B IQ4_NL, `-sm tensor`, unless noted.

| test | result |
|---|---:|
| MoE pp512 | 2457 t/s |
| MoE pp2048 ub 512 / 1024 / 2048 | 2207 / 2540 / 2975 t/s |
| MoE tg128 | 79.9 t/s |
| 9B dense IQ4_NL TP2 pp512 | 1927 t/s |
| 9B dense IQ4_NL TP2 tg128 | 100.7 t/s |
| 9B dense IQ4_NL single GPU tg128 | 78.8 t/s |
| 9B dense Q8 single GPU pp512 / tg128 | 1240 / 58.5 t/s |

Current as of `1416cfddf`. The 9B rows predate the dense tile change; re-measure
before comparing them against anything.

Commits: `133600a9a`, `d0695b3a1`, `4b281e210`, `931c3bd10`, `3a779a776`,
`85b110080`, `b77535738`, `e5241bae4`, `1416cfddf`.

PPL checks that must keep matching canonical:

- MoE c=512: 3.3537 (canonical 3.3834)
- MoE c=2048: 4.1616 (canonical 4.1818)
- MoE c=4096: 4.0885 (canonical 4.0988)
- dense Q8 c=64 / c=512: 3.2568 / 3.5306
- cli text for MoE TP2, dense Q8/Q4_0/IQ4_NL must match `--no-repack`

## Upstream sync

Two classes of change, with very different risk.

### Fork-only files: no merge conflict possible

`repack-*.{cu,cuh}` (about 3300 lines), `gated_delta_net_colthread.{cu,cuh}`
(574), `benches/ai-gfx906` (455). Upstream does not have these files, so every
rebase applies them untouched. All the tile, fusion and selector work lives here.

### Upstream files we patch: conflict and silent-breakage risk

| file | our diff | upstream commits since 2026-09-01 | risk |
|---|---:|---:|---|
| `src/llama-model.cpp` | +35/-21 | 89 | high |
| `ggml/src/ggml-cuda/ggml-cuda.cu` | +145 | 87 | high, but localised |
| `ggml/src/ggml-backend-meta.cpp` | +210 | 14 | medium |
| `ggml/src/ggml-cuda/common.cuh` | +30 | 26 | medium |
| `ggml/src/ggml-cuda/gated_delta_net.cu` | +8 | 4 | low |

Textual conflicts are not the worry; these are the silent ones:

- **`llama-model.cpp`, buft selection.** `make_gpu_buft_list` puts the extra
  buft (the repack buffer) ahead of the default so weights prefer it. If upstream
  reshapes that function the repack can stop firing with no error: the model
  loads canonical weights and prefill falls back to the `--repack 0` numbers.
- **`ggml-cuda.cu`, quantization cache.** +145 lines, mostly the q8_1 cache. The
  fork has the same patch, so if upstream takes theirs (or moves the cache into
  the scheduler) ours should be replaced rather than rebased.
- **`gated_delta_net.cu`.** Our hook sits at the top of
  `launch_gated_delta_net`. If upstream ever enables a GCN path in
  `select_gdn_mma_path`, the hook either shadows it or becomes dead code.
- **MoE up/gate fusion.** It fires on the "two MUL_MAT_ID plus GLU" graph shape.
  If upstream switches to a single fused `ffn_gate_up_exps` matmul (they already
  load its scales), the fusion stops firing silently and the 1.3% is lost.

### After every upstream sync

Run these four; a deviation in any of them is the signal for one of the above.

| check | expected |
|---|---:|
| pp512 | 2457 |
| pp2048 ub 2048 | 2975 |
| tg128 | 79.9 |
| PPL c=512 | 3.5779 |

A cheap extra assert: if no tensor ends up in a repack buffer type after model
load, the buft selection was disturbed.

## Done

### Activation quantization cache (prefill and decode)

One activation feeds several repacked matmuls back to back (q/k/v off one
attention norm, the shared expert and the routed experts off one ffn norm), and
each one quantized its own copy. `ggml_cuda_q8_1_cache_acquire` keeps up to two
quantized activations for the duration of one graph evaluation, keyed on the
source tensor, its data pointer, the block layout variant, the padded row length,
the strides, the extents and the byte count. Same design as the fork's
backend-generic cache, `GGML_CUDA_Q8_1_CACHE=0` disables it.

Same binary, Qwen3.6-35B-A3B IQ4_NL, `-sm tensor`:

| test | cache on | off | delta |
|---|---:|---:|---:|
| pp512 ub512 | 2457 | 2418 | +1.6% |
| pp2048 ub512 | 2207 | 2173 | +1.6% |
| pp2048 ub2048 | 2975 | 2925 | +1.7% |
| tg128 | 79.9 | 78.6 | +1.7% |

PPL at c=512 is 3.5779 either way and the cli output is byte-identical at ub 256
(GEMM path) and ub 8 (mat-vec path).

Only the repack paths use it so far. The canonical mat-vec and MMQ paths still
quantize a fresh copy per matmul; the same hook applies at their call sites and
would restore the gain under `--repack 0` (see P0 item 2).

### Dense GEMM token tile: choose by matrix size and batch

The dense repack GEMM used a 128-wide token tile (TN=2) for every batch above
128 tokens. The tile trades parallelism for weight reuse, and the two models we
track sit on opposite sides of it:

| pp2048 | BN=32 | BN=128 |
|---|---:|---:|
| 9B dense, ub 512 / 1024 / 2048 | 973 / 986 / 975 | **1280 / 1344 / 1341** |
| MoE, ub 512 | **2168** | 2067 |
| MoE, ub 1024 | 2538 | **2552** |
| MoE, ub 2048 | 2806 | **2921** |

The 9B (M >= 4096) loses 24% on the narrow tile. The MoE model wins 5% at
ub 512, where its dense matmuls are M=2048 attention projections and an M=512
shared expert. A choice by block count alone does not separate them: the M=512
shared expert wants the wide tile at ub 2048 even though its grid is just as
thin. The batch does separate them: short batches are parallelism-bound, long
ones are weight-bound.

Rule: `ne11 < 128 || (ne11 <= 512 && row_tiles <= 32)` selects BN=32, everything
else BN=128. `GGML_RP_DENSE_BN` overrides it for A/B.

MoE model: pp512 2305 -> **2423** (+5.1%), pp2048 at ub 512 2067 -> 2165 (+4.7%),
ub 1024 and 2048 unchanged, tg128 unchanged, PPL c=512 3.5779 either way. The
9B moves within noise.

### Profile: MoE prefill, pp2048 ub 2048

`rocprofv3 --kernel-trace --stats` writes an sqlite db; aggregate with
`select name, count(*), sum(end-start) from kernels group by name`. The run
measured 2879 t/s against 2924 unprofiled.

Shares of GPU kernel time, both devices, warmup and measured pass included
(2984 ms total, 149 ms of it the one-time weight repack):

| block | ms | share | calls | ms/call |
|---|---:|---:|---:|---:|
| dense GEMM (repack, no ids) | 662 | 23.3% | 1228 | 0.54 |
| MoE up+gate (fused) | 403 | 14.2% | 156 | 2.58 |
| NCCL allreduce | 389 | 13.7% | 320 | 1.22 |
| MoE down | 222 | 7.8% | 156 | 1.42 |
| GDN colthread | 219 | 7.7% | 120 | 1.82 |
| concat_non_cont | 204 | 7.2% | 120 | 1.70 |
| binbcast | 148 | 5.2% | 1188 | 0.12 |
| quantize activation | 131 | 4.6% | 1540 | 0.09 |
| flash_attn tile | 102 | 3.6% | 120 | 0.85 |
| rocBLAS gemm | 90 | 3.2% | 156 | 0.58 |
| ids helper | 32 | 1.1% | 312 | 0.10 |

MoE GEMM is 403 + 222 = 625 ms, 22% of prefill. BM 64 -> 128 halves the
activation re-read inside it, which is the same 6%-of-the-GEMM effect the
up/gate fusion already bought (1.3% end to end), and it pays the same occupancy
step that ate most of that win: LDS goes 14 KB -> 22 KB and the block count per
CU 3 -> 2. **Not worth it.**

Since the GEMM is compute-bound rather than load-bound, the remaining prefill
work is elsewhere: the dense GEMM is the largest single block and still runs the
TN=2 / BN=128 tile, the NCCL allreduce is 13.7% at ub 2048 (8 MB per call), and
the GDN + concat + ssm group is 15%.

At decode the same comm is 161 calls per token at about 43 us each, which is
launch and sync latency rather than bandwidth. Both trees pay it (see the
tp-allreduce note under Do not repeat), so the lever is batching the per-layer
allreduces into one group, which neither has tried.

### Fuse up and gate in the MoE prefill GEMM

`mmq_gemm_repacked_narrow_gated` keeps two accumulators per row tile and streams
both weight matrices through one pass over K, so the activation slab and its
quantization are paid once instead of twice. The gate planes need four LDS
weight buffers, so this instantiation runs at occupancy 2 against 3 for the
plain tile.

Same binary, `GGML_CUDA_REPACK_NO_FUSE=1` restores the two independent nodes:

| test | fused | two nodes | delta |
|---|---:|---:|---:|
| pp512 ub512 | 2307 | 2261 | +2.1% |
| pp2048 ub512 | 2073 | 2049 | +1.2% |
| pp2048 ub1024 | 2556 | 2522 | +1.3% |
| pp2048 ub2048 | 2924 | 2883 | +1.4% |
| tg128 | 79.0 | 79.0 | 0 |

PPL at c=512 is 3.5779 both ways and the cli output at ub 256 is byte-identical.

The ceiling is the activation share of the GEMM, not the tile fill: at ub 2048
the activation is about a quarter of its traffic, so one saved read is worth a
couple of percent. The fused tile needs four LDS weight planes, which drops it
from 3 blocks per CU to 2 (97 VGPR, 22784 B LDS, against 77 and 14080 for the
plain tile). A BK=2 instantiation halves the planes and does reach 3 blocks
(84 VGPR, 11776 B) but loses everywhere: pp512 2147, pp2048 ub512 1952, ub2048
2858, because the slab count and its syncs double. Kept at BK=4.

### MoE prefill ub scaling: the BN=64 cliff (fixed)

The adaptive tile width from `d0695b3a1` was only ever exercised at ub 512,
where the average expert load is 16 rows and BN=32 is always selected. At
ub 2048 (64 rows per expert) it picked the narrow CW=32/TN=2 instantiation,
which is 2.5x slower than BN=32, so our prefill collapsed exactly where the
fork we compared against scaled up.

Qwen3.6-35B-A3B IQ4_NL, 2x MI50, `-sm tensor`, f16 KV:

| ub | before | after (BN=32) | llama.cpp-gfx906 |
|---:|---:|---:|---:|
| 512 | 2055 | 2054 | 2161 |
| 1024 | 2521 | 2522 | 2559 |
| 2048 | 1149 | **2885** | 2877 |
| 4096 | 1048 | 2747 | - |

BN=32 beat the wide 64 tile (2578 at ub 2048, 2480 at ub 4096) and the wide
128 tile (2522) at every size, so the load-based selection is gone; the MoE
GEMM now uses one fixed 32-wide tile. PPL at c=512 is unchanged (3.5779 both
before and after in the same build).

`-ub 1024`/`-ub 2048` are therefore free prefill gains (2054 -> 2522 -> 2885
at pp2048). The FA mask scales with n_kv x ub, so re-check VRAM at long
context before adopting them by default.

### Port the GDN colthread kernel (prefill)

`gated_delta_net_colthread.{cu,cuh}` copied from `llama.cpp-gfx906`, called from
`launch_gated_delta_net` before the fallback path. `GGML_GDN_NO_COLTHREAD=1`
restores the old path.

Same binary A/B, Qwen3.6-35B-A3B IQ4_NL, `-sm tensor`, r=5:

| test | colthread | old | delta |
|---|---:|---:|---:|
| pp512 | 2297.3 | 2196.8 | +4.6% |
| pp2048 | 2173.8 | 2042.4 | +6.4% |
| tg128 | 81.4 | 81.1 | +0.4% |

PPL (c=512, 4 chunks): 3.5773 vs 3.5819 (-0.13%). With `-ub 1` (decode path):
3.3769 vs 3.3758 (+0.03%). Prefill numerics use the colthread summation order,
so prefill output is not bit-identical to the old kernel.

## P0

### 1. Allreduce: take upstream's internal two-shot for small messages (blocked)

`GGML_CUDA_ALLREDUCE=internal` (upstream's own 2-way allreduce, `allreduce.cu`,
staged through pinned host memory) is much faster at decode sizes than the NCCL
ring on this PCIe pair, and slower at prefill sizes:

| test | NCCL | internal | delta |
|---|---:|---:|---:|
| tg128 | 80.5 | **97.7** | **+21%** |
| pp2048 ub512 | 2202 | 1944 | -12% |

Size sweep, `ne` is elements per allreduce (ub x 2048):

| ub | ne | NCCL | internal |
|---:|---:|---:|---:|
| 8 | 16K | 317 | 362 (+14%) |
| 16 | 32K | 320 | 339 (+6%) |
| 32 | 65K | 529 | 555 (+4.9%) |
| 64 | 131K | 871 | 889 (+2.0%) |
| 128 | 262K | 1347 | 1359 (+0.9%) |
| 256 | 524K | 1804 | 1715 (-5%) |
| 512 | 1M | 2202 | 1944 (-12%) |

So the two paths should be selected per message size. Tried it: `comm_init_nccl`
also brings up the internal pipeline and a dispatcher takes internal when
`ne <= 131072` (ub 64). Decode landed at 97.7 as predicted, but **every prefill
run hangs**: 100% CPU, both GPUs idle, 23 GB of VRAM held, no output, and it
survives a 150 s timeout.

What isolates it: the hang needs *both* paths initialized in one comm context.
`GGML_CUDA_ALLREDUCE=internal` alone runs prefill fine (1944), and the new code
with `GGML_CUDA_ALLREDUCE_MAX_NE=0` (dispatch forced to NCCL, pipeline still up)
still hangs. Next suspects: `ggml_cuda_ar_pipeline_init` changes the current
device and does not restore it, and it creates its own streams, events and pinned
staging; check the device restore before anything else. Reverted meanwhile.

Upstream is extending this path (PR #32195, 4-way internal allreduce, open).

### 2. Convert the canonical paths to the quantization cache

The repack paths use `ggml_cuda_q8_1_cache_acquire`; the canonical mat-vec and
MMQ paths still quantize a fresh copy per matmul. The same hook applies at their
call sites (`mmvq.cu`, `mmq.cu`), and it restores the gain for `--repack 0` and
for types the repack does not cover.

### 3. MoE prefill GEMM: fill the expert tiles

At pp512 there are 512*8/256 = 16 rows per expert, so a BN=32 tile is half
padding. Weight traffic per expert is `ceil(rows_e/BN) * M * K * bpw`, which is
why a global BN=16 lost: experts with 17-32 rows needed two token tiles and
read their weights twice.

Per-expert BN bucketing was the plan here, but the ub scaling measurement above
killed it: BN=32 is faster than the 64 and 128 tiles at every batch size, so a
wider tile is not worth chasing. What is left:

- Profile pp512 to see the real split now that ub scaling works. The pp2048
  ub 2048 profile above puts the whole MoE GEMM at 22% of prefill.
- BM 64 -> 128 halves the activation re-reads per expert-matrix (2 MB -> 1 MB,
  about 23% of its traffic) if the LDS budget still holds. Measured against the
  profile above: the GEMM is compute-bound, the activation side is worth about
  6% of it (the up/gate fusion's 1.3% end to end), and BM=128 costs the same
  3 -> 2 blocks per CU step. Not worth it.
- Larger `-ub` is now the cheap way to fill the tiles: pp2048 goes 2054 (ub
  512) -> 2522 (ub 1024) -> 2885 (ub 2048) with no code change.
- Above ~8192 tokens the 128-wide tile may take over: llama.cpp-gfx906 uses a
  two-way split, `use_w32 = n_assign < 2*128*n_expert` (VGPR spill drops
  occupancy 2 vs 3 for the wide tile). Our tile is fixed at 32, which matches
  their choice up to that threshold. Measure pp8192/pp16384 before adopting.

### 4. Fuse router matvec with topk (decode)

- `mul_mat_vec_f` (F32 router) 0.62 ms/token + `topk_moe` 0.47 ms/token.
- Consider a fused router + sigmoid/scale + topk kernel in `topk-moe.cu`.

## P1

### 5. Fuse MoE down with the weighted reduction

The combine path still runs binbcast mul/add and a separate activation pass. Fuse
the down GEMM epilogue with the per-expert weighted sum and residual. Note
`concat_non_cont` (7.2% of prefill, 1.7 ms per call, 120 calls) belongs to the
attention/GDN group, not to this combine path, so it is a separate item.

### 6. Validate single-GPU / non-TP MoE

Whole repacked gate/up now go through the repack fused branches, but the 20 GB
model does not fit a 16 GB card. Test with a sliced MoE model: text and PPL
against canonical.

### 7. `repack_expert_bounds` LDS limit

Dynamic LDS is `n_expert * 4` bytes; more than 16384 experts exceeds the 64 KB
budget. Add a fallback if a model ever needs it.

### 8. GCN `kv_rows` (optional, multi-sequence long context only)

`ggml_flash_attn_ext_rows` is enabled only for NVIDIA Turing+ MMA
(`fattn.cu:685`). A GCN gather kernel would help multi-slot long-context decode.
No benefit for a single stream.

### 9. IQ4_XS repack kernel

Extend the repack path to native `GGML_TYPE_IQ4_XS` (4.25 bpw). Used by Unsloth
UD dynamic mixes; those tensors currently stay canonical and miss the whole
repack path.

This is the only quant type still planned for the repack path. It reuses the
IQ4_NL nibble body and codebook, so no new kernel is needed; every other type
is out of scope, see "Do not repeat".

The repacked layout is identical to IQ4_NL: split each 256-weight super-block
into 8 sub-blocks of 32, copy the 16 nibbles verbatim, and store one f16
`dl = d * (ls - 32)` per sub-block in the scale plane. Row stride, the `+16`
de-alias bump and `ne1 * (ne0/2 + ne0/32*2)` bytes all match IQ4_NL.

No kernel changes are needed: add `rp_traits<GGML_TYPE_IQ4_XS>` with the IQ4_NL
body (raw nibbles + `rp_iq4nl_expand` + f16 scale). The HIP
`get_int_from_table_16` already uses `v_perm`, so reinstinct's `iq4xs_lut4` is
not needed.

6-bit scale decode (per sub-block `ib = sb & 7`):

```
ls = ((b->scales_l[ib/2] >> (4*(ib%2))) & 0xf) | (((b->scales_h >> (2*ib)) & 3) << 4);
dl = GGML_FP16_TO_FP32(b->d) * (ls - 32);
```

Touch points:

- `repack-common.cuh`: packed-nibble branch in `repack_qs_row_stride` and
  `repack_gcn_nbytes`; trait; `RP_FOREACH_TYPE`; host declaration.
- `repack-common.cu`: `repack_tensor_supported` (require `ne0 % 256 == 0`);
  `repack_iq4xs_host`; `repack_host` case.
- `repack-buffer.cu`: device `repack_iq4xs_kernel` + async-upload switch.
- `repack-mul-mat.cu` / `repack-mul-mat-id.cu`: type switch cases pointing at
  the same IQ4_NL template instantiations.
- `ggml-backend-meta.cpp`: add IQ4_XS to the TP repack whitelist.

Precision: canonical computes `dl` in f32; the repack stores it as f16 (small
rounding, unlike IQ4_NL where `d` is already f16). Validate PPL.

Validation:

- nmse unit test: `repack_iq4xs_host` and the device kernel against
  `dequantize_row_iq4_xs` on random blocks; extend `benches/ai-gfx906`.
- `llama-quantize Qwen3.5-9B-Q8_0.gguf /tmp/q4xs.gguf IQ4_XS`, then cli text,
  PPL c=512 and pp512/tg128 vs `--no-repack` and vs IQ4_NL.
- Optional sliced/small MoE for the MUL_MAT_ID path.

Expected: 5.5% fewer weight bytes than IQ4_NL, about +3-5% decode on the
affected tensors; prefill around IQ4_NL (same codebook LUT). Effort about
250-350 lines.

Reference: `~/Projects/reinstinct/MANUAL.md` (quant table) and
`kernels/matvec_iq4xs_repacked.cpp`, `kernels/mmq_gemm_iq4xs_repacked.cpp`.

### 10. Row-split up/gate fusion (from gfx906-quartz)

`iacopPBK/gfx906-quartz` fuses gate and up in one GEMM block by splitting the row
tile: 32 rows come from the gate matrix and 32 from the up matrix, each thread
keeps a single accumulator, and the epilogue combines them through LDS (it
reuses the activation tile's space, `sAcc[row][token]`, and reads rows r and
r+32).

That is the opposite trade from our `mmq_gemm_repacked_narrow_gated`, which keeps
two accumulators over the same 64 rows:

| | ours | theirs |
|---|---|---|
| weight LDS | 16 KB (four planes) | 8 KB (one 64-row tile) |
| accumulators | two sets | one set |
| VGPR | 97 | not published, same shape as the plain tile |
| blocks per CU | 2 | 2 (their token tile is 128) |
| cost | none | twice the M tiles, so twice the activation reads |

We lost occupancy 3 -> 2 to the second accumulator, and that ate most of the
fusion's traffic win. The row-split form does not pay it, and the item it does
pay (activation reads) is the cheap one here: the GEMM is compute bound and the
up/gate fusion's whole traffic win was 1.3% end to end. Expected +1-2% on
prefill, and it removes the 97 VGPR / occupancy 2 corner rather than adding one.

### 11. Group blocks for L2 reuse

The same GEMM launches blocks in a grouped order, so the blocks in flight share
both weight rows and activation tiles in L2 instead of walking the grid in
(row, token) order. It is a few lines in the launch (remap `blockIdx`), and it
applies to the dense GEMM and to both MoE GEMMs. Untried here, so no expected
number.

### 12. Keep activations int8 between GEMMs

quartz quantizes the epilogue of the up/gate GEMM straight into int8 plus scales
and hands that to the down projection, so the activation never returns to f32
and is never re-quantized. Our pipeline does a fresh f32 to q8_1 conversion per
matmul; the quantization cache only removes the repeats of one input.

This is the largest of the three and the most invasive, because it changes the
shapes in the graph rather than a kernel. It is also where part of the Q8_0
share of prefill (the dense repack GEMM, 23%) could move.

All three come from reading `quartz/prefill.h` (`pf_gemm`); its `repack.h` is the
same [N][K] int8 plus scales layout we already use, and the inner loop is the
same 8x `sdot4`, so none of its advantage is in the dot product.

## Do not repeat

- Porting llama.cpp-gfx906's custom TP allreduce (`tp-allreduce.cu`): on
  MI50/PCIe its broadcast kernel is gated off, because `broadcast_ok` is false
  without coherent peer writes, and the comm falls back to NCCL. The only path
  that runs here (one-shot) measures 67.2 t/s against 70.0 through NCCL. Their
  own decode profile is pure `ncclDevKernel_Generic_4`: 20,642 calls and 922 ms
  against our 20,640 and 897 ms, so both trees pay the same allreduce cost.
  The lever is fewer calls per token, not a different kernel.
- BK=2 for the gated MoE tile: it does restore 3 blocks per CU (84 VGPR,
  11776 B LDS against 97 and 22784 at BK=4) but doubles the number of K slabs
  and syncs, and loses 2-7%: pp512 2147 against 2305, pp2048 ub512 1952
  against 2073, ub2048 2858 against 2924.
- Repack for any quant type other than IQ4_XS (K-quants, MXFP4, NVFP4, IQ3_S):
  out of scope. IQ4_XS reuses the IQ4_NL nibble body and codebook, so it needs
  no new kernel. The others need their own dequant code and their decode is
  slower than IQ4_NL, so a repack would not pay for itself. The fork's K-quant
  prefill numbers do not change this: they are `--repack`-style gains on a
  layout we do not run.
- narrow MoE GEMM CW=32/TN=2 (the BN=64 instantiation): 2.5x slower than
  BN=32 at ub 2048. Do not reintroduce it; BN=32 wins at every ub measured.
- BN=16 MoE GEMM tile applied globally: 787 vs 466 us per call. It helps the
  experts with <= 16 rows but doubles the token tiles, and with them the weight
  traffic, for everyone else. Per-expert BN bucketing is the fix (P0 item 2),
  not a global BN=16.
- mx-llama `mmid-bounds.cuh` helper: functionality is already covered by
  `repack_expert_bounds` (helper 27.4 us + bounds 9.7 us vs their 43.3 us).
- Do not route whole repacked weights into the canonical fused kernels: keep the
  guard in `ggml_cuda_should_fuse_mul_mat_vec_q`; the repack fused branches
  handle MoE and dense now.
- fork vega20 MMQ table (`mmq-config-vega.cuh`): pp512 596 vs 887 on 9B IQ4_NL
  `--repack 0`. Its occ=1/I=128 loses to upstream GCN occ=2/I=64 + stream_k.
- WY-chunked GDN (upstream #26001) on gfx906: 0.30-0.56x of the recurrent
  kernel (gfx906-llama-cpp E137). No matrix units, the op is latency-bound.
  The mx-llama-style "chunk" kernel is perf-neutral (tiles the same recurrence).
- Native q8_0 KV tile FA (no F16 shadow): saves 450 MiB/card at 256k but costs
  4.1% pp2048, 19.8% pp16384, 40.5% pp65536. K/V are re-dequantized per query
  block, and q8_0's 34-byte blocks are only 2-byte aligned so vector loads are
  not possible. Reverted; the F16 convert path stays.
- Forcing the VEC kernel for prefill (native q8, no shadow) saves the same
  450 MiB/card but costs 29% at pp2048 and 82% at pp16384.

## VRAM notes at 256k context (q8_0 KV, -sm layer)

- HIP compute buffer is 468 MiB/card larger than with f16 KV: that is the F16
  copy of one layer's K+V that the TILE path stages (not allocator
  fragmentation). Vulkan FA reads q8 natively and does not need it.
- `pipeline_parallel` (auto for `-sm layer`) sets sched copies = 4, so the
  `kq_mask` [n_kv x n_ubatch] F16 is staged 4 times: 1 GiB/card at 256k.
  `-sm tensor` disables pipeline parallel (copies = 1) but pays RCCL buffers.
- Capping the HIP buffer chunk (1 GiB) does not reduce the compute buffer; the
  greedy gallocr just moves the holes (1641.6 -> 1693.6 MiB).

## External references (watch)

Branches of `thelittlefireman/llama.cpp`; none has an upstream PR with gfx906
numbers, so re-check at each upstream sync instead of porting blind.

### `feature_fattn_tile_q8_0_native` (empty as of 2026-11)

Branch tip is upstream `254b1773` (2026-10-02), no local commits. The name
matches the experiment below, so the numbers are the answer if it ever lands.

Native q8_0 K/V inside the TILE kernel: dequantize block_q8_0 groups into the
tile during load, skip the F16 shadow. Implemented, measured, reverted.

Setup: Qwen3.6-35B-A3B IQ4_NL, 2x MI50, `-sm tensor`, q8_0 K/V, same binary
with an env A/B:

| test | convert + TILE | native q8 tile |
|---|---:|---:|
| pp2048 | 2036.4 | 1953.7 (-4.1%) |
| pp16384 | 1827.2 | 1466.2 (-19.8%) |
| pp65536 | 1227.5 | 730.5 (-40.5%) |
| tg128 | 73.6 | 73.7 |
| 256k peak VRAM | 26,404 MiB | 25,504 MiB (-450/card) |

Why it loses: the tile kernel dequantizes K/V per query block (32 rows for
head_dim 256), so a 512-row prefill repeats the dequant 16 times; the cost
scales with n_kv while the convert path pays it once. block_q8_0 is 34 bytes
and qs is only 2-byte aligned, so the load cannot be vectorized. Same verdict
as gfx906-llama-cpp E184-E190 (16k -3%, 160k fill -15%), which keeps it as a
fallback. Fallback value is real though: at 64k the native tile still beats
VEC (730 vs 324 t/s). Reaching parity needs split-KV prefill + merge so each
K/V element is dequantized once, which is a new kernel, not a variant.

### `feature_change_to_warp_size64_master` (upstream PR #29534, Draft)

GCN q8_0 (and q4_0) FA VEC with wave64 geometry. Our tree still has
`#define WARP_SIZE 32` everywhere, so the VEC kernel uses half a wave on
gfx906. Author numbers are from MI100 (CDNA) and are noise level; no gfx906
data. Applies to decode only; prefill uses TILE. Ceiling is small: at 256k the
attention read floor is ~2.9 GB/token. Watch for gfx906 numbers or a merge.

### `gfx906-dp4-ilp`

Dual accumulators in `vecdotq.cuh` (MMVQ q8_1 dots), +154 lines, no PR. No
effect on us: the repack path has its own dot code, and `mmvq_dp4a_u4` already
splits into two chains. Only matters for K-quant or `--no-repack` decode.

### `feature_flashdecodingpp_gcn_gemv`

Dedicated gfx906 single-token GEMV in `mmvq.cu` (+166 lines): one output row
per wave, no cross-wave shared-memory reduction, gate/bias/GLU fusion, gated
on `cc == VEGA20 && ncols_dst == 1 && !ids`. No PR, no numbers, and no effect
on us for the same reason as above: the repack MMV already runs one row per
wave. Worth a look only for K-quant decode.

### `HerRei/gfx906-decode-kernels` (MIT, AI-assisted, test-backend-ops checked)

A 162-line patch to `mmvq.cu` plus a flash-decoding kernel, measured on a
Radeon VII at stock clocks.

The matvec covers Q4_0, Q4_1 and IQ4_NL in the canonical block layout, with
expert ids and the fused gate, and picks lanes-per-row against rows-per-block
per shape (16x8 for experts, 2x64 under 2048 rows, 8x16 for K under 2048, 4x32
otherwise), always in 128-thread workgroups. 49 shapes at 512+ rows: median
-14% (Q4_0), -18% (Q4_1), -15% (IQ4_NL) per matvec. Whole model: +10% on a
Q4_0 MoE, +3% on a 27B IQ4_NL, prompt processing flat, PPL 6.031 -> 6.034.

It patches `mmvq.cu`, which our IQ4_NL decode does not use, so it only applies
with `--repack 0`. What is worth taking is the geometry and three findings:

- a 64-thread workgroup grid floors at about 57 us once it reaches about 2048
  workgroups, whatever the matrix size; 128 threads does not. Our dense MMV
  runs 1024-thread workgroups and the MoE one 256, so we should not sit on that
  floor, but it is the first thing to check if a small decode kernel looks slow.
- IQ4_NL is compute bound on this card, not memory bound: a higher memory clock
  changed nothing, a higher core clock did. Q4_1 is the fastest 4-bit format
  here, Q4_0 next, IQ4_NL 3-8% behind on the same shape.
- Q8_0 did not get faster with their layout and is not covered.

Their flash decoding handles head size 128 with 3 query heads per KV head over a
q8_0 cache (32k: 407 -> 226 us, 64k: 787 -> 453 us, below 6144 keys it defers).
Our attention instantiates at head size 256, so that kernel does not fit; see
Martin1701 for the matching shape.

### `Martin1701/MI50-llama.cpp` (attention for our shape)

gfx906 attention kernel with head size 256 and 6 query heads per KV head sharing
the q8_0 key/value loads, on upstream's split and combine. That matches the
shape our model runs (`flash_attn_tile<256, 256, ...>`), so this is the reference
for long-context decode, not the head-128 kernel above.

### `iacopPBK/gfx906-quartz` (standalone engine, architecture only)

A from-scratch inference engine for two gfx906 cards, Q8_0 GGUF only (other
quantisations are refused), Qwen3.8-27B class models. Runs 2x MI50 at 135 W:
pp2048 777 vs llama.cpp 654 at 0 context, 631 vs 499 at 16k, PPL 5.16 vs 5.18.

The architecture is the part that matters, because it is the answer to what our
profile shows:

- weights halved across the cards, and after each layer the cards write their
  partial sums straight into each other's memory over PCIe. No NCCL, no per-op
  launch: our decode makes 161 allreduce calls per token.
- decode is one persistent kernel per card for the whole model, every block
  resident, synchronised with a grid barrier (hence the 60-CU requirement).
  That removes the per-op launches and the graph replays too.
- decode always runs 3-4 rows (its own MTP layer) or 8-12 rows (a drafter), so
  it never decodes a single token. This is why that small-batch region matters.

Two cautions: it needs 32 GB cards (59 GB total at 256k with the drafter) and it
refuses anything but Q8_0, so it cannot run our 35B mix on 16 GB cards. Take the
structure, not the code.

Our Q8_0 path is real and is not marginal: the repack covers Q8_0, the 35B mix
runs Q8_0 for every non-expert tensor (the dense repack GEMM is 23% of prefill
in the profile above), and the 9B dense Q8 measures pp512 1240 / tg128 58.5. An
int8 path without the IQ4_NL table lookup is therefore where part of that 23%
could move, which is the same conclusion quartz reaches from the other side.

## Environment notes

- `-c 8` perplexity OOMs during context allocation, with and without
  `--no-repack`; unrelated to the repack code.
- GPU clocks can sit idle after crashes; measure with `-n 128 -r 5`.
- The `-mllvm -amdgpu-sched-strategy=max-ilp` CMake flag was evaluated and
  reverted; it is not in the tree.
