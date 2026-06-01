#include "pooling.cuh"
#include <mutex>
#include <unordered_map>

// ============================================================================
// Multi-dtype kernel implementations for bf16, int8, int16, fp8
// All 16 kernel versions (v0-v15) for both maxpool and avgpool
// ============================================================================

// ============================================================================
// BF16 (nv_bfloat16) MaxPool kernels
// ============================================================================

// --- v0: template instantiation ---
void maxpool_v0(const nv_bfloat16* input, nv_bfloat16* output, const PoolParams& params, cudaStream_t stream) {
    NVTX_RANGE_PUSH_C("maxpool_v0_bf16", NVTX_COLOR_MAXPOOL);
    const int64_t total = params.OH * params.OW * params.C;
    const int threads = 256;
    const int blocks_x = static_cast<int>((total + threads - 1) / threads);
    dim3 grid(blocks_x, 1, static_cast<int>(params.N));
    maxpool_v0_kernel<nv_bfloat16><<<grid, threads, 0, stream>>>(input, output, params);
    CUDA_CHECK(cudaGetLastError());
    NVTX_RANGE_POP();
}

// --- v1: explicit kernel (follow half pattern, uses float for shared memory) ---
template <int TILE_OH = 8, int TILE_OW = 8>
__global__ void maxpool_v1_kernel_bf16(
    const nv_bfloat16* __restrict__ input,
    nv_bfloat16* __restrict__ output,
    const PoolParams params,
    int blocks_oh, int blocks_ow,
    int smem_h, int smem_w)
{
    extern __shared__ float sdata[];

    const int n = blockIdx.z;
    const int c = blockIdx.y;
    const int tile_idx = blockIdx.x;
    const int tile_oh = tile_idx / blocks_ow;
    const int tile_ow = tile_idx % blocks_ow;

    const int th = threadIdx.y;
    const int tw = threadIdx.x;

    const int oh = tile_oh * TILE_OH + th;
    const int ow = tile_ow * TILE_OW + tw;

    const int ih_start = tile_oh * TILE_OH * params.sh - params.ph;
    const int iw_start = tile_ow * TILE_OW * params.sw - params.pw;

    const int total_smem = smem_h * smem_w;
    const int tid = th * TILE_OW + tw;
    for (int i = tid; i < total_smem; i += TILE_OH * TILE_OW) {
        const int sih = i / smem_w;
        const int siw = i % smem_w;
        const int ih = ih_start + sih;
        const int iw = iw_start + siw;
        if (ih >= 0 && ih < params.H && iw >= 0 && iw < params.W) {
            const int64_t in_idx = ((static_cast<int64_t>(n) * params.H + ih) * params.W + iw) * params.C + c;
            sdata[i] = __bfloat162float(input[in_idx]);
        } else {
            sdata[i] = -INFINITY;
        }
    }

    __syncthreads();

    if (oh >= params.OH || ow >= params.OW) return;

    float maxval = -INFINITY;
    for (int kh_i = 0; kh_i < params.kh; ++kh_i) {
        const int sih = th * params.sh + kh_i * params.dh;
        if (sih >= smem_h) continue;
        for (int kw_i = 0; kw_i < params.kw; ++kw_i) {
            const int siw = tw * params.sw + kw_i * params.dw;
            if (siw >= smem_w) continue;
            const float val = sdata[sih * smem_w + siw];
            if (val > maxval) maxval = val;
        }
    }

    const int64_t out_idx = ((static_cast<int64_t>(n) * params.OH + oh) * params.OW + ow) * params.C + c;
    output[out_idx] = __float2bfloat16(maxval);
}

void maxpool_v1(const nv_bfloat16* input, nv_bfloat16* output, const PoolParams& params, cudaStream_t stream) {
    NVTX_RANGE_PUSH_C("maxpool_v1_bf16", NVTX_COLOR_MAXPOOL);
    constexpr int TILE_OH = 8;
    constexpr int TILE_OW = 8;

    const int blocks_oh = static_cast<int>((params.OH + TILE_OH - 1) / TILE_OH);
    const int blocks_ow = static_cast<int>((params.OW + TILE_OW - 1) / TILE_OW);

    int smem_h = (TILE_OH - 1) * params.sh + (params.kh - 1) * params.dh + 1;
    int smem_w = (TILE_OW - 1) * params.sw + (params.kw - 1) * params.dw + 1;
    smem_h = min(smem_h, static_cast<int>(params.H + 2 * params.ph));
    smem_w = min(smem_w, static_cast<int>(params.W + 2 * params.pw));

    dim3 block(TILE_OW, TILE_OH);
    dim3 grid(blocks_oh * blocks_ow, static_cast<int>(params.C), static_cast<int>(params.N));
    size_t smem_bytes = static_cast<size_t>(smem_h) * smem_w * sizeof(float);

    maxpool_v1_kernel_bf16<TILE_OH, TILE_OW><<<grid, block, smem_bytes, stream>>>(
        input, output, params, blocks_oh, blocks_ow, smem_h, smem_w);
    CUDA_CHECK(cudaGetLastError());
    NVTX_RANGE_POP();
}

// --- v2: vectorized loads with nv_bfloat162 ---
template <int VEC>
__global__ void maxpool_v2_kernel_bf16(
    const nv_bfloat16* __restrict__ input,
    nv_bfloat16* __restrict__ output,
    const PoolParams params)
{
    const int64_t n = blockIdx.z;
    const int64_t C_vec = params.C / VEC;
    const int64_t flat = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t total = params.OH * params.OW * C_vec;
    if (n >= params.N || flat >= total) return;

    const int64_t oh = flat / (params.OW * C_vec);
    const int64_t ow = (flat / C_vec) % params.OW;
    const int64_t c_vec = flat % C_vec;
    const int64_t c = c_vec * VEC;

    float maxval[VEC];
    #pragma unroll
    for (int v = 0; v < VEC; ++v)
        maxval[v] = -INFINITY;

    for (int kh_i = 0; kh_i < params.kh; ++kh_i) {
        const int64_t ih = oh * params.sh - params.ph + static_cast<int64_t>(kh_i) * params.dh;
        if (ih < 0 || ih >= params.H) continue;
        for (int kw_i = 0; kw_i < params.kw; ++kw_i) {
            const int64_t iw = ow * params.sw - params.pw + static_cast<int64_t>(kw_i) * params.dw;
            if (iw < 0 || iw >= params.W) continue;

            const int64_t in_idx = ((n * params.H + ih) * params.W + iw) * params.C + c;

            // nv_bfloat162 vectorized load
            nv_bfloat162 vec = *reinterpret_cast<const nv_bfloat162*>(&input[in_idx]);
            float lo = __low2float(vec);
            float hi = __high2float(vec);
            if (lo > maxval[0]) maxval[0] = lo;
            if (hi > maxval[1]) maxval[1] = hi;
        }
    }

    const int64_t out_idx = ((n * params.OH + oh) * params.OW + ow) * params.C + c;
    nv_bfloat162 out_vec = __floats2bfloat162_rn(maxval[0], maxval[1]);
    *reinterpret_cast<nv_bfloat162*>(&output[out_idx]) = out_vec;
}

void maxpool_v2(const nv_bfloat16* input, nv_bfloat16* output, const PoolParams& params, cudaStream_t stream) {
    NVTX_RANGE_PUSH_C("maxpool_v2_bf16", NVTX_COLOR_MAXPOOL);
    constexpr int VEC = 2;
    if (params.C % VEC != 0) {
        NVTX_RANGE_POP();
        maxpool_v0(input, output, params, stream);
        return;
    }
    const int64_t C_vec = params.C / VEC;
    const int64_t total = params.OH * params.OW * C_vec;
    const int threads = 256;
    const int blocks_x = static_cast<int>((total + threads - 1) / threads);
    dim3 grid(blocks_x, 1, static_cast<int>(params.N));
    maxpool_v2_kernel_bf16<VEC><<<grid, threads, 0, stream>>>(input, output, params);
    CUDA_CHECK(cudaGetLastError());
    NVTX_RANGE_POP();
}

// --- v3: register blocking (template instantiation) ---
void maxpool_v3(const nv_bfloat16* input, nv_bfloat16* output, const PoolParams& params, cudaStream_t stream) {
    NVTX_RANGE_PUSH_C("maxpool_v3_bf16", NVTX_COLOR_MAXPOOL);
    constexpr int BLOCK = 4;
    if (params.OH < BLOCK || params.OH % BLOCK != 0) {
        NVTX_RANGE_POP();
        maxpool_v0(input, output, params, stream);
        return;
    }
    const int64_t OH_block = params.OH / BLOCK;
    const int64_t total = OH_block * params.OW * params.C;
    const int threads = 256;
    const int blocks_x = static_cast<int>((total + threads - 1) / threads);
    dim3 grid(blocks_x, 1, static_cast<int>(params.N));
    maxpool_v3_kernel<nv_bfloat16, BLOCK><<<grid, threads, 0, stream>>>(input, output, params);
    CUDA_CHECK(cudaGetLastError());
    NVTX_RANGE_POP();
}

// --- v4: warp-level reduce (template instantiation) ---
void maxpool_v4(const nv_bfloat16* input, nv_bfloat16* output, const PoolParams& params, cudaStream_t stream) {
    NVTX_RANGE_PUSH_C("maxpool_v4_bf16", NVTX_COLOR_MAXPOOL);
    const int64_t total = params.OH * params.OW * params.C;
    const int threads = 256;
    const int warps_per_block = threads / 32;
    const int blocks_x = static_cast<int>((total + warps_per_block - 1) / warps_per_block);
    dim3 grid(blocks_x, 1, static_cast<int>(params.N));
    maxpool_v4_kernel<nv_bfloat16><<<grid, threads, 0, stream>>>(input, output, params);
    CUDA_CHECK(cudaGetLastError());
    NVTX_RANGE_POP();
}

// --- v5: double buffer (explicit kernel) ---
template <int TILE_OH = 8, int TILE_OW = 8>
__global__ void maxpool_v5_kernel_bf16(
    const nv_bfloat16* __restrict__ input,
    nv_bfloat16* __restrict__ output,
    const PoolParams params,
    int blocks_oh, int blocks_ow,
    int smem_h, int smem_w)
{
    extern __shared__ float sdata[];
    float* buf[2] = { sdata, sdata + smem_h * smem_w };

    const int n = blockIdx.z;
    const int c_pair = blockIdx.y;
    const int c0 = c_pair * 2;
    const int c1 = c0 + 1;
    const bool has_c1 = (c1 < params.C);

    const int tile_idx = blockIdx.x;
    const int tile_oh = tile_idx / blocks_ow;
    const int tile_ow = tile_idx % blocks_ow;

    const int th = threadIdx.y;
    const int tw = threadIdx.x;

    const int oh = tile_oh * TILE_OH + th;
    const int ow = tile_ow * TILE_OW + tw;

    const int ih_start = tile_oh * TILE_OH * params.sh - params.ph;
    const int iw_start = tile_ow * TILE_OW * params.sw - params.pw;

    const int total_smem = smem_h * smem_w;
    const int tid = th * TILE_OW + tw;
    const int nthreads = TILE_OH * TILE_OW;

    for (int i = tid; i < total_smem; i += nthreads) {
        const int sih = i / smem_w;
        const int siw = i % smem_w;
        const int ih = ih_start + sih;
        const int iw = iw_start + siw;
        if (ih >= 0 && ih < params.H && iw >= 0 && iw < params.W) {
            const int64_t in_idx = ((static_cast<int64_t>(n) * params.H + ih) * params.W + iw) * params.C + c0;
            buf[0][i] = __bfloat162float(input[in_idx]);
        } else {
            buf[0][i] = -INFINITY;
        }
    }
    __syncthreads();

    float maxval_c0 = -INFINITY;
    if (oh < params.OH && ow < params.OW) {
        for (int kh_i = 0; kh_i < params.kh; ++kh_i) {
            const int sih = th * params.sh + kh_i * params.dh;
            if (sih >= smem_h) continue;
            for (int kw_i = 0; kw_i < params.kw; ++kw_i) {
                const int siw = tw * params.sw + kw_i * params.dw;
                if (siw >= smem_w) continue;
                const float val = buf[0][sih * smem_w + siw];
                if (val > maxval_c0) maxval_c0 = val;
            }
        }
    }
    if (oh < params.OH && ow < params.OW) {
        const int64_t out_idx = ((static_cast<int64_t>(n) * params.OH + oh) * params.OW + ow) * params.C + c0;
        output[out_idx] = __float2bfloat16(maxval_c0);
    }

    if (has_c1) {
        for (int i = tid; i < total_smem; i += nthreads) {
            const int sih = i / smem_w;
            const int siw = i % smem_w;
            const int ih = ih_start + sih;
            const int iw = iw_start + siw;
            if (ih >= 0 && ih < params.H && iw >= 0 && iw < params.W) {
                const int64_t in_idx = ((static_cast<int64_t>(n) * params.H + ih) * params.W + iw) * params.C + c1;
                buf[1][i] = __bfloat162float(input[in_idx]);
            } else {
                buf[1][i] = -INFINITY;
            }
        }
        __syncthreads();

        float maxval_c1 = -INFINITY;
        if (oh < params.OH && ow < params.OW) {
            for (int kh_i = 0; kh_i < params.kh; ++kh_i) {
                const int sih = th * params.sh + kh_i * params.dh;
                if (sih >= smem_h) continue;
                for (int kw_i = 0; kw_i < params.kw; ++kw_i) {
                    const int siw = tw * params.sw + kw_i * params.dw;
                    if (siw >= smem_w) continue;
                    const float val = buf[1][sih * smem_w + siw];
                    if (val > maxval_c1) maxval_c1 = val;
                }
            }
            const int64_t out_idx = ((static_cast<int64_t>(n) * params.OH + oh) * params.OW + ow) * params.C + c1;
            output[out_idx] = __float2bfloat16(maxval_c1);
        }
    }
}

void maxpool_v5(const nv_bfloat16* input, nv_bfloat16* output, const PoolParams& params, cudaStream_t stream) {
    NVTX_RANGE_PUSH_C("maxpool_v5_bf16", NVTX_COLOR_MAXPOOL);
    constexpr int TILE_OH = 8;
    constexpr int TILE_OW = 8;

    if (params.C < 2) {
        NVTX_RANGE_POP();
        maxpool_v1(input, output, params, stream);
        return;
    }

    const int blocks_oh = static_cast<int>((params.OH + TILE_OH - 1) / TILE_OH);
    const int blocks_ow = static_cast<int>((params.OW + TILE_OW - 1) / TILE_OW);

    int smem_h = (TILE_OH - 1) * params.sh + (params.kh - 1) * params.dh + 1;
    int smem_w = (TILE_OW - 1) * params.sw + (params.kw - 1) * params.dw + 1;
    smem_h = min(smem_h, static_cast<int>(params.H + 2 * params.ph));
    smem_w = min(smem_w, static_cast<int>(params.W + 2 * params.pw));

    size_t smem_bytes = 2 * static_cast<size_t>(smem_h) * smem_w * sizeof(float);

    int smem_limit = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&smem_limit, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0));
    if (smem_bytes > static_cast<size_t>(smem_limit)) {
        NVTX_RANGE_POP();
        maxpool_v1(input, output, params, stream);
        return;
    }

    const int c_pairs = static_cast<int>((params.C + 1) / 2);
    dim3 block(TILE_OW, TILE_OH);
    dim3 grid(blocks_oh * blocks_ow, c_pairs, static_cast<int>(params.N));

    if (smem_bytes > 49152) {
        CUDA_CHECK(cudaFuncSetAttribute(
            maxpool_v5_kernel_bf16<TILE_OH, TILE_OW>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(smem_bytes)));
    }

    maxpool_v5_kernel_bf16<TILE_OH, TILE_OW><<<grid, block, smem_bytes, stream>>>(
        input, output, params, blocks_oh, blocks_ow, smem_h, smem_w);
    CUDA_CHECK(cudaGetLastError());
    NVTX_RANGE_POP();
}

// --- v6: warp specialization (explicit kernel) ---
template <int TILE_OH = 8, int TILE_OW = 8, int NUM_LOAD_WARPS = 2, int NUM_COMPUTE_WARPS = 6>
__global__ void maxpool_v6_kernel_bf16(
    const nv_bfloat16* __restrict__ input,
    nv_bfloat16* __restrict__ output,
    const PoolParams params,
    int blocks_oh, int blocks_ow,
    int smem_h, int smem_w)
{
    constexpr int NUM_LOAD_THREADS = NUM_LOAD_WARPS * 32;
    constexpr int NUM_COMPUTE_THREADS = NUM_COMPUTE_WARPS * 32;
    constexpr int TILE_SIZE = TILE_OH * TILE_OW;

    extern __shared__ float sdata[];

    const int n = blockIdx.z;
    const int c = blockIdx.y;
    const int tile_idx = blockIdx.x;
    const int tile_oh = tile_idx / blocks_ow;
    const int tile_ow = tile_idx % blocks_ow;

    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const bool is_load_warp = warp_id < NUM_LOAD_WARPS;

    const int ih_start = tile_oh * TILE_OH * params.sh - params.ph;
    const int iw_start = tile_ow * TILE_OW * params.sw - params.pw;

    const int total_smem = smem_h * smem_w;

    if (is_load_warp) {
        for (int i = tid; i < total_smem; i += NUM_LOAD_THREADS) {
            const int sih = i / smem_w;
            const int siw = i % smem_w;
            const int ih = ih_start + sih;
            const int iw = iw_start + siw;
            if (ih >= 0 && ih < params.H && iw >= 0 && iw < params.W) {
                const int64_t in_idx = ((static_cast<int64_t>(n) * params.H + ih) * params.W + iw) * params.C + c;
                sdata[i] = __bfloat162float(input[in_idx]);
            } else {
                sdata[i] = -INFINITY;
            }
        }
    }
    __syncthreads();

    if (!is_load_warp) {
        const int compute_tid = tid - NUM_LOAD_THREADS;
        for (int i = compute_tid; i < TILE_SIZE; i += NUM_COMPUTE_THREADS) {
            const int th = i / TILE_OW;
            const int tw = i % TILE_OW;
            const int oh = tile_oh * TILE_OH + th;
            const int ow = tile_ow * TILE_OW + tw;

            if (oh >= params.OH || ow >= params.OW) continue;

            float maxval = -INFINITY;
            for (int kh_i = 0; kh_i < params.kh; ++kh_i) {
                const int sih = th * params.sh + kh_i * params.dh;
                if (sih >= smem_h) continue;
                for (int kw_i = 0; kw_i < params.kw; ++kw_i) {
                    const int siw = tw * params.sw + kw_i * params.dw;
                    if (siw >= smem_w) continue;
                    const float val = sdata[sih * smem_w + siw];
                    if (val > maxval) maxval = val;
                }
            }

            const int64_t out_idx = ((static_cast<int64_t>(n) * params.OH + oh) * params.OW + ow) * params.C + c;
            output[out_idx] = __float2bfloat16(maxval);
        }
    }
}

void maxpool_v6(const nv_bfloat16* input, nv_bfloat16* output, const PoolParams& params, cudaStream_t stream) {
    NVTX_RANGE_PUSH_C("maxpool_v6_bf16", NVTX_COLOR_MAXPOOL);
    constexpr int TILE_OH = 8;
    constexpr int TILE_OW = 8;
    constexpr int BLOCK_SIZE = 256;

    const int blocks_oh = static_cast<int>((params.OH + TILE_OH - 1) / TILE_OH);
    const int blocks_ow = static_cast<int>((params.OW + TILE_OW - 1) / TILE_OW);

    int smem_h = (TILE_OH - 1) * params.sh + (params.kh - 1) * params.dh + 1;
    int smem_w = (TILE_OW - 1) * params.sw + (params.kw - 1) * params.dw + 1;
    smem_h = min(smem_h, static_cast<int>(params.H + 2 * params.ph));
    smem_w = min(smem_w, static_cast<int>(params.W + 2 * params.pw));

    size_t smem_bytes = static_cast<size_t>(smem_h) * smem_w * sizeof(float);

    int smem_limit = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&smem_limit, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0));
    if (smem_bytes > static_cast<size_t>(smem_limit)) {
        NVTX_RANGE_POP();
        maxpool_v1(input, output, params, stream);
        return;
    }

    dim3 block(BLOCK_SIZE);
    dim3 grid(blocks_oh * blocks_ow, static_cast<int>(params.C), static_cast<int>(params.N));

    if (smem_bytes > 49152) {
        CUDA_CHECK(cudaFuncSetAttribute(
            maxpool_v6_kernel_bf16<TILE_OH, TILE_OW>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(smem_bytes)));
    }

    maxpool_v6_kernel_bf16<TILE_OH, TILE_OW><<<grid, block, smem_bytes, stream>>>(
        input, output, params, blocks_oh, blocks_ow, smem_h, smem_w);
    CUDA_CHECK(cudaGetLastError());
    NVTX_RANGE_POP();
}

// --- v7: alternative mappings ---
template <>
__global__ void maxpool_v7_mappingD_kernel<nv_bfloat16>(
    const nv_bfloat16* __restrict__ input,
    nv_bfloat16* __restrict__ output,
    const PoolParams params,
    int C_groups)
{
    const int n = blockIdx.z / C_groups;
    const int c_group = blockIdx.z - n * C_groups;
    const int c_block_base = c_group * 32;
    const int warp_id = threadIdx.y;
    const int lane = threadIdx.x;

    const int oh_tile_base = blockIdx.y * 4;
    const int ow_tile_base = blockIdx.x * 4;

    const int c_warp_base = c_block_base + warp_id * 4;
    if (n >= params.N || c_warp_base + 3 >= params.C) return;
    if (lane >= 16) return;

    const int sh = lane / 4;
    const int sw = lane % 4;
    const int oh = oh_tile_base + sh;
    const int ow = ow_tile_base + sw;
    if (oh >= params.OH || ow >= params.OW) return;

    float maxval[4];
    #pragma unroll
    for (int v = 0; v < 4; ++v)
        maxval[v] = -INFINITY;

    for (int kh_i = 0; kh_i < params.kh; ++kh_i) {
        const int64_t ih = oh * params.sh - params.ph + static_cast<int64_t>(kh_i) * params.dh;
        if (ih < 0 || ih >= params.H) continue;
        for (int kw_i = 0; kw_i < params.kw; ++kw_i) {
            const int64_t iw = ow * params.sw - params.pw + static_cast<int64_t>(kw_i) * params.dw;
            if (iw < 0 || iw >= params.W) continue;
            const int64_t in_idx = ((static_cast<int64_t>(n) * params.H + ih) * params.W + iw) * params.C + c_warp_base;

            // bf16: two nv_bfloat162 loads for 4 channels
            nv_bfloat162 vec0 = *reinterpret_cast<const nv_bfloat162*>(&input[in_idx]);
            nv_bfloat162 vec1 = *reinterpret_cast<const nv_bfloat162*>(&input[in_idx + 2]);
            float lo0 = __low2float(vec0), hi0 = __high2float(vec0);
            float lo1 = __low2float(vec1), hi1 = __high2float(vec1);
            if (lo0 > maxval[0]) maxval[0] = lo0;
            if (hi0 > maxval[1]) maxval[1] = hi0;
            if (lo1 > maxval[2]) maxval[2] = lo1;
            if (hi1 > maxval[3]) maxval[3] = hi1;
        }
    }

    const int64_t out_idx = ((static_cast<int64_t>(n) * params.OH + oh) * params.OW + ow) * params.C + c_warp_base;
    nv_bfloat162 out0 = __floats2bfloat162_rn(maxval[0], maxval[1]);
    nv_bfloat162 out1 = __floats2bfloat162_rn(maxval[2], maxval[3]);
    *reinterpret_cast<nv_bfloat162*>(&output[out_idx]) = out0;
    *reinterpret_cast<nv_bfloat162*>(&output[out_idx + 2]) = out1;
}

void maxpool_v7(const nv_bfloat16* input, nv_bfloat16* output, const PoolParams& params, int mapping, cudaStream_t stream) {
    NVTX_RANGE_PUSH_C("maxpool_v7_bf16", NVTX_COLOR_MAXPOOL);
    switch (mapping) {
        case 0: {
            NVTX_RANGE_POP();
            maxpool_v0(input, output, params, stream);
            return;
        }
        case 1: {
            const int C_groups = static_cast<int>((params.C + 3) / 4);
            const int64_t grid_z_64 = params.N * C_groups;
            if (grid_z_64 > 65535) { NVTX_RANGE_POP(); maxpool_v0(input, output, params, stream); return; }
            dim3 block(8, 8, 4);
            dim3 grid(static_cast<int>((params.OW + 7) / 8), static_cast<int>((params.OH + 7) / 8), static_cast<int>(grid_z_64));
            maxpool_v7_mappingB_kernel<nv_bfloat16><<<grid, block, 0, stream>>>(input, output, params, C_groups);
            CUDA_CHECK(cudaGetLastError());
            break;
        }
        case 2: {
            const int C_groups = static_cast<int>((params.C + 255) / 256);
            const int64_t grid_z_64 = params.N * C_groups;
            if (params.OW > 65535 || params.OH > 65535 || grid_z_64 > 65535) { NVTX_RANGE_POP(); maxpool_v0(input, output, params, stream); return; }
            dim3 block(256, 1, 1);
            dim3 grid(static_cast<int>(params.OW), static_cast<int>(params.OH), static_cast<int>(grid_z_64));
            maxpool_v7_mappingC_kernel<nv_bfloat16><<<grid, block, 0, stream>>>(input, output, params, C_groups);
            CUDA_CHECK(cudaGetLastError());
            break;
        }
        case 3: {
            if (params.C % 4 != 0) { NVTX_RANGE_POP(); maxpool_v0(input, output, params, stream); return; }
            const int C_groups = static_cast<int>((params.C + 31) / 32);
            const int64_t grid_z_64 = params.N * C_groups;
            if (grid_z_64 > 65535) { NVTX_RANGE_POP(); maxpool_v0(input, output, params, stream); return; }
            dim3 block(32, 8, 1);
            dim3 grid(static_cast<int>((params.OW + 3) / 4), static_cast<int>((params.OH + 3) / 4), static_cast<int>(grid_z_64));
            maxpool_v7_mappingD_kernel<nv_bfloat16><<<grid, block, 0, stream>>>(input, output, params, C_groups);
            CUDA_CHECK(cudaGetLastError());
            break;
        }
        default:
            NVTX_RANGE_POP();
            throw std::invalid_argument("unsupported mapping: " + std::to_string(mapping));
    }
    NVTX_RANGE_POP();
}

// --- v8: auto-tuned tiling (fallback to v2) ---
void maxpool_v8(const nv_bfloat16* input, nv_bfloat16* output, const PoolParams& params, cudaStream_t stream) {
    NVTX_RANGE_PUSH_C("maxpool_v8_bf16", NVTX_COLOR_MAXPOOL);
    maxpool_v2(input, output, params, stream);
    NVTX_RANGE_POP();
}

// --- v9: TMA warp-specialized pipeline (fallback to v2) ---
void maxpool_v9(const nv_bfloat16* input, nv_bfloat16* output, const PoolParams& params, cudaStream_t stream) {
    NVTX_RANGE_PUSH_C("maxpool_v9_bf16", NVTX_COLOR_MAXPOOL);
    maxpool_v2(input, output, params, stream);
    NVTX_RANGE_POP();
}

// --- v10: persistent kernel (fallback to v2) ---
void maxpool_v10(const nv_bfloat16* input, nv_bfloat16* output, const PoolParams& params, cudaStream_t stream) {
    NVTX_RANGE_PUSH_C("maxpool_v10_bf16", NVTX_COLOR_MAXPOOL);
    maxpool_v2(input, output, params, stream);
    NVTX_RANGE_POP();
}

// --- v11: warp-shuffle reduction (fallback to v4) ---
void maxpool_v11(const nv_bfloat16* input, nv_bfloat16* output, const PoolParams& params, cudaStream_t stream) {
    NVTX_RANGE_PUSH_C("maxpool_v11_bf16", NVTX_COLOR_MAXPOOL);
    maxpool_v4(input, output, params, stream);
    NVTX_RANGE_POP();
}

// --- v12: L2-aware persistent (fallback to v2) ---
void maxpool_v12(const nv_bfloat16* input, nv_bfloat16* output, const PoolParams& params, cudaStream_t stream) {
    NVTX_RANGE_PUSH_C("maxpool_v12_bf16", NVTX_COLOR_MAXPOOL);
    maxpool_v2(input, output, params, stream);
    NVTX_RANGE_POP();
}

// --- v13: channel-vectorized warp (fallback to v2) ---
void maxpool_v13(const nv_bfloat16* input, nv_bfloat16* output, const PoolParams& params, cudaStream_t stream) {
    NVTX_RANGE_PUSH_C("maxpool_v13_bf16", NVTX_COLOR_MAXPOOL);
    maxpool_v2(input, output, params, stream);
    NVTX_RANGE_POP();
}

// --- v14: adaptive dispatcher ---
void maxpool_v14(const nv_bfloat16* input, nv_bfloat16* output, const PoolParams& params, cudaStream_t stream) {
    NVTX_RANGE_PUSH_C("maxpool_v14_bf16", NVTX_COLOR_MAXPOOL);

    bool is_global = (params.kh >= params.H && params.kw >= params.W);
    if (is_global) {
        maxpool_v13(input, output, params, stream);
        NVTX_RANGE_POP();
        return;
    }

    int64_t total_output_elems = params.N * params.OH * params.OW * params.C;
    if (total_output_elems < 65536) {
        maxpool_v10(input, output, params, stream);
        NVTX_RANGE_POP();
        return;
    }

    if (params.sh == 1 && params.sw == 1 && params.kh == 3 && params.kw == 3) {
        maxpool_v15(input, output, params, stream);
        NVTX_RANGE_POP();
        return;
    }

    maxpool_v2(input, output, params, stream);
    NVTX_RANGE_POP();
}

// --- v15: swizzled shared memory ---
void maxpool_v15(const nv_bfloat16* input, nv_bfloat16* output, const PoolParams& params, cudaStream_t stream) {
    NVTX_RANGE_PUSH_C("maxpool_v15_bf16", NVTX_COLOR_MAXPOOL);
    constexpr int TILE_OH = 8;
    constexpr int TILE_OW = 8;

    const int blocks_oh = static_cast<int>((params.OH + TILE_OH - 1) / TILE_OH);
    const int blocks_ow = static_cast<int>((params.OW + TILE_OW - 1) / TILE_OW);

    int smem_h = (TILE_OH - 1) * params.sh + (params.kh - 1) * params.dh + 1;
    int smem_w = (TILE_OW - 1) * params.sw + (params.kw - 1) * params.dw + 1;
    size_t smem_bytes = static_cast<size_t>(smem_h) * (smem_w + 1) * sizeof(nv_bfloat16);

    dim3 block(TILE_OW, TILE_OH);
    dim3 grid(blocks_oh * blocks_ow, static_cast<int>(params.C), static_cast<int>(params.N));

    if (smem_bytes > 49152) {
        CUDA_CHECK(cudaFuncSetAttribute(maxpool_v15_kernel<nv_bfloat16, true>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem_bytes)));
    }

    maxpool_v15_kernel<nv_bfloat16, true><<<grid, block, smem_bytes, stream>>>(
        input, output, params, blocks_oh, blocks_ow, smem_h, smem_w, params.divisor_override);
    CUDA_CHECK(cudaGetLastError());
    NVTX_RANGE_POP();
}

// ============================================================================
// INT8 MaxPool kernels
// ============================================================================

// --- v0: template instantiation ---
void maxpool_v0(const int8_t* input, int8_t* output, const PoolParams& params, cudaStream_t stream) {
    NVTX_RANGE_PUSH_C("maxpool_v0_i8", NVTX_COLOR_MAXPOOL);
    const int64_t total = params.OH * params.OW * params.C;
    const int threads = 256;
    const int blocks_x = static_cast<int>((total + threads - 1) / threads);
    dim3 grid(blocks_x, 1, static_cast<int>(params.N));
    maxpool_v0_kernel<int8_t><<<grid, threads, 0, stream>>>(input, output, params);
    CUDA_CHECK(cudaGetLastError());
    NVTX_RANGE_POP();
}

// --- v1: explicit kernel (float shared memory, scalar loads) ---
template <int TILE_OH = 8, int TILE_OW = 8>
__global__ void maxpool_v1_kernel_i8(
    const int8_t* __restrict__ input,
    int8_t* __restrict__ output,
    const PoolParams params,
    int blocks_oh, int blocks_ow,
    int smem_h, int smem_w)
{
    extern __shared__ float sdata[];
    const int n = blockIdx.z;
    const int c = blockIdx.y;
    const int tile_idx = blockIdx.x;
    const int tile_oh = tile_idx / blocks_ow;
    const int tile_ow = tile_idx % blocks_ow;
    const int th = threadIdx.y;
    const int tw = threadIdx.x;
    const int oh = tile_oh * TILE_OH + th;
    const int ow = tile_ow * TILE_OW + tw;
    const int ih_start = tile_oh * TILE_OH * params.sh - params.ph;
    const int iw_start = tile_ow * TILE_OW * params.sw - params.pw;
    const int total_smem = smem_h * smem_w;
    const int tid = th * TILE_OW + tw;
    for (int i = tid; i < total_smem; i += TILE_OH * TILE_OW) {
        const int sih = i / smem_w;
        const int siw = i % smem_w;
        const int ih = ih_start + sih;
        const int iw = iw_start + siw;
        if (ih >= 0 && ih < params.H && iw >= 0 && iw < params.W) {
            const int64_t in_idx = ((static_cast<int64_t>(n) * params.H + ih) * params.W + iw) * params.C + c;
            sdata[i] = static_cast<float>(input[in_idx]);
        } else {
            sdata[i] = -INFINITY;
        }
    }
    __syncthreads();
    if (oh >= params.OH || ow >= params.OW) return;
    float maxval = -INFINITY;
    for (int kh_i = 0; kh_i < params.kh; ++kh_i) {
        const int sih = th * params.sh + kh_i * params.dh;
        if (sih >= smem_h) continue;
        for (int kw_i = 0; kw_i < params.kw; ++kw_i) {
            const int siw = tw * params.sw + kw_i * params.dw;
            if (siw >= smem_w) continue;
            const float val = sdata[sih * smem_w + siw];
            if (val > maxval) maxval = val;
        }
    }
    const int64_t out_idx = ((static_cast<int64_t>(n) * params.OH + oh) * params.OW + ow) * params.C + c;
    output[out_idx] = static_cast<int8_t>(maxval);
}

void maxpool_v1(const int8_t* input, int8_t* output, const PoolParams& params, cudaStream_t stream) {
    NVTX_RANGE_PUSH_C("maxpool_v1_i8", NVTX_COLOR_MAXPOOL);
    constexpr int TILE_OH = 8;
    constexpr int TILE_OW = 8;
    const int blocks_oh = static_cast<int>((params.OH + TILE_OH - 1) / TILE_OH);
    const int blocks_ow = static_cast<int>((params.OW + TILE_OW - 1) / TILE_OW);
    int smem_h = (TILE_OH - 1) * params.sh + (params.kh - 1) * params.dh + 1;
    int smem_w = (TILE_OW - 1) * params.sw + (params.kw - 1) * params.dw + 1;
    smem_h = min(smem_h, static_cast<int>(params.H + 2 * params.ph));
    smem_w = min(smem_w, static_cast<int>(params.W + 2 * params.pw));
    dim3 block(TILE_OW, TILE_OH);
    dim3 grid(blocks_oh * blocks_ow, static_cast<int>(params.C), static_cast<int>(params.N));
    size_t smem_bytes = static_cast<size_t>(smem_h) * smem_w * sizeof(float);
    maxpool_v1_kernel_i8<TILE_OH, TILE_OW><<<grid, block, smem_bytes, stream>>>(
        input, output, params, blocks_oh, blocks_ow, smem_h, smem_w);
    CUDA_CHECK(cudaGetLastError());
    NVTX_RANGE_POP();
}

// --- v2: int4 vectorized loads (16 channels per load = 128-bit) ---
template <int VEC>
__global__ void maxpool_v2_kernel_i8(
    const int8_t* __restrict__ input,
    int8_t* __restrict__ output,
    const PoolParams params)
{
    const int64_t n = blockIdx.z;
    const int64_t C_vec = params.C / VEC;
    const int64_t flat = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t total = params.OH * params.OW * C_vec;
    if (n >= params.N || flat >= total) return;
    const int64_t oh = flat / (params.OW * C_vec);
    const int64_t ow = (flat / C_vec) % params.OW;
    const int64_t c_vec = flat % C_vec;
    const int64_t c = c_vec * VEC;
    float maxval[VEC];
    #pragma unroll
    for (int v = 0; v < VEC; ++v) maxval[v] = -INFINITY;
    for (int kh_i = 0; kh_i < params.kh; ++kh_i) {
        const int64_t ih = oh * params.sh - params.ph + static_cast<int64_t>(kh_i) * params.dh;
        if (ih < 0 || ih >= params.H) continue;
        for (int kw_i = 0; kw_i < params.kw; ++kw_i) {
            const int64_t iw = ow * params.sw - params.pw + static_cast<int64_t>(kw_i) * params.dw;
            if (iw < 0 || iw >= params.W) continue;
            const int64_t in_idx = ((n * params.H + ih) * params.W + iw) * params.C + c;
            int4 vec = *reinterpret_cast<const int4*>(&input[in_idx]);
            #pragma unroll
            for (int v = 0; v < VEC; ++v) {
                float val = static_cast<float>(reinterpret_cast<const int8_t*>(&vec)[v]);
                if (val > maxval[v]) maxval[v] = val;
            }
        }
    }
    const int64_t out_idx = ((n * params.OH + oh) * params.OW + ow) * params.C + c;
    int8_t* out_ptr = &output[out_idx];
    #pragma unroll
    for (int v = 0; v < VEC; ++v) out_ptr[v] = static_cast<int8_t>(maxval[v]);
}

void maxpool_v2(const int8_t* input, int8_t* output, const PoolParams& params, cudaStream_t stream) {
    NVTX_RANGE_PUSH_C("maxpool_v2_i8", NVTX_COLOR_MAXPOOL);
    constexpr int VEC = 16;
    if (params.C % VEC != 0) {
        NVTX_RANGE_POP();
        maxpool_v0(input, output, params, stream);
        return;
    }
    const int64_t C_vec = params.C / VEC;
    const int64_t total = params.OH * params.OW * C_vec;
    const int threads = 256;
    const int blocks_x = static_cast<int>((total + threads - 1) / threads);
    dim3 grid(blocks_x, 1, static_cast<int>(params.N));
    maxpool_v2_kernel_i8<VEC><<<grid, threads, 0, stream>>>(input, output, params);
    CUDA_CHECK(cudaGetLastError());
    NVTX_RANGE_POP();
}

// --- v3: register blocking ---
void maxpool_v3(const int8_t* input, int8_t* output, const PoolParams& params, cudaStream_t stream) {
    NVTX_RANGE_PUSH_C("maxpool_v3_i8", NVTX_COLOR_MAXPOOL);
    constexpr int BLOCK = 4;
    if (params.OH < BLOCK || params.OH % BLOCK != 0) {
        NVTX_RANGE_POP();
        maxpool_v0(input, output, params, stream);
        return;
    }
    const int64_t OH_block = params.OH / BLOCK;
    const int64_t total = OH_block * params.OW * params.C;
    const int threads = 256;
    const int blocks_x = static_cast<int>((total + threads - 1) / threads);
    dim3 grid(blocks_x, 1, static_cast<int>(params.N));
    maxpool_v3_kernel<int8_t, BLOCK><<<grid, threads, 0, stream>>>(input, output, params);
    CUDA_CHECK(cudaGetLastError());
    NVTX_RANGE_POP();
}

// --- v4: warp-level reduce ---
void maxpool_v4(const int8_t* input, int8_t* output, const PoolParams& params, cudaStream_t stream) {
    NVTX_RANGE_PUSH_C("maxpool_v4_i8", NVTX_COLOR_MAXPOOL);
    const int64_t total = params.OH * params.OW * params.C;
    const int threads = 256;
    const int warps_per_block = threads / 32;
    const int blocks_x = static_cast<int>((total + warps_per_block - 1) / warps_per_block);
    dim3 grid(blocks_x, 1, static_cast<int>(params.N));
    maxpool_v4_kernel<int8_t><<<grid, threads, 0, stream>>>(input, output, params);
    CUDA_CHECK(cudaGetLastError());
    NVTX_RANGE_POP();
}

// --- v5: double buffer ---
template <int TILE_OH = 8, int TILE_OW = 8>
__global__ void maxpool_v5_kernel_i8(
    const int8_t* __restrict__ input,
    int8_t* __restrict__ output,
    const PoolParams params,
    int blocks_oh, int blocks_ow,
    int smem_h, int smem_w)
{
    extern __shared__ float sdata[];
    float* buf[2] = { sdata, sdata + smem_h * smem_w };
    const int n = blockIdx.z;
    const int c_pair = blockIdx.y;
    const int c0 = c_pair * 2;
    const int c1 = c0 + 1;
    const bool has_c1 = (c1 < params.C);
    const int tile_idx = blockIdx.x;
    const int tile_oh = tile_idx / blocks_ow;
    const int tile_ow = tile_idx % blocks_ow;
    const int th = threadIdx.y;
    const int tw = threadIdx.x;
    const int oh = tile_oh * TILE_OH + th;
    const int ow = tile_ow * TILE_OW + tw;
    const int ih_start = tile_oh * TILE_OH * params.sh - params.ph;
    const int iw_start = tile_ow * TILE_OW * params.sw - params.pw;
    const int total_smem = smem_h * smem_w;
    const int tid = th * TILE_OW + tw;
    const int nthreads = TILE_OH * TILE_OW;
    for (int i = tid; i < total_smem; i += nthreads) {
        const int sih = i / smem_w;
        const int siw = i % smem_w;
        const int ih = ih_start + sih;
        const int iw = iw_start + siw;
        if (ih >= 0 && ih < params.H && iw >= 0 && iw < params.W) {
            const int64_t in_idx = ((static_cast<int64_t>(n) * params.H + ih) * params.W + iw) * params.C + c0;
            buf[0][i] = static_cast<float>(input[in_idx]);
        } else {
            buf[0][i] = -INFINITY;
        }
    }
    __syncthreads();
    float maxval_c0 = -INFINITY;
    if (oh < params.OH && ow < params.OW) {
        for (int kh_i = 0; kh_i < params.kh; ++kh_i) {
            const int sih = th * params.sh + kh_i * params.dh;
            if (sih >= smem_h) continue;
            for (int kw_i = 0; kw_i < params.kw; ++kw_i) {
                const int siw = tw * params.sw + kw_i * params.dw;
                if (siw >= smem_w) continue;
                const float val = buf[0][sih * smem_w + siw];
                if (val > maxval_c0) maxval_c0 = val;
            }
        }
    }
    if (oh < params.OH && ow < params.OW) {
        const int64_t out_idx = ((static_cast<int64_t>(n) * params.OH + oh) * params.OW + ow) * params.C + c0;
        output[out_idx] = static_cast<int8_t>(maxval_c0);
    }
    if (has_c1) {
        for (int i = tid; i < total_smem; i += nthreads) {
            const int sih = i / smem_w;
            const int siw = i % smem_w;
            const int ih = ih_start + sih;
            const int iw = iw_start + siw;
            if (ih >= 0 && ih < params.H && iw >= 0 && iw < params.W) {
                const int64_t in_idx = ((static_cast<int64_t>(n) * params.H + ih) * params.W + iw) * params.C + c1;
                buf[1][i] = static_cast<float>(input[in_idx]);
            } else {
                buf[1][i] = -INFINITY;
            }
        }
        __syncthreads();
        float maxval_c1 = -INFINITY;
        if (oh < params.OH && ow < params.OW) {
            for (int kh_i = 0; kh_i < params.kh; ++kh_i) {
                const int sih = th * params.sh + kh_i * params.dh;
                if (sih >= smem_h) continue;
                for (int kw_i = 0; kw_i < params.kw; ++kw_i) {
                    const int siw = tw * params.sw + kw_i * params.dw;
                    if (siw >= smem_w) continue;
                    const float val = buf[1][sih * smem_w + siw];
                    if (val > maxval_c1) maxval_c1 = val;
                }
            }
            const int64_t out_idx = ((static_cast<int64_t>(n) * params.OH + oh) * params.OW + ow) * params.C + c1;
            output[out_idx] = static_cast<int8_t>(maxval_c1);
        }
    }
}

void maxpool_v5(const int8_t* input, int8_t* output, const PoolParams& params, cudaStream_t stream) {
    NVTX_RANGE_PUSH_C("maxpool_v5_i8", NVTX_COLOR_MAXPOOL);
    constexpr int TILE_OH = 8;
    constexpr int TILE_OW = 8;
    if (params.C < 2) { NVTX_RANGE_POP(); maxpool_v1(input, output, params, stream); return; }
    const int blocks_oh = static_cast<int>((params.OH + TILE_OH - 1) / TILE_OH);
    const int blocks_ow = static_cast<int>((params.OW + TILE_OW - 1) / TILE_OW);
    int smem_h = (TILE_OH - 1) * params.sh + (params.kh - 1) * params.dh + 1;
    int smem_w = (TILE_OW - 1) * params.sw + (params.kw - 1) * params.dw + 1;
    smem_h = min(smem_h, static_cast<int>(params.H + 2 * params.ph));
    smem_w = min(smem_w, static_cast<int>(params.W + 2 * params.pw));
    size_t smem_bytes = 2 * static_cast<size_t>(smem_h) * smem_w * sizeof(float);
    int smem_limit = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&smem_limit, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0));
    if (smem_bytes > static_cast<size_t>(smem_limit)) { NVTX_RANGE_POP(); maxpool_v1(input, output, params, stream); return; }
    const int c_pairs = static_cast<int>((params.C + 1) / 2);
    dim3 block(TILE_OW, TILE_OH);
    dim3 grid(blocks_oh * blocks_ow, c_pairs, static_cast<int>(params.N));
    if (smem_bytes > 49152) {
        CUDA_CHECK(cudaFuncSetAttribute(maxpool_v5_kernel_i8<TILE_OH, TILE_OW>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem_bytes)));
    }
    maxpool_v5_kernel_i8<TILE_OH, TILE_OW><<<grid, block, smem_bytes, stream>>>(
        input, output, params, blocks_oh, blocks_ow, smem_h, smem_w);
    CUDA_CHECK(cudaGetLastError());
    NVTX_RANGE_POP();
}

// --- v6: warp specialization ---
template <int TILE_OH = 8, int TILE_OW = 8, int NUM_LOAD_WARPS = 2, int NUM_COMPUTE_WARPS = 6>
__global__ void maxpool_v6_kernel_i8(
    const int8_t* __restrict__ input,
    int8_t* __restrict__ output,
    const PoolParams params,
    int blocks_oh, int blocks_ow,
    int smem_h, int smem_w)
{
    constexpr int NUM_LOAD_THREADS = NUM_LOAD_WARPS * 32;
    constexpr int NUM_COMPUTE_THREADS = NUM_COMPUTE_WARPS * 32;
    constexpr int TILE_SIZE = TILE_OH * TILE_OW;
    extern __shared__ float sdata[];
    const int n = blockIdx.z;
    const int c = blockIdx.y;
    const int tile_idx = blockIdx.x;
    const int tile_oh = tile_idx / blocks_ow;
    const int tile_ow = tile_idx % blocks_ow;
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const bool is_load_warp = warp_id < NUM_LOAD_WARPS;
    const int ih_start = tile_oh * TILE_OH * params.sh - params.ph;
    const int iw_start = tile_ow * TILE_OW * params.sw - params.pw;
    const int total_smem = smem_h * smem_w;
    if (is_load_warp) {
        for (int i = tid; i < total_smem; i += NUM_LOAD_THREADS) {
            const int sih = i / smem_w;
            const int siw = i % smem_w;
            const int ih = ih_start + sih;
            const int iw = iw_start + siw;
            if (ih >= 0 && ih < params.H && iw >= 0 && iw < params.W) {
                const int64_t in_idx = ((static_cast<int64_t>(n) * params.H + ih) * params.W + iw) * params.C + c;
                sdata[i] = static_cast<float>(input[in_idx]);
            } else {
                sdata[i] = -INFINITY;
            }
        }
    }
    __syncthreads();
    if (!is_load_warp) {
        const int compute_tid = tid - NUM_LOAD_THREADS;
        for (int i = compute_tid; i < TILE_SIZE; i += NUM_COMPUTE_THREADS) {
            const int th = i / TILE_OW;
            const int tw = i % TILE_OW;
            const int oh = tile_oh * TILE_OH + th;
            const int ow = tile_ow * TILE_OW + tw;
            if (oh >= params.OH || ow >= params.OW) continue;
            float maxval = -INFINITY;
            for (int kh_i = 0; kh_i < params.kh; ++kh_i) {
                const int sih = th * params.sh + kh_i * params.dh;
                if (sih >= smem_h) continue;
                for (int kw_i = 0; kw_i < params.kw; ++kw_i) {
                    const int siw = tw * params.sw + kw_i * params.dw;
                    if (siw >= smem_w) continue;
                    const float val = sdata[sih * smem_w + siw];
                    if (val > maxval) maxval = val;
                }
            }
            const int64_t out_idx = ((static_cast<int64_t>(n) * params.OH + oh) * params.OW + ow) * params.C + c;
            output[out_idx] = static_cast<int8_t>(maxval);
        }
    }
}

void maxpool_v6(const int8_t* input, int8_t* output, const PoolParams& params, cudaStream_t stream) {
    NVTX_RANGE_PUSH_C("maxpool_v6_i8", NVTX_COLOR_MAXPOOL);
    constexpr int TILE_OH = 8;
    constexpr int TILE_OW = 8;
    constexpr int BLOCK_SIZE = 256;
    const int blocks_oh = static_cast<int>((params.OH + TILE_OH - 1) / TILE_OH);
    const int blocks_ow = static_cast<int>((params.OW + TILE_OW - 1) / TILE_OW);
    int smem_h = (TILE_OH - 1) * params.sh + (params.kh - 1) * params.dh + 1;
    int smem_w = (TILE_OW - 1) * params.sw + (params.kw - 1) * params.dw + 1;
    smem_h = min(smem_h, static_cast<int>(params.H + 2 * params.ph));
    smem_w = min(smem_w, static_cast<int>(params.W + 2 * params.pw));
    size_t smem_bytes = static_cast<size_t>(smem_h) * smem_w * sizeof(float);
    int smem_limit = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&smem_limit, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0));
    if (smem_bytes > static_cast<size_t>(smem_limit)) { NVTX_RANGE_POP(); maxpool_v1(input, output, params, stream); return; }
    dim3 block(BLOCK_SIZE);
    dim3 grid(blocks_oh * blocks_ow, static_cast<int>(params.C), static_cast<int>(params.N));
    if (smem_bytes > 49152) {
        CUDA_CHECK(cudaFuncSetAttribute(maxpool_v6_kernel_i8<TILE_OH, TILE_OW>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem_bytes)));
    }
    maxpool_v6_kernel_i8<TILE_OH, TILE_OW><<<grid, block, smem_bytes, stream>>>(
        input, output, params, blocks_oh, blocks_ow, smem_h, smem_w);
    CUDA_CHECK(cudaGetLastError());
    NVTX_RANGE_POP();
}

// --- v7: alternative mappings ---
void maxpool_v7(const int8_t* input, int8_t* output, const PoolParams& params, int mapping, cudaStream_t stream) {
    NVTX_RANGE_PUSH_C("maxpool_v7_i8", NVTX_COLOR_MAXPOOL);
    switch (mapping) {
        case 0: { NVTX_RANGE_POP(); maxpool_v0(input, output, params, stream); return; }
        case 1: {
            const int C_groups = static_cast<int>((params.C + 3) / 4);
            const int64_t grid_z_64 = params.N * C_groups;
            if (grid_z_64 > 65535) { NVTX_RANGE_POP(); maxpool_v0(input, output, params, stream); return; }
            dim3 block(8, 8, 4);
            dim3 grid(static_cast<int>((params.OW + 7) / 8), static_cast<int>((params.OH + 7) / 8), static_cast<int>(grid_z_64));
            maxpool_v7_mappingB_kernel<int8_t><<<grid, block, 0, stream>>>(input, output, params, C_groups);
            CUDA_CHECK(cudaGetLastError());
            break;
        }
        case 2: {
            const int C_groups = static_cast<int>((params.C + 255) / 256);
            const int64_t grid_z_64 = params.N * C_groups;
            if (params.OW > 65535 || params.OH > 65535 || grid_z_64 > 65535) { NVTX_RANGE_POP(); maxpool_v0(input, output, params, stream); return; }
            dim3 block(256, 1, 1);
            dim3 grid(static_cast<int>(params.OW), static_cast<int>(params.OH), static_cast<int>(grid_z_64));
            maxpool_v7_mappingC_kernel<int8_t><<<grid, block, 0, stream>>>(input, output, params, C_groups);
            CUDA_CHECK(cudaGetLastError());
            break;
        }
        case 3: {
            // v7mD uses half2 vectorized loads in generic template — misaligned
            // for 1-byte types (int8_t, fp8). Fall back to v0.
            NVTX_RANGE_POP(); maxpool_v0(input, output, params, stream); return;
        }
        default:
            NVTX_RANGE_POP();
            throw std::invalid_argument("unsupported mapping: " + std::to_string(mapping));
    }
    NVTX_RANGE_POP();
}

// --- v8: auto-tuned tiling (fallback to v2) ---
void maxpool_v8(const int8_t* input, int8_t* output, const PoolParams& params, cudaStream_t stream) {
    NVTX_RANGE_PUSH_C("maxpool_v8_i8", NVTX_COLOR_MAXPOOL);
    if (params.C % 16 == 0) { NVTX_RANGE_POP(); maxpool_v2(input, output, params, stream); return; }
    NVTX_RANGE_POP();
    maxpool_v0(input, output, params, stream);
}

// --- v9: TMA (fallback) ---
void maxpool_v9(const int8_t* input, int8_t* output, const PoolParams& params, cudaStream_t stream) {
    NVTX_RANGE_PUSH_C("maxpool_v9_i8", NVTX_COLOR_MAXPOOL);
    if (params.C % 16 == 0) { maxpool_v2(input, output, params, stream); } else { maxpool_v0(input, output, params, stream); }
    NVTX_RANGE_POP();
}

// --- v10: persistent (fallback) ---
void maxpool_v10(const int8_t* input, int8_t* output, const PoolParams& params, cudaStream_t stream) {
    NVTX_RANGE_PUSH_C("maxpool_v10_i8", NVTX_COLOR_MAXPOOL);
    maxpool_v2(input, output, params, stream);
    NVTX_RANGE_POP();
}

// --- v11: warp-shuffle (fallback to v4) ---
void maxpool_v11(const int8_t* input, int8_t* output, const PoolParams& params, cudaStream_t stream) {
    NVTX_RANGE_PUSH_C("maxpool_v11_i8", NVTX_COLOR_MAXPOOL);
    maxpool_v4(input, output, params, stream);
    NVTX_RANGE_POP();
}

// --- v12: L2-aware (fallback) ---
void maxpool_v12(const int8_t* input, int8_t* output, const PoolParams& params, cudaStream_t stream) {
    NVTX_RANGE_PUSH_C("maxpool_v12_i8", NVTX_COLOR_MAXPOOL);
    maxpool_v2(input, output, params, stream);
    NVTX_RANGE_POP();
}

// --- v13: channel-vectorized (fallback) ---
void maxpool_v13(const int8_t* input, int8_t* output, const PoolParams& params, cudaStream_t stream) {
    NVTX_RANGE_PUSH_C("maxpool_v13_i8", NVTX_COLOR_MAXPOOL);
    if (params.C % 16 == 0) { maxpool_v2(input, output, params, stream); } else { maxpool_v0(input, output, params, stream); }
    NVTX_RANGE_POP();
}

// --- v14: adaptive dispatcher ---
void maxpool_v14(const int8_t* input, int8_t* output, const PoolParams& params, cudaStream_t stream) {
    NVTX_RANGE_PUSH_C("maxpool_v14_i8", NVTX_COLOR_MAXPOOL);
    bool is_global = (params.kh >= params.H && params.kw >= params.W);
    if (is_global) { maxpool_v13(input, output, params, stream); NVTX_RANGE_POP(); return; }
    int64_t total_output_elems = params.N * params.OH * params.OW * params.C;
    if (total_output_elems < 65536) { maxpool_v10(input, output, params, stream); NVTX_RANGE_POP(); return; }
    if (params.sh == 1 && params.sw == 1 && params.kh == 3 && params.kw == 3) { maxpool_v15(input, output, params, stream); NVTX_RANGE_POP(); return; }
    if (params.C % 16 == 0) { maxpool_v2(input, output, params, stream); } else { maxpool_v0(input, output, params, stream); }
    NVTX_RANGE_POP();
}

// --- v15: swizzled shared memory ---
void maxpool_v15(const int8_t* input, int8_t* output, const PoolParams& params, cudaStream_t stream) {
    NVTX_RANGE_PUSH_C("maxpool_v15_i8", NVTX_COLOR_MAXPOOL);
    constexpr int TILE_OH = 8;
    constexpr int TILE_OW = 8;
    const int blocks_oh = static_cast<int>((params.OH + TILE_OH - 1) / TILE_OH);
    const int blocks_ow = static_cast<int>((params.OW + TILE_OW - 1) / TILE_OW);
    int smem_h = (TILE_OH - 1) * params.sh + (params.kh - 1) * params.dh + 1;
    int smem_w = (TILE_OW - 1) * params.sw + (params.kw - 1) * params.dw + 1;
    size_t smem_bytes = static_cast<size_t>(smem_h) * (smem_w + 1) * sizeof(int8_t);
    dim3 block(TILE_OW, TILE_OH);
    dim3 grid(blocks_oh * blocks_ow, static_cast<int>(params.C), static_cast<int>(params.N));
    if (smem_bytes > 49152) {
        CUDA_CHECK(cudaFuncSetAttribute(maxpool_v15_kernel<int8_t, true>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem_bytes)));
    }
    maxpool_v15_kernel<int8_t, true><<<grid, block, smem_bytes, stream>>>(
        input, output, params, blocks_oh, blocks_ow, smem_h, smem_w, params.divisor_override);
    CUDA_CHECK(cudaGetLastError());
    NVTX_RANGE_POP();
}

// ============================================================================
// INT16 MaxPool kernels
// ============================================================================

// --- v0: template instantiation ---
void maxpool_v0(const int16_t* input, int16_t* output, const PoolParams& params, cudaStream_t stream) {
    NVTX_RANGE_PUSH_C("maxpool_v0_i16", NVTX_COLOR_MAXPOOL);
    const int64_t total = params.OH * params.OW * params.C;
    const int threads = 256;
    const int blocks_x = static_cast<int>((total + threads - 1) / threads);
    dim3 grid(blocks_x, 1, static_cast<int>(params.N));
    maxpool_v0_kernel<int16_t><<<grid, threads, 0, stream>>>(input, output, params);
    CUDA_CHECK(cudaGetLastError());
    NVTX_RANGE_POP();
}

// --- v1: explicit kernel (float shared memory, scalar loads) ---
template <int TILE_OH = 8, int TILE_OW = 8>
__global__ void maxpool_v1_kernel_i16(
    const int16_t* __restrict__ input,
    int16_t* __restrict__ output,
    const PoolParams params,
    int blocks_oh, int blocks_ow,
    int smem_h, int smem_w)
{
    extern __shared__ float sdata[];
    const int n = blockIdx.z;
    const int c = blockIdx.y;
    const int tile_idx = blockIdx.x;
    const int tile_oh = tile_idx / blocks_ow;
    const int tile_ow = tile_idx % blocks_ow;
    const int th = threadIdx.y;
    const int tw = threadIdx.x;
    const int oh = tile_oh * TILE_OH + th;
    const int ow = tile_ow * TILE_OW + tw;
    const int ih_start = tile_oh * TILE_OH * params.sh - params.ph;
    const int iw_start = tile_ow * TILE_OW * params.sw - params.pw;
    const int total_smem = smem_h * smem_w;
    const int tid = th * TILE_OW + tw;
    for (int i = tid; i < total_smem; i += TILE_OH * TILE_OW) {
        const int sih = i / smem_w;
        const int siw = i % smem_w;
        const int ih = ih_start + sih;
        const int iw = iw_start + siw;
        if (ih >= 0 && ih < params.H && iw >= 0 && iw < params.W) {
            const int64_t in_idx = ((static_cast<int64_t>(n) * params.H + ih) * params.W + iw) * params.C + c;
            sdata[i] = static_cast<float>(input[in_idx]);
        } else {
            sdata[i] = -INFINITY;
        }
    }
    __syncthreads();
    if (oh >= params.OH || ow >= params.OW) return;
    float maxval = -INFINITY;
    for (int kh_i = 0; kh_i < params.kh; ++kh_i) {
        const int sih = th * params.sh + kh_i * params.dh;
        if (sih >= smem_h) continue;
        for (int kw_i = 0; kw_i < params.kw; ++kw_i) {
            const int siw = tw * params.sw + kw_i * params.dw;
            if (siw >= smem_w) continue;
            const float val = sdata[sih * smem_w + siw];
            if (val > maxval) maxval = val;
        }
    }
    const int64_t out_idx = ((static_cast<int64_t>(n) * params.OH + oh) * params.OW + ow) * params.C + c;
    output[out_idx] = static_cast<int16_t>(maxval);
}

void maxpool_v1(const int16_t* input, int16_t* output, const PoolParams& params, cudaStream_t stream) {
    NVTX_RANGE_PUSH_C("maxpool_v1_i16", NVTX_COLOR_MAXPOOL);
    constexpr int TILE_OH = 8;
    constexpr int TILE_OW = 8;
    const int blocks_oh = static_cast<int>((params.OH + TILE_OH - 1) / TILE_OH);
    const int blocks_ow = static_cast<int>((params.OW + TILE_OW - 1) / TILE_OW);
    int smem_h = (TILE_OH - 1) * params.sh + (params.kh - 1) * params.dh + 1;
    int smem_w = (TILE_OW - 1) * params.sw + (params.kw - 1) * params.dw + 1;
    smem_h = min(smem_h, static_cast<int>(params.H + 2 * params.ph));
    smem_w = min(smem_w, static_cast<int>(params.W + 2 * params.pw));
    dim3 block(TILE_OW, TILE_OH);
    dim3 grid(blocks_oh * blocks_ow, static_cast<int>(params.C), static_cast<int>(params.N));
    size_t smem_bytes = static_cast<size_t>(smem_h) * smem_w * sizeof(float);
    maxpool_v1_kernel_i16<TILE_OH, TILE_OW><<<grid, block, smem_bytes, stream>>>(
        input, output, params, blocks_oh, blocks_ow, smem_h, smem_w);
    CUDA_CHECK(cudaGetLastError());
    NVTX_RANGE_POP();
}

// --- v2: short4 vectorized loads (8 channels per load = 128-bit) ---
template <int VEC>
__global__ void maxpool_v2_kernel_i16(
    const int16_t* __restrict__ input,
    int16_t* __restrict__ output,
    const PoolParams params)
{
    const int64_t n = blockIdx.z;
    const int64_t C_vec = params.C / VEC;
    const int64_t flat = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t total = params.OH * params.OW * C_vec;
    if (n >= params.N || flat >= total) return;
    const int64_t oh = flat / (params.OW * C_vec);
    const int64_t ow = (flat / C_vec) % params.OW;
    const int64_t c_vec = flat % C_vec;
    const int64_t c = c_vec * VEC;
    float maxval[VEC];
    #pragma unroll
    for (int v = 0; v < VEC; ++v) maxval[v] = -INFINITY;
    for (int kh_i = 0; kh_i < params.kh; ++kh_i) {
        const int64_t ih = oh * params.sh - params.ph + static_cast<int64_t>(kh_i) * params.dh;
        if (ih < 0 || ih >= params.H) continue;
        for (int kw_i = 0; kw_i < params.kw; ++kw_i) {
            const int64_t iw = ow * params.sw - params.pw + static_cast<int64_t>(kw_i) * params.dw;
            if (iw < 0 || iw >= params.W) continue;
            const int64_t in_idx = ((n * params.H + ih) * params.W + iw) * params.C + c;
            short4 vec = *reinterpret_cast<const short4*>(&input[in_idx]);
            #pragma unroll
            for (int v = 0; v < VEC; ++v) {
                float val = static_cast<float>(reinterpret_cast<const int16_t*>(&vec)[v]);
                if (val > maxval[v]) maxval[v] = val;
            }
        }
    }
    const int64_t out_idx = ((n * params.OH + oh) * params.OW + ow) * params.C + c;
    int16_t* out_ptr = &output[out_idx];
    #pragma unroll
    for (int v = 0; v < VEC; ++v) out_ptr[v] = static_cast<int16_t>(maxval[v]);
}

void maxpool_v2(const int16_t* input, int16_t* output, const PoolParams& params, cudaStream_t stream) {
    NVTX_RANGE_PUSH_C("maxpool_v2_i16", NVTX_COLOR_MAXPOOL);
    constexpr int VEC = 8;
    if (params.C % VEC != 0) { NVTX_RANGE_POP(); maxpool_v0(input, output, params, stream); return; }
    const int64_t C_vec = params.C / VEC;
    const int64_t total = params.OH * params.OW * C_vec;
    const int threads = 256;
    const int blocks_x = static_cast<int>((total + threads - 1) / threads);
    dim3 grid(blocks_x, 1, static_cast<int>(params.N));
    maxpool_v2_kernel_i16<VEC><<<grid, threads, 0, stream>>>(input, output, params);
    CUDA_CHECK(cudaGetLastError());
    NVTX_RANGE_POP();
}

// --- v3: register blocking ---
void maxpool_v3(const int16_t* input, int16_t* output, const PoolParams& params, cudaStream_t stream) {
    NVTX_RANGE_PUSH_C("maxpool_v3_i16", NVTX_COLOR_MAXPOOL);
    constexpr int BLOCK = 4;
    if (params.OH < BLOCK || params.OH % BLOCK != 0) { NVTX_RANGE_POP(); maxpool_v0(input, output, params, stream); return; }
    const int64_t OH_block = params.OH / BLOCK;
    const int64_t total = OH_block * params.OW * params.C;
    const int threads = 256;
    const int blocks_x = static_cast<int>((total + threads - 1) / threads);
    dim3 grid(blocks_x, 1, static_cast<int>(params.N));
    maxpool_v3_kernel<int16_t, BLOCK><<<grid, threads, 0, stream>>>(input, output, params);
    CUDA_CHECK(cudaGetLastError());
    NVTX_RANGE_POP();
}

// --- v4: warp-level reduce ---
void maxpool_v4(const int16_t* input, int16_t* output, const PoolParams& params, cudaStream_t stream) {
    NVTX_RANGE_PUSH_C("maxpool_v4_i16", NVTX_COLOR_MAXPOOL);
    const int64_t total = params.OH * params.OW * params.C;
    const int threads = 256;
    const int warps_per_block = threads / 32;
    const int blocks_x = static_cast<int>((total + warps_per_block - 1) / warps_per_block);
    dim3 grid(blocks_x, 1, static_cast<int>(params.N));
    maxpool_v4_kernel<int16_t><<<grid, threads, 0, stream>>>(input, output, params);
    CUDA_CHECK(cudaGetLastError());
    NVTX_RANGE_POP();
}

// --- v5: double buffer ---
template <int TILE_OH = 8, int TILE_OW = 8>
__global__ void maxpool_v5_kernel_i16(
    const int16_t* __restrict__ input,
    int16_t* __restrict__ output,
    const PoolParams params,
    int blocks_oh, int blocks_ow,
    int smem_h, int smem_w)
{
    extern __shared__ float sdata[];
    float* buf[2] = { sdata, sdata + smem_h * smem_w };
    const int n = blockIdx.z;
    const int c_pair = blockIdx.y;
    const int c0 = c_pair * 2;
    const int c1 = c0 + 1;
    const bool has_c1 = (c1 < params.C);
    const int tile_idx = blockIdx.x;
    const int tile_oh = tile_idx / blocks_ow;
    const int tile_ow = tile_idx % blocks_ow;
    const int th = threadIdx.y;
    const int tw = threadIdx.x;
    const int oh = tile_oh * TILE_OH + th;
    const int ow = tile_ow * TILE_OW + tw;
    const int ih_start = tile_oh * TILE_OH * params.sh - params.ph;
    const int iw_start = tile_ow * TILE_OW * params.sw - params.pw;
    const int total_smem = smem_h * smem_w;
    const int tid = th * TILE_OW + tw;
    const int nthreads = TILE_OH * TILE_OW;
    for (int i = tid; i < total_smem; i += nthreads) {
        const int sih = i / smem_w;
        const int siw = i % smem_w;
        const int ih = ih_start + sih;
        const int iw = iw_start + siw;
        if (ih >= 0 && ih < params.H && iw >= 0 && iw < params.W) {
            const int64_t in_idx = ((static_cast<int64_t>(n) * params.H + ih) * params.W + iw) * params.C + c0;
            buf[0][i] = static_cast<float>(input[in_idx]);
        } else { buf[0][i] = -INFINITY; }
    }
    __syncthreads();
    float maxval_c0 = -INFINITY;
    if (oh < params.OH && ow < params.OW) {
        for (int kh_i = 0; kh_i < params.kh; ++kh_i) {
            const int sih = th * params.sh + kh_i * params.dh;
            if (sih >= smem_h) continue;
            for (int kw_i = 0; kw_i < params.kw; ++kw_i) {
                const int siw = tw * params.sw + kw_i * params.dw;
                if (siw >= smem_w) continue;
                const float val = buf[0][sih * smem_w + siw];
                if (val > maxval_c0) maxval_c0 = val;
            }
        }
    }
    if (oh < params.OH && ow < params.OW) {
        const int64_t out_idx = ((static_cast<int64_t>(n) * params.OH + oh) * params.OW + ow) * params.C + c0;
        output[out_idx] = static_cast<int16_t>(maxval_c0);
    }
    if (has_c1) {
        for (int i = tid; i < total_smem; i += nthreads) {
            const int sih = i / smem_w;
            const int siw = i % smem_w;
            const int ih = ih_start + sih;
            const int iw = iw_start + siw;
            if (ih >= 0 && ih < params.H && iw >= 0 && iw < params.W) {
                const int64_t in_idx = ((static_cast<int64_t>(n) * params.H + ih) * params.W + iw) * params.C + c1;
                buf[1][i] = static_cast<float>(input[in_idx]);
            } else { buf[1][i] = -INFINITY; }
        }
        __syncthreads();
        float maxval_c1 = -INFINITY;
        if (oh < params.OH && ow < params.OW) {
            for (int kh_i = 0; kh_i < params.kh; ++kh_i) {
                const int sih = th * params.sh + kh_i * params.dh;
                if (sih >= smem_h) continue;
                for (int kw_i = 0; kw_i < params.kw; ++kw_i) {
                    const int siw = tw * params.sw + kw_i * params.dw;
                    if (siw >= smem_w) continue;
                    const float val = buf[1][sih * smem_w + siw];
                    if (val > maxval_c1) maxval_c1 = val;
                }
            }
            const int64_t out_idx = ((static_cast<int64_t>(n) * params.OH + oh) * params.OW + ow) * params.C + c1;
            output[out_idx] = static_cast<int16_t>(maxval_c1);
        }
    }
}

void maxpool_v5(const int16_t* input, int16_t* output, const PoolParams& params, cudaStream_t stream) {
    NVTX_RANGE_PUSH_C("maxpool_v5_i16", NVTX_COLOR_MAXPOOL);
    constexpr int TILE_OH = 8; constexpr int TILE_OW = 8;
    if (params.C < 2) { NVTX_RANGE_POP(); maxpool_v1(input, output, params, stream); return; }
    const int blocks_oh = static_cast<int>((params.OH + TILE_OH - 1) / TILE_OH);
    const int blocks_ow = static_cast<int>((params.OW + TILE_OW - 1) / TILE_OW);
    int smem_h = (TILE_OH - 1) * params.sh + (params.kh - 1) * params.dh + 1;
    int smem_w = (TILE_OW - 1) * params.sw + (params.kw - 1) * params.dw + 1;
    smem_h = min(smem_h, static_cast<int>(params.H + 2 * params.ph));
    smem_w = min(smem_w, static_cast<int>(params.W + 2 * params.pw));
    size_t smem_bytes = 2 * static_cast<size_t>(smem_h) * smem_w * sizeof(float);
    int smem_limit = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&smem_limit, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0));
    if (smem_bytes > static_cast<size_t>(smem_limit)) { NVTX_RANGE_POP(); maxpool_v1(input, output, params, stream); return; }
    const int c_pairs = static_cast<int>((params.C + 1) / 2);
    dim3 block(TILE_OW, TILE_OH);
    dim3 grid(blocks_oh * blocks_ow, c_pairs, static_cast<int>(params.N));
    if (smem_bytes > 49152) {
        CUDA_CHECK(cudaFuncSetAttribute(maxpool_v5_kernel_i16<TILE_OH, TILE_OW>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem_bytes)));
    }
    maxpool_v5_kernel_i16<TILE_OH, TILE_OW><<<grid, block, smem_bytes, stream>>>(
        input, output, params, blocks_oh, blocks_ow, smem_h, smem_w);
    CUDA_CHECK(cudaGetLastError());
    NVTX_RANGE_POP();
}

// --- v6: warp specialization ---
template <int TILE_OH = 8, int TILE_OW = 8, int NUM_LOAD_WARPS = 2, int NUM_COMPUTE_WARPS = 6>
__global__ void maxpool_v6_kernel_i16(
    const int16_t* __restrict__ input,
    int16_t* __restrict__ output,
    const PoolParams params,
    int blocks_oh, int blocks_ow,
    int smem_h, int smem_w)
{
    constexpr int NUM_LOAD_THREADS = NUM_LOAD_WARPS * 32;
    constexpr int NUM_COMPUTE_THREADS = NUM_COMPUTE_WARPS * 32;
    constexpr int TILE_SIZE = TILE_OH * TILE_OW;
    extern __shared__ float sdata[];
    const int n = blockIdx.z;
    const int c = blockIdx.y;
    const int tile_idx = blockIdx.x;
    const int tile_oh = tile_idx / blocks_ow;
    const int tile_ow = tile_idx % blocks_ow;
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const bool is_load_warp = warp_id < NUM_LOAD_WARPS;
    const int ih_start = tile_oh * TILE_OH * params.sh - params.ph;
    const int iw_start = tile_ow * TILE_OW * params.sw - params.pw;
    const int total_smem = smem_h * smem_w;
    if (is_load_warp) {
        for (int i = tid; i < total_smem; i += NUM_LOAD_THREADS) {
            const int sih = i / smem_w;
            const int siw = i % smem_w;
            const int ih = ih_start + sih;
            const int iw = iw_start + siw;
            if (ih >= 0 && ih < params.H && iw >= 0 && iw < params.W) {
                const int64_t in_idx = ((static_cast<int64_t>(n) * params.H + ih) * params.W + iw) * params.C + c;
                sdata[i] = static_cast<float>(input[in_idx]);
            } else { sdata[i] = -INFINITY; }
        }
    }
    __syncthreads();
    if (!is_load_warp) {
        const int compute_tid = tid - NUM_LOAD_THREADS;
        for (int i = compute_tid; i < TILE_SIZE; i += NUM_COMPUTE_THREADS) {
            const int th = i / TILE_OW;
            const int tw = i % TILE_OW;
            const int oh = tile_oh * TILE_OH + th;
            const int ow = tile_ow * TILE_OW + tw;
            if (oh >= params.OH || ow >= params.OW) continue;
            float maxval = -INFINITY;
            for (int kh_i = 0; kh_i < params.kh; ++kh_i) {
                const int sih = th * params.sh + kh_i * params.dh;
                if (sih >= smem_h) continue;
                for (int kw_i = 0; kw_i < params.kw; ++kw_i) {
                    const int siw = tw * params.sw + kw_i * params.dw;
                    if (siw >= smem_w) continue;
                    const float val = sdata[sih * smem_w + siw];
                    if (val > maxval) maxval = val;
                }
            }
            const int64_t out_idx = ((static_cast<int64_t>(n) * params.OH + oh) * params.OW + ow) * params.C + c;
            output[out_idx] = static_cast<int16_t>(maxval);
        }
    }
}

void maxpool_v6(const int16_t* input, int16_t* output, const PoolParams& params, cudaStream_t stream) {
    NVTX_RANGE_PUSH_C("maxpool_v6_i16", NVTX_COLOR_MAXPOOL);
    constexpr int TILE_OH = 8; constexpr int TILE_OW = 8; constexpr int BLOCK_SIZE = 256;
    const int blocks_oh = static_cast<int>((params.OH + TILE_OH - 1) / TILE_OH);
    const int blocks_ow = static_cast<int>((params.OW + TILE_OW - 1) / TILE_OW);
    int smem_h = (TILE_OH - 1) * params.sh + (params.kh - 1) * params.dh + 1;
    int smem_w = (TILE_OW - 1) * params.sw + (params.kw - 1) * params.dw + 1;
    smem_h = min(smem_h, static_cast<int>(params.H + 2 * params.ph));
    smem_w = min(smem_w, static_cast<int>(params.W + 2 * params.pw));
    size_t smem_bytes = static_cast<size_t>(smem_h) * smem_w * sizeof(float);
    int smem_limit = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&smem_limit, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0));
    if (smem_bytes > static_cast<size_t>(smem_limit)) { NVTX_RANGE_POP(); maxpool_v1(input, output, params, stream); return; }
    dim3 block(BLOCK_SIZE);
    dim3 grid(blocks_oh * blocks_ow, static_cast<int>(params.C), static_cast<int>(params.N));
    if (smem_bytes > 49152) {
        CUDA_CHECK(cudaFuncSetAttribute(maxpool_v6_kernel_i16<TILE_OH, TILE_OW>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem_bytes)));
    }
    maxpool_v6_kernel_i16<TILE_OH, TILE_OW><<<grid, block, smem_bytes, stream>>>(
        input, output, params, blocks_oh, blocks_ow, smem_h, smem_w);
    CUDA_CHECK(cudaGetLastError());
    NVTX_RANGE_POP();
}

// --- v7: alternative mappings ---
void maxpool_v7(const int16_t* input, int16_t* output, const PoolParams& params, int mapping, cudaStream_t stream) {
    NVTX_RANGE_PUSH_C("maxpool_v7_i16", NVTX_COLOR_MAXPOOL);
    switch (mapping) {
        case 0: { NVTX_RANGE_POP(); maxpool_v0(input, output, params, stream); return; }
        case 1: {
            const int C_groups = static_cast<int>((params.C + 3) / 4);
            const int64_t grid_z_64 = params.N * C_groups;
            if (grid_z_64 > 65535) { NVTX_RANGE_POP(); maxpool_v0(input, output, params, stream); return; }
            dim3 block(8, 8, 4);
            dim3 grid(static_cast<int>((params.OW + 7) / 8), static_cast<int>((params.OH + 7) / 8), static_cast<int>(grid_z_64));
            maxpool_v7_mappingB_kernel<int16_t><<<grid, block, 0, stream>>>(input, output, params, C_groups);
            CUDA_CHECK(cudaGetLastError()); break;
        }
        case 2: {
            const int C_groups = static_cast<int>((params.C + 255) / 256);
            const int64_t grid_z_64 = params.N * C_groups;
            if (params.OW > 65535 || params.OH > 65535 || grid_z_64 > 65535) { NVTX_RANGE_POP(); maxpool_v0(input, output, params, stream); return; }
            dim3 block(256, 1, 1);
            dim3 grid(static_cast<int>(params.OW), static_cast<int>(params.OH), static_cast<int>(grid_z_64));
            maxpool_v7_mappingC_kernel<int16_t><<<grid, block, 0, stream>>>(input, output, params, C_groups);
            CUDA_CHECK(cudaGetLastError()); break;
        }
        case 3: {
            // v7mD uses half2 vectorized loads — misaligned for 2-byte int16_t
            NVTX_RANGE_POP(); maxpool_v0(input, output, params, stream); return;
        }
        default:
            NVTX_RANGE_POP();
            throw std::invalid_argument("unsupported mapping: " + std::to_string(mapping));
    }
    NVTX_RANGE_POP();
}

// --- v8: auto-tuned tiling (fallback) ---
void maxpool_v8(const int16_t* input, int16_t* output, const PoolParams& params, cudaStream_t stream) {
    NVTX_RANGE_PUSH_C("maxpool_v8_i16", NVTX_COLOR_MAXPOOL);
    if (params.C % 8 == 0) { NVTX_RANGE_POP(); maxpool_v2(input, output, params, stream); return; }
    NVTX_RANGE_POP();
    maxpool_v0(input, output, params, stream);
}
void maxpool_v9(const int16_t* input, int16_t* output, const PoolParams& params, cudaStream_t stream) {
    NVTX_RANGE_PUSH_C("maxpool_v9_i16", NVTX_COLOR_MAXPOOL);
    if (params.C % 8 == 0) { maxpool_v2(input, output, params, stream); } else { maxpool_v0(input, output, params, stream); }
    NVTX_RANGE_POP();
}
void maxpool_v10(const int16_t* input, int16_t* output, const PoolParams& params, cudaStream_t stream) {
    NVTX_RANGE_PUSH_C("maxpool_v10_i16", NVTX_COLOR_MAXPOOL);
    maxpool_v2(input, output, params, stream);
    NVTX_RANGE_POP();
}
void maxpool_v11(const int16_t* input, int16_t* output, const PoolParams& params, cudaStream_t stream) {
    NVTX_RANGE_PUSH_C("maxpool_v11_i16", NVTX_COLOR_MAXPOOL);
    maxpool_v4(input, output, params, stream);
    NVTX_RANGE_POP();
}
void maxpool_v12(const int16_t* input, int16_t* output, const PoolParams& params, cudaStream_t stream) {
    NVTX_RANGE_PUSH_C("maxpool_v12_i16", NVTX_COLOR_MAXPOOL);
    maxpool_v2(input, output, params, stream);
    NVTX_RANGE_POP();
}
void maxpool_v13(const int16_t* input, int16_t* output, const PoolParams& params, cudaStream_t stream) {
    NVTX_RANGE_PUSH_C("maxpool_v13_i16", NVTX_COLOR_MAXPOOL);
    if (params.C % 8 == 0) { maxpool_v2(input, output, params, stream); } else { maxpool_v0(input, output, params, stream); }
    NVTX_RANGE_POP();
}

// --- v14: adaptive dispatcher ---
void maxpool_v14(const int16_t* input, int16_t* output, const PoolParams& params, cudaStream_t stream) {
    NVTX_RANGE_PUSH_C("maxpool_v14_i16", NVTX_COLOR_MAXPOOL);
    bool is_global = (params.kh >= params.H && params.kw >= params.W);
    if (is_global) { maxpool_v13(input, output, params, stream); NVTX_RANGE_POP(); return; }
    int64_t total_output_elems = params.N * params.OH * params.OW * params.C;
    if (total_output_elems < 65536) { maxpool_v10(input, output, params, stream); NVTX_RANGE_POP(); return; }
    if (params.sh == 1 && params.sw == 1 && params.kh == 3 && params.kw == 3) { maxpool_v15(input, output, params, stream); NVTX_RANGE_POP(); return; }
    if (params.C % 8 == 0) { maxpool_v2(input, output, params, stream); } else { maxpool_v0(input, output, params, stream); }
    NVTX_RANGE_POP();
}

// --- v15: swizzled shared memory ---
void maxpool_v15(const int16_t* input, int16_t* output, const PoolParams& params, cudaStream_t stream) {
    NVTX_RANGE_PUSH_C("maxpool_v15_i16", NVTX_COLOR_MAXPOOL);
    constexpr int TILE_OH = 8; constexpr int TILE_OW = 8;
    const int blocks_oh = static_cast<int>((params.OH + TILE_OH - 1) / TILE_OH);
    const int blocks_ow = static_cast<int>((params.OW + TILE_OW - 1) / TILE_OW);
    int smem_h = (TILE_OH - 1) * params.sh + (params.kh - 1) * params.dh + 1;
    int smem_w = (TILE_OW - 1) * params.sw + (params.kw - 1) * params.dw + 1;
    size_t smem_bytes = static_cast<size_t>(smem_h) * (smem_w + 1) * sizeof(int16_t);
    dim3 block(TILE_OW, TILE_OH);
    dim3 grid(blocks_oh * blocks_ow, static_cast<int>(params.C), static_cast<int>(params.N));
    if (smem_bytes > 49152) {
        CUDA_CHECK(cudaFuncSetAttribute(maxpool_v15_kernel<int16_t, true>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem_bytes)));
    }
    maxpool_v15_kernel<int16_t, true><<<grid, block, smem_bytes, stream>>>(
        input, output, params, blocks_oh, blocks_ow, smem_h, smem_w, params.divisor_override);
    CUDA_CHECK(cudaGetLastError());
    NVTX_RANGE_POP();
}

// ============================================================================
// FP8 E4M3 MaxPool kernels (scalar loads, compute in float)
// ============================================================================

// --- v0: template instantiation ---
void maxpool_v0_fp8_e4m3(const __nv_fp8_e4m3* input, __nv_fp8_e4m3* output, const PoolParams& params, cudaStream_t stream) {
    NVTX_RANGE_PUSH_C("maxpool_v0_fp8e4m3", NVTX_COLOR_MAXPOOL);
    const int64_t total = params.OH * params.OW * params.C;
    const int threads = 256;
    const int blocks_x = static_cast<int>((total + threads - 1) / threads);
    dim3 grid(blocks_x, 1, static_cast<int>(params.N));
    maxpool_v0_kernel<__nv_fp8_e4m3><<<grid, threads, 0, stream>>>(input, output, params);
    CUDA_CHECK(cudaGetLastError());
    NVTX_RANGE_POP();
}

// --- v1: explicit kernel (float shared memory) ---
template <int TILE_OH = 8, int TILE_OW = 8>
__global__ void maxpool_v1_kernel_fp8_e4m3(
    const __nv_fp8_e4m3* __restrict__ input,
    __nv_fp8_e4m3* __restrict__ output,
    const PoolParams params,
    int blocks_oh, int blocks_ow,
    int smem_h, int smem_w)
{
    extern __shared__ float sdata[];
    const int n = blockIdx.z; const int c = blockIdx.y;
    const int tile_idx = blockIdx.x;
    const int tile_oh = tile_idx / blocks_ow;
    const int tile_ow = tile_idx % blocks_ow;
    const int th = threadIdx.y; const int tw = threadIdx.x;
    const int oh = tile_oh * TILE_OH + th;
    const int ow = tile_ow * TILE_OW + tw;
    const int ih_start = tile_oh * TILE_OH * params.sh - params.ph;
    const int iw_start = tile_ow * TILE_OW * params.sw - params.pw;
    const int total_smem = smem_h * smem_w;
    const int tid = th * TILE_OW + tw;
    for (int i = tid; i < total_smem; i += TILE_OH * TILE_OW) {
        const int sih = i / smem_w; const int siw = i % smem_w;
        const int ih = ih_start + sih; const int iw = iw_start + siw;
        if (ih >= 0 && ih < params.H && iw >= 0 && iw < params.W) {
            const int64_t in_idx = ((static_cast<int64_t>(n) * params.H + ih) * params.W + iw) * params.C + c;
            sdata[i] = static_cast<float>(input[in_idx]);
        } else { sdata[i] = -INFINITY; }
    }
    __syncthreads();
    if (oh >= params.OH || ow >= params.OW) return;
    float maxval = -INFINITY;
    for (int kh_i = 0; kh_i < params.kh; ++kh_i) {
        const int sih = th * params.sh + kh_i * params.dh;
        if (sih >= smem_h) continue;
        for (int kw_i = 0; kw_i < params.kw; ++kw_i) {
            const int siw = tw * params.sw + kw_i * params.dw;
            if (siw >= smem_w) continue;
            const float val = sdata[sih * smem_w + siw];
            if (val > maxval) maxval = val;
        }
    }
    const int64_t out_idx = ((static_cast<int64_t>(n) * params.OH + oh) * params.OW + ow) * params.C + c;
    output[out_idx] = static_cast<__nv_fp8_e4m3>(maxval);
}

void maxpool_v1_fp8_e4m3(const __nv_fp8_e4m3* input, __nv_fp8_e4m3* output, const PoolParams& params, cudaStream_t stream) {
    NVTX_RANGE_PUSH_C("maxpool_v1_fp8e4m3", NVTX_COLOR_MAXPOOL);
    constexpr int TILE_OH = 8; constexpr int TILE_OW = 8;
    const int blocks_oh = static_cast<int>((params.OH + TILE_OH - 1) / TILE_OH);
    const int blocks_ow = static_cast<int>((params.OW + TILE_OW - 1) / TILE_OW);
    int smem_h = (TILE_OH - 1) * params.sh + (params.kh - 1) * params.dh + 1;
    int smem_w = (TILE_OW - 1) * params.sw + (params.kw - 1) * params.dw + 1;
    smem_h = min(smem_h, static_cast<int>(params.H + 2 * params.ph));
    smem_w = min(smem_w, static_cast<int>(params.W + 2 * params.pw));
    dim3 block(TILE_OW, TILE_OH);
    dim3 grid(blocks_oh * blocks_ow, static_cast<int>(params.C), static_cast<int>(params.N));
    size_t smem_bytes = static_cast<size_t>(smem_h) * smem_w * sizeof(float);
    maxpool_v1_kernel_fp8_e4m3<TILE_OH, TILE_OW><<<grid, block, smem_bytes, stream>>>(
        input, output, params, blocks_oh, blocks_ow, smem_h, smem_w);
    CUDA_CHECK(cudaGetLastError());
    NVTX_RANGE_POP();
}

// --- v2: fp8 vectorized loads using uint4 (128-bit, 16 fp8 elements) ---
template <typename T>
__global__ void maxpool_v2_fp8_kernel(
    const T* __restrict__ input, T* __restrict__ output, const PoolParams params)
{
    constexpr int VEC = 16;  // uint4 = 16 bytes = 16 fp8 elements
    const int64_t n = blockIdx.z;
    const int64_t C_vec = params.C / VEC;
    const int64_t flat = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t total = params.OH * params.OW * C_vec;
    if (n >= params.N || flat >= total) return;

    const int64_t oh = flat / (params.OW * C_vec);
    const int64_t ow = (flat / C_vec) % params.OW;
    const int64_t c_vec = flat % C_vec;
    const int64_t c = c_vec * VEC;

    float maxval[VEC];
    #pragma unroll
    for (int v = 0; v < VEC; ++v)
        maxval[v] = -INFINITY;

    for (int kh_i = 0; kh_i < params.kh; ++kh_i) {
        const int64_t ih = oh * params.sh - params.ph + static_cast<int64_t>(kh_i) * params.dh;
        if (ih < 0 || ih >= params.H) continue;
        for (int kw_i = 0; kw_i < params.kw; ++kw_i) {
            const int64_t iw = ow * params.sw - params.pw + static_cast<int64_t>(kw_i) * params.dw;
            if (iw < 0 || iw >= params.W) continue;

            const int64_t in_idx = ((n * params.H + ih) * params.W + iw) * params.C + c;
            uint4 vec = *reinterpret_cast<const uint4*>(&input[in_idx]);
            const uint8_t* bytes = reinterpret_cast<const uint8_t*>(&vec);
            #pragma unroll
            for (int v = 0; v < VEC; ++v) {
                T val;
                reinterpret_cast<uint8_t*>(&val)[0] = bytes[v];
                float fval = static_cast<float>(val);
                if (fval > maxval[v]) maxval[v] = fval;
            }
        }
    }

    const int64_t out_idx = ((n * params.OH + oh) * params.OW + ow) * params.C + c;
    uint4 out_vec;
    #pragma unroll
    for (int v = 0; v < VEC; ++v) {
        T val = static_cast<T>(maxval[v]);
        reinterpret_cast<uint8_t*>(&out_vec)[v] = reinterpret_cast<uint8_t*>(&val)[0];
    }
    *reinterpret_cast<uint4*>(&output[out_idx]) = out_vec;
}

// --- v2: fp8 vectorized loads (previously scalar fallback) ---
void maxpool_v2_fp8_e4m3(const __nv_fp8_e4m3* input, __nv_fp8_e4m3* output, const PoolParams& params, cudaStream_t stream) {
    NVTX_RANGE_PUSH_C("maxpool_v2_fp8e4m3", NVTX_COLOR_MAXPOOL);
    if (params.C % 16 != 0) { NVTX_RANGE_POP(); maxpool_v0_fp8_e4m3(input, output, params, stream); return; }
    const int threads = 256;
    const int64_t C_vec = params.C / 16;
    const int64_t total = params.OH * params.OW * C_vec;
    const int blocks_x = static_cast<int>((total + threads - 1) / threads);
    dim3 grid(blocks_x, 1, static_cast<int>(params.N));
    maxpool_v2_fp8_kernel<__nv_fp8_e4m3><<<grid, threads, 0, stream>>>(input, output, params);
    CUDA_CHECK(cudaGetLastError());
    NVTX_RANGE_POP();
}

void maxpool_v3_fp8_e4m3(const __nv_fp8_e4m3* input, __nv_fp8_e4m3* output, const PoolParams& params, cudaStream_t stream) {
    NVTX_RANGE_PUSH_C("maxpool_v3_fp8e4m3", NVTX_COLOR_MAXPOOL);
    constexpr int BLOCK = 4;
    if (params.OH < BLOCK || params.OH % BLOCK != 0) { NVTX_RANGE_POP(); maxpool_v0_fp8_e4m3(input, output, params, stream); return; }
    const int64_t OH_block = params.OH / BLOCK;
    const int64_t total = OH_block * params.OW * params.C;
    const int threads = 256;
    const int blocks_x = static_cast<int>((total + threads - 1) / threads);
    dim3 grid(blocks_x, 1, static_cast<int>(params.N));
    maxpool_v3_kernel<__nv_fp8_e4m3, BLOCK><<<grid, threads, 0, stream>>>(input, output, params);
    CUDA_CHECK(cudaGetLastError());
    NVTX_RANGE_POP();
}

void maxpool_v4_fp8_e4m3(const __nv_fp8_e4m3* input, __nv_fp8_e4m3* output, const PoolParams& params, cudaStream_t stream) {
    NVTX_RANGE_PUSH_C("maxpool_v4_fp8e4m3", NVTX_COLOR_MAXPOOL);
    const int64_t total = params.OH * params.OW * params.C;
    const int threads = 256;
    const int warps_per_block = threads / 32;
    const int blocks_x = static_cast<int>((total + warps_per_block - 1) / warps_per_block);
    dim3 grid(blocks_x, 1, static_cast<int>(params.N));
    maxpool_v4_kernel<__nv_fp8_e4m3><<<grid, threads, 0, stream>>>(input, output, params);
    CUDA_CHECK(cudaGetLastError());
    NVTX_RANGE_POP();
}

// v5/v6 for fp8: use v1 (vectorized loads don't help for fp8 scalar)
void maxpool_v5_fp8_e4m3(const __nv_fp8_e4m3* input, __nv_fp8_e4m3* output, const PoolParams& params, cudaStream_t stream) {
    NVTX_RANGE_PUSH_C("maxpool_v5_fp8e4m3", NVTX_COLOR_MAXPOOL);
    NVTX_RANGE_POP();
    maxpool_v1_fp8_e4m3(input, output, params, stream);
}
void maxpool_v6_fp8_e4m3(const __nv_fp8_e4m3* input, __nv_fp8_e4m3* output, const PoolParams& params, cudaStream_t stream) {
    NVTX_RANGE_PUSH_C("maxpool_v6_fp8e4m3", NVTX_COLOR_MAXPOOL);
    NVTX_RANGE_POP();
    maxpool_v1_fp8_e4m3(input, output, params, stream);
}

void maxpool_v7_fp8_e4m3(const __nv_fp8_e4m3* input, __nv_fp8_e4m3* output, const PoolParams& params, int mapping, cudaStream_t stream) {
    NVTX_RANGE_PUSH_C("maxpool_v7_fp8e4m3", NVTX_COLOR_MAXPOOL);
    switch (mapping) {
        case 0: { NVTX_RANGE_POP(); maxpool_v0_fp8_e4m3(input, output, params, stream); return; }
        case 1: {
            const int C_groups = static_cast<int>((params.C + 3) / 4);
            const int64_t grid_z_64 = params.N * C_groups;
            if (grid_z_64 > 65535) { NVTX_RANGE_POP(); maxpool_v0_fp8_e4m3(input, output, params, stream); return; }
            dim3 block(8, 8, 4);
            dim3 grid(static_cast<int>((params.OW + 7) / 8), static_cast<int>((params.OH + 7) / 8), static_cast<int>(grid_z_64));
            maxpool_v7_mappingB_kernel<__nv_fp8_e4m3><<<grid, block, 0, stream>>>(input, output, params, C_groups);
            CUDA_CHECK(cudaGetLastError()); break;
        }
        case 2: {
            const int C_groups = static_cast<int>((params.C + 255) / 256);
            const int64_t grid_z_64 = params.N * C_groups;
            if (params.OW > 65535 || params.OH > 65535 || grid_z_64 > 65535) { NVTX_RANGE_POP(); maxpool_v0_fp8_e4m3(input, output, params, stream); return; }
            dim3 block(256, 1, 1);
            dim3 grid(static_cast<int>(params.OW), static_cast<int>(params.OH), static_cast<int>(grid_z_64));
            maxpool_v7_mappingC_kernel<__nv_fp8_e4m3><<<grid, block, 0, stream>>>(input, output, params, C_groups);
            CUDA_CHECK(cudaGetLastError()); break;
        }
        case 3: {
            // v7mD uses half2 vectorized loads — misaligned for 1-byte fp8
            NVTX_RANGE_POP(); maxpool_v0_fp8_e4m3(input, output, params, stream); return;
        }
        default:
            NVTX_RANGE_POP();
            throw std::invalid_argument("unsupported mapping: " + std::to_string(mapping));
    }
    NVTX_RANGE_POP();
}

void maxpool_v8_fp8_e4m3(const __nv_fp8_e4m3* input, __nv_fp8_e4m3* output, const PoolParams& params, cudaStream_t stream) {
    NVTX_RANGE_PUSH_C("maxpool_v8_fp8e4m3", NVTX_COLOR_MAXPOOL);
    maxpool_v0_fp8_e4m3(input, output, params, stream);
    NVTX_RANGE_POP();
}

void maxpool_v9_fp8_e4m3(const __nv_fp8_e4m3* input, __nv_fp8_e4m3* output, const PoolParams& params, cudaStream_t stream) {
    NVTX_RANGE_PUSH_C("maxpool_v9_fp8e4m3", NVTX_COLOR_MAXPOOL);
    maxpool_v0_fp8_e4m3(input, output, params, stream);
    NVTX_RANGE_POP();
}

void maxpool_v10_fp8_e4m3(const __nv_fp8_e4m3* input, __nv_fp8_e4m3* output, const PoolParams& params, cudaStream_t stream) {
    NVTX_RANGE_PUSH_C("maxpool_v10_fp8e4m3", NVTX_COLOR_MAXPOOL);
    maxpool_v0_fp8_e4m3(input, output, params, stream);
    NVTX_RANGE_POP();
}

void maxpool_v11_fp8_e4m3(const __nv_fp8_e4m3* input, __nv_fp8_e4m3* output, const PoolParams& params, cudaStream_t stream) {
    NVTX_RANGE_PUSH_C("maxpool_v11_fp8e4m3", NVTX_COLOR_MAXPOOL);
    maxpool_v4_fp8_e4m3(input, output, params, stream);
    NVTX_RANGE_POP();
}

void maxpool_v12_fp8_e4m3(const __nv_fp8_e4m3* input, __nv_fp8_e4m3* output, const PoolParams& params, cudaStream_t stream) {
    NVTX_RANGE_PUSH_C("maxpool_v12_fp8e4m3", NVTX_COLOR_MAXPOOL);
    maxpool_v0_fp8_e4m3(input, output, params, stream);
    NVTX_RANGE_POP();
}

void maxpool_v13_fp8_e4m3(const __nv_fp8_e4m3* input, __nv_fp8_e4m3* output, const PoolParams& params, cudaStream_t stream) {
    NVTX_RANGE_PUSH_C("maxpool_v13_fp8e4m3", NVTX_COLOR_MAXPOOL);
    NVTX_RANGE_POP();
    maxpool_v0_fp8_e4m3(input, output, params, stream);
}

void maxpool_v14_fp8_e4m3(const __nv_fp8_e4m3* input, __nv_fp8_e4m3* output, const PoolParams& params, cudaStream_t stream) {
    NVTX_RANGE_PUSH_C("maxpool_v14_fp8e4m3", NVTX_COLOR_MAXPOOL);
    bool is_global = (params.kh >= params.H && params.kw >= params.W);
    if (is_global) { maxpool_v13_fp8_e4m3(input, output, params, stream); NVTX_RANGE_POP(); return; }
    int64_t total_output_elems = params.N * params.OH * params.OW * params.C;
    if (total_output_elems < 65536) { maxpool_v10_fp8_e4m3(input, output, params, stream); NVTX_RANGE_POP(); return; }
    if (params.sh == 1 && params.sw == 1 && params.kh == 3 && params.kw == 3) { maxpool_v15_fp8_e4m3(input, output, params, stream); NVTX_RANGE_POP(); return; }
    // Now v2 has uint4 vectorized loads — use it for fp8 too
    if (params.C % 16 == 0) { maxpool_v2_fp8_e4m3(input, output, params, stream); NVTX_RANGE_POP(); return; }
    maxpool_v0_fp8_e4m3(input, output, params, stream);
    NVTX_RANGE_POP();
}

void maxpool_v15_fp8_e4m3(const __nv_fp8_e4m3* input, __nv_fp8_e4m3* output, const PoolParams& params, cudaStream_t stream) {
    NVTX_RANGE_PUSH_C("maxpool_v15_fp8e4m3", NVTX_COLOR_MAXPOOL);
    constexpr int TILE_OH = 8; constexpr int TILE_OW = 8;
    const int blocks_oh = static_cast<int>((params.OH + TILE_OH - 1) / TILE_OH);
    const int blocks_ow = static_cast<int>((params.OW + TILE_OW - 1) / TILE_OW);
    int smem_h = (TILE_OH - 1) * params.sh + (params.kh - 1) * params.dh + 1;
    int smem_w = (TILE_OW - 1) * params.sw + (params.kw - 1) * params.dw + 1;
    size_t smem_bytes = static_cast<size_t>(smem_h) * (smem_w + 1) * sizeof(__nv_fp8_e4m3);
    dim3 block(TILE_OW, TILE_OH);
    dim3 grid(blocks_oh * blocks_ow, static_cast<int>(params.C), static_cast<int>(params.N));
    if (smem_bytes > 49152) {
        CUDA_CHECK(cudaFuncSetAttribute(maxpool_v15_kernel<__nv_fp8_e4m3, true>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem_bytes)));
    }
    maxpool_v15_kernel<__nv_fp8_e4m3, true><<<grid, block, smem_bytes, stream>>>(
        input, output, params, blocks_oh, blocks_ow, smem_h, smem_w, params.divisor_override);
    CUDA_CHECK(cudaGetLastError());
    NVTX_RANGE_POP();
}

// ============================================================================
// FP8 E5M2 MaxPool kernels (identical structure to e4m3, different type)
// ============================================================================

void maxpool_v0_fp8_e5m2(const __nv_fp8_e5m2* input, __nv_fp8_e5m2* output, const PoolParams& params, cudaStream_t stream) {
    NVTX_RANGE_PUSH_C("maxpool_v0_fp8e5m2", NVTX_COLOR_MAXPOOL);
    const int64_t total = params.OH * params.OW * params.C;
    const int threads = 256;
    const int blocks_x = static_cast<int>((total + threads - 1) / threads);
    dim3 grid(blocks_x, 1, static_cast<int>(params.N));
    maxpool_v0_kernel<__nv_fp8_e5m2><<<grid, threads, 0, stream>>>(input, output, params);
    CUDA_CHECK(cudaGetLastError());
    NVTX_RANGE_POP();
}

template <int TILE_OH = 8, int TILE_OW = 8>
__global__ void maxpool_v1_kernel_fp8_e5m2(
    const __nv_fp8_e5m2* __restrict__ input,
    __nv_fp8_e5m2* __restrict__ output,
    const PoolParams params,
    int blocks_oh, int blocks_ow,
    int smem_h, int smem_w)
{
    extern __shared__ float sdata[];
    const int n = blockIdx.z; const int c = blockIdx.y;
    const int tile_idx = blockIdx.x;
    const int tile_oh = tile_idx / blocks_ow;
    const int tile_ow = tile_idx % blocks_ow;
    const int th = threadIdx.y; const int tw = threadIdx.x;
    const int oh = tile_oh * TILE_OH + th;
    const int ow = tile_ow * TILE_OW + tw;
    const int ih_start = tile_oh * TILE_OH * params.sh - params.ph;
    const int iw_start = tile_ow * TILE_OW * params.sw - params.pw;
    const int total_smem = smem_h * smem_w;
    const int tid = th * TILE_OW + tw;
    for (int i = tid; i < total_smem; i += TILE_OH * TILE_OW) {
        const int sih = i / smem_w; const int siw = i % smem_w;
        const int ih = ih_start + sih; const int iw = iw_start + siw;
        if (ih >= 0 && ih < params.H && iw >= 0 && iw < params.W) {
            const int64_t in_idx = ((static_cast<int64_t>(n) * params.H + ih) * params.W + iw) * params.C + c;
            sdata[i] = static_cast<float>(input[in_idx]);
        } else { sdata[i] = -INFINITY; }
    }
    __syncthreads();
    if (oh >= params.OH || ow >= params.OW) return;
    float maxval = -INFINITY;
    for (int kh_i = 0; kh_i < params.kh; ++kh_i) {
        const int sih = th * params.sh + kh_i * params.dh;
        if (sih >= smem_h) continue;
        for (int kw_i = 0; kw_i < params.kw; ++kw_i) {
            const int siw = tw * params.sw + kw_i * params.dw;
            if (siw >= smem_w) continue;
            const float val = sdata[sih * smem_w + siw];
            if (val > maxval) maxval = val;
        }
    }
    const int64_t out_idx = ((static_cast<int64_t>(n) * params.OH + oh) * params.OW + ow) * params.C + c;
    output[out_idx] = static_cast<__nv_fp8_e5m2>(maxval);
}

void maxpool_v1_fp8_e5m2(const __nv_fp8_e5m2* input, __nv_fp8_e5m2* output, const PoolParams& params, cudaStream_t stream) {
    NVTX_RANGE_PUSH_C("maxpool_v1_fp8e5m2", NVTX_COLOR_MAXPOOL);
    constexpr int TILE_OH = 8; constexpr int TILE_OW = 8;
    const int blocks_oh = static_cast<int>((params.OH + TILE_OH - 1) / TILE_OH);
    const int blocks_ow = static_cast<int>((params.OW + TILE_OW - 1) / TILE_OW);
    int smem_h = (TILE_OH - 1) * params.sh + (params.kh - 1) * params.dh + 1;
    int smem_w = (TILE_OW - 1) * params.sw + (params.kw - 1) * params.dw + 1;
    smem_h = min(smem_h, static_cast<int>(params.H + 2 * params.ph));
    smem_w = min(smem_w, static_cast<int>(params.W + 2 * params.pw));
    dim3 block(TILE_OW, TILE_OH);
    dim3 grid(blocks_oh * blocks_ow, static_cast<int>(params.C), static_cast<int>(params.N));
    size_t smem_bytes = static_cast<size_t>(smem_h) * smem_w * sizeof(float);
    maxpool_v1_kernel_fp8_e5m2<TILE_OH, TILE_OW><<<grid, block, smem_bytes, stream>>>(
        input, output, params, blocks_oh, blocks_ow, smem_h, smem_w);
    CUDA_CHECK(cudaGetLastError());
    NVTX_RANGE_POP();
}

void maxpool_v2_fp8_e5m2(const __nv_fp8_e5m2* input, __nv_fp8_e5m2* output, const PoolParams& params, cudaStream_t stream) {
    NVTX_RANGE_PUSH_C("maxpool_v2_fp8e5m2", NVTX_COLOR_MAXPOOL);
    if (params.C % 16 != 0) { NVTX_RANGE_POP(); maxpool_v0_fp8_e5m2(input, output, params, stream); return; }
    const int threads = 256;
    const int64_t C_vec = params.C / 16;
    const int64_t total = params.OH * params.OW * C_vec;
    const int blocks_x = static_cast<int>((total + threads - 1) / threads);
    dim3 grid(blocks_x, 1, static_cast<int>(params.N));
    maxpool_v2_fp8_kernel<__nv_fp8_e5m2><<<grid, threads, 0, stream>>>(input, output, params);
    CUDA_CHECK(cudaGetLastError());
    NVTX_RANGE_POP();
}
void maxpool_v3_fp8_e5m2(const __nv_fp8_e5m2* input, __nv_fp8_e5m2* output, const PoolParams& params, cudaStream_t stream) {
    NVTX_RANGE_PUSH_C("maxpool_v3_fp8e5m2", NVTX_COLOR_MAXPOOL);
    constexpr int BLOCK = 4;
    if (params.OH < BLOCK || params.OH % BLOCK != 0) { NVTX_RANGE_POP(); maxpool_v0_fp8_e5m2(input, output, params, stream); return; }
    const int64_t OH_block = params.OH / BLOCK;
    const int64_t total = OH_block * params.OW * params.C;
    const int threads = 256;
    const int blocks_x = static_cast<int>((total + threads - 1) / threads);
    dim3 grid(blocks_x, 1, static_cast<int>(params.N));
    maxpool_v3_kernel<__nv_fp8_e5m2, BLOCK><<<grid, threads, 0, stream>>>(input, output, params);
    CUDA_CHECK(cudaGetLastError());
    NVTX_RANGE_POP();
}
void maxpool_v4_fp8_e5m2(const __nv_fp8_e5m2* input, __nv_fp8_e5m2* output, const PoolParams& params, cudaStream_t stream) {
    NVTX_RANGE_PUSH_C("maxpool_v4_fp8e5m2", NVTX_COLOR_MAXPOOL);
    const int64_t total = params.OH * params.OW * params.C;
    const int threads = 256;
    const int warps_per_block = threads / 32;
    const int blocks_x = static_cast<int>((total + warps_per_block - 1) / warps_per_block);
    dim3 grid(blocks_x, 1, static_cast<int>(params.N));
    maxpool_v4_kernel<__nv_fp8_e5m2><<<grid, threads, 0, stream>>>(input, output, params);
    CUDA_CHECK(cudaGetLastError());
    NVTX_RANGE_POP();
}
void maxpool_v5_fp8_e5m2(const __nv_fp8_e5m2* input, __nv_fp8_e5m2* output, const PoolParams& params, cudaStream_t stream) {
    NVTX_RANGE_PUSH_C("maxpool_v5_fp8e5m2", NVTX_COLOR_MAXPOOL);
    NVTX_RANGE_POP(); maxpool_v1_fp8_e5m2(input, output, params, stream);
}
void maxpool_v6_fp8_e5m2(const __nv_fp8_e5m2* input, __nv_fp8_e5m2* output, const PoolParams& params, cudaStream_t stream) {
    NVTX_RANGE_PUSH_C("maxpool_v6_fp8e5m2", NVTX_COLOR_MAXPOOL);
    NVTX_RANGE_POP(); maxpool_v1_fp8_e5m2(input, output, params, stream);
}
void maxpool_v7_fp8_e5m2(const __nv_fp8_e5m2* input, __nv_fp8_e5m2* output, const PoolParams& params, int mapping, cudaStream_t stream) {
    NVTX_RANGE_PUSH_C("maxpool_v7_fp8e5m2", NVTX_COLOR_MAXPOOL);
    switch (mapping) {
        case 0: { NVTX_RANGE_POP(); maxpool_v0_fp8_e5m2(input, output, params, stream); return; }
        case 1: {
            const int C_groups = static_cast<int>((params.C + 3) / 4);
            const int64_t grid_z_64 = params.N * C_groups;
            if (grid_z_64 > 65535) { NVTX_RANGE_POP(); maxpool_v0_fp8_e5m2(input, output, params, stream); return; }
            dim3 block(8, 8, 4);
            dim3 grid(static_cast<int>((params.OW + 7) / 8), static_cast<int>((params.OH + 7) / 8), static_cast<int>(grid_z_64));
            maxpool_v7_mappingB_kernel<__nv_fp8_e5m2><<<grid, block, 0, stream>>>(input, output, params, C_groups);
            CUDA_CHECK(cudaGetLastError()); break;
        }
        case 2: {
            const int C_groups = static_cast<int>((params.C + 255) / 256);
            const int64_t grid_z_64 = params.N * C_groups;
            if (params.OW > 65535 || params.OH > 65535 || grid_z_64 > 65535) { NVTX_RANGE_POP(); maxpool_v0_fp8_e5m2(input, output, params, stream); return; }
            dim3 block(256, 1, 1);
            dim3 grid(static_cast<int>(params.OW), static_cast<int>(params.OH), static_cast<int>(grid_z_64));
            maxpool_v7_mappingC_kernel<__nv_fp8_e5m2><<<grid, block, 0, stream>>>(input, output, params, C_groups);
            CUDA_CHECK(cudaGetLastError()); break;
        }
        case 3: {
            // v7mD uses half2 vectorized loads — misaligned for 1-byte fp8
            NVTX_RANGE_POP(); maxpool_v0_fp8_e5m2(input, output, params, stream); return;
        }
        default:
            NVTX_RANGE_POP();
            throw std::invalid_argument("unsupported mapping: " + std::to_string(mapping));
    }
    NVTX_RANGE_POP();
}
void maxpool_v8_fp8_e5m2(const __nv_fp8_e5m2* input, __nv_fp8_e5m2* output, const PoolParams& params, cudaStream_t stream) {
    NVTX_RANGE_PUSH_C("maxpool_v8_fp8e5m2", NVTX_COLOR_MAXPOOL);
    maxpool_v0_fp8_e5m2(input, output, params, stream);
    NVTX_RANGE_POP();
}
void maxpool_v9_fp8_e5m2(const __nv_fp8_e5m2* input, __nv_fp8_e5m2* output, const PoolParams& params, cudaStream_t stream) {
    NVTX_RANGE_PUSH_C("maxpool_v9_fp8e5m2", NVTX_COLOR_MAXPOOL);
    maxpool_v0_fp8_e5m2(input, output, params, stream);
    NVTX_RANGE_POP();
}
void maxpool_v10_fp8_e5m2(const __nv_fp8_e5m2* input, __nv_fp8_e5m2* output, const PoolParams& params, cudaStream_t stream) {
    NVTX_RANGE_PUSH_C("maxpool_v10_fp8e5m2", NVTX_COLOR_MAXPOOL);
    maxpool_v0_fp8_e5m2(input, output, params, stream);
    NVTX_RANGE_POP();
}
void maxpool_v11_fp8_e5m2(const __nv_fp8_e5m2* input, __nv_fp8_e5m2* output, const PoolParams& params, cudaStream_t stream) {
    NVTX_RANGE_PUSH_C("maxpool_v11_fp8e5m2", NVTX_COLOR_MAXPOOL);
    maxpool_v4_fp8_e5m2(input, output, params, stream);
    NVTX_RANGE_POP();
}
void maxpool_v12_fp8_e5m2(const __nv_fp8_e5m2* input, __nv_fp8_e5m2* output, const PoolParams& params, cudaStream_t stream) {
    NVTX_RANGE_PUSH_C("maxpool_v12_fp8e5m2", NVTX_COLOR_MAXPOOL);
    maxpool_v0_fp8_e5m2(input, output, params, stream);
    NVTX_RANGE_POP();
}
void maxpool_v13_fp8_e5m2(const __nv_fp8_e5m2* input, __nv_fp8_e5m2* output, const PoolParams& params, cudaStream_t stream) {
    NVTX_RANGE_PUSH_C("maxpool_v13_fp8e5m2", NVTX_COLOR_MAXPOOL);
    NVTX_RANGE_POP(); maxpool_v0_fp8_e5m2(input, output, params, stream);
}
void maxpool_v14_fp8_e5m2(const __nv_fp8_e5m2* input, __nv_fp8_e5m2* output, const PoolParams& params, cudaStream_t stream) {
    NVTX_RANGE_PUSH_C("maxpool_v14_fp8e5m2", NVTX_COLOR_MAXPOOL);
    bool is_global = (params.kh >= params.H && params.kw >= params.W);
    if (is_global) { maxpool_v13_fp8_e5m2(input, output, params, stream); NVTX_RANGE_POP(); return; }
    int64_t total_output_elems = params.N * params.OH * params.OW * params.C;
    if (total_output_elems < 65536) { maxpool_v10_fp8_e5m2(input, output, params, stream); NVTX_RANGE_POP(); return; }
    if (params.sh == 1 && params.sw == 1 && params.kh == 3 && params.kw == 3) { maxpool_v15_fp8_e5m2(input, output, params, stream); NVTX_RANGE_POP(); return; }
    // Now v2 has uint4 vectorized loads — use it for fp8 too
    if (params.C % 16 == 0) { maxpool_v2_fp8_e5m2(input, output, params, stream); NVTX_RANGE_POP(); return; }
    maxpool_v0_fp8_e5m2(input, output, params, stream);
    NVTX_RANGE_POP();
}
void maxpool_v15_fp8_e5m2(const __nv_fp8_e5m2* input, __nv_fp8_e5m2* output, const PoolParams& params, cudaStream_t stream) {
    NVTX_RANGE_PUSH_C("maxpool_v15_fp8e5m2", NVTX_COLOR_MAXPOOL);
    constexpr int TILE_OH = 8; constexpr int TILE_OW = 8;
    const int blocks_oh = static_cast<int>((params.OH + TILE_OH - 1) / TILE_OH);
    const int blocks_ow = static_cast<int>((params.OW + TILE_OW - 1) / TILE_OW);
    int smem_h = (TILE_OH - 1) * params.sh + (params.kh - 1) * params.dh + 1;
    int smem_w = (TILE_OW - 1) * params.sw + (params.kw - 1) * params.dw + 1;
    size_t smem_bytes = static_cast<size_t>(smem_h) * (smem_w + 1) * sizeof(__nv_fp8_e5m2);
    dim3 block(TILE_OW, TILE_OH);
    dim3 grid(blocks_oh * blocks_ow, static_cast<int>(params.C), static_cast<int>(params.N));
    if (smem_bytes > 49152) {
        CUDA_CHECK(cudaFuncSetAttribute(maxpool_v15_kernel<__nv_fp8_e5m2, true>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem_bytes)));
    }
    maxpool_v15_kernel<__nv_fp8_e5m2, true><<<grid, block, smem_bytes, stream>>>(
        input, output, params, blocks_oh, blocks_ow, smem_h, smem_w, params.divisor_override);
    CUDA_CHECK(cudaGetLastError());
    NVTX_RANGE_POP();
}
