#include "pooling.cuh"
#include <mutex>
#include <unordered_map>

// ============================================================================
// Multi-dtype avgpool kernel implementations for bf16, int8, int16, fp8
// ============================================================================

// ============================================================================
// BF16 (nv_bfloat16) AvgPool kernels
// ============================================================================

// --- v0: template instantiation ---
void avgpool_v0(const nv_bfloat16* input, nv_bfloat16* output, const AvgPoolParams& params, cudaStream_t stream) {
    NVTX_RANGE_PUSH_C("avgpool_v0_bf16", NVTX_COLOR_AVGPOOL);
    const int64_t total = params.OH * params.OW * params.C;
    const int threads = 256;
    const int blocks_x = static_cast<int>((total + threads - 1) / threads);
    dim3 grid(blocks_x, 1, static_cast<int>(params.N));
    avgpool_v0_kernel<nv_bfloat16><<<grid, threads, 0, stream>>>(input, output, params);
    CUDA_CHECK(cudaGetLastError());
    NVTX_RANGE_POP();
}

// --- v1: explicit kernel (float shared memory) ---
template <int TILE_OH = 8, int TILE_OW = 8>
__global__ void avgpool_v1_kernel_bf16(
    const nv_bfloat16* __restrict__ input,
    nv_bfloat16* __restrict__ output,
    const AvgPoolParams params,
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
            sdata[i] = 0.0f;
        }
    }
    __syncthreads();
    if (oh >= params.OH || ow >= params.OW) return;
    float sum = 0.0f;
    int count = 0;
    for (int kh_i = 0; kh_i < params.kh; ++kh_i) {
        const int sih = th * params.sh + kh_i * params.dh;
        if (sih >= smem_h) continue;
        const int64_t ih = static_cast<int64_t>(ih_start) + sih;
        for (int kw_i = 0; kw_i < params.kw; ++kw_i) {
            const int siw = tw * params.sw + kw_i * params.dw;
            if (siw >= smem_w) continue;
            const int64_t iw = static_cast<int64_t>(iw_start) + siw;
            const bool ih_in = (ih >= 0 && ih < params.H);
            const bool iw_in = (iw >= 0 && iw < params.W);
            if (ih_in && iw_in) {
                sum += sdata[sih * smem_w + siw];
                count++;
            } else if (params.count_include_pad) {
                const bool ih_in_pad = (ih >= -static_cast<int64_t>(params.ph) &&
                                        ih < params.H + static_cast<int64_t>(params.ph));
                const bool iw_in_pad = (iw >= -static_cast<int64_t>(params.pw) &&
                                        iw < params.W + static_cast<int64_t>(params.pw));
                if (ih_in_pad && iw_in_pad) { count++; }
            }
        }
    }
    float divisor;
    if (params.divisor_override > 0) {
        divisor = static_cast<float>(params.divisor_override);
    } else {
        divisor = static_cast<float>(count);
    }
    const int64_t out_idx = ((static_cast<int64_t>(n) * params.OH + oh) * params.OW + ow) * params.C + c;
    output[out_idx] = __float2bfloat16((divisor > 0.0f) ? (sum / divisor) : 0.0f);
}

void avgpool_v1(const nv_bfloat16* input, nv_bfloat16* output, const AvgPoolParams& params, cudaStream_t stream) {
    NVTX_RANGE_PUSH_C("avgpool_v1_bf16", NVTX_COLOR_AVGPOOL);
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
    avgpool_v1_kernel_bf16<TILE_OH, TILE_OW><<<grid, block, smem_bytes, stream>>>(
        input, output, params, blocks_oh, blocks_ow, smem_h, smem_w);
    CUDA_CHECK(cudaGetLastError());
    NVTX_RANGE_POP();
}

// --- v2: vectorized loads with nv_bfloat162 ---
template <int VEC>
__global__ void avgpool_v2_kernel_bf16(
    const nv_bfloat16* __restrict__ input,
    nv_bfloat16* __restrict__ output,
    const AvgPoolParams params)
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
    float sum[VEC];
    #pragma unroll
    for (int v = 0; v < VEC; ++v) sum[v] = 0.0f;
    int count = 0;
    for (int kh_i = 0; kh_i < params.kh; ++kh_i) {
        const int64_t ih = oh * params.sh - params.ph + static_cast<int64_t>(kh_i) * params.dh;
        const bool ih_in = (ih >= 0 && ih < params.H);
        for (int kw_i = 0; kw_i < params.kw; ++kw_i) {
            const int64_t iw = ow * params.sw - params.pw + static_cast<int64_t>(kw_i) * params.dw;
            const bool iw_in = (iw >= 0 && iw < params.W);
            if (ih_in && iw_in) {
                const int64_t in_idx = ((n * params.H + ih) * params.W + iw) * params.C + c;
                nv_bfloat162 vec = *reinterpret_cast<const nv_bfloat162*>(&input[in_idx]);
                sum[0] += __low2float(vec);
                sum[1] += __high2float(vec);
                count++;
            } else if (params.count_include_pad) {
                const bool ih_in_pad = (ih >= -static_cast<int64_t>(params.ph) &&
                                        ih < params.H + static_cast<int64_t>(params.ph));
                const bool iw_in_pad = (iw >= -static_cast<int64_t>(params.pw) &&
                                        iw < params.W + static_cast<int64_t>(params.pw));
                if (ih_in_pad && iw_in_pad) { count++; }
            }
        }
    }
    float divisor;
    if (params.divisor_override > 0) {
        divisor = static_cast<float>(params.divisor_override);
    } else {
        divisor = static_cast<float>(count);
    }
    const int64_t out_idx = ((n * params.OH + oh) * params.OW + ow) * params.C + c;
    float r0 = (divisor > 0.0f) ? (sum[0] / divisor) : 0.0f;
    float r1 = (divisor > 0.0f) ? (sum[1] / divisor) : 0.0f;
    nv_bfloat162 out_vec = __floats2bfloat162_rn(r0, r1);
    *reinterpret_cast<nv_bfloat162*>(&output[out_idx]) = out_vec;
}

void avgpool_v2(const nv_bfloat16* input, nv_bfloat16* output, const AvgPoolParams& params, cudaStream_t stream) {
    NVTX_RANGE_PUSH_C("avgpool_v2_bf16", NVTX_COLOR_AVGPOOL);
    constexpr int VEC = 2;
    if (params.C % VEC != 0) { NVTX_RANGE_POP(); avgpool_v0(input, output, params, stream); return; }
    const int64_t C_vec = params.C / VEC;
    const int64_t total = params.OH * params.OW * C_vec;
    const int threads = 256;
    const int blocks_x = static_cast<int>((total + threads - 1) / threads);
    dim3 grid(blocks_x, 1, static_cast<int>(params.N));
    avgpool_v2_kernel_bf16<VEC><<<grid, threads, 0, stream>>>(input, output, params);
    CUDA_CHECK(cudaGetLastError());
    NVTX_RANGE_POP();
}

// --- v3: register blocking (template) ---
void avgpool_v3(const nv_bfloat16* input, nv_bfloat16* output, const AvgPoolParams& params, cudaStream_t stream) {
    NVTX_RANGE_PUSH_C("avgpool_v3_bf16", NVTX_COLOR_AVGPOOL);
    constexpr int BLOCK = 4;
    if (params.OH < BLOCK || params.OH % BLOCK != 0) { NVTX_RANGE_POP(); avgpool_v0(input, output, params, stream); return; }
    const int64_t OH_block = params.OH / BLOCK;
    const int64_t total = OH_block * params.OW * params.C;
    const int threads = 256;
    const int blocks_x = static_cast<int>((total + threads - 1) / threads);
    dim3 grid(blocks_x, 1, static_cast<int>(params.N));
    avgpool_v3_kernel<nv_bfloat16, BLOCK><<<grid, threads, 0, stream>>>(input, output, params);
    CUDA_CHECK(cudaGetLastError());
    NVTX_RANGE_POP();
}

// --- v4: warp-level reduce (template) ---
void avgpool_v4(const nv_bfloat16* input, nv_bfloat16* output, const AvgPoolParams& params, cudaStream_t stream) {
    NVTX_RANGE_PUSH_C("avgpool_v4_bf16", NVTX_COLOR_AVGPOOL);
    const int64_t total = params.OH * params.OW * params.C;
    const int threads = 256;
    const int warps_per_block = threads / 32;
    const int blocks_x = static_cast<int>((total + warps_per_block - 1) / warps_per_block);
    dim3 grid(blocks_x, 1, static_cast<int>(params.N));
    avgpool_v4_kernel<nv_bfloat16><<<grid, threads, 0, stream>>>(input, output, params);
    CUDA_CHECK(cudaGetLastError());
    NVTX_RANGE_POP();
}

// --- v5: double buffer (explicit, follows half pattern) ---
template <int TILE_OH = 8, int TILE_OW = 8>
__global__ void avgpool_v5_kernel_bf16(
    const nv_bfloat16* __restrict__ input,
    nv_bfloat16* __restrict__ output,
    const AvgPoolParams params,
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

    // Load c0
    for (int i = tid; i < total_smem; i += nthreads) {
        const int sih = i / smem_w; const int siw = i % smem_w;
        const int ih = ih_start + sih; const int iw = iw_start + siw;
        if (ih >= 0 && ih < params.H && iw >= 0 && iw < params.W) {
            const int64_t in_idx = ((static_cast<int64_t>(n) * params.H + ih) * params.W + iw) * params.C + c0;
            buf[0][i] = __bfloat162float(input[in_idx]);
        } else { buf[0][i] = 0.0f; }
    }
    __syncthreads();

    // Compute c0
    float sum_c0 = 0.0f; int count_c0 = 0;
    if (oh < params.OH && ow < params.OW) {
        for (int kh_i = 0; kh_i < params.kh; ++kh_i) {
            const int sih = th * params.sh + kh_i * params.dh;
            if (sih >= smem_h) continue;
            const int64_t ih = static_cast<int64_t>(ih_start) + sih;
            for (int kw_i = 0; kw_i < params.kw; ++kw_i) {
                const int siw = tw * params.sw + kw_i * params.dw;
                if (siw >= smem_w) continue;
                const int64_t iw = static_cast<int64_t>(iw_start) + siw;
                const bool ih_in = (ih >= 0 && ih < params.H);
                const bool iw_in = (iw >= 0 && iw < params.W);
                if (ih_in && iw_in) { sum_c0 += buf[0][sih * smem_w + siw]; count_c0++; }
                else if (params.count_include_pad) {
                    const bool ih_in_pad = (ih >= -static_cast<int64_t>(params.ph) && ih < params.H + static_cast<int64_t>(params.ph));
                    const bool iw_in_pad = (iw >= -static_cast<int64_t>(params.pw) && iw < params.W + static_cast<int64_t>(params.pw));
                    if (ih_in_pad && iw_in_pad) count_c0++;
                }
            }
        }
    }
    if (oh < params.OH && ow < params.OW) {
        float divisor = (params.divisor_override > 0) ? static_cast<float>(params.divisor_override) : static_cast<float>(count_c0);
        const int64_t out_idx = ((static_cast<int64_t>(n) * params.OH + oh) * params.OW + ow) * params.C + c0;
        output[out_idx] = __float2bfloat16((divisor > 0.0f) ? (sum_c0 / divisor) : 0.0f);
    }

    // Load & compute c1
    if (has_c1) {
        for (int i = tid; i < total_smem; i += nthreads) {
            const int sih = i / smem_w; const int siw = i % smem_w;
            const int ih = ih_start + sih; const int iw = iw_start + siw;
            if (ih >= 0 && ih < params.H && iw >= 0 && iw < params.W) {
                const int64_t in_idx = ((static_cast<int64_t>(n) * params.H + ih) * params.W + iw) * params.C + c1;
                buf[1][i] = __bfloat162float(input[in_idx]);
            } else { buf[1][i] = 0.0f; }
        }
        __syncthreads();
        float sum_c1 = 0.0f; int count_c1 = 0;
        if (oh < params.OH && ow < params.OW) {
            for (int kh_i = 0; kh_i < params.kh; ++kh_i) {
                const int sih = th * params.sh + kh_i * params.dh;
                if (sih >= smem_h) continue;
                const int64_t ih = static_cast<int64_t>(ih_start) + sih;
                for (int kw_i = 0; kw_i < params.kw; ++kw_i) {
                    const int siw = tw * params.sw + kw_i * params.dw;
                    if (siw >= smem_w) continue;
                    const int64_t iw = static_cast<int64_t>(iw_start) + siw;
                    const bool ih_in = (ih >= 0 && ih < params.H);
                    const bool iw_in = (iw >= 0 && iw < params.W);
                    if (ih_in && iw_in) { sum_c1 += buf[1][sih * smem_w + siw]; count_c1++; }
                    else if (params.count_include_pad) {
                        const bool ih_in_pad = (ih >= -static_cast<int64_t>(params.ph) && ih < params.H + static_cast<int64_t>(params.ph));
                        const bool iw_in_pad = (iw >= -static_cast<int64_t>(params.pw) && iw < params.W + static_cast<int64_t>(params.pw));
                        if (ih_in_pad && iw_in_pad) count_c1++;
                    }
                }
            }
        }
        if (oh < params.OH && ow < params.OW) {
            float divisor = (params.divisor_override > 0) ? static_cast<float>(params.divisor_override) : static_cast<float>(count_c1);
            const int64_t out_idx = ((static_cast<int64_t>(n) * params.OH + oh) * params.OW + ow) * params.C + c1;
            output[out_idx] = __float2bfloat16((divisor > 0.0f) ? (sum_c1 / divisor) : 0.0f);
        }
    }
}

void avgpool_v5(const nv_bfloat16* input, nv_bfloat16* output, const AvgPoolParams& params, cudaStream_t stream) {
    NVTX_RANGE_PUSH_C("avgpool_v5_bf16", NVTX_COLOR_AVGPOOL);
    constexpr int TILE_OH = 8; constexpr int TILE_OW = 8;
    if (params.C < 2) { NVTX_RANGE_POP(); avgpool_v1(input, output, params, stream); return; }
    const int blocks_oh = static_cast<int>((params.OH + TILE_OH - 1) / TILE_OH);
    const int blocks_ow = static_cast<int>((params.OW + TILE_OW - 1) / TILE_OW);
    int smem_h = (TILE_OH - 1) * params.sh + (params.kh - 1) * params.dh + 1;
    int smem_w = (TILE_OW - 1) * params.sw + (params.kw - 1) * params.dw + 1;
    smem_h = min(smem_h, static_cast<int>(params.H + 2 * params.ph));
    smem_w = min(smem_w, static_cast<int>(params.W + 2 * params.pw));
    size_t smem_bytes = 2 * static_cast<size_t>(smem_h) * smem_w * sizeof(float);
    int smem_limit = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&smem_limit, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0));
    if (smem_bytes > static_cast<size_t>(smem_limit)) { NVTX_RANGE_POP(); avgpool_v1(input, output, params, stream); return; }
    const int c_pairs = static_cast<int>((params.C + 1) / 2);
    dim3 block(TILE_OW, TILE_OH);
    dim3 grid(blocks_oh * blocks_ow, c_pairs, static_cast<int>(params.N));
    if (smem_bytes > 49152) {
        CUDA_CHECK(cudaFuncSetAttribute(avgpool_v5_kernel_bf16<TILE_OH, TILE_OW>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem_bytes)));
    }
    avgpool_v5_kernel_bf16<TILE_OH, TILE_OW><<<grid, block, smem_bytes, stream>>>(
        input, output, params, blocks_oh, blocks_ow, smem_h, smem_w);
    CUDA_CHECK(cudaGetLastError());
    NVTX_RANGE_POP();
}

// --- v6: warp specialization (explicit) ---
template <int TILE_OH = 8, int TILE_OW = 8, int NUM_LOAD_WARPS = 2, int NUM_COMPUTE_WARPS = 6>
__global__ void avgpool_v6_kernel_bf16(
    const nv_bfloat16* __restrict__ input,
    nv_bfloat16* __restrict__ output,
    const AvgPoolParams params,
    int blocks_oh, int blocks_ow,
    int smem_h, int smem_w)
{
    constexpr int NUM_LOAD_THREADS = NUM_LOAD_WARPS * 32;
    constexpr int NUM_COMPUTE_THREADS = NUM_COMPUTE_WARPS * 32;
    constexpr int TILE_SIZE = TILE_OH * TILE_OW;
    extern __shared__ float sdata[];
    const int n = blockIdx.z; const int c = blockIdx.y;
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
            const int sih = i / smem_w; const int siw = i % smem_w;
            const int ih = ih_start + sih; const int iw = iw_start + siw;
            if (ih >= 0 && ih < params.H && iw >= 0 && iw < params.W) {
                const int64_t in_idx = ((static_cast<int64_t>(n) * params.H + ih) * params.W + iw) * params.C + c;
                sdata[i] = __bfloat162float(input[in_idx]);
            } else { sdata[i] = 0.0f; }
        }
    }
    __syncthreads();
    if (!is_load_warp) {
        const int compute_tid = tid - NUM_LOAD_THREADS;
        for (int i = compute_tid; i < TILE_SIZE; i += NUM_COMPUTE_THREADS) {
            const int th = i / TILE_OW; const int tw = i % TILE_OW;
            const int oh = tile_oh * TILE_OH + th; const int ow = tile_ow * TILE_OW + tw;
            if (oh >= params.OH || ow >= params.OW) continue;
            float sum = 0.0f; int count = 0;
            for (int kh_i = 0; kh_i < params.kh; ++kh_i) {
                const int sih = th * params.sh + kh_i * params.dh;
                if (sih >= smem_h) continue;
                const int64_t ih = static_cast<int64_t>(ih_start) + sih;
                for (int kw_i = 0; kw_i < params.kw; ++kw_i) {
                    const int siw = tw * params.sw + kw_i * params.dw;
                    if (siw >= smem_w) continue;
                    const int64_t iw = static_cast<int64_t>(iw_start) + siw;
                    const bool ih_in = (ih >= 0 && ih < params.H);
                    const bool iw_in = (iw >= 0 && iw < params.W);
                    if (ih_in && iw_in) { sum += sdata[sih * smem_w + siw]; count++; }
                    else if (params.count_include_pad) {
                        const bool ih_in_pad = (ih >= -static_cast<int64_t>(params.ph) && ih < params.H + static_cast<int64_t>(params.ph));
                        const bool iw_in_pad = (iw >= -static_cast<int64_t>(params.pw) && iw < params.W + static_cast<int64_t>(params.pw));
                        if (ih_in_pad && iw_in_pad) count++;
                    }
                }
            }
            float divisor = (params.divisor_override > 0) ? static_cast<float>(params.divisor_override) : static_cast<float>(count);
            const int64_t out_idx = ((static_cast<int64_t>(n) * params.OH + oh) * params.OW + ow) * params.C + c;
            output[out_idx] = __float2bfloat16((divisor > 0.0f) ? (sum / divisor) : 0.0f);
        }
    }
}

void avgpool_v6(const nv_bfloat16* input, nv_bfloat16* output, const AvgPoolParams& params, cudaStream_t stream) {
    NVTX_RANGE_PUSH_C("avgpool_v6_bf16", NVTX_COLOR_AVGPOOL);
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
    if (smem_bytes > static_cast<size_t>(smem_limit)) { NVTX_RANGE_POP(); avgpool_v1(input, output, params, stream); return; }
    dim3 block(BLOCK_SIZE);
    dim3 grid(blocks_oh * blocks_ow, static_cast<int>(params.C), static_cast<int>(params.N));
    if (smem_bytes > 49152) {
        CUDA_CHECK(cudaFuncSetAttribute(avgpool_v6_kernel_bf16<TILE_OH, TILE_OW>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem_bytes)));
    }
    avgpool_v6_kernel_bf16<TILE_OH, TILE_OW><<<grid, block, smem_bytes, stream>>>(
        input, output, params, blocks_oh, blocks_ow, smem_h, smem_w);
    CUDA_CHECK(cudaGetLastError());
    NVTX_RANGE_POP();
}

// --- v7: alternative mappings ---
void avgpool_v7(const nv_bfloat16* input, nv_bfloat16* output, const AvgPoolParams& params, int mapping, cudaStream_t stream) {
    NVTX_RANGE_PUSH_C("avgpool_v7_bf16", NVTX_COLOR_AVGPOOL);
    switch (mapping) {
        case 0: { NVTX_RANGE_POP(); avgpool_v0(input, output, params, stream); return; }
        case 1: {
            const int C_groups = static_cast<int>((params.C + 3) / 4);
            const int64_t grid_z_64 = params.N * C_groups;
            if (grid_z_64 > 65535) { NVTX_RANGE_POP(); avgpool_v0(input, output, params, stream); return; }
            dim3 block(8, 8, 4);
            dim3 grid(static_cast<int>((params.OW + 7) / 8), static_cast<int>((params.OH + 7) / 8), static_cast<int>(grid_z_64));
            avgpool_v7_mappingB_kernel<nv_bfloat16><<<grid, block, 0, stream>>>(input, output, params, C_groups);
            CUDA_CHECK(cudaGetLastError()); break;
        }
        case 2: {
            const int C_groups = static_cast<int>((params.C + 255) / 256);
            const int64_t grid_z_64 = params.N * C_groups;
            if (params.OW > 65535 || params.OH > 65535 || grid_z_64 > 65535) { NVTX_RANGE_POP(); avgpool_v0(input, output, params, stream); return; }
            dim3 block(256, 1, 1);
            dim3 grid(static_cast<int>(params.OW), static_cast<int>(params.OH), static_cast<int>(grid_z_64));
            avgpool_v7_mappingC_kernel<nv_bfloat16><<<grid, block, 0, stream>>>(input, output, params, C_groups);
            CUDA_CHECK(cudaGetLastError()); break;
        }
        case 3: {
            if (params.C % 4 != 0) { NVTX_RANGE_POP(); avgpool_v0(input, output, params, stream); return; }
            const int C_groups = static_cast<int>((params.C + 31) / 32);
            const int64_t grid_z_64 = params.N * C_groups;
            if (grid_z_64 > 65535) { NVTX_RANGE_POP(); avgpool_v0(input, output, params, stream); return; }
            dim3 block(32, 8, 1);
            dim3 grid(static_cast<int>((params.OW + 3) / 4), static_cast<int>((params.OH + 3) / 4), static_cast<int>(grid_z_64));
            avgpool_v7_mappingD_kernel<nv_bfloat16><<<grid, block, 0, stream>>>(input, output, params, C_groups);
            CUDA_CHECK(cudaGetLastError()); break;
        }
        default:
            NVTX_RANGE_POP();
            throw std::invalid_argument("unsupported mapping: " + std::to_string(mapping));
    }
    NVTX_RANGE_POP();
}

// --- v8: auto-tuned tiling (fallback to v2) ---
void avgpool_v8(const nv_bfloat16* input, nv_bfloat16* output, const AvgPoolParams& params, cudaStream_t stream) {
    NVTX_RANGE_PUSH_C("avgpool_v8_bf16", NVTX_COLOR_AVGPOOL);
    avgpool_v2(input, output, params, stream);
    NVTX_RANGE_POP();
}

// --- v9: TMA warp-specialized pipeline (fallback to v2) ---
void avgpool_v9(const nv_bfloat16* input, nv_bfloat16* output, const AvgPoolParams& params, cudaStream_t stream) {
    NVTX_RANGE_PUSH_C("avgpool_v9_bf16", NVTX_COLOR_AVGPOOL);
    avgpool_v2(input, output, params, stream);
    NVTX_RANGE_POP();
}

// --- v10: persistent kernel (fallback to v2) ---
void avgpool_v10(const nv_bfloat16* input, nv_bfloat16* output, const AvgPoolParams& params, cudaStream_t stream) {
    NVTX_RANGE_PUSH_C("avgpool_v10_bf16", NVTX_COLOR_AVGPOOL);
    avgpool_v2(input, output, params, stream);
    NVTX_RANGE_POP();
}

// --- v11: warp-shuffle reduction (fallback to v4) ---
void avgpool_v11(const nv_bfloat16* input, nv_bfloat16* output, const AvgPoolParams& params, cudaStream_t stream) {
    NVTX_RANGE_PUSH_C("avgpool_v11_bf16", NVTX_COLOR_AVGPOOL);
    avgpool_v4(input, output, params, stream);
    NVTX_RANGE_POP();
}

// --- v12: L2-aware persistent (fallback to v2) ---
void avgpool_v12(const nv_bfloat16* input, nv_bfloat16* output, const AvgPoolParams& params, cudaStream_t stream) {
    NVTX_RANGE_PUSH_C("avgpool_v12_bf16", NVTX_COLOR_AVGPOOL);
    avgpool_v2(input, output, params, stream);
    NVTX_RANGE_POP();
}

// --- v13: channel-vectorized warp (fallback to v2) ---
void avgpool_v13(const nv_bfloat16* input, nv_bfloat16* output, const AvgPoolParams& params, cudaStream_t stream) {
    NVTX_RANGE_PUSH_C("avgpool_v13_bf16", NVTX_COLOR_AVGPOOL);
    avgpool_v2(input, output, params, stream);
    NVTX_RANGE_POP();
}

// --- v14: adaptive dispatcher ---
void avgpool_v14(const nv_bfloat16* input, nv_bfloat16* output, const AvgPoolParams& params, cudaStream_t stream) {
    NVTX_RANGE_PUSH_C("avgpool_v14_bf16", NVTX_COLOR_AVGPOOL);
    bool is_global = (params.kh >= params.H && params.kw >= params.W);
    if (is_global) { avgpool_v13(input, output, params, stream); NVTX_RANGE_POP(); return; }
    int64_t total_output_elems = params.N * params.OH * params.OW * params.C;
    if (total_output_elems < 65536) { avgpool_v10(input, output, params, stream); NVTX_RANGE_POP(); return; }
    if (params.sh == 1 && params.sw == 1 && params.kh == 3 && params.kw == 3) { avgpool_v15(input, output, params, stream); NVTX_RANGE_POP(); return; }
    avgpool_v2(input, output, params, stream);
    NVTX_RANGE_POP();
}

// --- v15: swizzled shared memory ---
void avgpool_v15(const nv_bfloat16* input, nv_bfloat16* output, const AvgPoolParams& params, cudaStream_t stream) {
    NVTX_RANGE_PUSH_C("avgpool_v15_bf16", NVTX_COLOR_AVGPOOL);
    constexpr int TILE_OH = 8; constexpr int TILE_OW = 8;
    const int blocks_oh = static_cast<int>((params.OH + TILE_OH - 1) / TILE_OH);
    const int blocks_ow = static_cast<int>((params.OW + TILE_OW - 1) / TILE_OW);
    int smem_h = (TILE_OH - 1) * params.sh + (params.kh - 1) * params.dh + 1;
    int smem_w = (TILE_OW - 1) * params.sw + (params.kw - 1) * params.dw + 1;
    size_t smem_bytes = static_cast<size_t>(smem_h) * (smem_w + 1) * sizeof(nv_bfloat16);
    PoolParams pp = make_pool_params_from_avg(params);
    dim3 block(TILE_OW, TILE_OH);
    dim3 grid(blocks_oh * blocks_ow, static_cast<int>(params.C), static_cast<int>(params.N));
    if (smem_bytes > 49152) {
        CUDA_CHECK(cudaFuncSetAttribute(maxpool_v15_kernel<nv_bfloat16, false, true>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem_bytes)));
        CUDA_CHECK(cudaFuncSetAttribute(maxpool_v15_kernel<nv_bfloat16, false, false>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem_bytes)));
    }
    if (params.count_include_pad) {
        maxpool_v15_kernel<nv_bfloat16, false, true><<<grid, block, smem_bytes, stream>>>(
            reinterpret_cast<const nv_bfloat16*>(input), reinterpret_cast<nv_bfloat16*>(output),
            pp, blocks_oh, blocks_ow, smem_h, smem_w);
    } else {
        maxpool_v15_kernel<nv_bfloat16, false, false><<<grid, block, smem_bytes, stream>>>(
            reinterpret_cast<const nv_bfloat16*>(input), reinterpret_cast<nv_bfloat16*>(output),
            pp, blocks_oh, blocks_ow, smem_h, smem_w);
    }
    CUDA_CHECK(cudaGetLastError());
    NVTX_RANGE_POP();
}

// ============================================================================
// INT8 AvgPool kernels
// ============================================================================

// --- v0: template instantiation ---
void avgpool_v0(const int8_t* input, int8_t* output, const AvgPoolParams& params, cudaStream_t stream) {
    NVTX_RANGE_PUSH_C("avgpool_v0_i8", NVTX_COLOR_AVGPOOL);
    const int64_t total = params.OH * params.OW * params.C;
    const int threads = 256;
    const int blocks_x = static_cast<int>((total + threads - 1) / threads);
    dim3 grid(blocks_x, 1, static_cast<int>(params.N));
    avgpool_v0_kernel<int8_t><<<grid, threads, 0, stream>>>(input, output, params);
    CUDA_CHECK(cudaGetLastError());
    NVTX_RANGE_POP();
}

// --- v1: explicit kernel (float shared memory) ---
template <int TILE_OH = 8, int TILE_OW = 8>
__global__ void avgpool_v1_kernel_i8(
    const int8_t* __restrict__ input,
    int8_t* __restrict__ output,
    const AvgPoolParams params,
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
            sdata[i] = 0.0f;
        }
    }
    __syncthreads();
    if (oh >= params.OH || ow >= params.OW) return;
    float sum = 0.0f;
    int count = 0;
    for (int kh_i = 0; kh_i < params.kh; ++kh_i) {
        const int sih = th * params.sh + kh_i * params.dh;
        if (sih >= smem_h) continue;
        const int64_t ih = static_cast<int64_t>(ih_start) + sih;
        for (int kw_i = 0; kw_i < params.kw; ++kw_i) {
            const int siw = tw * params.sw + kw_i * params.dw;
            if (siw >= smem_w) continue;
            const int64_t iw = static_cast<int64_t>(iw_start) + siw;
            const bool ih_in = (ih >= 0 && ih < params.H);
            const bool iw_in = (iw >= 0 && iw < params.W);
            if (ih_in && iw_in) {
                sum += sdata[sih * smem_w + siw];
                count++;
            } else if (params.count_include_pad) {
                const bool ih_in_pad = (ih >= -static_cast<int64_t>(params.ph) &&
                                        ih < params.H + static_cast<int64_t>(params.ph));
                const bool iw_in_pad = (iw >= -static_cast<int64_t>(params.pw) &&
                                        iw < params.W + static_cast<int64_t>(params.pw));
                if (ih_in_pad && iw_in_pad) { count++; }
            }
        }
    }
    float divisor;
    if (params.divisor_override > 0) {
        divisor = static_cast<float>(params.divisor_override);
    } else {
        divisor = static_cast<float>(count);
    }
    const int64_t out_idx = ((static_cast<int64_t>(n) * params.OH + oh) * params.OW + ow) * params.C + c;
    output[out_idx] = static_cast<int8_t>(roundf((divisor > 0.0f) ? (sum / divisor) : 0.0f));
}

void avgpool_v1(const int8_t* input, int8_t* output, const AvgPoolParams& params, cudaStream_t stream) {
    NVTX_RANGE_PUSH_C("avgpool_v1_i8", NVTX_COLOR_AVGPOOL);
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
    avgpool_v1_kernel_i8<TILE_OH, TILE_OW><<<grid, block, smem_bytes, stream>>>(
        input, output, params, blocks_oh, blocks_ow, smem_h, smem_w);
    CUDA_CHECK(cudaGetLastError());
    NVTX_RANGE_POP();
}

// --- v2: vectorized loads with int4 (128-bit, 16 elements) ---
template <int VEC>
__global__ void avgpool_v2_kernel_i8(
    const int8_t* __restrict__ input,
    int8_t* __restrict__ output,
    const AvgPoolParams params)
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
    int32_t sum[VEC];
    #pragma unroll
    for (int v = 0; v < VEC; ++v) sum[v] = 0;
    int count = 0;
    for (int kh_i = 0; kh_i < params.kh; ++kh_i) {
        const int64_t ih = oh * params.sh - params.ph + static_cast<int64_t>(kh_i) * params.dh;
        const bool ih_in = (ih >= 0 && ih < params.H);
        for (int kw_i = 0; kw_i < params.kw; ++kw_i) {
            const int64_t iw = ow * params.sw - params.pw + static_cast<int64_t>(kw_i) * params.dw;
            const bool iw_in = (iw >= 0 && iw < params.W);
            if (ih_in && iw_in) {
                const int64_t in_idx = ((n * params.H + ih) * params.W + iw) * params.C + c;
                int4 vec = *reinterpret_cast<const int4*>(&input[in_idx]);
                sum[0]  += vec.x; sum[1]  += vec.y;
                sum[2]  += vec.z; sum[3]  += vec.w;
                sum[4]  += (reinterpret_cast<const int32_t*>(&vec)[4] & 0xFF);
                sum[5]  += ((reinterpret_cast<const int32_t*>(&vec)[4] >> 8) & 0xFF);
                sum[6]  += ((reinterpret_cast<const int32_t*>(&vec)[4] >> 16) & 0xFF);
                sum[7]  += ((reinterpret_cast<const int32_t*>(&vec)[4] >> 24) & 0xFF);
                sum[8]  += (reinterpret_cast<const int32_t*>(&vec)[5] & 0xFF);
                sum[9]  += ((reinterpret_cast<const int32_t*>(&vec)[5] >> 8) & 0xFF);
                sum[10] += ((reinterpret_cast<const int32_t*>(&vec)[5] >> 16) & 0xFF);
                sum[11] += ((reinterpret_cast<const int32_t*>(&vec)[5] >> 24) & 0xFF);
                sum[12] += (reinterpret_cast<const int32_t*>(&vec)[6] & 0xFF);
                sum[13] += ((reinterpret_cast<const int32_t*>(&vec)[6] >> 8) & 0xFF);
                sum[14] += ((reinterpret_cast<const int32_t*>(&vec)[6] >> 16) & 0xFF);
                sum[15] += ((reinterpret_cast<const int32_t*>(&vec)[6] >> 24) & 0xFF);
                count++;
            } else if (params.count_include_pad) {
                const bool ih_in_pad = (ih >= -static_cast<int64_t>(params.ph) &&
                                        ih < params.H + static_cast<int64_t>(params.ph));
                const bool iw_in_pad = (iw >= -static_cast<int64_t>(params.pw) &&
                                        iw < params.W + static_cast<int64_t>(params.pw));
                if (ih_in_pad && iw_in_pad) { count++; }
            }
        }
    }
    float divisor;
    if (params.divisor_override > 0) {
        divisor = static_cast<float>(params.divisor_override);
    } else {
        divisor = static_cast<float>(count);
    }
    const int64_t out_idx = ((n * params.OH + oh) * params.OW + ow) * params.C + c;
    int8_t* out_ptr = const_cast<int8_t*>(&output[out_idx]);
    if (divisor > 0.0f) {
        float rdiv = 1.0f / divisor;
        #pragma unroll
        for (int v = 0; v < VEC; ++v) {
            out_ptr[v] = static_cast<int8_t>(roundf(static_cast<float>(sum[v]) * rdiv));
        }
    } else {
        #pragma unroll
        for (int v = 0; v < VEC; ++v) out_ptr[v] = 0;
    }
}

void avgpool_v2(const int8_t* input, int8_t* output, const AvgPoolParams& params, cudaStream_t stream) {
    NVTX_RANGE_PUSH_C("avgpool_v2_i8", NVTX_COLOR_AVGPOOL);
    constexpr int VEC = 16;
    if (params.C % VEC != 0) { NVTX_RANGE_POP(); avgpool_v0(input, output, params, stream); return; }
    const int64_t C_vec = params.C / VEC;
    const int64_t total = params.OH * params.OW * C_vec;
    const int threads = 256;
    const int blocks_x = static_cast<int>((total + threads - 1) / threads);
    dim3 grid(blocks_x, 1, static_cast<int>(params.N));
    avgpool_v2_kernel_i8<VEC><<<grid, threads, 0, stream>>>(input, output, params);
    CUDA_CHECK(cudaGetLastError());
    NVTX_RANGE_POP();
}

// --- v3: register blocking (template) ---
void avgpool_v3(const int8_t* input, int8_t* output, const AvgPoolParams& params, cudaStream_t stream) {
    NVTX_RANGE_PUSH_C("avgpool_v3_i8", NVTX_COLOR_AVGPOOL);
    constexpr int BLOCK = 4;
    if (params.OH < BLOCK || params.OH % BLOCK != 0) { NVTX_RANGE_POP(); avgpool_v0(input, output, params, stream); return; }
    const int64_t OH_block = params.OH / BLOCK;
    const int64_t total = OH_block * params.OW * params.C;
    const int threads = 256;
    const int blocks_x = static_cast<int>((total + threads - 1) / threads);
    dim3 grid(blocks_x, 1, static_cast<int>(params.N));
    avgpool_v3_kernel<int8_t, BLOCK><<<grid, threads, 0, stream>>>(input, output, params);
    CUDA_CHECK(cudaGetLastError());
    NVTX_RANGE_POP();
}

// --- v4: warp-level reduce (template) ---
void avgpool_v4(const int8_t* input, int8_t* output, const AvgPoolParams& params, cudaStream_t stream) {
    NVTX_RANGE_PUSH_C("avgpool_v4_i8", NVTX_COLOR_AVGPOOL);
    const int64_t total = params.OH * params.OW * params.C;
    const int threads = 256;
    const int warps_per_block = threads / 32;
    const int blocks_x = static_cast<int>((total + warps_per_block - 1) / warps_per_block);
    dim3 grid(blocks_x, 1, static_cast<int>(params.N));
    avgpool_v4_kernel<int8_t><<<grid, threads, 0, stream>>>(input, output, params);
    CUDA_CHECK(cudaGetLastError());
    NVTX_RANGE_POP();
}

// --- v5: double buffer (explicit, float intermediate) ---
template <int TILE_OH = 8, int TILE_OW = 8>
__global__ void avgpool_v5_kernel_i8(
    const int8_t* __restrict__ input,
    int8_t* __restrict__ output,
    const AvgPoolParams params,
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
        const int sih = i / smem_w; const int siw = i % smem_w;
        const int ih = ih_start + sih; const int iw = iw_start + siw;
        if (ih >= 0 && ih < params.H && iw >= 0 && iw < params.W) {
            const int64_t in_idx = ((static_cast<int64_t>(n) * params.H + ih) * params.W + iw) * params.C + c0;
            buf[0][i] = static_cast<float>(input[in_idx]);
        } else { buf[0][i] = 0.0f; }
    }
    __syncthreads();

    float sum_c0 = 0.0f; int count_c0 = 0;
    if (oh < params.OH && ow < params.OW) {
        for (int kh_i = 0; kh_i < params.kh; ++kh_i) {
            const int sih = th * params.sh + kh_i * params.dh;
            if (sih >= smem_h) continue;
            const int64_t ih = static_cast<int64_t>(ih_start) + sih;
            for (int kw_i = 0; kw_i < params.kw; ++kw_i) {
                const int siw = tw * params.sw + kw_i * params.dw;
                if (siw >= smem_w) continue;
                const int64_t iw = static_cast<int64_t>(iw_start) + siw;
                const bool ih_in = (ih >= 0 && ih < params.H);
                const bool iw_in = (iw >= 0 && iw < params.W);
                if (ih_in && iw_in) { sum_c0 += buf[0][sih * smem_w + siw]; count_c0++; }
                else if (params.count_include_pad) {
                    const bool ih_in_pad = (ih >= -static_cast<int64_t>(params.ph) && ih < params.H + static_cast<int64_t>(params.ph));
                    const bool iw_in_pad = (iw >= -static_cast<int64_t>(params.pw) && iw < params.W + static_cast<int64_t>(params.pw));
                    if (ih_in_pad && iw_in_pad) count_c0++;
                }
            }
        }
    }
    if (oh < params.OH && ow < params.OW) {
        float divisor = (params.divisor_override > 0) ? static_cast<float>(params.divisor_override) : static_cast<float>(count_c0);
        const int64_t out_idx = ((static_cast<int64_t>(n) * params.OH + oh) * params.OW + ow) * params.C + c0;
        output[out_idx] = static_cast<int8_t>(roundf((divisor > 0.0f) ? (sum_c0 / divisor) : 0.0f));
    }

    if (has_c1) {
        for (int i = tid; i < total_smem; i += nthreads) {
            const int sih = i / smem_w; const int siw = i % smem_w;
            const int ih = ih_start + sih; const int iw = iw_start + siw;
            if (ih >= 0 && ih < params.H && iw >= 0 && iw < params.W) {
                const int64_t in_idx = ((static_cast<int64_t>(n) * params.H + ih) * params.W + iw) * params.C + c1;
                buf[1][i] = static_cast<float>(input[in_idx]);
            } else { buf[1][i] = 0.0f; }
        }
        __syncthreads();
        float sum_c1 = 0.0f; int count_c1 = 0;
        if (oh < params.OH && ow < params.OW) {
            for (int kh_i = 0; kh_i < params.kh; ++kh_i) {
                const int sih = th * params.sh + kh_i * params.dh;
                if (sih >= smem_h) continue;
                const int64_t ih = static_cast<int64_t>(ih_start) + sih;
                for (int kw_i = 0; kw_i < params.kw; ++kw_i) {
                    const int siw = tw * params.sw + kw_i * params.dw;
                    if (siw >= smem_w) continue;
                    const int64_t iw = static_cast<int64_t>(iw_start) + siw;
                    const bool ih_in = (ih >= 0 && ih < params.H);
                    const bool iw_in = (iw >= 0 && iw < params.W);
                    if (ih_in && iw_in) { sum_c1 += buf[1][sih * smem_w + siw]; count_c1++; }
                    else if (params.count_include_pad) {
                        const bool ih_in_pad = (ih >= -static_cast<int64_t>(params.ph) && ih < params.H + static_cast<int64_t>(params.ph));
                        const bool iw_in_pad = (iw >= -static_cast<int64_t>(params.pw) && iw < params.W + static_cast<int64_t>(params.pw));
                        if (ih_in_pad && iw_in_pad) count_c1++;
                    }
                }
            }
        }
        if (oh < params.OH && ow < params.OW) {
            float divisor = (params.divisor_override > 0) ? static_cast<float>(params.divisor_override) : static_cast<float>(count_c1);
            const int64_t out_idx = ((static_cast<int64_t>(n) * params.OH + oh) * params.OW + ow) * params.C + c1;
            output[out_idx] = static_cast<int8_t>(roundf((divisor > 0.0f) ? (sum_c1 / divisor) : 0.0f));
        }
    }
}

void avgpool_v5(const int8_t* input, int8_t* output, const AvgPoolParams& params, cudaStream_t stream) {
    NVTX_RANGE_PUSH_C("avgpool_v5_i8", NVTX_COLOR_AVGPOOL);
    constexpr int TILE_OH = 8; constexpr int TILE_OW = 8;
    if (params.C < 2) { NVTX_RANGE_POP(); avgpool_v1(input, output, params, stream); return; }
    const int blocks_oh = static_cast<int>((params.OH + TILE_OH - 1) / TILE_OH);
    const int blocks_ow = static_cast<int>((params.OW + TILE_OW - 1) / TILE_OW);
    int smem_h = (TILE_OH - 1) * params.sh + (params.kh - 1) * params.dh + 1;
    int smem_w = (TILE_OW - 1) * params.sw + (params.kw - 1) * params.dw + 1;
    smem_h = min(smem_h, static_cast<int>(params.H + 2 * params.ph));
    smem_w = min(smem_w, static_cast<int>(params.W + 2 * params.pw));
    size_t smem_bytes = 2 * static_cast<size_t>(smem_h) * smem_w * sizeof(float);
    int smem_limit = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&smem_limit, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0));
    if (smem_bytes > static_cast<size_t>(smem_limit)) { NVTX_RANGE_POP(); avgpool_v1(input, output, params, stream); return; }
    const int c_pairs = static_cast<int>((params.C + 1) / 2);
    dim3 block(TILE_OW, TILE_OH);
    dim3 grid(blocks_oh * blocks_ow, c_pairs, static_cast<int>(params.N));
    if (smem_bytes > 49152) {
        CUDA_CHECK(cudaFuncSetAttribute(avgpool_v5_kernel_i8<TILE_OH, TILE_OW>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem_bytes)));
    }
    avgpool_v5_kernel_i8<TILE_OH, TILE_OW><<<grid, block, smem_bytes, stream>>>(
        input, output, params, blocks_oh, blocks_ow, smem_h, smem_w);
    CUDA_CHECK(cudaGetLastError());
    NVTX_RANGE_POP();
}

// --- v6: warp specialization (explicit) ---
template <int TILE_OH = 8, int TILE_OW = 8, int NUM_LOAD_WARPS = 2, int NUM_COMPUTE_WARPS = 6>
__global__ void avgpool_v6_kernel_i8(
    const int8_t* __restrict__ input,
    int8_t* __restrict__ output,
    const AvgPoolParams params,
    int blocks_oh, int blocks_ow,
    int smem_h, int smem_w)
{
    constexpr int NUM_LOAD_THREADS = NUM_LOAD_WARPS * 32;
    constexpr int NUM_COMPUTE_THREADS = NUM_COMPUTE_WARPS * 32;
    constexpr int TILE_SIZE = TILE_OH * TILE_OW;
    extern __shared__ float sdata[];
    const int n = blockIdx.z; const int c = blockIdx.y;
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
            const int sih = i / smem_w; const int siw = i % smem_w;
            const int ih = ih_start + sih; const int iw = iw_start + siw;
            if (ih >= 0 && ih < params.H && iw >= 0 && iw < params.W) {
                const int64_t in_idx = ((static_cast<int64_t>(n) * params.H + ih) * params.W + iw) * params.C + c;
                sdata[i] = static_cast<float>(input[in_idx]);
            } else { sdata[i] = 0.0f; }
        }
    }
    __syncthreads();
    if (!is_load_warp) {
        const int compute_tid = tid - NUM_LOAD_THREADS;
        for (int i = compute_tid; i < TILE_SIZE; i += NUM_COMPUTE_THREADS) {
            const int th = i / TILE_OW; const int tw = i % TILE_OW;
            const int oh = tile_oh * TILE_OH + th; const int ow = tile_ow * TILE_OW + tw;
            if (oh >= params.OH || ow >= params.OW) continue;
            float sum = 0.0f; int count = 0;
            for (int kh_i = 0; kh_i < params.kh; ++kh_i) {
                const int sih = th * params.sh + kh_i * params.dh;
                if (sih >= smem_h) continue;
                const int64_t ih = static_cast<int64_t>(ih_start) + sih;
                for (int kw_i = 0; kw_i < params.kw; ++kw_i) {
                    const int siw = tw * params.sw + kw_i * params.dw;
                    if (siw >= smem_w) continue;
                    const int64_t iw = static_cast<int64_t>(iw_start) + siw;
                    const bool ih_in = (ih >= 0 && ih < params.H);
                    const bool iw_in = (iw >= 0 && iw < params.W);
                    if (ih_in && iw_in) { sum += sdata[sih * smem_w + siw]; count++; }
                    else if (params.count_include_pad) {
                        const bool ih_in_pad = (ih >= -static_cast<int64_t>(params.ph) && ih < params.H + static_cast<int64_t>(params.ph));
                        const bool iw_in_pad = (iw >= -static_cast<int64_t>(params.pw) && iw < params.W + static_cast<int64_t>(params.pw));
                        if (ih_in_pad && iw_in_pad) count++;
                    }
                }
            }
            float divisor = (params.divisor_override > 0) ? static_cast<float>(params.divisor_override) : static_cast<float>(count);
            const int64_t out_idx = ((static_cast<int64_t>(n) * params.OH + oh) * params.OW + ow) * params.C + c;
            output[out_idx] = static_cast<int8_t>(roundf((divisor > 0.0f) ? (sum / divisor) : 0.0f));
        }
    }
}

void avgpool_v6(const int8_t* input, int8_t* output, const AvgPoolParams& params, cudaStream_t stream) {
    NVTX_RANGE_PUSH_C("avgpool_v6_i8", NVTX_COLOR_AVGPOOL);
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
    if (smem_bytes > static_cast<size_t>(smem_limit)) { NVTX_RANGE_POP(); avgpool_v1(input, output, params, stream); return; }
    dim3 block(BLOCK_SIZE);
    dim3 grid(blocks_oh * blocks_ow, static_cast<int>(params.C), static_cast<int>(params.N));
    if (smem_bytes > 49152) {
        CUDA_CHECK(cudaFuncSetAttribute(avgpool_v6_kernel_i8<TILE_OH, TILE_OW>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem_bytes)));
    }
    avgpool_v6_kernel_i8<TILE_OH, TILE_OW><<<grid, block, smem_bytes, stream>>>(
        input, output, params, blocks_oh, blocks_ow, smem_h, smem_w);
    CUDA_CHECK(cudaGetLastError());
    NVTX_RANGE_POP();
}

// --- v7: alternative mappings ---
void avgpool_v7(const int8_t* input, int8_t* output, const AvgPoolParams& params, int mapping, cudaStream_t stream) {
    NVTX_RANGE_PUSH_C("avgpool_v7_i8", NVTX_COLOR_AVGPOOL);
    switch (mapping) {
        case 0: { NVTX_RANGE_POP(); avgpool_v0(input, output, params, stream); return; }
        case 1: {
            const int C_groups = static_cast<int>((params.C + 3) / 4);
            const int64_t grid_z_64 = params.N * C_groups;
            if (grid_z_64 > 65535) { NVTX_RANGE_POP(); avgpool_v0(input, output, params, stream); return; }
            dim3 block(8, 8, 4);
            dim3 grid(static_cast<int>((params.OW + 7) / 8), static_cast<int>((params.OH + 7) / 8), static_cast<int>(grid_z_64));
            avgpool_v7_mappingB_kernel<int8_t><<<grid, block, 0, stream>>>(input, output, params, C_groups);
            CUDA_CHECK(cudaGetLastError()); break;
        }
        case 2: {
            const int C_groups = static_cast<int>((params.C + 255) / 256);
            const int64_t grid_z_64 = params.N * C_groups;
            if (params.OW > 65535 || params.OH > 65535 || grid_z_64 > 65535) { NVTX_RANGE_POP(); avgpool_v0(input, output, params, stream); return; }
            dim3 block(256, 1, 1);
            dim3 grid(static_cast<int>(params.OW), static_cast<int>(params.OH), static_cast<int>(grid_z_64));
            avgpool_v7_mappingC_kernel<int8_t><<<grid, block, 0, stream>>>(input, output, params, C_groups);
            CUDA_CHECK(cudaGetLastError()); break;
        }
        case 3: {
            if (params.C % 4 != 0) { NVTX_RANGE_POP(); avgpool_v0(input, output, params, stream); return; }
            const int C_groups = static_cast<int>((params.C + 31) / 32);
            const int64_t grid_z_64 = params.N * C_groups;
            if (grid_z_64 > 65535) { NVTX_RANGE_POP(); avgpool_v0(input, output, params, stream); return; }
            dim3 block(32, 8, 1);
            dim3 grid(static_cast<int>((params.OW + 3) / 4), static_cast<int>((params.OH + 3) / 4), static_cast<int>(grid_z_64));
            avgpool_v7_mappingD_kernel<int8_t><<<grid, block, 0, stream>>>(input, output, params, C_groups);
            CUDA_CHECK(cudaGetLastError()); break;
        }
        default:
            NVTX_RANGE_POP();
            throw std::invalid_argument("unsupported mapping: " + std::to_string(mapping));
    }
    NVTX_RANGE_POP();
}

// --- v8-v13: fallbacks ---
void avgpool_v8(const int8_t* input, int8_t* output, const AvgPoolParams& params, cudaStream_t stream) {
    NVTX_RANGE_PUSH_C("avgpool_v8_i8", NVTX_COLOR_AVGPOOL);
    avgpool_v2(input, output, params, stream);
    NVTX_RANGE_POP();
}
void avgpool_v9(const int8_t* input, int8_t* output, const AvgPoolParams& params, cudaStream_t stream) {
    NVTX_RANGE_PUSH_C("avgpool_v9_i8", NVTX_COLOR_AVGPOOL);
    avgpool_v2(input, output, params, stream);
    NVTX_RANGE_POP();
}
void avgpool_v10(const int8_t* input, int8_t* output, const AvgPoolParams& params, cudaStream_t stream) {
    NVTX_RANGE_PUSH_C("avgpool_v10_i8", NVTX_COLOR_AVGPOOL);
    avgpool_v2(input, output, params, stream);
    NVTX_RANGE_POP();
}
void avgpool_v11(const int8_t* input, int8_t* output, const AvgPoolParams& params, cudaStream_t stream) {
    NVTX_RANGE_PUSH_C("avgpool_v11_i8", NVTX_COLOR_AVGPOOL);
    avgpool_v4(input, output, params, stream);
    NVTX_RANGE_POP();
}
void avgpool_v12(const int8_t* input, int8_t* output, const AvgPoolParams& params, cudaStream_t stream) {
    NVTX_RANGE_PUSH_C("avgpool_v12_i8", NVTX_COLOR_AVGPOOL);
    avgpool_v2(input, output, params, stream);
    NVTX_RANGE_POP();
}
void avgpool_v13(const int8_t* input, int8_t* output, const AvgPoolParams& params, cudaStream_t stream) {
    NVTX_RANGE_PUSH_C("avgpool_v13_i8", NVTX_COLOR_AVGPOOL);
    avgpool_v2(input, output, params, stream);
    NVTX_RANGE_POP();
}

// --- v14: adaptive dispatcher ---
void avgpool_v14(const int8_t* input, int8_t* output, const AvgPoolParams& params, cudaStream_t stream) {
    NVTX_RANGE_PUSH_C("avgpool_v14_i8", NVTX_COLOR_AVGPOOL);
    bool is_global = (params.kh >= params.H && params.kw >= params.W);
    if (is_global) { avgpool_v13(input, output, params, stream); NVTX_RANGE_POP(); return; }
    int64_t total_output_elems = params.N * params.OH * params.OW * params.C;
    if (total_output_elems < 65536) { avgpool_v10(input, output, params, stream); NVTX_RANGE_POP(); return; }
    if (params.sh == 1 && params.sw == 1 && params.kh == 3 && params.kw == 3) { avgpool_v15(input, output, params, stream); NVTX_RANGE_POP(); return; }
    avgpool_v2(input, output, params, stream);
    NVTX_RANGE_POP();
}

// --- v15: swizzled shared memory ---
void avgpool_v15(const int8_t* input, int8_t* output, const AvgPoolParams& params, cudaStream_t stream) {
    NVTX_RANGE_PUSH_C("avgpool_v15_i8", NVTX_COLOR_AVGPOOL);
    constexpr int TILE_OH = 8; constexpr int TILE_OW = 8;
    const int blocks_oh = static_cast<int>((params.OH + TILE_OH - 1) / TILE_OH);
    const int blocks_ow = static_cast<int>((params.OW + TILE_OW - 1) / TILE_OW);
    int smem_h = (TILE_OH - 1) * params.sh + (params.kh - 1) * params.dh + 1;
    int smem_w = (TILE_OW - 1) * params.sw + (params.kw - 1) * params.dw + 1;
    size_t smem_bytes = static_cast<size_t>(smem_h) * (smem_w + 1) * sizeof(int8_t);
    PoolParams pp = make_pool_params_from_avg(params);
    dim3 block(TILE_OW, TILE_OH);
    dim3 grid(blocks_oh * blocks_ow, static_cast<int>(params.C), static_cast<int>(params.N));
    if (smem_bytes > 49152) {
        CUDA_CHECK(cudaFuncSetAttribute(maxpool_v15_kernel<int8_t, false, true>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem_bytes)));
        CUDA_CHECK(cudaFuncSetAttribute(maxpool_v15_kernel<int8_t, false, false>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem_bytes)));
    }
    if (params.count_include_pad) {
        maxpool_v15_kernel<int8_t, false, true><<<grid, block, smem_bytes, stream>>>(
            reinterpret_cast<const int8_t*>(input), reinterpret_cast<int8_t*>(output),
            pp, blocks_oh, blocks_ow, smem_h, smem_w);
    } else {
        maxpool_v15_kernel<int8_t, false, false><<<grid, block, smem_bytes, stream>>>(
            reinterpret_cast<const int8_t*>(input), reinterpret_cast<int8_t*>(output),
            pp, blocks_oh, blocks_ow, smem_h, smem_w);
    }
    CUDA_CHECK(cudaGetLastError());
    NVTX_RANGE_POP();
}

// ============================================================================
// INT16 AvgPool kernels
// ============================================================================

// --- v0: template instantiation ---
void avgpool_v0(const int16_t* input, int16_t* output, const AvgPoolParams& params, cudaStream_t stream) {
    NVTX_RANGE_PUSH_C("avgpool_v0_i16", NVTX_COLOR_AVGPOOL);
    const int64_t total = params.OH * params.OW * params.C;
    const int threads = 256;
    const int blocks_x = static_cast<int>((total + threads - 1) / threads);
    dim3 grid(blocks_x, 1, static_cast<int>(params.N));
    avgpool_v0_kernel<int16_t><<<grid, threads, 0, stream>>>(input, output, params);
    CUDA_CHECK(cudaGetLastError());
    NVTX_RANGE_POP();
}

// --- v1: explicit kernel (float shared memory) ---
template <int TILE_OH = 8, int TILE_OW = 8>
__global__ void avgpool_v1_kernel_i16(
    const int16_t* __restrict__ input,
    int16_t* __restrict__ output,
    const AvgPoolParams params,
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
            sdata[i] = 0.0f;
        }
    }
    __syncthreads();
    if (oh >= params.OH || ow >= params.OW) return;
    float sum = 0.0f;
    int count = 0;
    for (int kh_i = 0; kh_i < params.kh; ++kh_i) {
        const int sih = th * params.sh + kh_i * params.dh;
        if (sih >= smem_h) continue;
        const int64_t ih = static_cast<int64_t>(ih_start) + sih;
        for (int kw_i = 0; kw_i < params.kw; ++kw_i) {
            const int siw = tw * params.sw + kw_i * params.dw;
            if (siw >= smem_w) continue;
            const int64_t iw = static_cast<int64_t>(iw_start) + siw;
            const bool ih_in = (ih >= 0 && ih < params.H);
            const bool iw_in = (iw >= 0 && iw < params.W);
            if (ih_in && iw_in) {
                sum += sdata[sih * smem_w + siw];
                count++;
            } else if (params.count_include_pad) {
                const bool ih_in_pad = (ih >= -static_cast<int64_t>(params.ph) &&
                                        ih < params.H + static_cast<int64_t>(params.ph));
                const bool iw_in_pad = (iw >= -static_cast<int64_t>(params.pw) &&
                                        iw < params.W + static_cast<int64_t>(params.pw));
                if (ih_in_pad && iw_in_pad) { count++; }
            }
        }
    }
    float divisor;
    if (params.divisor_override > 0) {
        divisor = static_cast<float>(params.divisor_override);
    } else {
        divisor = static_cast<float>(count);
    }
    const int64_t out_idx = ((static_cast<int64_t>(n) * params.OH + oh) * params.OW + ow) * params.C + c;
    output[out_idx] = static_cast<int16_t>(roundf((divisor > 0.0f) ? (sum / divisor) : 0.0f));
}

void avgpool_v1(const int16_t* input, int16_t* output, const AvgPoolParams& params, cudaStream_t stream) {
    NVTX_RANGE_PUSH_C("avgpool_v1_i16", NVTX_COLOR_AVGPOOL);
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
    avgpool_v1_kernel_i16<TILE_OH, TILE_OW><<<grid, block, smem_bytes, stream>>>(
        input, output, params, blocks_oh, blocks_ow, smem_h, smem_w);
    CUDA_CHECK(cudaGetLastError());
    NVTX_RANGE_POP();
}

// --- v2: vectorized loads with short4 (128-bit, 8 elements) ---
template <int VEC>
__global__ void avgpool_v2_kernel_i16(
    const int16_t* __restrict__ input,
    int16_t* __restrict__ output,
    const AvgPoolParams params)
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
    int32_t sum[VEC];
    #pragma unroll
    for (int v = 0; v < VEC; ++v) sum[v] = 0;
    int count = 0;
    for (int kh_i = 0; kh_i < params.kh; ++kh_i) {
        const int64_t ih = oh * params.sh - params.ph + static_cast<int64_t>(kh_i) * params.dh;
        const bool ih_in = (ih >= 0 && ih < params.H);
        for (int kw_i = 0; kw_i < params.kw; ++kw_i) {
            const int64_t iw = ow * params.sw - params.pw + static_cast<int64_t>(kw_i) * params.dw;
            const bool iw_in = (iw >= 0 && iw < params.W);
            if (ih_in && iw_in) {
                const int64_t in_idx = ((n * params.H + ih) * params.W + iw) * params.C + c;
                short4 vec = *reinterpret_cast<const short4*>(&input[in_idx]);
                sum[0] += vec.x; sum[1] += vec.y;
                sum[2] += vec.z; sum[3] += vec.w;
                count++;
            } else if (params.count_include_pad) {
                const bool ih_in_pad = (ih >= -static_cast<int64_t>(params.ph) &&
                                        ih < params.H + static_cast<int64_t>(params.ph));
                const bool iw_in_pad = (iw >= -static_cast<int64_t>(params.pw) &&
                                        iw < params.W + static_cast<int64_t>(params.pw));
                if (ih_in_pad && iw_in_pad) { count++; }
            }
        }
    }
    float divisor;
    if (params.divisor_override > 0) {
        divisor = static_cast<float>(params.divisor_override);
    } else {
        divisor = static_cast<float>(count);
    }
    const int64_t out_idx = ((n * params.OH + oh) * params.OW + ow) * params.C + c;
    int16_t* out_ptr = const_cast<int16_t*>(&output[out_idx]);
    if (divisor > 0.0f) {
        float rdiv = 1.0f / divisor;
        #pragma unroll
        for (int v = 0; v < VEC; ++v) {
            out_ptr[v] = static_cast<int16_t>(roundf(static_cast<float>(sum[v]) * rdiv));
        }
    } else {
        #pragma unroll
        for (int v = 0; v < VEC; ++v) out_ptr[v] = 0;
    }
}

void avgpool_v2(const int16_t* input, int16_t* output, const AvgPoolParams& params, cudaStream_t stream) {
    NVTX_RANGE_PUSH_C("avgpool_v2_i16", NVTX_COLOR_AVGPOOL);
    constexpr int VEC = 8;
    if (params.C % VEC != 0) { NVTX_RANGE_POP(); avgpool_v0(input, output, params, stream); return; }
    const int64_t C_vec = params.C / VEC;
    const int64_t total = params.OH * params.OW * C_vec;
    const int threads = 256;
    const int blocks_x = static_cast<int>((total + threads - 1) / threads);
    dim3 grid(blocks_x, 1, static_cast<int>(params.N));
    avgpool_v2_kernel_i16<VEC><<<grid, threads, 0, stream>>>(input, output, params);
    CUDA_CHECK(cudaGetLastError());
    NVTX_RANGE_POP();
}

// --- v3: register blocking (template) ---
void avgpool_v3(const int16_t* input, int16_t* output, const AvgPoolParams& params, cudaStream_t stream) {
    NVTX_RANGE_PUSH_C("avgpool_v3_i16", NVTX_COLOR_AVGPOOL);
    constexpr int BLOCK = 4;
    if (params.OH < BLOCK || params.OH % BLOCK != 0) { NVTX_RANGE_POP(); avgpool_v0(input, output, params, stream); return; }
    const int64_t OH_block = params.OH / BLOCK;
    const int64_t total = OH_block * params.OW * params.C;
    const int threads = 256;
    const int blocks_x = static_cast<int>((total + threads - 1) / threads);
    dim3 grid(blocks_x, 1, static_cast<int>(params.N));
    avgpool_v3_kernel<int16_t, BLOCK><<<grid, threads, 0, stream>>>(input, output, params);
    CUDA_CHECK(cudaGetLastError());
    NVTX_RANGE_POP();
}

// --- v4: warp-level reduce (template) ---
void avgpool_v4(const int16_t* input, int16_t* output, const AvgPoolParams& params, cudaStream_t stream) {
    NVTX_RANGE_PUSH_C("avgpool_v4_i16", NVTX_COLOR_AVGPOOL);
    const int64_t total = params.OH * params.OW * params.C;
    const int threads = 256;
    const int warps_per_block = threads / 32;
    const int blocks_x = static_cast<int>((total + warps_per_block - 1) / warps_per_block);
    dim3 grid(blocks_x, 1, static_cast<int>(params.N));
    avgpool_v4_kernel<int16_t><<<grid, threads, 0, stream>>>(input, output, params);
    CUDA_CHECK(cudaGetLastError());
    NVTX_RANGE_POP();
}

// --- v5: double buffer (explicit, float intermediate) ---
template <int TILE_OH = 8, int TILE_OW = 8>
__global__ void avgpool_v5_kernel_i16(
    const int16_t* __restrict__ input,
    int16_t* __restrict__ output,
    const AvgPoolParams params,
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
        const int sih = i / smem_w; const int siw = i % smem_w;
        const int ih = ih_start + sih; const int iw = iw_start + siw;
        if (ih >= 0 && ih < params.H && iw >= 0 && iw < params.W) {
            const int64_t in_idx = ((static_cast<int64_t>(n) * params.H + ih) * params.W + iw) * params.C + c0;
            buf[0][i] = static_cast<float>(input[in_idx]);
        } else { buf[0][i] = 0.0f; }
    }
    __syncthreads();

    float sum_c0 = 0.0f; int count_c0 = 0;
    if (oh < params.OH && ow < params.OW) {
        for (int kh_i = 0; kh_i < params.kh; ++kh_i) {
            const int sih = th * params.sh + kh_i * params.dh;
            if (sih >= smem_h) continue;
            const int64_t ih = static_cast<int64_t>(ih_start) + sih;
            for (int kw_i = 0; kw_i < params.kw; ++kw_i) {
                const int siw = tw * params.sw + kw_i * params.dw;
                if (siw >= smem_w) continue;
                const int64_t iw = static_cast<int64_t>(iw_start) + siw;
                const bool ih_in = (ih >= 0 && ih < params.H);
                const bool iw_in = (iw >= 0 && iw < params.W);
                if (ih_in && iw_in) { sum_c0 += buf[0][sih * smem_w + siw]; count_c0++; }
                else if (params.count_include_pad) {
                    const bool ih_in_pad = (ih >= -static_cast<int64_t>(params.ph) && ih < params.H + static_cast<int64_t>(params.ph));
                    const bool iw_in_pad = (iw >= -static_cast<int64_t>(params.pw) && iw < params.W + static_cast<int64_t>(params.pw));
                    if (ih_in_pad && iw_in_pad) count_c0++;
                }
            }
        }
    }
    if (oh < params.OH && ow < params.OW) {
        float divisor = (params.divisor_override > 0) ? static_cast<float>(params.divisor_override) : static_cast<float>(count_c0);
        const int64_t out_idx = ((static_cast<int64_t>(n) * params.OH + oh) * params.OW + ow) * params.C + c0;
        output[out_idx] = static_cast<int16_t>(roundf((divisor > 0.0f) ? (sum_c0 / divisor) : 0.0f));
    }

    if (has_c1) {
        for (int i = tid; i < total_smem; i += nthreads) {
            const int sih = i / smem_w; const int siw = i % smem_w;
            const int ih = ih_start + sih; const int iw = iw_start + siw;
            if (ih >= 0 && ih < params.H && iw >= 0 && iw < params.W) {
                const int64_t in_idx = ((static_cast<int64_t>(n) * params.H + ih) * params.W + iw) * params.C + c1;
                buf[1][i] = static_cast<float>(input[in_idx]);
            } else { buf[1][i] = 0.0f; }
        }
        __syncthreads();
        float sum_c1 = 0.0f; int count_c1 = 0;
        if (oh < params.OH && ow < params.OW) {
            for (int kh_i = 0; kh_i < params.kh; ++kh_i) {
                const int sih = th * params.sh + kh_i * params.dh;
                if (sih >= smem_h) continue;
                const int64_t ih = static_cast<int64_t>(ih_start) + sih;
                for (int kw_i = 0; kw_i < params.kw; ++kw_i) {
                    const int siw = tw * params.sw + kw_i * params.dw;
                    if (siw >= smem_w) continue;
                    const int64_t iw = static_cast<int64_t>(iw_start) + siw;
                    const bool ih_in = (ih >= 0 && ih < params.H);
                    const bool iw_in = (iw >= 0 && iw < params.W);
                    if (ih_in && iw_in) { sum_c1 += buf[1][sih * smem_w + siw]; count_c1++; }
                    else if (params.count_include_pad) {
                        const bool ih_in_pad = (ih >= -static_cast<int64_t>(params.ph) && ih < params.H + static_cast<int64_t>(params.ph));
                        const bool iw_in_pad = (iw >= -static_cast<int64_t>(params.pw) && iw < params.W + static_cast<int64_t>(params.pw));
                        if (ih_in_pad && iw_in_pad) count_c1++;
                    }
                }
            }
        }
        if (oh < params.OH && ow < params.OW) {
            float divisor = (params.divisor_override > 0) ? static_cast<float>(params.divisor_override) : static_cast<float>(count_c1);
            const int64_t out_idx = ((static_cast<int64_t>(n) * params.OH + oh) * params.OW + ow) * params.C + c1;
            output[out_idx] = static_cast<int16_t>(roundf((divisor > 0.0f) ? (sum_c1 / divisor) : 0.0f));
        }
    }
}

void avgpool_v5(const int16_t* input, int16_t* output, const AvgPoolParams& params, cudaStream_t stream) {
    NVTX_RANGE_PUSH_C("avgpool_v5_i16", NVTX_COLOR_AVGPOOL);
    constexpr int TILE_OH = 8; constexpr int TILE_OW = 8;
    if (params.C < 2) { NVTX_RANGE_POP(); avgpool_v1(input, output, params, stream); return; }
    const int blocks_oh = static_cast<int>((params.OH + TILE_OH - 1) / TILE_OH);
    const int blocks_ow = static_cast<int>((params.OW + TILE_OW - 1) / TILE_OW);
    int smem_h = (TILE_OH - 1) * params.sh + (params.kh - 1) * params.dh + 1;
    int smem_w = (TILE_OW - 1) * params.sw + (params.kw - 1) * params.dw + 1;
    smem_h = min(smem_h, static_cast<int>(params.H + 2 * params.ph));
    smem_w = min(smem_w, static_cast<int>(params.W + 2 * params.pw));
    size_t smem_bytes = 2 * static_cast<size_t>(smem_h) * smem_w * sizeof(float);
    int smem_limit = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&smem_limit, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0));
    if (smem_bytes > static_cast<size_t>(smem_limit)) { NVTX_RANGE_POP(); avgpool_v1(input, output, params, stream); return; }
    const int c_pairs = static_cast<int>((params.C + 1) / 2);
    dim3 block(TILE_OW, TILE_OH);
    dim3 grid(blocks_oh * blocks_ow, c_pairs, static_cast<int>(params.N));
    if (smem_bytes > 49152) {
        CUDA_CHECK(cudaFuncSetAttribute(avgpool_v5_kernel_i16<TILE_OH, TILE_OW>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem_bytes)));
    }
    avgpool_v5_kernel_i16<TILE_OH, TILE_OW><<<grid, block, smem_bytes, stream>>>(
        input, output, params, blocks_oh, blocks_ow, smem_h, smem_w);
    CUDA_CHECK(cudaGetLastError());
    NVTX_RANGE_POP();
}

// --- v6: warp specialization (explicit) ---
template <int TILE_OH = 8, int TILE_OW = 8, int NUM_LOAD_WARPS = 2, int NUM_COMPUTE_WARPS = 6>
__global__ void avgpool_v6_kernel_i16(
    const int16_t* __restrict__ input,
    int16_t* __restrict__ output,
    const AvgPoolParams params,
    int blocks_oh, int blocks_ow,
    int smem_h, int smem_w)
{
    constexpr int NUM_LOAD_THREADS = NUM_LOAD_WARPS * 32;
    constexpr int NUM_COMPUTE_THREADS = NUM_COMPUTE_WARPS * 32;
    constexpr int TILE_SIZE = TILE_OH * TILE_OW;
    extern __shared__ float sdata[];
    const int n = blockIdx.z; const int c = blockIdx.y;
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
            const int sih = i / smem_w; const int siw = i % smem_w;
            const int ih = ih_start + sih; const int iw = iw_start + siw;
            if (ih >= 0 && ih < params.H && iw >= 0 && iw < params.W) {
                const int64_t in_idx = ((static_cast<int64_t>(n) * params.H + ih) * params.W + iw) * params.C + c;
                sdata[i] = static_cast<float>(input[in_idx]);
            } else { sdata[i] = 0.0f; }
        }
    }
    __syncthreads();
    if (!is_load_warp) {
        const int compute_tid = tid - NUM_LOAD_THREADS;
        for (int i = compute_tid; i < TILE_SIZE; i += NUM_COMPUTE_THREADS) {
            const int th = i / TILE_OW; const int tw = i % TILE_OW;
            const int oh = tile_oh * TILE_OH + th; const int ow = tile_ow * TILE_OW + tw;
            if (oh >= params.OH || ow >= params.OW) continue;
            float sum = 0.0f; int count = 0;
            for (int kh_i = 0; kh_i < params.kh; ++kh_i) {
                const int sih = th * params.sh + kh_i * params.dh;
                if (sih >= smem_h) continue;
                const int64_t ih = static_cast<int64_t>(ih_start) + sih;
                for (int kw_i = 0; kw_i < params.kw; ++kw_i) {
                    const int siw = tw * params.sw + kw_i * params.dw;
                    if (siw >= smem_w) continue;
                    const int64_t iw = static_cast<int64_t>(iw_start) + siw;
                    const bool ih_in = (ih >= 0 && ih < params.H);
                    const bool iw_in = (iw >= 0 && iw < params.W);
                    if (ih_in && iw_in) { sum += sdata[sih * smem_w + siw]; count++; }
                    else if (params.count_include_pad) {
                        const bool ih_in_pad = (ih >= -static_cast<int64_t>(params.ph) && ih < params.H + static_cast<int64_t>(params.ph));
                        const bool iw_in_pad = (iw >= -static_cast<int64_t>(params.pw) && iw < params.W + static_cast<int64_t>(params.pw));
                        if (ih_in_pad && iw_in_pad) count++;
                    }
                }
            }
            float divisor = (params.divisor_override > 0) ? static_cast<float>(params.divisor_override) : static_cast<float>(count);
            const int64_t out_idx = ((static_cast<int64_t>(n) * params.OH + oh) * params.OW + ow) * params.C + c;
            output[out_idx] = static_cast<int16_t>(roundf((divisor > 0.0f) ? (sum / divisor) : 0.0f));
        }
    }
}

void avgpool_v6(const int16_t* input, int16_t* output, const AvgPoolParams& params, cudaStream_t stream) {
    NVTX_RANGE_PUSH_C("avgpool_v6_i16", NVTX_COLOR_AVGPOOL);
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
    if (smem_bytes > static_cast<size_t>(smem_limit)) { NVTX_RANGE_POP(); avgpool_v1(input, output, params, stream); return; }
    dim3 block(BLOCK_SIZE);
    dim3 grid(blocks_oh * blocks_ow, static_cast<int>(params.C), static_cast<int>(params.N));
    if (smem_bytes > 49152) {
        CUDA_CHECK(cudaFuncSetAttribute(avgpool_v6_kernel_i16<TILE_OH, TILE_OW>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem_bytes)));
    }
    avgpool_v6_kernel_i16<TILE_OH, TILE_OW><<<grid, block, smem_bytes, stream>>>(
        input, output, params, blocks_oh, blocks_ow, smem_h, smem_w);
    CUDA_CHECK(cudaGetLastError());
    NVTX_RANGE_POP();
}

// --- v7: alternative mappings ---
void avgpool_v7(const int16_t* input, int16_t* output, const AvgPoolParams& params, int mapping, cudaStream_t stream) {
    NVTX_RANGE_PUSH_C("avgpool_v7_i16", NVTX_COLOR_AVGPOOL);
    switch (mapping) {
        case 0: { NVTX_RANGE_POP(); avgpool_v0(input, output, params, stream); return; }
        case 1: {
            const int C_groups = static_cast<int>((params.C + 3) / 4);
            const int64_t grid_z_64 = params.N * C_groups;
            if (grid_z_64 > 65535) { NVTX_RANGE_POP(); avgpool_v0(input, output, params, stream); return; }
            dim3 block(8, 8, 4);
            dim3 grid(static_cast<int>((params.OW + 7) / 8), static_cast<int>((params.OH + 7) / 8), static_cast<int>(grid_z_64));
            avgpool_v7_mappingB_kernel<int16_t><<<grid, block, 0, stream>>>(input, output, params, C_groups);
            CUDA_CHECK(cudaGetLastError()); break;
        }
        case 2: {
            const int C_groups = static_cast<int>((params.C + 255) / 256);
            const int64_t grid_z_64 = params.N * C_groups;
            if (params.OW > 65535 || params.OH > 65535 || grid_z_64 > 65535) { NVTX_RANGE_POP(); avgpool_v0(input, output, params, stream); return; }
            dim3 block(256, 1, 1);
            dim3 grid(static_cast<int>(params.OW), static_cast<int>(params.OH), static_cast<int>(grid_z_64));
            avgpool_v7_mappingC_kernel<int16_t><<<grid, block, 0, stream>>>(input, output, params, C_groups);
            CUDA_CHECK(cudaGetLastError()); break;
        }
        case 3: {
            if (params.C % 4 != 0) { NVTX_RANGE_POP(); avgpool_v0(input, output, params, stream); return; }
            const int C_groups = static_cast<int>((params.C + 31) / 32);
            const int64_t grid_z_64 = params.N * C_groups;
            if (grid_z_64 > 65535) { NVTX_RANGE_POP(); avgpool_v0(input, output, params, stream); return; }
            dim3 block(32, 8, 1);
            dim3 grid(static_cast<int>((params.OW + 3) / 4), static_cast<int>((params.OH + 3) / 4), static_cast<int>(grid_z_64));
            avgpool_v7_mappingD_kernel<int16_t><<<grid, block, 0, stream>>>(input, output, params, C_groups);
            CUDA_CHECK(cudaGetLastError()); break;
        }
        default:
            NVTX_RANGE_POP();
            throw std::invalid_argument("unsupported mapping: " + std::to_string(mapping));
    }
    NVTX_RANGE_POP();
}

// --- v8-v13: fallbacks ---
void avgpool_v8(const int16_t* input, int16_t* output, const AvgPoolParams& params, cudaStream_t stream) {
    NVTX_RANGE_PUSH_C("avgpool_v8_i16", NVTX_COLOR_AVGPOOL);
    avgpool_v2(input, output, params, stream);
    NVTX_RANGE_POP();
}
void avgpool_v9(const int16_t* input, int16_t* output, const AvgPoolParams& params, cudaStream_t stream) {
    NVTX_RANGE_PUSH_C("avgpool_v9_i16", NVTX_COLOR_AVGPOOL);
    avgpool_v2(input, output, params, stream);
    NVTX_RANGE_POP();
}
void avgpool_v10(const int16_t* input, int16_t* output, const AvgPoolParams& params, cudaStream_t stream) {
    NVTX_RANGE_PUSH_C("avgpool_v10_i16", NVTX_COLOR_AVGPOOL);
    avgpool_v2(input, output, params, stream);
    NVTX_RANGE_POP();
}
void avgpool_v11(const int16_t* input, int16_t* output, const AvgPoolParams& params, cudaStream_t stream) {
    NVTX_RANGE_PUSH_C("avgpool_v11_i16", NVTX_COLOR_AVGPOOL);
    avgpool_v4(input, output, params, stream);
    NVTX_RANGE_POP();
}
void avgpool_v12(const int16_t* input, int16_t* output, const AvgPoolParams& params, cudaStream_t stream) {
    NVTX_RANGE_PUSH_C("avgpool_v12_i16", NVTX_COLOR_AVGPOOL);
    avgpool_v2(input, output, params, stream);
    NVTX_RANGE_POP();
}
void avgpool_v13(const int16_t* input, int16_t* output, const AvgPoolParams& params, cudaStream_t stream) {
    NVTX_RANGE_PUSH_C("avgpool_v13_i16", NVTX_COLOR_AVGPOOL);
    avgpool_v2(input, output, params, stream);
    NVTX_RANGE_POP();
}

// --- v14: adaptive dispatcher ---
void avgpool_v14(const int16_t* input, int16_t* output, const AvgPoolParams& params, cudaStream_t stream) {
    NVTX_RANGE_PUSH_C("avgpool_v14_i16", NVTX_COLOR_AVGPOOL);
    bool is_global = (params.kh >= params.H && params.kw >= params.W);
    if (is_global) { avgpool_v13(input, output, params, stream); NVTX_RANGE_POP(); return; }
    int64_t total_output_elems = params.N * params.OH * params.OW * params.C;
    if (total_output_elems < 65536) { avgpool_v10(input, output, params, stream); NVTX_RANGE_POP(); return; }
    if (params.sh == 1 && params.sw == 1 && params.kh == 3 && params.kw == 3) { avgpool_v15(input, output, params, stream); NVTX_RANGE_POP(); return; }
    avgpool_v2(input, output, params, stream);
    NVTX_RANGE_POP();
}

// --- v15: swizzled shared memory ---
void avgpool_v15(const int16_t* input, int16_t* output, const AvgPoolParams& params, cudaStream_t stream) {
    NVTX_RANGE_PUSH_C("avgpool_v15_i16", NVTX_COLOR_AVGPOOL);
    constexpr int TILE_OH = 8; constexpr int TILE_OW = 8;
    const int blocks_oh = static_cast<int>((params.OH + TILE_OH - 1) / TILE_OH);
    const int blocks_ow = static_cast<int>((params.OW + TILE_OW - 1) / TILE_OW);
    int smem_h = (TILE_OH - 1) * params.sh + (params.kh - 1) * params.dh + 1;
    int smem_w = (TILE_OW - 1) * params.sw + (params.kw - 1) * params.dw + 1;
    size_t smem_bytes = static_cast<size_t>(smem_h) * (smem_w + 1) * sizeof(int16_t);
    PoolParams pp = make_pool_params_from_avg(params);
    dim3 block(TILE_OW, TILE_OH);
    dim3 grid(blocks_oh * blocks_ow, static_cast<int>(params.C), static_cast<int>(params.N));
    if (smem_bytes > 49152) {
        CUDA_CHECK(cudaFuncSetAttribute(maxpool_v15_kernel<int16_t, false, true>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem_bytes)));
        CUDA_CHECK(cudaFuncSetAttribute(maxpool_v15_kernel<int16_t, false, false>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem_bytes)));
    }
    if (params.count_include_pad) {
        maxpool_v15_kernel<int16_t, false, true><<<grid, block, smem_bytes, stream>>>(
            reinterpret_cast<const int16_t*>(input), reinterpret_cast<int16_t*>(output),
            pp, blocks_oh, blocks_ow, smem_h, smem_w);
    } else {
        maxpool_v15_kernel<int16_t, false, false><<<grid, block, smem_bytes, stream>>>(
            reinterpret_cast<const int16_t*>(input), reinterpret_cast<int16_t*>(output),
            pp, blocks_oh, blocks_ow, smem_h, smem_w);
    }
    CUDA_CHECK(cudaGetLastError());
    NVTX_RANGE_POP();
}

// ============================================================================
// FP8 E4M3 AvgPool kernels
// ============================================================================

// --- v0: template instantiation ---
void avgpool_v0(const __nv_fp8_e4m3* input, __nv_fp8_e4m3* output, const AvgPoolParams& params, cudaStream_t stream) {
    NVTX_RANGE_PUSH_C("avgpool_v0_fp8e4m3", NVTX_COLOR_AVGPOOL);
    const int64_t total = params.OH * params.OW * params.C;
    const int threads = 256;
    const int blocks_x = static_cast<int>((total + threads - 1) / threads);
    dim3 grid(blocks_x, 1, static_cast<int>(params.N));
    avgpool_v0_kernel<__nv_fp8_e4m3><<<grid, threads, 0, stream>>>(input, output, params);
    CUDA_CHECK(cudaGetLastError());
    NVTX_RANGE_POP();
}

// --- v1: explicit kernel (float shared memory) ---
template <int TILE_OH = 8, int TILE_OW = 8>
__global__ void avgpool_v1_kernel_fp8_e4m3(
    const __nv_fp8_e4m3* __restrict__ input,
    __nv_fp8_e4m3* __restrict__ output,
    const AvgPoolParams params,
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
            sdata[i] = 0.0f;
        }
    }
    __syncthreads();
    if (oh >= params.OH || ow >= params.OW) return;
    float sum = 0.0f;
    int count = 0;
    for (int kh_i = 0; kh_i < params.kh; ++kh_i) {
        const int sih = th * params.sh + kh_i * params.dh;
        if (sih >= smem_h) continue;
        const int64_t ih = static_cast<int64_t>(ih_start) + sih;
        for (int kw_i = 0; kw_i < params.kw; ++kw_i) {
            const int siw = tw * params.sw + kw_i * params.dw;
            if (siw >= smem_w) continue;
            const int64_t iw = static_cast<int64_t>(iw_start) + siw;
            const bool ih_in = (ih >= 0 && ih < params.H);
            const bool iw_in = (iw >= 0 && iw < params.W);
            if (ih_in && iw_in) {
                sum += sdata[sih * smem_w + siw];
                count++;
            } else if (params.count_include_pad) {
                const bool ih_in_pad = (ih >= -static_cast<int64_t>(params.ph) &&
                                        ih < params.H + static_cast<int64_t>(params.ph));
                const bool iw_in_pad = (iw >= -static_cast<int64_t>(params.pw) &&
                                        iw < params.W + static_cast<int64_t>(params.pw));
                if (ih_in_pad && iw_in_pad) { count++; }
            }
        }
    }
    float divisor;
    if (params.divisor_override > 0) {
        divisor = static_cast<float>(params.divisor_override);
    } else {
        divisor = static_cast<float>(count);
    }
    const int64_t out_idx = ((static_cast<int64_t>(n) * params.OH + oh) * params.OW + ow) * params.C + c;
    output[out_idx] = static_cast<__nv_fp8_e4m3>((divisor > 0.0f) ? (sum / divisor) : 0.0f);
}

void avgpool_v1(const __nv_fp8_e4m3* input, __nv_fp8_e4m3* output, const AvgPoolParams& params, cudaStream_t stream) {
    NVTX_RANGE_PUSH_C("avgpool_v1_fp8e4m3", NVTX_COLOR_AVGPOOL);
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
    avgpool_v1_kernel_fp8_e4m3<TILE_OH, TILE_OW><<<grid, block, smem_bytes, stream>>>(
        input, output, params, blocks_oh, blocks_ow, smem_h, smem_w);
    CUDA_CHECK(cudaGetLastError());
    NVTX_RANGE_POP();
}

// --- v2: scalar loads (no native fp8 vectorization) ---
void avgpool_v2(const __nv_fp8_e4m3* input, __nv_fp8_e4m3* output, const AvgPoolParams& params, cudaStream_t stream) {
    NVTX_RANGE_PUSH_C("avgpool_v2_fp8e4m3", NVTX_COLOR_AVGPOOL);
    avgpool_v0(input, output, params, stream);
    NVTX_RANGE_POP();
}

// --- v3-v4: template instantiation ---
void avgpool_v3(const __nv_fp8_e4m3* input, __nv_fp8_e4m3* output, const AvgPoolParams& params, cudaStream_t stream) {
    NVTX_RANGE_PUSH_C("avgpool_v3_fp8e4m3", NVTX_COLOR_AVGPOOL);
    constexpr int BLOCK = 4;
    if (params.OH < BLOCK || params.OH % BLOCK != 0) { NVTX_RANGE_POP(); avgpool_v0(input, output, params, stream); return; }
    const int64_t OH_block = params.OH / BLOCK;
    const int64_t total = OH_block * params.OW * params.C;
    const int threads = 256;
    const int blocks_x = static_cast<int>((total + threads - 1) / threads);
    dim3 grid(blocks_x, 1, static_cast<int>(params.N));
    avgpool_v3_kernel<__nv_fp8_e4m3, BLOCK><<<grid, threads, 0, stream>>>(input, output, params);
    CUDA_CHECK(cudaGetLastError());
    NVTX_RANGE_POP();
}

void avgpool_v4(const __nv_fp8_e4m3* input, __nv_fp8_e4m3* output, const AvgPoolParams& params, cudaStream_t stream) {
    NVTX_RANGE_PUSH_C("avgpool_v4_fp8e4m3", NVTX_COLOR_AVGPOOL);
    const int64_t total = params.OH * params.OW * params.C;
    const int threads = 256;
    const int warps_per_block = threads / 32;
    const int blocks_x = static_cast<int>((total + warps_per_block - 1) / warps_per_block);
    dim3 grid(blocks_x, 1, static_cast<int>(params.N));
    avgpool_v4_kernel<__nv_fp8_e4m3><<<grid, threads, 0, stream>>>(input, output, params);
    CUDA_CHECK(cudaGetLastError());
    NVTX_RANGE_POP();
}

// --- v5: double buffer (fallback to v1) ---
void avgpool_v5(const __nv_fp8_e4m3* input, __nv_fp8_e4m3* output, const AvgPoolParams& params, cudaStream_t stream) {
    NVTX_RANGE_PUSH_C("avgpool_v5_fp8e4m3", NVTX_COLOR_AVGPOOL);
    avgpool_v1(input, output, params, stream);
    NVTX_RANGE_POP();
}

// --- v6: warp specialization (fallback to v1) ---
void avgpool_v6(const __nv_fp8_e4m3* input, __nv_fp8_e4m3* output, const AvgPoolParams& params, cudaStream_t stream) {
    NVTX_RANGE_PUSH_C("avgpool_v6_fp8e4m3", NVTX_COLOR_AVGPOOL);
    avgpool_v1(input, output, params, stream);
    NVTX_RANGE_POP();
}

// --- v7: alternative mappings ---
void avgpool_v7(const __nv_fp8_e4m3* input, __nv_fp8_e4m3* output, const AvgPoolParams& params, int mapping, cudaStream_t stream) {
    NVTX_RANGE_PUSH_C("avgpool_v7_fp8e4m3", NVTX_COLOR_AVGPOOL);
    switch (mapping) {
        case 0: { NVTX_RANGE_POP(); avgpool_v0(input, output, params, stream); return; }
        case 1: {
            const int C_groups = static_cast<int>((params.C + 3) / 4);
            const int64_t grid_z_64 = params.N * C_groups;
            if (grid_z_64 > 65535) { NVTX_RANGE_POP(); avgpool_v0(input, output, params, stream); return; }
            dim3 block(8, 8, 4);
            dim3 grid(static_cast<int>((params.OW + 7) / 8), static_cast<int>((params.OH + 7) / 8), static_cast<int>(grid_z_64));
            avgpool_v7_mappingB_kernel<__nv_fp8_e4m3><<<grid, block, 0, stream>>>(input, output, params, C_groups);
            CUDA_CHECK(cudaGetLastError()); break;
        }
        case 2: {
            const int C_groups = static_cast<int>((params.C + 255) / 256);
            const int64_t grid_z_64 = params.N * C_groups;
            if (params.OW > 65535 || params.OH > 65535 || grid_z_64 > 65535) { NVTX_RANGE_POP(); avgpool_v0(input, output, params, stream); return; }
            dim3 block(256, 1, 1);
            dim3 grid(static_cast<int>(params.OW), static_cast<int>(params.OH), static_cast<int>(grid_z_64));
            avgpool_v7_mappingC_kernel<__nv_fp8_e4m3><<<grid, block, 0, stream>>>(input, output, params, C_groups);
            CUDA_CHECK(cudaGetLastError()); break;
        }
        case 3: {
            if (params.C % 4 != 0) { NVTX_RANGE_POP(); avgpool_v0(input, output, params, stream); return; }
            const int C_groups = static_cast<int>((params.C + 31) / 32);
            const int64_t grid_z_64 = params.N * C_groups;
            if (grid_z_64 > 65535) { NVTX_RANGE_POP(); avgpool_v0(input, output, params, stream); return; }
            dim3 block(32, 8, 1);
            dim3 grid(static_cast<int>((params.OW + 3) / 4), static_cast<int>((params.OH + 3) / 4), static_cast<int>(grid_z_64));
            avgpool_v7_mappingD_kernel<__nv_fp8_e4m3><<<grid, block, 0, stream>>>(input, output, params, C_groups);
            CUDA_CHECK(cudaGetLastError()); break;
        }
        default:
            NVTX_RANGE_POP();
            throw std::invalid_argument("unsupported mapping: " + std::to_string(mapping));
    }
    NVTX_RANGE_POP();
}

// --- v8-v13: fallbacks ---
void avgpool_v8(const __nv_fp8_e4m3* input, __nv_fp8_e4m3* output, const AvgPoolParams& params, cudaStream_t stream) {
    NVTX_RANGE_PUSH_C("avgpool_v8_fp8e4m3", NVTX_COLOR_AVGPOOL);
    avgpool_v0(input, output, params, stream);
    NVTX_RANGE_POP();
}
void avgpool_v9(const __nv_fp8_e4m3* input, __nv_fp8_e4m3* output, const AvgPoolParams& params, cudaStream_t stream) {
    NVTX_RANGE_PUSH_C("avgpool_v9_fp8e4m3", NVTX_COLOR_AVGPOOL);
    avgpool_v0(input, output, params, stream);
    NVTX_RANGE_POP();
}
void avgpool_v10(const __nv_fp8_e4m3* input, __nv_fp8_e4m3* output, const AvgPoolParams& params, cudaStream_t stream) {
    NVTX_RANGE_PUSH_C("avgpool_v10_fp8e4m3", NVTX_COLOR_AVGPOOL);
    avgpool_v0(input, output, params, stream);
    NVTX_RANGE_POP();
}
void avgpool_v11(const __nv_fp8_e4m3* input, __nv_fp8_e4m3* output, const AvgPoolParams& params, cudaStream_t stream) {
    NVTX_RANGE_PUSH_C("avgpool_v11_fp8e4m3", NVTX_COLOR_AVGPOOL);
    avgpool_v4(input, output, params, stream);
    NVTX_RANGE_POP();
}
void avgpool_v12(const __nv_fp8_e4m3* input, __nv_fp8_e4m3* output, const AvgPoolParams& params, cudaStream_t stream) {
    NVTX_RANGE_PUSH_C("avgpool_v12_fp8e4m3", NVTX_COLOR_AVGPOOL);
    avgpool_v0(input, output, params, stream);
    NVTX_RANGE_POP();
}
void avgpool_v13(const __nv_fp8_e4m3* input, __nv_fp8_e4m3* output, const AvgPoolParams& params, cudaStream_t stream) {
    NVTX_RANGE_PUSH_C("avgpool_v13_fp8e4m3", NVTX_COLOR_AVGPOOL);
    avgpool_v0(input, output, params, stream);
    NVTX_RANGE_POP();
}

// --- v14: adaptive dispatcher ---
void avgpool_v14(const __nv_fp8_e4m3* input, __nv_fp8_e4m3* output, const AvgPoolParams& params, cudaStream_t stream) {
    NVTX_RANGE_PUSH_C("avgpool_v14_fp8e4m3", NVTX_COLOR_AVGPOOL);
    bool is_global = (params.kh >= params.H && params.kw >= params.W);
    if (is_global) { avgpool_v13(input, output, params, stream); NVTX_RANGE_POP(); return; }
    int64_t total_output_elems = params.N * params.OH * params.OW * params.C;
    if (total_output_elems < 65536) { avgpool_v10(input, output, params, stream); NVTX_RANGE_POP(); return; }
    if (params.sh == 1 && params.sw == 1 && params.kh == 3 && params.kw == 3) { avgpool_v15(input, output, params, stream); NVTX_RANGE_POP(); return; }
    avgpool_v0(input, output, params, stream);
    NVTX_RANGE_POP();
}

// --- v15: swizzled shared memory ---
void avgpool_v15(const __nv_fp8_e4m3* input, __nv_fp8_e4m3* output, const AvgPoolParams& params, cudaStream_t stream) {
    NVTX_RANGE_PUSH_C("avgpool_v15_fp8e4m3", NVTX_COLOR_AVGPOOL);
    constexpr int TILE_OH = 8; constexpr int TILE_OW = 8;
    const int blocks_oh = static_cast<int>((params.OH + TILE_OH - 1) / TILE_OH);
    const int blocks_ow = static_cast<int>((params.OW + TILE_OW - 1) / TILE_OW);
    int smem_h = (TILE_OH - 1) * params.sh + (params.kh - 1) * params.dh + 1;
    int smem_w = (TILE_OW - 1) * params.sw + (params.kw - 1) * params.dw + 1;
    size_t smem_bytes = static_cast<size_t>(smem_h) * (smem_w + 1) * sizeof(__nv_fp8_e4m3);
    PoolParams pp = make_pool_params_from_avg(params);
    dim3 block(TILE_OW, TILE_OH);
    dim3 grid(blocks_oh * blocks_ow, static_cast<int>(params.C), static_cast<int>(params.N));
    if (smem_bytes > 49152) {
        CUDA_CHECK(cudaFuncSetAttribute(maxpool_v15_kernel<__nv_fp8_e4m3, false, true>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem_bytes)));
        CUDA_CHECK(cudaFuncSetAttribute(maxpool_v15_kernel<__nv_fp8_e4m3, false, false>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem_bytes)));
    }
    if (params.count_include_pad) {
        maxpool_v15_kernel<__nv_fp8_e4m3, false, true><<<grid, block, smem_bytes, stream>>>(
            reinterpret_cast<const __nv_fp8_e4m3*>(input), reinterpret_cast<__nv_fp8_e4m3*>(output),
            pp, blocks_oh, blocks_ow, smem_h, smem_w);
    } else {
        maxpool_v15_kernel<__nv_fp8_e4m3, false, false><<<grid, block, smem_bytes, stream>>>(
            reinterpret_cast<const __nv_fp8_e4m3*>(input), reinterpret_cast<__nv_fp8_e4m3*>(output),
            pp, blocks_oh, blocks_ow, smem_h, smem_w);
    }
    CUDA_CHECK(cudaGetLastError());
    NVTX_RANGE_POP();
}

// ============================================================================
// FP8 E5M2 AvgPool kernels
// ============================================================================

// --- v0: template instantiation ---
void avgpool_v0(const __nv_fp8_e5m2* input, __nv_fp8_e5m2* output, const AvgPoolParams& params, cudaStream_t stream) {
    NVTX_RANGE_PUSH_C("avgpool_v0_fp8e5m2", NVTX_COLOR_AVGPOOL);
    const int64_t total = params.OH * params.OW * params.C;
    const int threads = 256;
    const int blocks_x = static_cast<int>((total + threads - 1) / threads);
    dim3 grid(blocks_x, 1, static_cast<int>(params.N));
    avgpool_v0_kernel<__nv_fp8_e5m2><<<grid, threads, 0, stream>>>(input, output, params);
    CUDA_CHECK(cudaGetLastError());
    NVTX_RANGE_POP();
}

// --- v1: explicit kernel (float shared memory) ---
template <int TILE_OH = 8, int TILE_OW = 8>
__global__ void avgpool_v1_kernel_fp8_e5m2(
    const __nv_fp8_e5m2* __restrict__ input,
    __nv_fp8_e5m2* __restrict__ output,
    const AvgPoolParams params,
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
            sdata[i] = 0.0f;
        }
    }
    __syncthreads();
    if (oh >= params.OH || ow >= params.OW) return;
    float sum = 0.0f;
    int count = 0;
    for (int kh_i = 0; kh_i < params.kh; ++kh_i) {
        const int sih = th * params.sh + kh_i * params.dh;
        if (sih >= smem_h) continue;
        const int64_t ih = static_cast<int64_t>(ih_start) + sih;
        for (int kw_i = 0; kw_i < params.kw; ++kw_i) {
            const int siw = tw * params.sw + kw_i * params.dw;
            if (siw >= smem_w) continue;
            const int64_t iw = static_cast<int64_t>(iw_start) + siw;
            const bool ih_in = (ih >= 0 && ih < params.H);
            const bool iw_in = (iw >= 0 && iw < params.W);
            if (ih_in && iw_in) {
                sum += sdata[sih * smem_w + siw];
                count++;
            } else if (params.count_include_pad) {
                const bool ih_in_pad = (ih >= -static_cast<int64_t>(params.ph) &&
                                        ih < params.H + static_cast<int64_t>(params.ph));
                const bool iw_in_pad = (iw >= -static_cast<int64_t>(params.pw) &&
                                        iw < params.W + static_cast<int64_t>(params.pw));
                if (ih_in_pad && iw_in_pad) { count++; }
            }
        }
    }
    float divisor;
    if (params.divisor_override > 0) {
        divisor = static_cast<float>(params.divisor_override);
    } else {
        divisor = static_cast<float>(count);
    }
    const int64_t out_idx = ((static_cast<int64_t>(n) * params.OH + oh) * params.OW + ow) * params.C + c;
    output[out_idx] = static_cast<__nv_fp8_e5m2>((divisor > 0.0f) ? (sum / divisor) : 0.0f);
}

void avgpool_v1(const __nv_fp8_e5m2* input, __nv_fp8_e5m2* output, const AvgPoolParams& params, cudaStream_t stream) {
    NVTX_RANGE_PUSH_C("avgpool_v1_fp8e5m2", NVTX_COLOR_AVGPOOL);
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
    avgpool_v1_kernel_fp8_e5m2<TILE_OH, TILE_OW><<<grid, block, smem_bytes, stream>>>(
        input, output, params, blocks_oh, blocks_ow, smem_h, smem_w);
    CUDA_CHECK(cudaGetLastError());
    NVTX_RANGE_POP();
}

// --- v2: scalar loads (no native fp8 vectorization) ---
void avgpool_v2(const __nv_fp8_e5m2* input, __nv_fp8_e5m2* output, const AvgPoolParams& params, cudaStream_t stream) {
    NVTX_RANGE_PUSH_C("avgpool_v2_fp8e5m2", NVTX_COLOR_AVGPOOL);
    avgpool_v0(input, output, params, stream);
    NVTX_RANGE_POP();
}

// --- v3-v4: template instantiation ---
void avgpool_v3(const __nv_fp8_e5m2* input, __nv_fp8_e5m2* output, const AvgPoolParams& params, cudaStream_t stream) {
    NVTX_RANGE_PUSH_C("avgpool_v3_fp8e5m2", NVTX_COLOR_AVGPOOL);
    constexpr int BLOCK = 4;
    if (params.OH < BLOCK || params.OH % BLOCK != 0) { NVTX_RANGE_POP(); avgpool_v0(input, output, params, stream); return; }
    const int64_t OH_block = params.OH / BLOCK;
    const int64_t total = OH_block * params.OW * params.C;
    const int threads = 256;
    const int blocks_x = static_cast<int>((total + threads - 1) / threads);
    dim3 grid(blocks_x, 1, static_cast<int>(params.N));
    avgpool_v3_kernel<__nv_fp8_e5m2, BLOCK><<<grid, threads, 0, stream>>>(input, output, params);
    CUDA_CHECK(cudaGetLastError());
    NVTX_RANGE_POP();
}

void avgpool_v4(const __nv_fp8_e5m2* input, __nv_fp8_e5m2* output, const AvgPoolParams& params, cudaStream_t stream) {
    NVTX_RANGE_PUSH_C("avgpool_v4_fp8e5m2", NVTX_COLOR_AVGPOOL);
    const int64_t total = params.OH * params.OW * params.C;
    const int threads = 256;
    const int warps_per_block = threads / 32;
    const int blocks_x = static_cast<int>((total + warps_per_block - 1) / warps_per_block);
    dim3 grid(blocks_x, 1, static_cast<int>(params.N));
    avgpool_v4_kernel<__nv_fp8_e5m2><<<grid, threads, 0, stream>>>(input, output, params);
    CUDA_CHECK(cudaGetLastError());
    NVTX_RANGE_POP();
}

// --- v5: double buffer (fallback to v1) ---
void avgpool_v5(const __nv_fp8_e5m2* input, __nv_fp8_e5m2* output, const AvgPoolParams& params, cudaStream_t stream) {
    NVTX_RANGE_PUSH_C("avgpool_v5_fp8e5m2", NVTX_COLOR_AVGPOOL);
    avgpool_v1(input, output, params, stream);
    NVTX_RANGE_POP();
}

// --- v6: warp specialization (fallback to v1) ---
void avgpool_v6(const __nv_fp8_e5m2* input, __nv_fp8_e5m2* output, const AvgPoolParams& params, cudaStream_t stream) {
    NVTX_RANGE_PUSH_C("avgpool_v6_fp8e5m2", NVTX_COLOR_AVGPOOL);
    avgpool_v1(input, output, params, stream);
    NVTX_RANGE_POP();
}

// --- v7: alternative mappings ---
void avgpool_v7(const __nv_fp8_e5m2* input, __nv_fp8_e5m2* output, const AvgPoolParams& params, int mapping, cudaStream_t stream) {
    NVTX_RANGE_PUSH_C("avgpool_v7_fp8e5m2", NVTX_COLOR_AVGPOOL);
    switch (mapping) {
        case 0: { NVTX_RANGE_POP(); avgpool_v0(input, output, params, stream); return; }
        case 1: {
            const int C_groups = static_cast<int>((params.C + 3) / 4);
            const int64_t grid_z_64 = params.N * C_groups;
            if (grid_z_64 > 65535) { NVTX_RANGE_POP(); avgpool_v0(input, output, params, stream); return; }
            dim3 block(8, 8, 4);
            dim3 grid(static_cast<int>((params.OW + 7) / 8), static_cast<int>((params.OH + 7) / 8), static_cast<int>(grid_z_64));
            avgpool_v7_mappingB_kernel<__nv_fp8_e5m2><<<grid, block, 0, stream>>>(input, output, params, C_groups);
            CUDA_CHECK(cudaGetLastError()); break;
        }
        case 2: {
            const int C_groups = static_cast<int>((params.C + 255) / 256);
            const int64_t grid_z_64 = params.N * C_groups;
            if (params.OW > 65535 || params.OH > 65535 || grid_z_64 > 65535) { NVTX_RANGE_POP(); avgpool_v0(input, output, params, stream); return; }
            dim3 block(256, 1, 1);
            dim3 grid(static_cast<int>(params.OW), static_cast<int>(params.OH), static_cast<int>(grid_z_64));
            avgpool_v7_mappingC_kernel<__nv_fp8_e5m2><<<grid, block, 0, stream>>>(input, output, params, C_groups);
            CUDA_CHECK(cudaGetLastError()); break;
        }
        case 3: {
            if (params.C % 4 != 0) { NVTX_RANGE_POP(); avgpool_v0(input, output, params, stream); return; }
            const int C_groups = static_cast<int>((params.C + 31) / 32);
            const int64_t grid_z_64 = params.N * C_groups;
            if (grid_z_64 > 65535) { NVTX_RANGE_POP(); avgpool_v0(input, output, params, stream); return; }
            dim3 block(32, 8, 1);
            dim3 grid(static_cast<int>((params.OW + 3) / 4), static_cast<int>((params.OH + 3) / 4), static_cast<int>(grid_z_64));
            avgpool_v7_mappingD_kernel<__nv_fp8_e5m2><<<grid, block, 0, stream>>>(input, output, params, C_groups);
            CUDA_CHECK(cudaGetLastError()); break;
        }
        default:
            NVTX_RANGE_POP();
            throw std::invalid_argument("unsupported mapping: " + std::to_string(mapping));
    }
    NVTX_RANGE_POP();
}

// --- v8-v13: fallbacks ---
void avgpool_v8(const __nv_fp8_e5m2* input, __nv_fp8_e5m2* output, const AvgPoolParams& params, cudaStream_t stream) {
    NVTX_RANGE_PUSH_C("avgpool_v8_fp8e5m2", NVTX_COLOR_AVGPOOL);
    avgpool_v0(input, output, params, stream);
    NVTX_RANGE_POP();
}
void avgpool_v9(const __nv_fp8_e5m2* input, __nv_fp8_e5m2* output, const AvgPoolParams& params, cudaStream_t stream) {
    NVTX_RANGE_PUSH_C("avgpool_v9_fp8e5m2", NVTX_COLOR_AVGPOOL);
    avgpool_v0(input, output, params, stream);
    NVTX_RANGE_POP();
}
void avgpool_v10(const __nv_fp8_e5m2* input, __nv_fp8_e5m2* output, const AvgPoolParams& params, cudaStream_t stream) {
    NVTX_RANGE_PUSH_C("avgpool_v10_fp8e5m2", NVTX_COLOR_AVGPOOL);
    avgpool_v0(input, output, params, stream);
    NVTX_RANGE_POP();
}
void avgpool_v11(const __nv_fp8_e5m2* input, __nv_fp8_e5m2* output, const AvgPoolParams& params, cudaStream_t stream) {
    NVTX_RANGE_PUSH_C("avgpool_v11_fp8e5m2", NVTX_COLOR_AVGPOOL);
    avgpool_v4(input, output, params, stream);
    NVTX_RANGE_POP();
}
void avgpool_v12(const __nv_fp8_e5m2* input, __nv_fp8_e5m2* output, const AvgPoolParams& params, cudaStream_t stream) {
    NVTX_RANGE_PUSH_C("avgpool_v12_fp8e5m2", NVTX_COLOR_AVGPOOL);
    avgpool_v0(input, output, params, stream);
    NVTX_RANGE_POP();
}
void avgpool_v13(const __nv_fp8_e5m2* input, __nv_fp8_e5m2* output, const AvgPoolParams& params, cudaStream_t stream) {
    NVTX_RANGE_PUSH_C("avgpool_v13_fp8e5m2", NVTX_COLOR_AVGPOOL);
    avgpool_v0(input, output, params, stream);
    NVTX_RANGE_POP();
}

// --- v14: adaptive dispatcher ---
void avgpool_v14(const __nv_fp8_e5m2* input, __nv_fp8_e5m2* output, const AvgPoolParams& params, cudaStream_t stream) {
    NVTX_RANGE_PUSH_C("avgpool_v14_fp8e5m2", NVTX_COLOR_AVGPOOL);
    bool is_global = (params.kh >= params.H && params.kw >= params.W);
    if (is_global) { avgpool_v13(input, output, params, stream); NVTX_RANGE_POP(); return; }
    int64_t total_output_elems = params.N * params.OH * params.OW * params.C;
    if (total_output_elems < 65536) { avgpool_v10(input, output, params, stream); NVTX_RANGE_POP(); return; }
    if (params.sh == 1 && params.sw == 1 && params.kh == 3 && params.kw == 3) { avgpool_v15(input, output, params, stream); NVTX_RANGE_POP(); return; }
    avgpool_v0(input, output, params, stream);
    NVTX_RANGE_POP();
}

// --- v15: swizzled shared memory ---
void avgpool_v15(const __nv_fp8_e5m2* input, __nv_fp8_e5m2* output, const AvgPoolParams& params, cudaStream_t stream) {
    NVTX_RANGE_PUSH_C("avgpool_v15_fp8e5m2", NVTX_COLOR_AVGPOOL);
    constexpr int TILE_OH = 8; constexpr int TILE_OW = 8;
    const int blocks_oh = static_cast<int>((params.OH + TILE_OH - 1) / TILE_OH);
    const int blocks_ow = static_cast<int>((params.OW + TILE_OW - 1) / TILE_OW);
    int smem_h = (TILE_OH - 1) * params.sh + (params.kh - 1) * params.dh + 1;
    int smem_w = (TILE_OW - 1) * params.sw + (params.kw - 1) * params.dw + 1;
    size_t smem_bytes = static_cast<size_t>(smem_h) * (smem_w + 1) * sizeof(__nv_fp8_e5m2);
    PoolParams pp = make_pool_params_from_avg(params);
    dim3 block(TILE_OW, TILE_OH);
    dim3 grid(blocks_oh * blocks_ow, static_cast<int>(params.C), static_cast<int>(params.N));
    if (smem_bytes > 49152) {
        CUDA_CHECK(cudaFuncSetAttribute(maxpool_v15_kernel<__nv_fp8_e5m2, false, true>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem_bytes)));
        CUDA_CHECK(cudaFuncSetAttribute(maxpool_v15_kernel<__nv_fp8_e5m2, false, false>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem_bytes)));
    }
    if (params.count_include_pad) {
        maxpool_v15_kernel<__nv_fp8_e5m2, false, true><<<grid, block, smem_bytes, stream>>>(
            reinterpret_cast<const __nv_fp8_e5m2*>(input), reinterpret_cast<__nv_fp8_e5m2*>(output),
            pp, blocks_oh, blocks_ow, smem_h, smem_w);
    } else {
        maxpool_v15_kernel<__nv_fp8_e5m2, false, false><<<grid, block, smem_bytes, stream>>>(
            reinterpret_cast<const __nv_fp8_e5m2*>(input), reinterpret_cast<__nv_fp8_e5m2*>(output),
            pp, blocks_oh, blocks_ow, smem_h, smem_w);
    }
    CUDA_CHECK(cudaGetLastError());
    NVTX_RANGE_POP();
}
