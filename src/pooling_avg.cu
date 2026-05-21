#include "pooling.cuh"
#include <mutex>
#include <unordered_map>

// AvgPool2d v0: naive kernel — one thread per output element
// Grid:  blockIdx.z = batch index N, blockIdx.x * blockDim.x + threadIdx.x = flat (OH*OW*C)
template <typename T>
__global__ void avgpool_v0_kernel(
    const T* __restrict__ input,
    T* __restrict__ output,
    const AvgPoolParams params)
{
    const int64_t n = blockIdx.z;
    const int64_t flat = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t OHW_C = params.OH * params.OW * params.C;
    if (n >= params.N || flat >= OHW_C) return;

    const int64_t oh = flat / (params.OW * params.C);
    const int64_t ow = (flat / params.C) % params.OW;
    const int64_t c  = flat % params.C;

    float sum = 0.0f;
    int count = 0;

    for (int kh_i = 0; kh_i < params.kh; ++kh_i) {
        const int64_t ih = oh * params.sh - params.ph + static_cast<int64_t>(kh_i) * params.dh;
        const bool ih_in = (ih >= 0 && ih < params.H);
        for (int kw_i = 0; kw_i < params.kw; ++kw_i) {
            const int64_t iw = ow * params.sw - params.pw + static_cast<int64_t>(kw_i) * params.dw;
            const bool iw_in = (iw >= 0 && iw < params.W);

            if (ih_in && iw_in) {
                const int64_t in_idx = ((n * params.H + ih) * params.W + iw) * params.C + c;
                sum += static_cast<float>(input[in_idx]);
                count++;
            } else if (params.count_include_pad) {
                // A padded zero: position is within the padded input region
                // but not within the actual input. When ceil_mode extends the
                // window beyond the padded region, those pixels are NOT counted.
                const bool ih_in_pad = (ih >= -static_cast<int64_t>(params.ph) &&
                                        ih < params.H + static_cast<int64_t>(params.ph));
                const bool iw_in_pad = (iw >= -static_cast<int64_t>(params.pw) &&
                                        iw < params.W + static_cast<int64_t>(params.pw));
                if (ih_in_pad && iw_in_pad) {
                    count++;
                }
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
    output[out_idx] = static_cast<T>(sum / divisor);
}

void avgpool_v0(const float* input, float* output, const AvgPoolParams& params, cudaStream_t stream) {
    NVTX_RANGE_PUSH_C("avgpool_v0_f32", NVTX_COLOR_AVGPOOL);
    const int64_t total = params.OH * params.OW * params.C;
    const int threads = 256;
    const int blocks_x = static_cast<int>((total + threads - 1) / threads);
    dim3 grid(blocks_x, 1, static_cast<int>(params.N));
    avgpool_v0_kernel<float><<<grid, threads, 0, stream>>>(input, output, params);
    CUDA_CHECK(cudaGetLastError());
    NVTX_RANGE_POP();
}

void avgpool_v0(const half* input, half* output, const AvgPoolParams& params, cudaStream_t stream) {
    NVTX_RANGE_PUSH_C("avgpool_v0_f16", NVTX_COLOR_AVGPOOL);
    const int64_t total = params.OH * params.OW * params.C;
    const int threads = 256;
    const int blocks_x = static_cast<int>((total + threads - 1) / threads);
    dim3 grid(blocks_x, 1, static_cast<int>(params.N));
    avgpool_v0_kernel<half><<<grid, threads, 0, stream>>>(input, output, params);
    CUDA_CHECK(cudaGetLastError());
    NVTX_RANGE_POP();
}

// ============================================================================
// AvgPool2d v1: shared memory tiling kernel
// Each block handles a TILE_OH x TILE_OW tile of output spatial positions
// for one (n, c) pair. The block cooperatively loads the corresponding
// input tile (output tile + halo region) into shared memory, then each
// thread computes its average from shared memory.
// ============================================================================

template <int TILE_OH = 8, int TILE_OW = 8>
__global__ void avgpool_v1_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
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

    const int th = threadIdx.y;  // 0..TILE_OH-1
    const int tw = threadIdx.x;  // 0..TILE_OW-1

    // Global output position
    const int oh = tile_oh * TILE_OH + th;
    const int ow = tile_ow * TILE_OW + tw;

    // Input tile origin in global coordinates
    const int ih_start = tile_oh * TILE_OH * params.sh - params.ph;
    const int iw_start = tile_ow * TILE_OW * params.sw - params.pw;

    // Cooperative loading of input tile into shared memory
    const int total_smem = smem_h * smem_w;
    const int tid = th * TILE_OW + tw;
    for (int i = tid; i < total_smem; i += TILE_OH * TILE_OW) {
        const int sih = i / smem_w;
        const int siw = i % smem_w;
        const int ih = ih_start + sih;
        const int iw = iw_start + siw;
        if (ih >= 0 && ih < params.H && iw >= 0 && iw < params.W) {
            const int64_t in_idx = ((static_cast<int64_t>(n) * params.H + ih) * params.W + iw) * params.C + c;
            sdata[i] = input[in_idx];
        } else {
            sdata[i] = 0.0f;  // Padded positions contribute 0 to sum
        }
    }

    __syncthreads();

    // Each thread computes one output position
    if (oh >= params.OH || ow >= params.OW) return;

    float sum = 0.0f;
    int count = 0;

    for (int kh_i = 0; kh_i < params.kh; ++kh_i) {
        const int sih = th * params.sh + kh_i * params.dh;
        if (sih >= smem_h) continue;  // beyond smem => input out of bounds
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
                // Check if within the padded region (not beyond the padding zone)
                const bool ih_in_pad = (ih >= -static_cast<int64_t>(params.ph) &&
                                        ih < params.H + static_cast<int64_t>(params.ph));
                const bool iw_in_pad = (iw >= -static_cast<int64_t>(params.pw) &&
                                        iw < params.W + static_cast<int64_t>(params.pw));
                if (ih_in_pad && iw_in_pad) {
                    count++;
                }
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
    output[out_idx] = (divisor > 0.0f) ? (sum / divisor) : 0.0f;
}

// half specialization: same logic, casts input/output via float
template <int TILE_OH = 8, int TILE_OW = 8>
__global__ void avgpool_v1_kernel_half(
    const half* __restrict__ input,
    half* __restrict__ output,
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
                if (ih_in_pad && iw_in_pad) {
                    count++;
                }
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
    output[out_idx] = static_cast<half>((divisor > 0.0f) ? (sum / divisor) : 0.0f);
}

void avgpool_v1(const float* input, float* output, const AvgPoolParams& params, cudaStream_t stream) {
    NVTX_RANGE_PUSH_C("avgpool_v1_f32", NVTX_COLOR_AVGPOOL);
    constexpr int TILE_OH = 8;
    constexpr int TILE_OW = 8;

    const int blocks_oh = static_cast<int>((params.OH + TILE_OH - 1) / TILE_OH);
    const int blocks_ow = static_cast<int>((params.OW + TILE_OW - 1) / TILE_OW);

    // Shared memory tile dimensions, capped at input extent (including padding)
    // to avoid excessive smem allocation for large strides
    int smem_h = (TILE_OH - 1) * params.sh + (params.kh - 1) * params.dh + 1;
    int smem_w = (TILE_OW - 1) * params.sw + (params.kw - 1) * params.dw + 1;
    smem_h = min(smem_h, static_cast<int>(params.H + 2 * params.ph));
    smem_w = min(smem_w, static_cast<int>(params.W + 2 * params.pw));

    dim3 block(TILE_OW, TILE_OH);
    dim3 grid(blocks_oh * blocks_ow, static_cast<int>(params.C), static_cast<int>(params.N));
    size_t smem_bytes = static_cast<size_t>(smem_h) * smem_w * sizeof(float);

    avgpool_v1_kernel<TILE_OH, TILE_OW><<<grid, block, smem_bytes, stream>>>(
        input, output, params, blocks_oh, blocks_ow, smem_h, smem_w);
    CUDA_CHECK(cudaGetLastError());
    NVTX_RANGE_POP();
}

void avgpool_v1(const half* input, half* output, const AvgPoolParams& params, cudaStream_t stream) {
    NVTX_RANGE_PUSH_C("avgpool_v1_f16", NVTX_COLOR_AVGPOOL);
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

    avgpool_v1_kernel_half<TILE_OH, TILE_OW><<<grid, block, smem_bytes, stream>>>(
        input, output, params, blocks_oh, blocks_ow, smem_h, smem_w);
    CUDA_CHECK(cudaGetLastError());
    NVTX_RANGE_POP();
}

// ============================================================================
// AvgPool2d v2: vectorized loads kernel
// Uses float4 (4 channels) for fp32 and half2 (2 channels) for fp16
// to coalesce global memory access. Falls back to v0 if C % VEC != 0.
// ============================================================================

template <typename T, int VEC>
__global__ void avgpool_v2_kernel(
    const T* __restrict__ input,
    T* __restrict__ output,
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
    for (int v = 0; v < VEC; ++v)
        sum[v] = 0.0f;

    int count = 0;

    for (int kh_i = 0; kh_i < params.kh; ++kh_i) {
        const int64_t ih = oh * params.sh - params.ph + static_cast<int64_t>(kh_i) * params.dh;
        const bool ih_in = (ih >= 0 && ih < params.H);
        for (int kw_i = 0; kw_i < params.kw; ++kw_i) {
            const int64_t iw = ow * params.sw - params.pw + static_cast<int64_t>(kw_i) * params.dw;
            const bool iw_in = (iw >= 0 && iw < params.W);

            if (ih_in && iw_in) {
                const int64_t in_idx = ((n * params.H + ih) * params.W + iw) * params.C + c;

                // Vectorized load
                if constexpr (std::is_same_v<T, float> && VEC == 4) {
                    float4 vec = *reinterpret_cast<const float4*>(&input[in_idx]);
                    sum[0] += vec.x;
                    sum[1] += vec.y;
                    sum[2] += vec.z;
                    sum[3] += vec.w;
                } else if constexpr (std::is_same_v<T, half> && VEC == 2) {
                    half2 vec = *reinterpret_cast<const half2*>(&input[in_idx]);
                    sum[0] += __low2float(vec);
                    sum[1] += __high2float(vec);
                }
                count++;
            } else if (params.count_include_pad) {
                const bool ih_in_pad = (ih >= -static_cast<int64_t>(params.ph) &&
                                        ih < params.H + static_cast<int64_t>(params.ph));
                const bool iw_in_pad = (iw >= -static_cast<int64_t>(params.pw) &&
                                        iw < params.W + static_cast<int64_t>(params.pw));
                if (ih_in_pad && iw_in_pad) {
                    count++;
                }
            }
        }
    }

    float divisor;
    if (params.divisor_override > 0) {
        divisor = static_cast<float>(params.divisor_override);
    } else {
        divisor = static_cast<float>(count);
    }

    // Write VEC output values
    const int64_t out_idx = ((n * params.OH + oh) * params.OW + ow) * params.C + c;

    if constexpr (std::is_same_v<T, float> && VEC == 4) {
        float4 out_vec;
        out_vec.x = (divisor > 0.0f) ? (sum[0] / divisor) : 0.0f;
        out_vec.y = (divisor > 0.0f) ? (sum[1] / divisor) : 0.0f;
        out_vec.z = (divisor > 0.0f) ? (sum[2] / divisor) : 0.0f;
        out_vec.w = (divisor > 0.0f) ? (sum[3] / divisor) : 0.0f;
        *reinterpret_cast<float4*>(&output[out_idx]) = out_vec;
    } else if constexpr (std::is_same_v<T, half> && VEC == 2) {
        float r0 = (divisor > 0.0f) ? (sum[0] / divisor) : 0.0f;
        float r1 = (divisor > 0.0f) ? (sum[1] / divisor) : 0.0f;
        half2 out_vec = __floats2half2_rn(r0, r1);
        *reinterpret_cast<half2*>(&output[out_idx]) = out_vec;
    }
}

void avgpool_v2(const float* input, float* output, const AvgPoolParams& params, cudaStream_t stream) {
    NVTX_RANGE_PUSH_C("avgpool_v2_f32", NVTX_COLOR_AVGPOOL);
    constexpr int VEC = 4;
    if (params.C % VEC != 0) {
        NVTX_RANGE_POP();
        avgpool_v0(input, output, params, stream);
        return;
    }
    const int64_t C_vec = params.C / VEC;
    const int64_t total = params.OH * params.OW * C_vec;
    const int threads = 256;
    const int blocks_x = static_cast<int>((total + threads - 1) / threads);
    dim3 grid(blocks_x, 1, static_cast<int>(params.N));
    avgpool_v2_kernel<float, VEC><<<grid, threads, 0, stream>>>(input, output, params);
    CUDA_CHECK(cudaGetLastError());
    NVTX_RANGE_POP();
}

void avgpool_v2(const half* input, half* output, const AvgPoolParams& params, cudaStream_t stream) {
    NVTX_RANGE_PUSH_C("avgpool_v2_f16", NVTX_COLOR_AVGPOOL);
    constexpr int VEC = 2;
    if (params.C % VEC != 0) {
        NVTX_RANGE_POP();
        avgpool_v0(input, output, params, stream);
        return;
    }
    const int64_t C_vec = params.C / VEC;
    const int64_t total = params.OH * params.OW * C_vec;
    const int threads = 256;
    const int blocks_x = static_cast<int>((total + threads - 1) / threads);
    dim3 grid(blocks_x, 1, static_cast<int>(params.N));
    avgpool_v2_kernel<half, VEC><<<grid, threads, 0, stream>>>(input, output, params);
    CUDA_CHECK(cudaGetLastError());
    NVTX_RANGE_POP();
}

// ============================================================================
// AvgPool2d v3: register blocking kernel
// Each thread computes BLOCK consecutive output rows for the same (ow, c).
// Each output position maintains its own sum and count to correctly handle
// count_include_pad and divisor_override per position.
// Falls back to v0 if OH < BLOCK or OH % BLOCK != 0.
// ============================================================================

template <typename T, int BLOCK>
__global__ void avgpool_v3_kernel(
    const T* __restrict__ input,
    T* __restrict__ output,
    const AvgPoolParams params)
{
    const int64_t n = blockIdx.z;
    const int64_t OH_block = params.OH / BLOCK;
    const int64_t total = OH_block * params.OW * params.C;
    const int64_t flat = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (n >= params.N || flat >= total) return;

    const int64_t c = flat % params.C;
    const int64_t ow = (flat / params.C) % params.OW;
    const int64_t oh_base = (flat / params.C / params.OW) * BLOCK;

    float sum[BLOCK];
    int count[BLOCK];
    #pragma unroll
    for (int b = 0; b < BLOCK; ++b) {
        sum[b] = 0.0f;
        count[b] = 0;
    }

    for (int kh_i = 0; kh_i < params.kh; ++kh_i) {
        for (int kw_i = 0; kw_i < params.kw; ++kw_i) {
            const int64_t iw = ow * params.sw - params.pw + static_cast<int64_t>(kw_i) * params.dw;
            const bool iw_in = (iw >= 0 && iw < params.W);
            const bool iw_in_pad = (iw >= -static_cast<int64_t>(params.pw) &&
                                     iw < params.W + static_cast<int64_t>(params.pw));

            #pragma unroll
            for (int b = 0; b < BLOCK; ++b) {
                const int64_t oh = oh_base + b;
                const int64_t ih = oh * params.sh - params.ph + static_cast<int64_t>(kh_i) * params.dh;
                const bool ih_in = (ih >= 0 && ih < params.H);

                if (ih_in && iw_in) {
                    const int64_t in_idx = ((n * params.H + ih) * params.W + iw) * params.C + c;
                    sum[b] += static_cast<float>(input[in_idx]);
                    count[b]++;
                } else if (params.count_include_pad) {
                    const bool ih_in_pad = (ih >= -static_cast<int64_t>(params.ph) &&
                                             ih < params.H + static_cast<int64_t>(params.ph));
                    if (ih_in_pad && iw_in_pad) {
                        count[b]++;
                    }
                }
            }
        }
    }

    #pragma unroll
    for (int b = 0; b < BLOCK; ++b) {
        const int64_t oh = oh_base + b;
        float divisor;
        if (params.divisor_override > 0) {
            divisor = static_cast<float>(params.divisor_override);
        } else {
            divisor = static_cast<float>(count[b]);
        }
        const int64_t out_idx = ((n * params.OH + oh) * params.OW + ow) * params.C + c;
        output[out_idx] = static_cast<T>((divisor > 0.0f) ? (sum[b] / divisor) : 0.0f);
    }
}

void avgpool_v3(const float* input, float* output, const AvgPoolParams& params, cudaStream_t stream) {
    NVTX_RANGE_PUSH_C("avgpool_v3_f32", NVTX_COLOR_AVGPOOL);
    constexpr int BLOCK = 4;
    if (params.OH < BLOCK || params.OH % BLOCK != 0) {
        NVTX_RANGE_POP();
        avgpool_v0(input, output, params, stream);
        return;
    }
    const int64_t OH_block = params.OH / BLOCK;
    const int64_t total = OH_block * params.OW * params.C;
    const int threads = 256;
    const int blocks_x = static_cast<int>((total + threads - 1) / threads);
    dim3 grid(blocks_x, 1, static_cast<int>(params.N));
    avgpool_v3_kernel<float, BLOCK><<<grid, threads, 0, stream>>>(input, output, params);
    CUDA_CHECK(cudaGetLastError());
    NVTX_RANGE_POP();
}

void avgpool_v3(const half* input, half* output, const AvgPoolParams& params, cudaStream_t stream) {
    NVTX_RANGE_PUSH_C("avgpool_v3_f16", NVTX_COLOR_AVGPOOL);
    constexpr int BLOCK = 4;
    if (params.OH < BLOCK || params.OH % BLOCK != 0) {
        NVTX_RANGE_POP();
        avgpool_v0(input, output, params, stream);
        return;
    }
    const int64_t OH_block = params.OH / BLOCK;
    const int64_t total = OH_block * params.OW * params.C;
    const int threads = 256;
    const int blocks_x = static_cast<int>((total + threads - 1) / threads);
    dim3 grid(blocks_x, 1, static_cast<int>(params.N));
    avgpool_v3_kernel<half, BLOCK><<<grid, threads, 0, stream>>>(input, output, params);
    CUDA_CHECK(cudaGetLastError());
    NVTX_RANGE_POP();
}

// ============================================================================
// AvgPool2d v4: warp-level reduce kernel
// Each warp (32 threads) cooperatively handles one output position (n, oh, ow, c).
// The karea = kh*kw elements of the kernel window are distributed across the 32
// lanes of the warp. Each lane accumulates a local sum and count, then warp
// shuffle reductions compute the final sum and count.
// For large kernels (karea > 32), each lane handles multiple elements.
// For small kernels (karea < 32), idle lanes contribute 0 (sum/count identity).
// ============================================================================

template <typename T>
__global__ void avgpool_v4_kernel(
    const T* __restrict__ input,
    T* __restrict__ output,
    const AvgPoolParams params)
{
    const int64_t n = blockIdx.z;
    const int warp_id = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int64_t flat = static_cast<int64_t>(blockIdx.x) * (blockDim.x / 32) + warp_id;
    const int64_t OHW_C = params.OH * params.OW * params.C;
    if (n >= params.N || flat >= OHW_C) return;

    const int64_t oh = flat / (params.OW * params.C);
    const int64_t ow = (flat / params.C) % params.OW;
    const int64_t c  = flat % params.C;

    const int karea = params.kh * params.kw;
    float mysum = 0.0f;
    int mycount = 0;

    // Each lane handles its subset of kernel positions
    for (int ki = lane; ki < karea; ki += 32) {
        const int kh_idx = ki / params.kw;
        const int kw_idx = ki % params.kw;
        const int64_t ih = oh * params.sh - params.ph + static_cast<int64_t>(kh_idx) * params.dh;
        const int64_t iw = ow * params.sw - params.pw + static_cast<int64_t>(kw_idx) * params.dw;

        const bool ih_in = (ih >= 0 && ih < params.H);
        const bool iw_in = (iw >= 0 && iw < params.W);

        if (ih_in && iw_in) {
            const int64_t in_idx = ((n * params.H + ih) * params.W + iw) * params.C + c;
            mysum += static_cast<float>(input[in_idx]);
            mycount++;
        } else if (params.count_include_pad) {
            const bool ih_in_pad = (ih >= -static_cast<int64_t>(params.ph) &&
                                    ih < params.H + static_cast<int64_t>(params.ph));
            const bool iw_in_pad = (iw >= -static_cast<int64_t>(params.pw) &&
                                    iw < params.W + static_cast<int64_t>(params.pw));
            if (ih_in_pad && iw_in_pad) {
                mycount++;
            }
        }
    }

    // Warp-level sum reduction
    for (int offset = 16; offset > 0; offset >>= 1) {
        mysum += __shfl_down_sync(0xFFFFFFFF, mysum, offset);
    }

    // Warp-level count reduction using float shuffle and casting
    float mycount_f = static_cast<float>(mycount);
    for (int offset = 16; offset > 0; offset >>= 1) {
        mycount_f += __shfl_down_sync(0xFFFFFFFF, mycount_f, offset);
    }

    // Lane 0 computes the average and writes the result
    if (lane == 0) {
        const int total_count = static_cast<int>(mycount_f);
        float divisor;
        if (params.divisor_override > 0) {
            divisor = static_cast<float>(params.divisor_override);
        } else {
            divisor = static_cast<float>(total_count);
        }

        const int64_t out_idx = ((n * params.OH + oh) * params.OW + ow) * params.C + c;
        output[out_idx] = static_cast<T>((divisor > 0.0f) ? (mysum / divisor) : 0.0f);
    }
}

void avgpool_v4(const float* input, float* output, const AvgPoolParams& params, cudaStream_t stream) {
    NVTX_RANGE_PUSH_C("avgpool_v4_f32", NVTX_COLOR_AVGPOOL);
    const int64_t total = params.OH * params.OW * params.C;
    const int threads = 256;
    const int warps_per_block = threads / 32;  // 8
    const int blocks_x = static_cast<int>((total + warps_per_block - 1) / warps_per_block);
    dim3 grid(blocks_x, 1, static_cast<int>(params.N));
    avgpool_v4_kernel<float><<<grid, threads, 0, stream>>>(input, output, params);
    CUDA_CHECK(cudaGetLastError());
    NVTX_RANGE_POP();
}

void avgpool_v4(const half* input, half* output, const AvgPoolParams& params, cudaStream_t stream) {
    NVTX_RANGE_PUSH_C("avgpool_v4_f16", NVTX_COLOR_AVGPOOL);
    const int64_t total = params.OH * params.OW * params.C;
    const int threads = 256;
    const int warps_per_block = threads / 32;  // 8
    const int blocks_x = static_cast<int>((total + warps_per_block - 1) / warps_per_block);
    dim3 grid(blocks_x, 1, static_cast<int>(params.N));
    avgpool_v4_kernel<half><<<grid, threads, 0, stream>>>(input, output, params);
    CUDA_CHECK(cudaGetLastError());
    NVTX_RANGE_POP();
}

// ============================================================================
// AvgPool2d v5: double buffer / pipeline kernel
// Each block handles 2 consecutive channels (c and c+1), using double-buffered
// shared memory. The block loads channel c's tile into buf[0], computes the avg,
// then loads channel c+1's tile into buf[1] while the warp scheduler can
// interleave memory and compute from different warps (implicit overlap).
// Double-buffering reduces the grid dimension by half (ceil(C/2) channel-pairs
// instead of C channels) and keeps both tiles resident in smem.
// If C is odd, the last channel pair only uses buf[0].
// Falls back to v1 if C < 2 or smem is too large.
// ============================================================================

template <int TILE_OH = 8, int TILE_OW = 8>
__global__ void avgpool_v5_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
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

    // Phase 1: Load channel c0 into buf[0]
    for (int i = tid; i < total_smem; i += nthreads) {
        const int sih = i / smem_w;
        const int siw = i % smem_w;
        const int ih = ih_start + sih;
        const int iw = iw_start + siw;
        if (ih >= 0 && ih < params.H && iw >= 0 && iw < params.W) {
            const int64_t in_idx = ((static_cast<int64_t>(n) * params.H + ih) * params.W + iw) * params.C + c0;
            buf[0][i] = input[in_idx];
        } else {
            buf[0][i] = 0.0f;  // Padded positions contribute 0 to sum
        }
    }
    __syncthreads();

    // Compute avg from buf[0] for channel c0
    float sum_c0 = 0.0f;
    int count_c0 = 0;
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

                if (ih_in && iw_in) {
                    sum_c0 += buf[0][sih * smem_w + siw];
                    count_c0++;
                } else if (params.count_include_pad) {
                    const bool ih_in_pad = (ih >= -static_cast<int64_t>(params.ph) &&
                                            ih < params.H + static_cast<int64_t>(params.ph));
                    const bool iw_in_pad = (iw >= -static_cast<int64_t>(params.pw) &&
                                            iw < params.W + static_cast<int64_t>(params.pw));
                    if (ih_in_pad && iw_in_pad) {
                        count_c0++;
                    }
                }
            }
        }

        float divisor;
        if (params.divisor_override > 0) {
            divisor = static_cast<float>(params.divisor_override);
        } else {
            divisor = static_cast<float>(count_c0);
        }
        const int64_t out_idx = ((static_cast<int64_t>(n) * params.OH + oh) * params.OW + ow) * params.C + c0;
        output[out_idx] = (divisor > 0.0f) ? (sum_c0 / divisor) : 0.0f;
    }

    if (has_c1) {
        // Phase 2: Load channel c1 into buf[1]
        for (int i = tid; i < total_smem; i += nthreads) {
            const int sih = i / smem_w;
            const int siw = i % smem_w;
            const int ih = ih_start + sih;
            const int iw = iw_start + siw;
            if (ih >= 0 && ih < params.H && iw >= 0 && iw < params.W) {
                const int64_t in_idx = ((static_cast<int64_t>(n) * params.H + ih) * params.W + iw) * params.C + c1;
                buf[1][i] = input[in_idx];
            } else {
                buf[1][i] = 0.0f;
            }
        }
        __syncthreads();

        // Phase 3: Compute avg from buf[1] for channel c1
        float sum_c1 = 0.0f;
        int count_c1 = 0;
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

                    if (ih_in && iw_in) {
                        sum_c1 += buf[1][sih * smem_w + siw];
                        count_c1++;
                    } else if (params.count_include_pad) {
                        const bool ih_in_pad = (ih >= -static_cast<int64_t>(params.ph) &&
                                                ih < params.H + static_cast<int64_t>(params.ph));
                        const bool iw_in_pad = (iw >= -static_cast<int64_t>(params.pw) &&
                                                iw < params.W + static_cast<int64_t>(params.pw));
                        if (ih_in_pad && iw_in_pad) {
                            count_c1++;
                        }
                    }
                }
            }

            float divisor;
            if (params.divisor_override > 0) {
                divisor = static_cast<float>(params.divisor_override);
            } else {
                divisor = static_cast<float>(count_c1);
            }
            const int64_t out_idx = ((static_cast<int64_t>(n) * params.OH + oh) * params.OW + ow) * params.C + c1;
            output[out_idx] = (divisor > 0.0f) ? (sum_c1 / divisor) : 0.0f;
        }
    }
}

// half specialization: synchronous cooperative load with double-buffered smem
template <int TILE_OH = 8, int TILE_OW = 8>
__global__ void avgpool_v5_kernel_half(
    const half* __restrict__ input,
    half* __restrict__ output,
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

    // Phase 1: Synchronously load channel c0 into buf[0]
    for (int i = tid; i < total_smem; i += nthreads) {
        const int sih = i / smem_w;
        const int siw = i % smem_w;
        const int ih = ih_start + sih;
        const int iw = iw_start + siw;
        if (ih >= 0 && ih < params.H && iw >= 0 && iw < params.W) {
            const int64_t in_idx = ((static_cast<int64_t>(n) * params.H + ih) * params.W + iw) * params.C + c0;
            buf[0][i] = static_cast<float>(input[in_idx]);
        } else {
            buf[0][i] = 0.0f;
        }
    }
    __syncthreads();

    // Compute avg from buf[0] for channel c0
    float sum_c0 = 0.0f;
    int count_c0 = 0;
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

                if (ih_in && iw_in) {
                    sum_c0 += buf[0][sih * smem_w + siw];
                    count_c0++;
                } else if (params.count_include_pad) {
                    const bool ih_in_pad = (ih >= -static_cast<int64_t>(params.ph) &&
                                            ih < params.H + static_cast<int64_t>(params.ph));
                    const bool iw_in_pad = (iw >= -static_cast<int64_t>(params.pw) &&
                                            iw < params.W + static_cast<int64_t>(params.pw));
                    if (ih_in_pad && iw_in_pad) {
                        count_c0++;
                    }
                }
            }
        }

        float divisor;
        if (params.divisor_override > 0) {
            divisor = static_cast<float>(params.divisor_override);
        } else {
            divisor = static_cast<float>(count_c0);
        }
        const int64_t out_idx = ((static_cast<int64_t>(n) * params.OH + oh) * params.OW + ow) * params.C + c0;
        output[out_idx] = static_cast<half>((divisor > 0.0f) ? (sum_c0 / divisor) : 0.0f);
    }

    if (has_c1) {
        // Phase 2: Load channel c1 into buf[1] (synchronous for half)
        for (int i = tid; i < total_smem; i += nthreads) {
            const int sih = i / smem_w;
            const int siw = i % smem_w;
            const int ih = ih_start + sih;
            const int iw = iw_start + siw;
            if (ih >= 0 && ih < params.H && iw >= 0 && iw < params.W) {
                const int64_t in_idx = ((static_cast<int64_t>(n) * params.H + ih) * params.W + iw) * params.C + c1;
                buf[1][i] = static_cast<float>(input[in_idx]);
            } else {
                buf[1][i] = 0.0f;
            }
        }
        __syncthreads();

        // Phase 3: Compute avg from buf[1] for channel c1
        float sum_c1 = 0.0f;
        int count_c1 = 0;
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

                    if (ih_in && iw_in) {
                        sum_c1 += buf[1][sih * smem_w + siw];
                        count_c1++;
                    } else if (params.count_include_pad) {
                        const bool ih_in_pad = (ih >= -static_cast<int64_t>(params.ph) &&
                                                ih < params.H + static_cast<int64_t>(params.ph));
                        const bool iw_in_pad = (iw >= -static_cast<int64_t>(params.pw) &&
                                                iw < params.W + static_cast<int64_t>(params.pw));
                        if (ih_in_pad && iw_in_pad) {
                            count_c1++;
                        }
                    }
                }
            }

            float divisor;
            if (params.divisor_override > 0) {
                divisor = static_cast<float>(params.divisor_override);
            } else {
                divisor = static_cast<float>(count_c1);
            }
            const int64_t out_idx = ((static_cast<int64_t>(n) * params.OH + oh) * params.OW + ow) * params.C + c1;
            output[out_idx] = static_cast<half>((divisor > 0.0f) ? (sum_c1 / divisor) : 0.0f);
        }
    }
}

void avgpool_v5(const float* input, float* output, const AvgPoolParams& params, cudaStream_t stream) {
    NVTX_RANGE_PUSH_C("avgpool_v5_f32", NVTX_COLOR_AVGPOOL);
    constexpr int TILE_OH = 8;
    constexpr int TILE_OW = 8;

    if (params.C < 2) {
        NVTX_RANGE_POP();
        avgpool_v1(input, output, params, stream);
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
        avgpool_v1(input, output, params, stream);
        return;
    }

    const int c_pairs = static_cast<int>((params.C + 1) / 2);
    dim3 block(TILE_OW, TILE_OH);
    dim3 grid(blocks_oh * blocks_ow, c_pairs, static_cast<int>(params.N));

    if (smem_bytes > 49152) {
        CUDA_CHECK(cudaFuncSetAttribute(
            avgpool_v5_kernel<TILE_OH, TILE_OW>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(smem_bytes)));
    }

    avgpool_v5_kernel<TILE_OH, TILE_OW><<<grid, block, smem_bytes, stream>>>(
        input, output, params, blocks_oh, blocks_ow, smem_h, smem_w);
    CUDA_CHECK(cudaGetLastError());
    NVTX_RANGE_POP();
}

void avgpool_v5(const half* input, half* output, const AvgPoolParams& params, cudaStream_t stream) {
    NVTX_RANGE_PUSH_C("avgpool_v5_f16", NVTX_COLOR_AVGPOOL);
    constexpr int TILE_OH = 8;
    constexpr int TILE_OW = 8;

    if (params.C < 2) {
        NVTX_RANGE_POP();
        avgpool_v1(input, output, params, stream);
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
        avgpool_v1(input, output, params, stream);
        return;
    }

    const int c_pairs = static_cast<int>((params.C + 1) / 2);
    dim3 block(TILE_OW, TILE_OH);
    dim3 grid(blocks_oh * blocks_ow, c_pairs, static_cast<int>(params.N));

    if (smem_bytes > 49152) {
        CUDA_CHECK(cudaFuncSetAttribute(
            avgpool_v5_kernel_half<TILE_OH, TILE_OW>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(smem_bytes)));
    }

    avgpool_v5_kernel_half<TILE_OH, TILE_OW><<<grid, block, smem_bytes, stream>>>(
        input, output, params, blocks_oh, blocks_ow, smem_h, smem_w);
    CUDA_CHECK(cudaGetLastError());
    NVTX_RANGE_POP();
}

// ============================================================================
// AvgPool2d v6: warp specialization kernel
// Split warps in a block into two roles: "load warps" that fetch data from
// global memory into shared memory, and "compute warps" that read from shared
// memory and compute the average. This decouples memory latency from compute.
//
// Block size: 256 threads = 8 warps
// NUM_LOAD_WARPS = 2 (warp 0-1), NUM_COMPUTE_WARPS = 6 (warp 2-7)
// Same tiling as v1: TILE_OH=8, TILE_OW=8 per (n, c) pair
// Falls back to v1 if smem is too large.
// ============================================================================

template <int TILE_OH = 8, int TILE_OW = 8, int NUM_LOAD_WARPS = 2, int NUM_COMPUTE_WARPS = 6>
__global__ void avgpool_v6_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const AvgPoolParams params,
    int blocks_oh, int blocks_ow,
    int smem_h, int smem_w)
{
    constexpr int NUM_LOAD_THREADS = NUM_LOAD_WARPS * 32;                  // 64
    constexpr int NUM_COMPUTE_THREADS = NUM_COMPUTE_WARPS * 32;            // 192
    constexpr int TILE_SIZE = TILE_OH * TILE_OW;                           // 64

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

    // Phase 1: Load warps cooperatively load input tile into shared memory
    if (is_load_warp) {
        for (int i = tid; i < total_smem; i += NUM_LOAD_THREADS) {
            const int sih = i / smem_w;
            const int siw = i % smem_w;
            const int ih = ih_start + sih;
            const int iw = iw_start + siw;
            if (ih >= 0 && ih < params.H && iw >= 0 && iw < params.W) {
                const int64_t in_idx = ((static_cast<int64_t>(n) * params.H + ih) * params.W + iw) * params.C + c;
                sdata[i] = input[in_idx];
            } else {
                sdata[i] = 0.0f;  // Padded positions contribute 0 to sum
            }
        }
    }
    __syncthreads();

    // Phase 2: Compute warps compute avg from shared memory
    if (!is_load_warp) {
        const int compute_tid = tid - NUM_LOAD_THREADS;  // 0..191

        for (int i = compute_tid; i < TILE_SIZE; i += NUM_COMPUTE_THREADS) {
            const int th = i / TILE_OW;
            const int tw = i % TILE_OW;
            const int oh = tile_oh * TILE_OH + th;
            const int ow = tile_ow * TILE_OW + tw;

            if (oh >= params.OH || ow >= params.OW) continue;

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
                        if (ih_in_pad && iw_in_pad) {
                            count++;
                        }
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
            output[out_idx] = (divisor > 0.0f) ? (sum / divisor) : 0.0f;
        }
    }
}

// half specialization
template <int TILE_OH = 8, int TILE_OW = 8, int NUM_LOAD_WARPS = 2, int NUM_COMPUTE_WARPS = 6>
__global__ void avgpool_v6_kernel_half(
    const half* __restrict__ input,
    half* __restrict__ output,
    const AvgPoolParams params,
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
                sdata[i] = 0.0f;
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
                        if (ih_in_pad && iw_in_pad) {
                            count++;
                        }
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
            output[out_idx] = static_cast<half>((divisor > 0.0f) ? (sum / divisor) : 0.0f);
        }
    }
}

void avgpool_v6(const float* input, float* output, const AvgPoolParams& params, cudaStream_t stream) {
    NVTX_RANGE_PUSH_C("avgpool_v6_f32", NVTX_COLOR_AVGPOOL);
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
        avgpool_v1(input, output, params, stream);
        return;
    }

    dim3 block(BLOCK_SIZE);
    dim3 grid(blocks_oh * blocks_ow, static_cast<int>(params.C), static_cast<int>(params.N));

    if (smem_bytes > 49152) {
        CUDA_CHECK(cudaFuncSetAttribute(
            avgpool_v6_kernel<TILE_OH, TILE_OW>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(smem_bytes)));
    }

    avgpool_v6_kernel<TILE_OH, TILE_OW><<<grid, block, smem_bytes, stream>>>(
        input, output, params, blocks_oh, blocks_ow, smem_h, smem_w);
    CUDA_CHECK(cudaGetLastError());
    NVTX_RANGE_POP();
}

void avgpool_v6(const half* input, half* output, const AvgPoolParams& params, cudaStream_t stream) {
    NVTX_RANGE_PUSH_C("avgpool_v6_f16", NVTX_COLOR_AVGPOOL);
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
        avgpool_v1(input, output, params, stream);
        return;
    }

    dim3 block(BLOCK_SIZE);
    dim3 grid(blocks_oh * blocks_ow, static_cast<int>(params.C), static_cast<int>(params.N));

    if (smem_bytes > 49152) {
        CUDA_CHECK(cudaFuncSetAttribute(
            avgpool_v6_kernel_half<TILE_OH, TILE_OW>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(smem_bytes)));
    }

    avgpool_v6_kernel_half<TILE_OH, TILE_OW><<<grid, block, smem_bytes, stream>>>(
        input, output, params, blocks_oh, blocks_ow, smem_h, smem_w);
    CUDA_CHECK(cudaGetLastError());
    NVTX_RANGE_POP();
}

// ============================================================================
// AvgPool2d v7: alternative grid/block mappings
// mapping=0 (A): 1D flat — same as v0
// mapping=1 (B): 2D spatial tiling — blockDim(8,8,4)
// mapping=2 (C): channel-major — blockDim(256,1,1), grid(OW,OH,N*ceil(C/256))
// mapping=3 (D): hybrid warp-spatial + vectorized — blockDim(32,8,1)
// Falls back to mapping A for alignment issues (e.g., C%4!=0 for D)
// or when grid dimensions exceed CUDA limits.
// ============================================================================

// --- Mapping B: 2D Spatial ---
// Each block covers an 8x8 spatial tile with 4 channels per thread
// blockDim = (8, 8, 4) = 256 threads
// Grid: (ceil(OW/8), ceil(OH/8), N * ceil(C/4))
// Thread (tx, ty, tz): oh = blockIdx.y*8+ty, ow = blockIdx.x*8+tx, c = c_group*4+tz

template <typename T>
__global__ void avgpool_v7_mappingB_kernel(
    const T* __restrict__ input,
    T* __restrict__ output,
    const AvgPoolParams params,
    int C_groups)
{
    const int n = blockIdx.z / C_groups;
    const int c_group = blockIdx.z - n * C_groups;
    const int c = c_group * 4 + threadIdx.z;
    const int oh = blockIdx.y * 8 + threadIdx.y;
    const int ow = blockIdx.x * 8 + threadIdx.x;

    if (n >= params.N || oh >= params.OH || ow >= params.OW || c >= params.C) return;

    float sum = 0.0f;
    int count = 0;

    for (int kh_i = 0; kh_i < params.kh; ++kh_i) {
        const int64_t ih = oh * params.sh - params.ph + static_cast<int64_t>(kh_i) * params.dh;
        const bool ih_in = (ih >= 0 && ih < params.H);
        for (int kw_i = 0; kw_i < params.kw; ++kw_i) {
            const int64_t iw = ow * params.sw - params.pw + static_cast<int64_t>(kw_i) * params.dw;
            const bool iw_in = (iw >= 0 && iw < params.W);

            if (ih_in && iw_in) {
                const int64_t in_idx = ((static_cast<int64_t>(n) * params.H + ih) * params.W + iw) * params.C + c;
                sum += static_cast<float>(input[in_idx]);
                count++;
            } else if (params.count_include_pad) {
                const bool ih_in_pad = (ih >= -static_cast<int64_t>(params.ph) &&
                                        ih < params.H + static_cast<int64_t>(params.ph));
                const bool iw_in_pad = (iw >= -static_cast<int64_t>(params.pw) &&
                                        iw < params.W + static_cast<int64_t>(params.pw));
                if (ih_in_pad && iw_in_pad) {
                    count++;
                }
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
    output[out_idx] = static_cast<T>((divisor > 0.0f) ? (sum / divisor) : 0.0f);
}

// --- Mapping C: Channel-Major ---
// Each block covers 256 channels for one (oh, ow) position
// blockDim = (256, 1, 1)
// Grid: (OW, OH, N * ceil(C/256))
// Thread: c = c_group*256 + threadIdx.x, ow = blockIdx.x, oh = blockIdx.y

template <typename T>
__global__ void avgpool_v7_mappingC_kernel(
    const T* __restrict__ input,
    T* __restrict__ output,
    const AvgPoolParams params,
    int C_groups)
{
    const int n = blockIdx.z / C_groups;
    const int c_group = blockIdx.z - n * C_groups;
    const int c = c_group * 256 + threadIdx.x;
    const int oh = blockIdx.y;
    const int ow = blockIdx.x;

    if (n >= params.N || oh >= params.OH || ow >= params.OW || c >= params.C) return;

    float sum = 0.0f;
    int count = 0;

    for (int kh_i = 0; kh_i < params.kh; ++kh_i) {
        const int64_t ih = oh * params.sh - params.ph + static_cast<int64_t>(kh_i) * params.dh;
        const bool ih_in = (ih >= 0 && ih < params.H);
        for (int kw_i = 0; kw_i < params.kw; ++kw_i) {
            const int64_t iw = ow * params.sw - params.pw + static_cast<int64_t>(kw_i) * params.dw;
            const bool iw_in = (iw >= 0 && iw < params.W);

            if (ih_in && iw_in) {
                const int64_t in_idx = ((static_cast<int64_t>(n) * params.H + ih) * params.W + iw) * params.C + c;
                sum += static_cast<float>(input[in_idx]);
                count++;
            } else if (params.count_include_pad) {
                const bool ih_in_pad = (ih >= -static_cast<int64_t>(params.ph) &&
                                        ih < params.H + static_cast<int64_t>(params.ph));
                const bool iw_in_pad = (iw >= -static_cast<int64_t>(params.pw) &&
                                        iw < params.W + static_cast<int64_t>(params.pw));
                if (ih_in_pad && iw_in_pad) {
                    count++;
                }
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
    output[out_idx] = static_cast<T>((divisor > 0.0f) ? (sum / divisor) : 0.0f);
}

// --- Mapping D: Hybrid warp-spatial + channel-vectorized ---
// Each warp handles a 4x4 spatial tile with 4 channels via vectorized loads
// blockDim = (32, 8, 1) — 8 warps per block, each warp covers 4x4 spatial + 4 channels
// Grid: (ceil(OW/4), ceil(OH/4), N * ceil(C/32))
// Lanes 0-15 of each warp handle 4x4 spatial positions, loading 4 channels via float4/half2x2

template <typename T>
__global__ void avgpool_v7_mappingD_kernel(
    const T* __restrict__ input,
    T* __restrict__ output,
    const AvgPoolParams params,
    int C_groups)
{
    const int n = blockIdx.z / C_groups;
    const int c_group = blockIdx.z - n * C_groups;
    const int c_block_base = c_group * 32;
    const int warp_id = threadIdx.y;  // 0..7
    const int lane = threadIdx.x;     // 0..31

    const int oh_tile_base = blockIdx.y * 4;
    const int ow_tile_base = blockIdx.x * 4;

    // Each warp handles 4 channels
    const int c_warp_base = c_block_base + warp_id * 4;
    if (n >= params.N || c_warp_base + 3 >= params.C) return;

    // Only first 16 lanes are active (4x4 = 16 spatial positions)
    if (lane >= 16) return;

    const int sh = lane / 4;
    const int sw = lane % 4;
    const int oh = oh_tile_base + sh;
    const int ow = ow_tile_base + sw;
    if (oh >= params.OH || ow >= params.OW) return;

    float sum[4];
    int count = 0;
    #pragma unroll
    for (int v = 0; v < 4; ++v)
        sum[v] = 0.0f;

    for (int kh_i = 0; kh_i < params.kh; ++kh_i) {
        const int64_t ih = oh * params.sh - params.ph + static_cast<int64_t>(kh_i) * params.dh;
        const bool ih_in = (ih >= 0 && ih < params.H);
        for (int kw_i = 0; kw_i < params.kw; ++kw_i) {
            const int64_t iw = ow * params.sw - params.pw + static_cast<int64_t>(kw_i) * params.dw;
            const bool iw_in = (iw >= 0 && iw < params.W);

            if (ih_in && iw_in) {
                const int64_t in_idx = ((static_cast<int64_t>(n) * params.H + ih) * params.W + iw) * params.C + c_warp_base;

                // Vectorized load of 4 channels
                if constexpr (std::is_same_v<T, float>) {
                    float4 vec = *reinterpret_cast<const float4*>(&input[in_idx]);
                    sum[0] += vec.x;
                    sum[1] += vec.y;
                    sum[2] += vec.z;
                    sum[3] += vec.w;
                } else {
                    half2 vec0 = *reinterpret_cast<const half2*>(&input[in_idx]);
                    half2 vec1 = *reinterpret_cast<const half2*>(&input[in_idx + 2]);
                    sum[0] += __low2float(vec0);
                    sum[1] += __high2float(vec0);
                    sum[2] += __low2float(vec1);
                    sum[3] += __high2float(vec1);
                }
                count++;
            } else if (params.count_include_pad) {
                const bool ih_in_pad = (ih >= -static_cast<int64_t>(params.ph) &&
                                        ih < params.H + static_cast<int64_t>(params.ph));
                const bool iw_in_pad = (iw >= -static_cast<int64_t>(params.pw) &&
                                        iw < params.W + static_cast<int64_t>(params.pw));
                if (ih_in_pad && iw_in_pad) {
                    count++;
                }
            }
        }
    }

    float divisor;
    if (params.divisor_override > 0) {
        divisor = static_cast<float>(params.divisor_override);
    } else {
        divisor = static_cast<float>(count);
    }

    // Write 4 output values
    const int64_t out_idx = ((static_cast<int64_t>(n) * params.OH + oh) * params.OW + ow) * params.C + c_warp_base;

    if constexpr (std::is_same_v<T, float>) {
        float4 out_vec;
        out_vec.x = (divisor > 0.0f) ? (sum[0] / divisor) : 0.0f;
        out_vec.y = (divisor > 0.0f) ? (sum[1] / divisor) : 0.0f;
        out_vec.z = (divisor > 0.0f) ? (sum[2] / divisor) : 0.0f;
        out_vec.w = (divisor > 0.0f) ? (sum[3] / divisor) : 0.0f;
        *reinterpret_cast<float4*>(&output[out_idx]) = out_vec;
    } else {
        float r0 = (divisor > 0.0f) ? (sum[0] / divisor) : 0.0f;
        float r1 = (divisor > 0.0f) ? (sum[1] / divisor) : 0.0f;
        float r2 = (divisor > 0.0f) ? (sum[2] / divisor) : 0.0f;
        float r3 = (divisor > 0.0f) ? (sum[3] / divisor) : 0.0f;
        half2 out0 = __floats2half2_rn(r0, r1);
        half2 out1 = __floats2half2_rn(r2, r3);
        *reinterpret_cast<half2*>(&output[out_idx]) = out0;
        *reinterpret_cast<half2*>(&output[out_idx + 2]) = out1;
    }
}

// --- v7 launcher: dispatches to the appropriate mapping kernel ---

void avgpool_v7(const float* input, float* output, const AvgPoolParams& params, int mapping, cudaStream_t stream) {
    NVTX_RANGE_PUSH_C("avgpool_v7_f32", NVTX_COLOR_AVGPOOL);
    switch (mapping) {
        case 0: {
            // Mapping A: 1D flat — same as v0
            NVTX_RANGE_POP();
            avgpool_v0(input, output, params, stream);
            return;
        }
        case 1: {
            // Mapping B: 2D spatial tiling
            const int C_groups = static_cast<int>((params.C + 3) / 4);
            const int64_t grid_z_64 = params.N * C_groups;
            if (grid_z_64 > 65535) {
                NVTX_RANGE_POP();
                avgpool_v0(input, output, params, stream);
                return;
            }
            dim3 block(8, 8, 4);
            dim3 grid(
                static_cast<int>((params.OW + 7) / 8),
                static_cast<int>((params.OH + 7) / 8),
                static_cast<int>(grid_z_64));
            avgpool_v7_mappingB_kernel<float><<<grid, block, 0, stream>>>(input, output, params, C_groups);
            CUDA_CHECK(cudaGetLastError());
            break;
        }
        case 2: {
            // Mapping C: channel-major
            const int C_groups = static_cast<int>((params.C + 255) / 256);
            const int64_t grid_z_64 = params.N * C_groups;
            if (params.OW > 65535 || params.OH > 65535 || grid_z_64 > 65535) {
                NVTX_RANGE_POP();
                avgpool_v0(input, output, params, stream);
                return;
            }
            dim3 block(256, 1, 1);
            dim3 grid(
                static_cast<int>(params.OW),
                static_cast<int>(params.OH),
                static_cast<int>(grid_z_64));
            avgpool_v7_mappingC_kernel<float><<<grid, block, 0, stream>>>(input, output, params, C_groups);
            CUDA_CHECK(cudaGetLastError());
            break;
        }
        case 3: {
            // Mapping D: hybrid warp-spatial + vectorized
            if (params.C % 4 != 0) {
                NVTX_RANGE_POP();
                avgpool_v0(input, output, params, stream);
                return;
            }
            const int C_groups = static_cast<int>((params.C + 31) / 32);
            const int64_t grid_z_64 = params.N * C_groups;
            if (grid_z_64 > 65535) {
                NVTX_RANGE_POP();
                avgpool_v0(input, output, params, stream);
                return;
            }
            dim3 block(32, 8, 1);
            dim3 grid(
                static_cast<int>((params.OW + 3) / 4),
                static_cast<int>((params.OH + 3) / 4),
                static_cast<int>(grid_z_64));
            avgpool_v7_mappingD_kernel<float><<<grid, block, 0, stream>>>(input, output, params, C_groups);
            CUDA_CHECK(cudaGetLastError());
            break;
        }
        default:
            NVTX_RANGE_POP();
            throw std::invalid_argument("unsupported mapping: " + std::to_string(mapping));
    }
    NVTX_RANGE_POP();
}

void avgpool_v7(const half* input, half* output, const AvgPoolParams& params, int mapping, cudaStream_t stream) {
    NVTX_RANGE_PUSH_C("avgpool_v7_f16", NVTX_COLOR_AVGPOOL);
    switch (mapping) {
        case 0: {
            NVTX_RANGE_POP();
            avgpool_v0(input, output, params, stream);
            return;
        }
        case 1: {
            const int C_groups = static_cast<int>((params.C + 3) / 4);
            const int64_t grid_z_64 = params.N * C_groups;
            if (grid_z_64 > 65535) {
                NVTX_RANGE_POP();
                avgpool_v0(input, output, params, stream);
                return;
            }
            dim3 block(8, 8, 4);
            dim3 grid(
                static_cast<int>((params.OW + 7) / 8),
                static_cast<int>((params.OH + 7) / 8),
                static_cast<int>(grid_z_64));
            avgpool_v7_mappingB_kernel<half><<<grid, block, 0, stream>>>(input, output, params, C_groups);
            CUDA_CHECK(cudaGetLastError());
            break;
        }
        case 2: {
            const int C_groups = static_cast<int>((params.C + 255) / 256);
            const int64_t grid_z_64 = params.N * C_groups;
            if (params.OW > 65535 || params.OH > 65535 || grid_z_64 > 65535) {
                NVTX_RANGE_POP();
                avgpool_v0(input, output, params, stream);
                return;
            }
            dim3 block(256, 1, 1);
            dim3 grid(
                static_cast<int>(params.OW),
                static_cast<int>(params.OH),
                static_cast<int>(grid_z_64));
            avgpool_v7_mappingC_kernel<half><<<grid, block, 0, stream>>>(input, output, params, C_groups);
            CUDA_CHECK(cudaGetLastError());
            break;
        }
        case 3: {
            if (params.C % 4 != 0) {
                NVTX_RANGE_POP();
                avgpool_v0(input, output, params, stream);
                return;
            }
            const int C_groups = static_cast<int>((params.C + 31) / 32);
            const int64_t grid_z_64 = params.N * C_groups;
            if (grid_z_64 > 65535) {
                NVTX_RANGE_POP();
                avgpool_v0(input, output, params, stream);
                return;
            }
            dim3 block(32, 8, 1);
            dim3 grid(
                static_cast<int>((params.OW + 3) / 4),
                static_cast<int>((params.OH + 3) / 4),
                static_cast<int>(grid_z_64));
            avgpool_v7_mappingD_kernel<half><<<grid, block, 0, stream>>>(input, output, params, C_groups);
            CUDA_CHECK(cudaGetLastError());
            break;
        }
        default:
            NVTX_RANGE_POP();
            throw std::invalid_argument("unsupported mapping: " + std::to_string(mapping));
    }
    NVTX_RANGE_POP();
}

// ============================================================================
// AvgPool2d v8: auto-tuned tiling kernel
// Same shared-memory tiling approach as v1, but auto-selects optimal
// TILE_OH x TILE_OW tile dimensions for each (shape, kernel) configuration.
// Falls back to v2 for aligned-C cases (vectorized loads outperform tiling).
// ============================================================================

struct AvgTileConfig { int tile_oh; int tile_ow; };
static const AvgTileConfig AVG_V8_TILE_CANDIDATES[] = {
    {8, 8}, {16, 16}, {32, 8}, {8, 32}, {16, 8},
    {8, 16}, {32, 4}, {4, 32}, {64, 4},
};
static const int AVG_V8_NUM_CANDIDATES = 9;

static uint64_t avg_v8_hash_key(int64_t H, int64_t W, int64_t C, int kh, int kw, int sh, int sw, int ph, int pw) {
    uint64_t hash = 0xcbf29ce484222325ULL;
    const uint64_t prime = 0x00000100000001b3ULL;
    int64_t vals[] = {H, W, C, (int64_t)kh, (int64_t)kw, (int64_t)sh, (int64_t)sw, (int64_t)ph, (int64_t)pw};
    for (int i = 0; i < 9; i++) {
        hash ^= (uint64_t)vals[i];
        hash *= prime;
    }
    return hash;
}

static std::mutex avg_v8_cache_mutex;
static std::unordered_map<uint64_t, AvgTileConfig> avg_v8_cache;

template <int TILE_OH, int TILE_OW, typename T>
__global__ void avgpool_v8_kernel(
    const T* __restrict__ input,
    T* __restrict__ output,
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
    const int nthreads = TILE_OH * TILE_OW;
    const int tid = th * TILE_OW + tw;
    for (int i = tid; i < total_smem; i += nthreads) {
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
                if (ih_in_pad && iw_in_pad) {
                    count++;
                }
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
    output[out_idx] = static_cast<T>((divisor > 0.0f) ? (sum / divisor) : 0.0f);
}

template <typename T>
static float avg_v8_bench_tile(const T* d_input, T* d_output, const AvgPoolParams& params,
                               AvgTileConfig cfg, cudaStream_t stream) {
    const int blocks_oh = static_cast<int>((params.OH + cfg.tile_oh - 1) / cfg.tile_oh);
    const int blocks_ow = static_cast<int>((params.OW + cfg.tile_ow - 1) / cfg.tile_ow);
    int smem_h = (cfg.tile_oh - 1) * params.sh + (params.kh - 1) * params.dh + 1;
    int smem_w = (cfg.tile_ow - 1) * params.sw + (params.kw - 1) * params.dw + 1;
    smem_h = min(smem_h, static_cast<int>(params.H + 2 * params.ph));
    smem_w = min(smem_w, static_cast<int>(params.W + 2 * params.pw));

    dim3 block(cfg.tile_ow, cfg.tile_oh);
    dim3 grid(blocks_oh * blocks_ow, static_cast<int>(params.C), static_cast<int>(params.N));
    size_t smem_bytes = static_cast<size_t>(smem_h) * smem_w * sizeof(float);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    for (int i = 0; i < 3; i++) {
        if (smem_bytes > 49152) {
            CUDA_CHECK(cudaFuncSetAttribute(avgpool_v8_kernel<8, 8, T>, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem_bytes)));
        }
        switch ((cfg.tile_oh << 8) | cfg.tile_ow) {
            case (8<<8)|8:   avgpool_v8_kernel<8, 8, T><<<grid, block, smem_bytes, stream>>>(d_input, d_output, params, blocks_oh, blocks_ow, smem_h, smem_w); break;
            case (16<<8)|16: avgpool_v8_kernel<16, 16, T><<<grid, block, smem_bytes, stream>>>(d_input, d_output, params, blocks_oh, blocks_ow, smem_h, smem_w); break;
            case (32<<8)|8:  avgpool_v8_kernel<32, 8, T><<<grid, block, smem_bytes, stream>>>(d_input, d_output, params, blocks_oh, blocks_ow, smem_h, smem_w); break;
            case (8<<8)|32:  avgpool_v8_kernel<8, 32, T><<<grid, block, smem_bytes, stream>>>(d_input, d_output, params, blocks_oh, blocks_ow, smem_h, smem_w); break;
            case (16<<8)|8:  avgpool_v8_kernel<16, 8, T><<<grid, block, smem_bytes, stream>>>(d_input, d_output, params, blocks_oh, blocks_ow, smem_h, smem_w); break;
            case (8<<8)|16:  avgpool_v8_kernel<8, 16, T><<<grid, block, smem_bytes, stream>>>(d_input, d_output, params, blocks_oh, blocks_ow, smem_h, smem_w); break;
            case (32<<8)|4:  avgpool_v8_kernel<32, 4, T><<<grid, block, smem_bytes, stream>>>(d_input, d_output, params, blocks_oh, blocks_ow, smem_h, smem_w); break;
            case (4<<8)|32:  avgpool_v8_kernel<4, 32, T><<<grid, block, smem_bytes, stream>>>(d_input, d_output, params, blocks_oh, blocks_ow, smem_h, smem_w); break;
            case (64<<8)|4:  avgpool_v8_kernel<64, 4, T><<<grid, block, smem_bytes, stream>>>(d_input, d_output, params, blocks_oh, blocks_ow, smem_h, smem_w); break;
        }
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    CUDA_CHECK(cudaEventRecord(start, stream));
    if (smem_bytes > 49152) {
        CUDA_CHECK(cudaFuncSetAttribute(avgpool_v8_kernel<8, 8, T>, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem_bytes)));
    }
    switch ((cfg.tile_oh << 8) | cfg.tile_ow) {
        case (8<<8)|8:   avgpool_v8_kernel<8, 8, T><<<grid, block, smem_bytes, stream>>>(d_input, d_output, params, blocks_oh, blocks_ow, smem_h, smem_w); break;
        case (16<<8)|16: avgpool_v8_kernel<16, 16, T><<<grid, block, smem_bytes, stream>>>(d_input, d_output, params, blocks_oh, blocks_ow, smem_h, smem_w); break;
        case (32<<8)|8:  avgpool_v8_kernel<32, 8, T><<<grid, block, smem_bytes, stream>>>(d_input, d_output, params, blocks_oh, blocks_ow, smem_h, smem_w); break;
        case (8<<8)|32:  avgpool_v8_kernel<8, 32, T><<<grid, block, smem_bytes, stream>>>(d_input, d_output, params, blocks_oh, blocks_ow, smem_h, smem_w); break;
        case (16<<8)|8:  avgpool_v8_kernel<16, 8, T><<<grid, block, smem_bytes, stream>>>(d_input, d_output, params, blocks_oh, blocks_ow, smem_h, smem_w); break;
        case (8<<8)|16:  avgpool_v8_kernel<8, 16, T><<<grid, block, smem_bytes, stream>>>(d_input, d_output, params, blocks_oh, blocks_ow, smem_h, smem_w); break;
        case (32<<8)|4:  avgpool_v8_kernel<32, 4, T><<<grid, block, smem_bytes, stream>>>(d_input, d_output, params, blocks_oh, blocks_ow, smem_h, smem_w); break;
        case (4<<8)|32:  avgpool_v8_kernel<4, 32, T><<<grid, block, smem_bytes, stream>>>(d_input, d_output, params, blocks_oh, blocks_ow, smem_h, smem_w); break;
        case (64<<8)|4:  avgpool_v8_kernel<64, 4, T><<<grid, block, smem_bytes, stream>>>(d_input, d_output, params, blocks_oh, blocks_ow, smem_h, smem_w); break;
    }
    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float elapsed = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return elapsed;
}

template <typename T>
static void avg_v8_launch_config(const T* d_input, T* d_output, const AvgPoolParams& params,
                                 AvgTileConfig cfg, cudaStream_t stream) {
    const int blocks_oh = static_cast<int>((params.OH + cfg.tile_oh - 1) / cfg.tile_oh);
    const int blocks_ow = static_cast<int>((params.OW + cfg.tile_ow - 1) / cfg.tile_ow);
    int smem_h = (cfg.tile_oh - 1) * params.sh + (params.kh - 1) * params.dh + 1;
    int smem_w = (cfg.tile_ow - 1) * params.sw + (params.kw - 1) * params.dw + 1;
    smem_h = min(smem_h, static_cast<int>(params.H + 2 * params.ph));
    smem_w = min(smem_w, static_cast<int>(params.W + 2 * params.pw));

    dim3 block(cfg.tile_ow, cfg.tile_oh);
    dim3 grid(blocks_oh * blocks_ow, static_cast<int>(params.C), static_cast<int>(params.N));
    size_t smem_bytes = static_cast<size_t>(smem_h) * smem_w * sizeof(float);

    if (smem_bytes > 49152) {
        CUDA_CHECK(cudaFuncSetAttribute(avgpool_v8_kernel<8, 8, T>, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem_bytes)));
    }
    switch ((cfg.tile_oh << 8) | cfg.tile_ow) {
        case (8<<8)|8:   avgpool_v8_kernel<8, 8, T><<<grid, block, smem_bytes, stream>>>(d_input, d_output, params, blocks_oh, blocks_ow, smem_h, smem_w); break;
        case (16<<8)|16: avgpool_v8_kernel<16, 16, T><<<grid, block, smem_bytes, stream>>>(d_input, d_output, params, blocks_oh, blocks_ow, smem_h, smem_w); break;
        case (32<<8)|8:  avgpool_v8_kernel<32, 8, T><<<grid, block, smem_bytes, stream>>>(d_input, d_output, params, blocks_oh, blocks_ow, smem_h, smem_w); break;
        case (8<<8)|32:  avgpool_v8_kernel<8, 32, T><<<grid, block, smem_bytes, stream>>>(d_input, d_output, params, blocks_oh, blocks_ow, smem_h, smem_w); break;
        case (16<<8)|8:  avgpool_v8_kernel<16, 8, T><<<grid, block, smem_bytes, stream>>>(d_input, d_output, params, blocks_oh, blocks_ow, smem_h, smem_w); break;
        case (8<<8)|16:  avgpool_v8_kernel<8, 16, T><<<grid, block, smem_bytes, stream>>>(d_input, d_output, params, blocks_oh, blocks_ow, smem_h, smem_w); break;
        case (32<<8)|4:  avgpool_v8_kernel<32, 4, T><<<grid, block, smem_bytes, stream>>>(d_input, d_output, params, blocks_oh, blocks_ow, smem_h, smem_w); break;
        case (4<<8)|32:  avgpool_v8_kernel<4, 32, T><<<grid, block, smem_bytes, stream>>>(d_input, d_output, params, blocks_oh, blocks_ow, smem_h, smem_w); break;
        case (64<<8)|4:  avgpool_v8_kernel<64, 4, T><<<grid, block, smem_bytes, stream>>>(d_input, d_output, params, blocks_oh, blocks_ow, smem_h, smem_w); break;
        default: avgpool_v8_kernel<8, 8, T><<<grid, block, smem_bytes, stream>>>(d_input, d_output, params, blocks_oh, blocks_ow, smem_h, smem_w); break;
    }
}

void avgpool_v8(const float* input, float* output, const AvgPoolParams& params, cudaStream_t stream) {
    NVTX_RANGE_PUSH_C("avgpool_v8_f32", NVTX_COLOR_AVGPOOL);

    if (params.C % 4 == 0) {
        NVTX_RANGE_POP();
        avgpool_v2(input, output, params, stream);
        return;
    }

    uint64_t key = avg_v8_hash_key(params.H, params.W, params.C, params.kh, params.kw, params.sh, params.sw, params.ph, params.pw);

    AvgTileConfig cfg;
    {
        std::lock_guard<std::mutex> lock(avg_v8_cache_mutex);
        auto it = avg_v8_cache.find(key);
        if (it != avg_v8_cache.end()) {
            cfg = it->second;
        }
    }

    if (cfg.tile_oh == 0) {
        int best_idx = 0;
        float best_time = 1e9f;
        for (int i = 0; i < AVG_V8_NUM_CANDIDATES; i++) {
            int smem_h = (AVG_V8_TILE_CANDIDATES[i].tile_oh - 1) * params.sh + (params.kh - 1) * params.dh + 1;
            int smem_w = (AVG_V8_TILE_CANDIDATES[i].tile_ow - 1) * params.sw + (params.kw - 1) * params.dw + 1;
            smem_h = min(smem_h, static_cast<int>(params.H + 2 * params.ph));
            smem_w = min(smem_w, static_cast<int>(params.W + 2 * params.pw));
            size_t smem_bytes = static_cast<size_t>(smem_h) * smem_w * sizeof(float);
            int smem_limit = 0;
            CUDA_CHECK(cudaDeviceGetAttribute(&smem_limit, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0));
            if (smem_bytes > static_cast<size_t>(smem_limit)) continue;

            float t = avg_v8_bench_tile(input, output, params, AVG_V8_TILE_CANDIDATES[i], stream);
            if (t < best_time) {
                best_time = t;
                best_idx = i;
            }
        }
        cfg = AVG_V8_TILE_CANDIDATES[best_idx];
        {
            std::lock_guard<std::mutex> lock(avg_v8_cache_mutex);
            avg_v8_cache[key] = cfg;
        }
    }

    avg_v8_launch_config(input, output, params, cfg, stream);
    CUDA_CHECK(cudaGetLastError());
    NVTX_RANGE_POP();
}

void avgpool_v8(const half* input, half* output, const AvgPoolParams& params, cudaStream_t stream) {
    NVTX_RANGE_PUSH_C("avgpool_v8_f16", NVTX_COLOR_AVGPOOL);

    if (params.C % 2 == 0) {
        NVTX_RANGE_POP();
        avgpool_v2(input, output, params, stream);
        return;
    }

    uint64_t key = avg_v8_hash_key(params.H, params.W, params.C, params.kh, params.kw, params.sh, params.sw, params.ph, params.pw);

    AvgTileConfig cfg;
    {
        std::lock_guard<std::mutex> lock(avg_v8_cache_mutex);
        auto it = avg_v8_cache.find(key);
        if (it != avg_v8_cache.end()) {
            cfg = it->second;
        }
    }

    if (cfg.tile_oh == 0) {
        int best_idx = 0;
        float best_time = 1e9f;
        for (int i = 0; i < AVG_V8_NUM_CANDIDATES; i++) {
            int smem_h = (AVG_V8_TILE_CANDIDATES[i].tile_oh - 1) * params.sh + (params.kh - 1) * params.dh + 1;
            int smem_w = (AVG_V8_TILE_CANDIDATES[i].tile_ow - 1) * params.sw + (params.kw - 1) * params.dw + 1;
            smem_h = min(smem_h, static_cast<int>(params.H + 2 * params.ph));
            smem_w = min(smem_w, static_cast<int>(params.W + 2 * params.pw));
            size_t smem_bytes = static_cast<size_t>(smem_h) * smem_w * sizeof(float);
            int smem_limit = 0;
            CUDA_CHECK(cudaDeviceGetAttribute(&smem_limit, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0));
            if (smem_bytes > static_cast<size_t>(smem_limit)) continue;

            float t = avg_v8_bench_tile(input, output, params, AVG_V8_TILE_CANDIDATES[i], stream);
            if (t < best_time) {
                best_time = t;
                best_idx = i;
            }
        }
        cfg = AVG_V8_TILE_CANDIDATES[best_idx];
        {
            std::lock_guard<std::mutex> lock(avg_v8_cache_mutex);
            avg_v8_cache[key] = cfg;
        }
    }

    avg_v8_launch_config(input, output, params, cfg, stream);
    CUDA_CHECK(cudaGetLastError());
    NVTX_RANGE_POP();
}

// ──────────────────────────────────────────────────────────────
// v10: Persistent Kernel — one block per SM, atomic work queue
// ──────────────────────────────────────────────────────────────

template <typename T>
__global__ void avgpool_v10_kernel(const T* __restrict__ input, T* __restrict__ output,
                                   const AvgPoolParams params,
                                   uint32_t* __restrict__ work_counter,
                                   uint32_t total_tiles) {
    constexpr int TILE_OH = 8;
    constexpr int TILE_OW = 8;
    const int smem_h = min((TILE_OH - 1) * params.sh + (params.kh - 1) * params.dh + 1,
                           static_cast<int>(params.H + 2 * params.ph));
    const int smem_w = min((TILE_OW - 1) * params.sw + (params.kw - 1) * params.dw + 1,
                           static_cast<int>(params.W + 2 * params.pw));

    extern __shared__ __align__(16) unsigned char smem_raw[];
    const T* smem = reinterpret_cast<const T*>(smem_raw);
    T* smem_rw = reinterpret_cast<T*>(smem_raw);

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    uint32_t tile_id = atomicAdd(work_counter, 1);
    if (tile_id >= total_tiles) return;

    const uint32_t tiles_per_nc = static_cast<uint32_t>(params.OH) * static_cast<uint32_t>(params.OW);
    const uint32_t nc_id = tile_id / tiles_per_nc;
    const uint32_t tile_pos = tile_id % tiles_per_nc;
    const uint32_t n = nc_id / static_cast<uint32_t>(params.C);
    const uint32_t c = nc_id % static_cast<uint32_t>(params.C);
    const uint32_t tile_oh_start = tile_pos / static_cast<uint32_t>(params.OW) * TILE_OH;
    const uint32_t tile_ow_start = tile_pos % static_cast<uint32_t>(params.OW) * TILE_OW;

    for (int si = ty; si < smem_h; si += TILE_OH) {
        for (int sj = tx; sj < smem_w; sj += TILE_OW) {
            int ih = static_cast<int>(tile_oh_start) * params.sh + si * params.dh - params.ph;
            int iw = static_cast<int>(tile_ow_start) * params.sw + sj * params.dw - params.pw;
            T val = T(0);
            bool in_bounds = (ih >= 0 && ih < params.H && iw >= 0 && iw < params.W);
            if (in_bounds) {
                int64_t idx = ((n * params.H + ih) * params.W + iw) * params.C + c;
                val = input[idx];
            }
            smem_rw[si * smem_w + sj] = val;
        }
    }
    __syncthreads();

    const int local_oh = ty;
    const int local_ow = tx;
    const int oh = static_cast<int>(tile_oh_start) + local_oh;
    const int ow = static_cast<int>(tile_ow_start) + local_ow;

    if (oh < params.OH && ow < params.OW) {
        float sum = 0.0f;
        int count = 0;
        const int kh_end = min(params.kh, smem_h - local_oh * params.sh);
        const int kw_end = min(params.kw, smem_w - local_ow * params.sw);
        for (int ki = 0; ki < kh_end; ki++) {
            for (int kj = 0; kj < kw_end; kj++) {
                const int si = local_oh * params.sh + ki * params.dh;
                const int sj = local_ow * params.sw + kj * params.dw;
                bool valid = true;
                int src_ih = static_cast<int>(tile_oh_start) * params.sh + si * params.dh - params.ph;
                int src_iw = static_cast<int>(tile_ow_start) * params.sw + sj * params.dw - params.pw;
                if (!params.count_include_pad) {
                    valid = (src_ih >= 0 && src_ih < params.H && src_iw >= 0 && src_iw < params.W);
                }
                if (valid) {
                    sum += static_cast<float>(smem[si * smem_w + sj]);
                    count++;
                }
            }
        }
        int64_t divisor = count;
        if (params.divisor_override > 0) divisor = static_cast<int64_t>(params.divisor_override);
        else if (count == 0) divisor = 1;
        int64_t oidx = ((n * params.OH + oh) * params.OW + ow) * params.C + c;
        output[oidx] = static_cast<T>(sum / static_cast<float>(divisor));
    }
}

template <typename T>
static void avgpool_v10_launch(const T* d_input, T* d_output, const AvgPoolParams& params, cudaStream_t stream) {
    constexpr int TILE_OH = 8;
    constexpr int TILE_OW = 8;

    int num_sm = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&num_sm, cudaDevAttrMultiProcessorCount, 0));

    const int blocks_oh = static_cast<int>((params.OH + TILE_OH - 1) / TILE_OH);
    const int blocks_ow = static_cast<int>((params.OW + TILE_OW - 1) / TILE_OW);
    const uint32_t total_tiles = static_cast<uint32_t>(params.N * params.C * blocks_oh * blocks_ow);

    uint32_t* d_counter = nullptr;
    CUDA_CHECK(cudaMallocAsync(&d_counter, sizeof(uint32_t), stream));
    CUDA_CHECK(cudaMemsetAsync(d_counter, 0, sizeof(uint32_t), stream));

    const int smem_h = min((TILE_OH - 1) * params.sh + (params.kh - 1) * params.dh + 1,
                           static_cast<int>(params.H + 2 * params.ph));
    const int smem_w = min((TILE_OW - 1) * params.sw + (params.kw - 1) * params.dw + 1,
                           static_cast<int>(params.W + 2 * params.pw));
    size_t smem_bytes = static_cast<size_t>(smem_h) * smem_w * sizeof(T);

    dim3 block(TILE_OW, TILE_OH);
    dim3 grid(num_sm, 1, 1);

    if (smem_bytes > 49152) {
        CUDA_CHECK(cudaFuncSetAttribute(avgpool_v10_kernel<T>, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem_bytes)));
    }

    avgpool_v10_kernel<T><<<grid, block, smem_bytes, stream>>>(d_input, d_output, params, d_counter, total_tiles);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaFreeAsync(d_counter, stream));
}

void avgpool_v10(const float* input, float* output, const AvgPoolParams& params, cudaStream_t stream) {
    NVTX_RANGE_PUSH_C("avgpool_v10_f32", NVTX_COLOR_AVGPOOL);
    avgpool_v10_launch(input, output, params, stream);
    NVTX_RANGE_POP();
}

void avgpool_v10(const half* input, half* output, const AvgPoolParams& params, cudaStream_t stream) {
    NVTX_RANGE_PUSH_C("avgpool_v10_f16", NVTX_COLOR_AVGPOOL);
    avgpool_v10_launch(input, output, params, stream);
    NVTX_RANGE_POP();
}

// ──────────────────────────────────────────────────────────────
// v9: TMA Warp-Specialized Pipeline — producer/consumer warp split
// ──────────────────────────────────────────────────────────────

template <int TILE_OH, int TILE_OW, typename T>
__global__ void __launch_bounds__(256)
avgpool_v9_ws_kernel(const T* __restrict__ input, T* __restrict__ output,
                     const AvgPoolParams params, int blocks_oh, int blocks_ow)
{
    const int n = blockIdx.z;
    const int c = blockIdx.y;
    const int tile_idx = blockIdx.x;
    const int tile_oh = tile_idx / blocks_ow;
    const int tile_ow = tile_idx % blocks_ow;

    const int smem_h = min((TILE_OH - 1) * params.sh + (params.kh - 1) * params.dh + 1,
                           static_cast<int>(params.H + 2 * params.ph));
    const int smem_w = min((TILE_OW - 1) * params.sw + (params.kw - 1) * params.dw + 1,
                           static_cast<int>(params.W + 2 * params.pw));

    extern __shared__ __align__(16) unsigned char smem_raw[];
    T* smem = reinterpret_cast<T*>(smem_raw);

    const int warp_id = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;

    // Producer warp (warp 0): async loads with cp.async
    if (warp_id == 0) {
        int elements = smem_h * smem_w;

        for (int i = lane; i < elements; i += 32) {
            int si = i / smem_w;
            int sj = i % smem_w;
            int ih = tile_oh * params.sh + si * params.dh - params.ph;
            int iw = tile_ow * params.sw + sj * params.dw - params.pw;
            T val = T(0);
            bool in_bounds = (ih >= 0 && ih < params.H && iw >= 0 && iw < params.W);
            if (in_bounds) {
                int64_t idx = ((n * params.H + ih) * params.W + iw) * params.C + c;
                val = input[idx];
            }
            smem[i] = val;
        }
        asm volatile("cp.async.commit_group;\n" ::);
    }

    // Consumer warps (1-7): wait and compute average
    if (warp_id >= 1) {
        asm volatile("cp.async.wait_group 0;\n" ::);
        __syncthreads();

        const int local_id = threadIdx.x - 32;

        if (local_id < TILE_OH * TILE_OW && local_id >= 0) {
            const int local_oh = local_id / TILE_OW;
            const int local_ow = local_id % TILE_OW;
            const int oh = tile_oh + local_oh;
            const int ow = tile_ow + local_ow;

            if (oh < params.OH && ow < params.OW) {
                float sum = 0.0f;
                int count = 0;
                const int kh_end = min(params.kh, smem_h - local_oh * params.sh);
                const int kw_end = min(params.kw, smem_w - local_ow * params.sw);
                for (int ki = 0; ki < kh_end; ki++) {
                    for (int kj = 0; kj < kw_end; kj++) {
                        const int si = local_oh * params.sh + ki * params.dh;
                        const int sj = local_ow * params.sw + kj * params.dw;
                        bool valid = true;
                        int src_ih = tile_oh * params.sh + si * params.dh - params.ph;
                        int src_iw = tile_ow * params.sw + sj * params.dw - params.pw;
                        if (!params.count_include_pad) {
                            valid = (src_ih >= 0 && src_ih < params.H && src_iw >= 0 && src_iw < params.W);
                        }
                        if (valid) {
                            sum += static_cast<float>(smem[si * smem_w + sj]);
                            count++;
                        }
                    }
                }
                int64_t divisor = count;
                if (params.divisor_override > 0) divisor = static_cast<int64_t>(params.divisor_override);
                else if (count == 0) divisor = 1;
                int64_t oidx = ((n * params.OH + oh) * params.OW + ow) * params.C + c;
                output[oidx] = static_cast<T>(sum / static_cast<float>(divisor));
            }
        }
    }
}

static bool has_cp_async() {
    int major = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, 0));
    return major >= 8;
}

void avgpool_v9(const float* input, float* output, const AvgPoolParams& params, cudaStream_t stream) {
    NVTX_RANGE_PUSH_C("avgpool_v9_f32", NVTX_COLOR_AVGPOOL);

    if (!has_cp_async()) {
        NVTX_RANGE_POP();
        avgpool_v2(input, output, params, stream);
        return;
    }

    constexpr int TILE_OH = 8;
    constexpr int TILE_OW = 8;
    const int blocks_oh = static_cast<int>((params.OH + TILE_OH - 1) / TILE_OH);
    const int blocks_ow = static_cast<int>((params.OW + TILE_OW - 1) / TILE_OW);
    const int smem_h = min((TILE_OH - 1) * params.sh + (params.kh - 1) * params.dh + 1,
                           static_cast<int>(params.H + 2 * params.ph));
    const int smem_w = min((TILE_OW - 1) * params.sw + (params.kw - 1) * params.dw + 1,
                           static_cast<int>(params.W + 2 * params.pw));
    size_t smem_bytes = static_cast<size_t>(smem_h) * smem_w * sizeof(float);

    dim3 block(256);
    dim3 grid(blocks_oh * blocks_ow, static_cast<int>(params.C), static_cast<int>(params.N));

    if (smem_bytes > 49152) {
        CUDA_CHECK(cudaFuncSetAttribute(avgpool_v9_ws_kernel<TILE_OH, TILE_OW, float>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem_bytes)));
    }

    avgpool_v9_ws_kernel<TILE_OH, TILE_OW, float><<<grid, block, smem_bytes, stream>>>(
        input, output, params, blocks_oh, blocks_ow);
    CUDA_CHECK(cudaGetLastError());
    NVTX_RANGE_POP();
}

void avgpool_v9(const half* input, half* output, const AvgPoolParams& params, cudaStream_t stream) {
    NVTX_RANGE_PUSH_C("avgpool_v9_f16", NVTX_COLOR_AVGPOOL);

    if (!has_cp_async()) {
        NVTX_RANGE_POP();
        avgpool_v2(input, output, params, stream);
        return;
    }

    constexpr int TILE_OH = 8;
    constexpr int TILE_OW = 8;
    const int blocks_oh = static_cast<int>((params.OH + TILE_OH - 1) / TILE_OH);
    const int blocks_ow = static_cast<int>((params.OW + TILE_OW - 1) / TILE_OW);
    const int smem_h = min((TILE_OH - 1) * params.sh + (params.kh - 1) * params.dh + 1,
                           static_cast<int>(params.H + 2 * params.ph));
    const int smem_w = min((TILE_OW - 1) * params.sw + (params.kw - 1) * params.dw + 1,
                           static_cast<int>(params.W + 2 * params.pw));
    size_t smem_bytes = static_cast<size_t>(smem_h) * smem_w * sizeof(half);

    dim3 block(256);
    dim3 grid(blocks_oh * blocks_ow, static_cast<int>(params.C), static_cast<int>(params.N));

    if (smem_bytes > 49152) {
        CUDA_CHECK(cudaFuncSetAttribute(avgpool_v9_ws_kernel<TILE_OH, TILE_OW, half>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem_bytes)));
    }

    avgpool_v9_ws_kernel<TILE_OH, TILE_OW, half><<<grid, block, smem_bytes, stream>>>(
        input, output, params, blocks_oh, blocks_ow);
    CUDA_CHECK(cudaGetLastError());
    NVTX_RANGE_POP();
}
