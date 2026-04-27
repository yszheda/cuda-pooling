#include "pooling.cuh"

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
