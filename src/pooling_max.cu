#include "pooling.cuh"

// MaxPool2d v0: naive kernel — one thread per output element
// Grid:  blockIdx.z = batch index N, blockIdx.x * blockDim.x + threadIdx.x = flat (OH*OW*C)
template <typename T>
__global__ void maxpool_v0_kernel(
    const T* __restrict__ input,
    T* __restrict__ output,
    const PoolParams params)
{
    const int64_t n = blockIdx.z;
    const int64_t flat = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t OHW_C = params.OH * params.OW * params.C;
    if (n >= params.N || flat >= OHW_C) return;

    const int64_t oh = flat / (params.OW * params.C);
    const int64_t ow = (flat / params.C) % params.OW;
    const int64_t c  = flat % params.C;

    float maxval = -INFINITY;

    for (int kh_i = 0; kh_i < params.kh; ++kh_i) {
        const int64_t ih = oh * params.sh - params.ph + static_cast<int64_t>(kh_i) * params.dh;
        if (ih < 0 || ih >= params.H) continue;
        for (int kw_i = 0; kw_i < params.kw; ++kw_i) {
            const int64_t iw = ow * params.sw - params.pw + static_cast<int64_t>(kw_i) * params.dw;
            if (iw < 0 || iw >= params.W) continue;

            const int64_t in_idx = ((n * params.H + ih) * params.W + iw) * params.C + c;
            float val = static_cast<float>(input[in_idx]);
            if (val > maxval) maxval = val;
        }
    }

    const int64_t out_idx = ((n * params.OH + oh) * params.OW + ow) * params.C + c;
    output[out_idx] = static_cast<T>(maxval);
}

void maxpool_v0(const float* input, float* output, const PoolParams& params, cudaStream_t stream) {
    const int64_t total = params.OH * params.OW * params.C;
    const int threads = 256;
    const int blocks_x = static_cast<int>((total + threads - 1) / threads);
    dim3 grid(blocks_x, 1, static_cast<int>(params.N));
    maxpool_v0_kernel<float><<<grid, threads, 0, stream>>>(input, output, params);
    CUDA_CHECK(cudaGetLastError());
}

void maxpool_v0(const half* input, half* output, const PoolParams& params, cudaStream_t stream) {
    const int64_t total = params.OH * params.OW * params.C;
    const int threads = 256;
    const int blocks_x = static_cast<int>((total + threads - 1) / threads);
    dim3 grid(blocks_x, 1, static_cast<int>(params.N));
    maxpool_v0_kernel<half><<<grid, threads, 0, stream>>>(input, output, params);
    CUDA_CHECK(cudaGetLastError());
}

// ============================================================================
// MaxPool2d v1: shared memory tiling kernel
// Each block handles a TILE_OH x TILE_OW tile of output spatial positions
// for one (n, c) pair. The block cooperatively loads the corresponding
// input tile (output tile + halo region) into shared memory, then each
// thread computes its max from shared memory.
// ============================================================================

template <int TILE_OH = 8, int TILE_OW = 8>
__global__ void maxpool_v1_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
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
            sdata[i] = -INFINITY;
        }
    }

    __syncthreads();

    // Each thread computes one output position
    if (oh >= params.OH || ow >= params.OW) return;

    float maxval = -INFINITY;
    for (int kh_i = 0; kh_i < params.kh; ++kh_i) {
        const int sih = th * params.sh + kh_i * params.dh;
        if (sih >= smem_h) continue;  // beyond smem => input out of bounds
        for (int kw_i = 0; kw_i < params.kw; ++kw_i) {
            const int siw = tw * params.sw + kw_i * params.dw;
            if (siw >= smem_w) continue;  // beyond smem => input out of bounds
            const float val = sdata[sih * smem_w + siw];
            if (val > maxval) maxval = val;
        }
    }

    const int64_t out_idx = ((static_cast<int64_t>(n) * params.OH + oh) * params.OW + ow) * params.C + c;
    output[out_idx] = maxval;
}

// half specialization: same logic, casts input/output via float
template <int TILE_OH = 8, int TILE_OW = 8>
__global__ void maxpool_v1_kernel_half(
    const half* __restrict__ input,
    half* __restrict__ output,
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
    output[out_idx] = static_cast<half>(maxval);
}

void maxpool_v1(const float* input, float* output, const PoolParams& params, cudaStream_t stream) {
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

    maxpool_v1_kernel<TILE_OH, TILE_OW><<<grid, block, smem_bytes, stream>>>(
        input, output, params, blocks_oh, blocks_ow, smem_h, smem_w);
    CUDA_CHECK(cudaGetLastError());
}

void maxpool_v1(const half* input, half* output, const PoolParams& params, cudaStream_t stream) {
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

    maxpool_v1_kernel_half<TILE_OH, TILE_OW><<<grid, block, smem_bytes, stream>>>(
        input, output, params, blocks_oh, blocks_ow, smem_h, smem_w);
    CUDA_CHECK(cudaGetLastError());
}

// ============================================================================
// MaxPool2d v2: vectorized loads kernel
// Uses float4 (4 channels) for fp32 and half2 (2 channels) for fp16
// to coalesce global memory access. Falls back to v0 if C % VEC != 0.
// ============================================================================

template <typename T, int VEC>
__global__ void maxpool_v2_kernel(
    const T* __restrict__ input,
    T* __restrict__ output,
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

    // Initialize VEC max values to -infinity
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

            // Vectorized load
            if constexpr (std::is_same_v<T, float> && VEC == 4) {
                float4 vec = *reinterpret_cast<const float4*>(&input[in_idx]);
                if (vec.x > maxval[0]) maxval[0] = vec.x;
                if (vec.y > maxval[1]) maxval[1] = vec.y;
                if (vec.z > maxval[2]) maxval[2] = vec.z;
                if (vec.w > maxval[3]) maxval[3] = vec.w;
            } else if constexpr (std::is_same_v<T, half> && VEC == 2) {
                half2 vec = *reinterpret_cast<const half2*>(&input[in_idx]);
                float lo = __low2float(vec);
                float hi = __high2float(vec);
                if (lo > maxval[0]) maxval[0] = lo;
                if (hi > maxval[1]) maxval[1] = hi;
            }
        }
    }

    // Write VEC output values
    const int64_t out_idx = ((n * params.OH + oh) * params.OW + ow) * params.C + c;

    if constexpr (std::is_same_v<T, float> && VEC == 4) {
        float4 out_vec;
        out_vec.x = maxval[0];
        out_vec.y = maxval[1];
        out_vec.z = maxval[2];
        out_vec.w = maxval[3];
        *reinterpret_cast<float4*>(&output[out_idx]) = out_vec;
    } else if constexpr (std::is_same_v<T, half> && VEC == 2) {
        half2 out_vec = __floats2half2_rn(maxval[0], maxval[1]);
        *reinterpret_cast<half2*>(&output[out_idx]) = out_vec;
    }
}

void maxpool_v2(const float* input, float* output, const PoolParams& params, cudaStream_t stream) {
    constexpr int VEC = 4;
    if (params.C % VEC != 0) {
        // Fall back to v0 if C is not aligned
        maxpool_v0(input, output, params, stream);
        return;
    }
    const int64_t C_vec = params.C / VEC;
    const int64_t total = params.OH * params.OW * C_vec;
    const int threads = 256;
    const int blocks_x = static_cast<int>((total + threads - 1) / threads);
    dim3 grid(blocks_x, 1, static_cast<int>(params.N));
    maxpool_v2_kernel<float, VEC><<<grid, threads, 0, stream>>>(input, output, params);
    CUDA_CHECK(cudaGetLastError());
}

void maxpool_v2(const half* input, half* output, const PoolParams& params, cudaStream_t stream) {
    constexpr int VEC = 2;
    if (params.C % VEC != 0) {
        maxpool_v0(input, output, params, stream);
        return;
    }
    const int64_t C_vec = params.C / VEC;
    const int64_t total = params.OH * params.OW * C_vec;
    const int threads = 256;
    const int blocks_x = static_cast<int>((total + threads - 1) / threads);
    dim3 grid(blocks_x, 1, static_cast<int>(params.N));
    maxpool_v2_kernel<half, VEC><<<grid, threads, 0, stream>>>(input, output, params);
    CUDA_CHECK(cudaGetLastError());
}

// ============================================================================
// MaxPool2d v3: register blocking kernel
// Each thread computes BLOCK consecutive output rows for the same (ow, c).
// This increases arithmetic intensity and reuses input data already in
// registers when stride < kernel_size (adjacent output windows share input).
// Falls back to v0 if OH < BLOCK or OH % BLOCK != 0.
// ============================================================================

template <typename T, int BLOCK>
__global__ void maxpool_v3_kernel(
    const T* __restrict__ input,
    T* __restrict__ output,
    const PoolParams params)
{
    const int64_t n = blockIdx.z;
    const int64_t OH_block = params.OH / BLOCK;  // number of row-blocks
    const int64_t total = OH_block * params.OW * params.C;
    const int64_t flat = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (n >= params.N || flat >= total) return;

    const int64_t c = flat % params.C;
    const int64_t ow = (flat / params.C) % params.OW;
    const int64_t oh_base = (flat / params.C / params.OW) * BLOCK;

    float maxval[BLOCK];
    #pragma unroll
    for (int b = 0; b < BLOCK; ++b)
        maxval[b] = -INFINITY;

    for (int kh_i = 0; kh_i < params.kh; ++kh_i) {
        for (int kw_i = 0; kw_i < params.kw; ++kw_i) {
            const int64_t iw = ow * params.sw - params.pw + static_cast<int64_t>(kw_i) * params.dw;
            if (iw < 0 || iw >= params.W) continue;

            #pragma unroll
            for (int b = 0; b < BLOCK; ++b) {
                const int64_t oh = oh_base + b;
                const int64_t ih = oh * params.sh - params.ph + static_cast<int64_t>(kh_i) * params.dh;
                if (ih < 0 || ih >= params.H) continue;

                const int64_t in_idx = ((n * params.H + ih) * params.W + iw) * params.C + c;
                float val = static_cast<float>(input[in_idx]);
                if (val > maxval[b]) maxval[b] = val;
            }
        }
    }

    #pragma unroll
    for (int b = 0; b < BLOCK; ++b) {
        const int64_t oh = oh_base + b;
        const int64_t out_idx = ((n * params.OH + oh) * params.OW + ow) * params.C + c;
        output[out_idx] = static_cast<T>(maxval[b]);
    }
}

void maxpool_v3(const float* input, float* output, const PoolParams& params, cudaStream_t stream) {
    constexpr int BLOCK = 4;
    if (params.OH < BLOCK || params.OH % BLOCK != 0) {
        maxpool_v0(input, output, params, stream);
        return;
    }
    const int64_t OH_block = params.OH / BLOCK;
    const int64_t total = OH_block * params.OW * params.C;
    const int threads = 256;
    const int blocks_x = static_cast<int>((total + threads - 1) / threads);
    dim3 grid(blocks_x, 1, static_cast<int>(params.N));
    maxpool_v3_kernel<float, BLOCK><<<grid, threads, 0, stream>>>(input, output, params);
    CUDA_CHECK(cudaGetLastError());
}

void maxpool_v3(const half* input, half* output, const PoolParams& params, cudaStream_t stream) {
    constexpr int BLOCK = 4;
    if (params.OH < BLOCK || params.OH % BLOCK != 0) {
        maxpool_v0(input, output, params, stream);
        return;
    }
    const int64_t OH_block = params.OH / BLOCK;
    const int64_t total = OH_block * params.OW * params.C;
    const int threads = 256;
    const int blocks_x = static_cast<int>((total + threads - 1) / threads);
    dim3 grid(blocks_x, 1, static_cast<int>(params.N));
    maxpool_v3_kernel<half, BLOCK><<<grid, threads, 0, stream>>>(input, output, params);
    CUDA_CHECK(cudaGetLastError());
}

// ============================================================================
// MaxPool2d v4: warp-level reduce kernel
// Each warp (32 threads) cooperatively handles one output position (n, oh, ow, c).
// The karea = kh*kw elements of the kernel window are distributed across the 32
// lanes of the warp. Each lane processes its subset (serial loop within lane),
// then a warp shuffle reduction computes the final max.
// For large kernels (karea > 32), each lane handles multiple elements.
// For small kernels (karea < 32), idle lanes contribute -INFINITY (max identity).
// ============================================================================

template <typename T>
__global__ void maxpool_v4_kernel(
    const T* __restrict__ input,
    T* __restrict__ output,
    const PoolParams params)
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
    float myval = -INFINITY;

    // Each lane handles its subset of kernel positions
    for (int ki = lane; ki < karea; ki += 32) {
        const int kh_idx = ki / params.kw;
        const int kw_idx = ki % params.kw;
        const int64_t ih = oh * params.sh - params.ph + static_cast<int64_t>(kh_idx) * params.dh;
        const int64_t iw = ow * params.sw - params.pw + static_cast<int64_t>(kw_idx) * params.dw;
        if (ih >= 0 && ih < params.H && iw >= 0 && iw < params.W) {
            const int64_t in_idx = ((n * params.H + ih) * params.W + iw) * params.C + c;
            float val = static_cast<float>(input[in_idx]);
            if (val > myval) myval = val;
        }
    }

    // Warp-level max reduction using __shfl_down_sync
    for (int offset = 16; offset > 0; offset >>= 1) {
        float other = __shfl_down_sync(0xFFFFFFFF, myval, offset);
        if (other > myval) myval = other;
    }

    // Lane 0 writes the result
    if (lane == 0) {
        const int64_t out_idx = ((n * params.OH + oh) * params.OW + ow) * params.C + c;
        output[out_idx] = static_cast<T>(myval);
    }
}

void maxpool_v4(const float* input, float* output, const PoolParams& params, cudaStream_t stream) {
    const int64_t total = params.OH * params.OW * params.C;
    const int threads = 256;
    const int warps_per_block = threads / 32;  // 8
    const int blocks_x = static_cast<int>((total + warps_per_block - 1) / warps_per_block);
    dim3 grid(blocks_x, 1, static_cast<int>(params.N));
    maxpool_v4_kernel<float><<<grid, threads, 0, stream>>>(input, output, params);
    CUDA_CHECK(cudaGetLastError());
}

void maxpool_v4(const half* input, half* output, const PoolParams& params, cudaStream_t stream) {
    const int64_t total = params.OH * params.OW * params.C;
    const int threads = 256;
    const int warps_per_block = threads / 32;  // 8
    const int blocks_x = static_cast<int>((total + warps_per_block - 1) / warps_per_block);
    dim3 grid(blocks_x, 1, static_cast<int>(params.N));
    maxpool_v4_kernel<half><<<grid, threads, 0, stream>>>(input, output, params);
    CUDA_CHECK(cudaGetLastError());
}

// ============================================================================
// MaxPool2d v5: double buffer / pipeline kernel
// Each block handles 2 consecutive channels (c and c+1), using double-buffered
// shared memory. The block loads channel c's tile into buf[0], computes the max,
// then loads channel c+1's tile into buf[1] while the warp scheduler can
// interleave memory and compute from different warps (implicit overlap).
// Double-buffering reduces the grid dimension by half (ceil(C/2) channel-pairs
// instead of C channels) and keeps both tiles resident in smem.
// If C is odd, the last channel pair only uses buf[0].
// Falls back to v1 if C < 2 or smem is too large.
// ============================================================================

template <int TILE_OH = 8, int TILE_OW = 8>
__global__ void maxpool_v5_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
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
            buf[0][i] = -INFINITY;
        }
    }
    __syncthreads();

    // Compute max from buf[0] for channel c0
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

    // Write output for channel c0
    if (oh < params.OH && ow < params.OW) {
        const int64_t out_idx = ((static_cast<int64_t>(n) * params.OH + oh) * params.OW + ow) * params.C + c0;
        output[out_idx] = maxval_c0;
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
                buf[1][i] = -INFINITY;
            }
        }
        __syncthreads();

        // Phase 3: Compute max from buf[1] for channel c1
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
            output[out_idx] = maxval_c1;
        }
    }
}

// half specialization
template <int TILE_OH = 8, int TILE_OW = 8>
__global__ void maxpool_v5_kernel_half(
    const half* __restrict__ input,
    half* __restrict__ output,
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

    // Phase 1: Load channel c0 into buf[0]
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

    // Compute max from buf[0] for channel c0
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

    // Write output for channel c0
    if (oh < params.OH && ow < params.OW) {
        const int64_t out_idx = ((static_cast<int64_t>(n) * params.OH + oh) * params.OW + ow) * params.C + c0;
        output[out_idx] = static_cast<half>(maxval_c0);
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
                buf[1][i] = static_cast<float>(input[in_idx]);
            } else {
                buf[1][i] = -INFINITY;
            }
        }
        __syncthreads();

        // Phase 3: Compute max from buf[1] for channel c1
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
            output[out_idx] = static_cast<half>(maxval_c1);
        }
    }
}

void maxpool_v5(const float* input, float* output, const PoolParams& params, cudaStream_t stream) {
    constexpr int TILE_OH = 8;
    constexpr int TILE_OW = 8;

    if (params.C < 2) {
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
        maxpool_v1(input, output, params, stream);
        return;
    }

    const int c_pairs = static_cast<int>((params.C + 1) / 2);
    dim3 block(TILE_OW, TILE_OH);
    dim3 grid(blocks_oh * blocks_ow, c_pairs, static_cast<int>(params.N));

    if (smem_bytes > 49152) {
        CUDA_CHECK(cudaFuncSetAttribute(
            maxpool_v5_kernel<TILE_OH, TILE_OW>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(smem_bytes)));
    }

    maxpool_v5_kernel<TILE_OH, TILE_OW><<<grid, block, smem_bytes, stream>>>(
        input, output, params, blocks_oh, blocks_ow, smem_h, smem_w);
    CUDA_CHECK(cudaGetLastError());
}

void maxpool_v5(const half* input, half* output, const PoolParams& params, cudaStream_t stream) {
    constexpr int TILE_OH = 8;
    constexpr int TILE_OW = 8;

    if (params.C < 2) {
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
        maxpool_v1(input, output, params, stream);
        return;
    }

    const int c_pairs = static_cast<int>((params.C + 1) / 2);
    dim3 block(TILE_OW, TILE_OH);
    dim3 grid(blocks_oh * blocks_ow, c_pairs, static_cast<int>(params.N));

    if (smem_bytes > 49152) {
        CUDA_CHECK(cudaFuncSetAttribute(
            maxpool_v5_kernel_half<TILE_OH, TILE_OW>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(smem_bytes)));
    }

    maxpool_v5_kernel_half<TILE_OH, TILE_OW><<<grid, block, smem_bytes, stream>>>(
        input, output, params, blocks_oh, blocks_ow, smem_h, smem_w);
    CUDA_CHECK(cudaGetLastError());
}

// ============================================================================
// MaxPool2d v6: warp specialization kernel
// Split warps in a block into two roles: "load warps" that fetch data from
// global memory into shared memory, and "compute warps" that read from shared
// memory and compute the max. This decouples memory latency from compute.
//
// Block size: 256 threads = 8 warps
// NUM_LOAD_WARPS = 2 (warp 0-1), NUM_COMPUTE_WARPS = 6 (warp 2-7)
// Same tiling as v1: TILE_OH=8, TILE_OW=8 per (n, c) pair
// Falls back to v1 if smem is too large.
// ============================================================================

template <int TILE_OH = 8, int TILE_OW = 8, int NUM_LOAD_WARPS = 2, int NUM_COMPUTE_WARPS = 6>
__global__ void maxpool_v6_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const PoolParams params,
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

    // Input tile origin in global coordinates
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
                sdata[i] = -INFINITY;
            }
        }
    }
    __syncthreads();

    // Phase 2: Compute warps compute max from shared memory
    if (!is_load_warp) {
        const int compute_tid = tid - NUM_LOAD_THREADS;  // 0..191

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
            output[out_idx] = maxval;
        }
    }
}

// half specialization
template <int TILE_OH = 8, int TILE_OW = 8, int NUM_LOAD_WARPS = 2, int NUM_COMPUTE_WARPS = 6>
__global__ void maxpool_v6_kernel_half(
    const half* __restrict__ input,
    half* __restrict__ output,
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
            output[out_idx] = static_cast<half>(maxval);
        }
    }
}

void maxpool_v6(const float* input, float* output, const PoolParams& params, cudaStream_t stream) {
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

    // Fall back to v1 if smem is too large
    int smem_limit = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&smem_limit, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0));
    if (smem_bytes > static_cast<size_t>(smem_limit)) {
        maxpool_v1(input, output, params, stream);
        return;
    }

    dim3 block(BLOCK_SIZE);
    dim3 grid(blocks_oh * blocks_ow, static_cast<int>(params.C), static_cast<int>(params.N));

    if (smem_bytes > 49152) {
        CUDA_CHECK(cudaFuncSetAttribute(
            maxpool_v6_kernel<TILE_OH, TILE_OW>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(smem_bytes)));
    }

    maxpool_v6_kernel<TILE_OH, TILE_OW><<<grid, block, smem_bytes, stream>>>(
        input, output, params, blocks_oh, blocks_ow, smem_h, smem_w);
    CUDA_CHECK(cudaGetLastError());
}

void maxpool_v6(const half* input, half* output, const PoolParams& params, cudaStream_t stream) {
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
        maxpool_v1(input, output, params, stream);
        return;
    }

    dim3 block(BLOCK_SIZE);
    dim3 grid(blocks_oh * blocks_ow, static_cast<int>(params.C), static_cast<int>(params.N));

    if (smem_bytes > 49152) {
        CUDA_CHECK(cudaFuncSetAttribute(
            maxpool_v6_kernel_half<TILE_OH, TILE_OW>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(smem_bytes)));
    }

    maxpool_v6_kernel_half<TILE_OH, TILE_OW><<<grid, block, smem_bytes, stream>>>(
        input, output, params, blocks_oh, blocks_ow, smem_h, smem_w);
    CUDA_CHECK(cudaGetLastError());
}

// ============================================================================
// MaxPool2d v7: alternative grid/block mappings
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
__global__ void maxpool_v7_mappingB_kernel(
    const T* __restrict__ input,
    T* __restrict__ output,
    const PoolParams params,
    int C_groups)
{
    const int n = blockIdx.z / C_groups;
    const int c_group = blockIdx.z - n * C_groups;
    const int c = c_group * 4 + threadIdx.z;
    const int oh = blockIdx.y * 8 + threadIdx.y;
    const int ow = blockIdx.x * 8 + threadIdx.x;

    if (n >= params.N || oh >= params.OH || ow >= params.OW || c >= params.C) return;

    float maxval = -INFINITY;

    for (int kh_i = 0; kh_i < params.kh; ++kh_i) {
        const int64_t ih = oh * params.sh - params.ph + static_cast<int64_t>(kh_i) * params.dh;
        if (ih < 0 || ih >= params.H) continue;
        for (int kw_i = 0; kw_i < params.kw; ++kw_i) {
            const int64_t iw = ow * params.sw - params.pw + static_cast<int64_t>(kw_i) * params.dw;
            if (iw < 0 || iw >= params.W) continue;
            const int64_t in_idx = ((static_cast<int64_t>(n) * params.H + ih) * params.W + iw) * params.C + c;
            float val = static_cast<float>(input[in_idx]);
            if (val > maxval) maxval = val;
        }
    }

    const int64_t out_idx = ((static_cast<int64_t>(n) * params.OH + oh) * params.OW + ow) * params.C + c;
    output[out_idx] = static_cast<T>(maxval);
}

// --- Mapping C: Channel-Major ---
// Each block covers 256 channels for one (oh, ow) position
// blockDim = (256, 1, 1)
// Grid: (OW, OH, N * ceil(C/256))
// Thread: c = c_group*256 + threadIdx.x, ow = blockIdx.x, oh = blockIdx.y

template <typename T>
__global__ void maxpool_v7_mappingC_kernel(
    const T* __restrict__ input,
    T* __restrict__ output,
    const PoolParams params,
    int C_groups)
{
    const int n = blockIdx.z / C_groups;
    const int c_group = blockIdx.z - n * C_groups;
    const int c = c_group * 256 + threadIdx.x;
    const int oh = blockIdx.y;
    const int ow = blockIdx.x;

    if (n >= params.N || oh >= params.OH || ow >= params.OW || c >= params.C) return;

    float maxval = -INFINITY;

    for (int kh_i = 0; kh_i < params.kh; ++kh_i) {
        const int64_t ih = oh * params.sh - params.ph + static_cast<int64_t>(kh_i) * params.dh;
        if (ih < 0 || ih >= params.H) continue;
        for (int kw_i = 0; kw_i < params.kw; ++kw_i) {
            const int64_t iw = ow * params.sw - params.pw + static_cast<int64_t>(kw_i) * params.dw;
            if (iw < 0 || iw >= params.W) continue;
            const int64_t in_idx = ((static_cast<int64_t>(n) * params.H + ih) * params.W + iw) * params.C + c;
            float val = static_cast<float>(input[in_idx]);
            if (val > maxval) maxval = val;
        }
    }

    const int64_t out_idx = ((static_cast<int64_t>(n) * params.OH + oh) * params.OW + ow) * params.C + c;
    output[out_idx] = static_cast<T>(maxval);
}

// --- Mapping D: Hybrid warp-spatial + channel-vectorized ---
// Each warp handles a 4x4 spatial tile with 4 channels via vectorized loads
// blockDim = (32, 8, 1) — 8 warps per block, each warp covers 4x4 spatial + 4 channels
// Grid: (ceil(OW/4), ceil(OH/4), N * ceil(C/32))
// Lanes 0-15 of each warp handle 4x4 spatial positions, loading 4 channels via float4/half2x2

template <typename T>
__global__ void maxpool_v7_mappingD_kernel(
    const T* __restrict__ input,
    T* __restrict__ output,
    const PoolParams params,
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

            // Vectorized load of 4 channels
            if constexpr (std::is_same_v<T, float>) {
                float4 vec = *reinterpret_cast<const float4*>(&input[in_idx]);
                if (vec.x > maxval[0]) maxval[0] = vec.x;
                if (vec.y > maxval[1]) maxval[1] = vec.y;
                if (vec.z > maxval[2]) maxval[2] = vec.z;
                if (vec.w > maxval[3]) maxval[3] = vec.w;
            } else {
                half2 vec0 = *reinterpret_cast<const half2*>(&input[in_idx]);
                half2 vec1 = *reinterpret_cast<const half2*>(&input[in_idx + 2]);
                float lo0 = __low2float(vec0), hi0 = __high2float(vec0);
                float lo1 = __low2float(vec1), hi1 = __high2float(vec1);
                if (lo0 > maxval[0]) maxval[0] = lo0;
                if (hi0 > maxval[1]) maxval[1] = hi0;
                if (lo1 > maxval[2]) maxval[2] = lo1;
                if (hi1 > maxval[3]) maxval[3] = hi1;
            }
        }
    }

    // Write 4 output values
    const int64_t out_idx = ((static_cast<int64_t>(n) * params.OH + oh) * params.OW + ow) * params.C + c_warp_base;

    if constexpr (std::is_same_v<T, float>) {
        float4 out_vec;
        out_vec.x = maxval[0];
        out_vec.y = maxval[1];
        out_vec.z = maxval[2];
        out_vec.w = maxval[3];
        *reinterpret_cast<float4*>(&output[out_idx]) = out_vec;
    } else {
        half2 out0 = __floats2half2_rn(maxval[0], maxval[1]);
        half2 out1 = __floats2half2_rn(maxval[2], maxval[3]);
        *reinterpret_cast<half2*>(&output[out_idx]) = out0;
        *reinterpret_cast<half2*>(&output[out_idx + 2]) = out1;
    }
}

// --- v7 launcher: dispatches to the appropriate mapping kernel ---

void maxpool_v7(const float* input, float* output, const PoolParams& params, int mapping, cudaStream_t stream) {
    switch (mapping) {
        case 0: {
            // Mapping A: 1D flat — same as v0
            maxpool_v0(input, output, params, stream);
            break;
        }
        case 1: {
            // Mapping B: 2D spatial tiling
            const int C_groups = static_cast<int>((params.C + 3) / 4);
            const int64_t grid_z_64 = params.N * C_groups;
            if (grid_z_64 > 65535) {
                maxpool_v0(input, output, params, stream);
                break;
            }
            dim3 block(8, 8, 4);
            dim3 grid(
                static_cast<int>((params.OW + 7) / 8),
                static_cast<int>((params.OH + 7) / 8),
                static_cast<int>(grid_z_64));
            maxpool_v7_mappingB_kernel<float><<<grid, block, 0, stream>>>(input, output, params, C_groups);
            CUDA_CHECK(cudaGetLastError());
            break;
        }
        case 2: {
            // Mapping C: channel-major
            const int C_groups = static_cast<int>((params.C + 255) / 256);
            const int64_t grid_z_64 = params.N * C_groups;
            if (params.OW > 65535 || params.OH > 65535 || grid_z_64 > 65535) {
                maxpool_v0(input, output, params, stream);
                break;
            }
            dim3 block(256, 1, 1);
            dim3 grid(
                static_cast<int>(params.OW),
                static_cast<int>(params.OH),
                static_cast<int>(grid_z_64));
            maxpool_v7_mappingC_kernel<float><<<grid, block, 0, stream>>>(input, output, params, C_groups);
            CUDA_CHECK(cudaGetLastError());
            break;
        }
        case 3: {
            // Mapping D: hybrid warp-spatial + vectorized
            if (params.C % 4 != 0) {
                maxpool_v0(input, output, params, stream);
                break;
            }
            const int C_groups = static_cast<int>((params.C + 31) / 32);
            const int64_t grid_z_64 = params.N * C_groups;
            if (grid_z_64 > 65535) {
                maxpool_v0(input, output, params, stream);
                break;
            }
            dim3 block(32, 8, 1);
            dim3 grid(
                static_cast<int>((params.OW + 3) / 4),
                static_cast<int>((params.OH + 3) / 4),
                static_cast<int>(grid_z_64));
            maxpool_v7_mappingD_kernel<float><<<grid, block, 0, stream>>>(input, output, params, C_groups);
            CUDA_CHECK(cudaGetLastError());
            break;
        }
        default:
            throw std::invalid_argument("unsupported mapping: " + std::to_string(mapping));
    }
}

void maxpool_v7(const half* input, half* output, const PoolParams& params, int mapping, cudaStream_t stream) {
    switch (mapping) {
        case 0: {
            maxpool_v0(input, output, params, stream);
            break;
        }
        case 1: {
            const int C_groups = static_cast<int>((params.C + 3) / 4);
            const int64_t grid_z_64 = params.N * C_groups;
            if (grid_z_64 > 65535) {
                maxpool_v0(input, output, params, stream);
                break;
            }
            dim3 block(8, 8, 4);
            dim3 grid(
                static_cast<int>((params.OW + 7) / 8),
                static_cast<int>((params.OH + 7) / 8),
                static_cast<int>(grid_z_64));
            maxpool_v7_mappingB_kernel<half><<<grid, block, 0, stream>>>(input, output, params, C_groups);
            CUDA_CHECK(cudaGetLastError());
            break;
        }
        case 2: {
            const int C_groups = static_cast<int>((params.C + 255) / 256);
            const int64_t grid_z_64 = params.N * C_groups;
            if (params.OW > 65535 || params.OH > 65535 || grid_z_64 > 65535) {
                maxpool_v0(input, output, params, stream);
                break;
            }
            dim3 block(256, 1, 1);
            dim3 grid(
                static_cast<int>(params.OW),
                static_cast<int>(params.OH),
                static_cast<int>(grid_z_64));
            maxpool_v7_mappingC_kernel<half><<<grid, block, 0, stream>>>(input, output, params, C_groups);
            CUDA_CHECK(cudaGetLastError());
            break;
        }
        case 3: {
            if (params.C % 4 != 0) {
                maxpool_v0(input, output, params, stream);
                break;
            }
            const int C_groups = static_cast<int>((params.C + 31) / 32);
            const int64_t grid_z_64 = params.N * C_groups;
            if (grid_z_64 > 65535) {
                maxpool_v0(input, output, params, stream);
                break;
            }
            dim3 block(32, 8, 1);
            dim3 grid(
                static_cast<int>((params.OW + 3) / 4),
                static_cast<int>((params.OH + 3) / 4),
                static_cast<int>(grid_z_64));
            maxpool_v7_mappingD_kernel<half><<<grid, block, 0, stream>>>(input, output, params, C_groups);
            CUDA_CHECK(cudaGetLastError());
            break;
        }
        default:
            throw std::invalid_argument("unsupported mapping: " + std::to_string(mapping));
    }
}
