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
