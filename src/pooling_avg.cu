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
        // When ceil_mode extends the window beyond the padded input,
        // those out-of-bounds pixels are NOT counted even with count_include_pad.
        // Only pixels within the actual padding range are counted as padded zeros.
        const bool ih_pad = (!ih_in && params.count_include_pad &&
                             ih >= -static_cast<int64_t>(params.ph) &&
                             ih < params.H + static_cast<int64_t>(params.ph));
        for (int kw_i = 0; kw_i < params.kw; ++kw_i) {
            const int64_t iw = ow * params.sw - params.pw + static_cast<int64_t>(kw_i) * params.dw;
            const bool iw_in = (iw >= 0 && iw < params.W);
            const bool iw_pad = (!iw_in && params.count_include_pad &&
                                 iw >= -static_cast<int64_t>(params.pw) &&
                                 iw < params.W + static_cast<int64_t>(params.pw));

            if (ih_in && iw_in) {
                const int64_t in_idx = ((n * params.H + ih) * params.W + iw) * params.C + c;
                sum += static_cast<float>(input[in_idx]);
                count++;
            } else if (ih_pad && iw_pad) {
                // Padded zero contributes to count but not to sum
                count++;
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
    const int64_t total = params.OH * params.OW * params.C;
    const int threads = 256;
    const int blocks_x = static_cast<int>((total + threads - 1) / threads);
    dim3 grid(blocks_x, 1, static_cast<int>(params.N));
    avgpool_v0_kernel<float><<<grid, threads, 0, stream>>>(input, output, params);
    CUDA_CHECK(cudaGetLastError());
}

void avgpool_v0(const half* input, half* output, const AvgPoolParams& params, cudaStream_t stream) {
    const int64_t total = params.OH * params.OW * params.C;
    const int threads = 256;
    const int blocks_x = static_cast<int>((total + threads - 1) / threads);
    dim3 grid(blocks_x, 1, static_cast<int>(params.N));
    avgpool_v0_kernel<half><<<grid, threads, 0, stream>>>(input, output, params);
    CUDA_CHECK(cudaGetLastError());
}
