#pragma once
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define CUDA_CHECK(call) \
    do { cudaError_t err = call; if (err != cudaSuccess) \
         throw std::runtime_error(cudaGetErrorString(err)); } while(0)

struct PoolParams {
    int64_t N, H, W, C;
    int kh, kw;
    int sh, sw;
    int ph, pw;
    int dh, dw;
    bool ceil_mode;
    int64_t OH, OW;

    void compute_output_size() {
        OH = ceil_mode
            ? (int64_t)ceil((double)(H + 2 * ph - dh * (kh - 1) - 1) / sh + 1)
            : (int64_t)floor((double)(H + 2 * ph - dh * (kh - 1) - 1) / sh + 1);
        OW = ceil_mode
            ? (int64_t)ceil((double)(W + 2 * pw - dw * (kw - 1) - 1) / sw + 1)
            : (int64_t)floor((double)(W + 2 * pw - dw * (kw - 1) - 1) / sw + 1);
    }
};

struct AvgPoolParams : PoolParams {
    bool count_include_pad;
    int64_t divisor_override; // 0 means not set
};

// MaxPool2d launchers (overloaded for float/half)
void maxpool_v0(const float* input, float* output, const PoolParams& params, cudaStream_t stream);
void maxpool_v0(const half* input, half* output, const PoolParams& params, cudaStream_t stream);

// AvgPool2d launchers (overloaded for float/half)
void avgpool_v0(const float* input, float* output, const AvgPoolParams& params, cudaStream_t stream);
void avgpool_v0(const half* input, half* output, const AvgPoolParams& params, cudaStream_t stream);
