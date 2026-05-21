#pragma once
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <stdexcept>
#include <type_traits>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define CUDA_CHECK(call) \
    do { cudaError_t err = call; if (err != cudaSuccess) \
         throw std::runtime_error(cudaGetErrorString(err)); } while(0)

// NVTX profiling macros — no-ops when ENABLE_NVTX is not defined
#ifdef ENABLE_NVTX
#include <nvtx3/nvToolsExt.h>

// NVTX color constants (ARGB)
#define NVTX_COLOR_MAXPOOL  0xFF4169E1  // Blue
#define NVTX_COLOR_AVGPOOL  0xFF2E8B57  // Green

inline void _nvtx_range_push_color(const char* name, uint32_t color) {
    nvtxEventAttributes_t attr = {};
    attr.version = NVTX_VERSION;
    attr.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    attr.colorType = NVTX_COLOR_ARGB;
    attr.color = color;
    attr.messageType = NVTX_MESSAGE_TYPE_ASCII;
    attr.message.ascii = name;
    nvtxRangePushEx(&attr);
}

#define NVTX_RANGE_PUSH(name)        nvtxRangePushA(name)
#define NVTX_RANGE_POP()             nvtxRangePop()
#define NVTX_RANGE_PUSH_C(name, c)   _nvtx_range_push_color(name, c)
#else
#define NVTX_RANGE_PUSH(name)
#define NVTX_RANGE_POP()
#define NVTX_RANGE_PUSH_C(name, c)
#endif

struct PoolParams {
    int64_t N, H, W, C;
    int kh, kw;
    int sh, sw;
    int ph, pw;
    int dh, dw;
    bool ceil_mode;
    int64_t OH, OW;

    void compute_output_size() {
        if (ceil_mode) {
            // PyTorch caps ceil_mode output so that each window starts within
            // the input (not the padding zone): oh*sh - ph < H
            // => oh < (H + ph) / sh => OH_max = floor((H + ph - 1) / sh) + 1
            OH = (int64_t)ceil((double)(H + 2 * ph - dh * (kh - 1) - 1) / sh + 1);
            int64_t OH_cap = (H + ph - 1) / sh + 1;
            if (OH > OH_cap) OH = OH_cap;

            OW = (int64_t)ceil((double)(W + 2 * pw - dw * (kw - 1) - 1) / sw + 1);
            int64_t OW_cap = (W + pw - 1) / sw + 1;
            if (OW > OW_cap) OW = OW_cap;
        } else {
            OH = (int64_t)floor((double)(H + 2 * ph - dh * (kh - 1) - 1) / sh + 1);
            OW = (int64_t)floor((double)(W + 2 * pw - dw * (kw - 1) - 1) / sw + 1);
        }
    }
};

struct AvgPoolParams : PoolParams {
    bool count_include_pad;
    int64_t divisor_override; // 0 means not set
};

// MaxPool2d launchers (overloaded for float/half)
void maxpool_v0(const float* input, float* output, const PoolParams& params, cudaStream_t stream);
void maxpool_v0(const half* input, half* output, const PoolParams& params, cudaStream_t stream);
void maxpool_v1(const float* input, float* output, const PoolParams& params, cudaStream_t stream);
void maxpool_v1(const half* input, half* output, const PoolParams& params, cudaStream_t stream);
void maxpool_v2(const float* input, float* output, const PoolParams& params, cudaStream_t stream);
void maxpool_v2(const half* input, half* output, const PoolParams& params, cudaStream_t stream);
void maxpool_v3(const float* input, float* output, const PoolParams& params, cudaStream_t stream);
void maxpool_v3(const half* input, half* output, const PoolParams& params, cudaStream_t stream);
void maxpool_v4(const float* input, float* output, const PoolParams& params, cudaStream_t stream);
void maxpool_v4(const half* input, half* output, const PoolParams& params, cudaStream_t stream);
void maxpool_v5(const float* input, float* output, const PoolParams& params, cudaStream_t stream);
void maxpool_v5(const half* input, half* output, const PoolParams& params, cudaStream_t stream);
void maxpool_v6(const float* input, float* output, const PoolParams& params, cudaStream_t stream);
void maxpool_v6(const half* input, half* output, const PoolParams& params, cudaStream_t stream);
void maxpool_v7(const float* input, float* output, const PoolParams& params, int mapping, cudaStream_t stream);
void maxpool_v7(const half* input, half* output, const PoolParams& params, int mapping, cudaStream_t stream);
void maxpool_v8(const float* input, float* output, const PoolParams& params, cudaStream_t stream);
void maxpool_v8(const half* input, half* output, const PoolParams& params, cudaStream_t stream);
void maxpool_v9(const float* input, float* output, const PoolParams& params, cudaStream_t stream);
void maxpool_v9(const half* input, half* output, const PoolParams& params, cudaStream_t stream);
void maxpool_v10(const float* input, float* output, const PoolParams& params, cudaStream_t stream);
void maxpool_v10(const half* input, half* output, const PoolParams& params, cudaStream_t stream);

// AvgPool2d launchers (overloaded for float/half)
void avgpool_v0(const float* input, float* output, const AvgPoolParams& params, cudaStream_t stream);
void avgpool_v0(const half* input, half* output, const AvgPoolParams& params, cudaStream_t stream);
void avgpool_v1(const float* input, float* output, const AvgPoolParams& params, cudaStream_t stream);
void avgpool_v1(const half* input, half* output, const AvgPoolParams& params, cudaStream_t stream);
void avgpool_v2(const float* input, float* output, const AvgPoolParams& params, cudaStream_t stream);
void avgpool_v2(const half* input, half* output, const AvgPoolParams& params, cudaStream_t stream);
void avgpool_v3(const float* input, float* output, const AvgPoolParams& params, cudaStream_t stream);
void avgpool_v3(const half* input, half* output, const AvgPoolParams& params, cudaStream_t stream);
void avgpool_v4(const float* input, float* output, const AvgPoolParams& params, cudaStream_t stream);
void avgpool_v4(const half* input, half* output, const AvgPoolParams& params, cudaStream_t stream);
void avgpool_v5(const float* input, float* output, const AvgPoolParams& params, cudaStream_t stream);
void avgpool_v5(const half* input, half* output, const AvgPoolParams& params, cudaStream_t stream);
void avgpool_v6(const float* input, float* output, const AvgPoolParams& params, cudaStream_t stream);
void avgpool_v6(const half* input, half* output, const AvgPoolParams& params, cudaStream_t stream);
void avgpool_v7(const float* input, float* output, const AvgPoolParams& params, int mapping, cudaStream_t stream);
void avgpool_v7(const half* input, half* output, const AvgPoolParams& params, int mapping, cudaStream_t stream);
void avgpool_v8(const float* input, float* output, const AvgPoolParams& params, cudaStream_t stream);
void avgpool_v8(const half* input, half* output, const AvgPoolParams& params, cudaStream_t stream);
void avgpool_v9(const float* input, float* output, const AvgPoolParams& params, cudaStream_t stream);
void avgpool_v9(const half* input, half* output, const AvgPoolParams& params, cudaStream_t stream);
void avgpool_v10(const float* input, float* output, const AvgPoolParams& params, cudaStream_t stream);
void avgpool_v10(const half* input, half* output, const AvgPoolParams& params, cudaStream_t stream);
