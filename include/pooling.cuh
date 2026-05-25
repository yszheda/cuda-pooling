#pragma once
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <stdexcept>
#include <type_traits>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

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
void maxpool_v11(const float* input, float* output, const PoolParams& params, cudaStream_t stream);
void maxpool_v11(const half* input, half* output, const PoolParams& params, cudaStream_t stream);
void maxpool_v12(const float* input, float* output, const PoolParams& params, cudaStream_t stream);
void maxpool_v12(const half* input, half* output, const PoolParams& params, cudaStream_t stream);
void maxpool_v13(const float* input, float* output, const PoolParams& params, cudaStream_t stream);
void maxpool_v13(const half* input, half* output, const PoolParams& params, cudaStream_t stream);

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
void avgpool_v11(const float* input, float* output, const AvgPoolParams& params, cudaStream_t stream);
void avgpool_v11(const half* input, half* output, const AvgPoolParams& params, cudaStream_t stream);
void avgpool_v12(const float* input, float* output, const AvgPoolParams& params, cudaStream_t stream);
void avgpool_v12(const half* input, half* output, const AvgPoolParams& params, cudaStream_t stream);
void avgpool_v13(const float* input, float* output, const AvgPoolParams& params, cudaStream_t stream);
void avgpool_v13(const half* input, half* output, const AvgPoolParams& params, cudaStream_t stream);

// v14: adaptive kernel dispatcher
void maxpool_v14(const float* input, float* output, const PoolParams& params, cudaStream_t stream);
void maxpool_v14(const half* input, half* output, const PoolParams& params, cudaStream_t stream);
void avgpool_v14(const float* input, float* output, const AvgPoolParams& params, cudaStream_t stream);
void avgpool_v14(const half* input, half* output, const AvgPoolParams& params, cudaStream_t stream);

// v15: swizzled shared memory
void maxpool_v15(const float* input, float* output, const PoolParams& params, cudaStream_t stream);
void maxpool_v15(const half* input, half* output, const PoolParams& params, cudaStream_t stream);
void avgpool_v15(const float* input, float* output, const AvgPoolParams& params, cudaStream_t stream);
void avgpool_v15(const half* input, half* output, const AvgPoolParams& params, cudaStream_t stream);

// ---------------------------------------------------------------------------
// dtype traits for pybind template instantiation
// ---------------------------------------------------------------------------
enum class PoolDType { F32, F16, BF16, FP8_E4M3, FP8_E5M2, I8, I16 };

template <PoolDType D> struct dtype_traits;

template <> struct dtype_traits<PoolDType::F32> {
    using ctype = float;
    static constexpr const char* name = "f32";
    static constexpr size_t itemsize = 4;
    static constexpr char np_kind = 'f';
    static bool validate_np_kind(char k) { return k == 'f'; }
};
template <> struct dtype_traits<PoolDType::F16> {
    using ctype = half;
    static constexpr const char* name = "f16";
    static constexpr size_t itemsize = 2;
    static constexpr char np_kind = 'f';
    static bool validate_np_kind(char k) { return k == 'f'; }
};
template <> struct dtype_traits<PoolDType::BF16> {
    using ctype = nv_bfloat16;
    static constexpr const char* name = "bf16";
    static constexpr size_t itemsize = 2;
    static constexpr char np_kind = 'u';  // numpy uint16 view
    static bool validate_np_kind(char k) { return k == 'u'; }
};
template <> struct dtype_traits<PoolDType::FP8_E4M3> {
    using ctype = __nv_fp8_e4m3;
    static constexpr const char* name = "fp8_e4m3";
    static constexpr size_t itemsize = 1;
    static constexpr char np_kind = 'u';  // numpy uint8 view
    static bool validate_np_kind(char k) { return k == 'u'; }
};
template <> struct dtype_traits<PoolDType::FP8_E5M2> {
    using ctype = __nv_fp8_e5m2;
    static constexpr const char* name = "fp8_e5m2";
    static constexpr size_t itemsize = 1;
    static constexpr char np_kind = 'u';
    static bool validate_np_kind(char k) { return k == 'u'; }
};
template <> struct dtype_traits<PoolDType::I8> {
    using ctype = int8_t;
    static constexpr const char* name = "i8";
    static constexpr size_t itemsize = 1;
    static constexpr char np_kind = 'i';
    static bool validate_np_kind(char k) { return k == 'i'; }
};
template <> struct dtype_traits<PoolDType::I16> {
    using ctype = int16_t;
    static constexpr const char* name = "i16";
    static constexpr size_t itemsize = 2;
    static constexpr char np_kind = 'i';
    static bool validate_np_kind(char k) { return k == 'i'; }
};
