// maxpool_v15_kernel template — shared between pooling_max.cu and pooling_avg.cu
// Both TUs need the full template definition to instantiate for all dtypes.
#pragma once
#include "pooling.cuh"

// V15 swizzled shared memory: use column padding to break 8-way bank conflicts.
// For fp32 (4-byte), elements at columns 0,8,16,... hit the same bank.
// By allocating smem_w+1 columns (padding), element (row, col) maps to
// row*(smem_w+1) + col, so consecutive columns are always in different banks.
template <typename T>
__device__ __forceinline__ int v15_smem_idx(int row, int col, int smem_w_padded) {
    return row * smem_w_padded + col;
}

template <typename T, bool IS_MAXPOOL, bool COUNT_INCLUDE_PAD>
__global__ void maxpool_v15_kernel(const T* __restrict__ input, T* __restrict__ output,
                                   const PoolParams params,
                                   int blocks_oh, int blocks_ow,
                                   int smem_h, int smem_w,
                                   int64_t divisor_override = 0)
{
    constexpr int TILE_OH = 8;
    constexpr int TILE_OW = 8;

    const int n = blockIdx.z;
    const int c = blockIdx.y;
    const int tile_idx = blockIdx.x;
    const int tile_oh = tile_idx / blocks_ow;
    const int tile_ow = tile_idx % blocks_ow;

    const int th = threadIdx.y;
    const int tw = threadIdx.x;

    const int smem_w_padded = smem_w + 1;

    extern __shared__ __align__(16) unsigned char smem_raw[];
    T* smem = reinterpret_cast<T*>(smem_raw);

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
        T val = IS_MAXPOOL ? static_cast<T>(-INFINITY) : static_cast<T>(0);
        if (ih >= 0 && ih < params.H && iw >= 0 && iw < params.W) {
            const int64_t in_idx = ((static_cast<int64_t>(n) * params.H + ih) * params.W + iw) * params.C + c;
            val = input[in_idx];
        }
        smem[v15_smem_idx<T>(sih, siw, smem_w_padded)] = val;
    }

    __syncthreads();

    if (oh >= params.OH || ow >= params.OW) return;

    T result_max = static_cast<T>(-INFINITY);
    float result_avg = 0.0f;
    int count = 0;
    for (int kh_i = 0; kh_i < params.kh; ++kh_i) {
        const int sih = th * params.sh + kh_i * params.dh;
        for (int kw_i = 0; kw_i < params.kw; ++kw_i) {
            const int siw = tw * params.sw + kw_i * params.dw;
            T val = smem[v15_smem_idx<T>(sih, siw, smem_w_padded)];
            if constexpr (IS_MAXPOOL) {
                if constexpr (std::is_same<T, half>::value) {
                    result_max = __hmax(result_max, val);
                } else if constexpr (std::is_same<T, nv_bfloat16>::value) {
                    result_max = __hmax(result_max, val);
                } else if constexpr (std::is_same<T, __nv_fp8_e4m3>::value) {
                    float r = static_cast<float>(result_max);
                    float v = static_cast<float>(val);
                    result_max = static_cast<__nv_fp8_e4m3>(r > v ? r : v);
                } else if constexpr (std::is_same<T, __nv_fp8_e5m2>::value) {
                    float r = static_cast<float>(result_max);
                    float v = static_cast<float>(val);
                    result_max = static_cast<__nv_fp8_e5m2>(r > v ? r : v);
                } else {
                    result_max = result_max > val ? result_max : val;
                }
            } else {
                result_avg += static_cast<float>(val);
                if constexpr (COUNT_INCLUDE_PAD) {
                    int ih_coord = tile_oh * TILE_OH * params.sh - params.ph + th * params.sh + kh_i * params.dh;
                    int iw_coord = tile_ow * TILE_OW * params.sw - params.pw + tw * params.sw + kw_i * params.dw;
                    if (ih_coord >= -params.ph && ih_coord < params.H + params.ph &&
                        iw_coord >= -params.pw && iw_coord < params.W + params.pw) {
                        count++;
                    }
                } else {
                    int ih_coord = tile_oh * TILE_OH * params.sh - params.ph + th * params.sh + kh_i * params.dh;
                    int iw_coord = tile_ow * TILE_OW * params.sw - params.pw + tw * params.sw + kw_i * params.dw;
                    if (ih_coord >= 0 && ih_coord < params.H && iw_coord >= 0 && iw_coord < params.W) {
                        count++;
                    }
                }
            }
        }
    }

    int64_t out_idx = ((static_cast<int64_t>(n) * params.OH + oh) * params.OW + ow) * params.C + c;
    if constexpr (IS_MAXPOOL) {
        output[out_idx] = result_max;
    } else {
        int64_t denom = (divisor_override > 0) ? divisor_override : count;
        output[out_idx] = static_cast<T>(result_avg / denom);
    }
}
