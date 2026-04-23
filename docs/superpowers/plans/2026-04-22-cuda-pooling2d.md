# CUDA Pooling2D Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a high-performance CUDA Max/Avg Pooling2D implementation with NHWC layout, pybind11 Python bindings, step-by-step optimization from naive to warp-specialized kernels, comprehensive tests, and real-model benchmarks.

**Architecture:** CMake project with CUDA kernels (8 optimization stages per pool type), pybind11 Python bindings exposing versioned `maxpool2d`/`avgpool2d` functions that accept/return numpy NHWC arrays. Tests use PyTorch as golden reference. Benchmarks cover synthetic shapes and real CNN model configurations. Build/test/benchmark on remote GPU via SSH.

**Tech Stack:** CUDA 13.0, CMake, pybind11, Python 3.12, NumPy, PyTorch 2.9.1, pytest

---

## File Structure

| File | Responsibility |
|------|---------------|
| `CMakeLists.txt` | Build system: CUDA compilation, pybind11, Python detection |
| `include/pooling.cuh` | Kernel launcher declarations, output-size calculation, parameter structs |
| `src/pooling_max.cu` | All MaxPool2d kernel implementations (v0-v7) |
| `src/pooling_avg.cu` | All AvgPool2d kernel implementations (v0-v7) |
| `src/pybind_module.cpp` | pybind11 bindings: `maxpool2d()` and `avgpool2d()` Python functions |
| `tests/conftest.py` | Shared test fixtures: golden reference helpers, tolerance constants |
| `tests/test_maxpool.py` | MaxPool2d unit tests covering full parameter matrix |
| `tests/test_avgpool.py` | AvgPool2d unit tests covering full parameter matrix |
| `benchmarks/bench_pooling.py` | Performance benchmarks: synthetic + real model cases |

---

## Task 1: CMake Build System & Project Skeleton

**Files:**
- Create: `CMakeLists.txt`
- Create: `include/pooling.cuh`
- Create: `src/pybind_module.cpp`

- [ ] **Step 1: Create CMakeLists.txt**

```cmake
cmake_minimum_required(VERSION 3.24)
project(cuda_pooling LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 80;90;100;110)

find_package(Python3 REQUIRED COMPONENTS Interpreter Development NumPy)

include(FetchContent)
FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG v2.13.6
)
FetchContent_MakeAvailable(pybind11)

add_library(pooling_kernels STATIC
    src/pooling_max.cu
    src/pooling_avg.cu
)
target_include_directories(pooling_kernels PUBLIC include)
target_compile_options(pooling_kernels PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

pybind11_add_module(_pooling src/pybind_module.cpp)
target_link_libraries(_pooling PRIVATE pooling_kernels Python3::NumPy)
target_compile_options(_pooling PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
```

- [ ] **Step 2: Create minimal pooling.cuh with parameter structs and output-size calculation**

```cpp
#pragma once
#include <cstdint>
#include <cstdlib>
#include <cmath>

struct PoolParams {
    int64_t N, H, W, C;
    int kh, kw;
    int sh, sw;
    int ph, pw;
    int dh, dw;
    bool ceil_mode;

    // Computed output dimensions
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

// MaxPool2d launchers (one per version)
void maxpool_v0(const float* input, float* output, const PoolParams& params, cudaStream_t stream);
void maxpool_v0_half(const half* input, half* output, const PoolParams& params, cudaStream_t stream);

// AvgPool2d launchers (one per version)
void avgpool_v0(const float* input, float* output, const AvgPoolParams& params, cudaStream_t stream);
void avgpool_v0_half(const half* input, half* output, const AvgPoolParams& params, cudaStream_t stream);
```

- [ ] **Step 3: Create minimal pybind_module.cpp with placeholder bindings**

```cpp
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "pooling.cuh"

namespace py = pybind11;

template <typename T>
py::array_t<T> maxpool2d_impl(py::array_t<T, py::array::c_style> input,
                               py::object kernel_size, py::object stride,
                               py::object padding, py::object dilation,
                               bool ceil_mode, int version) {
    auto buf = input.request();
    if (buf.ndim != 4)
        throw std::runtime_error("Input must be 4D (N, H, W, C)");

    int64_t N = buf.shape[0], H = buf.shape[1], W = buf.shape[2], C = buf.shape[3];
    auto parse_pair = [](py::object obj, int default_val) -> std::pair<int,int> {
        if (obj.is_none()) return {default_val, default_val};
        try { int v = obj.cast<int>(); return {v, v}; }
        catch (...) {
            auto t = obj.cast<std::tuple<int,int>>();
            return {std::get<0>(t), std::get<1>(t)};
        }
    };
    auto [kh, kw] = parse_pair(kernel_size, 1);
    auto [sh, sw] = parse_pair(stride, kh); // default stride = kernel_size
    auto [ph, pw] = parse_pair(padding, 0);
    auto [dh, dw] = parse_pair(dilation, 1);

    PoolParams params{N, H, W, C, kh, kw, sh, sw, ph, pw, dh, dw, ceil_mode, 0, 0};
    params.compute_output_size();

    auto output = py::array_t<T>({N, params.OH, params.OW, C});
    auto out_buf = output.request();

    // Dispatch to version
    switch (version) {
        case 0: maxpool_v0(static_cast<const T*>(buf.ptr), static_cast<T*>(out_buf.ptr), params, 0); break;
        default: throw std::runtime_error("Unsupported version: " + std::to_string(version));
    }
    return output;
}

template <typename T>
py::array_t<T> avgpool2d_impl(py::array_t<T, py::array::c_style> input,
                               py::object kernel_size, py::object stride,
                               py::object padding, bool ceil_mode,
                               bool count_include_pad, py::object divisor_override,
                               int version) {
    auto buf = input.request();
    if (buf.ndim != 4)
        throw std::runtime_error("Input must be 4D (N, H, W, C)");

    int64_t N = buf.shape[0], H = buf.shape[1], W = buf.shape[2], C = buf.shape[3];
    auto parse_pair = [](py::object obj, int default_val) -> std::pair<int,int> {
        if (obj.is_none()) return {default_val, default_val};
        try { int v = obj.cast<int>(); return {v, v}; }
        catch (...) {
            auto t = obj.cast<std::tuple<int,int>>();
            return {std::get<0>(t), std::get<1>(t)};
        }
    };
    auto [kh, kw] = parse_pair(kernel_size, 1);
    auto [sh, sw] = parse_pair(stride, kh);
    auto [ph, pw] = parse_pair(padding, 0);

    int64_t div_override = 0;
    if (!divisor_override.is_none()) div_override = divisor_override.cast<int64_t>();

    AvgPoolParams params{{N, H, W, C, kh, kw, sh, sw, ph, pw, 1, 1, ceil_mode, 0, 0},
                         count_include_pad, div_override};
    params.compute_output_size();

    auto output = py::array_t<T>({N, params.OH, params.OW, C});
    auto out_buf = output.request();

    switch (version) {
        case 0: avgpool_v0(static_cast<const T*>(buf.ptr), static_cast<T*>(out_buf.ptr), params, 0); break;
        default: throw std::runtime_error("Unsupported version: " + std::to_string(version));
    }
    return output;
}

PYBIND11_MODULE(_pooling, m) {
    m.def("maxpool2d_f32", &maxpool2d_impl<float>,
          py::arg("input"), py::arg("kernel_size"), py::arg("stride") = py::none(),
          py::arg("padding") = 0, py::arg("dilation") = 1, py::arg("ceil_mode") = false,
          py::arg("version") = 0);
    m.def("maxpool2d_f16", &maxpool2d_impl<half>,
          py::arg("input"), py::arg("kernel_size"), py::arg("stride") = py::none(),
          py::arg("padding") = 0, py::arg("dilation") = 1, py::arg("ceil_mode") = false,
          py::arg("version") = 0);
    m.def("avgpool2d_f32", &avgpool2d_impl<float>,
          py::arg("input"), py::arg("kernel_size"), py::arg("stride") = py::none(),
          py::arg("padding") = 0, py::arg("ceil_mode") = true,
          py::arg("count_include_pad") = true, py::arg("divisor_override") = py::none(),
          py::arg("version") = 0);
    m.def("avgpool2d_f16", &avgpool2d_impl<half>,
          py::arg("input"), py::arg("kernel_size"), py::arg("stride") = py::none(),
          py::arg("padding") = 0, py::arg("ceil_mode") = true,
          py::arg("count_include_pad") = true, py::arg("divisor_override") = py::none(),
          py::arg("version") = 0);
}
```

- [ ] **Step 4: Create stub kernel files so CMake can compile**

Create `src/pooling_max.cu`:
```cpp
#include "pooling.cuh"

void maxpool_v0(const float* input, float* output, const PoolParams& params, cudaStream_t stream) {}
void maxpool_v0_half(const half* input, half* output, const PoolParams& params, cudaStream_t stream) {}
```

Create `src/pooling_avg.cu`:
```cpp
#include "pooling.cuh"

void avgpool_v0(const float* input, float* output, const AvgPoolParams& params, cudaStream_t stream) {}
void avgpool_v0_half(const half* input, half* output, const AvgPoolParams& params, cudaStream_t stream) {}
```

- [ ] **Step 5: Deploy to remote and verify build**

```bash
rsync -avz --exclude='.git' --exclude='build' ./ shuyua01@10.190.0.91:/home/shuyua01/Development/cuda-pooling/
ssh shuyua01@10.190.0.91 "cd /home/shuyua01/Development/cuda-pooling && mkdir -p build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release && make -j\$(nproc)"
```

Expected: Build succeeds, `_pooling.cpython-312-x86_64-linux-gnu.so` produced.

- [ ] **Step 6: Verify Python import on remote**

```bash
ssh shuyua01@10.190.0.91 "cd /home/shuyua01/Development/cuda-pooling/build && python3 -c 'import _pooling; print(dir(_pooling))'"
```

Expected: Prints `['avgpool2d_f16', 'avgpool2d_f32', 'maxpool2d_f16', 'maxpool2d_f32', ...]`.

- [ ] **Step 7: Commit**

```bash
git add CMakeLists.txt include/pooling.cuh src/pooling_max.cu src/pooling_avg.cu src/pybind_module.cpp
git commit -m "feat: CMake build system, parameter structs, pybind11 skeleton"
```

---

## Task 2: MaxPool2d v0 (Naive Kernel)

**Files:**
- Modify: `src/pooling_max.cu`
- Modify: `include/pooling.cuh` (add helper macros)

- [ ] **Step 1: Add CUDA helper macros and NHWC index function to pooling.cuh**

Add below the struct definitions in `pooling.cuh`:

```cpp
#include <cuda_runtime.h>

#define CUDA_CHECK(call) \
    do { cudaError_t err = call; if (err != cudaSuccess) \
         throw std::runtime_error(cudaGetErrorString(err)); } while(0)

__device__ __forceinline__
int64_t nhwc_index(int64_t n, int64_t h, int64_t w, int64_t c, int64_t H, int64_t W, int64_t C) {
    return ((n * H + h) * W + w) * C + c;
}
```

- [ ] **Step 2: Implement maxpool_v0 kernel and launchers in `src/pooling_max.cu`**

Replace the stub with:

```cpp
#include "pooling.cuh"
#include <cfloat>
#include <cuda_fp16.h>

template <typename T>
__global__ void maxpool_v0_kernel(const T* __restrict__ input, T* __restrict__ output,
                                   int64_t N, int64_t OH, int64_t OW, int64_t H, int64_t W, int64_t C,
                                   int kh, int kw, int sh, int sw, int ph, int pw, int dh, int dw) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = N * OH * OW * C;

    if (idx < total) {
        int64_t c = idx % C;
        int64_t rem = idx / C;
        int64_t ow = rem % OW;
        rem /= OW;
        int64_t oh = rem;
        int64_t n = blockIdx.z;

        T maxval = T(-INFINITY);

        for (int kh_idx = 0; kh_idx < kh; kh_idx++) {
            for (int kw_idx = 0; kw_idx < kw; kw_idx++) {
                int ih = oh * sh - ph + kh_idx * dh;
                int iw = ow * sw - pw + kw_idx * dw;
                if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                    T val = input[nhwc_index(n, ih, iw, c, H, W, C)];
                    if (val > maxval) maxval = val;
                }
            }
        }
    }
}
```

```cpp
#include "pooling.cuh"
#include <cfloat>
#include <cuda_fp16.h>

template <typename T>
__global__ void maxpool_v0_kernel(const T* __restrict__ input, T* __restrict__ output,
                                   int64_t OH, int64_t OW, int64_t H, int64_t W, int64_t C,
                                   int kh, int kw, int sh, int sw, int ph, int pw, int dh, int dw) {
    int64_t n = blockIdx.z;
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = OH * OW * C;

    if (idx < total) {
        int64_t c = idx % C;
        int64_t rem = idx / C;
        int64_t ow = rem % OW;
        int64_t oh = rem / OW;

        float maxval = -INFINITY;

        for (int kh_idx = 0; kh_idx < kh; kh_idx++) {
            for (int kw_idx = 0; kw_idx < kw; kw_idx++) {
                int ih = oh * sh - ph + kh_idx * dh;
                int iw = ow * sw - pw + kw_idx * dw;
                if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                    float val;
                    if constexpr (std::is_same_v<T, half>) {
                        val = __half2float(input[((n * H + ih) * W + iw) * C + c]);
                    } else {
                        val = input[((n * H + ih) * W + iw) * C + c];
                    }
                    if (val > maxval) maxval = val;
                }
            }
        }

        int64_t out_idx = ((n * OH + oh) * OW + ow) * C + c;
        if constexpr (std::is_same_v<T, half>) {
            output[out_idx] = __float2half(maxval);
        } else {
            output[out_idx] = maxval;
        }
    }
}

void maxpool_v0(const float* input, float* output, const PoolParams& params, cudaStream_t stream) {
    int64_t total = params.OH * params.OW * params.C;
    int block = 256;
    int grid_x = (total + block - 1) / block;
    dim3 grid(grid_x, 1, params.N);
    maxpool_v0_kernel<float><<<grid, block, 0, stream>>>(
        input, output, params.OH, params.OW, params.H, params.W, params.C,
        params.kh, params.kw, params.sh, params.sw, params.ph, params.pw, params.dh, params.dw);
    CUDA_CHECK(cudaGetLastError());
}

void maxpool_v0_half(const half* input, half* output, const PoolParams& params, cudaStream_t stream) {
    int64_t total = params.OH * params.OW * params.C;
    int block = 256;
    int grid_x = (total + block - 1) / block;
    dim3 grid(grid_x, 1, params.N);
    maxpool_v0_kernel<half><<<grid, block, 0, stream>>>(
        input, output, params.OH, params.OW, params.H, params.W, params.C,
        params.kh, params.kw, params.sh, params.sw, params.ph, params.pw, params.dh, params.dw);
    CUDA_CHECK(cudaGetLastError());
}
```

- [ ] **Step 3: Update pybind_module.cpp to use float/half dispatch correctly**

The pybind_module.cpp template already dispatches to `maxpool_v0<T>` for float and `maxpool_v0_half` for half. Update the float dispatch to call `maxpool_v0` and the half dispatch to call `maxpool_v0_half`. Since we use separate function names for half, update the `maxpool2d_impl` template specialization. Actually, since the template calls `maxpool_v0(static_cast<const T*>(buf.ptr), ...)`, we need to make it work for both. Change the approach — use overloaded function names:

In `include/pooling.cuh`, change declarations to use overloading:

```cpp
void maxpool_v0(const float* input, float* output, const PoolParams& params, cudaStream_t stream);
void maxpool_v0(const half* input, half* output, const PoolParams& params, cudaStream_t stream);
```

Remove `maxpool_v0_half`. The C++ compiler will resolve the overload based on pointer type. Update `src/pooling_max.cu` accordingly (rename `maxpool_v0_half` to `maxpool_v0` with `half*` params).

Similarly for avgpool:

```cpp
void avgpool_v0(const float* input, float* output, const AvgPoolParams& params, cudaStream_t stream);
void avgpool_v0(const half* input, half* output, const AvgPoolParams& params, cudaStream_t stream);
```

- [ ] **Step 4: Deploy, build, and do a quick smoke test on remote**

```bash
rsync -avz --exclude='.git' --exclude='build' ./ shuyua01@10.190.0.91:/home/shuyua01/Development/cuda-pooling/
ssh shuyua01@10.190.0.91 "cd /home/shuyua01/Development/cuda-pooling/build && cmake .. -DCMAKE_BUILD_TYPE=Release && make -j\$(nproc)"
ssh shuyua01@10.190.0.91 "cd /home/shuyua01/Development/cuda-pooling && python3 -c '
import sys; sys.path.insert(0, \"build\")
import _pooling, numpy as np
x = np.random.randn(1,8,8,3).astype(np.float32)
y = _pooling.maxpool2d_f32(x, 2, None, 0, 1, False, 0)
print(\"Input shape:\", x.shape, \"Output shape:\", y.shape)
'"
```

Expected: `Input shape: (1, 8, 8, 3) Output shape: (1, 4, 4, 3)`.

- [ ] **Step 5: Commit**

```bash
git add include/pooling.cuh src/pooling_max.cu src/pybind_module.cpp
git commit -m "feat: MaxPool2d v0 naive kernel with fp32/fp16 support"
```

---

## Task 3: AvgPool2d v0 (Naive Kernel)

**Files:**
- Modify: `src/pooling_avg.cu`
- Modify: `include/pooling.cuh` (ensure avgpool overload declarations)

- [ ] **Step 1: Implement avgpool_v0 kernel and launchers in `src/pooling_avg.cu`**

```cpp
#include "pooling.cuh"
#include <cfloat>
#include <cuda_fp16.h>

template <typename T>
__global__ void avgpool_v0_kernel(const T* __restrict__ input, T* __restrict__ output,
                                   int64_t OH, int64_t OW, int64_t H, int64_t W, int64_t C,
                                   int kh, int kw, int sh, int sw, int ph, int pw,
                                   bool count_include_pad, int64_t divisor_override) {
    int64_t n = blockIdx.z;
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = OH * OW * C;

    if (idx < total) {
        int64_t c = idx % C;
        int64_t rem = idx / C;
        int64_t ow = rem % OW;
        int64_t oh = rem / OW;

        float sum = 0.0f;
        int count = 0;

        for (int kh_idx = 0; kh_idx < kh; kh_idx++) {
            for (int kw_idx = 0; kw_idx < kw; kw_idx++) {
                int ih = oh * sh - ph + kh_idx;
                int iw = ow * sw - pw + kw_idx;
                bool in_bounds = (ih >= 0 && ih < H && iw >= 0 && iw < W);
                if (in_bounds) {
                    float val;
                    if constexpr (std::is_same_v<T, half>) {
                        val = __half2float(input[((n * H + ih) * W + iw) * C + c]);
                    } else {
                        val = input[((n * H + ih) * W + iw) * C + c];
                    }
                    sum += val;
                    count++;
                } else if (count_include_pad) {
                    // padded zero contributes to sum (already 0) and count
                    count++;
                }
            }
        }

        float avg;
        if (divisor_override > 0) {
            avg = sum / static_cast<float>(divisor_override);
        } else {
            avg = (count > 0) ? sum / static_cast<float>(count) : 0.0f;
        }

        int64_t out_idx = ((n * OH + oh) * OW + ow) * C + c;
        if constexpr (std::is_same_v<T, half>) {
            output[out_idx] = __float2half(avg);
        } else {
            output[out_idx] = avg;
        }
    }
}

void avgpool_v0(const float* input, float* output, const AvgPoolParams& params, cudaStream_t stream) {
    int64_t total = params.OH * params.OW * params.C;
    int block = 256;
    int grid_x = (total + block - 1) / block;
    dim3 grid(grid_x, 1, params.N);
    avgpool_v0_kernel<float><<<grid, block, 0, stream>>>(
        input, output, params.OH, params.OW, params.H, params.W, params.C,
        params.kh, params.kw, params.sh, params.sw, params.ph, params.pw,
        params.count_include_pad, params.divisor_override);
    CUDA_CHECK(cudaGetLastError());
}

void avgpool_v0(const half* input, half* output, const AvgPoolParams& params, cudaStream_t stream) {
    int64_t total = params.OH * params.OW * params.C;
    int block = 256;
    int grid_x = (total + block - 1) / block;
    dim3 grid(grid_x, 1, params.N);
    avgpool_v0_kernel<half><<<grid, block, 0, stream>>>(
        input, output, params.OH, params.OW, params.H, params.W, params.C,
        params.kh, params.kw, params.sh, params.sw, params.ph, params.pw,
        params.count_include_pad, params.divisor_override);
    CUDA_CHECK(cudaGetLastError());
}
```

- [ ] **Step 2: Deploy, build, and smoke test on remote**

```bash
rsync -avz --exclude='.git' --exclude='build' ./ shuyua01@10.190.0.91:/home/shuyua01/Development/cuda-pooling/
ssh shuyua01@10.190.0.91 "cd /home/shuyua01/Development/cuda-pooling/build && cmake .. -DCMAKE_BUILD_TYPE=Release && make -j\$(nproc)"
ssh shuyua01@10.190.0.91 "cd /home/shuyua01/Development/cuda-pooling && python3 -c '
import sys; sys.path.insert(0, \"build\")
import _pooling, numpy as np
x = np.random.randn(1,8,8,3).astype(np.float32)
y = _pooling.avgpool2d_f32(x, 2, None, 0, True, True, None, 0)
print(\"Input shape:\", x.shape, \"Output shape:\", y.shape)
'"
```

Expected: `Input shape: (1, 8, 8, 3) Output shape: (1, 4, 4, 3)`.

- [ ] **Step 3: Commit**

```bash
git add include/pooling.cuh src/pooling_avg.cu
git commit -m "feat: AvgPool2d v0 naive kernel with count_include_pad and divisor_override"
```

---

## Task 4: Test Infrastructure & MaxPool2d Tests

**Files:**
- Create: `tests/conftest.py`
- Create: `tests/test_maxpool.py`

- [ ] **Step 1: Create tests/conftest.py with shared test helpers**

```python
import numpy as np
import torch
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'build'))
import _pooling

TOLERANCES = {
    np.float32: 1e-5,
    np.float16: 1e-3,
}

def pytorch_maxpool2d(x_nhwc, kernel_size, stride, padding, dilation, ceil_mode):
    """Golden reference: NHWC numpy -> PyTorch NCHW maxpool2d -> NHWC numpy."""
    dtype = x_nhwc.dtype
    # NHWC -> NCHW
    x_nchw = np.ascontiguousarray(x_nhwc.transpose(0, 3, 1, 2))
    t = torch.from_numpy(x_nchw.copy())
    if dtype == np.float16:
        t = t.half()
    out = torch.nn.functional.max_pool2d(t, kernel_size, stride, padding, dilation, ceil_mode)
    out_np = out.numpy()
    # NCHW -> NHWC
    return np.ascontiguousarray(out_np.transpose(0, 2, 3, 1))

def pytorch_avgpool2d(x_nhwc, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override):
    """Golden reference: NHWC numpy -> PyTorch NCHW avgpool2d -> NHWC numpy."""
    dtype = x_nhwc.dtype
    x_nchw = np.ascontiguousarray(x_nhwc.transpose(0, 3, 1, 2))
    t = torch.from_numpy(x_nchw.copy())
    if dtype == np.float16:
        t = t.half()
    out = torch.nn.functional.avg_pool2d(t, kernel_size, stride, padding, ceil_mode,
                                          count_include_pad, divisor_override)
    out_np = out.numpy()
    return np.ascontiguousarray(out_np.transpose(0, 2, 3, 1))

def call_maxpool2d(x_nhwc, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False, version=0):
    """Call our CUDA maxpool2d."""
    dtype = x_nhwc.dtype
    fn = _pooling.maxpool2d_f16 if dtype == np.float16 else _pooling.maxpool2d_f32
    return fn(x_nhwc, kernel_size, stride, padding, dilation, ceil_mode, version)

def call_avgpool2d(x_nhwc, kernel_size, stride=None, padding=0, ceil_mode=True,
                   count_include_pad=True, divisor_override=None, version=0):
    """Call our CUDA avgpool2d."""
    dtype = x_nhwc.dtype
    fn = _pooling.avgpool2d_f16 if dtype == np.float16 else _pooling.avgpool2d_f32
    return fn(x_nhwc, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override, version)

def check_close(actual, expected, dtype):
    atol = TOLERANCES[dtype]
    np.testing.assert_allclose(actual, expected, atol=atol, rtol=0)
```

- [ ] **Step 2: Create tests/test_maxpool.py with parameterized tests**

```python
import numpy as np
import pytest
from conftest import call_maxpool2d, pytorch_maxpool2d, check_close

DTYPES = [np.float32, np.float16]

@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("kernel_size", [1, 2, 3, 5, 7, (3,2), (2,3)])
@pytest.mark.parametrize("stride,padding", [(None, 0), (1, 0), (2, 1), ((3,2), (1,2))])
@pytest.mark.parametrize("dilation", [1])
@pytest.mark.parametrize("ceil_mode", [False])
def test_maxpool_basic(dtype, kernel_size, stride, padding, dilation, ceil_mode):
    N, H, W, C = 2, 16, 16, 4
    x = np.random.randn(N, H, W, C).astype(dtype)
    expected = pytorch_maxpool2d(x, kernel_size, stride, padding, dilation, ceil_mode)
    actual = call_maxpool2d(x, kernel_size, stride, padding, dilation, ceil_mode, version=0)
    assert actual.shape == expected.shape, f"Shape mismatch: {actual.shape} vs {expected.shape}"
    check_close(actual, expected, dtype)

@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("dilation", [1, 2, 3])
def test_maxpool_dilation(dtype, dilation):
    x = np.random.randn(2, 16, 16, 4).astype(dtype)
    expected = pytorch_maxpool2d(x, 3, 2, 1, dilation, False)
    actual = call_maxpool2d(x, 3, 2, 1, dilation, False, version=0)
    assert actual.shape == expected.shape
    check_close(actual, expected, dtype)

@pytest.mark.parametrize("dtype", DTYPES)
def test_maxpool_ceil_mode(dtype):
    x = np.random.randn(2, 7, 7, 4).astype(dtype)
    expected = pytorch_maxpool2d(x, 3, 2, 1, 1, True)
    actual = call_maxpool2d(x, 3, 2, 1, 1, True, version=0)
    assert actual.shape == expected.shape
    check_close(actual, expected, dtype)

@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("H,W", [(8, 8), (32, 32), (64, 64)])
def test_maxpool_global(dtype, H, W):
    C = 4
    x = np.random.randn(2, H, W, C).astype(dtype)
    expected = pytorch_maxpool2d(x, (H, W), 1, 0, 1, False)
    actual = call_maxpool2d(x, (H, W), 1, 0, 1, False, version=0)
    assert actual.shape == (2, 1, 1, C)
    check_close(actual, expected, dtype)

@pytest.mark.parametrize("dtype", DTYPES)
def test_maxpool_kernel_equals_input(dtype):
    """kernel_size == input spatial size, no padding (global pooling)."""
    x = np.random.randn(2, 8, 8, 4).astype(dtype)
    expected = pytorch_maxpool2d(x, 8, 1, 0, 1, False)
    actual = call_maxpool2d(x, 8, 1, 0, 1, False, version=0)
    check_close(actual, expected, dtype)

@pytest.mark.parametrize("dtype", DTYPES)
def test_maxpool_large_padding(dtype):
    """Large padding causing output > input spatial size."""
    x = np.random.randn(1, 4, 4, 3).astype(dtype)
    expected = pytorch_maxpool2d(x, 3, 1, 2, 1, False)
    actual = call_maxpool2d(x, 3, 1, 2, 1, False, version=0)
    check_close(actual, expected, dtype)

@pytest.mark.parametrize("dtype", DTYPES)
def test_maxpool_nonsquare(dtype):
    x = np.random.randn(2, 16, 12, 4).astype(dtype)
    expected = pytorch_maxpool2d(x, (3, 2), (2, 1), (1, 0), 1, False)
    actual = call_maxpool2d(x, (3, 2), (2, 1), (1, 0), 1, False, version=0)
    check_close(actual, expected, dtype)

@pytest.mark.parametrize("dtype", DTYPES)
def test_maxpool_dilation_with_padding(dtype):
    x = np.random.randn(1, 16, 16, 4).astype(dtype)
    expected = pytorch_maxpool2d(x, 3, 2, 2, 2, False)
    actual = call_maxpool2d(x, 3, 2, 2, 2, False, version=0)
    check_close(actual, expected, dtype)

@pytest.mark.parametrize("dtype", DTYPES)
def test_maxpool_ceil_extra_row(dtype):
    """ceil_mode=True producing extra output row/column."""
    x = np.random.randn(1, 5, 5, 3).astype(dtype)
    expected = pytorch_maxpool2d(x, 3, 2, 0, 1, True)
    actual = call_maxpool2d(x, 3, 2, 0, 1, True, version=0)
    assert actual.shape == expected.shape
    check_close(actual, expected, dtype)
```

- [ ] **Step 3: Deploy and run maxpool tests on remote**

```bash
rsync -avz --exclude='.git' --exclude='build' ./ shuyua01@10.190.0.91:/home/shuyua01/Development/cuda-pooling/
ssh shuyua01@10.190.0.91 "cd /home/shuyua01/Development/cuda-pooling/build && cmake .. -DCMAKE_BUILD_TYPE=Release && make -j\$(nproc)"
ssh shuyua01@10.190.0.91 "cd /home/shuyua01/Development/cuda-pooling && PYTHONPATH=build:\$PYTHONPATH python3 -m pytest tests/test_maxpool.py -v"
```

Expected: All tests PASS.

- [ ] **Step 4: Commit**

```bash
git add tests/conftest.py tests/test_maxpool.py
git commit -m "feat: MaxPool2d test suite with PyTorch golden reference"
```

---

## Task 5: AvgPool2d Tests

**Files:**
- Create: `tests/test_avgpool.py`

- [ ] **Step 1: Create tests/test_avgpool.py**

```python
import numpy as np
import pytest
from conftest import call_avgpool2d, pytorch_avgpool2d, check_close

DTYPES = [np.float32, np.float16]

@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("kernel_size", [2, 3, 4, (3,2)])
@pytest.mark.parametrize("stride,padding", [(None, 0), (1, 0), (2, 0)])
@pytest.mark.parametrize("ceil_mode", [False, True])
@pytest.mark.parametrize("count_include_pad", [True, False])
def test_avgpool_basic(dtype, kernel_size, stride, padding, ceil_mode, count_include_pad):
    N, H, W, C = 2, 16, 16, 4
    x = np.random.randn(N, H, W, C).astype(dtype)
    expected = pytorch_avgpool2d(x, kernel_size, stride, padding, ceil_mode, count_include_pad, None)
    actual = call_avgpool2d(x, kernel_size, stride, padding, ceil_mode, count_include_pad, None, version=0)
    assert actual.shape == expected.shape, f"Shape mismatch: {actual.shape} vs {expected.shape}"
    check_close(actual, expected, dtype)

@pytest.mark.parametrize("dtype", DTYPES)
def test_avgpool_divisor_override(dtype):
    x = np.random.randn(2, 8, 8, 4).astype(dtype)
    expected = pytorch_avgpool2d(x, 3, 2, 1, False, True, 4)
    actual = call_avgpool2d(x, 3, 2, 1, False, True, 4, version=0)
    check_close(actual, expected, dtype)

@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("H,W", [(8, 8), (32, 32), (64, 64)])
def test_avgpool_global(dtype, H, W):
    C = 4
    x = np.random.randn(2, H, W, C).astype(dtype)
    expected = pytorch_avgpool2d(x, (H, W), 1, 0, False, True, None)
    actual = call_avgpool2d(x, (H, W), 1, 0, False, True, None, version=0)
    assert actual.shape == (2, 1, 1, C)
    check_close(actual, expected, dtype)

@pytest.mark.parametrize("dtype", DTYPES)
def test_avgpool_kernel_equals_input(dtype):
    x = np.random.randn(2, 8, 8, 4).astype(dtype)
    expected = pytorch_avgpool2d(x, 8, 1, 0, False, True, None)
    actual = call_avgpool2d(x, 8, 1, 0, False, True, None, version=0)
    check_close(actual, expected, dtype)

@pytest.mark.parametrize("dtype", DTYPES)
def test_avgpool_count_include_pad_false(dtype):
    """count_include_pad=False: only real elements count in denominator."""
    x = np.random.randn(1, 4, 4, 3).astype(dtype)
    expected = pytorch_avgpool2d(x, 3, 2, 1, False, False, None)
    actual = call_avgpool2d(x, 3, 2, 1, False, False, None, version=0)
    check_close(actual, expected, dtype)

@pytest.mark.parametrize("dtype", DTYPES)
def test_avgpool_large_padding(dtype):
    x = np.random.randn(1, 4, 4, 3).astype(dtype)
    expected = pytorch_avgpool2d(x, 3, 1, 2, False, True, None)
    actual = call_avgpool2d(x, 3, 1, 2, False, True, None, version=0)
    check_close(actual, expected, dtype)

@pytest.mark.parametrize("dtype", DTYPES)
def test_avgpool_nonsquare(dtype):
    x = np.random.randn(2, 16, 12, 4).astype(dtype)
    expected = pytorch_avgpool2d(x, (3, 2), (2, 1), (1, 0), False, True, None)
    actual = call_avgpool2d(x, (3, 2), (2, 1), (1, 0), False, True, None, version=0)
    check_close(actual, expected, dtype)

@pytest.mark.parametrize("dtype", DTYPES)
def test_avgpool_ceil_mode(dtype):
    x = np.random.randn(1, 7, 7, 4).astype(dtype)
    expected = pytorch_avgpool2d(x, 3, 2, 1, True, True, None)
    actual = call_avgpool2d(x, 3, 2, 1, True, True, None, version=0)
    assert actual.shape == expected.shape
    check_close(actual, expected, dtype)
```

- [ ] **Step 2: Deploy and run avgpool tests on remote**

```bash
rsync -avz --exclude='.git' --exclude='build' ./ shuyua01@10.190.0.91:/home/shuyua01/Development/cuda-pooling/
ssh shuyua01@10.190.0.91 "cd /home/shuyua01/Development/cuda-pooling/build && cmake .. -DCMAKE_BUILD_TYPE=Release && make -j\$(nproc)"
ssh shuyua01@10.190.0.91 "cd /home/shuyua01/Development/cuda-pooling && PYTHONPATH=build:\$PYTHONPATH python3 -m pytest tests/test_avgpool.py -v"
```

Expected: All tests PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/test_avgpool.py
git commit -m "feat: AvgPool2d test suite with count_include_pad and divisor_override"
```

---

## Task 6: MaxPool2d v1 — Shared Memory Tiling

**Files:**
- Modify: `src/pooling_max.cu`
- Modify: `include/pooling.cuh`
- Modify: `src/pybind_module.cpp`

- [ ] **Step 1: Add maxpool_v1 declarations to `include/pooling.cuh`**

```cpp
void maxpool_v1(const float* input, float* output, const PoolParams& params, cudaStream_t stream);
void maxpool_v1(const half* input, half* output, const PoolParams& params, cudaStream_t stream);
```

- [ ] **Step 2: Implement maxpool_v1 kernel in `src/pooling_max.cu`**

The shared memory tiling strategy: each block loads a tile of input data (output tile + halo region) into shared memory, then each thread reads from shared memory to compute its max.

```cpp
template <typename T, int TILE_OH, int TILE_OW>
__global__ void maxpool_v1_kernel(const T* __restrict__ input, T* __restrict__ output,
                                   int64_t OH, int64_t OW, int64_t H, int64_t W, int64_t C,
                                   int kh, int kw, int sh, int sw, int ph, int pw, int dh, int dw) {
    // Each block handles TILE_OH x TILE_OW output positions for one (n, c)
    // Shared memory holds the input tile that covers these output positions
    extern __shared__ char smem[];
    float* sdata = reinterpret_cast<float*>(smem);

    int64_t n = blockIdx.z;
    int64_t c = blockIdx.y;  // one channel per block.y
    int64_t tile_oh = blockIdx.x / ((OW + TILE_OW - 1) / TILE_OW);
    int64_t tile_ow = blockIdx.x % ((OW + TILE_OW - 1) / TILE_OW);

    int64_t oh_start = tile_oh * TILE_OH;
    int64_t ow_start = tile_ow * TILE_OW;

    // Input region: compute the input tile that covers this output tile
    int ih_start = oh_start * sh - ph;
    int iw_start = ow_start * sw - pw;
    int ih_end = min((int)((oh_start + TILE_OH - 1) * sh - ph + (kh - 1) * dh + 1), (int)H);
    int iw_end = min((int)((ow_start + TILE_OW - 1) * sw - pw + (kw - 1) * dw + 1), (int)W);
    int ih_lo = max(ih_start, 0);
    int iw_lo = max(iw_start, 0);

    int tile_h = ih_end - ih_start;
    int tile_w = iw_end - iw_start;
    int smem_stride = tile_w;

    // Cooperatively load input tile into shared memory
    for (int i = threadIdx.x; i < tile_h * tile_w; i += blockDim.x) {
        int ti = i / tile_w;
        int tj = i % tile_w;
        int ih = ih_start + ti;
        int iw = iw_start + tj;
        float val = -INFINITY;
        if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
            if constexpr (std::is_same_v<T, half>) {
                val = __half2float(input[((n * H + ih) * W + iw) * C + c]);
            } else {
                val = input[((n * H + ih) * W + iw) * C + c];
            }
        }
        sdata[ti * smem_stride + tj] = val;
    }
    __syncthreads();

    // Each thread computes one output position
    int tid = threadIdx.x;
    if (tid < TILE_OH * TILE_OW) {
        int local_oh = tid / TILE_OW;
        int local_ow = tid % TILE_OW;
        int64_t oh = oh_start + local_oh;
        int64_t ow = ow_start + local_ow;

        if (oh < OH && ow < OW) {
            float maxval = -INFINITY;
            for (int kh_idx = 0; kh_idx < kh; kh_idx++) {
                for (int kw_idx = 0; kw_idx < kw; kw_idx++) {
                    int si = (oh * sh - ph - ih_start) + kh_idx * dh;
                    int sj = (ow * sw - pw - iw_start) + kw_idx * dw;
                    if (si >= 0 && si < tile_h && sj >= 0 && sj < tile_w) {
                        float val = sdata[si * smem_stride + sj];
                        if (val > maxval) maxval = val;
                    }
                }
            }
            int64_t out_idx = ((n * OH + oh) * OW + ow) * C + c;
            if constexpr (std::is_same_v<T, half>) {
                output[out_idx] = __float2half(maxval);
            } else {
                output[out_idx] = maxval;
            }
        }
    }
}
```

Note: this v1 kernel iterates channel-by-channel (c = blockIdx.y). The launch configuration changes accordingly.

- [ ] **Step 3: Add maxpool_v1 launcher functions**

```cpp
void maxpool_v1(const float* input, float* output, const PoolParams& params, cudaStream_t stream) {
    constexpr int TILE_OH = 8, TILE_OW = 8;
    int blocks_ow = (params.OW + TILE_OW - 1) / TILE_OW;
    int blocks_oh = (params.OH + TILE_OH - 1) / TILE_OH;
    int smem_size = ((TILE_OH - 1) * params.sh + (params.kh - 1) * params.dh + 1) *
                    ((TILE_OW - 1) * params.sw + (params.kw - 1) * params.dw + 1) * sizeof(float);
    dim3 grid(blocks_oh * blocks_ow, params.C, params.N);
    maxpool_v1_kernel<float, TILE_OH, TILE_OW><<<grid, 256, smem_size, stream>>>(
        input, output, params.OH, params.OW, params.H, params.W, params.C,
        params.kh, params.kw, params.sh, params.sw, params.ph, params.pw, params.dh, params.dw);
    CUDA_CHECK(cudaGetLastError());
}

void maxpool_v1(const half* input, half* output, const PoolParams& params, cudaStream_t stream) {
    constexpr int TILE_OH = 8, TILE_OW = 8;
    int blocks_ow = (params.OW + TILE_OW - 1) / TILE_OW;
    int blocks_oh = (params.OH + TILE_OH - 1) / TILE_OH;
    int smem_size = ((TILE_OH - 1) * params.sh + (params.kh - 1) * params.dh + 1) *
                    ((TILE_OW - 1) * params.sw + (params.kw - 1) * params.dw + 1) * sizeof(float);
    dim3 grid(blocks_oh * blocks_ow, params.C, params.N);
    maxpool_v1_kernel<half, TILE_OH, TILE_OW><<<grid, 256, smem_size, stream>>>(
        input, output, params.OH, params.OW, params.H, params.W, params.C,
        params.kh, params.kw, params.sh, params.sw, params.ph, params.pw, params.dh, params.dw);
    CUDA_CHECK(cudaGetLastError());
}
```

- [ ] **Step 4: Update pybind_module.cpp to handle version=1 dispatch**

In the `maxpool2d_impl` switch statement, add:
```cpp
case 1: maxpool_v1(static_cast<const T*>(buf.ptr), static_cast<T*>(out_buf.ptr), params, 0); break;
```

- [ ] **Step 5: Update test conftest.py to test all versions**

In `tests/conftest.py`, add a `VERSIONS` constant and update tests to iterate over it. For now, add:

```python
MAXPOOL_VERSIONS = [0, 1]
AVGPOOL_VERSIONS = [0]
```

Update `test_maxpool.py` to parametrize over versions:
```python
from conftest import MAXPOOL_VERSIONS

@pytest.mark.parametrize("version", MAXPOOL_VERSIONS)
# ... add version param to all test functions
```

- [ ] **Step 6: Deploy, build, and run all tests on remote**

```bash
rsync -avz --exclude='.git' --exclude='build' ./ shuyua01@10.190.0.91:/home/shuyua01/Development/cuda-pooling/
ssh shuyua01@10.190.0.91 "cd /home/shuyua01/Development/cuda-pooling/build && cmake .. -DCMAKE_BUILD_TYPE=Release && make -j\$(nproc)"
ssh shuyua01@10.190.0.91 "cd /home/shuyua01/Development/cuda-pooling && PYTHONPATH=build:\$PYTHONPATH python3 -m pytest tests/ -v"
```

Expected: All tests PASS for both v0 and v1.

- [ ] **Step 7: Commit**

```bash
git add include/pooling.cuh src/pooling_max.cu src/pybind_module.cpp tests/conftest.py tests/test_maxpool.py
git commit -m "feat: MaxPool2d v1 shared memory tiling kernel"
```

---

## Task 7: AvgPool2d v1 — Shared Memory Tiling

**Files:**
- Modify: `src/pooling_avg.cu`
- Modify: `include/pooling.cuh`
- Modify: `src/pybind_module.cpp`
- Modify: `tests/conftest.py`
- Modify: `tests/test_avgpool.py`

- [ ] **Step 1: Add avgpool_v1 declarations to `include/pooling.cuh`**

```cpp
void avgpool_v1(const float* input, float* output, const AvgPoolParams& params, cudaStream_t stream);
void avgpool_v1(const half* input, half* output, const AvgPoolParams& params, cudaStream_t stream);
```

- [ ] **Step 2: Implement avgpool_v1 kernel in `src/pooling_avg.cu`**

Same tiling strategy as maxpool_v1, but accumulates sum and counts valid elements for the average.

```cpp
template <typename T, int TILE_OH, int TILE_OW>
__global__ void avgpool_v1_kernel(const T* __restrict__ input, T* __restrict__ output,
                                   int64_t OH, int64_t OW, int64_t H, int64_t W, int64_t C,
                                   int kh, int kw, int sh, int sw, int ph, int pw,
                                   bool count_include_pad, int64_t divisor_override) {
    extern __shared__ char smem[];
    float* sdata = reinterpret_cast<float*>(smem);
    // Also need a validity mask in shared memory
    bool* valid = reinterpret_cast<bool*>(sdata + /* tile_h * tile_w */);

    int64_t n = blockIdx.z;
    int64_t c = blockIdx.y;
    int64_t tile_oh = blockIdx.x / ((OW + TILE_OW - 1) / TILE_OW);
    int64_t tile_ow = blockIdx.x % ((OW + TILE_OW - 1) / TILE_OW);

    int64_t oh_start = tile_oh * TILE_OH;
    int64_t ow_start = tile_ow * TILE_OW;

    int ih_start = oh_start * sh - ph;
    int iw_start = ow_start * sw - pw;
    int ih_end = min((int)((oh_start + TILE_OH - 1) * sh - ph + kh - 1 + 1), (int)H);
    int iw_end = min((int)((ow_start + TILE_OW - 1) * sw - pw + kw - 1 + 1), (int)W);

    int tile_h = ih_end - ih_start;
    int tile_w = iw_end - iw_start;
    int smem_stride = tile_w;

    // Load input tile and validity mask
    for (int i = threadIdx.x; i < tile_h * tile_w; i += blockDim.x) {
        int ti = i / tile_w;
        int tj = i % tile_w;
        int ih = ih_start + ti;
        int iw = iw_start + tj;
        bool is_valid = (ih >= 0 && ih < H && iw >= 0 && iw < W);
        float val = 0.0f;
        if (is_valid) {
            if constexpr (std::is_same_v<T, half>) {
                val = __half2float(input[((n * H + ih) * W + iw) * C + c]);
            } else {
                val = input[((n * H + ih) * W + iw) * C + c];
            }
        }
        sdata[ti * smem_stride + tj] = val;
        valid[ti * smem_stride + tj] = is_valid;
    }
    __syncthreads();

    int tid = threadIdx.x;
    if (tid < TILE_OH * TILE_OW) {
        int local_oh = tid / TILE_OW;
        int local_ow = tid % TILE_OW;
        int64_t oh = oh_start + local_oh;
        int64_t ow = ow_start + local_ow;

        if (oh < OH && ow < OW) {
            float sum = 0.0f;
            int count = 0;
            for (int kh_idx = 0; kh_idx < kh; kh_idx++) {
                for (int kw_idx = 0; kw_idx < kw; kw_idx++) {
                    int si = (oh * sh - ph - ih_start) + kh_idx;
                    int sj = (ow * sw - pw - iw_start) + kw_idx;
                    if (si >= 0 && si < tile_h && sj >= 0 && sj < tile_w) {
                        sum += sdata[si * smem_stride + sj];
                        if (valid[si * smem_stride + sj] || count_include_pad) {
                            count++;
                        }
                    }
                }
            }
            float avg;
            if (divisor_override > 0) {
                avg = sum / static_cast<float>(divisor_override);
            } else {
                avg = (count > 0) ? sum / static_cast<float>(count) : 0.0f;
            }
            int64_t out_idx = ((n * OH + oh) * OW + ow) * C + c;
            if constexpr (std::is_same_v<T, half>) {
                output[out_idx] = __float2half(avg);
            } else {
                output[out_idx] = avg;
            }
        }
    }
}
```

Note: The shared memory allocation must include both `float[tile_h * tile_w]` and `bool[tile_h * tile_w]`. The launcher must compute the total smem size accordingly.

- [ ] **Step 3: Add avgpool_v1 launcher functions (similar to maxpool_v1 launchers)**

- [ ] **Step 4: Update pybind_module.cpp avgpool version dispatch, add `case 1`**

- [ ] **Step 5: Update `tests/conftest.py` AVGPOOL_VERSIONS to `[0, 1]`, add version param to test_avgpool.py**

- [ ] **Step 6: Deploy, build, run all tests on remote**

Expected: All tests PASS for both v0 and v1.

- [ ] **Step 7: Commit**

```bash
git add include/pooling.cuh src/pooling_avg.cu src/pybind_module.cpp tests/
git commit -m "feat: AvgPool2d v1 shared memory tiling kernel"
```

---

## Task 8: MaxPool2d v2 — Vectorized Loads

**Files:**
- Modify: `src/pooling_max.cu`
- Modify: `include/pooling.cuh`
- Modify: `src/pybind_module.cpp`

- [ ] **Step 1: Add maxpool_v2 declarations to `include/pooling.cuh`**

```cpp
void maxpool_v2(const float* input, float* output, const PoolParams& params, cudaStream_t stream);
void maxpool_v2(const half* input, half* output, const PoolParams& params, cudaStream_t stream);
```

- [ ] **Step 2: Implement maxpool_v2 kernel using float4/half2 vectorized loads**

Key idea: Load 4 channels at once via `float4` (or 2 via `half2`) to coalesce global memory access. Each thread still computes one output element but for a group of C channels simultaneously. Requires C to be a multiple of the vector width; fall back to v0 for non-aligned C.

```cpp
template <typename T, int VEC>
__global__ void maxpool_v2_kernel(const T* __restrict__ input, T* __restrict__ output,
                                   int64_t OH, int64_t OW, int64_t H, int64_t W, int64_t C,
                                   int kh, int kw, int sh, int sw, int ph, int pw, int dh, int dw) {
    int64_t n = blockIdx.z;
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = OH * OW * (C / VEC);

    if (idx < total) {
        int64_t c_vec = idx % (C / VEC);
        int64_t rem = idx / (C / VEC);
        int64_t ow = rem % OW;
        int64_t oh = rem / OW;
        int64_t c = c_vec * VEC;

        // Vectorized max values
        float maxvals[VEC];
        for (int v = 0; v < VEC; v++) maxvals[v] = -INFINITY;

        for (int kh_idx = 0; kh_idx < kh; kh_idx++) {
            for (int kw_idx = 0; kw_idx < kw; kw_idx++) {
                int ih = oh * sh - ph + kh_idx * dh;
                int iw = ow * sw - pw + kw_idx * dw;
                if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                    if constexpr (VEC == 4 && std::is_same_v<T, float>) {
                        float4 vec = *reinterpret_cast<const float4*>(&input[((n * H + ih) * W + iw) * C + c]);
                        for (int v = 0; v < 4; v++) {
                            if (vec.x > maxvals[v]) maxvals[v] = (&vec.x)[v];
                        }
                    } else if constexpr (VEC == 2 && std::is_same_v<T, half>) {
                        half2 vec = *reinterpret_cast<const half2*>(&input[((n * H + ih) * W + iw) * C + c]);
                        float f0 = __low2float(vec);
                        float f1 = __high2float(vec);
                        if (f0 > maxvals[0]) maxvals[0] = f0;
                        if (f1 > maxvals[1]) maxvals[1] = f1;
                    }
                }
            }
        }

        int64_t out_base = ((n * OH + oh) * OW + ow) * C + c;
        for (int v = 0; v < VEC; v++) {
            if constexpr (std::is_same_v<T, half>) {
                output[out_base + v] = __float2half(maxvals[v]);
            } else {
                output[out_base + v] = maxvals[v];
            }
        }
    }
}
```

- [ ] **Step 3: Add maxpool_v2 launchers. For fp32 use VEC=4, for fp16 use VEC=2. Fall back to v0 if C is not aligned.**

- [ ] **Step 4: Update pybind_module.cpp version dispatch, add `case 2`**

- [ ] **Step 5: Update MAXPOOL_VERSIONS to `[0, 1, 2]` in conftest.py**

- [ ] **Step 6: Deploy, build, run all tests on remote**

- [ ] **Step 7: Commit**

```bash
git commit -m "feat: MaxPool2d v2 vectorized loads (float4/half2)"
```

---

## Task 9: AvgPool2d v2 — Vectorized Loads

**Files:**
- Modify: `src/pooling_avg.cu`
- Modify: `include/pooling.cuh`
- Modify: `src/pybind_module.cpp`
- Modify: `tests/conftest.py`

- [ ] **Step 1: Add avgpool_v2 declarations, implement kernel using float4/half2, add launchers, update dispatch and test versions**

Follows same vectorized pattern as maxpool_v2 but accumulates sum and divides by count.

- [ ] **Step 2: Deploy, build, run all tests on remote**

- [ ] **Step 3: Commit**

```bash
git commit -m "feat: AvgPool2d v2 vectorized loads (float4/half2)"
```

---

## Task 10: MaxPool2d v3 — Register Blocking

**Files:**
- Modify: `src/pooling_max.cu`
- Modify: `include/pooling.cuh`
- Modify: `src/pybind_module.cpp`

- [ ] **Step 1: Add maxpool_v3 declarations to `include/pooling.cuh`**

```cpp
void maxpool_v3(const float* input, float* output, const PoolParams& params, cudaStream_t stream);
void maxpool_v3(const half* input, half* output, const PoolParams& params, cudaStream_t stream);
```

- [ ] **Step 2: Implement maxpool_v3 kernel with register blocking**

Key idea: Each thread computes multiple adjacent output spatial positions (e.g., 2x2 or 4x1) to reuse input data loaded into registers. When stride < kernel_size, adjacent output windows share input elements.

```cpp
template <typename T, int BLOCK_OH, int BLOCK_OW>
__global__ void maxpool_v3_kernel(const T* __restrict__ input, T* __restrict__ output,
                                   int64_t OH, int64_t OW, int64_t H, int64_t W, int64_t C,
                                   int kh, int kw, int sh, int sw, int ph, int pw, int dh, int dw) {
    int64_t n = blockIdx.z;
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = (OH / BLOCK_OH) * (OW / BLOCK_OW) * C;

    if (idx < total) {
        int64_t c = idx % C;
        int64_t rem = idx / C;
        int64_t ow_base = (rem % (OW / BLOCK_OW)) * BLOCK_OW;
        int64_t oh_base = (rem / (OW / BLOCK_OW)) * BLOCK_OH;

        // Compute the input rows needed
        int ih_start = oh_base * sh - ph;
        int ih_end = (oh_base + BLOCK_OH - 1) * sh + (kh - 1) * dh - ph;

        // Load input into registers for all rows in the window
        for (int bh = 0; bh < BLOCK_OH; bh++) {
            for (int bw = 0; bw < BLOCK_OW; bw++) {
                int64_t oh = oh_base + bh;
                int64_t ow = ow_base + bw;
                if (oh < OH && ow < OW) {
                    float maxval = -INFINITY;
                    for (int kh_idx = 0; kh_idx < kh; kh_idx++) {
                        for (int kw_idx = 0; kw_idx < kw; kw_idx++) {
                            int ih = oh * sh - ph + kh_idx * dh;
                            int iw = ow * sw - pw + kw_idx * dw;
                            if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                                float val;
                                if constexpr (std::is_same_v<T, half>) {
                                    val = __half2float(input[((n * H + ih) * W + iw) * C + c]);
                                } else {
                                    val = input[((n * H + ih) * W + iw) * C + c];
                                }
                                if (val > maxval) maxval = val;
                            }
                        }
                    }
                    int64_t out_idx = ((n * OH + oh) * OW + ow) * C + c;
                    if constexpr (std::is_same_v<T, half>) {
                        output[out_idx] = __float2half(maxval);
                    } else {
                        output[out_idx] = maxval;
                    }
                }
            }
        }
    }
}
```

Note: The register blocking benefit comes from the compiler keeping shared input rows in registers across the inner loop iterations. A more sophisticated version would explicitly pre-load shared rows. This implementation shows the structure; further tuning can be done based on profiler feedback.

- [ ] **Step 3: Add launchers, update dispatch, update test versions, deploy/build/test**

- [ ] **Step 4: Commit**

```bash
git commit -m "feat: MaxPool2d v3 register blocking"
```

---

## Task 11: AvgPool2d v3 — Register Blocking

Same pattern as Task 10 but for AvgPool2d.

- [ ] **Step 1: Add avgpool_v3 declarations, kernel, launchers, dispatch, tests, deploy/build/test**

- [ ] **Step 2: Commit**

```bash
git commit -m "feat: AvgPool2d v3 register blocking"
```

---

## Task 12: MaxPool2d v4 — Warp-Level Reduce

**Files:**
- Modify: `src/pooling_max.cu`
- Modify: `include/pooling.cuh`
- Modify: `src/pybind_module.cpp`

- [ ] **Step 1: Add maxpool_v4 declarations to `include/pooling.cuh`**

```cpp
void maxpool_v4(const float* input, float* output, const PoolParams& params, cudaStream_t stream);
void maxpool_v4(const half* input, half* output, const PoolParams& params, cudaStream_t stream);
```

- [ ] **Step 2: Implement maxpool_v4 kernel using warp shuffle max reduction**

Key idea: For small kernel sizes (e.g., 3x3 = 9 elements), distribute the 9 input elements across a warp and use `__shfl_down_sync` to compute the max in O(log2) steps instead of a per-thread serial loop.

```cpp
template <typename T>
__global__ void maxpool_v4_kernel(const T* __restrict__ input, T* __restrict__ output,
                                   int64_t OH, int64_t OW, int64_t H, int64_t W, int64_t C,
                                   int kh, int kw, int sh, int sw, int ph, int pw, int dh, int dw) {
    int64_t n = blockIdx.z;
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = OH * OW * C;

    if (idx < total) {
        int64_t c = idx % C;
        int64_t rem = idx / C;
        int64_t ow = rem % OW;
        int64_t oh = rem / OW;

        int lane = threadIdx.x % 32;
        int karea = kh * kw;

        // Each lane handles a subset of the kernel window elements
        float myval = -INFINITY;
        for (int ki = lane; ki < karea; ki += 32) {
            int kh_idx = ki / kw;
            int kw_idx = ki % kw;
            int ih = oh * sh - ph + kh_idx * dh;
            int iw = ow * sw - pw + kw_idx * dw;
            if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                if constexpr (std::is_same_v<T, half>) {
                    float val = __half2float(input[((n * H + ih) * W + iw) * C + c]);
                    if (val > myval) myval = val;
                } else {
                    float val = input[((n * H + ih) * W + iw) * C + c];
                    if (val > myval) myval = val;
                }
            }
        }

        // Warp-level max reduction
        for (int offset = 16; offset > 0; offset >>= 1) {
            float other = __shfl_down_sync(0xFFFFFFFF, myval, offset);
            if (other > myval) myval = other;
        }

        // Lane 0 writes the result
        if (lane == 0) {
            int64_t out_idx = ((n * OH + oh) * OW + ow) * C + c;
            if constexpr (std::is_same_v<T, half>) {
                output[out_idx] = __float2half(myval);
            } else {
                output[out_idx] = myval;
            }
        }
    }
}
```

Note: This approach is most beneficial when `karea < 32` (small kernels). For large kernels (karea >= 32), each lane handles fewer or exactly 1 element, and the warp reduction still helps.

- [ ] **Step 3: Add launchers, update dispatch, update test versions, deploy/build/test**

- [ ] **Step 4: Commit**

```bash
git commit -m "feat: MaxPool2d v4 warp-level reduce"
```

---

## Task 13: AvgPool2d v4 — Warp-Level Reduce

Same pattern but uses warp shuffle sum reduction instead of max.

- [ ] **Step 1: Add avgpool_v4 with warp shuffle sum reduce. Also needs warp-level count reduction for count_include_pad=False case.**

The sum reduction:
```cpp
for (int offset = 16; offset > 0; offset >>= 1) {
    mysum += __shfl_down_sync(0xFFFFFFFF, mysum, offset);
    mycount += __shfl_down_sync(0xFFFFFFFF, mycount, offset);
}
```

- [ ] **Step 2: Deploy, build, test, commit**

```bash
git commit -m "feat: AvgPool2d v4 warp-level reduce"
```

---

## Task 14: MaxPool2d v5 — Double Buffer / Pipeline

**Files:**
- Modify: `src/pooling_max.cu`
- Modify: `include/pooling.cuh`
- Modify: `src/pybind_module.cpp`

- [ ] **Step 1: Add maxpool_v5 declarations to `include/pooling.cuh`**

```cpp
void maxpool_v5(const float* input, float* output, const PoolParams& params, cudaStream_t stream);
void maxpool_v5(const half* input, half* output, const PoolParams& params, cudaStream_t stream);
```

- [ ] **Step 2: Implement maxpool_v5 kernel with double buffering**

Key idea: Use two shared memory buffers. While threads compute the max from buffer A (current tile), a subset of threads asynchronously loads the next tile into buffer B. Swap buffers after each tile.

```cpp
template <typename T, int TILE_OH, int TILE_OW>
__global__ void maxpool_v5_kernel(const T* __restrict__ input, T* __restrict__ output,
                                   int64_t OH, int64_t OW, int64_t H, int64_t W, int64_t C,
                                   int kh, int kw, int sh, int sw, int ph, int pw, int dh, int dw) {
    // Double buffer in shared memory
    extern __shared__ char smem[];
    float* buf[2];
    int tile_h = /* computed */, tile_w = /* computed */;
    int buf_size = tile_h * tile_w;
    buf[0] = reinterpret_cast<float*>(smem);
    buf[1] = buf[0] + buf_size;

    int64_t n = blockIdx.z;
    int64_t c = blockIdx.y;
    // ... same tile logic as v1

    // Pipeline: load first tile
    load_tile<T>(input, buf[0], ...);
    __syncthreads();

    int num_tiles = (OW + TILE_OW - 1) / TILE_OW; // iterate over tile columns
    for (int t = 0; t < num_tiles; t++) {
        int cur = t % 2;
        int nxt = 1 - cur;

        // If not last tile, start loading next tile
        if (t + 1 < num_tiles) {
            load_tile_async<T>(input, buf[nxt], /* next tile params */);
        }

        // Compute from current buffer
        compute_max_from_smem<T>(buf[cur], output, ...);

        __syncthreads(); // ensure async load + compute both done before next iteration
    }
}
```

Note: The actual implementation must use `cuda::memcpy_async` (CUDA 11+) or the `cp.async` PTX instruction for true asynchronous loads. The async pipeline API is: `__pipeline_memcpy_async(shared_dst, global_src, size)` to issue an async copy, `__pipeline_commit()` to commit the transaction, and `__pipeline_wait_prior(n)` to wait for completion. The shared memory buffers must be sized for the full input tile including halo region. The `tile_h` and `tile_w` values are computed as: `tile_h = (TILE_OH - 1) * sh + (kh - 1) * dh + 1`, `tile_w = (TILE_OW - 1) * sw + (kw - 1) * dw + 1`.

- [ ] **Step 3: Add launchers, update dispatch, update test versions, deploy/build/test**

- [ ] **Step 4: Commit**

```bash
git commit -m "feat: MaxPool2d v5 double buffer / pipeline"
```

---

## Task 15: AvgPool2d v5 — Double Buffer / Pipeline

Same double-buffer pattern as Task 14 but for AvgPool2d, with sum/count accumulation instead of max.

- [ ] **Step 1: Implement avgpool_v5 with double buffering, deploy/build/test/commit**

```bash
git commit -m "feat: AvgPool2d v5 double buffer / pipeline"
```

---

## Task 16: MaxPool2d v6 — Warp Specialization

**Files:**
- Modify: `src/pooling_max.cu`
- Modify: `include/pooling.cuh`
- Modify: `src/pybind_module.cpp`

- [ ] **Step 1: Add maxpool_v6 declarations to `include/pooling.cuh`**

```cpp
void maxpool_v6(const float* input, float* output, const PoolParams& params, cudaStream_t stream);
void maxpool_v6(const half* input, half* output, const PoolParams& params, cudaStream_t stream);
```

- [ ] **Step 2: Implement maxpool_v6 kernel with warp specialization**

Key idea: In a block with multiple warps (e.g., 256 threads = 8 warps), assign some warps as "load warps" that load data into shared memory, and other warps as "compute warps" that read from shared memory and compute max. Uses `__pipeline_memcpy_async` for load warps and `__pipeline_wait` for synchronization.

```cpp
template <typename T, int TILE_OH, int TILE_OW, int NUM_LOAD_WARPS, int NUM_COMPUTE_WARPS>
__global__ void maxpool_v6_kernel(const T* __restrict__ input, T* __restrict__ output,
                                   int64_t OH, int64_t OW, int64_t H, int64_t W, int64_t C,
                                   int kh, int kw, int sh, int sw, int ph, int pw, int dh, int dw) {
    extern __shared__ float smem[];
    int warp_id = threadIdx.x / 32;
    int num_warps = blockDim.x / 32;
    bool is_load_warp = warp_id < NUM_LOAD_WARPS;

    if (is_load_warp) {
        // Load warp: cooperatively load input tiles into shared memory
        // Signal compute warps via barrier
    } else {
        // Compute warp: wait for data, compute max, write output
    }
}
```

This uses cooperative groups or bar.sync for inter-warp synchronization within a block.

- [ ] **Step 3: Add launchers, update dispatch, update test versions, deploy/build/test**

- [ ] **Step 4: Commit**

```bash
git commit -m "feat: MaxPool2d v6 warp specialization"
```

---

## Task 17: AvgPool2d v6 — Warp Specialization

Same warp-specialization pattern as Task 16 but for AvgPool2d.

- [ ] **Step 1: Implement avgpool_v6 with warp specialization, deploy/build/test/commit**

```bash
git commit -m "feat: AvgPool2d v6 warp specialization"
```

---

## Task 18: MaxPool2d v7 — Alternative Grid/Block Mappings

**Files:**
- Modify: `src/pooling_max.cu`
- Modify: `include/pooling.cuh`
- Modify: `src/pybind_module.cpp`

- [ ] **Step 1: Add maxpool_v7 declarations to `include/pooling.cuh`**

```cpp
void maxpool_v7(const float* input, float* output, const PoolParams& params, int mapping, cudaStream_t stream);
void maxpool_v7(const half* input, half* output, const PoolParams& params, int mapping, cudaStream_t stream);
```

Note: v7 takes an extra `mapping` parameter (0=A, 1=B, 2=C, 3=D) to select the grid/block mapping.

- [ ] **Step 2: Implement mapping A (1D flat, same as v0 baseline)**

Already implemented in v0. v7 mapping A reuses v0's kernel logic.

- [ ] **Step 3: Implement mapping B (2D spatial)**

Each block covers a 2D tile of output spatial positions `(oh_tile, ow_tile)`. Threads within the block distribute across channels for those spatial positions.

```cpp
template <typename T, int TILE_OH, int TILE_OW>
__global__ void maxpool_v7_mappingB_kernel(...) {
    // blockIdx.x, blockIdx.y determine the spatial tile
    // threadIdx.x determines the channel index within the tile
    int64_t oh = blockIdx.y * TILE_OH + threadIdx.y;
    int64_t ow = blockIdx.x * TILE_OW + threadIdx.x;
    int64_t c = threadIdx.z; // channel dimension
    // ... compute max for (n, oh, ow, c)
}
```

- [ ] **Step 4: Implement mapping C (channel-major)**

Each block covers a range of channels. Spatial dimensions are distributed across grid dimensions.

```cpp
template <typename T, int C_TILE>
__global__ void maxpool_v7_mappingC_kernel(...) {
    int64_t oh = blockIdx.y;
    int64_t ow = blockIdx.x;
    int64_t c = blockIdx.z * C_TILE + threadIdx.x;
    // ... compute max for (n, oh, ow, c)
}
```

- [ ] **Step 5: Implement mapping D (hybrid: warp covers spatial+channel tile)**

Each warp handles a small spatial tile + 4 channels (via vectorized load). Different warps in the block handle different spatial tiles.

```cpp
template <typename T, int WARP_OH, int WARP_OW, int VEC=4>
__global__ void maxpool_v7_mappingD_kernel(...) {
    int warp_id = threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    // warp covers (WARP_OH, WARP_OW) spatial positions, VEC channels
    // different warps cover different spatial tiles
}
```

- [ ] **Step 6: Add v7 launcher that dispatches to the 4 mapping variants based on `mapping` parameter**

- [ ] **Step 7: Update pybind_module.cpp — v7 takes an extra `mapping` parameter**

```cpp
m.def("maxpool2d_f32", &maxpool2d_impl<float>,
      py::arg("input"), py::arg("kernel_size"), py::arg("stride") = py::none(),
      py::arg("padding") = 0, py::arg("dilation") = 1, py::arg("ceil_mode") = false,
      py::arg("version") = 0, py::arg("mapping") = 0);
```

- [ ] **Step 8: Update test versions and test all 4 mappings, deploy/build/test**

- [ ] **Step 9: Commit**

```bash
git commit -m "feat: MaxPool2d v7 alternative grid/block mappings (A/B/C/D)"
```

---

## Task 19: AvgPool2d v7 — Alternative Grid/Block Mappings

Same 4 mapping variants as Task 18 but for AvgPool2d.

- [ ] **Step 1: Implement avgpool_v7 with mappings A/B/C/D, update bindings, tests, deploy/build/test/commit**

```bash
git commit -m "feat: AvgPool2d v7 alternative grid/block mappings (A/B/C/D)"
```

---

## Task 20: Benchmark Suite

**Files:**
- Create: `benchmarks/bench_pooling.py`

- [ ] **Step 1: Create benchmarks/bench_pooling.py**

```python
import numpy as np
import time
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'build'))
import _pooling

def call_maxpool(x, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False, version=0, mapping=0):
    dtype = x.dtype
    fn = _pooling.maxpool2d_f16 if dtype == np.float16 else _pooling.maxpool2d_f32
    return fn(x, kernel_size, stride, padding, dilation, ceil_mode, version, mapping)

def call_avgpool(x, kernel_size, stride=None, padding=0, ceil_mode=True,
                 count_include_pad=True, divisor_override=None, version=0, mapping=0):
    dtype = x.dtype
    fn = _pooling.avgpool2d_f16 if dtype == np.float16 else _pooling.avgpool2d_f32
    return fn(x, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override, version, mapping)

def benchmark(fn, warmup=10, iters=50):
    for _ in range(warmup):
        fn()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    elapsed = (time.perf_counter() - start) / iters
    return elapsed  # seconds

def format_results(results):
    print(f"| {'Version':<8} | {'Time (ms)':>10} | {'GB/s':>8} | {'Speedup':>8} |")
    print(f"|----------|------------|----------|----------|")
    baseline = results[0][1]
    for name, t, bw in results:
        speedup = baseline / t if t > 0 else 0
        print(f"| {name:<8} | {t*1000:>10.3f} | {bw:>8.1f} | {speedup:>7.2f}x |")

# Synthetic benchmarks
SYNTHETIC = [
    ("small", (1, 32, 32, 64)),
    ("large_spatial", (1, 128, 128, 256)),
    ("batched", (16, 32, 32, 64)),
    ("large_c", (1, 28, 28, 512)),
]

KERNEL_CONFIGS = [
    ("3x3_s1_p0", 3, 1, 0),
    ("3x3_s2_p1", 3, 2, 1),
    ("5x5_s1_p2", 5, 1, 2),
    ("2x2_s2_p0", 2, 2, 0),
]

# Real model benchmarks
MODEL_CASES = [
    # (name, pool_type, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override, shape)
    ("resnet_maxpool", "max", 3, 2, 1, False, None, None, (1, 56, 56, 64)),
    ("resnet_global", "avg", None, 1, 0, False, True, None, (1, 7, 7, 512)),  # kernel_size=H,W
    ("vgg_maxpool", "max", 2, 2, 0, False, None, None, (1, 112, 112, 128)),
    ("densenet_maxpool", "max", 3, 2, 1, False, None, None, (1, 112, 112, 64)),
    ("densenet_avgpool", "avg", 2, 2, 0, False, True, None, (1, 56, 56, 64)),
    ("googlenet_maxpool", "max", 3, 2, 0, True, None, None, (1, 112, 112, 64)),
    ("googlenet_s1", "max", 3, 1, 1, True, None, None, (1, 28, 28, 480)),
    ("inception_v3_maxpool", "max", 3, 2, 0, False, None, None, (1, 35, 35, 288)),
    ("inception_v3_avgpool", "avg", 3, 1, 1, False, True, None, (1, 35, 35, 256)),
    ("inception_v3_aux", "avg", 5, 3, 0, False, True, None, (1, 17, 17, 768)),
    ("inception_v4_avgpool", "avg", 3, 1, 1, False, False, None, (1, 35, 35, 384)),
    ("inception_v4_maxpool", "max", 3, 2, 0, False, None, None, (1, 35, 35, 384)),
    ("inception_resnet_maxpool", "max", 3, 2, 0, False, None, None, (1, 73, 73, 192)),
    ("inception_resnet_avgpool", "avg", 3, 1, 1, False, False, None, (1, 35, 35, 192)),
    ("yolo_sppf", "max", 5, 1, 2, False, None, None, (1, 20, 20, 512)),
    ("yolo_spp_k9", "max", 9, 1, 4, False, None, None, (1, 19, 19, 512)),
    ("yolo_spp_k13", "max", 13, 1, 6, False, None, None, (1, 19, 19, 512)),
    ("yolov3_tiny_s1", "max", 2, 1, 0, False, None, None, (1, 13, 13, 512)),
    ("shufflenet_maxpool", "max", 3, 2, 1, False, None, None, (1, 112, 112, 24)),
    ("efficientnet_global", "avg", None, 1, 0, False, True, None, (1, 7, 7, 1280)),
    ("swin_global", "avg", None, 1, 0, False, True, None, (1, 7, 7, 768)),
]

ALL_VERSIONS = [0, 1, 2, 3, 4, 5, 6]
V7_MAPPINGS = [0, 1, 2, 3]

def run_benchmarks():
    dtype = np.float32
    print("=" * 70)
    print("SYNTHETIC BENCHMARKS (fp32)")
    print("=" * 70)

    for name, shape in SYNTHETIC:
        N, H, W, C = shape
        for kname, ks, stride, pad in KERNEL_CONFIGS:
            x = np.random.randn(*shape).astype(dtype)
            print(f"\n--- {name} {shape} kernel={kname} ---")
            results = []
            input_bytes = N * H * W * C * 4  # fp32
            for v in ALL_VERSIONS:
                fn = lambda v=v: call_maxpool(x, ks, stride, pad, 1, False, v)
                t = benchmark(fn)
                gb_per_s = input_bytes / t / 1e9
                results.append((f"v{v}", t, gb_per_s))
            # v7 mappings
            for m in V7_MAPPINGS:
                fn = lambda m=m: call_maxpool(x, ks, stride, pad, 1, False, 7, m)
                t = benchmark(fn)
                gb_per_s = input_bytes / t / 1e9
                results.append((f"v7m{'ABCD'[m]}", t, gb_per_s))
            format_results(results)

    print("\n" + "=" * 70)
    print("REAL MODEL BENCHMARKS (fp32)")
    print("=" * 70)

    for case in MODEL_CASES:
        name, ptype, ks, stride, pad, ceil, cip, div, shape = case
        N, H, W, C = shape
        x = np.random.randn(*shape).astype(dtype)
        if ks is None:
            ks = (H, W)  # global pooling
        print(f"\n--- {name} {shape} ---")
        results = []
        input_bytes = N * H * W * C * 4
        for v in ALL_VERSIONS:
            if ptype == "max":
                fn = lambda v=v: call_maxpool(x, ks, stride, pad, 1, ceil, v)
            else:
                fn = lambda v=v: call_avgpool(x, ks, stride, pad, ceil, cip, div, v)
            t = benchmark(fn)
            gb_per_s = input_bytes / t / 1e9
            results.append((f"v{v}", t, gb_per_s))
        for m in V7_MAPPINGS:
            if ptype == "max":
                fn = lambda m=m: call_maxpool(x, ks, stride, pad, 1, ceil, 7, m)
            else:
                fn = lambda m=m: call_avgpool(x, ks, stride, pad, ceil, cip, div, 7, m)
            t = benchmark(fn)
            gb_per_s = input_bytes / t / 1e9
            results.append((f"v7m{'ABCD'[m]}", t, gb_per_s))
        format_results(results)

if __name__ == "__main__":
    run_benchmarks()
```

- [ ] **Step 2: Deploy and run benchmarks on remote**

```bash
rsync -avz --exclude='.git' --exclude='build' ./ shuyua01@10.190.0.91:/home/shuyua01/Development/cuda-pooling/
ssh shuyua01@10.190.0.91 "cd /home/shuyua01/Development/cuda-pooling/build && cmake .. -DCMAKE_BUILD_TYPE=Release && make -j\$(nproc)"
ssh shuyua01@10.190.0.91 "cd /home/shuyua01/Development/cuda-pooling && python3 benchmarks/bench_pooling.py"
```

Expected: Full benchmark table printed with all versions and speedups.

- [ ] **Step 3: Commit**

```bash
git add benchmarks/bench_pooling.py
git commit -m "feat: benchmark suite with synthetic and real model cases"
```

---

## Task 21: Final Integration Test & Cleanup

**Files:**
- Modify: various (cleanup, documentation)

- [ ] **Step 1: Run full test suite on remote**

```bash
ssh shuyua01@10.190.0.91 "cd /home/shuyua01/Development/cuda-pooling && PYTHONPATH=build:\$PYTHONPATH python3 -m pytest tests/ -v --tb=short"
```

Expected: All tests PASS for all versions.

- [ ] **Step 2: Run benchmarks and save results**

```bash
ssh shuyua01@10.190.0.91 "cd /home/shuyua01/Development/cuda-pooling && python3 benchmarks/bench_pooling.py | tee benchmark_results.txt"
```

- [ ] **Step 3: Add .gitignore**

```gitignore
build/
__pycache__/
*.pyc
.claude/
benchmark_results.txt
```

- [ ] **Step 4: Final commit**

```bash
git add .gitignore
git commit -m "chore: add .gitignore and finalize project"
```
