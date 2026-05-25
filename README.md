# CUDA Pooling2D

High-performance CUDA Max/Avg Pooling2D kernels with PyTorch-compatible API, supporting fp32/fp16 and NHWC layout.

## Overview

This project implements 16 optimization stages (v0-v15) of 2D pooling kernels for NVIDIA GPUs, systematically exploring GPU optimization techniques from naive global-memory kernels through vectorized loads, shared memory tiling, warp-level primitives, auto-tuned tiling, TMA warp-specialized pipelines, persistent kernels, and bank-conflict-free shared memory layouts.

**Target hardware**: NVIDIA Thor (SM 11.0, Blackwell), CUDA 13.0. Multi-architecture builds support SM 80-110.

**Best result**: v2 (vectorized loads) achieves **2.5-3.8x** speedup over the naive baseline across all benchmark configurations. v15 (swizzled shared memory) is the fastest for 3x3 stride-1 patterns.

## Features

- **PyTorch API compatible**: Supports all MaxPool2d and AvgPool2d parameters (`kernel_size`, `stride`, `padding`, `dilation`, `ceil_mode`, `count_include_pad`, `divisor_override`)
- **fp32 and fp16**: Native `float` and `half` computation
- **NHWC layout**: Input and output shape `[N, H, W, C]`
- **16 kernel versions**: Selectable at runtime via `version` parameter
- **v14 adaptive dispatcher**: Automatically routes to the optimal kernel based on input shape and kernel parameters

## Project Structure

```
cuda-pooling/
├── CMakeLists.txt                      # Build system
├── include/
│   └── pooling.cuh                     # Kernel launcher declarations, param structs
├── src/
│   ├── pooling_max.cu                  # MaxPool2d kernels (v0-v15)
│   ├── pooling_avg.cu                  # AvgPool2d kernels (v0-v15)
│   └── pybind_module.cpp              # pybind11 Python bindings
├── tests/
│   ├── conftest.py                     # Test fixtures, PyTorch golden reference
│   ├── test_maxpool.py                 # MaxPool2d unit tests
│   └── test_avgpool.py                 # AvgPool2d unit tests
├── benchmarks/
│   ├── bench_pooling.py                # Synthetic + real-model benchmarks
│   ├── bench_timed.py                  # CUDA event-timed benchmarks
│   ├── profile_ncu.py                  # Nsight Compute hardware metrics
│   ├── profile_nsys.py                 # Nsight Systems timeline profiling
│   └── analyze_performance.py          # Performance analysis utilities
└── docs/
    ├── profiling_report.md             # Detailed per-version performance analysis
    └── superpowers/
        ├── specs/2026-04-22-cuda-pooling2d-design.md
        ├── specs/2026-05-20-cuda-pooling-advanced-optimizations.md
        └── plans/2026-04-22-cuda-pooling2d.md
```

## Build

### Prerequisites

- CMake >= 3.24
- CUDA Toolkit >= 13.0
- Python 3 with NumPy
- pybind11 (fetched automatically via FetchContent)

### Steps

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j
```

The build produces `_pooling.cpython-*.so`, a Python module importable as `import _pooling`.

### NVTX Profiling Annotations

NVTX annotations are enabled by default (`-DENABLE_NVTX=ON`). They add zero overhead when not running under Nsight Systems. Disable with `-DENABLE_NVTX=OFF`.

## Python API

```python
import _pooling

# MaxPool2d (fp32)
output = _pooling.maxpool2d_f32(
    input,              # numpy array [N, H, W, C], dtype float32
    kernel_size,        # int or (int, int)
    stride=None,        # int or (int, int); defaults to kernel_size
    padding=0,          # int or (int, int)
    dilation=1,         # int or (int, int)
    ceil_mode=False,    # bool
    version=0,          # int: kernel version (0-15)
    mapping=0,          # int: v7 mapping variant (0-3)
)

# MaxPool2d (fp16)
output = _pooling.maxpool2d_f16(input, ...)  # same args, numpy float16

# AvgPool2d (fp32)
output = _pooling.avgpool2d_f32(
    input,              # numpy array [N, H, W, C], dtype float32
    kernel_size,        # int or (int, int)
    stride=None,        # int or (int, int)
    padding=0,          # int or (int, int)
    dilation=1,         # int or (int, int)
    ceil_mode=True,     # bool (PyTorch default)
    count_include_pad=True,  # bool
    divisor_override=None,   # int or None
    version=0,          # int: kernel version (0-15)
    mapping=0,          # int: v7 mapping variant (0-3)
)

# AvgPool2d (fp16)
output = _pooling.avgpool2d_f16(input, ...)  # same args, numpy float16

# Timed variants (return (output, elapsed_ms) using CUDA events)
output, ms = _pooling.maxpool2d_timed_f32(input, ..., version=2)

# Synchronize
_pooling.cuda_synchronize()
```

## Kernel Versions

### Core kernels (v0-v7)

| Version | Name | Technique | Best For |
|---------|------|-----------|----------|
| v0 | Naive | 1D flat grid, global memory only | Fallback for any configuration |
| v1 | Shared Memory Tiling | 2D tile (8x8) with halo load | Stride-1 with large kernels |
| **v2** | **Vectorized Loads** | **`float4`/`half2` coalesced reads** | **General purpose (best overall)** |
| v3 | Register Blocking | Each thread computes 4 output rows | Stride-2 with large OH |
| v4 | Warp-Level Reduce | Warp shuffle reduction | Global pooling (large karea) |
| v5 | Double Buffer | Two-channel blocks (sequential) | Marginal benefit; architectural limitation |
| v6 | Warp Specialization | Load warps + compute warps | Marginal benefit; needs async pipeline |
| v7 | Alternative Mappings | 4 grid/block mapping variants | Research/comparison |

v7 has 4 mapping sub-variants (A/B/C/D):
- **mA**: 1D flat (identical to v0)
- **mB**: 2D spatial with channel distribution
- **mC**: Channel-major blocking
- **mD**: Hybrid warp-spatial + vectorized loads (best alternative after v2)

### Advanced optimizations (v8-v15)

| Version | Name | Technique | Hardware |
|---------|------|-----------|----------|
| v8 | Auto-Tuned Tiling | Sweeps tile dimensions, caches winner | All |
| v9 | TMA Warp-Specialized Pipeline | Producer/consumer warps, TMA bulk copy, mbarrier sync | SM90+ (falls back to v2) |
| v10 | Persistent Kernel | 1 block/SM, atomic work queue | All |
| v11 | Warp-Shuffle Fix | Corrected warp-shuffle for maxpool/avgpool | All |
| v12 | L2-Aware Persistent | L2 cache-aware persistent kernel | All |
| v13 | Channel-Vectorized Warp | Warp processes channel vectors | All |
| **v14** | **Adaptive Dispatcher** | **Auto-routes to optimal kernel** | **All (recommended default)** |
| **v15** | **Swizzled Shared Memory** | **Column-padded smem eliminates bank conflicts** | **3x3 stride-1** |

### v14 Adaptive Dispatcher Routing

v14 automatically selects the best kernel based on parameters:

| Pattern | Routed To |
|---------|-----------|
| 3x3 stride-1 | v15 (swizzled shared memory) |
| Global pooling (OH=OW=1) | v4 (warp-level reduce) |
| Other (aligned C) | v2 (vectorized loads) |
| Other (unaligned C) | v0 (naive fallback) |

## Benchmarks

### Synthetic benchmarks

| Shape | Description |
|-------|-------------|
| (1, 32, 32, 64) | Small |
| (1, 128, 128, 256) | Large spatial |
| (16, 32, 32, 64) | Batched |
| (1, 28, 28, 512) | Large C |

### Real model benchmarks

Covers ResNet, VGG, DenseNet, GoogLeNet, Inception v3/v4, YOLO SPP, ShuffleNet, EfficientNet, Swin Transformer pooling configurations.

### Running benchmarks

```bash
# Standard benchmarks (wall-clock time)
python benchmarks/bench_pooling.py

# CUDA event-timed benchmarks (kernel-only, excluding H2D/D2H)
python benchmarks/bench_timed.py

# Nsight Systems timeline profiling (requires nsys)
python benchmarks/profile_nsys.py

# Nsight Compute hardware metrics (requires ncu)
python benchmarks/profile_ncu.py
```

## Testing

Tests compare kernel output against PyTorch's `F.max_pool2d` / `F.avg_pool2d` as golden reference.

```bash
pytest tests/ -v
```

**Tolerance**: `atol=1e-5` (fp32), `atol=1e-3` (fp16)

**Test matrix** covers: kernel sizes (1x1 through 13x13), strides (1, 2, default), padding (0-2), dilation (1-3), ceil_mode (True/False), count_include_pad (True/False), divisor_override, global pooling, fp32/fp16, across multiple versions.

## Performance Summary

Based on comprehensive profiling on NVIDIA Thor (SM 11.0, Blackwell):

- **v2 (vectorized loads)** is the best general-purpose kernel: 2.5-3.8x over v0 for aligned channel counts
- **v15 (swizzled shared memory)** is optimal for 3x3 stride-1 patterns, eliminating 8-way shared memory bank conflicts
- **v4 (warp reduce)** wins only for global pooling with large kernels (7x7+): 1.6-1.9x over v0
- Pooling is **memory-bound**: arithmetic intensity < 0.5 ops/byte for all configurations
- The performance ceiling is DRAM bandwidth; best kernels achieve high bus utilization through coalesced 128-bit transactions

See `docs/profiling_report.md` for detailed per-version analysis including roofline model, stall analysis, and quantitative breakdown of why certain techniques (v5 double buffer, v6 warp specialization) underperform without async pipeline support.

## Development Workflow

This project is developed locally on Windows and deployed to a remote GPU server:

```bash
# 1. Edit code locally, commit
git commit -m "..."

# 2. Deploy to remote GPU
scp -r . shuyua01@10.190.0.91:/home/shuyua01/Development/cuda-pooling

# 3. Build and test on remote
ssh shuyua01@10.190.0.91 "cd /home/shuyua01/Development/cuda-pooling && \
  rm -rf build && mkdir build && cd build && \
  cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.0/bin/nvcc -DCMAKE_CUDA_ARCHITECTURES=110 && \
  make -j && cd .. && pytest tests/ -v"
```

## Architecture Notes

### Why v5 (Double Buffer) and v6 (Warp Specialization) Underperform

Both techniques require **async memory copies** (`cp.async` / `__pipeline_memcpy_async`) to overlap global-to-shared-memory loads with computation. Without async copies, the load and compute phases execute sequentially with `__syncthreads()` barriers between them. The 2x shared memory cost and extra synchronization make them equal to or worse than simpler approaches. See `docs/profiling_report.md` for the full quantitative analysis.

### Shared Memory Bank Conflicts (v15)

For NHWC layout with consecutive threads processing adjacent output positions, shared memory accesses for overlapping pooling windows cause 8-way bank conflicts (fp32, 4-byte words). v15 solves this by allocating `smem_w + 1` columns (column padding), breaking the bank alignment pattern entirely. This yields significant speedup for 3x3 stride-1 where adjacent threads share the most window overlap.

### Roofline Model

| Config | karea | FLOPs/output | Bytes/input+output | AI (ops/byte) |
|--------|-------|-------------|-------------------|---------------|
| 3x3 fp32 | 9 | 9 | 40 | 0.23 |
| 3x3 fp16 | 9 | 9 | 20 | 0.45 |
| 5x5 fp32 | 25 | 25 | 104 | 0.24 |
| 7x7 fp32 | 49 | 49 | 200 | 0.25 |

With AI < 0.5, all pooling kernels are firmly memory-bound. The performance ceiling is DRAM bandwidth.

## License

MIT
