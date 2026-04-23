# CUDA Pooling2D Design

High-performance CUDA Max/Avg Pooling2D implementation compatible with PyTorch API, using NHWC layout.

## 1. Project Structure

```
cuda-pooling/
├── CMakeLists.txt
├── include/
│   └── pooling.cuh          # Kernel launcher declarations
├── src/
│   ├── pooling_max.cu        # MaxPool2d kernels (all optimization stages)
│   ├── pooling_avg.cu        # AvgPool2d kernels (all optimization stages)
│   └── pybind_module.cpp     # pybind11 bindings
├── tests/
│   ├── test_maxpool.py       # MaxPool2d unit tests
│   └── test_avgpool.py       # AvgPool2d unit tests
└── benchmarks/
    └── bench_pooling.py      # Performance benchmarks at each optimization stage
```

All optimization stages live in the same `.cu` file with different function names (e.g., `maxpool_v0`, `maxpool_v1`, ...). The pybind11 module exposes versioned kernels. Single header `pooling.cuh` declares all launcher functions.

## 2. API Surface

### Python bindings (pybind11)

```python
# MaxPool2d
maxpool2d(input, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False, version=0)
# -> output numpy array

# AvgPool2d
avgpool2d(input, kernel_size, stride=None, padding=0, ceil_mode=True, count_include_pad=True, divisor_override=None, version=0)
# -> output numpy array
```

### Input/Output contract

- `input`: numpy array, dtype `float32` or `float16`, shape `[N, H, W, C]`
- `kernel_size`, `stride`, `padding`, `dilation`: `int` or `tuple(int, int)`
- `stride` defaults to `kernel_size` when None (matching PyTorch)
- `version`: selects optimization stage (0=baseline through 7)
- Returns: numpy array with same dtype, shape `[N, OH, OW, C]`

### Output size calculation (matching PyTorch)

- Floor mode: `OH = floor((H + 2*padding - dilation*(kernel_size-1) - 1) / stride + 1)`
- Ceil mode: `OH = ceil((H + 2*padding - dilation*(kernel_size-1) - 1) / stride + 1)`

### AvgPool2d specifics

- `count_include_pad=True`: padded zeros count toward the average denominator
- `count_include_pad=False`: only real input elements count
- `divisor_override`: when set, replaces the denominator with this value

### Not supported

- `return_indices` for MaxPool2d (explicitly excluded)

## 3. Kernel Design & Optimization Stages

### MaxPool2d stages

| Version | Name | Technique |
|---------|------|-----------|
| v0 | Naive | Per-output-element, global memory only |
| v1 | Shared memory tiling | Load input tile + halo into shared memory |
| v2 | Vectorized loads | `float4`/`half2` coalesced reads across channels |
| v3 | Register blocking | Each thread computes multiple output positions |
| v4 | Warp-level reduce | Replace per-thread loop over kernel window with warp shuffle `max` reduction for small kernel sizes |
| v5 | Double buffer / pipeline | Overlap shared memory loads with compute: one buffer loading next tile while another computes current tile |
| v6 | Warp specialization | Split warps in a block: some warps handle data loading into shared memory, others handle computation |
| v7 | Alternative grid/block mapping | Compare mapping strategies (see below) |

### AvgPool2d stages

Same progression (v0-v7), with warp-level reduce using warp shuffle `sum` instead of `max`.

### Grid/Block mapping variants (v7)

| Mapping | Description | Best for |
|---------|-------------|----------|
| A | 1D flat: tid -> (oh, ow, c) linearized, N in grid z | General purpose |
| B | 2D spatial: block covers (oh_tile, ow_tile), channels distributed across threads in the tile | Large spatial dims |
| C | Channel-major: block covers (c_tile), spatial dims in grid | Large C (e.g. 512+) |
| D | Hybrid: warp covers (oh_tile, ow_tile, c_tile=4 via vectorized load), warps in block cover different spatial tiles | Balanced workloads |

For v7, all 4 mappings are implemented and benchmarked against each other to document which works best under which conditions.

### Default grid/block mapping (v0-v6)

- Block: `(256, 1, 1)` — 256 threads per block
- Thread -> output mapping: `tid` maps to a flat output index across `(OH, OW, C)`, then batch `N` is the grid z-dimension
- `c = tid % C`, `ow = (tid / C) % OW`, `oh = tid / (C * OW)` within a block
- Grid: `(ceil(OH*OW*C / 256), 1, N)`

### fp16 strategy

Use `half` natively in CUDA. Launcher dispatches via template specialization. No mixed-precision — input fp16 means compute fp16.

## 4. Testing Strategy

### Framework

pytest with numpy + PyTorch as golden reference. Tests run on remote GPU via `ssh shuyua01@10.190.0.91`.

### Test method

Generate random NHWC numpy input -> call our kernel -> compare against PyTorch (convert NHWC->NCHW, run `F.max_pool2d`/`F.avg_pool2d`, convert back to NHWC).

Tolerance: `atol=1e-5` for fp32, `atol=1e-3` for fp16.

### MaxPool2d test matrix

| Parameter | Values tested |
|-----------|--------------|
| kernel_size | 1x1, 2x2, 3x3, 5x5, 7x7, (3,2), (2,3) |
| stride | default (=kernel_size), 1, 2, (3,2) |
| padding | 0, 1, 2, (1,2) |
| dilation | 1, 2, 3 |
| ceil_mode | True, False |
| dtype | float32, float16 |
| global pooling | kernel_size = (H, W), stride = 1, padding = 0 |

### AvgPool2d test matrix

| Parameter | Values tested |
|-----------|--------------|
| kernel_size | 2x2, 3x3, 4x4, (3,2) |
| stride | default (=kernel_size), 1, 2 |
| padding | 0, 1, (1,2) |
| ceil_mode | True, False |
| count_include_pad | True, False |
| divisor_override | None, 4 |
| dtype | float32, float16 |
| global pooling | kernel_size = (H, W), stride = 1, padding = 0 |

### Edge case tests

- Input size exactly equal to kernel size (output = 1x1)
- Large padding (output larger than input)
- Non-square kernel/stride/padding
- Dilation > 1 with padding > 0
- ceil_mode=True producing extra output row/column
- Global max/avg pooling at multiple spatial sizes: small (8x8), medium (32x32), large (64x64)

## 5. Benchmark Strategy

### Framework

pytest-benchmark with custom timing, run on remote GPU.

### Synthetic benchmarks

| Input shape | Description |
|-------------|-------------|
| (1, 32, 32, 64) | Small |
| (1, 128, 128, 256) | Large spatial |
| (16, 32, 32, 64) | Batched |
| (1, 28, 28, 512) | Large C |

### Real model benchmarks

| Case | Model | Pool Type | kernel_size | stride | padding | ceil_mode | count_include_pad | Input (H,W,C) |
|------|-------|-----------|-------------|--------|---------|-----------|-------------------|----------------|
| resnet_maxpool | ResNet | MaxPool | 3 | 2 | 1 | False | - | (56, 56, 64) |
| resnet_global | ResNet | AvgPool | global | 1 | 0 | False | - | (7, 7, 512) |
| vgg_maxpool | VGG | MaxPool | 2 | 2 | 0 | False | - | (112, 112, 128) |
| densenet_maxpool | DenseNet | MaxPool | 3 | 2 | 1 | False | - | (112, 112, 64) |
| densenet_avgpool | DenseNet | AvgPool | 2 | 2 | 0 | False | True | (56, 56, C) |
| googlenet_maxpool | GoogLeNet | MaxPool | 3 | 2 | 0 | True | - | (112, 112, 64) |
| googlenet_s1 | GoogLeNet | MaxPool | 3 | 1 | 1 | True | - | (28, 28, 480) |
| inception_v3_maxpool | Inception v3 | MaxPool | 3 | 2 | 0 | False | - | (35, 35, 288) |
| inception_v3_avgpool | Inception v3 | AvgPool | 3 | 1 | 1 | False | True | (35, 35, C) |
| inception_v3_aux | Inception v3 | AvgPool | 5 | 3 | 0 | False | True | (17, 17, 768) |
| inception_v4_avgpool | Inception v4 | AvgPool | 3 | 1 | 1 | False | False | (35, 35, 384) |
| inception_v4_maxpool | Inception v4 | MaxPool | 3 | 2 | 0 | False | - | (35, 35, 384) |
| inception_resnet_maxpool | Inception-ResNet | MaxPool | 3 | 2 | 0 | False | - | (73, 73, 192) |
| inception_resnet_avgpool | Inception-ResNet | AvgPool | 3 | 1 | 1 | False | False | (35, 35, 192) |
| yolo_sppf | YOLOv5/v8 | MaxPool | 5 | 1 | 2 | False | - | (20, 20, 512) |
| yolo_spp_k9 | YOLOv4-SPP | MaxPool | 9 | 1 | 4 | False | - | (19, 19, 512) |
| yolo_spp_k13 | YOLOv4-SPP | MaxPool | 13 | 1 | 6 | False | - | (19, 19, 512) |
| yolov3_tiny_s1 | YOLOv3-Tiny | MaxPool | 2 | 1 | 0 | False | - | (13, 13, 512) |
| shufflenet_maxpool | ShuffleNetV2 | MaxPool | 3 | 2 | 1 | False | - | (112, 112, 24) |
| efficientnet_global | EfficientNet | AvgPool | global | 1 | 0 | False | - | (7, 7, C) |
| swin_global | Swin Transformer | AvgPool | global | 1 | 0 | False | - | (7, 7, 768) |

### Kernel configs per synthetic shape

(3,3)/s1/p0, (3,3)/s2/p1, (5,5)/s1/p2, (2,2)/s2/p0, global pooling

### Benchmark procedure

- Warmup: 10 iterations (skip timing)
- Measurement: 50 iterations, report median time
- For each (shape, kernel_config, dtype) combo, run all kernel versions (v0-v7)
- Report: wall clock time (ms), throughput (GB/s of input data read), speedup vs v0

### Output format

```
| Version | Time (ms) | GB/s   | Speedup |
|---------|-----------|--------|---------|
| v0      | 0.42      | 18.3   | 1.00x   |
| v1      | 0.19      | 40.5   | 2.21x   |
| ...     |           |        |         |
```

## 6. Build & Deploy Workflow

### Local (Windows, no GPU)

- Write and edit code locally
- `git commit` changes locally

### Remote (10.190.0.91, NVIDIA Thor, CUDA 13.0)

- `rsync` repo to `/home/shuyua01/Development/cuda-pooling/`
- Build: `mkdir build && cd build && cmake .. && make -j`
- Test: `cd .. && pytest tests/ -v`
- Benchmark: `python benchmarks/bench_pooling.py`

### CI-like flow

1. Local: edit -> git commit
2. `rsync -avz --exclude='.git' ./ shuyua01@10.190.0.91:/home/shuyua01/Development/cuda-pooling/`
3. SSH: `cd /home/shuyua01/Development/cuda-pooling && mkdir -p build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release && make -j$(nproc)`
4. SSH: `cd /home/shuyua01/Development/cuda-pooling && pytest tests/ -v`
5. SSH: `python benchmarks/bench_pooling.py`

### CMake configuration

- CUDA architecture: `80;90;100;110` (Ampere through Blackwell)
- pybind11: fetch via CMake `FetchContent`
- Python: detect via `find_package(Python3 COMPONENTS Interpreter Development NumPy)`

## 7. Environment

- **Remote GPU**: NVIDIA Thor (compute capability 11.0, Blackwell)
- **CUDA**: 13.0
- **PyTorch**: 2.9.1+cu130
- **Python**: 3.12.3
