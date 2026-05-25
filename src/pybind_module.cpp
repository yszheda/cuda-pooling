#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <utility>
#include <vector>
#include <stdexcept>
#include <string>
#include <mutex>
#include "pooling.cuh"

namespace py = pybind11;

// Pre-allocated device memory arena to avoid cudaMalloc/cudaFree during runtime.
// Sequential malloc/free across different kernel types caused "misaligned address"
// errors in batch tests. We reserve large buffers once and reuse them.
struct DeviceArena {
    void* ptr = nullptr;
    size_t capacity = 0;

    ~DeviceArena() { if (ptr) cudaFree(ptr); }

    void ensure(size_t bytes) {
        if (bytes <= capacity) return;
        if (ptr) cudaFree(ptr);
        CUDA_CHECK(cudaMalloc(&ptr, bytes));
        capacity = bytes;
    }
};

static std::mutex arena_mutex;
static DeviceArena arena_f32_in;   // max 256 MB for f32 input
static DeviceArena arena_f32_out;  // max 256 MB for f32 output
static DeviceArena arena_f16_in;   // max 128 MB for f16 input
static DeviceArena arena_f16_out;  // max 128 MB for f16 output

static constexpr size_t MAX_F32_BYTES = 256ULL * 1024 * 1024;
static constexpr size_t MAX_F16_BYTES = 128ULL * 1024 * 1024;

static void init_arenas() {
    arena_f32_in.ensure(MAX_F32_BYTES);
    arena_f32_out.ensure(MAX_F32_BYTES);
    arena_f16_in.ensure(MAX_F16_BYTES);
    arena_f16_out.ensure(MAX_F16_BYTES);
}

// Parse a parameter that can be int or tuple(int, int) into (h, w)
static std::pair<int, int> parse_pair(const py::object& obj) {
    if (py::isinstance<py::int_>(obj)) {
        int v = obj.cast<int>();
        return {v, v};
    } else if (py::isinstance<py::tuple>(obj)) {
        auto t = obj.cast<py::tuple>();
        if (t.size() != 2)
            throw std::invalid_argument("tuple must have exactly 2 elements");
        return {t[0].cast<int>(), t[1].cast<int>()};
    }
    throw std::invalid_argument("expected int or tuple(int, int)");
}

// Build PoolParams from shape and pooling args
static PoolParams make_pool_params(
    const std::vector<py::ssize_t>& shape,
    int kh, int kw, int sh, int sw, int ph, int pw, int dh, int dw,
    bool ceil_mode)
{
    PoolParams params;
    params.N = shape[0];
    params.H = shape[1];
    params.W = shape[2];
    params.C = shape[3];
    params.kh = kh; params.kw = kw;
    params.sh = sh; params.sw = sw;
    params.ph = ph; params.pw = pw;
    params.dh = dh; params.dw = dw;
    params.ceil_mode = ceil_mode;
    params.compute_output_size();
    return params;
}

// Convert a generic py::array's shape to vector<ssize_t>
static std::vector<py::ssize_t> array_shape(const py::array& arr) {
    auto ndim = arr.ndim();
    std::vector<py::ssize_t> shape(ndim);
    for (ssize_t i = 0; i < ndim; ++i)
        shape[i] = arr.shape(i);
    return shape;
}

// Ensure array is C-contiguous; return a reference to the original or a copy
static py::array ensure_c_contiguous(const py::array& arr) {
    if (arr.flags() & py::array::c_style)
        return arr;
    // Make a contiguous copy with same dtype
    py::module_ np = py::module_::import("numpy");
    return np.attr("ascontiguousarray")(arr).cast<py::array>();
}

// Validate that array dtype is float16 (2-byte float)
static void validate_float16(const py::array& arr) {
    if (arr.dtype().kind() != 'f' || arr.dtype().itemsize() != 2)
        throw std::invalid_argument("input must be float16 (numpy float16 / 2-byte float)");
}

// ==================== MaxPool2d ====================

py::array_t<float> maxpool2d_f32(
    py::array_t<float, py::array::c_style | py::array::forcecast> input,
    const py::object& kernel_size,
    const py::object& stride,
    const py::object& padding,
    const py::object& dilation,
    bool ceil_mode,
    int version,
    int mapping)
{
    auto buf = input.request();
    if (buf.ndim != 4)
        throw std::invalid_argument("input must be 4-D [N, H, W, C]");

    auto [kh, kw] = parse_pair(kernel_size);
    auto [sh, sw] = stride.is_none() ? std::make_pair(kh, kw) : parse_pair(stride);
    auto [ph, pw] = parse_pair(padding);
    auto [dh, dw] = parse_pair(dilation);

    auto params = make_pool_params(buf.shape, kh, kw, sh, sw, ph, pw, dh, dw, ceil_mode);
    size_t in_nelms = buf.size;
    size_t out_nelms = static_cast<size_t>(params.N * params.OH * params.OW * params.C);

    // Allocate device buffers from arena and copy input H2D
    {
        std::lock_guard<std::mutex> lock(arena_mutex);
        arena_f32_in.ensure(in_nelms * sizeof(float));
        arena_f32_out.ensure(out_nelms * sizeof(float));
    }
    float* d_input = static_cast<float*>(arena_f32_in.ptr);
    float* d_output = static_cast<float*>(arena_f32_out.ptr);
    CUDA_CHECK(cudaMemcpy(d_input, buf.ptr, in_nelms * sizeof(float), cudaMemcpyHostToDevice));

    // Launch kernel
    switch (version) {
        case 0:
            maxpool_v0(d_input, d_output, params, 0);
            break;
        case 1:
            maxpool_v1(d_input, d_output, params, 0);
            break;
        case 2:
            maxpool_v2(d_input, d_output, params, 0);
            break;
        case 3:
            maxpool_v3(d_input, d_output, params, 0);
            break;
        case 4:
            maxpool_v4(d_input, d_output, params, 0);
            break;
        case 5:
            maxpool_v5(d_input, d_output, params, 0);
            break;
        case 6:
            maxpool_v6(d_input, d_output, params, 0);
            break;
        case 7:
            maxpool_v7(d_input, d_output, params, mapping, 0);
            break;
        case 8:
            maxpool_v8(d_input, d_output, params, 0);
            break;
        case 9:
            maxpool_v9(d_input, d_output, params, 0);
            break;
        case 10:
            maxpool_v10(d_input, d_output, params, 0);
            break;
        case 11:
            maxpool_v11(d_input, d_output, params, 0);
            break;
        case 12:
            maxpool_v12(d_input, d_output, params, 0);
            break;
        case 13:
            maxpool_v13(d_input, d_output, params, 0);
            break;
        case 14:
            maxpool_v14(d_input, d_output, params, 0);
            break;
        case 15:
            maxpool_v15(d_input, d_output, params, 0);
            break;
        default:
            throw std::invalid_argument("unsupported kernel version: " + std::to_string(version));
    }

    // Copy output D2H
    std::vector<py::ssize_t> out_shape = {params.N, params.OH, params.OW, params.C};
    auto output = py::array_t<float>(out_shape);
    auto out_buf = output.request();
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(out_buf.ptr, d_output, out_nelms * sizeof(float), cudaMemcpyDeviceToHost));

    return output;
}

// fp16 version: numpy float16 has same memory layout as CUDA half (IEEE 754 binary16).
// We use py::array (generic) and validate dtype manually, then access raw pointers.
py::array maxpool2d_f16(
    py::array input,
    const py::object& kernel_size,
    const py::object& stride,
    const py::object& padding,
    const py::object& dilation,
    bool ceil_mode,
    int version,
    int mapping)
{
    validate_float16(input);
    input = ensure_c_contiguous(input);

    if (input.ndim() != 4)
        throw std::invalid_argument("input must be 4-D [N, H, W, C]");

    auto shape = array_shape(input);
    size_t in_nbytes = input.size() * sizeof(half);

    auto [kh, kw] = parse_pair(kernel_size);
    auto [sh, sw] = stride.is_none() ? std::make_pair(kh, kw) : parse_pair(stride);
    auto [ph, pw] = parse_pair(padding);
    auto [dh, dw] = parse_pair(dilation);

    auto params = make_pool_params(shape, kh, kw, sh, sw, ph, pw, dh, dw, ceil_mode);
    size_t out_nelms = static_cast<size_t>(params.N * params.OH * params.OW * params.C);
    size_t out_nbytes = out_nelms * sizeof(half);

    // Allocate device buffers and copy input H2D
    {
        std::lock_guard<std::mutex> lock(arena_mutex);
        arena_f16_in.ensure(in_nbytes);
        arena_f16_out.ensure(out_nbytes);
    }
    half* d_input = static_cast<half*>(arena_f16_in.ptr);
    half* d_output = static_cast<half*>(arena_f16_out.ptr);
    CUDA_CHECK(cudaMemcpy(d_input, input.data(), in_nbytes, cudaMemcpyHostToDevice));

    // Launch kernel
    switch (version) {
        case 0:
            maxpool_v0(d_input, d_output, params, 0);
            break;
        case 1:
            maxpool_v1(d_input, d_output, params, 0);
            break;
        case 2:
            maxpool_v2(d_input, d_output, params, 0);
            break;
        case 3:
            maxpool_v3(d_input, d_output, params, 0);
            break;
        case 4:
            maxpool_v4(d_input, d_output, params, 0);
            break;
        case 5:
            maxpool_v5(d_input, d_output, params, 0);
            break;
        case 6:
            maxpool_v6(d_input, d_output, params, 0);
            break;
        case 7:
            maxpool_v7(d_input, d_output, params, mapping, 0);
            break;
        case 8:
            maxpool_v8(d_input, d_output, params, 0);
            break;
        case 9:
            maxpool_v9(d_input, d_output, params, 0);
            break;
        case 10:
            maxpool_v10(d_input, d_output, params, 0);
            break;
        case 11:
            maxpool_v11(d_input, d_output, params, 0);
            break;
        case 12:
            maxpool_v12(d_input, d_output, params, 0);
            break;
        case 13:
            maxpool_v13(d_input, d_output, params, 0);
            break;
        case 14:
            maxpool_v14(d_input, d_output, params, 0);
            break;
        case 15:
            maxpool_v15(d_input, d_output, params, 0);
            break;
        default:
            throw std::invalid_argument("unsupported kernel version: " + std::to_string(version));
    }

    // Copy result back before allowing next call to reuse arena
    CUDA_CHECK(cudaDeviceSynchronize());

    // Allocate output numpy array with float16 dtype
    std::vector<py::ssize_t> out_shape = {params.N, params.OH, params.OW, params.C};
    py::module_ np = py::module_::import("numpy");
    auto output = np.attr("empty")(out_shape, np.attr("float16")).cast<py::array>();

    // Copy output D2H and free device memory
    CUDA_CHECK(cudaMemcpy(output.mutable_data(), d_output, out_nbytes, cudaMemcpyDeviceToHost));

    return output;
}

// ==================== AvgPool2d ====================

static AvgPoolParams make_avgpool_params(
    const std::vector<py::ssize_t>& shape,
    int kh, int kw, int sh, int sw, int ph, int pw, int dh, int dw,
    bool ceil_mode, bool count_include_pad, int64_t divisor_override)
{
    AvgPoolParams params;
    params.N = shape[0];
    params.H = shape[1];
    params.W = shape[2];
    params.C = shape[3];
    params.kh = kh; params.kw = kw;
    params.sh = sh; params.sw = sw;
    params.ph = ph; params.pw = pw;
    params.dh = dh; params.dw = dw;
    params.ceil_mode = ceil_mode;
    params.count_include_pad = count_include_pad;
    params.divisor_override = divisor_override;
    params.compute_output_size();
    return params;
}

py::array_t<float> avgpool2d_f32(
    py::array_t<float, py::array::c_style | py::array::forcecast> input,
    const py::object& kernel_size,
    const py::object& stride,
    const py::object& padding,
    const py::object& dilation,
    bool ceil_mode,
    bool count_include_pad,
    const py::object& divisor_override,
    int version,
    int mapping)
{
    auto buf = input.request();
    if (buf.ndim != 4)
        throw std::invalid_argument("input must be 4-D [N, H, W, C]");

    auto [kh, kw] = parse_pair(kernel_size);
    auto [sh, sw] = stride.is_none() ? std::make_pair(kh, kw) : parse_pair(stride);
    auto [ph, pw] = parse_pair(padding);
    auto [dh, dw] = parse_pair(dilation);

    int64_t div_over = divisor_override.is_none() ? 0 : divisor_override.cast<int64_t>();
    auto params = make_avgpool_params(buf.shape, kh, kw, sh, sw, ph, pw, dh, dw,
                                      ceil_mode, count_include_pad, div_over);
    size_t in_nelms = buf.size;
    size_t out_nelms = static_cast<size_t>(params.N * params.OH * params.OW * params.C);

    // Allocate device buffers from arena and copy input H2D
    {
        std::lock_guard<std::mutex> lock(arena_mutex);
        arena_f32_in.ensure(in_nelms * sizeof(float));
        arena_f32_out.ensure(out_nelms * sizeof(float));
    }
    float* d_input = static_cast<float*>(arena_f32_in.ptr);
    float* d_output = static_cast<float*>(arena_f32_out.ptr);
    CUDA_CHECK(cudaMemcpy(d_input, buf.ptr, in_nelms * sizeof(float), cudaMemcpyHostToDevice));

    // Launch kernel
    switch (version) {
        case 0:
            avgpool_v0(d_input, d_output, params, 0);
            break;
        case 1:
            avgpool_v1(d_input, d_output, params, 0);
            break;
        case 2:
            avgpool_v2(d_input, d_output, params, 0);
            break;
        case 3:
            avgpool_v3(d_input, d_output, params, 0);
            break;
        case 4:
            avgpool_v4(d_input, d_output, params, 0);
            break;
        case 5:
            avgpool_v5(d_input, d_output, params, 0);
            break;
        case 6:
            avgpool_v6(d_input, d_output, params, 0);
            break;
        case 7:
            avgpool_v7(d_input, d_output, params, mapping, 0);
            break;
        case 8:
            avgpool_v8(d_input, d_output, params, 0);
            break;
        case 9:
            avgpool_v9(d_input, d_output, params, 0);
            break;
        case 10:
            avgpool_v10(d_input, d_output, params, 0);
            break;
        case 11:
            avgpool_v11(d_input, d_output, params, 0);
            break;
        case 12:
            avgpool_v12(d_input, d_output, params, 0);
            break;
        case 13:
            avgpool_v13(d_input, d_output, params, 0);
            break;
        case 14:
            avgpool_v14(d_input, d_output, params, 0);
            break;
        case 15:
            avgpool_v15(d_input, d_output, params, 0);
            break;
        default:
            throw std::invalid_argument("unsupported kernel version: " + std::to_string(version));
    }

    // Copy output D2H
    std::vector<py::ssize_t> out_shape = {params.N, params.OH, params.OW, params.C};
    auto output = py::array_t<float>(out_shape);
    auto out_buf = output.request();
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(out_buf.ptr, d_output, out_nelms * sizeof(float), cudaMemcpyDeviceToHost));

    return output;
}

py::array avgpool2d_f16(
    py::array input,
    const py::object& kernel_size,
    const py::object& stride,
    const py::object& padding,
    const py::object& dilation,
    bool ceil_mode,
    bool count_include_pad,
    const py::object& divisor_override,
    int version,
    int mapping)
{
    validate_float16(input);
    input = ensure_c_contiguous(input);

    if (input.ndim() != 4)
        throw std::invalid_argument("input must be 4-D [N, H, W, C]");

    auto shape = array_shape(input);
    size_t in_nbytes = input.size() * sizeof(half);

    auto [kh, kw] = parse_pair(kernel_size);
    auto [sh, sw] = stride.is_none() ? std::make_pair(kh, kw) : parse_pair(stride);
    auto [ph, pw] = parse_pair(padding);
    auto [dh, dw] = parse_pair(dilation);

    int64_t div_over = divisor_override.is_none() ? 0 : divisor_override.cast<int64_t>();
    auto params = make_avgpool_params(shape, kh, kw, sh, sw, ph, pw, dh, dw,
                                      ceil_mode, count_include_pad, div_over);
    size_t out_nelms = static_cast<size_t>(params.N * params.OH * params.OW * params.C);
    size_t out_nbytes = out_nelms * sizeof(half);

    // Allocate device buffers from arena and copy input H2D
    {
        std::lock_guard<std::mutex> lock(arena_mutex);
        arena_f16_in.ensure(in_nbytes);
        arena_f16_out.ensure(out_nbytes);
    }
    half* d_input = static_cast<half*>(arena_f16_in.ptr);
    half* d_output = static_cast<half*>(arena_f16_out.ptr);
    CUDA_CHECK(cudaMemcpy(d_input, input.data(), in_nbytes, cudaMemcpyHostToDevice));

    // Launch kernel
    switch (version) {
        case 0:
            avgpool_v0(d_input, d_output, params, 0);
            break;
        case 1:
            avgpool_v1(d_input, d_output, params, 0);
            break;
        case 2:
            avgpool_v2(d_input, d_output, params, 0);
            break;
        case 3:
            avgpool_v3(d_input, d_output, params, 0);
            break;
        case 4:
            avgpool_v4(d_input, d_output, params, 0);
            break;
        case 5:
            avgpool_v5(d_input, d_output, params, 0);
            break;
        case 6:
            avgpool_v6(d_input, d_output, params, 0);
            break;
        case 7:
            avgpool_v7(d_input, d_output, params, mapping, 0);
            break;
        case 8:
            avgpool_v8(d_input, d_output, params, 0);
            break;
        case 9:
            avgpool_v9(d_input, d_output, params, 0);
            break;
        case 10:
            avgpool_v10(d_input, d_output, params, 0);
            break;
        case 11:
            avgpool_v11(d_input, d_output, params, 0);
            break;
        case 12:
            avgpool_v12(d_input, d_output, params, 0);
            break;
        case 13:
            avgpool_v13(d_input, d_output, params, 0);
            break;
        case 14:
            avgpool_v14(d_input, d_output, params, 0);
            break;
        case 15:
            avgpool_v15(d_input, d_output, params, 0);
            break;
        default:
            throw std::invalid_argument("unsupported kernel version: " + std::to_string(version));
    }

    // Wait for kernel to finish before allowing next call to reuse arena
    CUDA_CHECK(cudaDeviceSynchronize());

    // Allocate output numpy array with float16 dtype
    std::vector<py::ssize_t> out_shape = {params.N, params.OH, params.OW, params.C};
    py::module_ np = py::module_::import("numpy");
    auto output = np.attr("empty")(out_shape, np.attr("float16")).cast<py::array>();

    CUDA_CHECK(cudaMemcpy(output.mutable_data(), d_output, out_nbytes, cudaMemcpyDeviceToHost));

    return output;
}

// ==================== CUDA synchronize ====================

void cuda_synchronize() {
    CUDA_CHECK(cudaDeviceSynchronize());
}

// ==================== Timed MaxPool2d (CUDA events, kernel-only) ====================

// Helper: launch maxpool kernel for a given version, record CUDA events around it.
// Returns elapsed milliseconds for the kernel only.
static float maxpool_launch_timed(
    const float* d_input, float* d_output, const PoolParams& params,
    int version, int mapping, cudaStream_t stream)
{
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start, stream));
    switch (version) {
        case 0: maxpool_v0(d_input, d_output, params, stream); break;
        case 1: maxpool_v1(d_input, d_output, params, stream); break;
        case 2: maxpool_v2(d_input, d_output, params, stream); break;
        case 3: maxpool_v3(d_input, d_output, params, stream); break;
        case 4: maxpool_v4(d_input, d_output, params, stream); break;
        case 5: maxpool_v5(d_input, d_output, params, stream); break;
        case 6: maxpool_v6(d_input, d_output, params, stream); break;
        case 7: maxpool_v7(d_input, d_output, params, mapping, stream); break;
        case 8: maxpool_v8(d_input, d_output, params, stream); break;
        case 9: maxpool_v9(d_input, d_output, params, stream); break;
        case 10: maxpool_v10(d_input, d_output, params, stream); break;
        case 11: maxpool_v11(d_input, d_output, params, stream); break;
        case 12: maxpool_v12(d_input, d_output, params, stream); break;
        case 13: maxpool_v13(d_input, d_output, params, stream); break;
        case 14: maxpool_v14(d_input, d_output, params, stream); break;
        case 15: maxpool_v15(d_input, d_output, params, stream); break;
        default:
            CUDA_CHECK(cudaEventDestroy(start));
            CUDA_CHECK(cudaEventDestroy(stop));
            throw std::invalid_argument("unsupported kernel version: " + std::to_string(version));
    }
    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return elapsed_ms;
}

static float maxpool_launch_timed(
    const half* d_input, half* d_output, const PoolParams& params,
    int version, int mapping, cudaStream_t stream)
{
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start, stream));
    switch (version) {
        case 0: maxpool_v0(d_input, d_output, params, stream); break;
        case 1: maxpool_v1(d_input, d_output, params, stream); break;
        case 2: maxpool_v2(d_input, d_output, params, stream); break;
        case 3: maxpool_v3(d_input, d_output, params, stream); break;
        case 4: maxpool_v4(d_input, d_output, params, stream); break;
        case 5: maxpool_v5(d_input, d_output, params, stream); break;
        case 6: maxpool_v6(d_input, d_output, params, stream); break;
        case 7: maxpool_v7(d_input, d_output, params, mapping, stream); break;
        case 8: maxpool_v8(d_input, d_output, params, stream); break;
        case 9: maxpool_v9(d_input, d_output, params, stream); break;
        case 10: maxpool_v10(d_input, d_output, params, stream); break;
        case 11: maxpool_v11(d_input, d_output, params, stream); break;
        case 12: maxpool_v12(d_input, d_output, params, stream); break;
        case 13: maxpool_v13(d_input, d_output, params, stream); break;
        case 14: maxpool_v14(d_input, d_output, params, stream); break;
        case 15: maxpool_v15(d_input, d_output, params, stream); break;
        default:
            CUDA_CHECK(cudaEventDestroy(start));
            CUDA_CHECK(cudaEventDestroy(stop));
            throw std::invalid_argument("unsupported kernel version: " + std::to_string(version));
    }
    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return elapsed_ms;
}

py::tuple maxpool2d_timed_f32(
    py::array_t<float, py::array::c_style | py::array::forcecast> input,
    const py::object& kernel_size,
    const py::object& stride,
    const py::object& padding,
    const py::object& dilation,
    bool ceil_mode,
    int version,
    int mapping)
{
    NVTX_RANGE_PUSH("maxpool2d_timed_f32");
    auto buf = input.request();
    if (buf.ndim != 4)
        throw std::invalid_argument("input must be 4-D [N, H, W, C]");

    auto [kh, kw] = parse_pair(kernel_size);
    auto [sh, sw] = stride.is_none() ? std::make_pair(kh, kw) : parse_pair(stride);
    auto [ph, pw] = parse_pair(padding);
    auto [dh, dw] = parse_pair(dilation);

    auto params = make_pool_params(buf.shape, kh, kw, sh, sw, ph, pw, dh, dw, ceil_mode);
    size_t in_nelms = buf.size;
    size_t out_nelms = static_cast<size_t>(params.N * params.OH * params.OW * params.C);

    // Allocate device buffers from arena
    {
        std::lock_guard<std::mutex> lock(arena_mutex);
        arena_f32_in.ensure(in_nelms * sizeof(float));
        arena_f32_out.ensure(out_nelms * sizeof(float));
    }
    float* d_input = static_cast<float*>(arena_f32_in.ptr);
    float* d_output = static_cast<float*>(arena_f32_out.ptr);
    CUDA_CHECK(cudaMemcpy(d_input, buf.ptr, in_nelms * sizeof(float), cudaMemcpyHostToDevice));

    float elapsed_ms = maxpool_launch_timed(d_input, d_output, params, version, mapping, 0);

    std::vector<py::ssize_t> out_shape = {params.N, params.OH, params.OW, params.C};
    auto output = py::array_t<float>(out_shape);
    auto out_buf = output.request();
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(out_buf.ptr, d_output, out_nelms * sizeof(float), cudaMemcpyDeviceToHost));

    NVTX_RANGE_POP();
    return py::make_tuple(output, elapsed_ms);
}

py::tuple maxpool2d_timed_f16(
    py::array input,
    const py::object& kernel_size,
    const py::object& stride,
    const py::object& padding,
    const py::object& dilation,
    bool ceil_mode,
    int version,
    int mapping)
{
    NVTX_RANGE_PUSH("maxpool2d_timed_f16");
    validate_float16(input);
    input = ensure_c_contiguous(input);

    if (input.ndim() != 4)
        throw std::invalid_argument("input must be 4-D [N, H, W, C]");

    auto shape = array_shape(input);
    size_t in_nbytes = input.size() * sizeof(half);

    auto [kh, kw] = parse_pair(kernel_size);
    auto [sh, sw] = stride.is_none() ? std::make_pair(kh, kw) : parse_pair(stride);
    auto [ph, pw] = parse_pair(padding);
    auto [dh, dw] = parse_pair(dilation);

    auto params = make_pool_params(shape, kh, kw, sh, sw, ph, pw, dh, dw, ceil_mode);
    size_t out_nelms = static_cast<size_t>(params.N * params.OH * params.OW * params.C);
    size_t out_nbytes = out_nelms * sizeof(half);

    // Allocate device buffers from arena
    {
        std::lock_guard<std::mutex> lock(arena_mutex);
        arena_f16_in.ensure(in_nbytes);
        arena_f16_out.ensure(out_nbytes);
    }
    half* d_input = static_cast<half*>(arena_f16_in.ptr);
    half* d_output = static_cast<half*>(arena_f16_out.ptr);
    CUDA_CHECK(cudaMemcpy(d_input, input.data(), in_nbytes, cudaMemcpyHostToDevice));

    float elapsed_ms = maxpool_launch_timed(d_input, d_output, params, version, mapping, 0);

    std::vector<py::ssize_t> out_shape = {params.N, params.OH, params.OW, params.C};
    py::module_ np = py::module_::import("numpy");
    auto output = np.attr("empty")(out_shape, np.attr("float16")).cast<py::array>();

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(output.mutable_data(), d_output, out_nbytes, cudaMemcpyDeviceToHost));

    NVTX_RANGE_POP();
    return py::make_tuple(output, elapsed_ms);
}

// ==================== Timed AvgPool2d (CUDA events, kernel-only) ====================

static float avgpool_launch_timed(
    const float* d_input, float* d_output, const AvgPoolParams& params,
    int version, int mapping, cudaStream_t stream)
{
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start, stream));
    switch (version) {
        case 0: avgpool_v0(d_input, d_output, params, stream); break;
        case 1: avgpool_v1(d_input, d_output, params, stream); break;
        case 2: avgpool_v2(d_input, d_output, params, stream); break;
        case 3: avgpool_v3(d_input, d_output, params, stream); break;
        case 4: avgpool_v4(d_input, d_output, params, stream); break;
        case 5: avgpool_v5(d_input, d_output, params, stream); break;
        case 6: avgpool_v6(d_input, d_output, params, stream); break;
        case 7: avgpool_v7(d_input, d_output, params, mapping, stream); break;
        case 8: avgpool_v8(d_input, d_output, params, stream); break;
        case 9: avgpool_v9(d_input, d_output, params, stream); break;
        case 10: avgpool_v10(d_input, d_output, params, stream); break;
        case 11: avgpool_v11(d_input, d_output, params, stream); break;
        case 12: avgpool_v12(d_input, d_output, params, stream); break;
        case 13: avgpool_v13(d_input, d_output, params, stream); break;
        case 14: avgpool_v14(d_input, d_output, params, stream); break;
        case 15: avgpool_v15(d_input, d_output, params, stream); break;
        default:
            CUDA_CHECK(cudaEventDestroy(start));
            CUDA_CHECK(cudaEventDestroy(stop));
            throw std::invalid_argument("unsupported kernel version: " + std::to_string(version));
    }
    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return elapsed_ms;
}

static float avgpool_launch_timed(
    const half* d_input, half* d_output, const AvgPoolParams& params,
    int version, int mapping, cudaStream_t stream)
{
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start, stream));
    switch (version) {
        case 0: avgpool_v0(d_input, d_output, params, stream); break;
        case 1: avgpool_v1(d_input, d_output, params, stream); break;
        case 2: avgpool_v2(d_input, d_output, params, stream); break;
        case 3: avgpool_v3(d_input, d_output, params, stream); break;
        case 4: avgpool_v4(d_input, d_output, params, stream); break;
        case 5: avgpool_v5(d_input, d_output, params, stream); break;
        case 6: avgpool_v6(d_input, d_output, params, stream); break;
        case 7: avgpool_v7(d_input, d_output, params, mapping, stream); break;
        case 8: avgpool_v8(d_input, d_output, params, stream); break;
        case 9: avgpool_v9(d_input, d_output, params, stream); break;
        case 10: avgpool_v10(d_input, d_output, params, stream); break;
        case 11: avgpool_v11(d_input, d_output, params, stream); break;
        case 12: avgpool_v12(d_input, d_output, params, stream); break;
        case 13: avgpool_v13(d_input, d_output, params, stream); break;
        case 14: avgpool_v14(d_input, d_output, params, stream); break;
        case 15: avgpool_v15(d_input, d_output, params, stream); break;
        default:
            CUDA_CHECK(cudaEventDestroy(start));
            CUDA_CHECK(cudaEventDestroy(stop));
            throw std::invalid_argument("unsupported kernel version: " + std::to_string(version));
    }
    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return elapsed_ms;
}

py::tuple avgpool2d_timed_f32(
    py::array_t<float, py::array::c_style | py::array::forcecast> input,
    const py::object& kernel_size,
    const py::object& stride,
    const py::object& padding,
    const py::object& dilation,
    bool ceil_mode,
    bool count_include_pad,
    const py::object& divisor_override,
    int version,
    int mapping)
{
    NVTX_RANGE_PUSH("avgpool2d_timed_f32");
    auto buf = input.request();
    if (buf.ndim != 4)
        throw std::invalid_argument("input must be 4-D [N, H, W, C]");

    auto [kh, kw] = parse_pair(kernel_size);
    auto [sh, sw] = stride.is_none() ? std::make_pair(kh, kw) : parse_pair(stride);
    auto [ph, pw] = parse_pair(padding);
    auto [dh, dw] = parse_pair(dilation);

    int64_t div_over = divisor_override.is_none() ? 0 : divisor_override.cast<int64_t>();
    auto params = make_avgpool_params(buf.shape, kh, kw, sh, sw, ph, pw, dh, dw,
                                      ceil_mode, count_include_pad, div_over);
    size_t in_nelms = buf.size;
    size_t out_nelms = static_cast<size_t>(params.N * params.OH * params.OW * params.C);

    // Allocate device buffers from arena
    {
        std::lock_guard<std::mutex> lock(arena_mutex);
        arena_f32_in.ensure(in_nelms * sizeof(float));
        arena_f32_out.ensure(out_nelms * sizeof(float));
    }
    float* d_input = static_cast<float*>(arena_f32_in.ptr);
    float* d_output = static_cast<float*>(arena_f32_out.ptr);
    CUDA_CHECK(cudaMemcpy(d_input, buf.ptr, in_nelms * sizeof(float), cudaMemcpyHostToDevice));

    float elapsed_ms = avgpool_launch_timed(d_input, d_output, params, version, mapping, 0);

    std::vector<py::ssize_t> out_shape = {params.N, params.OH, params.OW, params.C};
    auto output = py::array_t<float>(out_shape);
    auto out_buf = output.request();
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(out_buf.ptr, d_output, out_nelms * sizeof(float), cudaMemcpyDeviceToHost));

    NVTX_RANGE_POP();
    return py::make_tuple(output, elapsed_ms);
}

py::tuple avgpool2d_timed_f16(
    py::array input,
    const py::object& kernel_size,
    const py::object& stride,
    const py::object& padding,
    const py::object& dilation,
    bool ceil_mode,
    bool count_include_pad,
    const py::object& divisor_override,
    int version,
    int mapping)
{
    NVTX_RANGE_PUSH("avgpool2d_timed_f16");
    validate_float16(input);
    input = ensure_c_contiguous(input);

    if (input.ndim() != 4)
        throw std::invalid_argument("input must be 4-D [N, H, W, C]");

    auto shape = array_shape(input);
    size_t in_nbytes = input.size() * sizeof(half);

    auto [kh, kw] = parse_pair(kernel_size);
    auto [sh, sw] = stride.is_none() ? std::make_pair(kh, kw) : parse_pair(stride);
    auto [ph, pw] = parse_pair(padding);
    auto [dh, dw] = parse_pair(dilation);

    int64_t div_over = divisor_override.is_none() ? 0 : divisor_override.cast<int64_t>();
    auto params = make_avgpool_params(shape, kh, kw, sh, sw, ph, pw, dh, dw,
                                      ceil_mode, count_include_pad, div_over);
    size_t out_nelms = static_cast<size_t>(params.N * params.OH * params.OW * params.C);
    size_t out_nbytes = out_nelms * sizeof(half);

    // Allocate device buffers from arena
    {
        std::lock_guard<std::mutex> lock(arena_mutex);
        arena_f16_in.ensure(in_nbytes);
        arena_f16_out.ensure(out_nbytes);
    }
    half* d_input = static_cast<half*>(arena_f16_in.ptr);
    half* d_output = static_cast<half*>(arena_f16_out.ptr);
    CUDA_CHECK(cudaMemcpy(d_input, input.data(), in_nbytes, cudaMemcpyHostToDevice));

    float elapsed_ms = avgpool_launch_timed(d_input, d_output, params, version, mapping, 0);

    std::vector<py::ssize_t> out_shape = {params.N, params.OH, params.OW, params.C};
    py::module_ np = py::module_::import("numpy");
    auto output = np.attr("empty")(out_shape, np.attr("float16")).cast<py::array>();

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(output.mutable_data(), d_output, out_nbytes, cudaMemcpyDeviceToHost));

    NVTX_RANGE_POP();
    return py::make_tuple(output, elapsed_ms);
}

PYBIND11_MODULE(_pooling, m) {
    m.doc() = "CUDA Pooling2D kernels";

    m.def("maxpool2d_f32", &maxpool2d_f32,
          py::arg("input"),
          py::arg("kernel_size"),
          py::arg("stride") = py::none(),
          py::arg("padding") = 0,
          py::arg("dilation") = 1,
          py::arg("ceil_mode") = false,
          py::arg("version") = 0,
          py::arg("mapping") = 0);

    m.def("maxpool2d_f16", &maxpool2d_f16,
          py::arg("input"),
          py::arg("kernel_size"),
          py::arg("stride") = py::none(),
          py::arg("padding") = 0,
          py::arg("dilation") = 1,
          py::arg("ceil_mode") = false,
          py::arg("version") = 0,
          py::arg("mapping") = 0);

    m.def("avgpool2d_f32", &avgpool2d_f32,
          py::arg("input"),
          py::arg("kernel_size"),
          py::arg("stride") = py::none(),
          py::arg("padding") = 0,
          py::arg("dilation") = 1,
          py::arg("ceil_mode") = true,
          py::arg("count_include_pad") = true,
          py::arg("divisor_override") = py::none(),
          py::arg("version") = 0,
          py::arg("mapping") = 0);

    m.def("avgpool2d_f16", &avgpool2d_f16,
          py::arg("input"),
          py::arg("kernel_size"),
          py::arg("stride") = py::none(),
          py::arg("padding") = 0,
          py::arg("dilation") = 1,
          py::arg("ceil_mode") = true,
          py::arg("count_include_pad") = true,
          py::arg("divisor_override") = py::none(),
          py::arg("version") = 0,
          py::arg("mapping") = 0);

    // CUDA synchronize
    m.def("cuda_synchronize", &cuda_synchronize,
          "Synchronize the CUDA device");

    // Timed variants (return (output, elapsed_ms) tuple)
    m.def("maxpool2d_timed_f32", &maxpool2d_timed_f32,
          py::arg("input"),
          py::arg("kernel_size"),
          py::arg("stride") = py::none(),
          py::arg("padding") = 0,
          py::arg("dilation") = 1,
          py::arg("ceil_mode") = false,
          py::arg("version") = 0,
          py::arg("mapping") = 0);

    m.def("maxpool2d_timed_f16", &maxpool2d_timed_f16,
          py::arg("input"),
          py::arg("kernel_size"),
          py::arg("stride") = py::none(),
          py::arg("padding") = 0,
          py::arg("dilation") = 1,
          py::arg("ceil_mode") = false,
          py::arg("version") = 0,
          py::arg("mapping") = 0);

    m.def("avgpool2d_timed_f32", &avgpool2d_timed_f32,
          py::arg("input"),
          py::arg("kernel_size"),
          py::arg("stride") = py::none(),
          py::arg("padding") = 0,
          py::arg("dilation") = 1,
          py::arg("ceil_mode") = true,
          py::arg("count_include_pad") = true,
          py::arg("divisor_override") = py::none(),
          py::arg("version") = 0,
          py::arg("mapping") = 0);

    m.def("avgpool2d_timed_f16", &avgpool2d_timed_f16,
          py::arg("input"),
          py::arg("kernel_size"),
          py::arg("stride") = py::none(),
          py::arg("padding") = 0,
          py::arg("dilation") = 1,
          py::arg("ceil_mode") = true,
          py::arg("count_include_pad") = true,
          py::arg("divisor_override") = py::none(),
          py::arg("version") = 0,
          py::arg("mapping") = 0);
}
