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

// ---------------------------------------------------------------------------
// Pre-allocated device memory arena (generic, 2 arenas for in/out)
// ---------------------------------------------------------------------------
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
static DeviceArena arena_in;   // max 512 MB shared across all dtypes
static DeviceArena arena_out;  // max 512 MB shared across all dtypes

static constexpr size_t MAX_BYTES = 512ULL * 1024 * 1024;

// ---------------------------------------------------------------------------
// Parameter parsing helpers
// ---------------------------------------------------------------------------
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

static std::vector<py::ssize_t> array_shape(const py::array& arr) {
    auto ndim = arr.ndim();
    std::vector<py::ssize_t> shape(ndim);
    for (ssize_t i = 0; i < ndim; ++i)
        shape[i] = arr.shape(i);
    return shape;
}

static py::array ensure_c_contiguous(const py::array& arr) {
    if (arr.flags() & py::array::c_style)
        return arr;
    py::module_ np = py::module_::import("numpy");
    return np.attr("ascontiguousarray")(arr).cast<py::array>();
}

static void validate_dtype(const py::array& arr, char expected_kind, size_t expected_size, const char* type_name) {
    if (arr.dtype().kind() != expected_kind || (size_t)arr.dtype().itemsize() != expected_size)
        throw std::invalid_argument(std::string("input must be ") + type_name);
}

// ---------------------------------------------------------------------------
// Version dispatch helpers (templated, overloaded on pointer type)
// ---------------------------------------------------------------------------
template <typename T>
void dispatch_maxpool(const T* d_in, T* d_out, const PoolParams& params, int version, int mapping, cudaStream_t stream) {
    switch (version) {
        case 0: maxpool_v0(d_in, d_out, params, stream); break;
        case 1: maxpool_v1(d_in, d_out, params, stream); break;
        case 2: maxpool_v2(d_in, d_out, params, stream); break;
        case 3: maxpool_v3(d_in, d_out, params, stream); break;
        case 4: maxpool_v4(d_in, d_out, params, stream); break;
        case 5: maxpool_v5(d_in, d_out, params, stream); break;
        case 6: maxpool_v6(d_in, d_out, params, stream); break;
        case 7: maxpool_v7(d_in, d_out, params, mapping, stream); break;
        case 8: maxpool_v8(d_in, d_out, params, stream); break;
        case 9: maxpool_v9(d_in, d_out, params, stream); break;
        case 10: maxpool_v10(d_in, d_out, params, stream); break;
        case 11: maxpool_v11(d_in, d_out, params, stream); break;
        case 12: maxpool_v12(d_in, d_out, params, stream); break;
        case 13: maxpool_v13(d_in, d_out, params, stream); break;
        case 14: maxpool_v14(d_in, d_out, params, stream); break;
        case 15: maxpool_v15(d_in, d_out, params, stream); break;
        default:
            throw std::invalid_argument("unsupported kernel version: " + std::to_string(version));
    }
}

template <typename T>
void dispatch_avgpool(const T* d_in, T* d_out, const AvgPoolParams& params, int version, int mapping, cudaStream_t stream) {
    switch (version) {
        case 0: avgpool_v0(d_in, d_out, params, stream); break;
        case 1: avgpool_v1(d_in, d_out, params, stream); break;
        case 2: avgpool_v2(d_in, d_out, params, stream); break;
        case 3: avgpool_v3(d_in, d_out, params, stream); break;
        case 4: avgpool_v4(d_in, d_out, params, stream); break;
        case 5: avgpool_v5(d_in, d_out, params, stream); break;
        case 6: avgpool_v6(d_in, d_out, params, stream); break;
        case 7: avgpool_v7(d_in, d_out, params, mapping, stream); break;
        case 8: avgpool_v8(d_in, d_out, params, stream); break;
        case 9: avgpool_v9(d_in, d_out, params, stream); break;
        case 10: avgpool_v10(d_in, d_out, params, stream); break;
        case 11: avgpool_v11(d_in, d_out, params, stream); break;
        case 12: avgpool_v12(d_in, d_out, params, stream); break;
        case 13: avgpool_v13(d_in, d_out, params, stream); break;
        case 14: avgpool_v14(d_in, d_out, params, stream); break;
        case 15: avgpool_v15(d_in, d_out, params, stream); break;
        default:
            throw std::invalid_argument("unsupported kernel version: " + std::to_string(version));
    }
}

// ---------------------------------------------------------------------------
// Timed launch helpers (templated)
// ---------------------------------------------------------------------------
template <typename T>
static float maxpool_launch_timed(const T* d_in, T* d_out, const PoolParams& params, int version, int mapping, cudaStream_t stream) {
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start, stream));
    dispatch_maxpool(d_in, d_out, params, version, mapping, stream);
    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return elapsed_ms;
}

template <typename T>
static float avgpool_launch_timed(const T* d_in, T* d_out, const AvgPoolParams& params, int version, int mapping, cudaStream_t stream) {
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start, stream));
    dispatch_avgpool(d_in, d_out, params, version, mapping, stream);
    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return elapsed_ms;
}

// ---------------------------------------------------------------------------
// MaxPool2d (py::array_t<T> variant — for float, int8_t, int16_t)
// ---------------------------------------------------------------------------
template <typename T>
py::array_t<T> maxpool2d_typed(
    py::array_t<T, py::array::c_style | py::array::forcecast> input,
    const py::object& kernel_size, const py::object& stride,
    const py::object& padding, const py::object& dilation,
    bool ceil_mode, int version, int mapping)
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

    {
        std::lock_guard<std::mutex> lock(arena_mutex);
        arena_in.ensure(in_nelms * sizeof(T));
        arena_out.ensure(out_nelms * sizeof(T));
    }
    T* d_input = static_cast<T*>(arena_in.ptr);
    T* d_output = static_cast<T*>(arena_out.ptr);
    CUDA_CHECK(cudaMemcpy(d_input, buf.ptr, in_nelms * sizeof(T), cudaMemcpyHostToDevice));

    dispatch_maxpool(d_input, d_output, params, version, mapping, 0);

    std::vector<py::ssize_t> out_shape = {params.N, params.OH, params.OW, params.C};
    auto output = py::array_t<T>(out_shape);
    auto out_buf = output.request();
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(out_buf.ptr, d_output, out_nelms * sizeof(T), cudaMemcpyDeviceToHost));

    return output;
}

// ---------------------------------------------------------------------------
// MaxPool2d (generic py::array variant — for half, bf16, fp8 where numpy has no native dtype)
// ---------------------------------------------------------------------------
template <typename T>
py::array maxpool2d_raw(
    py::array input,
    const py::object& kernel_size, const py::object& stride,
    const py::object& padding, const py::object& dilation,
    bool ceil_mode, int version, int mapping,
    const char* np_dtype_name)
{
    input = ensure_c_contiguous(input);
    if (input.ndim() != 4)
        throw std::invalid_argument("input must be 4-D [N, H, W, C]");

    auto shape = array_shape(input);
    size_t in_nbytes = input.size() * sizeof(T);

    auto [kh, kw] = parse_pair(kernel_size);
    auto [sh, sw] = stride.is_none() ? std::make_pair(kh, kw) : parse_pair(stride);
    auto [ph, pw] = parse_pair(padding);
    auto [dh, dw] = parse_pair(dilation);

    auto params = make_pool_params(shape, kh, kw, sh, sw, ph, pw, dh, dw, ceil_mode);
    size_t out_nelms = static_cast<size_t>(params.N * params.OH * params.OW * params.C);
    size_t out_nbytes = out_nelms * sizeof(T);

    {
        std::lock_guard<std::mutex> lock(arena_mutex);
        arena_in.ensure(in_nbytes);
        arena_out.ensure(out_nbytes);
    }
    T* d_input = static_cast<T*>(arena_in.ptr);
    T* d_output = static_cast<T*>(arena_out.ptr);
    CUDA_CHECK(cudaMemcpy(d_input, input.data(), in_nbytes, cudaMemcpyHostToDevice));

    dispatch_maxpool(d_input, d_output, params, version, mapping, 0);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<py::ssize_t> out_shape = {params.N, params.OH, params.OW, params.C};
    py::module_ np = py::module_::import("numpy");
    auto output = np.attr("empty")(out_shape, np.attr(np_dtype_name)).cast<py::array>();

    CUDA_CHECK(cudaMemcpy(output.mutable_data(), d_output, out_nbytes, cudaMemcpyDeviceToHost));
    return output;
}

// ---------------------------------------------------------------------------
// AvgPool2d (py::array_t<T> variant)
// ---------------------------------------------------------------------------
template <typename T>
py::array_t<T> avgpool2d_typed(
    py::array_t<T, py::array::c_style | py::array::forcecast> input,
    const py::object& kernel_size, const py::object& stride,
    const py::object& padding, const py::object& dilation,
    bool ceil_mode, bool count_include_pad,
    const py::object& divisor_override, int version, int mapping)
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

    {
        std::lock_guard<std::mutex> lock(arena_mutex);
        arena_in.ensure(in_nelms * sizeof(T));
        arena_out.ensure(out_nelms * sizeof(T));
    }
    T* d_input = static_cast<T*>(arena_in.ptr);
    T* d_output = static_cast<T*>(arena_out.ptr);
    CUDA_CHECK(cudaMemcpy(d_input, buf.ptr, in_nelms * sizeof(T), cudaMemcpyHostToDevice));

    dispatch_avgpool(d_input, d_output, params, version, mapping, 0);

    std::vector<py::ssize_t> out_shape = {params.N, params.OH, params.OW, params.C};
    auto output = py::array_t<T>(out_shape);
    auto out_buf = output.request();
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(out_buf.ptr, d_output, out_nelms * sizeof(T), cudaMemcpyDeviceToHost));

    return output;
}

// ---------------------------------------------------------------------------
// AvgPool2d (generic py::array variant)
// ---------------------------------------------------------------------------
template <typename T>
py::array avgpool2d_raw(
    py::array input,
    const py::object& kernel_size, const py::object& stride,
    const py::object& padding, const py::object& dilation,
    bool ceil_mode, bool count_include_pad,
    const py::object& divisor_override, int version, int mapping,
    const char* np_dtype_name)
{
    input = ensure_c_contiguous(input);
    if (input.ndim() != 4)
        throw std::invalid_argument("input must be 4-D [N, H, W, C]");

    auto shape = array_shape(input);
    size_t in_nbytes = input.size() * sizeof(T);

    auto [kh, kw] = parse_pair(kernel_size);
    auto [sh, sw] = stride.is_none() ? std::make_pair(kh, kw) : parse_pair(stride);
    auto [ph, pw] = parse_pair(padding);
    auto [dh, dw] = parse_pair(dilation);

    int64_t div_over = divisor_override.is_none() ? 0 : divisor_override.cast<int64_t>();
    auto params = make_avgpool_params(shape, kh, kw, sh, sw, ph, pw, dh, dw,
                                      ceil_mode, count_include_pad, div_over);
    size_t out_nelms = static_cast<size_t>(params.N * params.OH * params.OW * params.C);
    size_t out_nbytes = out_nelms * sizeof(T);

    {
        std::lock_guard<std::mutex> lock(arena_mutex);
        arena_in.ensure(in_nbytes);
        arena_out.ensure(out_nbytes);
    }
    T* d_input = static_cast<T*>(arena_in.ptr);
    T* d_output = static_cast<T*>(arena_out.ptr);
    CUDA_CHECK(cudaMemcpy(d_input, input.data(), in_nbytes, cudaMemcpyHostToDevice));

    dispatch_avgpool(d_input, d_output, params, version, mapping, 0);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<py::ssize_t> out_shape = {params.N, params.OH, params.OW, params.C};
    py::module_ np = py::module_::import("numpy");
    auto output = np.attr("empty")(out_shape, np.attr(np_dtype_name)).cast<py::array>();

    CUDA_CHECK(cudaMemcpy(output.mutable_data(), d_output, out_nbytes, cudaMemcpyDeviceToHost));
    return output;
}

// ---------------------------------------------------------------------------
// Concrete function definitions for pybind11
// ---------------------------------------------------------------------------

// --- fp32 ---
py::array_t<float> maxpool2d_f32(
    py::array_t<float, py::array::c_style | py::array::forcecast> input,
    const py::object& kernel_size, const py::object& stride,
    const py::object& padding, const py::object& dilation,
    bool ceil_mode, int version, int mapping)
{
    return maxpool2d_typed<float>(input, kernel_size, stride, padding, dilation, ceil_mode, version, mapping);
}

py::array maxpool2d_f16(
    py::array input,
    const py::object& kernel_size, const py::object& stride,
    const py::object& padding, const py::object& dilation,
    bool ceil_mode, int version, int mapping)
{
    validate_dtype(input, 'f', 2, "float16");
    return maxpool2d_raw<half>(input, kernel_size, stride, padding, dilation, ceil_mode, version, mapping, "float16");
}

py::array_t<float> avgpool2d_f32(
    py::array_t<float, py::array::c_style | py::array::forcecast> input,
    const py::object& kernel_size, const py::object& stride,
    const py::object& padding, const py::object& dilation,
    bool ceil_mode, bool count_include_pad,
    const py::object& divisor_override, int version, int mapping)
{
    return avgpool2d_typed<float>(input, kernel_size, stride, padding, dilation, ceil_mode, count_include_pad, divisor_override, version, mapping);
}

py::array avgpool2d_f16(
    py::array input,
    const py::object& kernel_size, const py::object& stride,
    const py::object& padding, const py::object& dilation,
    bool ceil_mode, bool count_include_pad,
    const py::object& divisor_override, int version, int mapping)
{
    validate_dtype(input, 'f', 2, "float16");
    return avgpool2d_raw<half>(input, kernel_size, stride, padding, dilation, ceil_mode, count_include_pad, divisor_override, version, mapping, "float16");
}

void cuda_synchronize() {
    CUDA_CHECK(cudaDeviceSynchronize());
}

// --- Timed variants (fp32) ---
py::tuple maxpool2d_timed_f32(
    py::array_t<float, py::array::c_style | py::array::forcecast> input,
    const py::object& kernel_size, const py::object& stride,
    const py::object& padding, const py::object& dilation,
    bool ceil_mode, int version, int mapping)
{
    NVTX_RANGE_PUSH("maxpool2d_timed_f32");
    auto buf = input.request();
    if (buf.ndim != 4) throw std::invalid_argument("input must be 4-D [N, H, W, C]");
    auto [kh, kw] = parse_pair(kernel_size);
    auto [sh, sw] = stride.is_none() ? std::make_pair(kh, kw) : parse_pair(stride);
    auto [ph, pw] = parse_pair(padding);
    auto [dh, dw] = parse_pair(dilation);
    auto params = make_pool_params(buf.shape, kh, kw, sh, sw, ph, pw, dh, dw, ceil_mode);
    size_t in_nelms = buf.size;
    size_t out_nelms = static_cast<size_t>(params.N * params.OH * params.OW * params.C);
    { std::lock_guard<std::mutex> lock(arena_mutex); arena_in.ensure(in_nelms * sizeof(float)); arena_out.ensure(out_nelms * sizeof(float)); }
    float* d_input = static_cast<float*>(arena_in.ptr);
    float* d_output = static_cast<float*>(arena_out.ptr);
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

py::tuple avgpool2d_timed_f32(
    py::array_t<float, py::array::c_style | py::array::forcecast> input,
    const py::object& kernel_size, const py::object& stride,
    const py::object& padding, const py::object& dilation,
    bool ceil_mode, bool count_include_pad,
    const py::object& divisor_override, int version, int mapping)
{
    NVTX_RANGE_PUSH("avgpool2d_timed_f32");
    auto buf = input.request();
    if (buf.ndim != 4) throw std::invalid_argument("input must be 4-D [N, H, W, C]");
    auto [kh, kw] = parse_pair(kernel_size);
    auto [sh, sw] = stride.is_none() ? std::make_pair(kh, kw) : parse_pair(stride);
    auto [ph, pw] = parse_pair(padding);
    auto [dh, dw] = parse_pair(dilation);
    int64_t div_over = divisor_override.is_none() ? 0 : divisor_override.cast<int64_t>();
    auto params = make_avgpool_params(buf.shape, kh, kw, sh, sw, ph, pw, dh, dw, ceil_mode, count_include_pad, div_over);
    size_t in_nelms = buf.size;
    size_t out_nelms = static_cast<size_t>(params.N * params.OH * params.OW * params.C);
    { std::lock_guard<std::mutex> lock(arena_mutex); arena_in.ensure(in_nelms * sizeof(float)); arena_out.ensure(out_nelms * sizeof(float)); }
    float* d_input = static_cast<float*>(arena_in.ptr);
    float* d_output = static_cast<float*>(arena_out.ptr);
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

// ---------------------------------------------------------------------------
// PYBIND11_MODULE
// ---------------------------------------------------------------------------
PYBIND11_MODULE(_pooling, m) {
    m.doc() = "CUDA Pooling2D kernels";

    m.def("maxpool2d_f32", &maxpool2d_f32,
          py::arg("input"), py::arg("kernel_size"), py::arg("stride") = py::none(),
          py::arg("padding") = 0, py::arg("dilation") = 1, py::arg("ceil_mode") = false,
          py::arg("version") = 0, py::arg("mapping") = 0);

    m.def("maxpool2d_f16", &maxpool2d_f16,
          py::arg("input"), py::arg("kernel_size"), py::arg("stride") = py::none(),
          py::arg("padding") = 0, py::arg("dilation") = 1, py::arg("ceil_mode") = false,
          py::arg("version") = 0, py::arg("mapping") = 0);

    m.def("avgpool2d_f32", &avgpool2d_f32,
          py::arg("input"), py::arg("kernel_size"), py::arg("stride") = py::none(),
          py::arg("padding") = 0, py::arg("dilation") = 1, py::arg("ceil_mode") = true,
          py::arg("count_include_pad") = true, py::arg("divisor_override") = py::none(),
          py::arg("version") = 0, py::arg("mapping") = 0);

    m.def("avgpool2d_f16", &avgpool2d_f16,
          py::arg("input"), py::arg("kernel_size"), py::arg("stride") = py::none(),
          py::arg("padding") = 0, py::arg("dilation") = 1, py::arg("ceil_mode") = true,
          py::arg("count_include_pad") = true, py::arg("divisor_override") = py::none(),
          py::arg("version") = 0, py::arg("mapping") = 0);

    m.def("cuda_synchronize", &cuda_synchronize, "Synchronize the CUDA device");

    m.def("maxpool2d_timed_f32", &maxpool2d_timed_f32,
          py::arg("input"), py::arg("kernel_size"), py::arg("stride") = py::none(),
          py::arg("padding") = 0, py::arg("dilation") = 1, py::arg("ceil_mode") = false,
          py::arg("version") = 0, py::arg("mapping") = 0);

    m.def("avgpool2d_timed_f32", &avgpool2d_timed_f32,
          py::arg("input"), py::arg("kernel_size"), py::arg("stride") = py::none(),
          py::arg("padding") = 0, py::arg("dilation") = 1, py::arg("ceil_mode") = true,
          py::arg("count_include_pad") = true, py::arg("divisor_override") = py::none(),
          py::arg("version") = 0, py::arg("mapping") = 0);
}
