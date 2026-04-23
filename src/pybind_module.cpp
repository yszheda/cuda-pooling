#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <utility>
#include <vector>
#include <stdexcept>
#include <string>
#include "pooling.cuh"

namespace py = pybind11;

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

template <typename T>
py::array_t<T> maxpool2d_impl(
    py::array_t<T, py::array::c_style | py::array::forcecast> input,
    const py::object& kernel_size,
    const py::object& stride,
    const py::object& padding,
    const py::object& dilation,
    bool ceil_mode,
    int version
) {
    auto buf = input.request();
    if (buf.ndim != 4)
        throw std::invalid_argument("input must be 4-D [N, H, W, C]");

    auto [kh, kw] = parse_pair(kernel_size);
    auto [sh, sw] = stride.is_none() ? std::make_pair(kh, kw) : parse_pair(stride);
    auto [ph, pw] = parse_pair(padding);
    auto [dh, dw] = parse_pair(dilation);

    PoolParams params;
    params.N = buf.shape[0];
    params.H = buf.shape[1];
    params.W = buf.shape[2];
    params.C = buf.shape[3];
    params.kh = kh; params.kw = kw;
    params.sh = sh; params.sw = sw;
    params.ph = ph; params.pw = pw;
    params.dh = dh; params.dw = dw;
    params.ceil_mode = ceil_mode;
    params.compute_output_size();

    std::vector<py::ssize_t> out_shape = {params.N, params.OH, params.OW, params.C};
    auto output = py::array_t<T>(out_shape);
    auto out_buf = output.request();

    switch (version) {
        case 0:
            maxpool_v0(static_cast<const T*>(buf.ptr),
                       static_cast<T*>(out_buf.ptr),
                       params, 0);
            break;
        default:
            throw std::invalid_argument("unsupported kernel version: " + std::to_string(version));
    }

    return output;
}

template <typename T>
py::array_t<T> avgpool2d_impl(
    py::array_t<T, py::array::c_style | py::array::forcecast> input,
    const py::object& kernel_size,
    const py::object& stride,
    const py::object& padding,
    const py::object& dilation,
    bool ceil_mode,
    bool count_include_pad,
    int64_t divisor_override,
    int version
) {
    auto buf = input.request();
    if (buf.ndim != 4)
        throw std::invalid_argument("input must be 4-D [N, H, W, C]");

    auto [kh, kw] = parse_pair(kernel_size);
    auto [sh, sw] = stride.is_none() ? std::make_pair(kh, kw) : parse_pair(stride);
    auto [ph, pw] = parse_pair(padding);
    auto [dh, dw] = parse_pair(dilation);

    AvgPoolParams params;
    params.N = buf.shape[0];
    params.H = buf.shape[1];
    params.W = buf.shape[2];
    params.C = buf.shape[3];
    params.kh = kh; params.kw = kw;
    params.sh = sh; params.sw = sw;
    params.ph = ph; params.pw = pw;
    params.dh = dh; params.dw = dw;
    params.ceil_mode = ceil_mode;
    params.count_include_pad = count_include_pad;
    params.divisor_override = divisor_override;
    params.compute_output_size();

    std::vector<py::ssize_t> out_shape = {params.N, params.OH, params.OW, params.C};
    auto output = py::array_t<T>(out_shape);
    auto out_buf = output.request();

    switch (version) {
        case 0:
            avgpool_v0(static_cast<const T*>(buf.ptr),
                       static_cast<T*>(out_buf.ptr),
                       params, 0);
            break;
        default:
            throw std::invalid_argument("unsupported kernel version: " + std::to_string(version));
    }

    return output;
}

PYBIND11_MODULE(_pooling, m) {
    m.doc() = "CUDA Pooling2D kernels";

    m.def("maxpool2d_f32", &maxpool2d_impl<float>,
          py::arg("input"),
          py::arg("kernel_size"),
          py::arg("stride") = py::none(),
          py::arg("padding") = 0,
          py::arg("dilation") = 1,
          py::arg("ceil_mode") = false,
          py::arg("version") = 0);

    m.def("maxpool2d_f16", &maxpool2d_impl<half>,
          py::arg("input"),
          py::arg("kernel_size"),
          py::arg("stride") = py::none(),
          py::arg("padding") = 0,
          py::arg("dilation") = 1,
          py::arg("ceil_mode") = false,
          py::arg("version") = 0);

    m.def("avgpool2d_f32", &avgpool2d_impl<float>,
          py::arg("input"),
          py::arg("kernel_size"),
          py::arg("stride") = py::none(),
          py::arg("padding") = 0,
          py::arg("dilation") = 1,
          py::arg("ceil_mode") = false,
          py::arg("count_include_pad") = true,
          py::arg("divisor_override") = 0,
          py::arg("version") = 0);

    m.def("avgpool2d_f16", &avgpool2d_impl<half>,
          py::arg("input"),
          py::arg("kernel_size"),
          py::arg("stride") = py::none(),
          py::arg("padding") = 0,
          py::arg("dilation") = 1,
          py::arg("ceil_mode") = false,
          py::arg("count_include_pad") = true,
          py::arg("divisor_override") = 0,
          py::arg("version") = 0);
}
