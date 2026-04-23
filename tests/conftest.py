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

MAXPOOL_VERSIONS = [0, 1]
AVGPOOL_VERSIONS = [0, 1]


def _is_valid_maxpool_padding(kernel_size, padding, dilation):
    """Check if padding is valid for PyTorch max_pool2d.
    PyTorch requires: padding <= dilation * (kernel_size - 1) / 2
    """
    if isinstance(kernel_size, (list, tuple)):
        kh, kw = kernel_size
    else:
        kh = kw = kernel_size
    if isinstance(padding, (list, tuple)):
        ph, pw = padding
    else:
        ph = pw = padding
    if isinstance(dilation, (list, tuple)):
        dh, dw = dilation
    else:
        dh = dw = dilation
    return ph <= dh * (kh - 1) // 2 and pw <= dw * (kw - 1) // 2


def _is_valid_avgpool_padding(kernel_size, padding):
    """Check if padding is valid for PyTorch avg_pool2d.
    PyTorch requires: padding <= kernel_size / 2
    """
    if isinstance(kernel_size, (list, tuple)):
        kh, kw = kernel_size
    else:
        kh = kw = kernel_size
    if isinstance(padding, (list, tuple)):
        ph, pw = padding
    else:
        ph = pw = padding
    return ph <= kh // 2 and pw <= kw // 2


def pytorch_maxpool2d(x_nhwc, kernel_size, stride, padding, dilation, ceil_mode):
    """Golden reference: NHWC numpy -> PyTorch NCHW maxpool2d -> NHWC numpy."""
    dtype = x_nhwc.dtype
    x_nchw = np.ascontiguousarray(x_nhwc.transpose(0, 3, 1, 2))
    t = torch.from_numpy(x_nchw.copy())
    if dtype == np.float16:
        t = t.half()
    try:
        out = torch.nn.functional.max_pool2d(t, kernel_size, stride, padding, dilation, ceil_mode)
    except RuntimeError as e:
        pytest.skip(f"PyTorch rejects this parameter combination: {e}")
    out_np = out.numpy()
    return np.ascontiguousarray(out_np.transpose(0, 2, 3, 1))


def pytorch_avgpool2d(x_nhwc, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override):
    """Golden reference: NHWC numpy -> PyTorch NCHW avgpool2d -> NHWC numpy."""
    dtype = x_nhwc.dtype
    x_nchw = np.ascontiguousarray(x_nhwc.transpose(0, 3, 1, 2))
    t = torch.from_numpy(x_nchw.copy())
    if dtype == np.float16:
        t = t.half()
    try:
        out = torch.nn.functional.avg_pool2d(t, kernel_size, stride, padding, ceil_mode,
                                              count_include_pad, divisor_override)
    except RuntimeError as e:
        pytest.skip(f"PyTorch rejects this parameter combination: {e}")
    out_np = out.numpy()
    return np.ascontiguousarray(out_np.transpose(0, 2, 3, 1))


def call_maxpool2d(x_nhwc, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False, version=0):
    """Call our CUDA maxpool2d."""
    dtype = x_nhwc.dtype
    if dtype == np.float16:
        return _pooling.maxpool2d_f16(x_nhwc, kernel_size, stride, padding, dilation, ceil_mode, version)
    else:
        return _pooling.maxpool2d_f32(x_nhwc, kernel_size, stride, padding, dilation, ceil_mode, version)


def call_avgpool2d(x_nhwc, kernel_size, stride=None, padding=0, ceil_mode=True,
                   count_include_pad=True, divisor_override=None, version=0):
    """Call our CUDA avgpool2d."""
    dtype = x_nhwc.dtype
    if dtype == np.float16:
        return _pooling.avgpool2d_f16(x_nhwc, kernel_size, stride, padding, 1, ceil_mode,
                                      count_include_pad, divisor_override, version)
    else:
        return _pooling.avgpool2d_f32(x_nhwc, kernel_size, stride, padding, 1, ceil_mode,
                                      count_include_pad, divisor_override, version)


def check_close(actual, expected, dtype):
    atol = TOLERANCES[dtype]
    np.testing.assert_allclose(actual, expected, atol=atol, rtol=0)
