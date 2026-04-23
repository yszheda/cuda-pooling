import numpy as np
import pytest
from conftest import (
    pytorch_avgpool2d, call_avgpool2d, check_close,
    AVGPOOL_VERSIONS, TOLERANCES,
)


# ---------- helpers ----------

def _rand_nhwc(N, H, W, C, dtype=np.float32):
    """Generate random NHWC array."""
    if dtype == np.float16:
        return np.random.randn(N, H, W, C).astype(np.float16)
    return np.random.randn(N, H, W, C).astype(np.float32)


# ---------- basic kernel_size, stride, padding combos ----------

@pytest.mark.parametrize("version", AVGPOOL_VERSIONS)
@pytest.mark.parametrize("dtype", [np.float32, np.float16])
@pytest.mark.parametrize("kernel_size", [2, 3, 4, (3, 2)])
@pytest.mark.parametrize("stride,padding", [
    (1, 0), (2, 0), (2, 1),
])
@pytest.mark.parametrize("ceil_mode", [False, True])
@pytest.mark.parametrize("count_include_pad", [True, False])
def test_basic(version, dtype, kernel_size, stride, padding, ceil_mode, count_include_pad):
    x = _rand_nhwc(2, 16, 16, 4, dtype)
    expected = pytorch_avgpool2d(x, kernel_size, stride, padding, ceil_mode, count_include_pad, None)
    actual = call_avgpool2d(x, kernel_size, stride, padding, ceil_mode, count_include_pad, None, version)
    check_close(actual, expected, dtype)


# ---------- stride=None defaults to kernel_size ----------

@pytest.mark.parametrize("version", AVGPOOL_VERSIONS)
@pytest.mark.parametrize("dtype", [np.float32, np.float16])
@pytest.mark.parametrize("kernel_size", [2, 3, (3, 2)])
def test_stride_none(version, dtype, kernel_size):
    x = _rand_nhwc(2, 16, 16, 4, dtype)
    expected = pytorch_avgpool2d(x, kernel_size, None, 0, False, True, None)
    actual = call_avgpool2d(x, kernel_size, None, 0, False, True, None, version)
    check_close(actual, expected, dtype)


# ---------- divisor_override ----------

@pytest.mark.parametrize("version", AVGPOOL_VERSIONS)
@pytest.mark.parametrize("dtype", [np.float32, np.float16])
@pytest.mark.parametrize("divisor_override", [1, 2, 3, 6, 9])
def test_divisor_override(version, dtype, divisor_override):
    x = _rand_nhwc(2, 8, 8, 4, dtype)
    expected = pytorch_avgpool2d(x, 3, 2, 1, False, True, divisor_override)
    actual = call_avgpool2d(x, 3, 2, 1, False, True, divisor_override, version)
    check_close(actual, expected, dtype)


# ---------- global pooling ----------

@pytest.mark.parametrize("version", AVGPOOL_VERSIONS)
@pytest.mark.parametrize("dtype", [np.float32, np.float16])
@pytest.mark.parametrize("H,W", [(8, 8), (32, 32), (64, 64)])
def test_global_pooling(version, dtype, H, W):
    x = _rand_nhwc(2, H, W, 4, dtype)
    expected = pytorch_avgpool2d(x, (H, W), None, 0, False, True, None)
    actual = call_avgpool2d(x, (H, W), None, 0, False, True, None, version)
    check_close(actual, expected, dtype)


# ---------- kernel_size == input size ----------

@pytest.mark.parametrize("version", AVGPOOL_VERSIONS)
@pytest.mark.parametrize("dtype", [np.float32, np.float16])
def test_kernel_equals_input(version, dtype):
    x = _rand_nhwc(2, 8, 8, 3, dtype)
    expected = pytorch_avgpool2d(x, 8, None, 0, False, True, None)
    actual = call_avgpool2d(x, 8, None, 0, False, True, None, version)
    check_close(actual, expected, dtype)


# ---------- count_include_pad=False specifically ----------

@pytest.mark.parametrize("version", AVGPOOL_VERSIONS)
@pytest.mark.parametrize("dtype", [np.float32, np.float16])
def test_count_include_pad_false(version, dtype):
    x = _rand_nhwc(2, 7, 7, 4, dtype)
    expected = pytorch_avgpool2d(x, 3, 2, 1, False, False, None)
    actual = call_avgpool2d(x, 3, 2, 1, False, False, None, version)
    check_close(actual, expected, dtype)


# ---------- count_include_pad=False with more padding ----------
# PyTorch requires padding <= kernel_size // 2. For k=3: max p=1

@pytest.mark.parametrize("version", AVGPOOL_VERSIONS)
@pytest.mark.parametrize("dtype", [np.float32, np.float16])
def test_count_include_pad_false_with_padding(version, dtype):
    x = _rand_nhwc(1, 5, 5, 3, dtype)
    expected = pytorch_avgpool2d(x, 3, 1, 1, False, False, None)
    actual = call_avgpool2d(x, 3, 1, 1, False, False, None, version)
    check_close(actual, expected, dtype)


# ---------- large padding ----------
# PyTorch requires padding <= kernel_size // 2. For k=7: max p=3

@pytest.mark.parametrize("version", AVGPOOL_VERSIONS)
@pytest.mark.parametrize("dtype", [np.float32, np.float16])
def test_large_padding(version, dtype):
    x = _rand_nhwc(1, 8, 8, 3, dtype)
    # k=7, p=3 is valid (3 <= 7//2 = 3) and produces larger output with stride=1
    expected = pytorch_avgpool2d(x, 7, 1, 3, False, True, None)
    actual = call_avgpool2d(x, 7, 1, 3, False, True, None, version)
    check_close(actual, expected, dtype)


# ---------- non-square kernel/stride/padding ----------

@pytest.mark.parametrize("version", AVGPOOL_VERSIONS)
@pytest.mark.parametrize("dtype", [np.float32, np.float16])
def test_nonsquare_params(version, dtype):
    x = _rand_nhwc(2, 16, 20, 4, dtype)
    # (3,5) with padding (1,2): 1 <= 3//2=1 and 2 <= 5//2=2, valid
    expected = pytorch_avgpool2d(x, (3, 5), (2, 3), (1, 2), False, True, None)
    actual = call_avgpool2d(x, (3, 5), (2, 3), (1, 2), False, True, None, version)
    check_close(actual, expected, dtype)


# ---------- ceil_mode ----------

@pytest.mark.parametrize("version", AVGPOOL_VERSIONS)
@pytest.mark.parametrize("dtype", [np.float32, np.float16])
def test_ceil_mode_true(version, dtype):
    x = _rand_nhwc(2, 7, 7, 4, dtype)
    expected = pytorch_avgpool2d(x, 3, 2, 1, True, True, None)
    actual = call_avgpool2d(x, 3, 2, 1, True, True, None, version)
    check_close(actual, expected, dtype)


@pytest.mark.parametrize("version", AVGPOOL_VERSIONS)
@pytest.mark.parametrize("dtype", [np.float32, np.float16])
def test_ceil_mode_false(version, dtype):
    x = _rand_nhwc(2, 7, 7, 4, dtype)
    expected = pytorch_avgpool2d(x, 3, 2, 1, False, True, None)
    actual = call_avgpool2d(x, 3, 2, 1, False, True, None, version)
    check_close(actual, expected, dtype)


# ---------- ceil_mode with count_include_pad=False ----------

@pytest.mark.parametrize("version", AVGPOOL_VERSIONS)
@pytest.mark.parametrize("dtype", [np.float32, np.float16])
def test_ceil_mode_with_count_include_pad_false(version, dtype):
    x = _rand_nhwc(2, 5, 5, 3, dtype)
    expected = pytorch_avgpool2d(x, 2, 2, 0, True, False, None)
    actual = call_avgpool2d(x, 2, 2, 0, True, False, None, version)
    check_close(actual, expected, dtype)


# ---------- ceil_mode with padding and count_include_pad ----------

@pytest.mark.parametrize("version", AVGPOOL_VERSIONS)
@pytest.mark.parametrize("dtype", [np.float32, np.float16])
@pytest.mark.parametrize("count_include_pad", [True, False])
def test_ceil_mode_with_padding(version, dtype, count_include_pad):
    x = _rand_nhwc(2, 7, 7, 4, dtype)
    expected = pytorch_avgpool2d(x, 3, 2, 1, True, count_include_pad, None)
    actual = call_avgpool2d(x, 3, 2, 1, True, count_include_pad, None, version)
    check_close(actual, expected, dtype)


# ---------- divisor_override with ceil_mode ----------

@pytest.mark.parametrize("version", AVGPOOL_VERSIONS)
@pytest.mark.parametrize("dtype", [np.float32, np.float16])
def test_divisor_override_ceil_mode(version, dtype):
    x = _rand_nhwc(2, 7, 7, 4, dtype)
    expected = pytorch_avgpool2d(x, 3, 2, 1, True, True, 9)
    actual = call_avgpool2d(x, 3, 2, 1, True, True, 9, version)
    check_close(actual, expected, dtype)


# ---------- various spatial sizes ----------

@pytest.mark.parametrize("version", AVGPOOL_VERSIONS)
@pytest.mark.parametrize("dtype", [np.float32, np.float16])
@pytest.mark.parametrize("H,W", [(7, 7), (15, 13), (17, 31), (64, 48)])
def test_various_spatial_sizes(version, dtype, H, W):
    x = _rand_nhwc(1, H, W, 3, dtype)
    expected = pytorch_avgpool2d(x, 3, 2, 1, False, True, None)
    actual = call_avgpool2d(x, 3, 2, 1, False, True, None, version)
    check_close(actual, expected, dtype)


# ---------- batch size 1 ----------

@pytest.mark.parametrize("version", AVGPOOL_VERSIONS)
@pytest.mark.parametrize("dtype", [np.float32, np.float16])
def test_batch1(version, dtype):
    x = _rand_nhwc(1, 8, 8, 3, dtype)
    expected = pytorch_avgpool2d(x, 2, 2, 0, False, True, None)
    actual = call_avgpool2d(x, 2, 2, 0, False, True, None, version)
    check_close(actual, expected, dtype)


# ---------- larger batch ----------

@pytest.mark.parametrize("version", AVGPOOL_VERSIONS)
@pytest.mark.parametrize("dtype", [np.float32, np.float16])
def test_large_batch(version, dtype):
    x = _rand_nhwc(8, 8, 8, 3, dtype)
    expected = pytorch_avgpool2d(x, 2, 2, 0, False, True, None)
    actual = call_avgpool2d(x, 2, 2, 0, False, True, None, version)
    check_close(actual, expected, dtype)
