import numpy as np
import pytest
from conftest import (
    pytorch_maxpool2d, call_maxpool2d, check_close,
    MAXPOOL_VERSIONS, TOLERANCES, MAPPING_VERSIONS,
)


# ---------- helpers ----------

def _rand_nhwc(N, H, W, C, dtype=np.float32):
    """Generate random NHWC array."""
    if dtype == np.float16:
        return np.random.randn(N, H, W, C).astype(np.float16)
    return np.random.randn(N, H, W, C).astype(np.float32)


# ---------- basic kernel_size, stride, padding combos ----------

@pytest.mark.parametrize("version", MAXPOOL_VERSIONS)
@pytest.mark.parametrize("dtype", [np.float32, np.float16])
@pytest.mark.parametrize("kernel_size", [1, 2, 3, 5, 7, (3, 2), (2, 3)])
@pytest.mark.parametrize("stride,padding", [
    (1, 0), (2, 0), (2, 1), (3, 1),
])
def test_basic(version, dtype, kernel_size, stride, padding):
    x = _rand_nhwc(2, 32, 32, 4, dtype)
    expected = pytorch_maxpool2d(x, kernel_size, stride, padding, 1, False)
    actual = call_maxpool2d(x, kernel_size, stride, padding, 1, False, version)
    check_close(actual, expected, dtype)


# ---------- stride=None defaults to kernel_size ----------

@pytest.mark.parametrize("version", MAXPOOL_VERSIONS)
@pytest.mark.parametrize("dtype", [np.float32, np.float16])
@pytest.mark.parametrize("kernel_size", [2, 3, 5, (3, 2)])
def test_stride_none(version, dtype, kernel_size):
    x = _rand_nhwc(2, 16, 16, 4, dtype)
    expected = pytorch_maxpool2d(x, kernel_size, None, 0, 1, False)
    actual = call_maxpool2d(x, kernel_size, None, 0, 1, False, version)
    check_close(actual, expected, dtype)


# ---------- dilation ----------

@pytest.mark.parametrize("version", MAXPOOL_VERSIONS)
@pytest.mark.parametrize("dtype", [np.float32, np.float16])
@pytest.mark.parametrize("dilation", [1, 2, 3])
def test_dilation(version, dtype, dilation):
    x = _rand_nhwc(2, 32, 32, 4, dtype)
    # Use padding that is valid for all dilations: p=1 works for dilation=1,2,3 with k=3
    padding = 1
    expected = pytorch_maxpool2d(x, 3, 2, padding, dilation, False)
    actual = call_maxpool2d(x, 3, 2, padding, dilation, False, version)
    check_close(actual, expected, dtype)


# ---------- ceil_mode=True ----------

@pytest.mark.parametrize("version", MAXPOOL_VERSIONS)
@pytest.mark.parametrize("dtype", [np.float32, np.float16])
def test_ceil_mode(version, dtype):
    x = _rand_nhwc(2, 7, 7, 4, dtype)
    expected = pytorch_maxpool2d(x, 3, 2, 1, 1, True)
    actual = call_maxpool2d(x, 3, 2, 1, 1, True, version)
    check_close(actual, expected, dtype)


@pytest.mark.parametrize("version", MAXPOOL_VERSIONS)
@pytest.mark.parametrize("dtype", [np.float32, np.float16])
def test_ceil_mode_extra_output(version, dtype):
    """ceil_mode=True that produces an extra output row/column."""
    x = _rand_nhwc(1, 5, 5, 3, dtype)
    expected = pytorch_maxpool2d(x, 2, 2, 0, 1, True)
    actual = call_maxpool2d(x, 2, 2, 0, 1, True, version)
    check_close(actual, expected, dtype)


# ---------- global pooling ----------

@pytest.mark.parametrize("version", MAXPOOL_VERSIONS)
@pytest.mark.parametrize("dtype", [np.float32, np.float16])
@pytest.mark.parametrize("H,W", [(8, 8), (32, 32), (64, 64)])
def test_global_pooling(version, dtype, H, W):
    x = _rand_nhwc(2, H, W, 4, dtype)
    expected = pytorch_maxpool2d(x, (H, W), None, 0, 1, False)
    actual = call_maxpool2d(x, (H, W), None, 0, 1, False, version)
    check_close(actual, expected, dtype)


# ---------- kernel_size == input size (no padding) ----------

@pytest.mark.parametrize("version", MAXPOOL_VERSIONS)
@pytest.mark.parametrize("dtype", [np.float32, np.float16])
def test_kernel_equals_input(version, dtype):
    x = _rand_nhwc(2, 8, 8, 3, dtype)
    expected = pytorch_maxpool2d(x, 8, None, 0, 1, False)
    actual = call_maxpool2d(x, 8, None, 0, 1, False, version)
    check_close(actual, expected, dtype)


# ---------- large padding (output > input) ----------
# PyTorch requires padding <= dilation * (kernel_size - 1) / 2
# For kernel=7, dilation=1: max padding = 3
# For kernel=5, dilation=3: max padding = 6

@pytest.mark.parametrize("version", MAXPOOL_VERSIONS)
@pytest.mark.parametrize("dtype", [np.float32, np.float16])
def test_large_padding(version, dtype):
    x = _rand_nhwc(1, 8, 8, 3, dtype)
    # Use k=7, p=3 which is valid (3 <= 1*(7-1)/2 = 3) and produces larger output
    expected = pytorch_maxpool2d(x, 7, 1, 3, 1, False)
    actual = call_maxpool2d(x, 7, 1, 3, 1, False, version)
    check_close(actual, expected, dtype)


# ---------- non-square kernel/stride/padding ----------

@pytest.mark.parametrize("version", MAXPOOL_VERSIONS)
@pytest.mark.parametrize("dtype", [np.float32, np.float16])
def test_nonsquare_params(version, dtype):
    x = _rand_nhwc(2, 16, 20, 4, dtype)
    expected = pytorch_maxpool2d(x, (3, 5), (2, 3), (1, 2), 1, False)
    actual = call_maxpool2d(x, (3, 5), (2, 3), (1, 2), 1, False, version)
    check_close(actual, expected, dtype)


# ---------- dilation with padding ----------

@pytest.mark.parametrize("version", MAXPOOL_VERSIONS)
@pytest.mark.parametrize("dtype", [np.float32, np.float16])
def test_dilation_with_padding(version, dtype):
    x = _rand_nhwc(2, 16, 16, 4, dtype)
    # dilation=2, kernel=3: max padding = 2*(3-1)/2 = 2, so p=2 is valid
    expected = pytorch_maxpool2d(x, 3, 2, 2, 2, False)
    actual = call_maxpool2d(x, 3, 2, 2, 2, False, version)
    check_close(actual, expected, dtype)


# ---------- various spatial sizes ----------

@pytest.mark.parametrize("version", MAXPOOL_VERSIONS)
@pytest.mark.parametrize("dtype", [np.float32, np.float16])
@pytest.mark.parametrize("H,W", [(7, 7), (15, 13), (17, 31), (64, 48)])
def test_various_spatial_sizes(version, dtype, H, W):
    x = _rand_nhwc(1, H, W, 3, dtype)
    expected = pytorch_maxpool2d(x, 3, 2, 1, 1, False)
    actual = call_maxpool2d(x, 3, 2, 1, 1, False, version)
    check_close(actual, expected, dtype)


# ---------- batch size 1 ----------

@pytest.mark.parametrize("version", MAXPOOL_VERSIONS)
@pytest.mark.parametrize("dtype", [np.float32, np.float16])
def test_batch1(version, dtype):
    x = _rand_nhwc(1, 8, 8, 3, dtype)
    expected = pytorch_maxpool2d(x, 2, 2, 0, 1, False)
    actual = call_maxpool2d(x, 2, 2, 0, 1, False, version)
    check_close(actual, expected, dtype)


# ---------- larger batch ----------

@pytest.mark.parametrize("version", MAXPOOL_VERSIONS)
@pytest.mark.parametrize("dtype", [np.float32, np.float16])
def test_large_batch(version, dtype):
    x = _rand_nhwc(8, 8, 8, 3, dtype)
    expected = pytorch_maxpool2d(x, 2, 2, 0, 1, False)
    actual = call_maxpool2d(x, 2, 2, 0, 1, False, version)
    check_close(actual, expected, dtype)


# ---------- v7 mapping-specific tests ----------

@pytest.mark.parametrize("mapping", MAPPING_VERSIONS)
@pytest.mark.parametrize("dtype", [np.float32, np.float16])
@pytest.mark.parametrize("kernel_size", [2, 3, (3, 2)])
@pytest.mark.parametrize("stride,padding", [(1, 0), (2, 0), (2, 1)])
def test_v7_mapping_basic(mapping, dtype, kernel_size, stride, padding):
    x = _rand_nhwc(2, 32, 32, 8, dtype)
    expected = pytorch_maxpool2d(x, kernel_size, stride, padding, 1, False)
    actual = call_maxpool2d(x, kernel_size, stride, padding, 1, False, version=7, mapping=mapping)
    check_close(actual, expected, dtype)


@pytest.mark.parametrize("mapping", MAPPING_VERSIONS)
@pytest.mark.parametrize("dtype", [np.float32, np.float16])
def test_v7_mapping_ceil_mode(mapping, dtype):
    x = _rand_nhwc(2, 7, 7, 4, dtype)
    expected = pytorch_maxpool2d(x, 3, 2, 1, 1, True)
    actual = call_maxpool2d(x, 3, 2, 1, 1, True, version=7, mapping=mapping)
    check_close(actual, expected, dtype)


@pytest.mark.parametrize("mapping", MAPPING_VERSIONS)
@pytest.mark.parametrize("dtype", [np.float32, np.float16])
def test_v7_mapping_dilation(mapping, dtype):
    x = _rand_nhwc(2, 32, 32, 8, dtype)
    expected = pytorch_maxpool2d(x, 3, 2, 1, 2, False)
    actual = call_maxpool2d(x, 3, 2, 1, 2, False, version=7, mapping=mapping)
    check_close(actual, expected, dtype)


@pytest.mark.parametrize("mapping", MAPPING_VERSIONS)
@pytest.mark.parametrize("dtype", [np.float32, np.float16])
def test_v7_mapping_nonsquare(mapping, dtype):
    x = _rand_nhwc(2, 16, 20, 8, dtype)
    expected = pytorch_maxpool2d(x, (3, 5), (2, 3), (1, 2), 1, False)
    actual = call_maxpool2d(x, (3, 5), (2, 3), (1, 2), 1, False, version=7, mapping=mapping)
    check_close(actual, expected, dtype)


@pytest.mark.parametrize("mapping", MAPPING_VERSIONS)
@pytest.mark.parametrize("dtype", [np.float32, np.float16])
def test_v7_mapping_large_C(mapping, dtype):
    """Test with larger channel count to exercise mapping C and D."""
    x = _rand_nhwc(1, 8, 8, 64, dtype)
    expected = pytorch_maxpool2d(x, 2, 2, 0, 1, False)
    actual = call_maxpool2d(x, 2, 2, 0, 1, False, version=7, mapping=mapping)
    check_close(actual, expected, dtype)


@pytest.mark.parametrize("mapping", MAPPING_VERSIONS)
@pytest.mark.parametrize("dtype", [np.float32, np.float16])
def test_v7_mapping_odd_C(mapping, dtype):
    """Test with odd C — mapping D should fall back to A."""
    x = _rand_nhwc(1, 8, 8, 3, dtype)
    expected = pytorch_maxpool2d(x, 2, 2, 0, 1, False)
    actual = call_maxpool2d(x, 2, 2, 0, 1, False, version=7, mapping=mapping)
    check_close(actual, expected, dtype)


# ---------- v9 TMA pipeline — SM90+ only ----------

@pytest.mark.skipif("not is_sm90_or_newer()")
@pytest.mark.parametrize("dtype", [np.float32, np.float16])
def test_v9_basic(dtype):
    x = _rand_nhwc(1, 8, 8, 3, dtype)
    expected = pytorch_maxpool2d(x, 2, 2, 0, 1, False)
    actual = call_maxpool2d(x, 2, 2, 0, 1, False, version=9)
    check_close(actual, expected, dtype)


@pytest.mark.skipif("not is_sm90_or_newer()")
@pytest.mark.parametrize("dtype", [np.float32, np.float16])
def test_v9_stride2(dtype):
    x = _rand_nhwc(1, 32, 32, 8, dtype)
    expected = pytorch_maxpool2d(x, 3, 2, 0, 1, False)
    actual = call_maxpool2d(x, 3, 2, 0, 1, False, version=9)
    check_close(actual, expected, dtype)


@pytest.mark.skipif("not is_sm90_or_newer()")
@pytest.mark.parametrize("dtype", [np.float32, np.float16])
def test_v9_batch(dtype):
    x = _rand_nhwc(4, 16, 16, 4, dtype)
    expected = pytorch_maxpool2d(x, 2, 2, 0, 1, False)
    actual = call_maxpool2d(x, 2, 2, 0, 1, False, version=9)
    check_close(actual, expected, dtype)
