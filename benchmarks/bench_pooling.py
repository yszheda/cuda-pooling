"""Benchmark suite for CUDA Pooling2D kernels.

Covers synthetic shapes and real CNN model configurations.
Runs all kernel versions (v0-v7) for comparison.
"""

import numpy as np
import time
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'build'))
import _pooling


def call_maxpool(x, kernel_size, stride=None, padding=0, dilation=1,
                 ceil_mode=False, version=0, mapping=0):
    dtype = x.dtype
    fn = _pooling.maxpool2d_f16 if dtype == np.float16 else _pooling.maxpool2d_f32
    return fn(x, kernel_size, stride, padding, dilation, ceil_mode, version, mapping)


def call_avgpool(x, kernel_size, stride=None, padding=0, dilation=1,
                 ceil_mode=True, count_include_pad=True, divisor_override=None,
                 version=0, mapping=0):
    dtype = x.dtype
    fn = _pooling.avgpool2d_f16 if dtype == np.float16 else _pooling.avgpool2d_f32
    return fn(x, kernel_size, stride, padding, dilation, ceil_mode,
              count_include_pad, divisor_override, version, mapping)


def benchmark(fn, warmup=10, iters=50):
    for _ in range(warmup):
        fn()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    elapsed = (time.perf_counter() - start) / iters
    return elapsed


def format_results(results):
    print(f"| {'Version':<10} | {'Time (ms)':>10} | {'GB/s':>8} | {'Speedup':>8} |")
    print(f"|------------|------------|----------|----------|")
    baseline = results[0][1]
    for name, t, bw in results:
        speedup = baseline / t if t > 0 else 0
        print(f"| {name:<10} | {t*1000:>10.4f} | {bw:>8.2f} | {speedup:>7.2f}x |")


# --- Synthetic benchmarks ---

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

# --- Real model benchmarks ---

MODEL_CASES = [
    # (name, pool_type, kernel_size, stride, padding, dilation, ceil_mode, count_include_pad, divisor_override, shape)
    ("resnet_maxpool", "max", 3, 2, 1, 1, False, None, None, (1, 56, 56, 64)),
    ("resnet_global", "avg", None, 1, 0, 1, False, True, None, (1, 7, 7, 512)),
    ("vgg_maxpool", "max", 2, 2, 0, 1, False, None, None, (1, 112, 112, 128)),
    ("densenet_maxpool", "max", 3, 2, 1, 1, False, None, None, (1, 112, 112, 64)),
    ("densenet_avgpool", "avg", 2, 2, 0, 1, False, True, None, (1, 56, 56, 64)),
    ("googlenet_maxpool", "max", 3, 2, 0, 1, True, None, None, (1, 112, 112, 64)),
    ("googlenet_s1", "max", 3, 1, 1, 1, True, None, None, (1, 28, 28, 480)),
    ("inception_v3_maxpool", "max", 3, 2, 0, 1, False, None, None, (1, 35, 35, 288)),
    ("inception_v3_avgpool", "avg", 3, 1, 1, 1, False, True, None, (1, 35, 35, 256)),
    ("inception_v3_aux", "avg", 5, 3, 0, 1, False, True, None, (1, 17, 17, 768)),
    ("inception_v4_avgpool", "avg", 3, 1, 1, 1, False, False, None, (1, 35, 35, 384)),
    ("inception_v4_maxpool", "max", 3, 2, 0, 1, False, None, None, (1, 35, 35, 384)),
    ("inception_resnet_maxpool", "max", 3, 2, 0, 1, False, None, None, (1, 73, 73, 192)),
    ("inception_resnet_avgpool", "avg", 3, 1, 1, 1, False, False, None, (1, 35, 35, 192)),
    ("yolo_sppf", "max", 5, 1, 2, 1, False, None, None, (1, 20, 20, 512)),
    ("yolo_spp_k9", "max", 9, 1, 4, 1, False, None, None, (1, 19, 19, 512)),
    ("yolo_spp_k13", "max", 13, 1, 6, 1, False, None, None, (1, 19, 19, 512)),
    ("yolov3_tiny_s1", "max", 2, 1, 0, 1, False, None, None, (1, 13, 13, 512)),
    ("shufflenet_maxpool", "max", 3, 2, 1, 1, False, None, None, (1, 112, 112, 24)),
    ("efficientnet_global", "avg", None, 1, 0, 1, False, True, None, (1, 7, 7, 1280)),
    ("swin_global", "avg", None, 1, 0, 1, False, True, None, (1, 7, 7, 768)),
]

ALL_VERSIONS = [0, 1, 2, 3, 4, 5, 6]
V7_MAPPINGS = [0, 1, 2, 3]


def bench_one(case_name, ptype, x, ks, stride, pad, dilation=1, ceil=False,
              cip=True, div=None):
    """Benchmark one case across all versions and v7 mappings."""
    N, H, W, C = x.shape
    input_bytes = N * H * W * C * x.itemsize
    results = []

    for v in ALL_VERSIONS:
        try:
            if ptype == "max":
                fn = lambda v=v: call_maxpool(x, ks, stride, pad, dilation, ceil, v, 0)
            else:
                fn = lambda v=v: call_avgpool(x, ks, stride, pad, dilation, ceil, cip, div, v, 0)
            t = benchmark(fn)
            gb_per_s = input_bytes / t / 1e9
            results.append((f"v{v}", t, gb_per_s))
        except Exception as e:
            results.append((f"v{v}", float('nan'), 0))

    for m in V7_MAPPINGS:
        try:
            if ptype == "max":
                fn = lambda m=m: call_maxpool(x, ks, stride, pad, dilation, ceil, 7, m)
            else:
                fn = lambda m=m: call_avgpool(x, ks, stride, pad, dilation, ceil, cip, div, 7, m)
            t = benchmark(fn)
            gb_per_s = input_bytes / t / 1e9
            results.append((f"v7m{'ABCD'[m]}", t, gb_per_s))
        except Exception as e:
            results.append((f"v7m{'ABCD'[m]}", float('nan'), 0))

    return results


def run_benchmarks(dtype=np.float32):
    print("=" * 70)
    print(f"POOLING2D BENCHMARKS ({'fp32' if dtype == np.float32 else 'fp16'})")
    print("=" * 70)

    # Synthetic benchmarks
    print("\n" + "=" * 70)
    print("SYNTHETIC BENCHMARKS")
    print("=" * 70)

    for name, shape in SYNTHETIC:
        N, H, W, C = shape
        for kname, ks, stride, pad in KERNEL_CONFIGS:
            x = np.random.randn(*shape).astype(dtype)
            print(f"\n--- {name} {shape} kernel={kname} ---")
            results = bench_one(f"{name}_{kname}", "max", x, ks, stride, pad)
            format_results(results)

    # Real model benchmarks
    print("\n" + "=" * 70)
    print("REAL MODEL BENCHMARKS")
    print("=" * 70)

    for case in MODEL_CASES:
        name, ptype, ks, stride, pad, dilation, ceil, cip, div, shape = case
        x = np.random.randn(*shape).astype(dtype)
        if ks is None:
            ks = (shape[1], shape[2])  # global pooling
        print(f"\n--- {name} {shape} ---")
        results = bench_one(name, ptype, x, ks, stride, pad, dilation, ceil, cip, div)
        format_results(results)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="CUDA Pooling2D benchmarks")
    parser.add_argument("--fp16", action="store_true", help="Use fp16 instead of fp32")
    args = parser.parse_args()
    dtype = np.float16 if args.fp16 else np.float32
    run_benchmarks(dtype)
