"""Multi-dtype CUDA Pooling2D benchmark.

Tests all 7 dtypes (fp32, fp16, bf16, fp8_e4m3, fp8_e5m2, int8, int16)
across 4 canonical configurations. Uses timed pybind functions where
available (kernel-only CUDA event timing), falls back to wall-clock
timing with cuda_synchronize for dtypes without timed variants.

Usage:
    python benchmarks/bench_multidtype.py              # all dtypes
    python benchmarks/bench_multidtype.py --dtype fp32 # single dtype
    python benchmarks/bench_multidtype.py --pool avg   # avgpool only
"""

import numpy as np
import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'build'))
import _pooling


# --- Dtype descriptors ---
# For dtypes with timed variants: use kernel-only CUDA event timing
# For dtypes without timed variants: use wall-clock + cuda_synchronize
DTYPE_DESC = {
    'fp32':      {'np': np.float32, 'sz': 4, 'has_timed': True,  'label': 'fp32'},
    'fp16':      {'np': np.float16, 'sz': 2, 'has_timed': False, 'label': 'fp16'},
    'bf16':      {'np': np.uint16,  'sz': 2, 'has_timed': True,  'label': 'bf16'},
    'fp8_e4m3':  {'np': np.uint8,   'sz': 1, 'has_timed': True,  'label': 'fp8_e4m3'},
    'fp8_e5m2':  {'np': np.uint8,   'sz': 1, 'has_timed': True,  'label': 'fp8_e5m2'},
    'int8':      {'np': np.int8,    'sz': 1, 'has_timed': True,  'label': 'int8'},
    'int16':     {'np': np.int16,   'sz': 2, 'has_timed': True,  'label': 'int16'},
}

ALL_DTYPES = list(DTYPE_DESC.keys())
ALL_VERSIONS = [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15]
V7_MAPPINGS = [0, 1, 2, 3]

# Canonical configs from the plan
CANONICAL = [
    ("mem_bound",    "max", (1, 128, 128, 256), 3, 2, 1),
    ("global_avg",   "avg", (1, 7, 7, 512),      7, 1, 0),
    ("dense_3x3s1",  "max", (1, 56, 56, 64),     3, 1, 1),
    ("large_k13",    "max", (1, 32, 32, 64),    13, 1, 6),
]

WARMUP = 5
ITERS = 20


def make_input(dtype_key, shape):
    """Generate random input for the dtype."""
    desc = DTYPE_DESC[dtype_key]
    np_type = desc['np']
    if dtype_key in ('fp32', 'fp16'):
        return np.random.randn(*shape).astype(np_type)
    elif dtype_key == 'bf16':
        # Generate fp32, convert to bf16 bit representation (upper 16 bits of fp32)
        fp32 = np.random.randn(*shape).astype(np.float32)
        u32 = fp32.view(np.uint32)
        return (u32 >> 16).astype(np.uint16)
    elif dtype_key in ('fp8_e4m3', 'fp8_e5m2'):
        return np.random.randint(0, 120, shape).astype(np.uint8)
    elif dtype_key == 'int8':
        return np.random.randint(-64, 64, shape).astype(np.int8)
    elif dtype_key == 'int16':
        return np.random.randint(-256, 256, shape).astype(np.int16)
    return np.random.randn(*shape).astype(np_type)


def get_maxpool_fn(dtype_key, timed):
    prefix = 'maxpool2d_timed_' if timed else 'maxpool2d_'
    name = prefix + DTYPE_DESC[dtype_key]['label']
    return getattr(_pooling, name)


def get_avgpool_fn(dtype_key, timed):
    prefix = 'avgpool2d_timed_' if timed else 'avgpool2d_'
    name = prefix + DTYPE_DESC[dtype_key]['label']
    return getattr(_pooling, name)


def benchmark_one(dtype_key, pool_type, x, ks, stride, pad, version=0, mapping=0):
    """Benchmark one config. Returns median kernel time in ms."""
    desc = DTYPE_DESC[dtype_key]
    has_timed = desc['has_timed']

    if pool_type == 'max':
        fn = get_maxpool_fn(dtype_key, has_timed)
        call = lambda: fn(x, ks, stride, pad, 1, False, version, mapping)
    else:
        fn = get_avgpool_fn(dtype_key, has_timed)
        call = lambda: fn(x, ks, stride, pad, 1, False, True, None, version, mapping)

    if has_timed:
        # Warmup
        for _ in range(WARMUP):
            call()
        _pooling.cuda_synchronize()

        # Timed
        times = []
        for _ in range(ITERS):
            _, ms = call()
            times.append(ms)
        return np.median(times)
    else:
        # Wall-clock fallback: cuda_synchronize + perf_counter
        _pooling.cuda_synchronize()
        for _ in range(WARMUP):
            call()
        _pooling.cuda_synchronize()

        t0 = time.perf_counter()
        for _ in range(ITERS):
            call()
        _pooling.cuda_synchronize()
        t1 = time.perf_counter()
        return (t1 - t0) / ITERS * 1000.0


def run_canonical(dtype_key, pool_only=None):
    """Run canonical benchmarks for one dtype."""
    desc = DTYPE_DESC[dtype_key]
    print(f"\n{'='*70}")
    print(f"MULTI-DTYPE BENCHMARKS - {dtype_key.upper()}")
    timing_note = "(kernel-only CUDA event timing)" if desc['has_timed'] else "(wall-clock timing, less precise)"
    print(f"  {timing_note}")
    print(f"{'='*70}")

    for name, pool_type, shape, ks, stride, pad in CANONICAL:
        if pool_only and pool_type != pool_only:
            continue
        N, H, W, C = shape
        x = make_input(dtype_key, shape)
        input_bytes = N * H * W * C * desc['sz']

        print(f"\n--- {name} {shape} {pool_type} k={ks}s={stride}p={pad} ---")
        print(f"| {'Version':<10} | {'Kernel (ms)':>12} | {'GB/s':>8} | {'Speedup':>8} |")
        print(f"|------------|--------------|----------|----------|")

        results = []
        for v in ALL_VERSIONS:
            try:
                t = benchmark_one(dtype_key, pool_type, x, ks, stride, pad, v, 0)
                gb = input_bytes / (t * 1e-3) / 1e9 if t > 0 else 0
                results.append((f"v{v}", t, gb))
            except Exception as e:
                results.append((f"v{v}", -1, 0))

        baseline = results[0][1] if results[0][1] > 0 else 1e-9
        for vname, t, gb in results:
            if t < 0:
                print(f"| {vname:<10} | {'ERROR':>12} | {'-':>8} | {'-':>8} |")
                continue
            speedup = baseline / t if t > 0 else 0
            print(f"| {vname:<10} | {t:>12.4f} | {gb:>8.2f} | {speedup:>7.2f}x |")

        # v7 variants
        for m in V7_MAPPINGS:
            try:
                t = benchmark_one(dtype_key, pool_type, x, ks, stride, pad, 7, m)
                gb = input_bytes / (t * 1e-3) / 1e9 if t > 0 else 0
                speedup = baseline / t if t > 0 else 0
                print(f"| {'v7m'+'ABCD'[m]:<10} | {t:>12.4f} | {gb:>8.2f} | {speedup:>7.2f}x |")
            except Exception as e:
                print(f"| {'v7m'+'ABCD'[m]:<10} | {'ERROR':>12} | {'-':>8} | {'-':>8} |")


def run_cross_dtype_summary(pool_only=None):
    """Cross-dtype comparison: v0 vs best for each dtype."""
    print(f"\n{'='*70}")
    print(f"CROSS-DTYPE PERFORMANCE SUMMARY")
    print(f"{'='*70}")

    for name, pool_type, shape, ks, stride, pad in CANONICAL:
        if pool_only and pool_type != pool_only:
            continue

        print(f"\n--- {name} ({pool_type}, {shape}, k={ks}s={stride}) ---")
        print(f"| {'Dtype':<12} | {'v0 (ms)':>9} | {'Best (ms)':>9} | {'v0->Best':>10} | {'v0 BW':>9} |")
        print(f"|------------|-----------|-----------|------------|-----------|")

        for dtype_key in ALL_DTYPES:
            desc = DTYPE_DESC[dtype_key]
            x = make_input(dtype_key, shape)
            N, H, W, C = shape
            input_bytes = N * H * W * C * desc['sz']

            try:
                t0 = benchmark_one(dtype_key, pool_type, x, ks, stride, pad, 0, 0)
                best_t = t0
                best_v = 0
                for v in [1, 2, 8, 10, 14, 15]:
                    try:
                        t = benchmark_one(dtype_key, pool_type, x, ks, stride, pad, v, 0)
                        if t > 0 and t < best_t:
                            best_t = t
                            best_v = v
                    except:
                        pass
                bw0 = input_bytes / (t0 * 1e-3) / 1e9 if t0 > 0 else 0
                speedup = t0 / best_t if best_t > 0 else 1
                print(f"| {dtype_key:<12} | {t0:>9.4f} | {best_t:>9.4f} | v{best_v} {speedup:>6.2f}x | {bw0:>8.2f} |")
            except Exception as e:
                print(f"| {dtype_key:<12} | {'ERROR':>9} | {'-':>9} | {'-':>10} | {'-':>9} |")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Multi-dtype CUDA Pooling2D benchmarks")
    parser.add_argument("--dtype", choices=ALL_DTYPES, help="Benchmark only this dtype")
    parser.add_argument("--pool", choices=["max", "avg"], help="Benchmark only this pool type")
    parser.add_argument("--summary", action="store_true", help="Only run cross-dtype summary")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup iterations")
    parser.add_argument("--iters", type=int, default=20, help="Timed iterations")
    args = parser.parse_args()

    WARMUP = args.warmup
    ITERS = args.iters

    dtypes = [args.dtype] if args.dtype else ALL_DTYPES

    if args.summary:
        run_cross_dtype_summary(args.pool)
    else:
        for d in dtypes:
            run_canonical(d, args.pool)
        run_cross_dtype_summary(args.pool)
