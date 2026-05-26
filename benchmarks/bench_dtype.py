"""Single-dtype benchmark runner. Can be run standalone or imported by bench_multidtype.py.

Usage (standalone):
    python benchmarks/bench_dtype.py fp32
    python benchmarks/bench_dtype.py fp8_e4m3 None 2 3

Usage (imported):
    from bench_dtype import run_dtype
    results = run_dtype('fp32', P=_pooling_module, warmup=5, iters=20)
"""
import numpy as np
import sys
import os
import json
import time

DTYPE_MAP = {
    'fp32':     {'np_type': np.float32, 'fn_suffix': 'f32',     'sz': 4, 'has_timed': True},
    'fp16':     {'np_type': np.float16, 'fn_suffix': 'f16',     'sz': 2, 'has_timed': False},
    'bf16':     {'np_type': np.uint16,  'fn_suffix': 'bf16',    'sz': 2, 'has_timed': True},
    'fp8_e4m3': {'np_type': np.uint8,   'fn_suffix': 'fp8_e4m3','sz': 1, 'has_timed': True},
    'fp8_e5m2': {'np_type': np.uint8,   'fn_suffix': 'fp8_e5m2','sz': 1, 'has_timed': True},
    'int8':     {'np_type': np.int8,    'fn_suffix': 'i8',      'sz': 1, 'has_timed': True},
    'int16':    {'np_type': np.int16,   'fn_suffix': 'i16',     'sz': 2, 'has_timed': True},
}

CANONICAL = [
    ("mem_bound",    "max", (1, 128, 128, 256), 3, 2, 1),
    ("global_avg",   "avg", (1, 7, 7, 512),      7, 1, 0),
    ("dense_3x3s1",  "max", (1, 56, 56, 64),     3, 1, 1),
    ("large_k13",    "max", (1, 32, 32, 64),    13, 1, 6),
]

ALL_VERSIONS = [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15]
V7_MAPPINGS = [0, 1, 2, 3]


def make_input(dtype_info, shape):
    np_type = dtype_info['np_type']
    if np_type in (np.float32, np.float16):
        return np.random.randn(*shape).astype(np_type)
    elif np_type == np.uint16:
        fp32 = np.random.randn(*shape).astype(np.float32)
        return (fp32.view(np.uint32) >> 16).astype(np.uint16)
    elif np_type == np.uint8:
        return np.random.randint(0, 120, shape).astype(np.uint8)
    elif np_type == np.int8:
        return np.random.randint(-64, 64, shape).astype(np.int8)
    elif np_type == np.int16:
        return np.random.randint(-256, 256, shape).astype(np.int16)
    return np.random.randn(*shape)


def get_fn(pool_type, suffix, timed, P):
    if timed:
        prefix = 'maxpool2d_timed_' if pool_type == 'max' else 'avgpool2d_timed_'
    else:
        prefix = 'maxpool2d_' if pool_type == 'max' else 'avgpool2d_'
    return getattr(P, prefix + suffix)


def bench_one(pool_type, suffix, has_timed, x, ks, stride, pad, version, mapping, P):
    fn = get_fn(pool_type, suffix, has_timed, P)
    if pool_type == 'max':
        call = lambda: fn(x, ks, stride, pad, 1, False, version, mapping)
    else:
        call = lambda: fn(x, ks, stride, pad, 1, False, True, None, version, mapping)

    if has_timed:
        for _ in range(WARMUP):
            call()
        P.cuda_synchronize()
        times = []
        for _ in range(ITERS):
            _, ms = call()
            times.append(ms)
        return np.median(times)
    else:
        P.cuda_synchronize()
        for _ in range(WARMUP):
            call()
        P.cuda_synchronize()
        t0 = time.perf_counter()
        for _ in range(ITERS):
            call()
        P.cuda_synchronize()
        return (time.perf_counter() - t0) / ITERS * 1000.0


def run_dtype(dtype_key, pool_only=None, warmup=5, iters=20, P=None):
    """Run benchmarks for one dtype. Returns dict of results.

    If P is None, imports _pooling from the build directory.
    """
    global WARMUP, ITERS
    WARMUP = warmup
    ITERS = iters

    if P is None:
        build_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'build')
        sys.path.insert(0, build_dir)
        import _pooling as P

    dtype_info = DTYPE_MAP[dtype_key]
    suffix = dtype_info['fn_suffix']
    has_timed = dtype_info['has_timed']
    sz = dtype_info['sz']

    results = {}
    for name, pool_type, shape, ks, stride, pad in CANONICAL:
        if pool_only and pool_type != pool_only:
            continue
        N, H, W, C = shape
        x = make_input(dtype_info, shape)
        input_bytes = N * H * W * C * sz

        r = {
            'pool_type': pool_type, 'shape': list(shape), 'ks': ks,
            'stride': stride, 'pad': pad, 'input_bytes': input_bytes,
            'versions': {}, 'v7': {}
        }

        for v in ALL_VERSIONS:
            try:
                t = bench_one(pool_type, suffix, has_timed, x, ks, stride, pad, v, 0, P)
                r['versions'][str(v)] = round(float(t), 6)
            except Exception as e:
                r['versions'][str(v)] = {'error': str(e)}
                break  # context corrupted, skip rest of versions

        # Save config results before running v7 mappings (which may crash)
        results[name] = r

        v7_crashed = False
        for m in V7_MAPPINGS:
            if r['versions'].get('0') and isinstance(r['versions']['0'], dict):
                break  # v0 already failed, skip v7
            try:
                t = bench_one(pool_type, suffix, has_timed, x, ks, stride, pad, 7, m, P)
                r['v7'][str(m)] = round(float(t), 6)
            except Exception as e:
                r['v7'][str(m)] = {'error': str(e)}
                # v7 kernel crashed - context is likely corrupted.
                # Skip remaining configs for this dtype.
                v7_crashed = True
                break

        if v7_crashed:
            return results

    return results


def main():
    dtype_key = sys.argv[1]
    pool_only = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] != 'None' else None
    warmup = int(sys.argv[3]) if len(sys.argv) > 3 else 5
    iters = int(sys.argv[4]) if len(sys.argv) > 4 else 20

    results = run_dtype(dtype_key, pool_only, warmup, iters)
    print(json.dumps(results))


if __name__ == "__main__":
    main()
