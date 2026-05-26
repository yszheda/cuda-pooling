"""Full benchmark runner - subprocess per dtype to avoid CUDA context corruption.

Usage:
    python bench_full.py              # Run all dtypes (orchestrator)
    python bench_full.py fp32         # Run single dtype (worker)
"""
import sys
import os

# Extended benchmark configs
CANONICAL = [
    ("mem_bound",     "max", (1, 128, 128, 256), 3,  2, 1),
    ("global_avg",    "avg", (1, 7,   7,   512),  7,  1, 0),
    ("dense_3x3s1",   "max", (1, 56,  56,  64),   3,  1, 1),
    ("large_k13",     "max", (1, 32,  32,  64),   13, 1, 6),
    ("small_2x2s2",   "max", (1, 64,  64,  32),   2,  2, 0),
    ("mid_5x5s2",     "max", (1, 28,  28,  128),  5,  2, 2),
    ("batch_3x3s1",   "max", (4, 32,  32,  64),   3,  1, 1),
    ("wide_k7",       "max", (1, 16,  16,  256),  7,  1, 3),
    ("global_max",    "max", (1, 7,   7,   1024), 7,  1, 0),
    ("avg_dense",     "avg", (1, 28,  28,  256),  3,  1, 1),
]

ALL_VERSIONS = [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15]
V7_MAPPINGS = [0, 1, 2, 3]
DTYPE_KEYS = ['fp32', 'bf16', 'fp8_e4m3', 'fp8_e5m2', 'int8', 'int16']

DTYPE_MAP = {
    'fp32':     {'np_type': 'float32',     'suffix': 'f32',     'sz': 4},
    'bf16':     {'np_type': 'uint16',      'suffix': 'bf16',    'sz': 2},
    'fp8_e4m3': {'np_type': 'uint8',       'suffix': 'fp8_e4m3','sz': 1},
    'fp8_e5m2': {'np_type': 'uint8',       'suffix': 'fp8_e5m2','sz': 1},
    'int8':     {'np_type': 'int8',        'suffix': 'i8',      'sz': 1},
    'int16':    {'np_type': 'int16',       'suffix': 'i16',     'sz': 2},
}

WARMUP = 3
ITERS = 10


def run_single_dtype(dtype_key):
    """Benchmark one dtype. Returns dict of results. Runs in isolated process."""
    import numpy as np

    BUILD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'build')
    sys.path.insert(0, BUILD_DIR)
    import _pooling as P

    dtype_info = DTYPE_MAP[dtype_key]
    suffix = dtype_info['suffix']
    sz = dtype_info['sz']
    np_type_name = dtype_info['np_type']

    def make_input(shape):
        if np_type_name in ('float32',):
            return np.random.randn(*shape).astype(np.float32)
        elif np_type_name == 'uint16':
            return (np.random.randn(*shape).astype(np.float32).view(np.uint32) >> 16).astype(np.uint16)
        elif np_type_name == 'uint8':
            return np.random.randint(0, 120, shape).astype(np.uint8)
        elif np_type_name == 'int8':
            return np.random.randint(-64, 64, shape).astype(np.int8)
        elif np_type_name == 'int16':
            return np.random.randint(-256, 256, shape).astype(np.int16)
        return np.random.randn(*shape)

    results = {}
    for name, pool_type, shape, ks, stride, pad in CANONICAL:
        N, H, W, C = shape
        x = make_input(shape)
        input_bytes = N * H * W * C * sz

        r = {
            'pool_type': pool_type, 'shape': list(shape),
            'ks': ks, 'stride': stride, 'pad': pad,
            'input_bytes': input_bytes,
            'versions': {}, 'v7': {}
        }

        pfx = 'maxpool2d_timed_' if pool_type == 'max' else 'avgpool2d_timed_'
        fn = getattr(P, pfx + suffix)

        for v in ALL_VERSIONS:
            try:
                call = lambda: fn(x, ks, stride, pad, 1, False, v, 0) if pool_type == 'max' \
                    else fn(x, ks, stride, pad, 1, False, True, None, v, 0)
                for _ in range(WARMUP):
                    call()
                P.cuda_synchronize()
                times = []
                for _ in range(ITERS):
                    _, ms = call()
                    times.append(ms)
                t = float(np.median(times))
                r['versions'][str(v)] = round(t, 6)
            except Exception as e:
                r['versions'][str(v)] = {'error': str(e)}
                break

        for m in V7_MAPPINGS:
            try:
                call = lambda: fn(x, ks, stride, pad, 1, False, 7, m) if pool_type == 'max' \
                    else fn(x, ks, stride, pad, 1, False, True, None, 7, m)
                for _ in range(WARMUP):
                    call()
                P.cuda_synchronize()
                _, ms = call()
                r['v7'][str(m)] = round(float(ms), 6)
            except Exception as e:
                r['v7'][str(m)] = {'error': str(e)}

        results[name] = r

    return results


def run_all_dtypes():
    """Run each dtype in a subprocess and merge results."""
    import subprocess

    BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
    results = {}
    for dk in DTYPE_KEYS:
        print(f"\n>>> Running {dk}...", file=sys.stderr)
        sys.stderr.flush()
        cmd = [sys.executable, os.path.join(BENCH_DIR, 'bench_full.py'), dk]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300,
                                start_new_session=True)
        if result.returncode != 0:
            print(f"  {dk}: subprocess failed (rc={result.returncode})", file=sys.stderr)
            print(f"  stderr: {result.stderr[:300]}", file=sys.stderr)
            continue
        try:
            # Extract only the last JSON object from stdout (in case of mixed output)
            stdout = result.stdout.strip()
            last_brace = stdout.rfind('}')
            first_brace = stdout.find('{')
            if first_brace == -1 or last_brace == -1:
                print(f"  {dk}: no JSON found in stdout", file=sys.stderr)
                print(f"  stdout: {stdout[:200]}", file=sys.stderr)
                continue
            json_str = stdout[first_brace:last_brace+1]
            data = json.loads(json_str)
            results.update(data)
            n_ok = sum(1 for dv in data.values()
                       if isinstance(dv, dict)
                       for val in dv.get('versions', {}).values()
                       if not isinstance(val, dict))
            print(f"  {dk}: {n_ok} version timings", file=sys.stderr)
        except json.JSONDecodeError as e:
            print(f"  {dk}: JSON error: {e}", file=sys.stderr)
            print(f"  stdout: {result.stdout[:200]}", file=sys.stderr)

    print(json.dumps(results, indent=2))


def main():
    if len(sys.argv) > 1:
        # Single dtype mode (worker)
        dtype_key = sys.argv[1]
        results = run_single_dtype(dtype_key)
        print(json.dumps({dtype_key: results}, indent=2))
    else:
        # Orchestrator mode
        run_all_dtypes()


if __name__ == "__main__":
    import json
    main()
