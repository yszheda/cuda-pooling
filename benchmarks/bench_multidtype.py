"""Multi-dtype CUDA Pooling2D benchmark with subprocess isolation per dtype.

Each dtype runs in a separate subprocess to avoid CUDA context pollution
from kernel crashes (e.g. fp16 v9+ illegal memory access).

Usage:
    python benchmarks/bench_multidtype.py              # all dtypes
    python benchmarks/bench_multidtype.py --dtype fp32 # single dtype
    python benchmarks/bench_multidtype.py --pool avg   # avgpool only
    python benchmarks/bench_multidtype.py --summary    # cross-dtype summary only
"""

import sys
import os
import subprocess
import json

BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
DTYPE_DESC = {
    'fp32':      {'np': 'float32', 'sz': 4, 'has_timed': True},
    'fp16':      {'np': 'float16', 'sz': 2, 'has_timed': False},
    'bf16':      {'np': 'uint16',  'sz': 2, 'has_timed': True},
    'fp8_e4m3':  {'np': 'uint8',   'sz': 1, 'has_timed': True},
    'fp8_e5m2':  {'np': 'uint8',   'sz': 1, 'has_timed': True},
    'int8':      {'np': 'int8',    'sz': 1, 'has_timed': True},
    'int16':     {'np': 'int16',   'sz': 2, 'has_timed': True},
}
ALL_DTYPES = list(DTYPE_DESC.keys())
# Run fp16 last since its v9+ kernels crash and may corrupt the
# subprocess parent's CUDA context, preventing subsequent dtypes from running.
ALL_DTYPES_ORDERED = ['fp32', 'bf16', 'fp8_e4m3', 'fp8_e5m2', 'int8', 'int16', 'fp16']
CANONICAL = [
    ("mem_bound",    "max", (1, 128, 128, 256), 3, 2, 1),
    ("global_avg",   "avg", (1, 7, 7, 512),      7, 1, 0),
    ("dense_3x3s1",  "max", (1, 56, 56, 64),     3, 1, 1),
    ("large_k13",    "max", (1, 32, 32, 64),    13, 1, 6),
]
ALL_VERSIONS = [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15]
V7_MAPPINGS = [0, 1, 2, 3]


def run_dtype(dtype_key, pool_only=None, warmup=5, iters=20):
    """Run bench_dtype.py subprocess for one dtype.

    Each subprocess gets a clean CUDA context.
    """
    cmd = [sys.executable, os.path.join(BENCH_DIR, 'bench_dtype.py'), dtype_key]
    if pool_only:
        cmd.append(pool_only)
    else:
        cmd.append('None')  # placeholder so warmup/iters align correctly
    cmd.append(str(warmup))
    cmd.append(str(iters))

    # Run in a new process group to avoid inheriting any CUDA state.
    kwargs = {}
    if sys.platform != 'win32':
        kwargs['start_new_session'] = True

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300, **kwargs)
    if result.returncode != 0:
        print(f"  SUBPROCESS ERROR for {dtype_key} (rc={result.returncode}):", file=sys.stderr)
        print(result.stderr[:500], file=sys.stderr)
        return None
    try:
        data = json.loads(result.stdout.strip())
        print(f"  {dtype_key}: got {len(data)} configs", file=sys.stderr)
        return data
    except json.JSONDecodeError as e:
        print(f"  JSON parse error for {dtype_key}: {e}", file=sys.stderr)
        print(f"  stdout: {result.stdout[:200]}", file=sys.stderr)
        print(f"  stderr: {result.stderr[:200]}", file=sys.stderr)
        return None


def print_dtype_results(dtype_key, results, pool_only=None):
    """Print formatted results for one dtype."""
    desc = DTYPE_DESC[dtype_key]
    timing = "kernel-only CUDA event timing" if desc['has_timed'] else "wall-clock timing"
    print(f"\n{'='*70}")
    print(f"MULTI-DTYPE BENCHMARKS - {dtype_key.upper()}")
    print(f"  ({timing})")
    print(f"{'='*70}")

    if not results:
        print("  No results (subprocess failed)")
        return

    for name in [c[0] for c in CANONICAL if not pool_only or c[1] == pool_only]:
        if name not in results:
            continue
        r = results[name]
        shape = tuple(r['shape'])
        pool_type = r['pool_type']
        input_bytes = r['input_bytes']
        print(f"\n--- {name} {shape} {pool_type} k={r['ks']}s={r['stride']}p={r['pad']} ---")
        print(f"| {'Version':<10} | {'Kernel (ms)':>12} | {'GB/s':>8} | {'Speedup':>8} |")
        print(f"|------------|--------------|----------|----------|")

        baseline = None
        entries = []
        for v_str in [str(v) for v in ALL_VERSIONS]:
            val = r['versions'].get(v_str)
            if isinstance(val, dict) and 'error' in val:
                entries.append((f"v{v_str}", -1, 0))
                continue
            t = float(val) if val else -1
            gb = input_bytes / (t * 1e-3) / 1e9 if t > 0 else 0
            entries.append((f"v{v_str}", t, gb))
            if baseline is None and t > 0:
                baseline = t

        if baseline is None:
            baseline = 1e-9
        for vname, t, gb in entries:
            if t < 0:
                print(f"| {vname:<10} | {'ERROR':>12} | {'-':>8} | {'-':>8} |")
            else:
                speedup = baseline / t if t > 0 else 0
                print(f"| {vname:<10} | {t:>12.4f} | {gb:>8.2f} | {speedup:>7.2f}x |")

        for m in V7_MAPPINGS:
            val = r['v7'].get(str(m))
            if isinstance(val, dict) and 'error' in val:
                print(f"| {'v7m'+'ABCD'[m]:<10} | {'ERROR':>12} | {'-':>8} | {'-':>8} |")
                continue
            t = float(val) if val else -1
            if t < 0:
                print(f"| {'v7m'+'ABCD'[m]:<10} | {'ERROR':>12} | {'-':>8} | {'-':>8} |")
            else:
                gb = input_bytes / (t * 1e-3) / 1e9 if t > 0 else 0
                speedup = baseline / t if t > 0 else 0
                print(f"| {'v7m'+'ABCD'[m]:<10} | {t:>12.4f} | {gb:>8.2f} | {speedup:>7.2f}x |")


def run_cross_dtype_summary(pool_only=None, warmup=3, iters=10):
    """Cross-dtype comparison with per-dtype subprocess isolation."""
    print(f"\n{'='*70}")
    print(f"CROSS-DTYPE PERFORMANCE SUMMARY")
    print(f"{'='*70}")

    dtype_results = {}
    for dtype_key in ALL_DTYPES_ORDERED:
        print(f"\n  Running {dtype_key}...", file=sys.stderr)
        sys.stderr.flush()
        result = run_dtype(dtype_key, pool_only, warmup, iters)
        dtype_results[dtype_key] = result

    for name, pool_type, shape, ks, stride, pad in CANONICAL:
        if pool_only and pool_type != pool_only:
            continue

        print(f"\n--- {name} ({pool_type}, {shape}, k={ks}s={stride}) ---")
        print(f"| {'Dtype':<12} | {'v0 (ms)':>9} | {'Best (ms)':>9} | {'v0->Best':>10} | {'v0 BW':>9} |")
        print(f"|------------|-----------|-----------|------------|-----------|")

        for dtype_key in ALL_DTYPES_ORDERED:
            results = dtype_results.get(dtype_key)
            if not results or name not in results:
                print(f"| {dtype_key:<12} | {'N/A':>9} | {'-':>9} | {'-':>10} | {'-':>9} |")
                continue

            r = results[name]
            input_bytes = r['input_bytes']

            v0_val = r['versions'].get('0')
            if isinstance(v0_val, dict) and 'error' in v0_val:
                print(f"| {dtype_key:<12} | {'ERROR':>9} | {'-':>9} | {'-':>10} | {'-':>9} |")
                continue
            t0 = float(v0_val) if v0_val else 0
            if t0 <= 0:
                print(f"| {dtype_key:<12} | {'ERROR':>9} | {'-':>9} | {'-':>10} | {'-':>9} |")
                continue

            best_t = t0
            best_v = 0
            for v in [1, 2, 8, 10, 14, 15]:
                val = r['versions'].get(str(v))
                if val and not (isinstance(val, dict) and 'error' in val):
                    t = float(val)
                    if t > 0 and t < best_t:
                        best_t = t
                        best_v = v

            bw0 = input_bytes / (t0 * 1e-3) / 1e9
            speedup = t0 / best_t if best_t > 0 else 1
            print(f"| {dtype_key:<12} | {t0:>9.4f} | {best_t:>9.4f} | v{best_v} {speedup:>6.2f}x | {bw0:>8.2f} |")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Multi-dtype CUDA Pooling2D benchmarks (process-isolated)")
    parser.add_argument("--dtype", choices=ALL_DTYPES, help="Benchmark only this dtype")
    parser.add_argument("--pool", choices=["max", "avg"], help="Benchmark only this pool type")
    parser.add_argument("--summary", action="store_true", help="Only run cross-dtype summary")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup iterations")
    parser.add_argument("--iters", type=int, default=20, help="Timed iterations")
    args = parser.parse_args()

    dtypes = [args.dtype] if args.dtype else ALL_DTYPES_ORDERED

    if args.summary:
        run_cross_dtype_summary(args.pool, args.warmup, args.iters)
    else:
        for d in dtypes:
            print(f"\n>>> Benchmarking {d} (isolated subprocess) ...", file=sys.stderr)
            results = run_dtype(d, args.pool, args.warmup, args.iters)
            print_dtype_results(d, results, args.pool)

        print(f"\n>>> Running cross-dtype summary ...", file=sys.stderr)
        run_cross_dtype_summary(args.pool, 3, 10)
