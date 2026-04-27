"""Nsight Compute profiling for CUDA Pooling2D kernels.

Runs ncu on each (version, shape, kernel_config) combination, collecting
key hardware metrics: occupancy, DRAM throughput, L1/L2 traffic, register/smem
usage, stall reasons, SM utilization.

Output: CSV file with one row per (pool_type, version, dtype, shape, kernel_config, metric_name, value).
"""

import numpy as np
import subprocess
import csv
import sys
import os
import argparse
import tempfile
import json

NCU_PATH = "/usr/local/cuda-13.0/bin/ncu"

# Metrics to collect — primary names with fallback alternatives for different GPU architectures
METRICS = {
    "occupancy": [
        "sm__warps_active.avg.pct_of_peak",
        "sm__occupancy.avg.pct_of_peak_active_warps",
    ],
    "dram_throughput": [
        "dram__throughput.avg.pct_of_peak_sustained",
        "dram__throughput.avg.pct_of_peak_sustained.elapsed",
    ],
    "dram_read_bytes": [
        "dram__bytes_read.sum",
    ],
    "dram_write_bytes": [
        "dram__bytes_write.sum",
    ],
    "l1_read_sectors": [
        "l1tex__t_sectors_pipe_lsu.mem_global_op_ld.sum",
        "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum",
    ],
    "l2_read_sectors": [
        "lts__t_sectors_op_read.sum",
    ],
    "l2_write_sectors": [
        "lts__t_sectors_op_write.sum",
    ],
    "registers_per_thread": [
        "launch__registers_per_thread",
    ],
    "shared_mem_per_block": [
        "launch__shared_memory_per_block",
    ],
    "sm_utilization": [
        "smsp__cycles_active.avg.pct_of_peak_sustained_elapsed",
        "sm__cycles_active.avg.pct_of_peak_sustained_elapsed",
    ],
    "stall_not_selected": [
        "smsp__warps_issue_stalled_not_selected.per_cycle_active",
    ],
    "stall_wait": [
        "smsp__warps_issue_stalled_wait.per_cycle_active",
    ],
    "stall_long_scoreboard": [
        "smsp__warps_issue_stalled_long_scoreboard.per_cycle_active",
    ],
    "stall_short_scoreboard": [
        "smsp__warps_issue_stalled_short_scoreboard.per_cycle_active",
    ],
    "stall_math_pipe_throttle": [
        "smsp__warps_issue_stalled_math_pipe_throttle.per_cycle_active",
    ],
    "stall_mio_throttle": [
        "smsp__warps_issue_stalled_mio_throttle.per_cycle_active",
    ],
    "stall_mem_barrier": [
        "smsp__warps_issue_stalled_mem_barrier.per_cycle_active",
    ],
    "stall_sleeping": [
        "smsp__warps_issue_stalled_sleeping.per_cycle_active",
    ],
    "stall_execution_dependency": [
        "smsp__warps_issue_stalled_execution_dependency.per_cycle_active",
    ],
    "block_size": [
        "launch__block_size",
    ],
    "grid_size": [
        "launch__grid_size",
    ],
    "kernel_name": [
        "kernel__name",
    ],
    "duration_ns": [
        "gpu__time_duration.sum",
    ],
    "achieved_occupancy": [
        "sm__occupancy.avg.pct_of_peak_active_warps",
    ],
}

# Flatten all metric names for ncu command
def get_all_metric_names():
    names = []
    for category, alts in METRICS.items():
        names.extend(alts)
    return names


# Profiling configurations
PROFILING_CASES = [
    # (name, pool_type, kernel_size, stride, padding, dilation, ceil_mode, count_include_pad, divisor_override, shape)
    # Small spatial
    ("small_3x3s2p1_max", "max", 3, 2, 1, 1, False, None, None, (1, 32, 32, 64)),
    ("small_3x3s2p1_avg", "avg", 3, 2, 1, 1, False, True, None, (1, 32, 32, 64)),
    # Large spatial
    ("large_3x3s2p1_max", "max", 3, 2, 1, 1, False, None, None, (1, 128, 128, 256)),
    ("large_3x3s2p1_avg", "avg", 3, 2, 1, 1, False, True, None, (1, 128, 128, 256)),
    # Large C
    ("largeC_3x3s2p1_max", "max", 3, 2, 1, 1, False, None, None, (1, 28, 28, 512)),
    ("largeC_3x3s2p1_avg", "avg", 3, 2, 1, 1, False, True, None, (1, 28, 28, 512)),
    # Real model cases
    ("resnet_maxpool", "max", 3, 2, 1, 1, False, None, None, (1, 56, 56, 64)),
    ("resnet_global", "avg", 7, 1, 0, 1, False, True, None, (1, 7, 7, 512)),
    ("vgg_maxpool", "max", 2, 2, 0, 1, False, None, None, (1, 112, 112, 128)),
    ("yolo_sppf", "max", 5, 1, 2, 1, False, None, None, (1, 20, 20, 512)),
]

ALL_VERSIONS = [0, 1, 2, 3, 4, 5, 6]
V7_MAPPINGS = [0, 1, 2, 3]


def write_bench_script(pool_type, version, mapping, dtype, shape,
                       kernel_size, stride, padding, dilation, ceil_mode,
                       count_include_pad, divisor_override, out_path):
    """Write a small Python script that calls the pooling kernel once."""
    N, H, W, C = shape
    lines = [
        "import numpy as np",
        "import sys",
        "sys.path.insert(0, 'build')",
        "import _pooling",
        "",
        f"x = np.random.randn({N}, {H}, {W}, {C}).astype(np.{dtype})",
        "",
    ]

    if pool_type == "max":
        fn_name = "_pooling.maxpool2d_f32" if dtype == "float32" else "_pooling.maxpool2d_f16"
        lines.append(f"_pooling.cuda_synchronize()")
        lines.append(f"out = {fn_name}(x, {kernel_size}, {stride}, {padding}, {dilation}, {ceil_mode}, {version}, {mapping})")
    else:
        fn_name = "_pooling.avgpool2d_f32" if dtype == "float32" else "_pooling.avgpool2d_f16"
        cip_str = str(count_include_pad) if count_include_pad is not None else "True"
        div_str = str(divisor_override) if divisor_override is not None else "None"
        lines.append(f"_pooling.cuda_synchronize()")
        lines.append(f"out = {fn_name}(x, {kernel_size}, {stride}, {padding}, {dilation}, {ceil_mode}, {cip_str}, {div_str}, {version}, {mapping})")

    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")


def run_ncu(ncu_path, script_path, metric_names, cwd):
    """Run ncu on the given script and return parsed CSV results."""
    cmd = [
        ncu_path,
        "--target-processes", "all",
        "--set", "full",
        "--metrics", ",".join(metric_names),
        "--csv",
        "--force-overwrite",
        "--launch-skip", "1",  # Skip the cuda_synchronize call
        "--launch-count", "1",  # Only profile the kernel call
        "--cache-control", "none",
        "--clock-control", "none",
        sys.executable, script_path,
    ]

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=300,
            cwd=cwd, env={**os.environ, "CUDA_VISIBLE_DEVICES": "0"}
        )
    except subprocess.TimeoutExpired:
        return None

    if result.returncode != 0 and not result.stdout:
        return None

    return result.stdout


def parse_ncu_csv(csv_text):
    """Parse ncu CSV output into a dict of metric_name -> value."""
    if not csv_text or not csv_text.strip():
        return {}

    lines = csv_text.strip().split("\n")
    if len(lines) < 3:
        return {}

    # ncu CSV: first line is header, second line is unit, third+ are data
    reader = csv.reader(lines)
    rows = list(reader)
    if len(rows) < 2:
        return {}

    headers = rows[0]
    values = rows[2] if len(rows) > 2 else rows[1]  # Skip unit row

    result = {}
    for h, v in zip(headers, values):
        h = h.strip('"').strip()
        v = v.strip('"').strip()
        try:
            result[h] = float(v)
        except (ValueError, TypeError):
            result[h] = v  # Keep as string (e.g., kernel name)

    return result


def find_metric_value(ncu_results, metric_alts):
    """Try each alternative metric name and return the first found value."""
    for name in metric_alts:
        if name in ncu_results:
            return ncu_results[name]
    return None


def profile_one_case(case_name, pool_type, kernel_size, stride, padding, dilation,
                     ceil_mode, count_include_pad, divisor_override, shape,
                     version, mapping, dtype, ncu_path, cwd, all_metric_names):
    """Profile a single (version, case) combination with ncu."""
    label = f"{case_name}_v{version}" + (f"_m{'ABCD'[mapping]}" if version == 7 else "")
    script_path = os.path.join(cwd, "_ncu_bench_script.py")

    write_bench_script(
        pool_type, version, mapping, dtype, shape,
        kernel_size, stride, padding, dilation, ceil_mode,
        count_include_pad, divisor_override, script_path
    )

    csv_text = run_ncu(ncu_path, script_path, all_metric_names, cwd)

    if csv_text is None:
        print(f"  {label}: TIMEOUT or ERROR", file=sys.stderr)
        return []

    ncu_results = parse_ncu_csv(csv_text)
    if not ncu_results:
        print(f"  {label}: no results parsed", file=sys.stderr)
        return []

    rows = []
    for category, alts in METRICS.items():
        val = find_metric_value(ncu_results, alts)
        if val is not None:
            rows.append({
                "case": case_name,
                "pool_type": pool_type,
                "version": version,
                "mapping": mapping if version == 7 else 0,
                "dtype": dtype,
                "shape": f"({shape[0]},{shape[1]},{shape[2]},{shape[3]})",
                "kernel_size": kernel_size,
                "stride": stride,
                "padding": padding,
                "metric": category,
                "value": val,
            })

    print(f"  {label}: {len(rows)} metrics collected")
    return rows


def main():
    parser = argparse.ArgumentParser(description="Nsight Compute profiling for CUDA Pooling2D")
    parser.add_argument("--output", default="ncu_results.csv", help="Output CSV path")
    parser.add_argument("--ncu", default=NCU_PATH, help="Path to ncu binary")
    parser.add_argument("--dtype", choices=["float32", "float16"], default="float32")
    parser.add_argument("--cases", nargs="*", help="Subset of case names to profile")
    parser.add_argument("--versions", nargs="*", type=int, help="Subset of versions to profile")
    args = parser.parse_args()

    all_metric_names = get_all_metric_names()
    all_rows = []

    cases = PROFILING_CASES
    if args.cases:
        cases = [c for c in cases if c[0] in args.cases]

    versions = ALL_VERSIONS
    if args.versions is not None:
        versions = args.versions

    cwd = os.getcwd()
    tmpdir = tempfile.mkdtemp(prefix="ncu_profile_")

    total = len(cases) * (len(versions) + len(V7_MAPPINGS))
    count = 0

    for case in cases:
        name, ptype, ks, st, pad, dil, ceil, cip, div, shape = case
        print(f"\n--- {name} ---")

        for v in versions:
            count += 1
            rows = profile_one_case(
                name, ptype, ks, st, pad, dil, ceil, cip, div, shape,
                v, 0, args.dtype, args.ncu, cwd, all_metric_names
            )
            all_rows.extend(rows)

        for m in V7_MAPPINGS:
            count += 1
            rows = profile_one_case(
                name, ptype, ks, st, pad, dil, ceil, cip, div, shape,
                7, m, args.dtype, args.ncu, cwd, all_metric_names
            )
            all_rows.extend(rows)

    # Write CSV
    if all_rows:
        fieldnames = ["case", "pool_type", "version", "mapping", "dtype", "shape",
                      "kernel_size", "stride", "padding", "metric", "value"]
        with open(args.output, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"\nWrote {len(all_rows)} metric rows to {args.output}")
    else:
        print("\nNo results collected!")

    # Cleanup temp script
    script_path = os.path.join(cwd, "_ncu_bench_script.py")
    if os.path.exists(script_path):
        os.remove(script_path)


if __name__ == "__main__":
    main()
