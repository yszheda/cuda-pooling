"""Nsight Systems timeline profiling for CUDA Pooling2D kernels.

Runs nsys profile with NVTX annotations visible in the timeline.
Generates .nsys-rep files per benchmark case for visual inspection.
"""

import numpy as np
import subprocess
import sys
import os
import argparse
import time

NSYS_PATH = "/usr/local/bin/nsys"

# Same profiling cases as profile_ncu.py
PROFILING_CASES = [
    ("small_3x3s2p1_max", "max", 3, 2, 1, 1, False, None, None, (1, 32, 32, 64)),
    ("small_3x3s2p1_avg", "avg", 3, 2, 1, 1, False, True, None, (1, 32, 32, 64)),
    ("large_3x3s2p1_max", "max", 3, 2, 1, 1, False, None, None, (1, 128, 128, 256)),
    ("large_3x3s2p1_avg", "avg", 3, 2, 1, 1, False, True, None, (1, 128, 128, 256)),
    ("largeC_3x3s2p1_max", "max", 3, 2, 1, 1, False, None, None, (1, 28, 28, 512)),
    ("largeC_3x3s2p1_avg", "avg", 3, 2, 1, 1, False, True, None, (1, 28, 28, 512)),
    ("resnet_maxpool", "max", 3, 2, 1, 1, False, None, None, (1, 56, 56, 64)),
    ("resnet_global", "avg", 7, 1, 0, 1, False, True, None, (1, 7, 7, 512)),
    ("vgg_maxpool", "max", 2, 2, 0, 1, False, None, None, (1, 112, 112, 128)),
    ("yolo_sppf", "max", 5, 1, 2, 1, False, None, None, (1, 20, 20, 512)),
]

ALL_VERSIONS = [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15]
V7_MAPPINGS = [0, 1, 2, 3]


def write_nsys_script(pool_type, dtype, shape, kernel_size, stride, padding,
                      dilation, ceil_mode, count_include_pad, divisor_override,
                      warmup, iters, out_path):
    """Write a Python script that runs all versions for nsys profiling."""
    N, H, W, C = shape
    lines = [
        "import numpy as np",
        "import sys",
        "sys.path.insert(0, 'build')",
        "import _pooling",
        "",
        f"x = np.random.randn({N}, {H}, {W}, {C}).astype(np.{dtype})",
        "",
        "_pooling.cuda_synchronize()",
        "",
    ]

    # Warmup with v0
    fn_prefix = "_pooling.maxpool2d_f32" if pool_type == "max" else "_pooling.avgpool2d_f32"
    if dtype == "float16":
        fn_prefix = "_pooling.maxpool2d_f16" if pool_type == "max" else "_pooling.avgpool2d_f16"

    for _ in range(warmup):
        if pool_type == "max":
            lines.append(f"{fn_prefix}(x, {kernel_size}, {stride}, {padding}, {dilation}, {ceil_mode}, 0, 0)")
        else:
            cip_str = str(count_include_pad) if count_include_pad is not None else "True"
            div_str = str(divisor_override) if divisor_override is not None else "None"
            lines.append(f"{fn_prefix}(x, {kernel_size}, {stride}, {padding}, {dilation}, {ceil_mode}, {cip_str}, {div_str}, 0, 0)")

    lines.append("_pooling.cuda_synchronize()")
    lines.append("")

    # Profile each version
    for v in ALL_VERSIONS:
        if pool_type == "max":
            lines.append(f"# Version {v}")
            lines.append(f"{fn_prefix}(x, {kernel_size}, {stride}, {padding}, {dilation}, {ceil_mode}, {v}, 0)")
        else:
            lines.append(f"# Version {v}")
            lines.append(f"{fn_prefix}(x, {kernel_size}, {stride}, {padding}, {dilation}, {ceil_mode}, {cip_str}, {div_str}, {v}, 0)")

    for m in V7_MAPPINGS:
        if pool_type == "max":
            lines.append(f"# Version 7 mapping {m}")
            lines.append(f"{fn_prefix}(x, {kernel_size}, {stride}, {padding}, {dilation}, {ceil_mode}, 7, {m})")
        else:
            lines.append(f"# Version 7 mapping {m}")
            lines.append(f"{fn_prefix}(x, {kernel_size}, {stride}, {padding}, {dilation}, {ceil_mode}, {cip_str}, {div_str}, 7, {m})")

    lines.append("_pooling.cuda_synchronize()")

    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")


def run_nsys(case_name, script_path, output_dir, cwd):
    """Run nsys profile and save the .nsys-rep file."""
    output_path = os.path.join(output_dir, f"{case_name}")
    cmd = [
        NSYS_PATH, "profile",
        "--trace=cuda,nvtx",
        "--gpu-metrics-device=all",
        "--output", output_path,
        "--force-overwrite", "true",
        "--duration", "10",
        sys.executable, script_path,
    ]

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=120,
            cwd=cwd, env={**os.environ, "CUDA_VISIBLE_DEVICES": "0"}
        )
        nsys_rep = output_path + ".nsys-rep"
        if os.path.exists(nsys_rep):
            print(f"  {case_name}: OK ({nsys_rep})")
            return nsys_rep
        else:
            print(f"  {case_name}: no .nsys-rep generated", file=sys.stderr)
            return None
    except subprocess.TimeoutExpired:
        print(f"  {case_name}: TIMEOUT", file=sys.stderr)
        return None


def main():
    parser = argparse.ArgumentParser(description="Nsight Systems profiling for CUDA Pooling2D")
    parser.add_argument("--output", default="nsys_profiles", help="Output directory for .nsys-rep files")
    parser.add_argument("--nsys", default=NSYS_PATH, help="Path to nsys binary")
    parser.add_argument("--dtype", choices=["float32", "float16"], default="float32")
    parser.add_argument("--cases", nargs="*", help="Subset of case names to profile")
    parser.add_argument("--warmup", type=int, default=3, help="Warmup iterations")
    parser.add_argument("--iters", type=int, default=1, help="Profile iterations per version")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    cwd = os.getcwd()

    cases = PROFILING_CASES
    if args.cases:
        cases = [c for c in cases if c[0] in args.cases]

    for case in cases:
        name, ptype, ks, st, pad, dil, ceil, cip, div, shape = case
        print(f"\n--- {name} ---")

        script_path = os.path.join(cwd, f"_nsys_bench_{name}.py")
        write_nsys_script(
            ptype, args.dtype, shape, ks, st, pad, dil,
            ceil, cip, div, args.warmup, args.iters, script_path
        )

        run_nsys(name, script_path, args.output, cwd)

        if os.path.exists(script_path):
            os.remove(script_path)

    print(f"\nDone. Profile files in {args.output}/")
    print("Open with: nsys-ui <file>.nsys-rep")


if __name__ == "__main__":
    main()
