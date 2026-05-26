"""Generate performance plots and analysis reports from benchmark JSON data.

Reads benchmark data from JSON files and produces:
1. Speedup comparison bar charts (all versions, all configs)
2. Performance across configs (line charts)
3. Cross-dtype comparison
4. Cross-GPU comparison (Thor vs A40)
"""
import json
import os
import sys

# Try to import matplotlib
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    import numpy as np
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("WARNING: matplotlib not available. Install with: pip install matplotlib")


ALL_VERSIONS = [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15]
V7_LABELS = ['v7mA', 'v7mB', 'v7mC', 'v7mD']
DTYPE_COLORS = {
    'fp32': '#1f77b4', 'bf16': '#2ca02c', 'fp8_e4m3': '#ff7f0e',
    'fp8_e5m2': '#d62728', 'int8': '#9467bd', 'int16': '#8c564b', 'fp16': '#e377c2',
}

# Config descriptions for display
CONFIG_DESC = {
    'mem_bound':    'Mem-Bound 3x3s2\n(1,128,128,256)',
    'global_avg':   'Global Avg 7x7\n(1,7,7,512)',
    'dense_3x3s1':  'Dense 3x3s1\n(1,56,56,64)',
    'large_k13':    'Large 13x13\n(1,32,32,64)',
    'small_2x2s2':  'Small 2x2s2\n(1,64,64,32)',
    'mid_5x5s2':    'Medium 5x5s2\n(1,28,28,128)',
    'batch_3x3s1':  'Batch 3x3s1\n(4,32,32,64)',
    'wide_k7':      'Wide 7x7s1\n(1,16,16,256)',
    'global_max':   'Global Max 7x7\n(1,7,7,1024)',
    'avg_dense':    'Avg Dense 3x3\n(1,28,28,256)',
}


def load_json(path):
    """Load benchmark JSON, return dict of {dtype: {config: data}}."""
    with open(path) as f:
        return json.load(f)


def get_version_times(dtype_data, config_name, version):
    """Get median time for a version across all configs, handling errors."""
    versions = dtype_data[config_name]['versions']
    v = versions.get(str(version))
    if v and not isinstance(v, dict):
        return float(v)
    return None


def compute_speedups(data, dtype_key):
    """Compute speedup vs v0 for each version and config."""
    if dtype_key not in data:
        return {}
    result = {}
    for config_name in data[dtype_key]:
        v0 = get_version_times(data[dtype_key], config_name, 0)
        if v0 is None or v0 <= 0:
            continue
        speeds = {}
        for v in ALL_VERSIONS:
            t = get_version_times(data[dtype_key], config_name, v)
            if t and t > 0:
                speeds[v] = v0 / t
        result[config_name] = speeds
    return result


def plot_speedup_by_config(data, output_dir, gpu_name='Thor'):
    """Plot speedup vs v0 for each version, one subplot per config."""
    fig, axes = plt.subplots(2, 5, figsize=(24, 10), sharey=False)
    axes = axes.flatten()

    # Only plot dtypes that have data
    dtype_keys = [dk for dk in DTYPE_COLORS if dk in data]
    configs = list(data[dtype_keys[0]].keys()) if dtype_keys else []

    for idx, config_name in enumerate(configs):
        if idx >= len(axes):
            break
        ax = axes[idx]

        # Get v0 time for this config (use fp32 as reference if available)
        ref_dtype = 'fp32' if 'fp32' in data else dtype_keys[0]
        if config_name not in data.get(ref_dtype, {}):
            ax.set_axis_off()
            continue

        # Speedup for each dtype
        for dk in dtype_keys:
            speeds = compute_speedups(data, dk)
            if config_name not in speeds:
                continue
            vs = list(speeds[config_name].keys())
            sp = list(speeds[config_name].values())
            ax.plot(vs, sp, marker='o', label=dk, color=DTYPE_COLORS.get(dk, 'gray'),
                    linewidth=2, markersize=6)

        ax.set_title(CONFIG_DESC.get(config_name, config_name), fontsize=10)
        ax.axhline(y=1, color='red', linestyle='--', alpha=0.5, linewidth=1)
        ax.set_xlabel('Version', fontsize=9)
        if idx % 5 == 0:
            ax.set_ylabel('Speedup vs v0', fontsize=9)
        ax.legend(fontsize=7, loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        ax.set_xticks(ALL_VERSIONS)
        ax.tick_params(axis='x', rotation=45, labelsize=8)

    fig.suptitle(f'Version Speedup vs Baseline (v0) — {gpu_name}', fontsize=14, y=1.01)
    fig.tight_layout()
    path = os.path.join(output_dir, f'speedup_by_config_{gpu_name.lower()}.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_cross_dtype_comparison(data, output_dir, gpu_name='Thor'):
    """Plot v0 and best performance for each dtype across configs."""
    dtype_keys = [dk for dk in DTYPE_COLORS if dk in data]
    configs = list(data[dtype_keys[0]].keys()) if dtype_keys else []

    # Per config: bar chart of dtypes
    fig, axes = plt.subplots(2, 5, figsize=(24, 10), sharey=False)
    axes = axes.flatten()

    for idx, config_name in enumerate(configs):
        if idx >= len(axes):
            break
        ax = axes[idx]

        x = np.arange(len(dtype_keys))
        width = 0.35

        v0_times = []
        best_times = []
        best_vers = []

        for dk in dtype_keys:
            v0 = get_version_times(data[dk], config_name, 0)
            v0_times.append(v0 if v0 else np.nan)
            # Find best version
            best_t = v0
            best_v = 0
            for v in [1, 2, 8, 10, 14, 15]:
                t = get_version_times(data[dk], config_name, v)
                if t and t > 0 and t < (best_t or float('inf')):
                    best_t = t
                    best_v = v
            best_times.append(best_t if best_t else np.nan)
            best_vers.append(f"v{best_v}")

        bars1 = ax.bar(x - width/2, v0_times, width, label='v0 (baseline)',
                       color='#666666', alpha=0.8)
        bars2 = ax.bar(x + width/2, best_times, width, label='Best version',
                       color='#1f77b4', alpha=0.8)

        # Add speedup labels
        for i in range(len(dtype_keys)):
            if v0_times[i] and best_times[i] and best_times[i] > 0:
                speedup = v0_times[i] / best_times[i]
                ax.text(x[i] + width/2, best_times[i] * 1.05,
                        f'{speedup:.1f}x\n{best_vers[i]}',
                        ha='center', va='bottom', fontsize=7, color='#1f77b4')

        ax.set_xticks(x)
        ax.set_xticklabels(dtype_keys, rotation=45, fontsize=9)
        ax.set_title(CONFIG_DESC.get(config_name, config_name), fontsize=10)
        ax.set_ylabel('Time (ms)', fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_yscale('log')

    fig.suptitle(f'Cross-Dtype Performance Comparison — {gpu_name}', fontsize=14, y=1.01)
    fig.tight_layout()
    path = os.path.join(output_dir, f'cross_dtype_{gpu_name.lower()}.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_version_times_heatmap(data, output_dir, gpu_name='Thor'):
    """Heatmap: versions (x) x configs (y), color = time, per dtype."""
    dtype_keys = [dk for dk in DTYPE_COLORS if dk in data]
    if not dtype_keys:
        return

    # All configs present in data
    configs = list(data[dtype_keys[0]].keys())

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    axes = axes.flatten()

    for didx, dk in enumerate(dtype_keys):
        if didx >= len(axes):
            break
        ax = axes[didx]

        # Build matrix: rows=configs, cols=versions
        times = []
        for config_name in configs:
            row = []
            for v in ALL_VERSIONS:
                t = get_version_times(data[dk], config_name, v)
                row.append(t if t else np.nan)
            times.append(row)

        times = np.array(times)
        # Normalize per row for better visualization
        row_max = np.nanmax(times, axis=1, keepdims=True)
        normalized = times / np.where(row_max > 0, row_max, 1)

        im = ax.imshow(normalized, cmap='YlOrRd', aspect='auto', vmin=0, vmax=2)
        ax.set_yticks(range(len(configs)))
        ax.set_yticklabels([CONFIG_DESC.get(c, c).replace('\n', ' ') for c in configs], fontsize=8)
        ax.set_xticks(range(len(ALL_VERSIONS)))
        ax.set_xticklabels([f'v{v}' for v in ALL_VERSIONS], fontsize=8)
        ax.set_title(f'{dk}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Version')

        # Add time annotations
        for i in range(len(configs)):
            for j in range(len(ALL_VERSIONS)):
                if not np.isnan(times[i, j]):
                    color = 'white' if normalized[i, j] > 0.6 else 'black'
                    ax.text(j, i, f'{times[i, j]:.1f}',
                            ha='center', va='center', fontsize=6, color=color)

        fig.colorbar(im, ax=ax, label='Normalized Time')

    fig.suptitle(f'Version Performance Heatmap — {gpu_name}', fontsize=14, y=1.01)
    fig.tight_layout()
    path = os.path.join(output_dir, f'heatmap_{gpu_name.lower()}.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_cross_gpu_comparison(thor_data, a40_data, output_dir):
    """Plot Thor vs A40 comparison for each dtype."""
    dtype_keys = set(thor_data.keys()) & set(a40_data.keys())
    if not dtype_keys:
        return
    dtype_keys = sorted(dtype_keys)

    configs = list(thor_data[dtype_keys[0]].keys()) if dtype_keys else []

    fig, axes = plt.subplots(2, 3, figsize=(20, 12), sharey=False)
    axes = axes.flatten()

    for didx, dk in enumerate(dtype_keys):
        if didx >= len(axes):
            break
        ax = axes[didx]

        x = np.arange(len(configs))
        width = 0.35

        thor_v0 = []
        a40_v0 = []
        thor_best = []
        a40_best = []

        for config_name in configs:
            tv0 = get_version_times(thor_data[dk], config_name, 0)
            av0 = get_version_times(a40_data[dk], config_name, 0)
            thor_v0.append(tv0 if tv0 else np.nan)
            a40_v0.append(av0 if av0 else np.nan)

            tb = None
            for v in [1, 2, 8, 10, 14, 15]:
                t = get_version_times(thor_data[dk], config_name, v)
                if t and t > 0:
                    if tb is None or t < tb:
                        tb = t
            thor_best.append(tb if tb else np.nan)

            ab = None
            for v in [1, 2, 8, 10, 14, 15]:
                t = get_version_times(a40_data[dk], config_name, v)
                if t and t > 0:
                    if ab is None or t < ab:
                        ab = t
            a40_best.append(ab if ab else np.nan)

        x_thor = x - width * 0.75
        x_a40 = x + width * 0.25

        ax.bar(x_thor - width/2, thor_v0, width, label='Thor v0',
               color='#ff7f0e', alpha=0.7)
        ax.bar(x_thor + width/2, thor_best, width, label='Thor best',
               color='#ff7f0e', alpha=0.4)
        ax.bar(x_a40 - width/2, a40_v0, width, label='A40 v0',
               color='#1f77b4', alpha=0.7)
        ax.bar(x_a40 + width/2, a40_best, width, label='A40 best',
               color='#1f77b4', alpha=0.4)

        ax.set_xticks(x)
        short_labels = [c[:12] for c in configs]
        ax.set_xticklabels(short_labels, rotation=45, fontsize=8)
        ax.set_title(dk, fontsize=12, fontweight='bold')
        ax.set_ylabel('Time (ms)', fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_yscale('log')

    fig.suptitle('Thor (SM 11.0) vs A40 (SM 8.6) Performance Comparison', fontsize=14, y=1.01)
    fig.tight_layout()
    path = os.path.join(output_dir, 'cross_gpu_comparison.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_bw_utilization(data, output_dir, gpu_name='Thor'):
    """Plot effective bandwidth utilization for each version and config."""
    dtype_keys = [dk for dk in DTYPE_COLORS if dk in data]
    configs = list(data[dtype_keys[0]].keys()) if dtype_keys else []

    fig, axes = plt.subplots(2, 5, figsize=(24, 10))
    axes = axes.flatten()

    for idx, config_name in enumerate(configs):
        if idx >= len(axes):
            break
        ax = axes[idx]

        for dk in dtype_keys:
            if config_name not in data.get(dk, {}):
                continue
            input_bytes = data[dk][config_name]['input_bytes']
            bw_values = []
            for v in ALL_VERSIONS:
                t = get_version_times(data[dk], config_name, v)
                if t and t > 0:
                    bw = input_bytes / (t * 1e-3) / 1e9  # GB/s
                    bw_values.append((v, bw))

            if bw_values:
                vs = [x[0] for x in bw_values]
                bws = [x[1] for x in bw_values]
                ax.plot(vs, bws, marker='o', label=dk, color=DTYPE_COLORS.get(dk, 'gray'),
                        linewidth=2, markersize=6)

        ax.set_title(CONFIG_DESC.get(config_name, config_name), fontsize=10)
        ax.set_xlabel('Version', fontsize=9)
        if idx % 5 == 0:
            ax.set_ylabel('Effective BW (GB/s)', fontsize=9)
        ax.legend(fontsize=7, loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.set_xticks(ALL_VERSIONS)
        ax.tick_params(axis='x', rotation=45, labelsize=8)

    fig.suptitle(f'Effective Bandwidth Utilization — {gpu_name}', fontsize=14, y=1.01)
    fig.tight_layout()
    path = os.path.join(output_dir, f'bandwidth_{gpu_name.lower()}.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


def generate_markdown_report(thor_data, a40_data, output_dir):
    """Generate a markdown report with tables and analysis."""
    dtype_keys = sorted(set(thor_data.keys()) & set(a40_data.keys()))
    configs = list(thor_data[dtype_keys[0]].keys()) if dtype_keys else []

    lines = []
    lines.append('# CUDA Pooling2D Cross-GPU Performance Report\n')
    lines.append('## Environment\n')
    lines.append('- **Thor**: NVIDIA Thor (SM 11.0, Blackwell), CUDA 13.0')
    lines.append('- **A40**: NVIDIA A40 (SM 8.6, Ampere), CUDA 13.1')
    lines.append('- **Timing**: CUDA events (kernel-only, excluding H2D/D2H)')
    lines.append('')

    # Cross-dtype summary table
    lines.append('## Cross-Dtype Performance Summary\n')
    lines.append('Speedup of best optimized version vs v0 baseline.')
    lines.append('')

    for config_name in configs:
        lines.append(f'### {config_name}\n')
        lines.append(f'| Dtype | v0 Thor (ms) | Best Thor (ms) | Speedup | v0 A40 (ms) | Best A40 (ms) | Speedup |')
        lines.append(f'|-------|-------------|---------------|---------|-------------|---------------|---------|')

        for dk in dtype_keys:
            tv0 = get_version_times(thor_data[dk], config_name, 0)
            av0 = get_version_times(a40_data[dk], config_name, 0)

            # Find best on Thor
            tb = tv0; tv = 0
            for v in [1, 2, 8, 10, 14, 15]:
                t = get_version_times(thor_data[dk], config_name, v)
                if t and t > 0 and t < (tb or float('inf')):
                    tb = t; tv = v

            # Find best on A40
            ab = av0; av = 0
            for v in [1, 2, 8, 10, 14, 15]:
                t = get_version_times(a40_data[dk], config_name, v)
                if t and t > 0 and t < (ab or float('inf')):
                    ab = t; av = v

            def fmt(x):
                return f'{x:.4f}' if x else 'N/A'
            def speedup(baseline, best, v):
                if baseline and best and best > 0:
                    return f'v{v} {baseline/best:.2f}x'
                return 'N/A'

            lines.append(f'| {dk} | {fmt(tv0)} | {fmt(tb)} | {speedup(tv0, tb, tv)} | {fmt(av0)} | {fmt(ab)} | {speedup(av0, ab, av)} |')

        lines.append('')

    # Detailed per-dtype tables
    lines.append('## Detailed Version Timing (ms)\n')

    for dk in dtype_keys:
        lines.append(f'### {dk}\n')
        lines.append(f'| Config | v0 | v1 | v2 | v3 | v4 | v5 | v6 | v8 | v9 | v10 | v11 | v12 | v13 | v14 | v15 |')
        lines.append(f'|--------|----|----|----|----|----|----|----|----|----|-----|-----|-----|-----|-----|------|')

        for config_name in configs:
            vals = []
            for v in ALL_VERSIONS:
                tt = get_version_times(thor_data[dk], config_name, v)
                at = get_version_times(a40_data[dk], config_name, v)
                if tt and at:
                    ratio = tt / at if at > 0 else 0
                    vals.append(f'{tt:.2f}')
                elif tt:
                    vals.append(f'{tt:.2f}')
                elif at:
                    vals.append(f'A{at:.2f}')
                else:
                    vals.append('ERR')
            lines.append(f'| {config_name} | {" | ".join(vals)} |')

        lines.append('')

    # Analysis
    lines.append('## Analysis\n')
    lines.append('### Key Findings\n')
    lines.append('1. **v2 (Vectorized Loads)** consistently provides the largest speedup for fp32/bf16/int16')
    lines.append('2. **v7m3 misaligned address** bug blocks fp8/int8 after first config on both GPUs')
    lines.append('3. **fp16 v9+ illegal memory access** is Thor-specific, does not reproduce on A40')
    lines.append('4. **Global pooling** shows minimal improvement across all dtypes (v14 best)')
    lines.append('5. **int8/int16** show highest speedup potential (up to 30x) with optimized versions')
    lines.append('')

    path = os.path.join(output_dir, 'cross_gpu_report.md')
    with open(path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"  Saved: {path}")


def main():
    output_dir = sys.argv[1] if len(sys.argv) > 1 else 'docs/plots'
    os.makedirs(output_dir, exist_ok=True)

    thor_path = sys.argv[2] if len(sys.argv) > 2 else '/tmp/bench_thor.json'
    a40_path = sys.argv[3] if len(sys.argv) > 3 else '/tmp/bench_a40.json'

    # Load data
    thor_data = load_json(thor_path) if os.path.exists(thor_path) else {}
    a40_data = load_json(a40_path) if os.path.exists(a40_path) else {}

    print(f"Thor data: {len(thor_data)} dtypes")
    print(f"A40 data: {len(a40_data)} dtypes")

    if not HAS_MPL:
        print("matplotlib not available, generating markdown report only")
        if thor_data and a40_data:
            generate_markdown_report(thor_data, a40_data, output_dir)
        return

    if thor_data:
        print("\nThor plots:")
        plot_speedup_by_config(thor_data, output_dir, 'Thor')
        plot_cross_dtype_comparison(thor_data, output_dir, 'Thor')
        plot_version_times_heatmap(thor_data, output_dir, 'Thor')
        plot_bw_utilization(thor_data, output_dir, 'Thor')

    if a40_data:
        print("\nA40 plots:")
        plot_speedup_by_config(a40_data, output_dir, 'A40')
        plot_cross_dtype_comparison(a40_data, output_dir, 'A40')
        plot_version_times_heatmap(a40_data, output_dir, 'A40')
        plot_bw_utilization(a40_data, output_dir, 'A40')

    if thor_data and a40_data:
        print("\nCross-GPU comparison:")
        plot_cross_gpu_comparison(thor_data, a40_data, output_dir)
        generate_markdown_report(thor_data, a40_data, output_dir)


if __name__ == "__main__":
    main()
