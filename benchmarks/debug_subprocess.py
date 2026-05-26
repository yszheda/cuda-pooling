import sys
import os
import subprocess
import json

BENCH_DIR = os.path.dirname(os.path.abspath(__file__))

dtype_key = 'fp32'
cmd = [sys.executable, os.path.join(BENCH_DIR, 'bench_dtype.py'), dtype_key]
print(f"cmd: {cmd}", file=sys.stderr)
result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
print(f"returncode: {result.returncode}", file=sys.stderr)
print(f"stdout len: {len(result.stdout)}", file=sys.stderr)
print(f"stderr: {result.stderr[:300]}", file=sys.stderr)
print(f"stdout first 100: {result.stdout[:100]}", file=sys.stderr)
try:
    data = json.loads(result.stdout.strip())
    print(f"JSON keys: {list(data.keys())}", file=sys.stderr)
    print("SUCCESS", file=sys.stderr)
except Exception as e:
    print(f"JSON error: {e}", file=sys.stderr)
