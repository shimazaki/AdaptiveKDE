#!/usr/bin/env python
"""Benchmark runner for AdaptiveKDE functions.

Runs each function N times, records wall/CPU mean+std, and appends results
to tests/benchmarks.json.

Usage:
    python tests/run_benchmarks.py [--n-runs 10]
"""
import argparse
import json
import os
import platform
import socket
import sys
import time

import numpy as np

from adaptivekde.sshist import sshist
from adaptivekde.sskernel import sskernel
from adaptivekde.ssvkernel import ssvkernel

# Old Faithful eruption durations (minutes) — 107 observations
# Source: Azzalini & Bowman (1990)
OLD_FAITHFUL = np.array([
    4.37, 3.87, 4.00, 4.03, 3.50, 4.08, 2.25, 4.70, 1.73, 4.93,
    1.73, 4.62, 3.43, 4.25, 1.68, 3.92, 3.68, 3.10, 4.03, 1.77,
    4.08, 1.75, 3.20, 1.85, 4.62, 1.97, 4.50, 3.92, 4.35, 2.33,
    3.83, 1.88, 4.60, 1.80, 4.73, 1.77, 4.57, 1.85, 3.52, 4.00,
    3.70, 3.72, 4.25, 3.58, 3.80, 3.77, 3.75, 2.50, 4.50, 4.10,
    3.70, 3.80, 3.43, 4.00, 2.27, 4.40, 4.05, 4.25, 3.33, 2.00,
    4.33, 2.93, 4.58, 1.90, 3.58, 3.73, 3.73, 1.82, 4.63, 3.50,
    4.00, 3.67, 1.67, 4.60, 1.67, 4.00, 1.80, 4.42, 1.90, 4.63,
    2.93, 3.50, 1.97, 4.28, 1.83, 4.13, 1.83, 4.65, 4.20, 3.93,
    4.33, 1.83, 4.53, 2.03, 4.18, 4.43, 4.07, 4.13, 3.95, 4.10,
    2.72, 4.58, 1.90, 4.50, 1.95, 4.83, 4.12,
])

BENCHMARKS_FILE = os.path.join(os.path.dirname(__file__), "benchmarks.json")


def bench_sshist(x):
    """Benchmark sshist (deterministic, no seed needed)."""
    t0_wall = time.perf_counter()
    t0_cpu = time.process_time()
    sshist(x)
    wall = time.perf_counter() - t0_wall
    cpu = time.process_time() - t0_cpu
    return wall, cpu


def bench_sskernel(x):
    """Benchmark sskernel."""
    np.random.seed(0)
    t0_wall = time.perf_counter()
    t0_cpu = time.process_time()
    sskernel(x, nbs=200)
    wall = time.perf_counter() - t0_wall
    cpu = time.process_time() - t0_cpu
    return wall, cpu


def bench_ssvkernel(x):
    """Benchmark ssvkernel."""
    np.random.seed(0)
    t0_wall = time.perf_counter()
    t0_cpu = time.process_time()
    ssvkernel(x, nbs=50)
    wall = time.perf_counter() - t0_wall
    cpu = time.process_time() - t0_cpu
    return wall, cpu


def run_benchmarks(n_runs):
    x = OLD_FAITHFUL.copy()

    functions = [
        ("sshist", bench_sshist),
        ("sskernel", bench_sskernel),
        ("ssvkernel", bench_ssvkernel),
    ]

    # Warmup run (not counted)
    for _, func in functions:
        func(x)

    results = {}
    for name, func in functions:
        walls = []
        cpus = []
        for _ in range(n_runs):
            wall, cpu = func(x)
            walls.append(wall)
            cpus.append(cpu)
        results[name] = {
            "wall_mean": float(np.mean(walls)),
            "wall_std": float(np.std(walls)),
            "cpu_mean": float(np.mean(cpus)),
            "cpu_std": float(np.std(cpus)),
        }

    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark AdaptiveKDE functions")
    parser.add_argument("--n-runs", type=int, default=10,
                        help="Number of benchmark runs per function (default: 10)")
    args = parser.parse_args()

    print(f"Python {sys.version}")
    print(f"NumPy  {np.__version__}")
    print(f"Machine: {socket.gethostname()}")
    print(f"Running {args.n_runs} benchmark iterations per function...")
    print()

    results = run_benchmarks(args.n_runs)

    # Build record
    record = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "machine": socket.gethostname(),
        "n_runs": args.n_runs,
        "functions": results,
    }

    # Load existing or create new
    if os.path.exists(BENCHMARKS_FILE):
        with open(BENCHMARKS_FILE, "r") as f:
            data = json.load(f)
    else:
        data = []

    data.append(record)

    with open(BENCHMARKS_FILE, "w") as f:
        json.dump(data, f, indent=2)

    # Print summary
    print(f"  {'Function':<12} {'Wall mean':>10} {'Wall std':>10} {'CPU mean':>10} {'CPU std':>10}")
    print("  " + "-" * 56)
    for name, stats in results.items():
        print(f"  {name:<12} {stats['wall_mean']:>10.4f} {stats['wall_std']:>10.4f} "
              f"{stats['cpu_mean']:>10.4f} {stats['cpu_std']:>10.4f}")

    print()
    print(f"Results appended to {BENCHMARKS_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
