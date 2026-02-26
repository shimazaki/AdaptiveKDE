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
OLD_FAITHFUL = np.array([
    1.80, 1.80, 1.80, 1.85, 1.87, 1.90, 1.92, 1.93, 1.93, 1.95,
    1.95, 1.97, 2.00, 2.03, 2.05, 2.07, 2.08, 2.10, 2.10, 2.13,
    2.15, 2.17, 2.18, 2.20, 2.22, 2.23, 2.25, 2.27, 2.30, 2.30,
    2.33, 2.33, 2.35, 2.37, 2.40, 2.42, 2.43, 2.45, 2.47, 2.50,
    2.50, 2.52, 2.55, 2.57, 2.58, 2.60, 2.63, 2.65, 2.67, 2.70,
    2.72, 2.73, 2.75, 2.78, 2.80, 2.83, 2.85, 2.87, 2.90, 2.92,
    2.93, 2.95, 2.97, 3.00, 3.02, 3.03, 3.05, 3.07, 3.08, 3.10,
    3.15, 3.17, 3.20, 3.22, 3.25, 3.27, 3.30, 3.33, 3.35, 3.37,
    3.40, 3.42, 3.45, 3.47, 3.50, 3.53, 3.55, 3.57, 3.60, 3.63,
    3.65, 3.67, 3.70, 3.72, 3.75, 3.78, 3.80, 3.83, 3.85, 3.88,
    3.90, 3.93, 3.95, 3.98, 4.00, 4.02, 4.05,
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
