#!/usr/bin/env python
"""Optimization experiment benchmark harness for AdaptiveKDE.

Usage:
    python tests/run_opt_experiments.py --experiment baseline
    python tests/run_opt_experiments.py --experiment A1-searchsorted
    python tests/run_opt_experiments.py --report
"""
import argparse
import json
import os
import statistics
import sys
import time

import numpy as np

from adaptivekde.sshist import sshist
from adaptivekde.sskernel import sskernel
from adaptivekde.ssvkernel import ssvkernel

# --- Test datasets ---
# 107-point Old Faithful (Azzalini & Bowman 1990)
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

# 1000-point synthetic bimodal
_rs = np.random.RandomState(42)
SYNTHETIC_1K = np.concatenate([
    _rs.normal(2.0, 0.3, 400),
    _rs.normal(4.5, 0.4, 600),
])

DATASETS = {
    "n107": OLD_FAITHFUL,
    "n1000": SYNTHETIC_1K,
}

RESULTS_FILE = os.path.join(os.path.dirname(__file__), "experiment_results.json")

N_WARMUP = 3
N_RUNS_DEFAULT = 20


def bench_function(func, x, seed, n_warmup, n_runs):
    """Run func(x) with warmup, return dict of CPU timing stats."""
    for _ in range(n_warmup):
        if seed is not None:
            np.random.seed(seed)
        func(x)
    cpus = []
    for _ in range(n_runs):
        if seed is not None:
            np.random.seed(seed)
        t0 = time.process_time()
        func(x)
        cpu = time.process_time() - t0
        cpus.append(cpu)
    return {
        "cpu_median": statistics.median(cpus),
        "cpu_mean": statistics.mean(cpus),
        "cpu_std": statistics.stdev(cpus) if len(cpus) > 1 else 0.0,
        "cpu_min": min(cpus),
    }


def run_experiment(n_runs):
    """Run all 3 functions on all datasets, return results dict."""
    results = {}
    funcs = [
        ("sshist", lambda x: sshist(x), None),
        ("sskernel", lambda x: sskernel(x, nbs=200), 0),
        ("ssvkernel", lambda x: ssvkernel(x, nbs=50), 0),
    ]
    for ds_name, ds in DATASETS.items():
        results[ds_name] = {}
        for fname, func, seed in funcs:
            stats = bench_function(func, ds.copy(), seed, N_WARMUP, n_runs)
            results[ds_name][fname] = stats
    return results


def print_results(experiment, results):
    """Print results for one experiment."""
    print(f"\n  Experiment: {experiment}")
    print(f"  {'Dataset':<8} {'sshist':>12} {'sskernel':>12} {'ssvkernel':>12} {'Total':>12}")
    print("  " + "-" * 60)
    for ds_name in DATASETS:
        r = results[ds_name]
        total = sum(r[f]["cpu_median"] for f in r)
        print(f"  {ds_name:<8} {r['sshist']['cpu_median']:>12.4f}"
              f" {r['sskernel']['cpu_median']:>12.4f}"
              f" {r['ssvkernel']['cpu_median']:>12.4f}"
              f" {total:>12.4f}")


def print_report():
    """Print comparison table from all saved experiments."""
    if not os.path.exists(RESULTS_FILE):
        print("No results file found.")
        return

    with open(RESULTS_FILE, "r") as f:
        records = json.load(f)

    if not records:
        print("No experiments recorded.")
        return

    # Find baseline for speedup calculation
    baseline = None
    for rec in records:
        if rec["experiment"] == "baseline":
            baseline = rec["results"]
            break

    print("\n" + "=" * 90)
    print("  AdaptiveKDE Optimization Experiment Results (CPU time, median of 20 runs)")
    print("=" * 90)

    for ds_name in DATASETS:
        print(f"\n  Dataset: {ds_name} (n={len(DATASETS[ds_name])})")
        print(f"  {'Experiment':<28} {'sshist':>9} {'sskernel':>9} {'ssvkernel':>9}"
              f" {'Total':>9} {'Speedup':>8}")
        print("  " + "-" * 76)

        for rec in records:
            exp = rec["experiment"]
            r = rec["results"].get(ds_name, {})
            if not r:
                continue
            sh = r.get("sshist", {}).get("cpu_median", 0)
            sk = r.get("sskernel", {}).get("cpu_median", 0)
            sv = r.get("ssvkernel", {}).get("cpu_median", 0)
            total = sh + sk + sv

            speedup = ""
            if baseline and ds_name in baseline:
                bl = baseline[ds_name]
                bl_total = sum(bl[f]["cpu_median"] for f in bl)
                if total > 0:
                    speedup = f"{bl_total / total:>7.2f}x"

            print(f"  {exp:<28} {sh:>9.4f} {sk:>9.4f} {sv:>9.4f}"
                  f" {total:>9.4f} {speedup:>8}")

    print("\n" + "=" * 90)


def main():
    parser = argparse.ArgumentParser(description="AdaptiveKDE optimization experiments")
    parser.add_argument("--experiment", type=str, help="Experiment label")
    parser.add_argument("--n-runs", type=int, default=N_RUNS_DEFAULT)
    parser.add_argument("--report", action="store_true", help="Print comparison table")
    args = parser.parse_args()

    if args.report:
        print_report()
        return 0

    if not args.experiment:
        parser.error("--experiment or --report required")

    print(f"Python {sys.version.split()[0]}, NumPy {np.__version__}")
    print(f"Experiment: {args.experiment}")
    print(f"Runs: {N_WARMUP} warmup + {args.n_runs} timed")

    results = run_experiment(args.n_runs)
    print_results(args.experiment, results)

    # Load or create results file
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, "r") as f:
            records = json.load(f)
    else:
        records = []

    # Remove previous record with same experiment name (allow re-runs)
    records = [r for r in records if r["experiment"] != args.experiment]

    records.append({
        "experiment": args.experiment,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "python_version": sys.version.split()[0],
        "numpy_version": np.__version__,
        "n_runs": args.n_runs,
        "results": results,
    })

    with open(RESULTS_FILE, "w") as f:
        json.dump(records, f, indent=2)

    print(f"\n  Saved to {RESULTS_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
