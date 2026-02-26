#!/usr/bin/env python
"""Run all three AdaptiveKDE functions, verify against golden references,
and display timing + output summary.

Usage:
    python tests/run_tests.py
"""
import os
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

RTOL = 1e-10
REF_DIR = os.path.join(os.path.dirname(__file__), "reference_data")
W = 70  # table width


def check_allclose(actual, expected, label):
    """Return (pass: bool, max_rel_err: float)."""
    try:
        np.testing.assert_allclose(actual, expected, rtol=RTOL)
        err = np.max(np.abs((actual - expected) / np.where(expected == 0, 1, expected)))
        return True, err
    except AssertionError:
        err = np.max(np.abs((actual - expected) / np.where(expected == 0, 1, expected)))
        return False, err


def run_sshist(x):
    t0_wall = time.perf_counter()
    t0_cpu = time.process_time()
    optN, optD, edges, C, N = sshist(x)
    wall = time.perf_counter() - t0_wall
    cpu = time.process_time() - t0_cpu

    ref = np.load(os.path.join(REF_DIR, "sshist_ref.npz"))
    tests = []
    tests.append(("optN", optN == int(ref["optN"])))
    tests.append(("optD", check_allclose(optD, float(ref["optD"]), "optD")[0]))
    tests.append(("edges", check_allclose(edges, ref["edges"], "edges")[0]))
    tests.append(("C", check_allclose(C, ref["C"], "C")[0]))

    outputs = [
        ("optN", str(optN)),
        ("optD", f"{optD:.6f}"),
        ("len(edges)", str(len(edges))),
        ("len(C)", str(len(C))),
    ]
    return wall, cpu, tests, outputs


def run_sskernel(x):
    np.random.seed(0)
    t0_wall = time.perf_counter()
    t0_cpu = time.process_time()
    y, t, optw, W, C, confb95, yb = sskernel(x, nbs=200)
    wall = time.perf_counter() - t0_wall
    cpu = time.process_time() - t0_cpu

    ref = np.load(os.path.join(REF_DIR, "sskernel_ref.npz"))
    tests = []
    tests.append(("y", check_allclose(y, ref["y"], "y")[0]))
    tests.append(("t", check_allclose(t, ref["t"], "t")[0]))
    tests.append(("optw", check_allclose(optw, ref["optw"], "optw")[0]))
    tests.append(("W", check_allclose(W, ref["W"], "W")[0]))
    tests.append(("C", check_allclose(C, ref["C"], "C")[0]))
    tests.append(("confb95", check_allclose(confb95, ref["confb95"], "confb95")[0]))
    tests.append(("yb", check_allclose(yb, ref["yb"], "yb")[0]))

    dt = np.diff(t)
    integral = np.sum(0.5 * (y[:-1] + y[1:]) * dt)
    outputs = [
        ("optw", f"{optw:.6f}"),
        ("len(y)", str(len(y))),
        ("integral(y)", f"{integral:.6f}"),
        ("yb.shape", str(yb.shape)),
    ]
    return wall, cpu, tests, outputs


def run_ssvkernel(x):
    np.random.seed(0)
    t0_wall = time.perf_counter()
    t0_cpu = time.process_time()
    y, t, optw, gs, C, confb95, yb = ssvkernel(x, nbs=50)
    wall = time.perf_counter() - t0_wall
    cpu = time.process_time() - t0_cpu

    ref = np.load(os.path.join(REF_DIR, "ssvkernel_ref.npz"))
    tests = []
    tests.append(("y", check_allclose(y, ref["y"], "y")[0]))
    tests.append(("t", check_allclose(t, ref["t"], "t")[0]))
    tests.append(("optw", check_allclose(optw, ref["optw"], "optw")[0]))
    tests.append(("gs", check_allclose(gs, ref["gs"], "gs")[0]))
    tests.append(("C", check_allclose(C, ref["C"], "C")[0]))
    tests.append(("confb95", check_allclose(confb95, ref["confb95"], "confb95")[0]))
    tests.append(("yb", check_allclose(yb, ref["yb"], "yb")[0]))

    dt = np.diff(t)
    integral = np.sum(0.5 * (y[:-1] + y[1:]) * dt)
    outputs = [
        ("optw range", f"[{optw.min():.4f}, {optw.max():.4f}]"),
        ("len(y)", str(len(y))),
        ("integral(y)", f"{integral:.6f}"),
        ("gs.shape", str(gs.shape)),
        ("yb.shape", str(yb.shape)),
    ]
    return wall, cpu, tests, outputs


def main():
    x = OLD_FAITHFUL.copy()

    functions = [
        ("sshist", run_sshist),
        ("sskernel", run_sskernel),
        ("ssvkernel", run_ssvkernel),
    ]

    all_results = []
    for name, func in functions:
        wall, cpu, tests, outputs = func(x)
        all_results.append((name, wall, cpu, tests, outputs))

    # --- Timing table ---
    print()
    print("=" * W)
    print(f"  AdaptiveKDE Summary  (n={len(x)} Old Faithful samples, rtol={RTOL})")
    print("=" * W)
    print()
    print("  Timing")
    print("  " + "-" * (W - 4))
    print(f"  {'Function':<12} {'Wall (s)':>10} {'CPU (s)':>10}")
    print("  " + "-" * (W - 4))
    total_wall = 0.0
    total_cpu = 0.0
    for name, wall, cpu, tests, outputs in all_results:
        print(f"  {name:<12} {wall:>10.4f} {cpu:>10.4f}")
        total_wall += wall
        total_cpu += cpu
    print("  " + "-" * (W - 4))
    print(f"  {'Total':<12} {total_wall:>10.4f} {total_cpu:>10.4f}")
    print()

    # --- Per-function output + test results ---
    total_pass = 0
    total_tests = 0
    for name, wall, cpu, tests, outputs in all_results:
        n_pass = sum(1 for _, ok in tests if ok)
        n_total = len(tests)
        total_pass += n_pass
        total_tests += n_total
        status = "PASS" if n_pass == n_total else "FAIL"

        print(f"  {name}  [{status}: {n_pass}/{n_total} golden checks]")
        print("  " + "-" * (W - 4))

        # Output values
        for key, val in outputs:
            print(f"    {key:<16} {val}")

        # Golden test results
        print()
        for field, ok in tests:
            mark = "PASS" if ok else "FAIL"
            print(f"    {field:<16} {mark}")
        print()

    # --- Final summary ---
    all_ok = total_pass == total_tests
    print("=" * W)
    if all_ok:
        print(f"  ALL PASSED  ({total_pass}/{total_tests} golden checks)")
    else:
        print(f"  FAILURES    ({total_pass}/{total_tests} golden checks passed)")
    print("=" * W)
    print()

    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
