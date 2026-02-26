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
