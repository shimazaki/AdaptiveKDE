import os
import time
import numpy as np
import pytest
from adaptivekde.sshist import sshist


class TestSshistGolden:
    """Compare sshist output against saved reference data."""

    def test_golden_match(self, faithful, ref_dir):
        t0_wall = time.perf_counter()
        t0_cpu = time.process_time()
        optN, optD, edges, C, N = sshist(faithful)
        wall = time.perf_counter() - t0_wall
        cpu = time.process_time() - t0_cpu
        print(f"\n  sshist: {wall:.4f} s (cpu: {cpu:.4f} s)")
        ref = np.load(os.path.join(ref_dir, "sshist_ref.npz"))

        assert optN == int(ref["optN"])
        np.testing.assert_allclose(optD, float(ref["optD"]), rtol=1e-10)
        np.testing.assert_allclose(edges, ref["edges"], rtol=1e-10)
        np.testing.assert_allclose(C, ref["C"], rtol=1e-10)


class TestSshistProperties:
    """Property-based sanity checks for sshist."""

    def test_optN_positive_integer(self, faithful):
        optN, optD, edges, C, N = sshist(faithful)
        assert isinstance(optN, (int, np.integer))
        assert optN >= 2

    def test_optD_positive(self, faithful):
        optN, optD, edges, C, N = sshist(faithful)
        assert optD > 0

    def test_edges_sorted(self, faithful):
        optN, optD, edges, C, N = sshist(faithful)
        assert np.all(np.diff(edges) > 0)

    def test_edges_span_data(self, faithful):
        optN, optD, edges, C, N = sshist(faithful)
        assert edges[0] <= np.min(faithful)
        assert edges[-1] >= np.max(faithful)

    def test_cost_finite(self, faithful):
        optN, optD, edges, C, N = sshist(faithful)
        assert np.all(np.isfinite(C))

    def test_unimodal_data(self):
        rs = np.random.RandomState(42)
        x = rs.randn(500)
        optN, optD, edges, C, N = sshist(x)
        assert optN >= 2
        assert optD > 0
