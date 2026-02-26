import os
import numpy as np
import pytest
from adaptivekde.sskernel import sskernel


class TestSskernelGolden:
    """Compare sskernel output against saved reference data."""

    def test_golden_match(self, faithful, ref_dir):
        np.random.seed(0)
        y, t, optw, W, C, confb95, yb = sskernel(faithful, nbs=200)
        ref = np.load(os.path.join(ref_dir, "sskernel_ref.npz"))

        np.testing.assert_allclose(y, ref["y"], rtol=1e-10)
        np.testing.assert_allclose(t, ref["t"], rtol=1e-10)
        np.testing.assert_allclose(optw, ref["optw"], rtol=1e-10)
        np.testing.assert_allclose(W, ref["W"], rtol=1e-10)
        np.testing.assert_allclose(C, ref["C"], rtol=1e-10)
        np.testing.assert_allclose(confb95, ref["confb95"], rtol=1e-10)
        np.testing.assert_allclose(yb, ref["yb"], rtol=1e-10)


class TestSskernelProperties:
    """Property-based sanity checks for sskernel."""

    @pytest.fixture(autouse=True)
    def run_sskernel(self, faithful):
        np.random.seed(0)
        self.y, self.t, self.optw, self.W, self.C, self.confb95, self.yb = \
            sskernel(faithful, nbs=200)

    def test_density_nonnegative(self):
        assert np.all(self.y >= 0)

    def test_density_integrates_to_one(self):
        dt = np.diff(self.t)
        integral = np.sum(0.5 * (self.y[:-1] + self.y[1:]) * dt)
        np.testing.assert_allclose(integral, 1.0, atol=0.05)

    def test_bandwidth_positive(self):
        assert self.optw > 0

    def test_t_sorted(self):
        assert np.all(np.diff(self.t) > 0)

    def test_confidence_bands_bracket_estimate(self):
        lower = self.confb95[0, :]
        upper = self.confb95[1, :]
        assert np.all(upper >= lower)

    def test_yb_shape(self):
        assert self.yb.shape[0] == 200
        assert self.yb.shape[1] == len(self.t)
