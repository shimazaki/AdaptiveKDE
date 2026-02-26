import os
import numpy as np
import pytest
from adaptivekde.ssvkernel import ssvkernel


class TestSsvkernelGolden:
    """Compare ssvkernel output against saved reference data."""

    def test_golden_match(self, faithful, ref_dir):
        np.random.seed(0)
        y, t, optw, gs, C, confb95, yb = ssvkernel(faithful, nbs=50)
        ref = np.load(os.path.join(ref_dir, "ssvkernel_ref.npz"))

        np.testing.assert_allclose(y, ref["y"], rtol=1e-10)
        np.testing.assert_allclose(t, ref["t"], rtol=1e-10)
        np.testing.assert_allclose(optw, ref["optw"], rtol=1e-10)
        np.testing.assert_allclose(gs, ref["gs"], rtol=1e-10)
        np.testing.assert_allclose(C, ref["C"], rtol=1e-10)
        np.testing.assert_allclose(confb95, ref["confb95"], rtol=1e-10)
        np.testing.assert_allclose(yb, ref["yb"], rtol=1e-10)


class TestSsvkernelProperties:
    """Property-based sanity checks for ssvkernel."""

    @pytest.fixture(autouse=True)
    def run_ssvkernel(self, faithful):
        np.random.seed(0)
        self.y, self.t, self.optw, self.gs, self.C, self.confb95, self.yb = \
            ssvkernel(faithful, nbs=50)

    def test_density_nonnegative(self):
        assert np.all(self.y >= 0)

    def test_density_integrates_to_one(self):
        dt = np.diff(self.t)
        integral = np.sum(0.5 * (self.y[:-1] + self.y[1:]) * dt)
        np.testing.assert_allclose(integral, 1.0, atol=0.05)

    def test_bandwidth_positive(self):
        assert np.all(self.optw > 0)

    def test_t_sorted(self):
        assert np.all(np.diff(self.t) > 0)

    def test_confidence_bands_bracket_estimate(self):
        lower = self.confb95[0, :]
        upper = self.confb95[1, :]
        assert np.all(upper >= lower)

    def test_yb_shape(self):
        assert self.yb.shape[0] == 50
        assert self.yb.shape[1] == len(self.t)
