import numpy as np
import torch

from noise_correlations.discriminability import lfi, mv_normal_jeffreys
from numpy.testing import assert_array_almost_equal


def test_lfi_low_rank():
    """Tests that LFI is calculated correctly for low-rank covariances.

    Pytorch version does not allow low-rank matrices.
    """
    d = 10
    mu0 = np.random.randn(1, d)
    mu1 = np.random.randn(1, d)
    cov = np.diag(np.arange(10, dtype=float))
    lfi_val = lfi(mu0, cov, mu1, cov)
    lfi_val2 = lfi(mu0[:, 1:], cov[1:, 1:], mu1[:, 1:], cov[1:, 1:])
    diff = (mu1 - mu0)
    lfi_solve = diff[:, 1:] @ np.linalg.solve(cov[1:, 1:], diff[:, 1:].T)

    assert_array_almost_equal(lfi_val, lfi_solve)
    assert_array_almost_equal(lfi_val, lfi_val2)

    mu0 = torch.tensor(mu0[:, 1:])
    mu1 = torch.tensor(mu1[:, 1:])
    cov = torch.tensor(cov[1:, 1:])
    lfi_valt = lfi(mu0, cov, mu1, cov).numpy()

    assert_array_almost_equal(lfi_val, lfi_valt)


def test_jeffreys():
    """Tests that SDKL gives reasonable values.
    """
    d = 10
    mu0 = np.random.randn(1, d)
    mu1 = np.random.randn(1, d)
    cov0 = np.diag(np.arange(10, dtype=float)) + 1.
    cov1 = np.diag(np.arange(10, dtype=float)[::-1]) + 1.
    sdkl_val = mv_normal_jeffreys(mu0, cov0, mu1, cov1)
    sdkl_val0 = mv_normal_jeffreys(mu0, cov0, mu0, cov0)

    assert_array_almost_equal(sdkl_val0, 0.)
    assert sdkl_val > sdkl_val0
