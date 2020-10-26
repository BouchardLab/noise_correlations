import numpy as np

from noise_correlations.null_models import random_rotation
from numpy.testing import assert_array_almost_equal


def test_random_rotation():
    """Tests that random rotations are correctly generated and applied."""
    n_dim = 5
    mu = np.zeros(n_dim)
    diag = np.arange(n_dim) + 1
    cov = np.diag(diag)

    # Test one rotation
    _, new_cov = random_rotation(mu, cov, n_rotations=1)
    assert_array_almost_equal(np.linalg.eigh(new_cov)[0], diag)
    # Test multiple rotations
    _, new_covs = random_rotation(mu, cov, n_rotations=5)
    [assert_array_almost_equal(np.linalg.eigh(new_cov)[0], diag)
     for new_cov in new_covs]
    # Test multiple covariances
    mus = [mu, mu]
    covs = [cov, 2 * cov]
    _, new_covs = random_rotation(mus, covs, n_rotations=5)
    [[assert_array_almost_equal(np.linalg.eigh(new_cov)[0], diag * (idx + 1))
      for new_cov in cov_temp]
     for idx, cov_temp in enumerate(new_covs)]
