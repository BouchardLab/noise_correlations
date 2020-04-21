import numpy as np

from scipy.stats import special_ortho_group


def mean_cov(x):
    """Calculate the mean and covariance of a data matrix.

    Parameters
    ----------
    x : ndarray (samples, dim)
        Data array.

    Returns
    -------
    Mean vector and covariance matrix for the data matrix.
    """
    return x.mean(axis=0), np.cov(x, rowvar=False)


def uniform_correlation_matrix(dim, var, corr, noise_std=0., rng=None):
    """Create a uniform covariance matrix with constant variance and uniform
    pairwise covariance. Will have a single e0 eigenvalue and dim-1 e1
    eigenvalues
    """
    cov = np.ones((dim, dim)) * var * corr
    cov[np.arange(dim), np.arange(dim)] = var
    if noise_std > 0.:
        if rng is None:
            rng = np.random
        noise = rng.randn(dim, dim) * noise_std
        cov += noise.dot(noise.T)
    return cov


def subsample_cov(mus, covs, keep, rng):
    if isinstance(mus, list):
        dim = mus[0].size
    else:
        dim = mus.size
    idxs = rng.permutation(dim)[:keep]
    if isinstance(mus, list):
        mus = [mu[idxs] for mu in mus]
    else:
        mus = mus[idxs]
    if isinstance(covs, list):
        covs = [cov[idxs][:, idxs] for cov in covs]
    else:
        covs = covs[idxs][:, idxs]
    return mus, covs
