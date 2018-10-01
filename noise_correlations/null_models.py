import numpy as np
import matplotlib.pyplot as plt
from . import discriminability
from scipy.stats import special_ortho_group


def erase_offdiag(mu, cov):
    """Remove off-diagonal entries of cov."""
    return mu, np.diag(np.diag(cov))


def shuffle_data(xx):
    """Permute each column independently.
    Model-agnostic analog of removing off-diagonal entries of covariance."""
    shuffled = [np.random.permutation(xx[:, ii]) for ii in range(xx.shape[1])]
    return np.vstack(shuffled).T


def diag_and_scale(mu, cov):
    """Remove off-diagonal entries, then rescale cov so the determinant
    is the same as it was originally."""
    det = np.linalg.det(cov)
    diag = np.diag(cov)
    newdet = np.prod(diag)
    diagmat = np.diag(diag)
    size = cov.shape[0]
    factor = (det / newdet)**(1. / size)
    return mu, factor * diagmat


def random_rotation(mu, cov):
    """Apply a random rotation to cov."""
    rotmat = special_ortho_group.rvs(cov.shape[0])
    return mu, rotmat.dot(cov).dot(rotmat.T)


def random_rotation_data(x):
    """Apply a random rotation to data.

    Parameters
    ----------
    x : ndarray (examples, dim)
    """
    mu = x.mean(axis=0, keepdims=True)
    rotmat = special_ortho_group.rvs(x.shape[1])
    return (x - mu).dot(rotmat.T) + mu


def eval_null(mu0, cov0, mu1, cov1, null, measures, nsamples):
    """
    Plot a histogram of nsamples values of measure evaluated on orig0 and orig1
    after applying trans. Original value plotted as vertical line.

    Parameters:
    -----------
    orig0       (examples, dim) data
    orig1       (examples, dim) data
    trans       (callable) data transformation
    measure     (callable) takes 2 data arguments, returns comparison measure
    nsamples    (int) number of times to apply trans and evaluate measure

    Returns:
    measure(orig0, orig1)
    fraction of samples with measure less than original
    matplotlib axis object
    """
    if not isinstance(measures, list):
        measures = [measures]
    orig_val = np.array([m(mu0, cov0, mu1, cov1) for m in measures])
    values = np.zeros((len(measures), nsamples))
    for ii in range(nsamples):
        mu0p, cov0p = null(mu0, cov0)
        mu1p, cov1p = null(mu1, cov1)
        for jj, m in enumerate(measures):
            values[jj, ii] = m(mu0p, cov0p, mu1p, cov1p)
    frac_less = np.count_nonzero(values >= orig_val[:, np.newaxis],
                                 axis=1) / nsamples
    return orig_val, values, frac_less


def eval_null_data(x0, x1, null, measures, nsamples):
    """
    Plot a histogram of nsamples values of measure evaluated on orig0 and orig1
    after applying trans. Original value plotted as vertical line.

    Parameters:
    -----------
    orig0       (examples, dim) data
    orig1       (examples, dim) data
    trans       (callable) data transformation
    measure     (callable) takes 2 data arguments, returns comparison measure
    nsamples    (int) number of times to apply trans and evaluate measure

    Returns:
    measure(orig0, orig1)
    fraction of samples with measure less than original
    matplotlib axis object
    """
    if not isinstance(measures, list):
        measures = [measures]
    orig_val = np.array([m(x0, x1) for m in measures])
    values = np.zeros((len(measures), nsamples))
    for ii in range(nsamples):
        x0p = null(x0)
        x1p = null(x1)
        for jj, m in enumerate(measures):
            values[jj, ii] = m(x0p, x1p)
    frac_less = np.count_nonzero(values >= orig_val[:, np.newaxis],
                                 axis=1) / nsamples
    return orig_val, values, frac_less
