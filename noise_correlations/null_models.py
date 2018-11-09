import numpy as np
import matplotlib.pyplot as plt
from . import discriminability
from scipy.stats import special_ortho_group


def erase_offdiag(mu, cov):
    """Remove off-diagonal entries of cov."""
    return mu, np.diag(np.diag(cov))


def shuffle_data(x, size=1, rng=None):
    """Permute each column independently.
    Model-agnostic analog of removing off-diagonal entries of covariance."""
    if rng is None:
        rng = np.random
    x = x.copy()
    if size == 1:
        for ii in range(x.shape[1]):
            x[:, ii] = rng.permutation(x[:, ii])
        return x
    else:
        out = np.zeros((size,) +  x.shape)
        for ii in range(size):
            for jj in range(x.shape[1]):
                out[ii, :, jj] = rng.permutation(x[:, jj])
        return out


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


def random_rotation(mu, cov, size=1, rng=None):
    """Apply a random rotation to cov."""
    if rng is None:
        rng = np.random
    sog = special_ortho_group(cov.shape[0])
    if size == 1:
        rot = sog.rvs(cov.shape[0], random_state=rng)
        return mu, rot.dot(cov).dot(rot.T)
    else:
        rots = sog.rvs(size, random_state=rng)
        return mu[np.newaxis], np.einsum('nij, jk, nlk->nil', rots, cov, rots)


def random_rotation_data(x, size=1, rng=None):
    """Apply a random rotation to data.

    Parameters
    ----------
    x : ndarray (examples, dim)
    """
    if rng is None:
        rng = np.random
    sog = special_ortho_group(x.shape[1])
    mu = x.mean(axis=0, keepdims=True)
    if size == 1:
        rot = sog.rvs(random_state=rng)
        return (x - mu).dot(rot.T) + mu
    else:
        rots = sog.rvs(size, random_state=rng)
        rval = np.einsum('ij, klj', x-mu, rots)
        return np.transpose(rval, axes=(1, 0, 2)) + mu[np.newaxis]


def eval_null(mu0, cov0, mu1, cov1, null, measures, nsamples, seed=201811071, same_null=False):
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
    rng = np.random.RandomState(seed)
    if not isinstance(measures, list):
        measures = [measures]
    orig_val = np.array([m(mu0, cov0, mu1, cov1) for m in measures])
    values = np.zeros((len(measures), nsamples))
    if same_null:
        rng2 = np.random.RandomState()
        rng2.set_state(rng.__getstate__())
    else:
        rng2 = rng
    for ii in range(nsamples):
        mu0p, cov0p = null(mu0, cov0, rng)
        mu1p, cov1p = null(mu1, cov1, rng2)
        for jj, m in enumerate(measures):
            values[jj, ii] = m(mu0p, cov0p, mu1p, cov1p)
    frac_less = np.count_nonzero(values >= orig_val[:, np.newaxis],
                                 axis=1) / nsamples
    return orig_val, values, frac_less


def eval_null_data(x0, x1, null, measures, nsamples, seed=201811072, same_null=False):
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
    rng = np.random.RandomState(seed)
    if not isinstance(measures, list):
        measures = [measures]
    orig_val = np.array([m(x0, x1) for m in measures])
    values = np.zeros((len(measures), nsamples))
    if same_null:
        rng2 = np.random.RandomState()
        rng2.set_state(rng.__getstate__())
    else:
        rng2 = rng
    x0ps = null(x0, size=nsamples, rng=rng)
    x1ps = null(x1, size=nsamples, rng=rng2)
    for ii in range(nsamples):
        x0p = x0ps[ii]
        x1p = x1ps[ii]
        for jj, m in enumerate(measures):
            values[jj, ii] = m(x0p, x1p)
    frac_less = np.count_nonzero(values >= orig_val[:, np.newaxis],
                                 axis=1) / nsamples
    return orig_val, values, frac_less
