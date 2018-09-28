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


def histo_samples(orig0, orig1, trans, measure, nsamples, faxes=None):
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
    values = np.zeros(nsamples)
    for ii in range(nsamples):
        new0 = trans(orig0)
        new1 = trans(orig1)
        values[ii] = measure(new0, new1)
    if faxes is None:
        faxes = plt.subplots(1, 1)
    fig, ax = faxes
    vals, bins, patches = ax.hist(values, bins=50, density=True)
    orig_val = measure(orig0, orig1)
    ax.vlines(orig_val, 0, np.max(vals))
    frac_less = np.count_nonzero(orig_val > values) / nsamples
    return orig_val, frac_less, ax


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
    frac_less = np.count_nonzero(orig_val[:, np.newaxis] > values,
                                 axis=1) / nsamples
    return orig_val, values, frac_less


def rot_plot(mu0, cov0, mu1, cov1, measure, nsamples=10000):
    """Plots comparison to random-rotation null model."""
    def unpack_rot(args):
        return random_rotation(*args)

    def unpack_measure(arg0, arg1):
        return measure(*arg0, *arg1)
    val, frac, ax = histo_samples((mu0, cov0), (mu1, cov1),
                                  unpack_rot, unpack_measure, nsamples)
    print('Value: ', val)
    print('Fraction of rotations giving smaller values: ', frac)
    return val, frac, ax


def ellipses_and_measure_normal_data(x0, x1):
    mu0 = x0.mean(0)
    cov0 = np.cov(x0)
    mu1 = x1.mean(0)
    cov1 = np.cov(x1)
    discriminability.plot_ellipses(mu0, cov0, mu1, cov1)
    print('Jeffreys:', discriminability.mv_normal_jeffreys(mu0, cov0,
                                                           mu1, cov1))


def scatter_and_measures(x0, x1):
    fig, ax = plt.subplots(1)
    c0, c1 = u'#1f77b4', u'#ff7f0e'
    ax.plot(x0[:, 0], x0[:, 1], '.', color=c0)
    ax.plot(x1[:, 0], x1[:, 1], '.', color=c1)
    print('Jeffreys:', discriminability.mv_normal_jeffreys_data(x0, x1))
    print('Linear discriminability',
          discriminability.linear_discriminability_data(x0, x1))
    print('Quadratic disciminability', discriminability.qda_data(x0, x1))


def shuffle_plot_data(x0, x1, measure, ntrials=1000):
    val, frac = histo_samples(x0, x1, shuffle_data, measure, ntrials)
    print('Value:', val)
    print('Fraction of shuffles giving smaller values: ', frac)


def rot_plot_data(x0, x1, measure, nsamples=10000):
    val, frac, ax = histo_samples(x0, x1, random_rotation_data, measure,
                                  nsamples)
    print('Value: ', val)
    print('Fraction of rotations giving smaller values: ', frac)
    return val, frac, ax
