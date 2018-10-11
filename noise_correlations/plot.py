import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse


def plot_ellipses(mu0, cov0, mu1, cov1, ld_cov=None, faxes=None):
    """Plot ellipses corresponding to bivariate normal distributions
    with means mu0, mu1 and covariances cov0, cov1. Can also include
    an ellipse for the linear discriminability covariance.

    Parameters
    ----------
    mu0 : ndarray (2,)
    cov0 : ndarray (2, 2)
    mu1 : ndarray (2,)
    cov1: ndarray (2, 2)
    ld_cov : ndarray (2, 2)
    """
    if mu0.size != 2:
        raise ValueError
    if faxes is None:
        faxes = plt.subplots(1, figsize=(5, 5))
    f, ax = faxes
    c0, c1 = u'#1f77b4', u'#ff7f0e'
    for mu, cov, c in [(mu0, cov0, c0), (mu1, cov1, c1)]:
        e, v = np.linalg.eigh(cov)
        e = np.sqrt(e)
        ell = Ellipse(mu, e[1], e[0],
                      180. * np.arctan2(v[0, -1], v[1, -1]) / np.pi,
                      facecolor=c, alpha=.5)
        ax.plot(mu[0], mu[1], 'o', c=c)
        ax.add_artist(ell)
    if ld_cov is not None:
        e, v = np.linalg.eigh(ld_cov)
        e = np.sqrt(e)
        ell = Ellipse(.5 * (mu0 + mu1), e[1], e[0],
                      180. * np.arctan2(v[0, -1], v[1, -1]) / np.pi,
                      facecolor='None', alpha=.5, edgecolor='k')
        ax.add_artist(ell)
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2)
    return


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
