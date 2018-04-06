import numpy as np
import matplotlib.pyplot as plt
import discriminability
from scipy.stats import special_ortho_group


def erase_offdiag(mu, sigma):
    """Remove off-diagonal entries of sigma."""
    return mu, np.diag(np.diag(sigma))


def shuffle_data(xx):
    """Permute each column independently.
    Model-agnostic analog of removing off-diagonal entries of covariance."""
    shuffled = [np.random.permutation(xx[:, ii]) for ii in range(xx.shape[1])]
    return np.vstack(shuffled).T


def diag_and_scale(mu, sigma):
    """Remove off-diagonal entries, then rescale sigma so the determinant
    is the same as it was originally."""
    det = np.linalg.det(sigma)
    diag = np.diag(sigma)
    newdet = np.prod(diag)
    diagmat = np.diag(diag)
    size = sigma.shape[0]
    factor = (det / newdet)**(1/size)
    return mu, factor*diagmat


def random_rotation(mu, sigma):
    """Apply a random rotation to sigma."""
    rotmat = special_ortho_group.rvs(sigma.shape[0])
    return mu, rotmat.dot(sigma).dot(rotmat.T)


def histo_samples(orig0, orig1, trans, measure, nsamples):
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
    fig, ax = plt.subplots(1, 1)
    vals, bins, patches = ax.hist(values, bins=50, density=True)
    orig_val = measure(orig0, orig1)
    ax.vlines(orig_val, 0, np.max(vals))
    frac_less = np.count_nonzero(orig_val > values)/nsamples
    return orig_val, frac_less, ax


def rot_plot(mu0, sigma0, mu1, sigma1, measure, nsamples=10000):
    """Plots comparison to random-rotation null model."""
    def unpack_rot(args):
        return random_rotation(*args)

    def unpack_measure(arg0, arg1):
        return measure(*arg0, *arg1)
    val, frac, ax = histo_samples((mu0, sigma0), (mu1, sigma1),
                                  unpack_rot, unpack_measure, nsamples)
    print('Value: ', val)
    print('Fraction of rotations giving smaller values: ', frac)
    return val, frac, ax


def ellipses_and_measure_normal_data(x0, x1):
    mu0 = x0.mean(0)
    sigma0 = np.cov(x0)
    mu1 = x1.mean(0)
    sigma1 = np.cov(x1)
    discriminability.plot_ellipses(mu0, sigma0, mu1, sigma1)
    print('Jeffreys:', discriminability.mv_normal_jeffreys(mu0, sigma0, mu1, sigma1))


def scatter_and_measures(x0, x1):
    fig, ax = plt.subplots(1,1)
    c0, c1 = u'#1f77b4', u'#ff7f0e'
    ax.plot(x0[:, 0], x0[:, 1], '.', color=c0)
    ax.plot(x1[:, 0], x1[:, 1], '.', color=c1)
    print('Jeffreys:', discriminability.mv_normal_jeffreys_data(x0, x1))
    print('Linear discriminability', discriminability.linear_discriminability_data(x0, x1))
    print('Quadratic disciminability', discriminability.qda_data(x0, x1))


def shuffle_plot_data(x0, x1, measure, ntrials=1000):
    val, frac = histo_samples(x0, x1, shuffle_data, measure, ntrials)
    print('Value:', val)
    print('Fraction of shuffles giving smaller values: ', frac)


def random_rotation_data(xx):
    mu = np.mean(xx, keepdims=True)
    centered = xx - mu
    rotmat = special_ortho_group.rvs(xx.shape[1])
    return centered.dot(rotmat.T) + mu


def rot_plot_data(x0, x1, measure, nsamples=10000):
    val, frac, ax = histo_samples(x0, x1, random_rotation_data, measure, nsamples)
    print('Value: ', val)
    print('Fraction of rotations giving smaller values: ', frac)
    return val, frac, ax


def ellipses_and_measure(mu0, sigma0, mu1, sigma1):
    discriminability.plot_ellipses(mu0, sigma0, mu1, sigma1)
    print('Jeffreys:', discriminability.mv_normal_jeffreys(mu0, sigma0, mu1, sigma1))


def null_model_example(null_func):
    mu0 = np.array([1., .5])
    mu1 = np.array([.5, 1.])
    sigma0 = np.array([[1.5, 1.], [1., 1.5]])
    sigma1 = np.array([[1.5, 1.], [1., 1.5]])
    ellipses_and_measure(*null_func(mu0, sigma0),
                         *null_func(mu1, sigma1))