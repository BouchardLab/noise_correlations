import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA


def mean_cov(x):
    return x.mean(axis=0), np.cov(x, rowvar=False)


def mv_normal_kl(mu0, sigma0, mu1, sigma1):
    sigma1_inv = np.linalg.inv(sigma1)
    mean_diff = mu1 - mu0
    d = mu1.size
    tr = np.trace(sigma1_inv.dot(sigma0))
    means = mean_diff.dot(sigma1_inv).dot(mean_diff)
    logdets = np.log(np.linalg.det(sigma1)) - np.log(np.linalg.det(sigma0))
    return .5 * (tr + means + logdets - d)


def mv_normal_jeffreys(mu0, sigma0, mu1, sigma1):
    """Calculate Jeffreys divergence, a symmetrized version of KL."""
    return .5 * (mv_normal_kl(mu0, sigma0, mu1, sigma1) +
                 mv_normal_kl(mu1, sigma1, mu0, sigma0))


def mv_normal_jeffreys_data(x0, x1):
    """Calculate Jeffreys divergence from data."""
    mu0, sigma0 = mean_cov(x0)
    mu1, sigma1 = mean_cov(x1)
    return mv_normal_jeffreys(mu0, sigma0, mu1, sigma1)


def linear_discriminability(mu0, mu1, sigma):
    """Calculate linear discriminability when total covariance sigma is known."""
    mean_diff = mu1 - mu0
    return mean_diff.dot(np.linalg.inv(sigma)).dot(mean_diff)


def linear_discriminability_data(x0, x1):
    """Calculate linear discriminability from data."""
    mu0 = x0.mean(axis=0)
    mu1 = x1.mean(axis=0)
    sigma = np.cov(np.concatenate((x0, x1)), rowvar=False)
    return linear_discriminability(mu0, mu1, sigma)


def qda_data(x0, x1):
    """Calculate quadratic disciminability from data."""
    X = np.concatenate((x0, x1))
    Y = np.zeros(X.shape[0])
    Y[:x0.shape[0]] = 1
    model = QDA().fit(X, Y)
    return model.score(X, Y)


def plot_ellipses(mu0, sigma0, mu1, sigma1, ld_sigma=None):
    """Plot ellipses corresponding to bivariate normal distributions
    with means mu0, mu1 and covariances sigma0, sigma1."""
    assert mu0.size == 2
    f, ax = plt.subplots(1, figsize=(5, 5))
    c0, c1 = u'#1f77b4', u'#ff7f0e'
    for mu, sigma, c in [(mu0, sigma0, c0), (mu1, sigma1, c1)]:
        e, v = np.linalg.eigh(sigma)
        e = np.sqrt(e)
        ell = Ellipse(mu, e[1], e[0], 180. * np.arctan2(v[0, -1], v[1, -1]) / np.pi,
                      facecolor=c, alpha=.5)
        ax.plot(mu[0], mu[1], 'o', c=c)
        ax.add_artist(ell)
    if ld_sigma is not None:
        e, v = np.linalg.eigh(ld_sigma)
        e = np.sqrt(e)
        ell = Ellipse(.5*(mu0+mu1), e[1], e[0], 180. * np.arctan2(v[0, -1], v[1, -1]) / np.pi,
                      facecolor='None', alpha=.5, edgecolor='k')
        ax.add_artist(ell)
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2)