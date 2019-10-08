import numpy as np
from sklearn.discriminant_analysis import (QuadraticDiscriminantAnalysis as QDA,
                                           LinearDiscriminantAnalysis as LDA)
import torch

from .utils import mean_cov


def mv_normal_kl(mu0, cov0, mu1, cov1):
    """Calculates the KL Divergence between two multivariate normal
    distributions given their means and covariances.

    Parameters
    ----------
    mu0 : ndarray (dim,)
    cov0 : ndarray (dim, dim)
    mu1 : ndarray (dim,)
    cov1: ndarray (dim, dim)

    Returns
    -------
    KL Divergence
    """
    use_torch = any([isinstance(item, torch.Tensor)
                     for item in [mu0, cov0, mu1, cov1]])
    mean_diff = mu1 - mu0
    if use_torch:
        d = torch.prod(mu1.size)
        tr = torch.trace(torch.solve(cov1, cov0)[0])
        means = mean_diff.mm(torch.solve(cov1, mean_diff)[0])
        logdets = torch.slogdet(cov1)[1] - torch.slogdet(cov0)
    else:
        d = mu1.size
        tr = np.trace(np.linalg.solve(cov1, cov0))
        means = mean_diff.dot(np.linalg.solve(cov1, mean_diff))
        logdets = np.linalg.slogdet(cov1)[1] - np.linalg.slogdet(cov0)[1]
    return .5 * (tr + means + logdets - d)


def mv_normal_jeffreys(mu0, cov0, mu1, cov1):
    """Calculate the symmetric KL Divergence for two multivariate
    normal distributions.

    Parameters
    ----------
    mu0 : ndarray (dim,)
    cov0 : ndarray (dim, dim)
    mu1 : ndarray (dim,)
    cov1: ndarray (dim, dim)

    Returns
    -------
    Symmetric KL Divergence
    """
    kl1 = mv_normal_kl(mu0, cov0, mu1, cov1)
    kl2 = mv_normal_kl(mu1, cov1, mu0, cov0)
    return .5 * (kl1 + kl2)


def mv_normal_jeffreys_data(x0, x1):
    """Calculate the symmetric KL Divergence for two multivariate
    normal distributions from two data matrices.

    Parameters
    ----------
    x0 : ndarray (samples, dim)
    x1 : ndarray (samples, dim)

    Returns
    -------
    Symmetric KL Divergence
    """
    mu0, cov0 = mean_cov(x0)
    mu1, cov1 = mean_cov(x1)
    return mv_normal_jeffreys(mu0, cov0, mu1, cov1)


def linear_discriminability(mu0, cov0, mu1, cov1):
    """Calculate the linear discriminability for two distributions with
    known individual means and covariances.

    Parameters
    ----------
    mu0 : ndarray (dim,)
    mu1 : ndarray (dim,)
    cov: ndarray (dim, dim)

    Returns
    -------
    Linear discriminability
    """
    return lfi(mu0, cov0, mu1, cov1)


def linear_discriminability_data(x0, x1):
    """Calculate the linear discriminability for two distributions from data.

    Parameters
    ----------
    x0 : ndarray (samples, dim)
    x1 : ndarray (samples, dim)

    Returns
    -------
    Linear discriminability
    """
    return lfi_data(x0, x1)


def lda_data(x0, x1):
    """Calculate the training accuracy from a Linear
    Discriminant Analysis (LDA) model from data.

    Parameters
    ----------
    x0 : ndarray (samples, dim)
    x1 : ndarray (samples, dim)

    Returns
    -------
    LDA accuracy
    """
    X = np.concatenate((x0, x1))
    Y = np.zeros(X.shape[0])
    Y[:x0.shape[0]] = 1
    model = LDA().fit(X, Y)
    return model.score(X, Y)


def lda_samples(mu0, cov0, mu1, cov1, size=10000):
    """Calculate the training accuracy from a Linear
    Discriminant Analysis (LDA) model from two normal distributions
    by sampling from them.

    Parameters
    ----------
    mu0 : ndarray (dim,)
    cov0 : ndarray (dim, dim)
    mu1 : ndarray (dim,)
    cov1: ndarray (dim, dim)

    Returns
    -------
    LDA accuracy
    """
    x0 = np.random.multivariate_normal(mu0, cov0, size=size)
    x1 = np.random.multivariate_normal(mu1, cov1, size=size)
    return lda_data(x0, x1)


def qda_data(x0, x1):
    """Calculate the training accuracy from a Quadratic
    Discriminant Analysis (QDA) model from data.

    Parameters
    ----------
    x0 : ndarray (samples, dim)
    x1 : ndarray (samples, dim)

    Returns
    -------
    QDA accuracy
    """
    X = np.concatenate((x0, x1))
    Y = np.zeros(X.shape[0])
    Y[:x0.shape[0]] = 1
    model = QDA().fit(X, Y)
    return model.score(X, Y)


def qda_samples(mu0, cov0, mu1, cov1, size=10000):
    """Calculate the training accuracy from a Quadratic
    Discriminant Analysis (QDA) model from two normal distributions
    by sampling from them.

    Parameters
    ----------
    mu0 : ndarray (dim,)
    cov0 : ndarray (dim, dim)
    mu1 : ndarray (dim,)
    cov1: ndarray (dim, dim)

    Returns
    -------
    QDA accuracy
    """
    x0 = np.random.multivariate_normal(mu0, cov0, size=size)
    x1 = np.random.multivariate_normal(mu1, cov1, size=size)
    return qda_data(x0, x1)


def lfi(mu0, cov0, mu1, cov1, dtheta=1.):
    """Calculate the linear Fisher information from two data matrices.

    Parameters
    ----------
    x0 : ndarray (samples, dim)
    x1 : ndarray (samples, dim)
    dtheta : float
        Change in stimulus between x0 and x1.

    Returns
    -------
    Symmetric KL Divergence
    """
    use_torch = any([isinstance(item, torch.Tensor)
                     for item in [mu0, cov0, mu1, cov1]])
    dmu_dtheta = (mu1 - mu0) / dtheta
    cov = (cov0 + cov1) / 2.

    if use_torch:
        return dmu_dtheta.mm(torch.solve(cov, dmu_dtheta.t()))
    else:
        return dmu_dtheta.dot(np.linalg.pinv(cov).dot(dmu_dtheta.T))


def lfi_data(x0, x1, dtheta=1.):
    """Calculate the linear Fisher information from two data matrices.

    Parameters
    ----------
    x0 : ndarray (samples, dim)
    x1 : ndarray (samples, dim)
    dtheta : float
        Change in stimulus between x0 and x1.

    Returns
    -------
    Symmetric KL Divergence
    """
    mu0, cov0 = mean_cov(x0)
    mu1, cov1 = mean_cov(x1)

    return lfi(mu0, cov0, mu1, cov1, dtheta)


def corrected_lfi(mu0, cov0, mu1, cov1, T, N, dtheta=1.):
    """Calculate the corrected linear Fisher information from two data matrices.
    https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1004218

    Parameters
    ----------
    x0 : ndarray (samples, dim)
    x1 : ndarray (samples, dim)
    dtheta : float
        Change in stimulus between x0 and x1.

    Returns
    -------
    Symmetric KL Divergence
    """
    c0 = (2 * T - N - 3.) / (2. * T - 2)
    c1 = (2. * N) / (T * dtheta**2)

    return (lfi(mu0, cov0, mu1, cov1, dtheta=dtheta) * c0) - c1


def corrected_lfi_data(x0, x1, dtheta=1.):
    """Calculate the corrected linear Fisher information from two data matrices.
    https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1004218

    Parameters
    ----------
    x0 : ndarray (samples, dim)
    x1 : ndarray (samples, dim)
    dtheta : float
        Change in stimulus between x0 and x1.

    Returns
    -------
    Symmetric KL Divergence
    """
    T = x0.shape[0]
    N = x0.shape[1]
    c0 = (2 * T - N - 3.) / (2. * T - 2)
    c1 = (2. * N) / (T * dtheta**2)
    if x0.shape[0] != x1.shape[0]:
        raise ValueError

    return (lfi_data(x0, x1, dtheta) * c0) - c1


def corrected_lfi_samples(mu0, cov0, mu1, cov1, size=10000, dtheta=1.):
    """Calculate the corrected linear Fisher information from samples from two
    multivariante normal distributions.
    https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1004218

    Parameters
    ----------
    x0 : ndarray (samples, dim)
    x1 : ndarray (samples, dim)
    dtheta : float
        Change in stimulus between x0 and x1.

    Returns
    -------
    Symmetric KL Divergence
    """
    x0 = np.random.multivariate_normal(mu0, cov0, size=size)
    x1 = np.random.multivariate_normal(mu1, cov1, size=size)
    T = x0.shape[0]
    N = x0.shape[1]
    c0 = (2 * T - N - 3.) / (2. * T - 2)
    c1 = (2. * N) / (T * dtheta**2)
    if x0.shape[0] != x1.shape[1]:
        raise ValueError

    return (lfi_data(x0, x1, dtheta) * c0) - c1


def lfi_shuffle_data(x0, x1, dtheta=1.):
    """Calculate the shuffled linear Fisher information from two data matrices.

    Parameters
    ----------
    x0 : ndarray (samples, dim)
    x1 : ndarray (samples, dim)
    dtheta : float
        Change in stimulus between x0 and x1.

    Returns
    -------
    Symmetric KL Divergence
    """
    mu0 = x0.mean(axis=0)
    mu1 = x1.mean(axis=0)
    var = np.var(np.concatenate((x0, x1)), axis=0, ddof=1)

    dmu_dtheta = (mu1 - mu0) / dtheta

    return np.sum(dmu_dtheta**2 / var)


def corrected_lfi_shuffle_data(x0, x1, dtheta=1.):
    """Calculate the shuffled linear Fisher information from two data matrices.

    Parameters
    ----------
    x0 : ndarray (samples, dim)
    x1 : ndarray (samples, dim)
    dtheta : float
        Change in stimulus between x0 and x1.

    Returns
    -------
    Symmetric KL Divergence
    """
    T = x0.shape[0]
    N = x0.shape[1]
    c0 = (T - 2.) / (T - 1.)
    c1 = (2. * N) / (T * dtheta**2)

    return (lfi_shuffle_data(x0, x1) * c0) - c1
