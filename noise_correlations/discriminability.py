import numpy as np
from sklearn.linear_model import LogisticRegression as LR
from sklearn.discriminant_analysis import (QuadraticDiscriminantAnalysis as QDA,
                                           LinearDiscriminantAnalysis as LDA)
from sklearn.metrics import log_loss
import torch

from .utils import mean_cov, _lfi


def mv_normal_kl(mu0, cov0, mu1, cov1, return_trace=False):
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
    mean_diff = mu1 - mu0
    d = mu1.size
    tr = np.trace(np.linalg.solve(cov1, cov0))
    means = mean_diff @ np.linalg.solve(cov1, mean_diff.T)
    logdets = np.linalg.slogdet(cov1)[1] - np.linalg.slogdet(cov0)[1]
    kl = .5 * (tr + means + logdets - d)
    if return_trace:
        return kl, tr
    else:
        return kl


def mv_normal_jeffreys(mu0, cov0, mu1, cov1, return_trace=False):
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
    kl1, tr1 = mv_normal_kl(mu0, cov0, mu1, cov1, True)
    kl2, tr2 = mv_normal_kl(mu1, cov1, mu0, cov0, True)
    sdkl = 0.5 * (kl1 + kl2)
    if return_trace:
        return sdkl, 0.5 * (tr1 + tr2)
    else:
        return sdkl


def mv_normal_jeffreys_data(x0, x1, return_trace=False):
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
    return mv_normal_jeffreys(mu0, cov0, mu1, cov1, return_trace=return_trace)


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


def logistic_data(x0, x1):
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
    model = LR(C=1e3, solver='lbfgs').fit(X, Y)
    Yhat = model.predict_proba(X)
    return np.mean(Yhat.argmax(axis=1) == Y), log_loss(Y, Yhat)


def logistic_samples(mu0, cov0, mu1, cov1, size=10000):
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
    return logistic_data(x0, x1)


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
    Linear Fisher information
    """
    use_torch = any([isinstance(item, torch.Tensor)
                     for item in [mu0, cov0, mu1, cov1]])
    dmu_dtheta = (mu1 - mu0) / dtheta
    cov = (cov0 + cov1) / 2.

    if use_torch:
        return _lfi(mu0, mu1, cov, dtheta)
    else:
        return dmu_dtheta @ np.linalg.lstsq(cov, dmu_dtheta.T, rcond=None)[0]


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
    Linear Fisher information
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
    Corrected linear Fisher information
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
    Corrected linear Fisher information
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
    Corrected linear Fisher information
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
    Linear Fisher information
    """
    mu0 = x0.mean(axis=0)
    mu1 = x1.mean(axis=0)
    var = np.var(np.concatenate((x0, x1)), axis=0, ddof=1)

    dmu_dtheta = (mu1 - mu0) / dtheta

    return np.dot(dmu_dtheta**2, var)


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
    Linear Fisher information
    """
    T = x0.shape[0]
    N = x0.shape[1]
    c0 = (T - 2.) / (T - 1.)
    c1 = (2. * N) / (T * dtheta**2)

    return (lfi_shuffle_data(x0, x1) * c0) - c1
