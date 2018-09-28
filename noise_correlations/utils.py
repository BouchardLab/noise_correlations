import numpy as np


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
