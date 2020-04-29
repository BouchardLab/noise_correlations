import matplotlib.pyplot as plt
import numpy as np


def X_stimuli(X, stimuli):
    """Preprocesses input design matrix and stimuli per trials.

    Parameters
    ----------
    X : ndarray (samples, units)
        Neural data design matrix.
    stimuli : ndarray (samples,)
        The stimulus value for each trial.

    Returns
    -------
    n_samples : int
        The number of samples in the dataset.
    n_units : int
        The number of units in the dataset.
    n_stimuli : int
        The number of unique stimuli.
    unique_stimuli : ndarray (stimuli,)
        The unique stimuli, sorted by value.
    """
    n_samples, n_units = X.shape
    unique_stimuli = np.unique(stimuli)
    n_stimuli = unique_stimuli.size
    return n_samples, n_units, n_stimuli, unique_stimuli


def check_fax(fax=None, n_rows=1, n_cols=1, figsize=(10, 10)):
    """Checks an incoming set of axes, and creates new ones if needed.

    Parameters
    ----------
    fax : tuple of mpl.figure and mpl.axes, or None
        The figure and axes. If None, a new set will be created.

    figsize : tuple or None
        The figure size, if fax is None.

    Returns
    -------
    fig, ax : mpl.figure and mpl.axes
        The matplotlib axes objects.
    """
    # no axes provided
    if fax is None:
        fig, ax = plt.subplots(int(n_rows), int(n_cols), figsize=figsize)
    else:
        fig, ax = fax
    return fig, ax


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


def get_tuned_units(
    X, stimuli, aggregator=np.median, peak_response=None, min_modulation=None
):
    """Gets the units in a neural design matrix which satisfy a chosen
    criteria for being tuned.

    Parameters
    ----------
    X : ndarray (samples, units)
        Neural data design matrix.
    stimuli : ndarray (samples,)
        The stimulus value for each trial.
    aggregator : function
        Function used to aggregate statistics across samples per stimulus.
    peak_response : float or None
        If None, peak response criteria is not used. If float, serves as the
        minimum peak response required for a neuron to be tuned.
    min_modulation : float or None
        If None, peak response criteria is not used. If float, serves as the
        minimum modulation (min-to-max distance of the tuning curve) required
        for the neuron to be considered tuned.

    Returns
    -------
    keep : ndarray (units,)
        A Boolean mask denoting which neurons to keep.
    """
    n_samples, n_units, n_stimuli, unique_stimuli = X_stimuli(X, stimuli)
    # calculate average responses
    avg_responses = np.zeros((n_units, n_stimuli))
    for idx, stimulus in enumerate(unique_stimuli):
        avg_responses[:, idx] = aggregator(X[stimuli == stimulus], axis=0)

    keep = np.ones(n_units)
    if peak_response is not None:
        valid = avg_responses.max(axis=1) > peak_response
        keep = np.logical_and(keep, valid)
    if min_modulation is not None:
        valid = avg_responses.max(axis=1) >= min_modulation * avg_responses.min(axis=1)
        keep = np.logical_and(keep, valid)

    return keep


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
