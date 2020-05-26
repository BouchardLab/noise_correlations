import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as ss


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


def cartesian_product(x, y):
    """Calculates the Cartesian product between two 1-d numpy arrays."""
    return np.dstack(np.meshgrid(x, y)).reshape(-1, 2)


def get_nonresponsive_for_stim(X, stimuli):
    """Gets the units that have no response to a specific stimulus.

    Parameters
    ----------
    X : ndarray (samples, units)
        Neural data design matrix.
    stimuli : ndarray (samples,)
        The stimulus value for each trial.

    Returns
    -------
    unit_idxs : np.ndarray
        The units that have no response to a specific stimulus.
    """
    _, n_units, n_stimuli, unique_stimuli = X_stimuli(X, stimuli)
    unit_idxs = np.unique([
        unit for unit in range(n_units) for stim in unique_stimuli
        if X[stimuli == stim][:, unit].sum() == 0
    ])
    return unit_idxs


def get_variance_to_mean_ratio(X, stimuli):
    """Gets variance to mean ratio, averaged over unique stimuli.

    Parameters
    ----------
    X : ndarray (samples, units)
        Neural data design matrix.
    stimuli : ndarray (samples,)
        The stimulus value for each trial.

    Returns
    -------
    variance_to_mean : ndarray (units,)
        The variance-to-mean ratio, averaged over unique stimuli, for each
        neuron.
    """
    _, n_units, n_stimuli, unique_stimuli = X_stimuli(X, stimuli)
    variance_to_mean = np.zeros((n_units, n_stimuli))

    # calculate variance-to-mean ratio for each stimulus
    for idx, stimulus in enumerate(unique_stimuli):
        X_sub = X[stimuli == stimulus]
        mean_response = np.mean(X_sub, axis=0)
        variance = np.var(X_sub, axis=0)
        variance_to_mean[:, idx] = variance / mean_response

    return np.nan_to_num(variance_to_mean).mean(axis=1)


def get_tuning_curve(X, stimuli, aggregator=np.mean):
    """Gets the tuning curves for a neural design matrix and set of stimuli.

    Parameters
    ----------
    X : ndarray (samples, units)
        Neural data design matrix.
    stimuli : ndarray (samples,)
        The stimulus value for each trial.
    aggregator : function
        Function used to aggregate statistics across samples per stimulus.

    Returns
    -------
    tuning_curves : ndarray (units, unique_stimuli)
        The mean response to unique stimuli for each neuron (i.e., their tuning
        curves).
    """
    _, n_units, n_stimuli, unique_stimuli = X_stimuli(X, stimuli)
    # calculate average responses over unique stimuli
    tuning_curves = np.zeros((n_units, n_stimuli))
    for idx, stimulus in enumerate(unique_stimuli):
        tuning_curves[:, idx] = aggregator(X[stimuli == stimulus], axis=0)
    return tuning_curves


def get_tuning_modulation(X, stimuli, aggregator=np.mean):
    """Gets the tuning modulation (min-to-max distance) for each unit in a
    neural design matrix and set of stimuli.

    Parameters
    ----------
    X : ndarray (samples, units)
        Neural data design matrix.
    stimuli : ndarray (samples,)
        The stimulus value for each trial.
    aggregator : function
        Function used to aggregate statistics across samples per stimulus.

    Returns
    -------
    modulations : ndarray (units,)
        The min-to-max distance of each neuron's tuning curve.
    """
    tuning_curves = get_tuning_curve(X, stimuli, aggregator)
    modulations = np.max(tuning_curves, axis=1) - np.min(tuning_curves, axis=1)
    return modulations


def get_tuning_modulation_fraction(X, stimuli, aggregator=np.mean):
    """Gets the tuning modulation fraction (modulation to peak ratio) for each
    unit in a neural design matrix and set of stimuli.

    Parameters
    ----------
    X : ndarray (samples, units)
        Neural data design matrix.
    stimuli : ndarray (samples,)
        The stimulus value for each trial.
    aggregator : function
        Function used to aggregate statistics across samples per stimulus.

    Returns
    -------
    modulations : ndarray (units,)
        The min-to-max distance of each neuron's tuning curve.
    """
    tuning_curves = get_tuning_curve(X, stimuli, aggregator)
    modulations = np.max(tuning_curves, axis=1) - np.min(tuning_curves, axis=1)
    modulation_fractions = modulations / np.max(tuning_curves, axis=1)
    return modulation_fractions


def get_tuning_modulation_pvalue(X, stimuli, aggregator=np.mean):
    """Performs a Wilcoxon rank-sum test on the distribution of the max and
    min points of the tuning curve, and returns the p-value.

    Parameters
    ----------
    X : ndarray (samples, units)
        Neural data design matrix.
    stimuli : ndarray (samples,)
        The stimulus value for each trial.
    aggregator : function
        Function used to aggregate statistics across samples per stimulus.

    Returns
    -------
    pvalues : ndarray (units,)
        The p-values comparing the min-to-max distribution of the tuning curve.
    """
    n_samples, n_units, n_stimuli, unique_stimuli = X_stimuli(X, stimuli)
    # calculate tuning curves
    tuning_curves = get_tuning_curve(X, stimuli, aggregator)
    # get best and worst stimuli index for each unit
    max_idxs = np.argmax(tuning_curves, axis=1)
    min_idxs = np.argmin(tuning_curves, axis=1)

    # calculate p-value for each unit
    pvalues = np.zeros(n_units)
    for unit in range(n_units):
        # get best/worst stim for unit
        max_stim = unique_stimuli[max_idxs[unit]]
        min_stim = unique_stimuli[min_idxs[unit]]
        # get max/min response distribution for unit
        max_responses = X[stimuli == max_stim][:, unit]
        min_responses = X[stimuli == min_stim][:, unit]
        # calculate p-value for each unit
        _, pvalues[unit] = ss.ttest_ind(max_responses, min_responses, equal_var=False)

    return pvalues


def get_selectivity_index(X, stimuli, aggregator=np.mean, circular=360):
    """Gets the orientation/direction selectivity index for a set of neural
    responses and circular stimuli.

    Parameters
    ----------
    X : ndarray (samples, units)
        Neural data design matrix.
    stimuli : ndarray (samples,)
        The stimulus value for each trial.
    aggregator : function
        Function used to aggregate statistics across samples per stimulus.
    circular : float
        The circular bound of the stimulus set.

    Returns
    -------
    si : ndarray (units,)
        The selectivity index.
    """
    unique_stimuli = np.unique(stimuli)
    # calculate difference between preferred / orthogonal stimuli
    stim_difference = circular / 2
    # get preferred tuning
    tuning_curves = get_tuning_curve(X, stimuli, aggregator)
    preferred_stimuli = unique_stimuli[np.argmax(tuning_curves, 1)]
    # calculate orthogonal direction
    orthogonal_direction = (preferred_stimuli + stim_difference) % 360
    orthogonal_idx = np.searchsorted(unique_stimuli, orthogonal_direction)
    # calculate selectivity index
    r_max = np.max(tuning_curves, axis=1)
    r_orth = tuning_curves[np.arange(X.shape[1]), orthogonal_idx]
    si = (r_max - r_orth) / (r_max + r_orth)
    return si


def get_peak_response(X, stimuli, aggregator=np.mean):
    """Gets the tuning modulation (min-to-max distance) for each unit in a
    neural design matrix and set of stimuli.

    Parameters
    ----------
    X : ndarray (samples, units)
        Neural data design matrix.
    stimuli : ndarray (samples,)
        The stimulus value for each trial.
    aggregator : function
        Function used to aggregate statistics across samples per stimulus.

    Returns
    -------
    modulations : ndarray (units,)
        The min-to-max distance of each neuron's tuning curve.
    """
    tuning_curves = get_tuning_curve(X, stimuli, aggregator)
    peak_responses = np.max(tuning_curves, axis=1)
    return peak_responses


def get_tuned_units(
    X, stimuli, aggregator=np.mean, peak_response=2, tuning_criteria='p-value',
    alpha=0.05, modulation=4, modulation_frac=None, variance_to_mean=10
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
    modulation_frac : float or None
        If None, peak response criteria is not used. If float, serves as the
        minimum modulation (min-to-max distance of the tuning curve) required
        for the neuron to be considered tuned.
    variance_to_mean : float or None
        The maximum variance-to-mean ratio of each neural unit.

    Returns
    -------
    keep : ndarray (units,)
        A Boolean mask denoting which neurons to keep.
    """
    n_units = X.shape[1]
    keep = np.ones(n_units)
    # peak of each tuning curve must be some minimum value
    peak_responses = get_peak_response(X, stimuli, aggregator=aggregator)
    valid = peak_responses > peak_response
    keep = np.logical_and(keep, valid)
    # the min-to-max distance must be at least some minimum value
    if tuning_criteria == 'p-value':
        pvalues = get_tuning_modulation_pvalue(X, stimuli, aggregator)
        valid = pvalues < alpha
    elif tuning_criteria == 'modulation':
        tuning_modulations = get_tuning_modulation(X, stimuli, aggregator=aggregator)
        valid = tuning_modulations > modulation
    elif tuning_criteria == 'modulation_frac':
        tuning_modulations = get_tuning_modulation(X, stimuli, aggregator=aggregator)
        valid = (tuning_modulations / peak_responses) > modulation_frac
    elif tuning_criteria is None:
        return keep
    else:
        raise ValueError('Invalid tuning criteria.')
    keep = np.logical_and(keep, valid)

    return keep


def get_responsive_units(
    X, stimuli, aggregator=np.mean, peak_response=None, variance_to_mean=10.
):
    """Gets the units in a neural design matrix which exhibit enough neural
    activity to be considered responsive. Acts as a wrapper for the
    get_tuned_units function.

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
    variance_to_mean : float or None
        The maximum variance-to-mean ratio of each neural unit.

    Returns
    -------
    keep : ndarray (units,)
        A Boolean mask denoting which neurons to keep.
    """
    keep = get_tuned_units(X=X, stimuli=stimuli, aggregator=aggregator,
                           peak_response=peak_response, tuning_criteria=None,
                           variance_to_mean=variance_to_mean)
    return keep


def p_value_regions(p0, p1, alpha=0.01):
    """Returns the fraction of p-values in different regions.

    Parameters
    ----------
    p0, p1 : np.ndarray
        The p-values.

    Returns
    -------
    f_both : float
        The fraction of p-values that are significant in both cases.
    f0_only, f1_only : float
        The fraction of p-values that are significant for one set, but not the
        other.
    """
    n_samples = p0.size
    sig0 = p0 < alpha
    sig1 = p1 < alpha

    # calculate fractions in each region
    f_both = np.sum(sig0 & sig1) / n_samples
    f0_only = np.sum(sig0 & ~sig1) / n_samples
    f1_only = np.sum(~sig0 & sig1) / n_samples

    return f0_only, f1_only, f_both


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
