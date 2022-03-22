import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as ss
from scipy.optimize import minimize
from sklearn.decomposition import FactorAnalysis as FA
import warnings
import torch


def _lfi(mu0, mu1, cov, dtheta=1.):
    """Calculate the linear Fisher information from two data matrices.
    Pytorch version should always have full rank cov.

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
    dmu_dtheta = (mu1 - mu0) / dtheta
    return dmu_dtheta.mm(torch.linalg.solve(cov, dmu_dtheta.t()))


class FACov:
    def __init__(self, X, k=None):
        """Class to represent a multivariate covarince parameterized by a
        Factor Analysis model. Can rotate the shared variability.

        Parameters
        ----------
        X : ndarray (samples, units)
            Neural data.
        k : int
            Number of factors to include. If `None`, a heuristic will be used to
            chose the largest `k` such that the model is identifiable.
        """
        d = X.shape[1]
        if d < 3:
            raise ValueError('FA model only works for d > 2.')
        self.k = k
        self.mean = X.mean(axis=0, keepdims=True)
        X = X - self.mean
        if k is None:
            kmax = (d - 1) // 2
            for ki in range(1, kmax + 1):
                model = FA(n_components=ki, tol=1e-4,
                           svd_method='lapack', noise_variance_init=np.var(X, axis=0))
                model0 = FA(n_components=ki, tol=1e-4,
                            svd_method='lapack')
                model.fit(X)
                model0.fit(X)
                cc = np.corrcoef(model.noise_variance_,
                                 model0.noise_variance_)[0, 1]
                if cc < .975:
                    self.k = ki - 1
                    break
                self.k = ki
            if self.k == 0:
                warnings.warn("FA model was not well constrained for any `k`,"
                              " setting `k=1`.", RuntimeWarning)
            self.k = max(self.k, 1)
            model = FA(n_components=self.k, tol=1e-4, svd_method='lapack',
                       noise_variance_init=np.var(X, axis=0))
            model.fit(X)
        else:
            model = FA(n_components=k, tol=1e-4, svd_method='lapack',
                       noise_variance_init=np.var(X, axis=0))
        model.fit(X)
        self.private = model.noise_variance_
        self.shared = model.components_

    def params(self, R=None):
        """Return the mean and covariance.

        Parameters
        ----------
        R : ndarray
            Optional rotation matrix.
        Returns
        -------
        mean, cov
        """
        if R is None:
            cov = np.diag(self.private) + self.shared.T @ self.shared
        else:
            shared = self.shared @ R.T
            cov = np.diag(self.private) + shared.T @ shared
        return self.mean.ravel(), cov

    def get_optimal_orientation(self, mu0, mu1):
        """Calculate the optimal cov by rotating the shared variability in the
        FA model.

        Parameters
        ----------
        mu0 : ndarray (dim,)
        mu1 : ndarray (dim,)

        Returns
        -------
        cov
        """
        def make_cov(paramst, shared, private):
            dim = shared.shape[1]
            At = paramst.reshape(dim, dim)
            At = (At - At.t()) / 2.
            R = torch.matrix_exp(At)
            cov = torch.chain_matmul(R, shared.t(), shared, R.t()) + private
            return cov

        def f_df(params, shared, private, mu0, mu1):
            paramst = torch.tensor(params, requires_grad=True)
            cov = make_cov(paramst, shared, private)
            loss = -_lfi(mu0[np.newaxis], mu1[np.newaxis], cov)
            loss.backward()
            loss = loss.detach().numpy()
            grad = paramst.grad.detach().numpy()
            return loss, grad

        dim = self.shared.shape[1]
        shared = torch.tensor(self.shared)
        private = torch.tensor(np.diag(self.private))
        args = shared, private, torch.tensor(mu0), torch.tensor(mu1)
        x0 = np.zeros(dim**2)
        x = minimize(f_df, x0, method='L-BFGS-B', jac=True, args=args).x
        paramst = torch.tensor(x)
        opt_cov = make_cov(paramst, shared, private).numpy()
        return opt_cov


def make_corr(paramst, d):
    """Helper function for turning a list of params into a
    correlations matrix.
    """
    X = paramst.reshape(d, d)
    X = X / torch.norm(X, dim=0)
    return X.t() @ X


def f_df_corr(params, d, mu0, mu1, sigma):
    """Loss and gradient to optimize correlation matrix.
    """
    paramst = torch.tensor(params, requires_grad=True)
    corr = make_corr(paramst, d)
    cov = torch.eye(d, dtype=paramst.dtype) * 1e-10 + corr * torch.outer(sigma, sigma)
    loss = -_lfi(mu0, mu1, cov)
    X = paramst.reshape(d, d)
    loss = loss + 0.1 * torch.sum((1. - torch.norm(X, dim=0))**2)
    loss.backward()
    loss = loss.detach().numpy()
    grad = paramst.grad.detach().numpy()
    return loss, grad


def lfi_uniform_corr_opt_cov(var, mu0, mu1, rng, n_restarts=10):
    """Optimize the covariance correlations to maximize LFI.

    Parameters
    ----------
    var : ndarray (d,)
        Array from the covariance diagonal.
    mu0 : ndarray (1, d)
        Mean for stim 0
    mu1 : ndarray (1, d)
        Mean for stim 1
    rng : RandomState
        Numpy random generator.
    n_restarts : int
        Number of re-initializations. This problem is not convex, so
        we try and find the highest LFI over a few restarts.
    """
    lfi_keep = -np.inf
    cov_keep = None
    sigma = torch.tensor(np.sqrt(var))
    d = sigma.shape[0]
    mu0 = torch.tensor(mu0)[np.newaxis]
    mu1 = torch.tensor(mu1)[np.newaxis]
    args = d, mu0, mu1, sigma

    for ii in range(n_restarts):
        X = rng.standard_normal(size=d**2)
        X /= np.linalg.norm(X, axis=0)
        try:
            params = minimize(f_df_corr, X, method='L-BFGS-B', jac=True, args=args).x
            paramst = torch.tensor(params)
            opt_cov = torch.eye(d, dtype=paramst.dtype) * 1e-10 + make_corr(paramst, d)
            opt_cov = opt_cov * torch.outer(sigma, sigma)
            lfi = np.squeeze(_lfi(mu0, mu1, opt_cov).numpy())
        except RuntimeError:
            lfi = -np.inf
        if lfi > lfi_keep:
            cov_keep = opt_cov.numpy()
            lfi_keep = lfi
    if lfi_keep < 0.:
        cov_keep = np.zeros((d, d))
    return cov_keep


def circular_difference(v1, v2, maximum=360):
    """Calculates the circular difference between two vectors, with some
    maximum."""
    smaller = np.minimum(v1, v2)
    bigger = np.maximum(v1, v2)
    diff1 = bigger - smaller
    diff2 = smaller + (maximum - bigger)
    return np.minimum(diff1, diff2)


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
    if peak_response is not None:
        valid = peak_responses > peak_response
    else:
        valid = True
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


def compute_angle(v1, v2):
    """Computes the angle between two vectors.

    Parameters
    ----------
    v1, v2 : np.ndarray
        The vectors.

    Returns
    -------
    angle : float
        The angle, in degrees, between the two vectors.
    """
    angle = np.rad2deg(np.arccos(
        np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    ))
    return angle


def angle2R(angle):
    """Creates a rotation matrix in two dimensions. Assumes incoming angle is
    in degrees."""
    angle = np.deg2rad(angle)
    R = np.array([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle), np.cos(angle)]])
    return R


def participation_ratio(cov):
    """Calculate the participation ratio of a covariance matrix.

    Parameters
    ----------
    cov : np.ndarray
        Covariance matrix.

    Returns
    -------
    pr : float
        The participation ratio.
    """
    u, _ = np.linalg.eigh(cov)
    pr = np.sum(u)**2 / np.sum(u**2)
    return pr


def participation_ratio_eig(eig):
    """Calculate the participation ratio of a set of eigenvalues.

    Parameters
    ----------
    eig : np.ndarray
        The eigenvalues.

    Returns
    -------
    pr : float
        The participation ratio.
    """
    pr = np.sum(eig)**2 / np.sum(eig**2)
    return pr


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


def reflection(A, v):
    """The reflection function. Useful helper for calculating rotation matrices.

    Parameters
    ----------
    A : np.ndarray
        A matrix.
    v : np.ndarray
        A vector.
    """
    norm = np.dot(v, v)
    return A - (2 / norm) * np.outer(v, A @ v)


def get_rotation_for_vectors(v1, v2):
    """The reflection function. Useful helper for calculating rotation matrices.

    Parameters
    ----------
    v1, v2 : np.ndarray
        The vectors between which to find a rotation matrix. The rotation matrix
        rotates from v1 to v2.

    Returns
    -------
    R : np.ndarray
        The rotation matrix.
    """
    dim = v1.size
    S = reflection(np.identity(dim), v1 + v2)
    R = reflection(S, v2)
    return R


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


def cov2corr(cov):
    """Convert a covariance matrix to a correlation matrix.

    Parameters
    ----------
    cov : np.ndarray, shape (n_units, n_units)
        The covariance matrix.

    Returns
    -------
    corr : np.ndarray, shape (n_units, n_units)
        The correlation matrix.
    """
    stdevs = np.sqrt(np.diag(cov))
    outer = np.outer(stdevs, stdevs)
    corr = cov / outer
    return corr


def corr2cov(corr, var):
    """Converts a correlation matrix to a covariance matrix, given a set of
    variances.

    Parameters
    ----------
    corr : np.ndarray, shape (n_units, n_units)
        The correlation matrix.
    var : np.ndarray, shape (n_units,)
        A vector of variances for the units in the correlation matrix.

    Returns
    -------
    cov : np.ndarray, shape (n_units, n_units)
        The covariance matrix.
    """
    stdevs = np.sqrt(var)
    outer = np.outer(stdevs, stdevs)
    cov = corr * outer
    return cov


def get_dimstim_responses(X, stimuli, units, stims):
    """Subsets the response matrix into two separate matrices, corresponding
    to the responses for a dimlet of neurons to a pair of stimuli.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_units)
        Neural data design matrix.
    stimuli : np.ndarray, shape (n_samples,)
        The stimulus value for each trial.
    units : np.ndarray, shape (n_units,)
        The units in the dimlet.
    stims : np.ndarray, shape (2,)
        The stimulus values for the stimulus pairing.
    """
    stim1, stim2 = stims
    X1 = X[stimuli == stim1][:, units]
    X2 = X[stimuli == stim2][:, units]
    return X1, X2


def get_dimstim_responses_from_h5(h5, dim_idx, dimstim_idx):
    # Obtain all data
    X = h5['X'][:]
    stimuli = h5['stimuli'][:]
    # Base dimension is the number of units in the smallest dimlet of this
    # experiment
    base_dim = np.max(np.argwhere(h5['units'][0, 0] != 0).ravel()) + 1
    # Read in the dimlet and dimstim
    dimlet = h5['units'][dim_idx, dimstim_idx, :dim_idx + base_dim].astype('int')
    stims = h5['stims'][dim_idx, dimstim_idx]
    # Get separate design matrices
    X1, X2 = get_dimstim_responses(X, stimuli, dimlet, stims)
    return X1, X2


def read_avg_cov(h5, dim_idx, dimstim_idx):
    # Get dimstim responses
    X1, X2 = get_dimstim_responses_from_h5(h5, dim_idx, dimstim_idx)
    avg_cov = 0.5 * (np.cov(X1.T) + np.cov(X2.T))
    return avg_cov


def find_fraction(ps, value=2/3):
    """For a series of percentile arrays, `ps`, find the fraction of samples such
    that the median is at least `value`.

    Parameters
    ----------
    ps : ndarray (dims, percentiles)
        Array of percentiles, analysis is done row-wise.
    value : float
        Percentile value for median to be above or equal to.

    """
    ps = np.atleast_2d(ps)
    ps = np.sort(ps, axis=1)
    ndims, n_samples = ps.shape
    fractions = np.zeros(ndims)
    for ii in range(ndims):
        p = ps[ii]
        if p[-1] < value:
            fractions[ii] = 0.
        elif p[0] >= value:
            fractions[ii] = 1.
        else:
            lower = 0
            upper = n_samples - 1
            medl = np.median(p[lower:])
            medu = np.median(p[upper:])
            while upper > lower + 1:
                idx = (lower + upper) // 2
                medi = np.median(p[idx:])
                if medi > value:
                    upper = idx
                else:
                    lower = idx
                upper = max(upper, lower + 1)
            fractions[ii] = (n_samples - upper) / n_samples
    return np.squeeze(fractions)


def run_bootstrap(data, func, ci):
    """Calculate bootstrap confidence intervals on data for the
    statistics defined by func().

    Parameters
    ----------
    data : ndarray (n_samples,) or (dims, n_samples)
        Data to bootstrap.
    func : callable
        Should accept an array of samples and return a scalar.
    ci : float
        Desired confidence interval.
    """
    many = False
    if isinstance(data, np.ndarray):
        ndim = data.ndim
    else:
        ndim = data[0].ndim
        many = True

    if ndim == 1:
        out = np.zeros(3)
        if many:
            out[1] = func(*data)
            stats = ss.bootstrap(data, func, vectorized=False, method='basic',
                                 confidence_level=ci, paired=True).confidence_interval
        else:
            out[1] = func(data)
            stats = ss.bootstrap([data], func, vectorized=False, method='basic',
                                 confidence_level=ci).confidence_interval
        out[0] = stats.low
        out[2] = stats.high
    elif ndim == 2:
        if many:
            out = np.zeros((data[0].shape[0], 3))
            for ii in range(data[0].shape[0]):
                out[ii, 1] = func(*[d[ii] for d in data])
                stats = ss.bootstrap([d[ii] for d in data], func, vectorized=False, method='basic',
                                     confidence_level=ci, paired=True).confidence_interval
                out[ii, 0] = stats.low
                out[ii, 2] = stats.high
        else:
            out = np.zeros((data.shape[0], 3))
            for ii in range(data.shape[0]):
                out[ii, 1] = func(data[ii])
                stats = ss.bootstrap([data[ii]], func, vectorized=False, method='basic',
                                     confidence_level=ci).confidence_interval
                out[ii, 0] = stats.low
                out[ii, 2] = stats.high
    else:
        raise ValueError

    return out
