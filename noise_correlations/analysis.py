import numpy as np
from scipy.special import comb
from scipy.stats import spearmanr as sr
from scipy.stats import special_ortho_group as sog
from itertools import combinations

from .null_models import shuffle_data, random_rotation
from .utils import mean_cov, uniform_correlation_matrix
from .discriminability import (lfi, lfi_data,
                               mv_normal_jeffreys as sdkl,
                               mv_normal_jeffreys_data as sdkl_data)


def all_correlations(X, stimuli, spearmanr=False):
    """Compute all pairwise correlations.

    Parameters
    ----------
    X : ndarray (samples, units)
        Neural data design matrix.
    stimuli : ndarray (samples,)
        The stimulus value for each trial.

    Returns
    -------
    corrs: ndarray ((units choose 2) * stimuli)
        All pairwise correlations across stimuli.
    """
    corrs = []
    unique_stimuli = np.unique(stimuli)
    # get correlations across trials for unique stimuli
    for stim in unique_stimuli:
        # get responses corresponding to the current stimulus
        X_stim = X[np.argwhere(stimuli == stim).ravel()].T
        if spearmanr:
            cc = sr(X_stim, axis=1)[0]
        else:
            cc = np.corrcoef(X_stim)
        idxs = np.triu_indices_from(cc, k=1)
        cc = cc[idxs]
        corrs.append(cc)
    corrs = np.concatenate(corrs)
    return corrs


def generate_dimlet(X, stimuli, dim, rng, is_stim_circular=False):
    """Generates a random dimlet of given size from a neural design matrix and
    a random pair of stimuli from a stimulus vector.

    Parameters
    ----------
    X : ndarray (samples, units)
        Neural data design matrix.
    stimuli : ndarray (samples,)
        The stimulus value for each trial.
    dim : int
        Number of units to consider.
    rng : RandomState
        Random state instance.
    is_stim_circular : bool
        If True, stimulus values are assumed to be circular.

    Returns
    -------
    unit_idxs : ndarray (dim,)
        The indices for the units in the dimlet.
    stim_vals : ndarray
        The values of a randomly chosen pair of stimuli.
    """
    n_samples, n_units = X.shape
    unique_stimuli = np.unique(stimuli)
    n_stimuli = unique_stimuli.size
    # get a random sub-population of size dim
    unit_idxs = rng.permutation(n_units)[:dim]
    # choose random unique stimulus
    stim_idx = rng.randint(n_stimuli - 1)
    # get a pair of random stimuli values
    if is_stim_circular:
        flip = 2 * rng.binomial(1, .5) - 1
        stim_idxs = np.mod(np.array([stim_idx, stim_idx + flip]),
                           n_stimuli)
        stim_vals = np.sort(unique_stimuli[stim_idxs])
    else:
        stim_vals = unique_stimuli[[stim_idx, stim_idx + 1]]

    return unit_idxs, stim_vals


def inner_compare_nulls_measures(X, stimuli, unit_idxs, stim_vals, rng, n_reps,
                                 circular_stim=None):
    """Calculates values of metrics on a dimlet of a neural design matrix under
    both the shuffled and rotation null models.

    Parameters
    ----------
    X : ndarray (samples, units)
        Neural data design matrix.
    stimuli : ndarray (samples,)
        The stimulus value for each trial.
    unit_idxs : ndarray (dim,)
        The indices for the units in the dimlet.
    stim_vals : ndarray
        The values of a randomly chosen pair of stimuli.
    rng : RandomState
        Random state instance.
    n_reps : int
        The number of repetitions to consider when evaluating null models.
    circular_stim : None or float
        If None, stimulus will not be treated as circular. If int, takes on the
        value of the stimulus upper bound.

    Returns
    -------
    v_s_lfi, v_s_sdkl : ndarray (reps,)
        The values of the LFI/sDKL on the shuffled dimlets.

    v_r_lfi, v_r_sdkl : ndarray (reps,)
        The values of the LFI/sDKL on the rotated dimlets.

    v_lfi, v_sdkl : float
        The values of the LFI/sDKL on the original dimlet.
    """
    n_samples, n_units = X.shape
    # segment design matrix according to stimuli and units
    stim0_idx = np.argwhere(stimuli == stim_vals[0]).ravel()
    stim1_idx = np.argwhere(stimuli == stim_vals[1]).ravel()
    X0 = X[stim0_idx][:, unit_idxs]
    X1 = X[stim1_idx][:, unit_idxs]
    # sub-design matrix statistics
    mu0, cov0 = mean_cov(X0)
    mu1, cov1 = mean_cov(X1)
    # calculate stimulus difference
    dtheta = np.diff(stim_vals)
    if circular_stim is not None:
        dtheta = min(dtheta, np.diff(stim_vals[::-1]) + circular_stim)

    # calculate values of LFI and sDKL for original datasets
    v_lfi = lfi(mu0, cov0, mu1, cov1, dtheta=dtheta)
    v_sdkl = sdkl(mu0, cov0, mu1, cov1)
    # values for measures on shuffled data
    v_s_lfi = np.zeros(n_reps)
    v_s_sdkl = np.zeros(n_reps)
    # values for measures on rotated data
    v_r_lfi = np.zeros(n_reps)
    v_r_sdkl = np.zeros(n_reps)

    for jj in range(n_reps):
        # shuffle null model
        X0s = shuffle_data(X0, rng=rng)
        X1s = shuffle_data(X1, rng=rng)
        v_s_lfi[jj] = lfi_data(X0s, X1s, dtheta=dtheta)
        v_s_sdkl[jj] = sdkl_data(X0s, X1s)
        # rotation null model
        (mu0r, mu1r), (cov0r, cov1r) = random_rotation([mu0, mu1], [cov0, cov1], rng=rng)
        v_r_lfi[jj] = lfi(mu0r, cov0r, mu1r, cov1r, dtheta=dtheta)
        mu1r, cov1r = random_rotation(mu1, cov1, rng=rng)
        v_r_sdkl[jj] = sdkl(mu0r, cov0r, mu1r, cov1r)
    return v_s_lfi, v_s_sdkl, v_r_lfi, v_r_sdkl, v_lfi, v_sdkl


def compare_nulls_measures(X, stimuli, dim, n_dimlets, rng, n_reps=10000,
                           circular_stim=None):
    """Compare p-values across null models for linear Fisher information and
    symmetric KL-divergence.

    Parameters
    ----------
    X : ndarray (units, stimuli, trials)
        Neural data.
    stimuli : ndarray (samples,)
        The stimulus value for each trial.
    dim : int
        Number of units to consider in each dimlet.
    n_dimlets : int
        The number of dimlets over which to average measures.
    rng : RandomState
        Random state instance.
    n_reps : int
        The number of repetitions to consider when evaluating null models.
    circular_stim : None or float
        If None, stimulus will not be treated as circular. If int, takes on the
        value of the stimulus upper bound.

    Returns
    -------
    p_s_lfi, p_s_sdkl : ndarray (reps,)
        The p-values on the shuffled dimlets.

    p_r_lfi, p_r_sdkl : ndarray (reps,)
        The p-values on the rotated dimlets.

    v_lfi, v_sdkl : float
        The values of the LFI/sDKL on the original dimlet.
    """
    n_samples, n_units = X.shape
    is_stim_circular = circular_stim is not None
    # p-values for shuffled null model
    p_s_lfi = np.zeros(n_dimlets)
    p_s_sdkl = np.zeros(n_dimlets)
    # p-values for rotation null model
    p_r_lfi = np.zeros(n_dimlets)
    p_r_sdkl = np.zeros(n_dimlets)
    # values of metrics
    v_lfi = np.zeros(n_dimlets)
    v_sdkl = np.zeros(n_dimlets)

    # calculate p-values over dimlets
    for ii in range(n_dimlets):
        unit_idxs, stim_vals = generate_dimlet(X, stimuli, dim, rng, is_stim_circular)
        # calculate values under shuffle and rotation null models
        (vs_lfi, vs_sdkl,
         vr_lfi, vr_sdkl,
         vi_lfi, vi_sdkl) = inner_compare_nulls_measures(X, stimuli, unit_idxs,
                                                         stim_vals, rng, n_reps,
                                                         circular_stim)
        v_lfi[ii] = vi_lfi
        v_sdkl[ii] = vi_sdkl
        # calculate p-values
        p_s_lfi[ii] = np.mean(vs_lfi >= v_lfi[ii])
        p_s_sdkl[ii] = np.mean(vs_sdkl >= v_sdkl[ii])
        p_r_lfi[ii] = np.mean(vr_lfi >= v_lfi[ii])
        p_r_sdkl[ii] = np.mean(vr_sdkl >= v_sdkl[ii])
    return p_s_lfi, p_s_sdkl, p_r_lfi, p_r_sdkl, v_lfi, v_sdkl


def dist_compare_nulls_measures(X, dim, n_dimlets, rng, comm,
                                n_samples=10000, circular_stim=False,
                                all_stim=True):
    """Compare p-values across null models.

    Parameters
    ----------
    X: ndarray (units, stimuli, trials)
        Neural data.
    dim: int
        Number of units to consider.
    """
    from mpi_utils.ndarray import Gatherv_rows

    n_units, n_stimuli, n_trials = X.shape
    size = comm.size
    rank = comm.rank

    if all_stim:
        if circular_stim:
            stims = np.arange(n_stimuli)
        else:
            stims = np.arange(n_stimuli - 1)
        if n_dimlets >= comb(n_units, dim, exact=True):
            units = np.array(list(combinations(np.arange(n_units), dim)))
        else:
            units = np.zeros((n_dimlets, dim), dtype=int)
            for ii in range(n_dimlets):
                unit_idxs, _ = generate_dimlet(X, dim, rng, circular_stim)
                units[ii] = unit_idxs
        n_comb = units.shape[0]
        n_stim = stims.shape[0]
        units = np.concatenate([units] * n_stim)
        stims = np.tile(stims[np.newaxis], (n_comb, 1)).T.ravel()
        stims = np.stack([stims, stims + 1], axis=-1)
        if circular_stim:
            stims = np.mod(stims, n_stimuli)
    else:
        if n_dimlets >= comb(n_units, dim, exact=True):
            units = np.array(list(combinations(np.arange(n_units), dim)))
            if circular_stim:
                stims = rng.randint(n_stimuli, size=units.shape[0])
            else:
                stims = rng.randint(n_stimuli - 1, size=units.shape[0])
            stims = np.concatenate([stims, stims + 1])
            if circular_stim:
                stims = np.mod(stims, n_stimuli)
        else:
            units = np.zeros((n_dimlets, dim), dtype=int)
            stims = np.zeros((n_dimlets, 2), dtype=int)
            for ii in range(n_dimlets):
                unit_idxs, stim_idxs = generate_dimlet(X, dim, rng, circular_stim)
                units[ii] = unit_idxs
                stims[ii] = stim_idxs
    units = np.array_split(units, size)[rank]
    stims = np.array_split(stims, size)[rank]

    my_dimlets = units.shape[0]
    p_s_lfi = np.zeros(my_dimlets)
    p_s_sdkl = np.zeros(my_dimlets)
    p_r_lfi = np.zeros(my_dimlets)
    p_r_sdkl = np.zeros(my_dimlets)
    v_lfi = np.zeros(my_dimlets)
    v_sdkl = np.zeros(my_dimlets)

    for ii in range(my_dimlets):
        if rank == 0:
            print(dim, '{} out of {}'.format(ii + 1, my_dimlets))
        unit_idxs, stim_idxs = units[ii], stims[ii]
        (vs_lfi, vs_sdkl,
         vr_lfi, vr_sdkl,
         vi_lfi, vi_sdkl) = inner_compare_nulls_measures(X, unit_idxs,
                                                         stim_idxs,
                                                         rng,
                                                         n_samples)
        v_lfi[ii] = vi_lfi
        v_sdkl[ii] = vi_sdkl
        p_s_lfi[ii] = np.mean(vs_lfi >= v_lfi[ii])
        p_s_sdkl[ii] = np.mean(vs_sdkl >= v_sdkl[ii])
        p_r_lfi[ii] = np.mean(vr_lfi >= v_lfi[ii])
        p_r_sdkl[ii] = np.mean(vr_sdkl >= v_sdkl[ii])
    p_s_lfi = Gatherv_rows(p_s_lfi, comm)
    p_s_sdkl = Gatherv_rows(p_s_sdkl, comm)
    p_r_lfi = Gatherv_rows(p_r_lfi, comm)
    p_r_sdkl = Gatherv_rows(p_r_sdkl, comm)
    v_lfi = Gatherv_rows(v_lfi, comm)
    v_sdkl = Gatherv_rows(v_sdkl, comm)
    return p_s_lfi, p_s_sdkl, p_r_lfi, p_r_sdkl, v_lfi, v_sdkl


def dist_compare_dtheta(X, dim, n_dimlets, rng, comm, n_samples=10000,
                        circular_stim=True):
    """Compare p-values across null models.

    Parameters
    ----------
    X: ndarray (units, stimuli, trials)
        Neural data.
    dim: int
        Number of units to consider.
    """
    from mpi_utils.ndarray import Gatherv_rows

    n_units, n_stimuli, n_trials = X.shape
    size = comm.size
    rank = comm.rank

    stims = np.array(list(combinations(np.arange(n_stimuli), 2)))
    if n_dimlets >= comb(n_units, dim, exact=True):
        units = np.array(list(combinations(np.arange(n_units), dim)))
    else:
        units = np.zeros((n_dimlets, dim), dtype=int)
        for ii in range(n_dimlets):
            unit_idxs, _ = generate_dimlet(X, dim, rng, circular_stim)
            units[ii] = unit_idxs
    n_comb = units.shape[0]
    n_stim = stims.shape[0]
    units = np.concatenate([units] * n_stim)
    o_stims = np.tile(stims[:, np.newaxis], (1, n_comb, 1)).reshape(-1, 2)
    units = np.array_split(units, size)[rank]
    stims = np.array_split(o_stims, size)[rank]

    my_dimlets = units.shape[0]
    v_lfi = np.zeros(my_dimlets)
    v_sdkl = np.zeros(my_dimlets)
    p_s_lfi = np.zeros(my_dimlets)
    p_s_sdkl = np.zeros(my_dimlets)
    p_r_lfi = np.zeros(my_dimlets)
    p_r_sdkl = np.zeros(my_dimlets)

    for ii in range(my_dimlets):
        if rank == 0:
            print(dim, '{} out of {}'.format(ii + 1, my_dimlets))
        unit_idxs, stim_idxs = units[ii], stims[ii]
        stim_idxs = np.sort(stim_idxs)
        (vs_lfi, vs_sdkl,
         vr_lfi, vr_sdkl,
         vi_lfi, vi_sdkl) = inner_compare_nulls_measures(X, unit_idxs,
                                                         stim_idxs,
                                                         rng,
                                                         n_samples)
        v_lfi[ii] = vi_lfi
        v_sdkl[ii] = vi_sdkl
        p_s_lfi[ii] = np.mean(vs_lfi >= v_lfi[ii])
        p_s_sdkl[ii] = np.mean(vs_sdkl >= v_sdkl[ii])
        p_r_lfi[ii] = np.mean(vr_lfi >= v_lfi[ii])
        p_r_sdkl[ii] = np.mean(vr_sdkl >= v_sdkl[ii])
    p_s_lfi = Gatherv_rows(p_s_lfi, comm)
    p_s_sdkl = Gatherv_rows(p_s_sdkl, comm)
    p_r_lfi = Gatherv_rows(p_r_lfi, comm)
    p_r_sdkl = Gatherv_rows(p_r_sdkl, comm)
    v_lfi = Gatherv_rows(v_lfi, comm)
    v_sdkl = Gatherv_rows(v_sdkl, comm)
    return p_s_lfi, p_s_sdkl, p_r_lfi, p_r_sdkl, v_lfi, v_sdkl, o_stims


def dist_synthetic_data(dim, n_deltas, n_rotations, rng, comm, dim_size=10,
                        n_samples=10000):
    """Compare p-values across null models.

    Parameters
    ----------
    X: ndarray (units, stimuli, trials)
        Neural data.
    dim: int
        Number of units to consider.
    """
    from mpi_utils.ndarray import Gatherv_rows

    cov0 = uniform_correlation_matrix(dim, 5, .1, noise_std=0.01, rng=rng)
    cov1 = cov0.copy()
    deltas = np.logspace(-1.5, 0, n_deltas)
    n_deltas = deltas.size
    size = comm.size
    rank = comm.rank

    deltas = np.concatenate([deltas] * n_rotations)
    my_deltas = np.array_split(deltas, size)[rank]
    R0s = sog.rvs(dim, size=n_samples, random_state=rng)
    R1s = sog.rvs(dim, size=n_samples, random_state=rng)

    v_lfi = np.zeros(my_deltas.size)
    v_sdkl = np.zeros(my_deltas.size)
    p_s_lfi = np.zeros(my_deltas.size)
    p_s_sdkl = np.zeros(my_deltas.size)
    p_r_lfi = np.zeros(my_deltas.size)
    p_r_sdkl = np.zeros(my_deltas.size)

    for ii, delta in enumerate(my_deltas):
        mu0 = delta * np.ones(dim) / np.sqrt(dim)
        mu1 = -mu0
        if rank == 0:
            print('{} out of {}'.format(ii + 1, my_deltas.size))
        X0_zm = rng.multivariate_normal([0., 0.], cov0, size=size)
        X1_zm = rng.multivariate_normal([0., 0.], cov1, size=size)
        r = sog.rvs(dim, random_state=rng)
        X0 = X0_zm.dot(r) + mu0[np.newaxis]
        r = sog.rvs(dim, random_state=rng)
        X1 = X1_zm.dot(r) + mu1[np.newaxis]
        vs_lfi = np.zeros(n_samples)
        vs_sdkl = np.zeros(n_samples)
        vr_lfi = np.zeros(n_samples)
        vr_sdkl = np.zeros(n_samples)
        mu0, cov0 = mean_cov(X0)
        mu1, cov1 = mean_cov(X1)
        v_lfi[ii] = lfi(mu0, cov0, mu1, cov1)
        v_sdkl[ii] = sdkl(mu0, cov0, mu1, cov1)
        for jj, (R0, R1) in enumerate(zip(R0s, R1s)):
            X0s = shuffle_data(X0, rng=rng)
            X1s = shuffle_data(X1, rng=rng)
            vs_lfi[jj] = lfi_data(X0s, X1s)
            vs_sdkl[jj] = sdkl_data(X0s, X1s)
            cov0r = R0.dot(cov0.dot(R0.T))
            cov1r = R0.dot(cov1.dot(R0.T))
            vr_lfi[jj] = lfi(mu0, cov0r, mu1, cov1r)
            cov1r = R1.dot(cov1.dot(R1.T))
            vr_sdkl[jj] = sdkl(mu0, cov0r, mu1, cov1r)
        p_s_lfi[ii] = np.mean(vs_lfi >= v_lfi[ii])
        p_s_sdkl[ii] = np.mean(vs_sdkl >= v_sdkl[ii])
        p_r_lfi[ii] = np.mean(vr_lfi >= v_lfi[ii])
        p_r_sdkl[ii] = np.mean(vr_sdkl >= v_sdkl[ii])
    p_s_lfi = Gatherv_rows(p_s_lfi, comm)
    p_s_sdkl = Gatherv_rows(p_s_sdkl, comm)
    p_r_lfi = Gatherv_rows(p_r_lfi, comm)
    p_r_sdkl = Gatherv_rows(p_r_sdkl, comm)
    v_lfi = Gatherv_rows(v_lfi, comm)
    v_sdkl = Gatherv_rows(v_sdkl, comm)
    return p_s_lfi, p_s_sdkl, p_r_lfi, p_r_sdkl, v_lfi, v_sdkl
