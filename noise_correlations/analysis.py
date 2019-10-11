import numpy as np
from scipy.special import comb
from itertools import combinations

from . null_models import shuffle_data, random_rotation
from .utils import mean_cov
from .discriminability import (corrected_lfi as clfi,
                               corrected_lfi_data as clfi_data,
                               mv_normal_jeffreys as sdkl,
                               mv_normal_jeffreys_data as sdkl_data)


def all_correlations(X):
    """Compute all pairwise correlations.

    Parameters
    ----------
    X: ndarray (units, stimuli, trials)
        Neural data.

    Returns
    -------
    corrs: ndarray ((units choose 2) * stimuli)
        All pairwise correlations across stimuli.
    """
    corrs = []
    for s in range(X.shape[1]):
        cc = np.corrcoef(X[:, s])
        idxs = np.triu_indices_from(cc, k=1)
        cc = cc[idxs[0], idxs[1]]
        corrs.append(cc)
    corrs = np.concatenate(corrs)
    corrs = corrs[~np.isnan(corrs)]
    return corrs


def generate_dimlet(X, dim, rng, circular_stim=False):
    n_units, n_stimuli, n_trials = X.shape
    unit_idxs = rng.permutation(n_units)[:dim]
    if circular_stim:
        stim_idxs = rng.randint(n_stimuli - 1)
        flip = 2 * rng.binomial(1, .5) - 1
        stim_idxs = np.mod(np.array([stim_idxs, stim_idxs + flip]),
                           n_stimuli)
    else:
        stim_idxs = rng.randint(n_stimuli - 1)
        stim_idxs = np.array([stim_idxs, stim_idxs + 1])
    return unit_idxs, stim_idxs


def inner_compare_nulls_measures(X, unit_idxs, stim_idxs, rng, n_samples):
    n_units, n_stimuli, n_trials = X.shape
    dim = unit_idxs.size
    Xu = X[unit_idxs]
    X0 = Xu[:, stim_idxs[0]].T
    X1 = Xu[:, stim_idxs[1]].T
    mu0, cov0 = mean_cov(X0)
    mu1, cov1 = mean_cov(X1)
    v_lfi = clfi(mu0, cov0, mu1, cov1, n_trials, dim, dtheta=1.)
    v_sdkl = sdkl(mu0, cov0, mu1, cov1)
    vs_lfi = np.zeros(n_samples)
    vs_sdkl = np.zeros(n_samples)
    vr_lfi = np.zeros(n_samples)
    vr_sdkl = np.zeros(n_samples)
    for jj in range(n_samples):
        shuffle_X = shuffle_data(np.concatenate([X0, X1], axis=1))
        X0s = shuffle_X[:, :dim]
        X1s = shuffle_X[:, dim:]
        vs_lfi[jj] = clfi_data(X0s, X1s, dtheta=1)
        vs_sdkl[jj] = sdkl_data(X0s, X1s)
        (mu0r, mu1r), (cov0r, cov1r) = random_rotation([mu0, mu1], [cov0, cov1], rng=rng)
        vr_lfi[jj] = clfi(mu0r, cov0r, mu1r, cov1r, n_samples, dim, dtheta=1)
        mu1r, cov1r = random_rotation(mu1, cov1, rng=rng)
        vr_sdkl[jj] = sdkl(mu0r, cov0r, mu1r, cov1r)
    return vs_lfi, vs_sdkl, vr_lfi, vr_sdkl, v_lfi, v_sdkl


def compare_nulls_measures(X, dim, n_dimlets, rng, n_samples=10000,
                           circular_stim=False):
    """Compare p-values across null models.

    Parameters
    ----------
    X: ndarray (units, stimuli, trials)
        Neural data.
    dim: int
        Number of units to consider.
    """
    n_units, n_stimuli, n_trials = X.shape

    p_s_lfi = np.zeros(n_dimlets)
    p_s_sdkl = np.zeros(n_dimlets)
    p_r_lfi = np.zeros(n_dimlets)
    p_r_sdkl = np.zeros(n_dimlets)
    v_lfi = np.zeros(n_dimlets)
    v_sdkl = np.zeros(n_dimlets)

    for ii in range(n_dimlets):
        unit_idxs, stim_idxs = generate_dimlet(X, dim, rng, circular_stim)
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
    return p_s_lfi, p_s_sdkl, p_r_lfi, p_r_sdkl, v_lfi, v_sdkl
