import numpy as np
from scipy.special import comb
from scipy.stats import spearmanr as sr
from scipy.stats import special_ortho_group as sog
from itertools import combinations

from .null_models import shuffle_data, random_rotation
from .utils import mean_cov, uniform_correlation_matrix, X_stimuli
from .discriminability import (lfi, lfi_data,
                               mv_normal_jeffreys as sdkl,
                               mv_normal_jeffreys_data as sdkl_data)


def all_correlations(X, stimuli, u1=None, u2=None, spearmanr=False):
    """Compute all pairwise correlations.

    Parameters
    ----------
    X : ndarray (samples, units)
        Neural data design matrix.
    stimuli : ndarray (samples,)
        The stimulus value for each trial.
    u1, u2 : ndarray (units,)
        The sub-population to consider correlations between. If None, all
        correlations are stored.

    Returns
    -------
    corrs: ndarray ((units choose 2) * stimuli)
        All pairwise correlations across stimuli.
    """
    corrs = []
    n_samples, n_units, n_stimuli, unique_stimuli = X_stimuli(X, stimuli)

    # no sub-populations provided, examine all correlations
    if u1 is None and u2 is None:
        idxs = np.triu_indices(n=n_units, k=1)
    # examine within sub-population correlation specified by u1
    elif u1 is not None and u2 is None:
        if u1.dtype == 'bool':
            u1 = np.argwhere(u1).ravel()
        idxs = np.array([[idx1, idx2]
                         for idx1 in u1 for idx2 in u1
                         if idx2 < idx1])
        idxs = (idxs[:, 0], idxs[:, 1])
    # examine between sub-population correlation specified by u1/u2
    elif u1 is not None and u2 is not None:
        if u1.dtype == 'bool':
            u1 = np.argwhere(u1).ravel()
        if u2.dtype == 'bool':
            u2 = np.argwhere(u2).ravel()
        idxs = np.array([[idx1, idx2]
                         for idx1 in u1 for idx2 in u2])
        idxs = (idxs[:, 0], idxs[:, 1])

    # get correlations across trials for unique stimuli
    for stim in unique_stimuli:
        # get responses corresponding to the current stimulus
        X_stim = X[np.argwhere(stimuli == stim).ravel()].T
        if spearmanr:
            cc = sr(X_stim, axis=1)[0]
        else:
            cc = np.corrcoef(X_stim)
        cc = cc[idxs]
        corrs.append(cc)
    corrs = np.concatenate(corrs)
    return corrs


def generate_dimlet(n_units, n_dim, rng):
    """Generates a random dimlet of given size from a neural design matrix and
    a random pair of stimuli from a stimulus vector.

    Parameters
    ----------
    n_units : int
        The number of units in the population.
    n_dim : int
        The number of dimensions in the dimlet.
    rng : RandomState
        Random state instance.

    Returns
    -------
    unit_idxs : ndarray (n_dim,)
        The indices for the units in the dimlet.
    stim_vals : ndarray
        The values of a randomly chosen pair of stimuli.
    """
    unit_idxs = rng.permutation(n_units)[:n_dim]
    return unit_idxs


def generate_stim_pair(stimuli, rng, circular_stim=False):
    """Generates a random dimlet of given size from a neural design matrix and
    a random pair of stimuli from a stimulus vector.

    Parameters
    ----------
    stimuli : ndarray (n_samples,)
        The stimulus for each sample in the population.
    rng : RandomState
        Random state instance.
    circular_stim : bool
        Indicates whether the stimulus is circular.

    Returns
    -------
    stim_vals : ndarray
        The values of a randomly chosen pair of stimuli.
    """
    unique_stimuli = np.unique(stimuli)
    n_stimuli = unique_stimuli.size

    # choose random index for stimulus
    if circular_stim:
        stim_idx = rng.randint(n_stimuli)
    else:
        stim_idx = rng.randint(n_stimuli - 1)

    # get stim values from unique stimuli set
    stim_vals = np.sort(np.array(
        [unique_stimuli[stim_idx], unique_stimuli[(stim_idx + 1) % n_stimuli]]
    ))
    return stim_vals


def generate_dimlets_and_stim_pairs(
    n_units, stimuli, n_dim, n_dimlets, rng, all_stim=True, circular_stim=False
):
    """Generates a set of dimlets and stimulus pairs collectively, based on
    provided criteria.

    Parameters
    ----------
    n_units : int
        The number of units in the population.
    stimuli : ndarray (n_samples,)
        The stimulus for each sample in the population.
    n_dim : int
        The number of dimensions in the dimlet.
    n_dimlets : int
        The number of dimlets over which to average measures.
    rng : RandomState
        Random state instance.
    all_stim : bool
        If True, all neighboring pairs of stimuli are used for each dimlet.
        If False, one stimulus pair is chosen randomly for each dimlet.
    circular_stim : bool
        Indicates whether the stimulus is circular.

    Returns
    -------
    units : ndarray (n_sets, n_dim)
        The units in each dimlet. The number of unique dimlets is given by
        n_dimlets, but this array may contain repetitions corresponding
        to multiple stimuli-pairs, depending on all_stim.
    stims : ndarray (n_sets, n_dim)
        The stimulus-pair for all comparisons to make. The pairs are chosen
        to span all possible neighboring pairs for the stimuli (all_stim), or
        randomly chosen for each dimlet.
    """
    unique_stimuli = np.unique(stimuli)
    n_stimuli = unique_stimuli.size

    # calculate maximum possible number of dimlets
    max_dimlets = comb(n_units, n_dim, exact=True)

    # each dimlet will be evaluated on all neighboring pairwise stim combinations
    if all_stim:
        # get left side of each stimulus pair
        if circular_stim:
            stim_idx = np.arange(n_stimuli)
        else:
            stim_idx = np.arange(n_stimuli - 1)
        stims = np.stack([unique_stimuli[stim_idx],
                          unique_stimuli[(stim_idx + 1) % n_stimuli]],
                         axis=1)
        stims = np.sort(stims, axis=1)

        # check if number of dimlets is greater than maximum possible
        if n_dimlets >= max_dimlets:
            units = np.array(list(combinations(np.arange(n_units), n_dim)))
        else:
            # otherwise, randomly choose dimlets
            units = np.zeros((n_dimlets, n_dim), dtype=int)
            for ii in range(n_dimlets):
                units[ii] = generate_dimlet(n_units, n_dim, rng)

        n_unit_sets = units.shape[0]
        n_stim_sets = stims.shape[0]
        # repeat the units for each stim-set
        units = np.repeat(units, n_stim_sets, axis=0)
        # repeat the stims for each dimlet
        stims = np.tile(stims, (n_unit_sets, 1))

    # dimlet and stim-pair are chosen randomly, together
    else:
        # if number of dimlets is too large, use all dimlets
        if n_dimlets >= max_dimlets:
            units = np.array(list(combinations(np.arange(n_units), n_dim)))
            if circular_stim:
                stim_idx = rng.randint(n_stimuli, size=units.shape[0])
            else:
                stim_idx = rng.randint(n_stimuli - 1, size=units.shape[0])
            stims = np.stack([unique_stimuli[stim_idx],
                              unique_stimuli[(stim_idx + 1) % n_stimuli]],
                             axis=1)
        else:
            # choose each dimlet and stim-pair
            units = np.zeros((n_dimlets, n_dim), dtype=int)
            stims = np.zeros((n_dimlets, 2), dtype=int)
            for ii in range(n_dimlets):
                units[ii] = generate_dimlet(n_units, n_dim, rng)
                stims[ii] = generate_stim_pair(stimuli, rng, circular_stim)

    return units, stims


def inner_compare_nulls_measures(X, stimuli, unit_idxs, stim_vals, rng, n_repeats,
                                 circular_stim=False):
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
    n_repeats : int
        The number of repetitions to consider when evaluating null models.
    circular_stim : bool
        Indicates whether the stimulus is circular.

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
    if circular_stim:
        dtheta = np.ediff1d(np.unique(stimuli))[0]
    else:
        dtheta = np.diff(stim_vals).item()

    # calculate values of LFI and sDKL for original datasets
    v_lfi = lfi(mu0, cov0, mu1, cov1, dtheta=dtheta)
    v_sdkl = sdkl(mu0, cov0, mu1, cov1)
    # values for measures on shuffled data
    v_s_lfi = np.zeros(n_repeats)
    v_s_sdkl = np.zeros(n_repeats)
    # values for measures on rotated data
    v_r_lfi = np.zeros(n_repeats)
    v_r_sdkl = np.zeros(n_repeats)

    for jj in range(n_repeats):
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


def compare_nulls_measures(X, stimuli, n_dim, n_dimlets, rng, n_repeats=10000,
                           circular_stim=False):
    """Compare p-values across null models for linear Fisher information and
    symmetric KL-divergence.

    Parameters
    ----------
    X : ndarray (units, stimuli, trials)
        Neural data.
    stimuli : ndarray (samples,)
        The stimulus value for each trial.
    n_dim : int
        Number of units to consider in each dimlet.
    n_dimlets : int
        The number of dimlets over which to average measures.
    rng : RandomState
        Random state instance.
    n_repeats : int
        The number of repetitions to consider when evaluating null models.
    circular_stim : bool
        Indicates whether the stimulus is circular.

    Returns
    -------
    p_s_lfi, p_s_sdkl : ndarray (dimlets,)
        The p-values on the shuffled dimlets.
    p_r_lfi, p_r_sdkl : ndarray (dimlets,)
        The p-values on the rotated dimlets.
    v_lfi, v_sdkl : ndarray (dimlets,)
        The values of the LFI/sDKL on the original dimlet.
    """
    n_samples, n_units = X.shape
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
        unit_idxs = generate_dimlet(n_units, n_dim, rng)
        stim_vals = generate_stim_pair(stimuli, rng, circular_stim)
        # calculate values under shuffle and rotation null models
        (v_s_lfi, v_s_sdkl,
         v_r_lfi, v_r_sdkl,
         v_lfi[ii], v_sdkl[ii]) = \
            inner_compare_nulls_measures(
                X=X, stimuli=stimuli, unit_idxs=unit_idxs, stim_vals=stim_vals,
                rng=rng, n_repeats=n_repeats, circular_stim=circular_stim)

        # calculate p-values
        p_s_lfi[ii] = np.mean(v_s_lfi >= v_lfi[ii])
        p_s_sdkl[ii] = np.mean(v_s_sdkl >= v_sdkl[ii])
        p_r_lfi[ii] = np.mean(v_r_lfi >= v_lfi[ii])
        p_r_sdkl[ii] = np.mean(v_r_sdkl >= v_sdkl[ii])
    return p_s_lfi, p_s_sdkl, p_r_lfi, p_r_sdkl, v_lfi, v_sdkl


def calculate_null_measures(X, stimuli, n_dim, n_dimlets, rng,
                            n_repeats=10000, circular_stim=False,
                            all_stim=True):
    """Calculates null model distributions for linear Fisher information and
    symmetric KL-divergence.

    This function will calculate values for random dimlets, with neighboring
    pairwise stimuli.

    Parameters
    ----------
    X : ndarray (units, stimuli, trials)
        Neural data.
    stimuli : ndarray (samples,)
        The stimulus value for each trial.
    n_dim : int
        Number of units to consider in each dimlet.
    n_dimlets : int
        The number of dimlets over which to calculate p-values.
    rng : RandomState
        Random state instance.
    n_repeats : int
        The number of repetitions to consider when evaluating null models.
    circular_stim : bool
        Indicates whether the stimulus is circular.
    all_stim : bool
        If True, all consecutive pairs of stimuli are used.

    Returns
    -------
    p_s_lfi, p_s_sdkl : ndarray (dimlets,)
        The p-values on the shuffled dimlets.
    p_r_lfi, p_r_sdkl : ndarray (dimlets,)
        The p-values on the rotated dimlets.
    v_lfi, v_sdkl : ndarray (dimlets,)
        The values of the LFI/sDKL on the original dimlet.
    """
    n_samples, n_units = X.shape

    units, stims = generate_dimlets_and_stim_pairs(
        n_units=n_units, stimuli=stimuli, n_dim=n_dim, n_dimlets=n_dimlets,
        rng=rng, all_stim=all_stim, circular_stim=circular_stim
    )

    # Allocate storage for this rank's p-values
    n_pairings = units.shape[0]
    v_s_lfi = np.zeros((n_pairings, n_repeats))
    v_s_sdkl = np.zeros_like(v_s_lfi)
    v_r_lfi = np.zeros_like(v_s_lfi)
    v_r_sdkl = np.zeros_like(v_s_lfi)
    v_lfi = np.zeros(n_pairings)
    v_sdkl = np.zeros(n_pairings)

    for ii in range(n_pairings):
        unit_idxs, stim_vals = units[ii], stims[ii]
        # Calculate values under shuffle and rotation null models
        (v_s_lfi[ii], v_s_sdkl[ii],
         v_r_lfi[ii], v_r_sdkl[ii],
         v_lfi[ii], v_sdkl[ii]) = \
            inner_compare_nulls_measures(
                X=X, stimuli=stimuli, unit_idxs=unit_idxs, stim_vals=stim_vals,
                rng=rng, n_repeats=n_repeats, circular_stim=circular_stim)

    return v_s_lfi, v_s_sdkl, v_r_lfi, v_r_sdkl, v_lfi, v_sdkl, units, stims


def dist_compare_nulls_measures(X, stimuli, n_dim, n_dimlets, rng, comm,
                                n_repeats=10000, circular_stim=False,
                                all_stim=True):
    """Compare p-values across null models for linear Fisher information and
    symmetric KL-divergence, in a distributed manner.

    This function will calculate p-values for random dimlets, with neighboring
    pairwise stimuli.

    Parameters
    ----------
    X : ndarray (units, stimuli, trials)
        Neural data.
    stimuli : ndarray (samples,)
        The stimulus value for each trial.
    n_dim : int
        Number of units to consider in each dimlet.
    n_dimlets : int
        The number of dimlets over which to calculate p-values.
    rng : RandomState
        Random state instance.
    n_repeats : int
        The number of repetitions to consider when evaluating null models.
    circular_stim : bool
        Indicates whether the stimulus is circular.
    all_stim : bool
        If True, all consecutive pairs of stimuli are used.

    Returns
    -------
    p_s_lfi, p_s_sdkl : ndarray (dimlets,)
        The p-values on the shuffled dimlets.
    p_r_lfi, p_r_sdkl : ndarray (dimlets,)
        The p-values on the rotated dimlets.
    v_lfi, v_sdkl : ndarray (dimlets,)
        The values of the LFI/sDKL on the original dimlet.
    """
    from mpi_utils.ndarray import Gatherv_rows

    n_samples, n_units = X.shape

    size = comm.size
    rank = comm.rank

    units, stims = generate_dimlets_and_stim_pairs(
        n_units=n_units, stimuli=stimuli, n_dim=n_dim, n_dimlets=n_dimlets,
        rng=rng, all_stim=all_stim, circular_stim=circular_stim
    )
    # allocate units and stims to the current rank
    units = np.array_split(units, size)[rank]
    stims = np.array_split(stims, size)[rank]

    # allocate storage for this rank's p-values
    my_dimlets = units.shape[0]
    p_s_lfi = np.zeros(my_dimlets)
    p_s_sdkl = np.zeros(my_dimlets)
    p_r_lfi = np.zeros(my_dimlets)
    p_r_sdkl = np.zeros(my_dimlets)
    v_lfi = np.zeros(my_dimlets)
    v_sdkl = np.zeros(my_dimlets)

    for ii in range(my_dimlets):
        if rank == 0:
            print('Dimension %s' % n_dim, '{} out of {}'.format(ii + 1, my_dimlets))
        unit_idxs, stim_vals = units[ii], stims[ii]
        # calculate values under shuffle and rotation null models
        (v_s_lfi, v_s_sdkl,
         v_r_lfi, v_r_sdkl,
         v_lfi[ii], v_sdkl[ii]) = \
            inner_compare_nulls_measures(
                X=X, stimuli=stimuli, unit_idxs=unit_idxs, stim_vals=stim_vals,
                rng=rng, n_repeats=n_repeats, circular_stim=circular_stim)

        # calculate p-values
        p_s_lfi[ii] = np.mean(v_s_lfi >= v_lfi[ii])
        p_s_sdkl[ii] = np.mean(v_s_sdkl >= v_sdkl[ii])
        p_r_lfi[ii] = np.mean(v_r_lfi >= v_lfi[ii])
        p_r_sdkl[ii] = np.mean(v_r_sdkl >= v_sdkl[ii])

    # gather p-values across ranks
    p_s_lfi = Gatherv_rows(p_s_lfi, comm)
    p_s_sdkl = Gatherv_rows(p_s_sdkl, comm)
    p_r_lfi = Gatherv_rows(p_r_lfi, comm)
    p_r_sdkl = Gatherv_rows(p_r_sdkl, comm)
    v_lfi = Gatherv_rows(v_lfi, comm)
    v_sdkl = Gatherv_rows(v_sdkl, comm)
    return p_s_lfi, p_s_sdkl, p_r_lfi, p_r_sdkl, v_lfi, v_sdkl


def dist_calculate_nulls_measures(X, stimuli, n_dim, n_dimlets, rng, comm,
                                  n_repeats=10000, circular_stim=False,
                                  all_stim=True):
    """Calculates null model distributions for linear Fisher information and
    symmetric KL-divergence, in a distributed manner.

    This function will calculate values for random dimlets, with neighboring
    pairwise stimuli.

    Parameters
    ----------
    X : ndarray (units, stimuli, trials)
        Neural data.
    stimuli : ndarray (samples,)
        The stimulus value for each trial.
    n_dim : int
        Number of units to consider in each dimlet.
    n_dimlets : int
        The number of dimlets over which to calculate p-values.
    rng : RandomState
        Random state instance.
    n_repeats : int
        The number of repetitions to consider when evaluating null models.
    circular_stim : bool
        Indicates whether the stimulus is circular.
    all_stim : bool
        If True, all consecutive pairs of stimuli are used.

    Returns
    -------
    p_s_lfi, p_s_sdkl : ndarray (dimlets,)
        The p-values on the shuffled dimlets.
    p_r_lfi, p_r_sdkl : ndarray (dimlets,)
        The p-values on the rotated dimlets.
    v_lfi, v_sdkl : ndarray (dimlets,)
        The values of the LFI/sDKL on the original dimlet.
    """
    from mpi_utils.ndarray import Gatherv_rows

    n_samples, n_units = X.shape

    size = comm.size
    rank = comm.rank

    units, stims = generate_dimlets_and_stim_pairs(
        n_units=n_units, stimuli=stimuli, n_dim=n_dim, n_dimlets=n_dimlets,
        rng=rng, all_stim=all_stim, circular_stim=circular_stim
    )
    # allocate units and stims to the current rank
    units = np.array_split(units, size)[rank]
    stims = np.array_split(stims, size)[rank]

    # allocate storage for this rank's p-values
    my_dimlets = units.shape[0]
    v_s_lfi = np.zeros((my_dimlets, n_repeats))
    v_s_sdkl = np.zeros_like(v_s_lfi)
    v_r_lfi = np.zeros_like(v_s_lfi)
    v_r_sdkl = np.zeros_like(v_s_lfi)
    v_lfi = np.zeros(my_dimlets)
    v_sdkl = np.zeros(my_dimlets)

    for ii in range(my_dimlets):
        if rank == 0:
            print('Dimension %s' % n_dim, '{} out of {}'.format(ii + 1, my_dimlets))
        unit_idxs, stim_vals = units[ii], stims[ii]
        # calculate values under shuffle and rotation null models
        (v_s_lfi[ii], v_s_sdkl[ii],
         v_r_lfi[ii], v_r_sdkl[ii],
         v_lfi[ii], v_sdkl[ii]) = \
            inner_compare_nulls_measures(
                X=X, stimuli=stimuli, unit_idxs=unit_idxs, stim_vals=stim_vals,
                rng=rng, n_repeats=n_repeats, circular_stim=circular_stim)

    # gather p-values across ranks
    v_s_lfi = Gatherv_rows(v_s_lfi, comm)
    v_s_sdkl = Gatherv_rows(v_s_sdkl, comm)
    v_r_lfi = Gatherv_rows(v_r_lfi, comm)
    v_r_sdkl = Gatherv_rows(v_r_sdkl, comm)
    v_lfi = Gatherv_rows(v_lfi, comm)
    v_sdkl = Gatherv_rows(v_sdkl, comm)
    return v_s_lfi, v_s_sdkl, v_r_lfi, v_r_sdkl, v_lfi, v_sdkl


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
