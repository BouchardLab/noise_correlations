import numpy as np

from noise_correlations import analysis
from numpy.testing import assert_equal
from scipy.special import comb


def test_generate_dimlet():
    """Tests whether dimlets are correctly generated."""
    n_repeats = 100
    n_units = 10
    rng = np.random.RandomState(2332)

    # iterate over dimlet sizes
    for n_dim in [2, 3, 4, 5]:
        # multiple repetitions for randomness
        for rep in range(n_repeats):
            # generate dimlet, no circular stim
            unit_idxs = analysis.generate_dimlet(n_units, n_dim, rng)
            assert_equal(n_dim, unit_idxs.size)
            assert np.all(unit_idxs < n_units)


def test_generate_stim_pair():
    """Tests whether stimulus pairs are correctly generated."""
    # create stims
    n_repeats = 100
    stim_spacing = 25
    n_stims = 4
    unique_stims = stim_spacing * np.arange(n_stims)
    n_samples = 12
    stimuli = np.repeat(unique_stims, n_samples / n_stims)
    # random state
    rng = np.random.default_rng(2332)

    # multiple repetitions for randomness
    for rep in range(n_repeats):
        # check non-circular case
        stim_vals = analysis.generate_stim_pair(stimuli, rng, circular_stim=False)
        assert_equal(stim_spacing, np.ediff1d(stim_vals).item())
        # check circular stim
        stim_vals = analysis.generate_stim_pair(stimuli, rng, circular_stim=True)
        assert np.ediff1d(stim_vals).item() in [stim_spacing, stim_spacing * (n_stims - 1)]


def test_generate_dimlets_and_stim_pairs():
    """Tests whether the dimlets and stimulus pairs are collectively generated
    correctly."""
    n_units = 10
    n_dim = 5
    n_dimlets = 20
    # generate stimuli
    stim_spacing = 25
    n_stims = 4
    unique_stims = stim_spacing * np.arange(n_stims)
    n_samples = 12
    stimuli = np.repeat(unique_stims, n_samples / n_stims)
    # random state
    rng = np.random.default_rng(2332)

    # case 1: all_stim is True, no circular stim
    units, stims = analysis.generate_dimlets_and_stim_pairs(
        n_units=n_units, stimuli=stimuli, n_dim=n_dim, n_dimlets=n_dimlets,
        rng=rng, all_stim=True, circular_stim=False
    )
    assert units.shape[0] == stims.shape[0]
    assert units.shape[0] == n_dimlets * (n_stims - 1)
    # check that stims are neighboring pairs
    for stim_pair in stims:
        idx = np.argwhere(unique_stims == stim_pair[0]).item()
        assert unique_stims[idx + 1] == stim_pair[1]

    # case 2: all_stim is True, with circular stim
    units, stims = analysis.generate_dimlets_and_stim_pairs(
        n_units=n_units, stimuli=stimuli, n_dim=n_dim, n_dimlets=n_dimlets,
        rng=rng, all_stim=True, circular_stim=True
    )
    assert units.shape[0] == stims.shape[0]
    assert units.shape[0] == n_dimlets * n_stims
    # check that stims are neighboring pairs
    for stim_pair in stims:
        idx = np.argwhere(unique_stims == stim_pair[0]).item()
        if idx == 0:
            assert (
                (unique_stims[idx + 1] == stim_pair[1]) or (unique_stims[-1] == stim_pair[1])
            )
        else:
            assert unique_stims[idx + 1] == stim_pair[1]

    # case 3: all_stim is False, no circular stim
    units, stims = analysis.generate_dimlets_and_stim_pairs(
        n_units=n_units, stimuli=stimuli, n_dim=n_dim, n_dimlets=n_dimlets,
        rng=rng, all_stim=False, circular_stim=False
    )
    assert units.shape[0] == stims.shape[0]
    assert units.shape[0] == n_dimlets
    # check that stims are neighboring pairs
    for stim_pair in stims:
        idx = np.argwhere(unique_stims == stim_pair[0]).item()
        assert unique_stims[idx + 1] == stim_pair[1]

    # case 4: all_stim is False, with circular stim
    units, stims = analysis.generate_dimlets_and_stim_pairs(
        n_units=n_units, stimuli=stimuli, n_dim=n_dim, n_dimlets=n_dimlets,
        rng=rng, all_stim=False, circular_stim=True
    )
    assert units.shape[0] == stims.shape[0]
    assert units.shape[0] == n_dimlets
    # check that stims are neighboring pairs
    for stim_pair in stims:
        idx = np.argwhere(unique_stims == stim_pair[0]).item()
        if idx == 0:
            assert (
                (unique_stims[idx + 1] == stim_pair[1]) or (unique_stims[-1] == stim_pair[1])
            )
        else:
            assert unique_stims[idx + 1] == stim_pair[1]

    # check edge case where n_dimlets is too large
    n_dimlets = 5 * comb(n_units, n_dim)
    units, stims = analysis.generate_dimlets_and_stim_pairs(
        n_units=n_units, stimuli=stimuli, n_dim=n_dim, n_dimlets=n_dimlets,
        rng=rng, all_stim=True, circular_stim=False
    )
    assert units.shape[0] == stims.shape[0]
    assert units.shape[0] == comb(n_units, n_dim) * (n_stims - 1)

    # same as above, but circular stim is True
    n_dimlets = 5 * comb(n_units, n_dim)
    units, stims = analysis.generate_dimlets_and_stim_pairs(
        n_units=n_units, stimuli=stimuli, n_dim=n_dim, n_dimlets=n_dimlets,
        rng=rng, all_stim=True, circular_stim=True
    )
    assert units.shape[0] == stims.shape[0]
    assert units.shape[0] == comb(n_units, n_dim) * n_stims
