import numpy as np

from noise_correlations import analysis
from numpy.testing import assert_equal


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
    rng = np.random.RandomState(2332)

    # multiple repetitions for randomness
    for rep in range(n_repeats):
        # check non-circular case
        stim_vals = analysis.generate_stim_pair(stimuli, rng, circular_stim=False)
        assert_equal(stim_spacing, np.ediff1d(stim_vals).item())
        # check circular stim
        stim_vals = analysis.generate_stim_pair(stimuli, rng, circular_stim=True)
        assert np.ediff1d(stim_vals).item() in [stim_spacing, stim_spacing * (n_stims - 1)]
