import argparse
import h5py
import numpy as np

from noise_correlations import analysis, utils


# Command line arguments
parser = argparse.ArgumentParser(description='Calculate optimality statistics.')
parser.add_argument('--fits_path', type=str)
parser.add_argument('--verbose', action='store_true')
args = parser.parse_args()

with h5py.File(args.fits_path, 'a') as results:
    # Create groups for optimal orientations and covariances
    if 'opt_covariances' in results:
        covs_group = results['opt_covariances']
    else:
        covs_group = results.create_group('opt_covariances')
    if 'opt_rotations' in results:
        rots_group = results['opt_rotations']
    else:
        rots_group = results.create_group('opt_rotations')

    X = results['X'][:]
    stimuli = results['stimuli'][:]
    units = results['units'][:]
    stims = results['stims'][:]
    v_lfi = results['v_lfi'][:]
    dim_idxs = np.arange(results['units'].shape[0])

    n_pairings = v_lfi.shape[1]

    # Calculate optimal rotations and covariances
    for idx, dim_idx in enumerate(dim_idxs):
        n_dim = dim_idx + 2
        optimal_cov = np.zeros((n_pairings, n_dim, n_dim))
        optimal_R = np.zeros_like(optimal_cov)
        # Iterate over dimlet-stim pairings
        for pairing in range(n_pairings):
            # Extract data for the dimlet-stim pairing
            stim_pairing = stims[dim_idx, pairing]
            dimlet = units[dim_idx, pairing][:dim_idx + 2].astype('int')
            stim0_idx = np.argwhere(stimuli == stim_pairing[0]).ravel()
            stim1_idx = np.argwhere(stimuli == stim_pairing[1]).ravel()
            X0 = X[stim0_idx][:, dimlet]
            X1 = X[stim1_idx][:, dimlet]
            # Get means and covariances
            mu0, cov0 = utils.mean_cov(X0)
            mu1, cov1 = utils.mean_cov(X1)
            # Differential correlation direction
            fpr = mu1 - mu0
            fpr /= np.linalg.norm(fpr)
            # Average covariance is used by LFI
            avg_cov = (cov0 + cov1) / 2.
            # Get smallest eigenvector from covariance
            w_small = np.linalg.eigh(avg_cov)[1][:, 0]
            # Get rotation matrix that brings the smallest eigenvector to the
            # optimal orientation
            R = analysis.get_rotation_for_vectors(w_small, fpr)
            optimal_cov[pairing] = R @ avg_cov @ R.T
            optimal_R[pairing] = R
        key = f"{idx}"
        if key in covs_group:
            covs_group[key][...] = optimal_cov
        else:
            covs_group[key] = optimal_cov
        if key in rots_group:
            rots_group[key][...] = optimal_cov
        else:
            rots_group[f"{idx}"] = optimal_R
