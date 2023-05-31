"""Performs a comparison of discriminability metrics on neural data across
randomly chosen sub-populations of different sizes between a shuffle and
rotation null model."""
import argparse
import h5py
import neuropacks as packs
import numpy as np
import os
import time
from pathlib import Path

from mpi4py import MPI
from mpi_utils.ndarray import Bcast_from_root
from noise_correlations.analysis import dist_calculate_nulls_measures_w_rotations
from noise_correlations import utils


def main(args):
    # Path to the neural dataset
    data_path = args.data_path
    # Path to rotation dataset
    rotation_path = args.rotation_path
    # Path to correlation dataset
    correlation_path = args.correlation_path
    # Folder in which to save the values
    save_folder = args.save_folder
    # Tag to indicate the file in which to save the values
    save_tag = args.save_tag
    # Which neural dataset to use
    dataset = args.dataset
    # Dimlet dimensions
    dim_min = args.dim_min
    dim_max = args.dim_max
    dims = np.arange(dim_min, dim_max + 1)
    n_dims = dims.size
    # Number of dimlets per dimension
    n_dimlets = args.n_dimlets
    # Number of repeats for each dim-stim
    n_repeats = args.n_repeats
    # Whether all stim pairs are used for each dimlet
    all_stim = args.limit_stim
    # Create random state
    rng = np.random.default_rng(args.random_seed)

    # MPI communicator
    comm = MPI.COMM_WORLD
    size = comm.size
    rank = comm.rank

    if rank == 0:
        t0 = time.time()
        print('--------------------------------------------------------------')
        print('>>> Beginning Job...')
        print(f'{size} processes running, this is rank {rank}.')
        print(f'Running on dataset {dataset}, up to {dim_max} dimensions.')
        print(f'Using {n_dimlets} dimlets per dimension, and {n_repeats} repeats per dimlet.')
        print('--------------------------------------------------------------')

    # Obtain neural design matrix and broadcast to all ranks
    X = None
    stimuli = None

    # Kohn lab data (Monkey V1, single-units, drifting gratings)
    if dataset == 'pvc11':
        circular_stim = True
        unordered = False
        stim_transform = None
        if rank == 0:
            pack = packs.PVC11(data_path=data_path)
            # Get design matrix and stimuli
            X = pack.get_response_matrix(transform=args.transform)
            stimuli = pack.get_design_matrix(form='angle')
            non_responsive_stim = utils.get_nonresponsive_for_stim(X, stimuli)
            if non_responsive_stim.sum() > 0:
                X = np.delete(X, non_responsive_stim, axis=1)
            selected_units = utils.get_tuned_units(
                X=X, stimuli=stimuli,
                aggregator=np.mean,
                peak_response=args.peak_response,
                tuning_criteria=args.tuning_criteria,
                modulation=args.min_modulation,
                modulation_frac=args.modulation_frac,
                variance_to_mean=10.)
            X = X[:, selected_units]
    # Feller lab data (Mouse RGCs, single-units, drifting gratings)
    elif dataset == 'ret2':
        circular_stim = True
        unordered = False
        stim_transform = None
        if rank == 0:
            pack = packs.RET2(data_path=data_path)
            # get design matrix and stimuli
            X = pack.get_response_matrix(cells='all', response='max')
            stimuli = pack.get_design_matrix(form='angle')
            X = X[:, pack.tuned_cells]
    # Chang lab data (Human vSMC, ECoG, CV syllables)
    elif dataset == 'cv':
        circular_stim = False
        unordered = True
        all_stim = False
        if rank == 0:
            pack = packs.CV(data_path=data_path)
            X = pack.get_response_matrix(stimulus=args.cv_response)
            stimuli = pack.get_design_matrix(stimulus=args.cv_response)
            unique_stimuli = np.unique(stimuli)
            stimuli = np.array([np.argwhere(unique_stimuli == stim).item()
                               for stim in stimuli])
    elif dataset == 'ecog':
        circular_stim = False
        unordered = False
        all_stim = False
        stim_transform = 'log'
        if rank == 0:
            pack = packs.ECOG(data_path=data_path)
            X = pack.get_response_matrix(bounds=[40, 60],
                                         band='HG',
                                         amps=[4, 5, 6],
                                         electrodes=pack.pac_idxs)
            stimuli = pack.get_design_matrix(form='frequency')
            amplitudes = pack.get_design_matrix(form='amplitude')
            stimuli = stimuli[np.isin(amplitudes, [5, 6, 7])]
            unique_stimuli = np.unique(stimuli)
    # Ji lab dataset, drifting gratings, mouse v1, calcium imaging
    elif dataset == 'v1':
        circular_stim = True
        unordered = False
        stim_transform = None
        p = Path(data_path)
        sessions = [s.stem for s in p.iterdir() if s.is_dir()]
        data_path, depth = os.path.split(data_path)
        data_path, date_anim = os.path.split(data_path)
        date = date_anim.split('_')[0]
        files = []
        for sess in sessions:
            fname = f'ROI_{date}AllTif_RM_{sess}_Intensity_unweighted_s2p_NpMethod1_Coe0.75_Exclusion_NpSize30.mat'
            files.append(os.path.join(data_path, date_anim, depth, sess, fname))
        if rank == 0:
            pack = packs.V1(data_path=files)
            # get design matrix and stimuli
            X = pack.get_response_matrix(cells='all', response='max')
            stimuli = pack.get_design_matrix(form='angle')
            selected_units = utils.get_tuned_units(
                X=X, stimuli=stimuli,
                aggregator=np.mean,
                peak_response=30,
                tuning_criteria=args.tuning_criteria,
                modulation=args.min_modulation,
                modulation_frac=args.modulation_frac,
                variance_to_mean=10.)
            X = X[:, selected_units]
    else:
        raise ValueError('Dataset not available.')

    if rank == 0:
        print('>>> Loading data...')
        print(f'Loaded dataset {dataset}.')
        print(f'Dataset has {X.shape[1]} units and {X.shape[0]} samples.')
        print(f'Dataset has {np.unique(stimuli).size} unique stimuli.')
        if unordered:
            print(
                f'Dataset is unordered. Using {args.n_stims_per_dimlet} stims per dimlet.'
            )
        if circular_stim:
            print('Dataset has circular stim.')

    # Broadcast design matrix and stimuli
    X = Bcast_from_root(X, comm)
    stimuli = Bcast_from_root(stimuli, comm)
    if rank == 0:
        print(f'Broadcasted dataset {dataset}.')
        print('==============================================================')
        print('>>> Performing experiment...\n')

    # Obtain storage arrays
    if all_stim and circular_stim:
        n_dim_stims = n_dimlets * np.unique(stimuli).size
    elif all_stim and not circular_stim:
        n_dim_stims = n_dimlets * (np.unique(stimuli).size - 1)
    elif unordered:
        n_dim_stims = n_dimlets * args.n_stims_per_dimlet
    else:
        n_dim_stims = n_dimlets

    R_idxs = None
    corr_idxs = None
    if rank == 0:
        R_idxs = None
        corr_idxs = None
        # Get rotations for this dimension
        with h5py.File(rotation_path, 'r') as rotations:
            max_n_Rs = rotations['2'].shape[0]
            R_idxs = rng.integers(
                low=0,
                high=max_n_Rs,
                size=(n_dims, n_dim_stims, n_repeats, 2))
        with h5py.File(correlation_path, 'r') as correlations:
            max_n_corrs = correlations['2'].shape[0]
            corr_idxs = rng.integers(
                low=0,
                high=max_n_corrs,
                size=(n_dims, n_dim_stims, n_repeats, 2))
        # Creating results filename
        save_name = f'{dataset}_{dim_max}_{n_dimlets}_{n_repeats}.h5'
        if save_tag != '':
            save_name = save_tag + '_' + save_name
        save_name = os.path.join(save_folder, save_name)
        with h5py.File(save_name, 'w') as results:
            opt_covs_group = results.create_group('opt_covs')
            opt_u_covs_group = results.create_group('opt_u_covs')
            opt_fa_covs_group = results.create_group('opt_fa_covs')
            # Observed measures in neural data
            results['v_lfi'] = np.zeros((n_dims, n_dim_stims))
            # Values of measures across shuffles/rotations
            results['v_s_lfi'] = np.zeros((n_dims, n_dim_stims, n_repeats))
            results['v_u_lfi'] = np.zeros((n_dims, n_dim_stims, n_repeats))
            results['v_r_lfi'] = np.zeros((n_dims, n_dim_stims, n_repeats))
            results['v_fa_lfi'] = np.zeros((n_dims, n_dim_stims, n_repeats))
            results['v_fa_fit_lfi'] = np.zeros((n_dims, n_dim_stims))
            # Percentiles of measures
            results['p_s_lfi'] = np.zeros((n_dims, n_dim_stims))
            results['p_u_lfi'] = np.zeros((n_dims, n_dim_stims))
            results['p_r_lfi'] = np.zeros((n_dims, n_dim_stims))
            results['p_fa_lfi'] = np.zeros((n_dims, n_dim_stims))
            # Rotation indices
            results['R_idxs'] = R_idxs
            results['corr_idxs'] = corr_idxs
            # Dim-stim storage
            results['fa_ks'] = np.zeros((n_dims, n_dim_stims, 3), dtype=int)
            results['units'] = np.zeros((n_dims, n_dim_stims, np.max(dims)), dtype=int)
            results['stims'] = np.zeros((n_dims, n_dim_stims, 2))
    R_idxs = Bcast_from_root(R_idxs, comm)
    corr_idxs = Bcast_from_root(corr_idxs, comm)

    for idx, n_dim in enumerate(dims):
        if rank == 0:
            print(f'>>> Dimension {n_dim}')
            t1 = time.time()
        # Perform distributed evaluation across null measures
        (v_lfi_temp, v_sdkl_temp,
         v_s_lfi_temp, v_s_sdkl_temp,
         v_u_lfi_temp, v_u_sdkl_temp,
         v_r_lfi_temp, v_r_sdkl_temp,
         v_fa_lfi_temp, v_fa_sdkl_temp,
         v_fa_fit_lfi_temp, v_fa_fit_sdkl_temp,
         opt_covs_temp, opt_u_covs_temp, opt_fa_covs_temp,
         stims_temp, units_temp,
         fa_ks_temp) = dist_calculate_nulls_measures_w_rotations(
            X=X,
            stimuli=stimuli,
            n_dim=n_dim,
            n_dimlets=n_dimlets,
            Rs=rotation_path,
            R_idxs=R_idxs[idx],
            corrs=correlation_path,
            corr_idxs=corr_idxs[idx],
            rng=rng,
            comm=comm,
            circular_stim=circular_stim,
            all_stim=all_stim,
            unordered=unordered,
            n_stims_per_dimlet=args.n_stims_per_dimlet,
            verbose=args.inner_loop_verbose,
            stim_transform=stim_transform,
            k=None)

        if rank == 0:
            with h5py.File(save_name, 'a') as results:
                results['v_lfi'][idx] = v_lfi_temp
                results['v_s_lfi'][idx] = v_s_lfi_temp
                results['v_u_lfi'][idx] = v_u_lfi_temp
                results['v_r_lfi'][idx] = v_r_lfi_temp
                results['v_fa_lfi'][idx] = v_fa_lfi_temp
                results['v_fa_fit_lfi'][idx] = v_fa_fit_lfi_temp
                results['p_s_lfi'][idx] = np.mean(
                    v_lfi_temp[..., np.newaxis] > v_s_lfi_temp, axis=-1
                )
                results['p_u_lfi'][idx] = np.mean(
                    v_lfi_temp[..., np.newaxis] > v_u_lfi_temp, axis=-1
                )
                results['p_r_lfi'][idx] = np.mean(
                    v_lfi_temp[..., np.newaxis] > v_r_lfi_temp, axis=-1
                )
                results['p_fa_lfi'][idx] = np.mean(
                    v_lfi_temp[..., np.newaxis] > v_fa_lfi_temp, axis=-1
                )
                results['units'][idx, :, :n_dim] = units_temp
                results['stims'][idx] = stims_temp
                results['fa_ks'][idx] = fa_ks_temp
                opt_covs_group = results['opt_covs']
                opt_covs_group[str(n_dim)] = opt_covs_temp
                opt_u_covs_group = results['opt_u_covs']
                opt_u_covs_group[str(n_dim)] = opt_u_covs_temp
                opt_fa_covs_group = results['opt_fa_covs']
                opt_fa_covs_group[str(n_dim)] = opt_fa_covs_temp

            print(f'Loop took {time.time() - t1} seconds.')

    if rank == 0:
        print('==============================================================')
        print('>>> Saving additional data...')
        # Store datasets
        results = h5py.File(save_name, 'a')
        results['X'] = X
        results['stimuli'] = stimuli
        results.close()
        print('Successfully Saved.')
        print('Job complete. Total time:', time.time() - t0)
        print('--------------------------------------------------------------')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run noise correlations analysis.')
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--rotation_path', type=str)
    parser.add_argument('--correlation_path', type=str)
    parser.add_argument('--save_folder', default='', type=str)
    parser.add_argument('--save_tag', type=str, default='')
    parser.add_argument('--dataset', choices=['pvc11', 'ret2', 'ac1', 'cv', 'ecog', 'v1'])
    parser.add_argument('--dim_min', type=int, default=2)
    parser.add_argument('--dim_max', type=int)
    parser.add_argument('--n_dimlets', '-n', type=int, default=1000)
    parser.add_argument('--n_repeats', '-s', type=int, default=1000)
    parser.add_argument('--tuning_criteria', default='modulation_frac')
    parser.add_argument('--peak_response', type=float, default=10.)
    parser.add_argument('--min_modulation', type=float, default=2.)
    parser.add_argument('--modulation_frac', type=float, default=0.5)
    parser.add_argument('--random_seed', '-rs', type=int, default=0)
    parser.add_argument('--limit_stim', action='store_false')
    parser.add_argument('--transform')
    parser.add_argument('--inner_loop_verbose', action='store_true')
    parser.add_argument('--cv_response', type=str, default='cv')
    parser.add_argument('--n_stims_per_dimlet', type=int, default=1)
    args = parser.parse_args()

    main(args)
