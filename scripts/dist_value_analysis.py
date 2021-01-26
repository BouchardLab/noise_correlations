"""Performs a comparison of discriminability metrics on neural data across
randomly chosen sub-populations of different sizes between a shuffle and
rotation null model."""
import argparse
import h5py
import neuropacks as packs
import numpy as np
import os
import time

from mpi4py import MPI
from mpi_utils.ndarray import Bcast_from_root
from noise_correlations.analysis import dist_calculate_nulls_measures
from noise_correlations import utils


def main(args):
    # Filepath arguments
    data_path = args.data_path
    save_folder = args.save_folder
    save_tag = args.save_tag
    dataset = args.dataset
    # Dimlet dimensions
    dim_max = args.dim_max
    dims = np.arange(2, dim_max + 1)
    n_dims = dims.size
    # Number of dimlets per dimension
    n_dimlets = args.n_dimlets
    # Number of repeats for each dim-stim
    n_repeats = args.n_repeats
    # Identify which neural population we consider
    which = args.which
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
        if rank == 0:
            pack = packs.PVC11(data_path=data_path)
            # Get design matrix and stimuli
            X = pack.get_response_matrix(transform=None)
            stimuli = pack.get_design_matrix(form='angle')
            non_responsive_stim = utils.get_nonresponsive_for_stim(X, stimuli)
            if non_responsive_stim.sum() > 0:
                X = np.delete(X, non_responsive_stim, axis=1)
            # Sub-sample the design matrix
            if which == 'tuned':
                selected_units = utils.get_tuned_units(
                    X=X, stimuli=stimuli,
                    aggregator=np.mean,
                    peak_response=args.peak_response,
                    tuning_criteria=args.tuning_criteria,
                    modulation=args.min_modulation,
                    modulation_frac=args.modulation_frac,
                    variance_to_mean=10.)
            elif which == 'responsive':
                selected_units = utils.get_responsive_units(
                    X=X, stimuli=stimuli,
                    aggregator=np.mean,
                    peak_response=args.peak_response,
                    variance_to_mean=10.)
            X = X[:, selected_units]

    # Feller lab data (Mouse RGCs, single-units, drifting gratings)
    elif dataset == 'ret2':
        circular_stim = True
        unordered = False
        if rank == 0:
            pack = packs.RET2(data_path=data_path)
            # get design matrix and stimuli
            X = pack.get_response_matrix(cells='all', response='max')
            stimuli = pack.get_design_matrix(form='angle')
            if which == 'tuned':
                selected_units = pack.tuned_cells
            elif which == 'responsive':
                selected_units = utils.get_responsive_units(
                    X=X, stimuli=stimuli,
                    aggregator=np.mean,
                    peak_response=args.peak_response,
                    variance_to_mean=10.)
            X = X[:, selected_units]

    # Bouchard lab data (Rat AC, muECOG, tone pips)
    elif dataset == 'ac1':
        circular_stim = False
        unordered = False
        if rank == 0:
            pack = packs.AC1(data_path=data_path)
            # get design matrix and stimuli
            amplitudes = np.arange(2, 8)
            X = pack.get_response_matrix(amplitudes=amplitudes)
            stimuli = pack.get_design_matrix(amplitudes=amplitudes)
            if which == 'tuned':
                selected_units = utils.get_tuned_units(
                    X=X, stimuli=stimuli,
                    aggregator=np.mean,
                    peak_response=args.peak_response,
                    tuning_criteria=args.tuning_criteria,
                    modulation=args.min_modulation,
                    modulation_frac=args.modulation_frac,
                    variance_to_mean=10.)
            elif which == 'responsive':
                selected_units = utils.get_responsive_units(
                    X=X, stimuli=stimuli,
                    aggregator=np.mean,
                    peak_response=args.peak_response,
                    variance_to_mean=10.)
            X = X[:, selected_units]

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

    if rank == 0:
        # Observed measures in neural data
        v_lfi = np.zeros((n_dims, n_dim_stims))
        v_sdkl = np.zeros_like(v_lfi)
        # Values of measures across shuffles/rotations
        v_s_lfi = np.zeros((n_dims, n_dim_stims, n_repeats))
        v_s_sdkl = np.zeros_like(v_s_lfi)
        v_r_lfi = np.zeros_like(v_s_lfi)
        v_r_sdkl = np.zeros_like(v_s_lfi)
        # Dim-stim storage
        units = np.zeros((n_dims, n_dim_stims, np.max(dims)))
        stims = np.zeros((n_dims, n_dim_stims, 2))

    for idx, n_dim in enumerate(dims):
        if rank == 0:
            t1 = time.time()
            print(f'>>> Dimension {n_dim}')
        # evaluate p-values using MPI
        (v_s_lfi_temp, v_s_sdkl_temp,
         v_r_lfi_temp, v_r_sdkl_temp,
         v_lfi_temp, v_sdkl_temp,
         units_temp, stims_temp) = dist_calculate_nulls_measures(
            X=X, stimuli=stimuli, n_dim=n_dim, n_dimlets=n_dimlets, rng=rng,
            comm=comm, n_repeats=n_repeats, circular_stim=circular_stim,
            all_stim=all_stim, unordered=unordered,
            n_stims_per_dimlet=args.n_stims_per_dimlet,
            verbose=args.inner_loop_verbose)

        if rank == 0:
            v_lfi[idx] = v_lfi_temp
            v_sdkl[idx] = v_sdkl_temp
            v_s_lfi[idx] = v_s_lfi_temp
            v_s_sdkl[idx] = v_s_sdkl_temp
            v_r_lfi[idx] = v_r_lfi_temp
            v_r_sdkl[idx] = v_r_sdkl_temp
            units[idx, :, :n_dim] = units_temp
            stims[idx] = stims_temp

            print(f'Loop took {time.time() - t1} seconds.')
            if idx < dims.size - 1:
                print('')

    if rank == 0:
        print('==============================================================')
        print('>>> Saving data...')
        # Creating results filename
        save_name = f'values_{dataset}_{dim_max}_{n_dimlets}_{n_repeats}.h5'
        if save_tag != '':
            save_name = save_tag + '_' + save_name
        save_name = os.path.join(save_folder, save_name)
        # Store datasets
        results = h5py.File(save_name, 'w')
        results['v_lfi'] = v_lfi
        results['v_sdkl'] = v_sdkl
        results['v_s_lfi'] = v_s_lfi
        results['v_s_sdkl'] = v_s_sdkl
        results['v_r_lfi'] = v_r_lfi
        results['v_r_sdkl'] = v_r_sdkl
        results['units'] = units
        results['stims'] = stims
        results.close()
        print('Successfully Saved.')
        print('Job complete. Total time:', time.time() - t0)
        print('--------------------------------------------------------------')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run noise correlations analysis.')
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--save_folder', default='', type=str)
    parser.add_argument('--save_tag', type=str, default='')
    parser.add_argument('--dataset', choices=['pvc11', 'ret2', 'ac1', 'cv'])
    parser.add_argument('--dim_max', type=int)
    parser.add_argument('--n_dimlets', '-n', type=int, default=1000)
    parser.add_argument('--n_repeats', '-s', type=int, default=1000)
    parser.add_argument('--which', choices=['tuned', 'responsive', 'all'])
    parser.add_argument('--tuning_criteria', default='modulation_frac')
    parser.add_argument('--peak_response', type=float, default=10.)
    parser.add_argument('--min_modulation', type=float, default=2.)
    parser.add_argument('--modulation_frac', type=float, default=0.5)
    parser.add_argument('--random_seed', '-rs', type=int, default=0)
    parser.add_argument('--limit_stim', action='store_false')
    parser.add_argument('--inner_loop_verbose', action='store_true')
    parser.add_argument('--cv_response', type=str, default='cv')
    parser.add_argument('--n_stims_per_dimlet', type=int, default=1)
    args = parser.parse_args()

    main(args)
