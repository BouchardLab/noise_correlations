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
from noise_correlations.analysis import dist_compare_nulls_measures
from noise_correlations import utils


def main(args):
    # process command line arguments
    # filepath arguments
    data_path = args.data_path
    save_folder = args.save_folder
    dataset = args.dataset
    # the dimensions we consider
    dim_max = args.dim_max
    dims = np.arange(2, dim_max + 1)
    n_unique_dims = dims.size
    # number of dimlets per dimension
    n_dimlets = args.n_dimlets
    # number of repeats for each dimlet/stim pairing
    n_repeats = args.n_repeats
    # identify which type of neural population we consider
    which_neurons = args.which_neurons
    all_stim = args.limit_stim

    # create random state
    rng = np.random.RandomState(args.random_state)

    comm = MPI.COMM_WORLD
    size = comm.size
    rank = comm.rank

    if rank == 0:
        t0 = time.time()
        print('--------------------------------------------------------------')
        print('%s processes running, this is rank %s.' % (size, rank))
        print('Running on dataset %s, up to %s dimensions.' % (dataset, dim_max))
        print('Using %s dimlets per dimension, and %s repeats per dimlet.'
              % (n_dimlets, n_repeats))
        print('--------------------------------------------------------------')

    # obtain neural design matrix and broadcast to all ranks
    X = None
    stimuli = None
    # Kohn lab data (Monkey V1, single-units, drifting gratings)
    if dataset == 'pvc11':
        circular_stim = True
        if rank == 0:
            pvc11 = packs.PVC11(data_path=data_path)
            # get design matrix and stimuli
            X = pvc11.get_response_matrix(transform=None)
            stimuli = pvc11.get_design_matrix(form='angle')
            if which_neurons == 'tuned':
                tuned_units = utils.get_tuned_units(X, stimuli,
                                                    peak_response=args.peak_response,
                                                    min_modulation=args.min_modulation)
                X = X[:, tuned_units]
    # Feller lab data (Mouse RGCs, single-units, drifting gratings)
    elif dataset == 'ret2':
        circular_stim = True
        if rank == 0:
            ret2 = packs.RET2(data_path=data_path)
            # get design matrix and stimuli
            X = ret2.get_response_matrix(cells='all', response='max')
            stimuli = ret2.get_design_matrix(form='angle')
    # Bouchard lab data (Rat AC, muECOG, tone pips)
    elif dataset == 'ac_ecog':
        circular_stim = False
        if rank == 0:
            with h5py.File(data_path, 'r') as f:
                resp = f['final_rsp'][:]
                X = np.transpose(resp[..., 5], axes=(0, 2, 1))
                trial_medians = np.median(X, axis=-1)
                keep = (trial_medians.max(axis=1) - trial_medians.min(axis=1)) >= 5.
                X = X[keep]
    else:
        raise ValueError('Dataset not available.')

    if rank == 0:
        print('Loaded dataset %s' % dataset)
    X = Bcast_from_root(X, comm)
    stimuli = Bcast_from_root(stimuli, comm)
    if rank == 0:
        print('Broadcasted dataset %s' % dataset)

    # determine size of p-value array
    if all_stim and circular_stim:
        n_dimlet_stim_combs = n_dimlets * np.unique(stimuli).size
    elif all_stim and not circular_stim:
        n_dimlet_stim_combs = n_dimlets * (np.unique(stimuli).size - 1)
    else:
        n_dimlet_stim_combs = n_dimlets

    p_s_lfi = np.zeros((n_unique_dims, n_dimlet_stim_combs))
    p_s_sdkl = np.zeros((n_unique_dims, n_dimlet_stim_combs))
    p_r_lfi = np.zeros((n_unique_dims, n_dimlet_stim_combs))
    p_r_sdkl = np.zeros((n_unique_dims, n_dimlet_stim_combs))
    v_lfi = np.zeros((n_unique_dims, n_dimlet_stim_combs))
    v_sdkl = np.zeros((n_unique_dims, n_dimlet_stim_combs))

    # calculate p-values for many dimlets at difference dimensions
    for idx, n_dim in enumerate(dims):
        if rank == 0:
            print('=== Dimension %s ===' % n_dim)
        # evaluate p-values using MPI
        (p_s_lfi[idx], p_s_sdkl[idx],
         p_r_lfi[idx], p_r_sdkl[idx],
         v_lfi[idx], v_sdkl[idx]) = dist_compare_nulls_measures(
            X=X, stimuli=stimuli, n_dim=n_dim, n_dimlets=n_dimlets, rng=rng,
            comm=comm, n_repeats=n_repeats, circular_stim=circular_stim,
            all_stim=all_stim)
        if rank == 0:
            t1 = time.time()
            print('Loop took %s seconds.\n' % (t1 - t0))

    # save data in root
    if rank == 0:
        print('Pre-save.')
        save_name = '{}_{}_{}_{}.npz'.format(dataset, dim_max, n_dimlets, n_repeats)
        save_name = os.path.join(save_folder, save_name)
        np.savez(save_name,
                 p_s_lfi=p_s_lfi, p_s_sdkl=p_s_sdkl,
                 p_r_lfi=p_r_lfi, p_r_sdkl=p_r_sdkl,
                 v_lfi=v_lfi, v_sdkl=v_sdkl)
        print('Successfully Saved.')
        t2 = time.time()
        print('Job complete. Total time: ', t2 - t0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run noise correlations analysis.')
    parser.add_argument('--data_path', type=str,
                        help='Path to where the dataset is stored.')
    parser.add_argument('--save_folder', type=str,
                        help='Folder where results will be saved.')
    parser.add_argument('--dataset', choices=['pvc11', 'maxd'],
                        help='Which dataset to run analysis on.')
    parser.add_argument('--dim_max', type=int,
                        help='The maximum number of units to operate on.')
    parser.add_argument('--n_dimlets', '-n', type=int, default=1000,
                        help='How many dimlets to consider.')
    parser.add_argument('--n_repeats', '-s', type=int, default=10000,
                        help='How many repetitions to perform when evaluating'
                             'p-values.')
    parser.add_argument('--which_neurons', '-which', default='tuned',
                        choices=['tuned', 'responsive', 'all'])
    parser.add_argument('--peak_response', type=float, default=10.)
    parser.add_argument('--min_modulation', type=float, default=2.)
    parser.add_argument('--random_state', '-rs', type=int, default=0,
                        help='Random state seed.')
    parser.add_argument('--limit_stim', action='store_false',
                        help='Do not use all pairwise stimuli for each dimlet.')
    args = parser.parse_args()

    main(args)
