import argparse
import h5py
import neuropacks as packs
import numpy as np
import os

from mpi4py import MPI
from mpi_utils.ndarray import Bcast_from_root
from noise_correlations.analysis import dist_compare_nulls_measures


def main(args):
    # process command line arguments
    data_path = args.data_path
    save_folder = args.save_folder
    dataset = args.dataset
    n_dim = args.n_dim
    n_dimlets = args.n_dimlets
    n_repeats = args.n_repeats
    rng = np.random.RandomState(args.random_state)
    limit_stim = args.limit_stim

    comm = MPI.COMM_WORLD
    size = comm.size
    rank = comm.rank

    if rank == 0:
        print(size, rank)
        print(dataset, n_dim)

    X = None
    if dataset == 'pvc11':
        circular_stim = True
        if rank == 0:
            pvc11 = packs.PVC11(data_path=data_path)
            # get design matrix and stimuli
            X = pvc11.get_response_matrix(transform=None)
            stimuli = pvc11.get_design_matrix(form='angle')
            # TODO: keep criteria
    elif dataset == 'maxd':
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
        print(dataset, n_dim, 'load')
    X = Bcast_from_root(X, comm)
    if rank == 0:
        print(dataset, n_dim, 'bcast')

    # evaluate p-values using MPI
    (p_s_lfi, p_s_sdkl,
     p_r_lfi, p_r_sdkl,
     v_lfi, v_sdkl) = dist_compare_nulls_measures(
        X=X, stimuli=stimuli, n_dim=n_dim, n_dimlets=n_dimlets, rng=rng,
        comm=comm, n_repeats=n_repeats, circular_stim=circular_stim,
        all_stim=limit_stim)

    if rank == 0:
        print(dataset, n_dim, 'presave')
        save_name = '{}_{}_{}_{}.npz'.format(dataset, n_dim, n_dimlets, n_repeats)
        save_name = os.path.join(save_folder, save_name)
        np.savez(save_name,
                 p_s_lfi=p_s_lfi, p_s_sdkl=p_s_sdkl,
                 p_r_lfi=p_r_lfi, p_r_sdkl=p_r_sdkl,
                 v_lfi=v_lfi, v_sdkl=v_sdkl)
        print(dataset, n_dim, 'done')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run noise correlations analysis.')
    parser.add_argument('data_path', type=str,
                        help='Path to where the dataset is stored.')
    parser.add_argument('save_folder', type=str,
                        help='Folder where results will be saved.')
    parser.add_argument('dataset', choices=['pvc11', 'maxd'],
                        help='Which dataset to run analysis on.')
    parser.add_argument('n_dim', type=int,
                        help='How many units to operate on.')
    parser.add_argument('--n_dimlets', '-n', type=int, default=1000,
                        help='How many dimlets to consider.')
    parser.add_argument('--n_repeats', '-s', type=int, default=10000,
                        help='How many repetitions to perform when evaluating'
                             'p-values.')
    parser.add_argument('--random_state', '-rs', type=int, default=0,
                        help='Random state seed.')
    parser.add_argument('--limit_stim', action='store_false',
                        help='Do not use all pairwise stimuli for each dimlet.')
    args = parser.parse_args()

    main(args)
