import argparse, os, glob, h5py
import numpy as np

from mpi4py import MPI

from noise_correlations.data import datasets
from noise_correlations import discriminability, null_models
from noise_correlations.discriminability import lfi_data, lda_data, corrected_lfi_data, mv_normal_jeffreys_data
from noise_correlations.null_models import random_rotation_data, shuffle_data
from noise_correlations.analysis import dist_synthetic_data

from mpi_utils.ndarray import Bcast_from_root


parser = argparse.ArgumentParser(description='Run noise correlations analysis.')
parser.add_argument('save_folder', type=str,
                    help='Folder where results will be saved.')
parser.add_argument('dim', type=int,
                    help='How many units to operate on.')
parser.add_argument('--n_deltas', '-d', type=int, default=10,
                    help='How many deltas to consider.')
parser.add_argument('--n_rotations', '-r', type=int, default=10,
                    help='How many rotations to consider.')
parser.add_argument('--n_samples', '-s', type=int, default=10000,
                    help='How many dimlets to consider.')
args = parser.parse_args()

save_folder = args.save_folder
dim = args.dim
n_deltas = args.n_deltas
n_rotations = args.n_rotations
n_samples = args.n_samples

rng = np.random.RandomState(0)
comm = MPI.COMM_WORLD
size = comm.size
rank = comm.rank

if rank == 0:
    print(size, rank, dim)


(p_s_lfi, p_s_sdkl,
 p_r_lfi, p_r_sdkl,
 v_lfi, v_sdkl) = dist_synthetic_data(dim, n_deltas, n_rotations, rng, comm,
                                      n_samples=n_samples)

if rank == 0:
    print(dataset, dim, 'presave')
    save_name = 'synthetic_{}_{}_{}.npz'.format(dim, n_deltas, n_rotations)
    save_name = os.path.join(save_folder, save_name)
    np.savez(save_name,
             p_s_lfi=p_s_lfi, p_s_sdkl=p_s_sdkl,
             p_r_lfi=p_r_lfi, p_r_sdkl=p_r_sdkl,
             v_lfi=v_lfi, v_sdkl=v_sdkl)
    print(dim, 'done')
