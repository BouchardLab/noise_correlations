import argparse, os, glob
import numpy as np
import matplotlib.pyplot as plt

from mpi4py import MPI

from noise_correlations.data import datasets
from noise_correlations import discriminability, null_models
from noise_correlations.discriminability import lfi_data, lda_data, corrected_lfi_data, mv_normal_jeffreys_data
from noise_correlations.null_models import random_rotation_data, shuffle_data
from noise_correlations import plot
from noise_correlations.analysis import dist_compare_nulls_measures

from mpi_utils.ndarray import Bcast_from_root


parser = argparse.ArgumentParser(description='Run noise correlations analysis.')
parser.add_argument('folder', type=str,
                    help='Base folder where all datasets are stored.')
parser.add_argument('save_folder', type=str,
                    help='Folder where results will be saved.')
parser.add_argument('dataset', choices=['kohn'],
                    help='Which dataset to run analysis on.')
parser.add_argument('dim', type=int,
                    help='How many units to operate on.')
parser.add_argument('--n_dimlets', '-n', type=int, default=10000,
                    help='How many dimlets to consider.')
parser.add_argument('--n_samples', '-s', type=int, default=10000,
                    help='How many dimlets to consider.')
args = parser.parse_args()
folder = args.folder
save_folder = save_args.folder
dataset = args.dataset
dim = args.dim
n_dimlets = args.n_dimlets
n_samples = args.n_samples

rng = np.random.RandomState(0)
comm = MPI.COMM_WORLD
size = comm.size
rank = comm.rank

X = None
if dataset == 'kohn':
    circular_stim = True
    if rank == 0:
        path = os.path.join(folder, 'kohn_pvc-11')
        ds = datasets.KohnCRCNSpvc11_monkey(path)
        X = ds.data_tensor()
X = Bcast_from_root(X, comm)


(p_s_lfi, p_s_sdkl,
 p_r_lfi, p_r_sdkl,
 v_lfi, v_sdkl) = dist_compare_nulls_measures(X, dim, n_dimlets, rng,
                                              comm, n_samples=n_samples,
                                              circular_stim=circular_stim)
if rank == 0:
    save_name = '{}_{}_{}.npz'.format(dim, n_dimlets, n_samples)
    save_name = os.path.join(save_folder, save_name)
    np.savez(save_name,
             p_s_lfi=p_s_lfi, p_s_sdkl=p_s_sdkl,
             p_r_lfi=p_r_lfi, p_r_sdkl=p_r_sdkl,
             v_lfi=v_lfi, v_sdkl=v_sdkl)
