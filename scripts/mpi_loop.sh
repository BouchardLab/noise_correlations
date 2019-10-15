#!/bin/bash -l
#SBATCH --qos=premium
#SBATCH --constraint=knl
#SBATCH --nodes=48
#SBATCH --time=05:00:00

cores=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
export HDF5_USE_FILE_LOCKING=FALSE
export KMP_AFFINITY=disabled

source $HOME/anaconda3/bin/activate
source activate noise_correlations
fold=/project/projectdirs/m2043/jlivezey/noise_correlations

for dim in {5..10}; do
  srun -N 8 -n 544 -c $cores python -u dist_analysis.py $fold $fold maxd $dim &
done
wait
