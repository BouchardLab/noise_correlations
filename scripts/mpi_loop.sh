#!/bin/bash -l
#SBATCH --qos=premium
#SBATCH --constraint=knl
#SBATCH --nodes=36
#SBATCH --time=02:00:00

cores=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1

source $HOME/anaconda3/bin/activate
source activate noise_correlations
fold=/project/projectdirs/m2043/jlivezey/noise_correlations

for dim in {2..10}; do
  srun -N 4 -n 272 -c $cores python -u dist_analysis.py $fold $fold kohn $dim &
done
wait
