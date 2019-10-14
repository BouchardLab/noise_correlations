#!/bin/bash -l
#SBATCH --qos=debug
#SBATCH --constraint=knl
#SBATCH --nodes=4
#SBATCH --time=00:30:00

cores=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1

source $HOME/anaconda3/bin/activate
source activate noise_correlations
fold=/project/projectdirs/m2043/jlivezey/noise_correlations

for dim in {2..10}; do
  srun -N 2 -n 136 -c $cores python -u dist_analysis.py $fold $fold kohn $dim -n 100 -s 1000 &
done
wait
