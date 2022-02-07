#!/bin/bash
#SBATCH -N 256
#SBATCH -C haswell
#SBATCH -q regular
#SBATCH -J nc
#SBATCH --mail-user=pratik.sachdeva@berkeley.edu
#SBATCH --mail-type=ALL
#SBATCH -t 05:00:00
#SBATCH --image=docker:pssachdeva/neuro:latest
#SBATCH --output=/global/homes/s/sachdeva/scripts/neurocorr/exp10/exp10_ecog_out.o
#SBATCH --error=/global/homes/s/sachdeva/scripts/neurocorr/exp10/exp10_ecog_err.o

srun -n 4096 -c 2 shifter \
    python -u $HOME/noise_correlations/scripts/discrim_null_model_analysis.py \
    --data_path=/global/cscratch1/sd/sachdeva/data/ecog/r32_b7.mat \
    --rotation_path=/global/cscratch1/sd/sachdeva/neurocorr/rotations.h5 \
    --correlation_path=/global/cscratch1/sd/sachdeva/neurocorr/correlations.h5 \
    --save_folder=/global/cscratch1/sd/sachdeva/neurocorr/exp10 \
    --save_tag=exp10 \
    --dataset=ecog \
    --dim_min=3 \
    --dim_max=20 \
    --n_dimlets=3000 \
    --n_repeats=1000 \
    --random_seed=12112020 \
    --inner_loop_verbose
