#!/bin/bash
#SBATCH -N 256
#SBATCH -C haswell
#SBATCH -q regular
#SBATCH -J nc
#SBATCH --mail-user=pratik.sachdeva@berkeley.edu
#SBATCH --mail-type=ALL
#SBATCH -t 07:00:00
#SBATCH --image=docker:pssachdeva/neuro:latest
#SBATCH --output=/global/homes/s/sachdeva/scripts/neurocorr/exp10/exp10_pvc11-3_out.o
#SBATCH --error=/global/homes/s/sachdeva/scripts/neurocorr/exp10/exp10_pvc11-3_err.o

srun -n 4096 -c 2 shifter \
    python -u $HOME/noise_correlations/scripts/discrim_null_model_analysis.py \
    --data_path=/global/cscratch1/sd/sachdeva/data/pvc11/monkey3.mat \
    --rotation_path=/global/cscratch1/sd/sachdeva/neurocorr/rotations.h5 \
    --correlation_path=/global/cscratch1/sd/sachdeva/neurocorr/correlations.h5 \
    --save_folder=/global/cscratch1/sd/sachdeva/neurocorr/exp10 \
    --save_tag=exp10_3 \
    --dataset=pvc11 \
    --dim_min=3 \
    --dim_max=20 \
    --n_dimlets=1000 \
    --n_repeats=1000 \
    --tuning_criteria=modulation_frac \
    --modulation_frac=0.60 \
    --peak_response=2.56 \
    --random_seed=12112020 \
    --inner_loop_verbose
