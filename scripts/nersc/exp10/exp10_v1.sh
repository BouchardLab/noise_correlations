#!/bin/bash
#SBATCH -N 256
#SBATCH -C haswell
#SBATCH -q regular
#SBATCH -J nc
#SBATCH --mail-user=jlivezey@lbl.gov
#SBATCH --mail-type=ALL
#SBATCH -t 05:00:00
#SBATCH --image=docker:pssachdeva/neuro:latest
#SBATCH --output=/global/cscratch1/sd/jlivezey/exp10_v1_out.o
#SBATCH --error=/global/cscratch1/sd/jlivezey/exp10_v1_err.o

srun -n 4096 -c 2 shifter \
    python -u $HOME/noise_correlations/scripts/discrim_null_model_analysis.py \
    --data_path=$SCRATCH/data/ret2/200114_fov1_data.mat \
    --rotation_path=/global/cscratch1/sd/jlivezey/rotations.h5 \
    --correlation_path=/global/cscratch1/sd/jlivezey/correlations.h5 \
    --save_folder=/global/cscratch1/sd/jlivezey/exp10 \
    --save_tag=exp10 \
    --dataset=v1 \
    --dim_min=3 \
    --dim_max=20 \
    --n_dimlets=3000 \
    --n_repeats=1000 \
    --random_seed=12112020 \
    --inner_loop_verbose
