#!/bin/bash
#SBATCH -N 342
#SBATCH -C knl
#SBATCH -q regular
#SBATCH -J nc
#SBATCH --mail-user=jlivezey@lbl.gov
#SBATCH --mail-type=ALL
#SBATCH -t 08:00:00
#SBATCH --image=docker:pssachdeva/neuro:latest
#SBATCH --output=/global/cscratch1/sd/jlivezey/exp10_v1_20201015_Y35_360_out.o
#SBATCH --error=/global/cscratch1/sd/jlivezey/exp10_v1_20201015_Y35_360_err.o

export OMP_NUM_THREADS=5
export OMP_PLACES=threads
export OMP_PROC_BIND=true

srun -n 4104 -c 20 shifter \
    python -u $HOME/noise_correlations/scripts/discrim_null_model_analysis.py \
    --data_path=/global/cscratch1/sd/sachdeva/data/pvc11/monkey1.mat \
    --rotation_path=/global/cscratch1/sd/sachdeva/neurocorr/rotations.h5 \
    --correlation_path=/global/cscratch1/sd/sachdeva/neurocorr/correlations.h5 \
    --save_folder=/global/cscratch1/sd/sachdeva/neurocorr/exp10 \
    --save_tag=exp10_1_05 \
    --dataset=pvc11 \
    --dim_min=3 \
    --dim_max=20 \
    --n_dimlets=1000 \
    --n_repeats=1000 \
    --tuning_criteria=modulation_frac \
    --modulation_frac=0.60 \
    --frac=0.5 \
    --peak_response=2.56 \
    --random_seed=12112020 \
    --inner_loop_verbose
