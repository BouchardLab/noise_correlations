#!/bin/bash
#SBATCH -N 342
#SBATCH -C knl
#SBATCH -q premium
#SBATCH -J nc
#SBATCH --mail-user=jlivezey@lbl.gov
#SBATCH --mail-type=ALL
#SBATCH -t 08:00:00
#SBATCH --image=docker:jesselivezey/nc:latest
#SBATCH --output=/global/cscratch1/sd/jlivezey/exp10_pvc_11-1_025_out.o
#SBATCH --error=/global/cscratch1/sd/jlivezey/exp10_pvc_11-1_025_error.o

export OMP_NUM_THREADS=5
export OMP_PLACES=threads
export OMP_PROC_BIND=true

srun -n 4104 -c 20 shifter \
    python -u $HOME/noise_correlations/scripts/lfi_null_model_analysis.py \
    --data_path=/global/cfs/cdirs/m2043/jlivezey/noise_correlations/kohn_pvc-11/spikes_gratings/data_monkey1_gratings.mat \
    --rotation_path=/global/cscratch1/sd/jlivezey/nc/rotations.h5 \
    --correlation_path=/global/cscratch1/sd/jlivezey/nc/correlations.h5 \
    --save_folder=/global/cscratch1/sd/jlivezey/nc/exp10 \
    --save_tag=exp10_1_025 \
    --dataset=pvc11 \
    --dim_min=3 \
    --dim_max=6 \
    --n_dimlets=1000 \
    --n_repeats=1000 \
    --tuning_criteria=modulation_frac \
    --modulation_frac=0.60 \
    --frac=0.25 \
    --peak_response=2.56 \
    --random_seed=12112020 \
    --inner_loop_verbose
