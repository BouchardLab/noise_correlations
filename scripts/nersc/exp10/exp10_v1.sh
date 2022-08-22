#!/bin/bash
#SBATCH -N 256
#SBATCH -C haswell
#SBATCH -q regular
#SBATCH -J nc
#SBATCH --mail-user=jlivezey@lbl.gov
#SBATCH --mail-type=ALL
#SBATCH -t 08:00:00
#SBATCH --image=docker:jesselivezey/nc:latest
#SBATCH --output=/global/cscratch1/sd/jlivezey/exp10_v1_20201015_Y35_360_out.o
#SBATCH --error=/global/cscratch1/sd/jlivezey/exp10_v1_20201015_Y35_360_err.o

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OMP_PLACES=threads
export OMP_PROC_BIND=spread

srun -n 4096 -c 4 --cpu_bind=cores shifter \
    python -u $HOME/noise_correlations/scripts/lfi_null_model_analysis.py \
    --data_path=$SCRATCH/nc/v1/20201015_Y35/Z360 \
    --rotation_path=/global/cscratch1/sd/jlivezey/nc/rotations.h5 \
    --correlation_path=/global/cscratch1/sd/jlivezey/nc/correlations.h5 \
    --save_folder=/global/cscratch1/sd/jlivezey/nc/exp10 \
    --save_tag=exp10_20201015_Y35_Z360_Method1 \
    --dataset=v1 \
    --dim_min=3 \
    --dim_max=20 \
    --n_dimlets=3000 \
    --n_repeats=1000 \
    --random_seed=12112020 \
    --inner_loop_verbose

##SBATCH --output=/global/cscratch1/sd/jlivezey/exp10_v1_20200729_Y24_280_out.o
##SBATCH --error=/global/cscratch1/sd/jlivezey/exp10_v1_20200729_Y24_280_err.o
#    --data_path=$SCRATCH/nc/v1/20200729_Y24/Z280 \
#    --save_tag=exp10_20200729_Y24_Z280_Method1 \

##SBATCH --output=/global/cscratch1/sd/jlivezey/exp10_v1_20200729_Y24_310_out.o
##SBATCH --error=/global/cscratch1/sd/jlivezey/exp10_v1_20200729_Y24_310_err.o
#    --data_path=$SCRATCH/nc/v1/20200729_Y24/Z310 \
#    --save_tag=exp10_20200729_Y24_Z310_Method1 \

##SBATCH --output=/global/cscratch1/sd/jlivezey/exp10_v1_20201015_Y35_330_out.o
##SBATCH --error=/global/cscratch1/sd/jlivezey/exp10_v1_20201015_Y35_330_err.o
#    --data_path=$SCRATCH/nc/v1/20201015_Y35/Z330 \
#    --save_tag=exp10_20201015_Y35_Z330_Method1 \

##SBATCH --output=/global/cscratch1/sd/jlivezey/exp10_v1_20201018_Y35_330_out.o
##SBATCH --error=/global/cscratch1/sd/jlivezey/exp10_v1_20201018_Y35_330_err.o
#    --data_path=$SCRATCH/nc/v1/20201018_Y35/Z330 \
#    --save_tag=exp10_20201018_Y35_Z330_Method1 \

##SBATCH --output=/global/cscratch1/sd/jlivezey/exp10_v1_20201018_Y35_360_out.o
##SBATCH --error=/global/cscratch1/sd/jlivezey/exp10_v1_20201018_Y35_360_err.o
#    --data_path=$SCRATCH/nc/v1/20201018_Y35/Z360 \
#    --save_tag=exp10_20201018_Y35_Z360_Method1 \

##SBATCH --output=/global/cscratch1/sd/jlivezey/exp10_v1_20201021_Y36_300_out.o
##SBATCH --error=/global/cscratch1/sd/jlivezey/exp10_v1_20201021_Y36_300_err.o
#    --data_path=$SCRATCH/nc/v1/20201021_Y36/Z300 \
#    --save_tag=exp10_20201021_Y36_Z300_Method1 \

##SBATCH --output=/global/cscratch1/sd/jlivezey/exp10_v1_20201021_Y36_330_out.o
##SBATCH --error=/global/cscratch1/sd/jlivezey/exp10_v1_20201021_Y36_330_err.o
#    --data_path=$SCRATCH/nc/v1/20201021_Y36/Z330 \
#    --save_tag=exp10_20201021_Y36_Z330_Method1 \

##SBATCH --output=/global/cscratch1/sd/jlivezey/exp10_v1_20201022_Y36_360_out.o
##SBATCH --error=/global/cscratch1/sd/jlivezey/exp10_v1_20201022_Y36_360_err.o
#    --data_path=$SCRATCH/nc/v1/20201022_Y36/Z360 \
#    --save_tag=exp10_20201022_Y36_Z360_Method1 \
