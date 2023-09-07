eval "$(conda shell.bash hook)"
conda activate noise_correlations
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
mpiexec -n 24 python -u /home/jlivezey/Data/noise_correlations/scripts/discrim_null_model_analysis.py \
    --data_path=/home/jlivezey/Data/nc/pvc11/spikes_gratings/data_monkey1_gratings.mat \
    --rotation_path=/home/jlivezey/Data/nc/rotations.h5 \
    --correlation_path=/home/jlivezey/Data/nc/correlations.h5 \
    --save_folder=/home/jlivezey/Data/nc/exp10_1 \
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
    --inner_loop_verbose 2>&1 | tee five.txt
