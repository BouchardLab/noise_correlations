export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1

for dim in {2..10}; do
  echo $dim
  mpirun -n $1 python -u dist_analysis.py /data/noise_correlations . $2 $dim
done
