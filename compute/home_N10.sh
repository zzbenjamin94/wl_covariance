#!/bin/bash -l

#SBATCH --qos=debug
#SBATCH --constraint=haswell
#SBATCH -t 00:05:00
#SBATCH -N 10
#SBATCH -o home-N1.out
#SBATCH --mail-user=lastephey@lbl.gov
#SBATCH --mail-type=ALL

#OpenMP settings:
export OMP_NUM_THREADS=1
export OMP_PLACES=threads
export OMP_PROC_BIND=spread

echo "starting N 1 trials"
echo "Python located at /global/homes/s/stephey/.conda/envs/mpihome/bin/python"
for i in {1..5}
do
   srun -n 96 -c 2 python test_mpi4py.py
   echo "N 10 trial $i completed"
   sleep 10
done
