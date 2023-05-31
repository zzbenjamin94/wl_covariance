#!/bin/bash
#SBATCH --constraint=haswell
#SBATCH --nodes=3
#SBATCH --qos=regular
#SBATCH --time=8:00:00

source activate myLSST
#Contains 180 clusters
srun -N 1 -n 32 -c 2 --cpu-bind=cores python compute_sigma_mvir_5e14_1e15_z0p49.py  &

#Contains 350 clusters
srun -N 2 -n 64 -c 2 --cpu-bind=cores python compute_sigma_mvir_2e14_5e14_z1p03.py  &
wait