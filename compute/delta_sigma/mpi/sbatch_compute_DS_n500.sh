#!/bin/bash
#SBATCH --constraint=haswell
#SBATCH --nodes=4
#SBATCH --qos=regular
#SBATCH --time=5:00:00

source activate myLSST
#srun -N 2 -n 64 -c 2 --cpu-bind=cores python compute_sigma_mvir_2e14_5e14_z0p49.py  &
srun -N 4 -n 128 -c 2 --cpu-bind=cores python compute_sigma_mvir_5e14_1e15_z0p49.py  #&
#srun -N 2 -n 64 -c 2 --cpu-bind=cores python compute_sigma_mvir_2e14_5e14_z1p03.py  &
#wait