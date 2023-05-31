#!/bin/bash
#SBATCH --constraint=haswell
#SBATCH --nodes=26
#SBATCH --qos=regular
#SBATCH --time=6:00:00

source activate myLSST
srun -N 26 -n 832 -c 2 --cpu-bind=cores python compute_sigma_m200c_all_z0p00.py  #&
#srun -N 25 -n 800 -c 2 --cpu-bind=cores python compute_sigma_m200c_all_z0p49.py  &
#srun -N 5 -n 160 -c 2 --cpu-bind=cores python compute_sigma_m200c_all_z1p03.py  &
#wait