#!/bin/bash
#SBATCH --constraint=haswell
#SBATCH --nodes=2
#SBATCH --qos=debug
#SBATCH --time=00:30:00

source activate myLSST
srun -N 1 -n 32 -c 2 --cpu-bind=cores python compute_DS_multiprocess.py &
srun -N 1 -n 32 -c 2 --cpu-bind=cores python compute_DS_multiprocess.py &
wait