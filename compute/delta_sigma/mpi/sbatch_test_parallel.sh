#!/bin/bash
#SBATCH --constraint=haswell
#SBATCH --nodes=4
#SBATCH --time=00:30:00

source activate myLSST
srun -n 4 -c 16 --cpu-bind=cores python test_parallel.py
