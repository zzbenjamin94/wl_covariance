#!/bin/bash
#SBATCH --image=zzbenjamin94/mydesc:1.0 
#SBATCH --qos=debug
#SBATCH -N 1
#SBATCH -C haswell
#SBATCH -t 30

source /opt/miniconda3/etc/profile.d/conda.sh
conda activate myLSST
srun -n 4 shifter python test_mpi4py.py