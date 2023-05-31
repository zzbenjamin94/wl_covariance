import mpi4py
from mpi4py import MPI
import numpy as np
from pygadgetreader import * ##Test if this function exists

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

#create random matrix of 0s and 1s on rank 0, prepare
#empty buffer on other ranks
if rank == 0:
    asize = 5000
    rng = np.random.default_rng(seed=42)
    #low is inclusive and high is exclusive
    data = rng.integers(low=0, high=2, size=(asize, asize))
else:
    asize = 5000
    data = np.empty([asize, asize], dtype=int)

#broadcast random matrix to all ranks
comm.Bcast(data, root=0)

#now do scalar multiply by rank number
data_mult= data*rank

#now find matrix max
data_max = np.max(data_mult)

#gather our results back to rank 0
gather_data = np.empty(size, dtype=int)
comm.Gather(data_max, gather_data, root=0)

#check that our results are correct
if rank == 0:
    #adjust for zero indexing
    asize = size - 1
    max_gather = np.max(gather_data).astype(int)
    assert asize == max_gather