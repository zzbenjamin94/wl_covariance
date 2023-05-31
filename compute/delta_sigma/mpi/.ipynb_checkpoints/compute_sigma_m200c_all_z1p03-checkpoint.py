## testing multidark
import numpy as np
import matplotlib.pyplot as plt
from time import time
import seaborn as sns
from scipy.stats import kde
import h5py
import astropy.io.fits as fits
import csv
import pandas as pd
import h5py
import tables
import os
from astropy.table import Table
from astropy.coordinates import SkyCoord
from tqdm import tqdm
from astropy.io import ascii
import os
from glob import glob
import math
import pandas as pd
from pygadgetreader import *
import mpi4py as mpi
from mpi4py import MPI
import pickle

import sys
sys.path.append('/global/homes/z/zzhang13/BaryonPasting/CorrelatedStructures/')

##Ignore warnings
import warnings
warnings.filterwarnings("ignore")


## Add paths
ptcl_dir = '/global/cscratch1/sd/zzhang13/MultiDark/MDPL2_particles/z1p03/'
clusters_dir = '/global/homes/z/zzhang13/BaryonPasting/CorrelatedStructures/multiDark/data/'


##Global variables
box_length = 1000
ptcl_mass = 1.505e9 ##Msun/h
ptcl_samp = 10 
dz = 200 ## Projection depth Mpc/h

#Common variables.
m_low = 5e13; m_high = 1e15

mass_bin_edges = [5e13, 1e14, 2e14, 5e14, 1e15]
mass_bins = [[mass_bin_edges[i],mass_bin_edges[i+1]] for i in range(len(mass_bin_edges)-1)]

##This part needs to change. Program needs to read 500 clusters per bin, not total. 

r_bins_log_norm = np.linspace(-1,1,21)
r_bins_lin_norm = 10**r_bins_log_norm
r_cent_log_norm= (r_bins_log_norm[1:] + r_bins_log_norm[:-1])/2
r_cent_lin_norm = 10**r_cent_log_norm

##Compute background density
from numpy import sqrt
delta=200.0
omega_m=0.3089
omega_l=0.6911
omega_b = 0.0486 
hubble=0.6774
redshift = 1.03 ## This to change

aexp = 1./(1.+redshift)
Ez = sqrt(omega_m/aexp**3.0+omega_l)
fb = omega_b/omega_m
mu = 0.59
mue = 1.14
delta = float(delta)
erg_to_keV = 624150647.99632

# critical density of the Universe in h^2*Msun/Mpc^3
rho_crit = (2.77536627e11)*(Ez)**2.0

### Background density
rho_bkgr = rho_crit * omega_m
rho_2d_bkgr = rho_bkgr * 2 * dz
    
##Remove the object oriented programming. Write as simple functions. 
    
if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    cluster_df = pd.read_csv(clusters_dir + 'clusters_m200c_n500_all_z1p03.csv' ) ##File with n500 samples in each bin
    
    #Mass cut
    mask = (cluster_df.M200c >= m_low) & (cluster_df.M200c < m_high)
    cur_df = cluster_df[mask]
    #cur_df = cur_df.head(10) ## Testing a handful of clusters
    
    sample_num = len(cur_df)
        
    ##All processes sync here
    comm.Barrier()

    ## For root process read the cluster file and its properties
    if rank == 0:    
        x = np.array(cur_df.x); y = np.array(cur_df.y); z = np.array(cur_df.z)
        m200c = np.array(cur_df.M200c); r200c = np.array(cur_df.R200c)
        cluster_id = np.array(cur_df.CtreesHaloID)

        #Expand r_range to that of individual clusters. 
        r_range = r_bins_lin_norm*r200c[:,np.newaxis]

    else:
        x = np.zeros(sample_num); y = np.zeros(sample_num); z = np.zeros(sample_num)
        r_range = np.zeros((sample_num, len(r_bins_lin_norm)))


    ##Broadcast the datavectors
    comm.Bcast(x); comm.Bcast(y); comm.Bcast(z);
    comm.Bcast(r_range)

    sigma = np.zeros((sample_num, len(r_cent_lin_norm)))
    deltasigma = np.zeros((sample_num, len(r_cent_lin_norm)))

    ##Splitting the array by rank. Array splitting allows uneven split
    split = np.array_split(sigma,size,axis = 0) #Split input array by the number of available cores
    split_sizes = []

    for i in range(0,len(split),1):
        split_sizes = np.append(split_sizes, len(split[i]))

    n_start = int(np.insert(np.cumsum(split_sizes),0,0)[rank])
    n_end =  int(np.insert(np.cumsum(split_sizes),0,0)[rank]) + len(split[rank]) 
    n_local = len(split[rank])
    print("For rank ", rank, "n_start: ", n_start, ", n_end: ", n_end)
    print("Rank ", rank, "number of clusters: ", n_local)

    gather_length = split_sizes*len(sigma[0,:]) # multiplied by the length of the second dimension 
    gather_offset = np.insert(np.cumsum(gather_length),0,0)[0:-1] 

    if rank == 0:
        print('Gatherv_length: ', gather_length)
        print('Gatherv_offset: ', gather_offset)

    sigma_local = np.zeros((n_local, len(r_cent_lin_norm)))
    deltasigma_local = np.zeros((n_local, len(r_cent_lin_norm)))

    comm.Barrier()

    ## Compute Delta Sigma
    start = time()
    ptcl_files = glob(ptcl_dir + 'snap_098.*')

    for file in ptcl_files:
        start_file = time()
        ptcl = readsnap(file, 'pos', 'dm', nth=ptcl_samp, suppress=1, single=True)

    #if rank == 0: print("Completed reading ptcl file: ", file)

        for i in range(n_start, n_end): #This part is different for different clusters. 
            start_file = time()
            r_range_cl = r_range[i]
            sigma_cl = np.zeros(len(r_cent_lin_norm))
            deltasigma_cl = np.zeros(len(r_cent_lin_norm))

            ##Periodic boundary condition for annulus. Halos are conditions at [0,1000] Mpc boundaries.       
            dx_sqr = np.asarray([(ptcl[:,0]-x[i])**2, (ptcl[:,0]-x[i]+box_length)**2, (ptcl[:,0]-x[i]-box_length)**2]).min(0)
            dy_sqr = np.asarray([(ptcl[:,1]-y[i])**2, (ptcl[:,1]-y[i]+box_length)**2, (ptcl[:,1]-y[i]-box_length)**2]).min(0)
            dz_min = np.asarray([np.abs(ptcl[:,2]-z[i]), np.abs(ptcl[:,2]-z[i]+box_length), np.abs(ptcl[:,2]-z[i]-box_length)]).min(0)

            for j in range(len(r_cent_lin_norm)):
                #radius for annulus
                dr = r_range_cl[j+1] - r_range_cl[j]
                r_cur = r_range_cl[j]

                #Masking
                mask_DS = dx_sqr + dy_sqr < (r_cur+dr)**2 
                mask_DS &= dz_min < dz
                mask_Sigma = mask_DS & (dx_sqr + dy_sqr >= (r_cur)**2)

                #Building an annulus
                annulus_df = ptcl[mask_Sigma]

                #Find 2D density within the annulus
                area_annulus = np.pi * ((r_cur+dr)**2 - (r_cur)**2.)
                m_annulus = len(annulus_df)*ptcl_mass*ptcl_samp
                sigma_cl[j] = m_annulus/area_annulus - rho_2d_bkgr

                ##Finding DeltaSigma
                circle_df = ptcl[mask_DS]
                area_circle = np.pi*(r_cur+dr)**2. ##This modified from r_cur to r_cur+dr
                m_circle = len(circle_df)*ptcl_mass*ptcl_samp
                sigma_avg = m_circle/area_circle - rho_2d_bkgr
                deltasigma_cl[j] = sigma_avg - sigma_cl[j]

            sigma_local[i-n_start] += sigma_cl
            deltasigma_local[i-n_start] += deltasigma_cl
            end_file = time()
            #if rank == 0: print("Time elapsed for cluster is: ", end_file - start_file)


    comm.Gatherv(sigma_local, [sigma, gather_length, gather_offset, MPI.DOUBLE], root=0)
    comm.Gatherv(deltasigma_local, [deltasigma, gather_length, gather_offset, MPI.DOUBLE], root=0)

    comm.Barrier()
    end = time()
    if rank == 0: print("Total Time elapsed  = ", end-start)

    ##Save onto a pickle file
    output = {}
    if rank == 0:
        output['cluster_id'] = cluster_id
        output['sigma'] = sigma
        output['r_range'] = r_range
        output['delta_sigma'] = deltasigma

        with open(clusters_dir+'sigma_m200c_all_z1p03.pkl','wb') as handle:
            pickle.dump(output, handle)
    