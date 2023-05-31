from scipy.stats import kde
import h5py
import astropy.io.fits as fits
import csv
import pandas as pd
import numpy as np
import tables
import pickle
import os
from astropy.table import Table
from astropy.coordinates import SkyCoord
from tqdm import tqdm
from astropy.io import ascii
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import incredible as cr
from scipy import stats
import scipy.optimize as opt
import emcee
import tqdm

import warnings
warnings.filterwarnings("ignore")

ptcl_mass = 1.505e9 ## Msun/h

def ptcl_dir(redshift):
    redshift_str = '{:.2f}'.format(redshift)
    redshift_str = redshift_str.replace('.','p')
    ptcl_dir = '/global/cscratch1/sd/zzhang13/MultiDark/MDPL2_particles/z{}/'.format(redshift_str)
    if not os.path.exists(ptcl_dir):
        raise Exception('something is very wrong: %s does not exist'%ptcl_dir)   
    return ptcl_dir

def cluster_dir(redshift):
    redshift_str = '{:.2f}'.format(redshift)
    redshift_str = redshift_str.replace('.','p')
    
    cluster_dir = '/global/cscratch1/sd/zzhang13/MultiDark/MDPL2_ROCKSTAR_Halos/z{}/'.format(redshift_str)
    if not os.path.exists(cluster_dir):
        raise Exception('something is very wrong: %s does not exist'%cluster_dir)
    return cluster_dir

def repo_dir():
    repo_dir = '/global/homes/z/zzhang13/BaryonPasting/CorrelatedStructures/multiDark/'
    if not os.path.exists(repo_dir):
        raise Exception('something is very wrong: %s does not exist'%repo_dir)
    return repo_dir
    
def root_dir():
    root_dir = '/global/homes/z/zzhang13/'
    if not os.path.exists(root_dir):
        raise Exception('something is very wrong: %s does not exist'%root_dir)
    return root_dir

def chains_dir():
    chains_dir = '/global/homes/z/zzhang13/BaryonPasting/CorrelatedStructures/multiDark/data/chains/'
    if not os.path.exists(chains_dir):
        raise Exception('something is very wrong: %s does not exist'%chains_dir)
    return chains_dir

def plots_dir():
    plots_dir = '/global/homes/z/zzhang13/BaryonPasting/CorrelatedStructures/multiDark/plots/'
    if not os.path.exists(plots_dir):
        raise Exception('something is very wrong: %s does not exist'%plots_dir)
    return plots_dir

def data_dir():
    data_dir = '/global/homes/z/zzhang13/BaryonPasting/CorrelatedStructures/multiDark/data/'
    if not os.path.exists(data_dir):
        raise Exception('something is very wrong: %s does not exist'%data_dir)
    return data_dir

def kllr_dir():
    kllr_dir = '/global/homes/z/zzhang13/kllr/'
    if not os.path.exists(kllr_dir):
        raise Exception('something is very wrong: %s does not exist'%kllr_dir)
    return kllr_dir

def tools_dir():
    tools_dir = '/global/homes/z/zzhang13/BaryonPasting/CorrelatedStructures/multiDark/tools/'
    if not os.path.exists(tools_dir):
        raise Exception('something is very wrong: %s does not exist'%tools_dir)
    return tools_dir

