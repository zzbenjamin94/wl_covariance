import numpy as np

## Radial bins

## Log bins
r_bins_log_norm = np.linspace(-1,1,21)
r_bins_lin_norm = 10**r_bins_log_norm
r_cent_log_norm= (r_bins_log_norm[1:] + r_bins_log_norm[:-1])/2
r_cent_lin_norm = 10**r_cent_log_norm

## Mass bins
mass_bin_edges = [5e13, 1e14, 2e14, 5e14, 1e15]
mass_bins = [[mass_bin_edges[i],mass_bin_edges[i+1]] for i in range(len(mass_bin_edges)-1)]
mass_bin_cent = [(mass_bin_edges[i+1]+mass_bin_edges[i])/2 for i in range(len(mass_bin_edges)-1)]


## Redshift bins
a_bins = [1, 0.6712, 0.4922]