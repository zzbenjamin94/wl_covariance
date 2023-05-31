## Add paths
ptcl_dir = '/global/cscratch1/sd/zzhang13/MultiDark/MDPL2_particles/z0p00/'
clusters_dir = '/global/cscratch1/sd/zzhang13/MultiDark/MDPL2_ROCKSTAR_Halos/z0p00/'


### Reading the cluster_obj file. 
from setup import ptcl_dir, cluster_dir
box_length = 1000
ptcl_mass = 1.505e9 ##Msun/h


class clusters_obj(object):

    '''
    Initializes the cluster_obj for derivation of cluster properties. 
    
    (x,y,z): The 3D coordinates of the cluster in Mpc/h
    M: Mass of the cluster in Msun/h. Could be Mvir, M200c/M200m, M500c/M500m. 
    R: The radius of the cluster in Mpc/h to normalize. Could be R200c/R200m or R500c/R500m. 
    r_range_norm: the radii range in Mpc/h over the R_vir (or R200c/R200m) for clusters
    dz: Projection depth in Mpc/h. 
    '''
    def __init__(self, cluster_id, x, y, z, M, R, Ngal, r_range_norm, dz=200):
        r_bin_center = (r_range_norm[0:-1] + r_range_norm[1:])/2
        
        self.cluster_id = cluster_id
        self.x = x; self.y = y; self.z = z
        self.M = M
        self.R = R
        self.dz = dz
        self.Sigma = np.zeros((len(x), len(r_bin_center)))     #Sigma
        self.DS = np.zeros((len(x), len(r_bin_center)))        #Delta Sigma
        self.r_range = r_range_norm*R[:,np.newaxis] #Broadcasted to (N,len(r_range))
        self.Ngal = Ngal
        return
    
    '''
    Computes the Surface Density of all the clusters. This part requires parallelization. 
    '''
    def compute_Sigma_all(self):
        
        ptcl_files = glob(ptcl_dir() + 'snap_130.*')
        for file in ptcl_files:
            ptcl = readsnap(file, 'pos', 'dm', nth=10, suppress=1, single=True)
            
            ##Calculate the contribution of the particle file for each cluster
            ## This part of the code needs fixing.
            for i in range(len(self.x)):
                #Sigma_cluster_file = self.__compute_Sigma_percluster_perfile(ptcl, self.x[i], self. y[i], self.z[i], self.r_range[i], self.dz)
                r_range = self.r_range[i]
                sigma = np.zeros(len(r_range)-1)
        
                ##Periodic boundary condition for annulus. Halos are conditions at [0,1000] Mpc boundaries.       
                dx_sqr = np.asarray([(ptcl[:,0]-x[i])**2, (ptcl[:,0]-x[i]+box_length)**2, (ptcl[:,0]-x[i]-box_length)**2]).min(0)
                dy_sqr = np.asarray([(ptcl[:,1]-y[i])**2, (ptcl[:,1]-y[i]+box_length)**2, (ptcl[:,1]-y[i]-box_length)**2]).min(0)
                dz_min = np.asarray([np.abs(ptcl[:,2]-z[i]), np.abs(ptcl[:,2]-z[i]+box_length), np.abs(ptcl[:,2]-z[i]-box_length)]).min(0)

                for j in range(len(r_range)-1):
                    #radius for annulus
                    dr = r_range[j+1] - r_range[j]
                    r_cur = r_range[j]

                    #Masking
                    mask = dx_sqr + dy_sqr <= (r_cur+dr)**2
                    mask &= dz_min < self.dz

                    #Building an annulus
                    annulus_df = ptcl[mask]

                    #Find 2D density within the annulus
                    area = np.pi * ((r_cur+dr)**2 - (r_cur)**2.)

                    m_tot = len(annulus_df)*ptcl_mass

                    sigma[j] = m_tot/area
                
                self.Sigma[i] += sigma
            
        return  
    
    '''
    Computes the Surface Density of all the clusters. This part requires parallelization. 
    '''
    def compute_Sigma_all_multiprocess(self):
        
        ##Each rank gets a range of indices to read. 
        num_clusters = len(self.x)
        n_per_rank = math.floor(num_clusters/size) #This gives you the work that's evenly distributed across all ranks. 
        n_remain = num_clusters%n_per_rank #Remainder
        
        #Split evenly as much as possible. 
        if rank < n_remain:
            n_start = (n_per_rank+1)*rank
            n_end = (n_per_rank+1)*(rank+1)
        if rank >= n_remain: 
            n_start = (n_per_rank+1)*rank
            n_end = n_start + n_per_rank
            
        print("Indices for rank ", rank, ":", "[{:d},{;d}".format(n_start, n_end))
        
        ptcl_files = glob(ptcl_dir() + 'snap_130.*')
        for file in ptcl_files:
            if rank == 0: ptcl = readsnap(file, 'pos', 'dm', nth=10, suppress=1, single=True)
            comm.Bcast(ptcl, root = 0)
            print("Completed broadcasting the particle file. ")
            
            for i in range(n_start, n_end):
                r_range = self.r_range[i]
                sigma = np.zeros(len(r_range)-1)
        
                ##Periodic boundary condition for annulus. Halos are conditions at [0,1000] Mpc boundaries.       
                dx_sqr = np.asarray([(ptcl[:,0]-x[i])**2, (ptcl[:,0]-x[i]+box_length)**2, (ptcl[:,0]-x[i]-box_length)**2]).min(0)
                dy_sqr = np.asarray([(ptcl[:,1]-y[i])**2, (ptcl[:,1]-y[i]+box_length)**2, (ptcl[:,1]-y[i]-box_length)**2]).min(0)
                dz_min = np.asarray([np.abs(ptcl[:,2]-z[i]), np.abs(ptcl[:,2]-z[i]+box_length), np.abs(ptcl[:,2]-z[i]-box_length)]).min(0)

                for j in range(len(r_range)-1):
                    #radius for annulus
                    dr = r_range[j+1] - r_range[j]
                    r_cur = r_range[j]

                    #Masking
                    mask = dx_sqr + dy_sqr <= (r_cur+dr)**2
                    mask &= dz_min < self.dz

                    #Building an annulus
                    annulus_df = ptcl[mask]

                    #Find 2D density within the annulus
                    area = np.pi * ((r_cur+dr)**2 - (r_cur)**2.)

                    m_tot = len(annulus_df)*ptcl_mass

                    sigma[j] = m_tot/area
                
                self.Sigma[i] += sigma
            
        return
    
    '''
    Internal function to compute
    '''
    def __compute_Sigma_percluster_perfile(self, ptcl, x, y, z, r_range, dz):
        sigma = np.zeros(len(r_range)-1)
        
        ##Periodic boundary condition for annulus. Halos are conditions at [0,1000] Mpc boundaries.       
        dx_sqr = np.asarray([(ptcl[:,0]-x)**2, (ptcl[:,0]-x+box_length)**2, (ptcl[:,0]-x-box_length)**2]).min(0)
        dy_sqr = np.asarray([(ptcl[:,1]-y)**2, (ptcl[:,1]-y+box_length)**2, (ptcl[:,1]-y-box_length)**2]).min(0)
        dz_min = np.asarray([np.abs(ptcl[:,2]-z), np.abs(ptcl[:,2]-z+box_length), np.abs(ptcl[:,2]-z-box_length)]).min(0)

        for i in range(len(r_range)-1):
            #radius for annulus
            dr = r_range[i+1] - r_range[i]
            r_cur = r_range[i]

            #Masking
            mask = dx_sqr + dy_sqr <= (r_cur+dr)**2
            mask &= dz_min < dz

            #Building an annulus
            annulus_df = ptcl[mask]

            #Find 2D density within the annulus
            area = np.pi * ((r_cur+dr)**2 - (r_cur)**2.)
            m_tot = len(annulus_df)*ptcl_mass
            sigma[i] = m_tot/area
            
        return sigma
    
    def get_Sigma(self):
        return self.Sigma
    
    def get_DeltaSigma(self):
    ## DeltaSigma dimensions of M, L -- clusters and radial bins
        for i in range(len(self.x)):
            for j in range(r_bin_center):
                self.DS[i,j] = np.mean(self.Sigma[i,:j+1] - self.Sigma[i,j])
        return self.DS
    
    def get_Ngal(self):
        return self.Ngal
    
    def get_id(self):
        return self.cluster_id