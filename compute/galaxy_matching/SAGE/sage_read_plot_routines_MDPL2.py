from pylab import *
from scipy import signal as ss

#Not sure if float should be 32 or 64

def galdtype():
	# Define the data-type for the public version of SAGE
	Galdesc_full = [
        ('dbId'                         , np.int64),
        ('snapnum'                      , np.int16),
        ('redshift'                     , np.float64),
        ('rockstarId'                   , np.int64),
        ('depthFirstId'                 , np.int64),
        ('forestId'                     , np.int64),
        ('GalaxyID'                     , np.int64),
        ('HostHaloID'                   , np.int64),
        ('MainHaloID'                   , np.int64),
        ('GalaxyType'                   , np.int8),
        ('HaloMass'                     , np.float64),
        ('Vmax'                         , np.float64),
        ('spin'                         , np.float64), #Should this be 3 parameters for the 3 coordinates
        ('x'                            , np.float64),
        ('y'                            , np.float64),
        ('z'                            , np.float64),
        ('vx'                           , np.float64),
        ('vy'                           , np.float64),
        ('vz'                           , np.float64),
        ('MstarSpheroid'                , np.float64),
        ('MstarDisk'                    , np.float64),
        ('McoldDisk'                    , np.float64),
        ('Mhot'                         , np.float64),
        ('Mbh'                          , np.float64),
        ('SFRspheroid'                  , np.float64),
        ('SFRdisk'                      , np.float64),
        ('SFR'                          , np.float64),
        ('MZhotHalo'                    , np.float64),
        ('MZstarSpheroid'               , np.float64),
        ('MZstarDisk'                   , np.float64),
        ('MeanAgeStars'                 , np.float64),
        ('NInFile'                      , np.int32),
        ('fileNum'                      , np.int32),
        ('ix'                           , np.int32),
        ('iy'                           , np.int32),
        ('iz'                           , np.int32),
        ('phkey'                        , np.int64),
        ]
    
	names = [Galdesc_full[i][0] for i in range(len(Galdesc_full))]
	formats = [Galdesc_full[i][1] for i in range(len(Galdesc_full))]
	Galdesc = np.dtype({'names':names, 'formats':formats}, align=True)
	return Galdesc


def sageoutsingle(fname):
	# Read a single SAGE output file, intended only as a subroutine of read_sagesnap
	Galdesc_full = galdtype()
	fin = open(fname, 'rb')  # Open the file
	Ntrees = np.fromfile(fin,np.dtype(np.int32),1)  # Read number of trees in file
	NtotGals = int(np.fromfile(fin,np.dtype(np.int32),1)[0])  # Read number of gals in file.
	GalsPerTree = np.fromfile(fin, np.dtype((np.int32, Ntrees)),1) # Read the number of gals in each tree
	G = np.fromfile(fin, Galdesc_full, NtotGals) # Read all the galaxy data
	return G, NtotGals

##For Mvir > 5e13
def read_sagesnap_mvircut(fpre, Galdesc, firstfile=0, lastfile=7):
    # Read full SAGE snapshot, going through each file and compiling into 1 array
    #Galdesc = galdtype()
    
    columns = [Galdesc[i][0] for i in range(len(Galdesc))]
    Glist = []
    Ngal = np.array([])
    for i in range(firstfile,lastfile+1):
        G1, N1 = sageoutsingle(fpre+'_'+str(i))
        G_filt = G1['CentralMvir'] > 5e2 ##Applied a Mvir cut to this 
        G1 = G1[G_filt]
        N1 = len(G1)
        G1 = G1[columns]
        Glist += [G1]
        Ngal = np.append(Ngal,N1)

    G = np.empty(int(sum(Ngal)), dtype=Galdesc)
    for i in range(firstfile,lastfile+1):
        j = i-firstfile
        G[int(sum(Ngal[:j])):int(sum(Ngal[:j+1]))] = Glist[j][0:int(Ngal[j])].copy()
        
    
    G = G.view(np.recarray)
    return G
