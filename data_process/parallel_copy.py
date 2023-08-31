
import h5py
from mpi4py import MPI
import numpy as np
import time
from netCDF4 import Dataset as DS
from netCDF4 import Dataset
import os

def writetofile(time_steps, src, dest, channel_idx, varslist):
    if os.path.isfile(src):
        batch = 2**6
        rank = MPI.COMM_WORLD.rank
        Nproc = MPI.COMM_WORLD.size
        Nimgtot = time_steps #src_shape[0]

        Nimg = Nimgtot//Nproc
        print("Nimgtot",Nimgtot)
        print("Nproc",Nproc)
        print("Nimg",Nimg)
        base = rank*Nimg
        end = (rank+1)*Nimg if rank<Nproc - 1 else Nimgtot
        idx = base

        for variable_name in varslist:

            fsrc = DS(src, 'r', format="NETCDF4").variables[variable_name]
            #print('Source, ', fsrc[:])
            fdest = h5py.File(dest, 'a', driver='mpio', comm=MPI.COMM_WORLD)

            start = time.time()
            while idx<end:
                if end - idx < batch:
                    ims = fsrc[idx:end]
                    print(ims.shape)
                    fdest['fields'][idx:end, channel_idx, :, :] = ims
                    break
                else:
                    ims = fsrc[idx:idx+batch]
                    fdest['fields'][idx:idx+batch, channel_idx, :, :] = ims
                    idx+=batch
                    ttot = time.time() - start
                    eta = (end - base)/((idx - base)/ttot)
                    hrs = eta//3600
                    mins = (eta - 3600*hrs)//60
                    secs = (eta - 3600*hrs - 60*mins)

            ttot = time.time() - start
            hrs = ttot//3600
            mins = (ttot - 3600*hrs)//60
            secs = (ttot - 3600*hrs - 60*mins)
            channel_idx += 1 

            
np.set_printoptions(precision=6)

# Open the NetCDF file in read-write mode
year_lst = [2021,2022, 2023]


for year in year_lst:
    file_name = '45x45_' + str(year) +'.nc'
    file_path = '/home/mcsp/Downloads/FourCastNet/copernicus/' + file_name # File download from API
    # Open the NetCDF file in read-write mode
    with Dataset(file_path, 'r+') as nc_file:
        # Access the 'u10' and 'v10' variables
        u10_variable = nc_file.variables['u10']
        v10_variable = nc_file.variables['v10']
        u100_variable = nc_file.variables['u100']


        # Read the data from the variables into NumPy arrays
        u10_data = u10_variable[:]
        v10_data = v10_variable[:]
        u100_data = u100_variable[:]
 
        new_variable_data10 = np.sqrt(u10_data**2 + v10_data**2)
        u100_variable[:] = new_variable_data10
        nc_file.sync()

for year in year_lst:
    source_dest = '/home/mcsp/Downloads/FourCastNet/data/' # Srore processed data
    dest = source_dest + str(year) +'.h5' 
    file_name = '45x45_' + str(year) + '.nc'
    source_sfc = file_path
    fsrc = DS(source_sfc, 'r', format="NETCDF4")
    time_steps = 8760
    time_steps = fsrc['time'].shape[0]
    var = 1
    lat = fsrc['latitude'].shape[0]
    long = fsrc['longitude'].shape[0]
    with h5py.File(dest, 'w') as f:
        f.create_dataset('fields', shape = (time_steps, var, lat, long), dtype='f')
    src = source_sfc

    writetofile(time_steps, src, dest, 0, ['u100'])


with h5py.File(dest, 'r') as f:
    n_samples_per_year = f['fields'].shape[0]
    img_shape_x = f['fields'].shape[2]
    img_shape_y = f['fields'].shape[3]
    print('n_samples_per_year', n_samples_per_year)
    print('img_shape_x', img_shape_x)

"""
writetofile(src, dest, 2, ['t2m'])

#sp mslp
src = source_sfc
writetofile(src, dest, 3, ['sp'])
writetofile(src, dest, 4, ['msl'])

#t850
src = source_pl
writetofile(src, dest, 5, ['t'], 2)

#uvz1000
src = source_pl
writetofile(src, dest, 6, ['u'], 3)
writetofile(src, dest, 7, ['v'], 3)
writetofile(src, dest, 8, ['z'], 3)


#uvz850
src = source_pl
writetofile(src, dest, 9, ['u'])
writetofile(src, dest, 10, ['v'])
writetofile(src, dest, 11, ['z'])

#uvz 500
src = source_pl
writetofile(src, dest, 12, ['u'])
writetofile(src, dest, 13, ['v'])
writetofile(src, dest, 14, ['z'])

#t500
src = source_pl
writetofile(src, dest, 15, ['t'])

#z50
src = source_pl
writetofile(src, dest, 16, ['z'])

#r500 
src = source_pl
writetofile(src, dest, 17, ['r'])

#r850
src = source_pl
writetofile(src, dest, 18, ['r'])

#tcwv
src = source_sfc
writetofile(src, dest, 19, ['tcwv'])

"""