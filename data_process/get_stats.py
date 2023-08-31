

import torch
import numpy as np
import h5py


#years = [2010,2012, 2014, 2016, 2018, 2020] 
years = [2020, 2021, 2022] 
global_means = np.zeros((1,1,1,1))
global_stds = np.zeros((1,1,1,1))
time_means = np.zeros((1,1,45,45)) 

for ii, year in enumerate(years):

    with h5py.File('/home/mcsp/Downloads/FourCastNet/data/train/'+str(year) + '.h5', 'r') as f:

        rnd_idx = np.random.randint(0, 8760-500) #8760   1460

        global_means += np.mean(f['fields'][rnd_idx:rnd_idx+500], keepdims=True, axis = (0,2,3))
        global_stds += np.var(f['fields'][rnd_idx:rnd_idx+500], keepdims=True, axis = (0,2,3))

global_means = global_means/len(years)
global_stds = np.sqrt(global_stds/len(years))
time_means = time_means/len(years)

np.save('/home/mcsp/Downloads/FourCastNet/data/stats/global_means.npy', global_means)
np.save('/home/mcsp/Downloads/FourCastNet/data/stats/global_stds.npy', global_stds)
np.save('/home/mcsp/Downloads/FourCastNet/data/stats/time_means.npy', time_means)

print("means: ", global_means)
print("stds: ", global_stds)






