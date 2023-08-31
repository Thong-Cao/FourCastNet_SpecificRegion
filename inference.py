# libraries
import os, sys, time
import numpy as np
import h5py
# Get the model config from default configs
sys.path.insert(1, './FourCastNet/') # insert code repo into path
from utils.YParams import YParams
# We are going to use a default config. Please see github repo for other config examples
config_file = "./config/AFNO.yaml"
config_name = "afno_backbone"
params = YParams(config_file, config_name)
print("Model architecture used = {}".format(params["nettype"]))
import torch
import torchvision
import torch.nn as nn
import matplotlib.pyplot as plt
# import model
from networks.afnonet import AFNONet
from collections import OrderedDict

def load_model(model, params, checkpoint_file):
    ''' helper function to load model weights '''
    checkpoint_fname = checkpoint_file
    checkpoint = torch.load(checkpoint_fname)
    try:
        ''' FourCastNet is trained with distributed data parallel
            (DDP) which prepends 'module' to all keys. Non-DDP 
            models need to strip this prefix '''
        new_state_dict = OrderedDict()
        for key, val in checkpoint['model_state'].items():
            name = key[7:]
            if name != 'ged':
                new_state_dict[name] = val  
        model.load_state_dict(new_state_dict)
    except:
        model.load_state_dict(checkpoint['model_state'])
    model.eval() # set to inference mode
    return model

# define metrics from the definitions above
def lat(j: torch.Tensor, num_lat: int) -> torch.Tensor:
    return 90. - j * 180./float(num_lat-1)

def latitude_weighting_factor(j: torch.Tensor, num_lat: int, s: torch.Tensor) -> torch.Tensor:
    return num_lat * torch.cos(3.1416/180. * lat(j, num_lat))/s

def weighted_rmse_channels(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    #takes in arrays of size [n, c, h, w]  and returns latitude-weighted rmse for each channel
    num_lat = pred.shape[2]
    lat_t = torch.arange(start=0, end=num_lat, device=pred.device)
    s = torch.sum(torch.cos(3.1416/180. * lat(lat_t, num_lat)))
    weight = torch.reshape(latitude_weighting_factor(lat_t, num_lat, s), (1, 1, -1, 1))
    result = torch.sqrt(torch.mean((pred - target)**2., dim=(-1,-2)))
    return result

def weighted_acc_channels(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    #takes in arrays of size [n, c, h, w]  and returns latitude-weighted acc for each channel
    num_lat = pred.shape[2]
    lat_t = torch.arange(start=0, end=num_lat, device=pred.device)
    s = torch.sum(torch.cos(3.1416/180. * lat(lat_t, num_lat)))
    weight = torch.reshape(latitude_weighting_factor(lat_t, num_lat, s), (1, 1, -1, 1))
    result = torch.sum( pred * target, dim=(-1,-2)) / torch.sqrt(torch.sum( pred * pred, dim=(-1,-2)) * torch.sum( target *
    target, dim=(-1,-2)))
    return result
    
def inference(data_slice, model, prediction_length, idx):
    # create memory for the different stats
    n_out_channels = params['N_out_channels']
    acc = torch.zeros((prediction_length, n_out_channels)).to(device, dtype=torch.float)
    rmse = torch.zeros((prediction_length, n_out_channels)).to(device, dtype=torch.float)
    mape = torch.zeros((prediction_length, n_out_channels)).to(device, dtype=torch.float)
    # load time means: represents climatology

    # to conserve GPU mem, only save one channel (can be changed if sufficient GPU mem or move to CPU)
    targets = torch.zeros((prediction_length, 1, img_shape_x, img_shape_y)).to(device, dtype=torch.float)
    predictions = torch.zeros((prediction_length, 1, img_shape_x, img_shape_y)).to(device, dtype=torch.float)

     
    with torch.no_grad():
        for i in range(data_slice.shape[0]): 
            if i == 0:
                first = data_slice[0:1]
                future = data_slice[1:2]
                pred = first
                tar = first
                # also save out predictions for visualizing channel index idx
                targets[0,0] = first[0,idx]
                predictions[0,0] = first[0,idx]
                # predict
                future_pred = model(first)
            else:
                if i < prediction_length - 1:
                    future = data_slice[i+1:i+2]
                future_pred = model(future_pred) # autoregressive step

            if i < prediction_length - 1:
                predictions[i+1,0] = future_pred[0,idx]
                targets[i+1,0] = future[0,idx]
            # compute metrics using the ground truth ERA5 data as "true" predictions
            rmse[i] = weighted_rmse_channels(pred, tar) * std
            acc[i] = weighted_acc_channels(pred-m, tar-m)
 
            print('Predicted timestep {} of {}. {} RMSE Error: {}, ACC: {}, MAPE: {}'.format(i, prediction_length, field, rmse[i,idx], acc[i,idx], mape[i,idx]))  
            pred = future_pred
            tar = future
    # copy to cpu for plotting/vis
    acc_cpu = acc.cpu().numpy()
    rmse_cpu = rmse.cpu().numpy()
    predictions_cpu = predictions.cpu().numpy()
    targets_cpu = targets.cpu().numpy()
    
    return acc_cpu, rmse_cpu, predictions_cpu, targets_cpu
##################################################################################################################
# data and model paths
data_file = '/home/mcsp/Downloads/FourCastNet/data/test/2022.h5'
model_path = '/home/mcsp/Downloads/FourCastNet/weight/afno_backbone/0/training_checkpoints/best_ckpt.tar'

time_means_path =  '/home/mcsp/Downloads/FourCastNet/stats/time_means.npy'
global_means_path = '/home/mcsp/Downloads/FourCastNet/stats/global_means.npy'
global_stds_path =  '/home/mcsp/Downloads/FourCastNet/stats/global_stds.npy'

img_shape_x =45
img_shape_y =45

dt = 1 # time step (x 1 hours)
ic = 0 # start the inference from here
prediction_length = 25 # number of steps (x 6 hours)

variables = ['wind_10'] 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# in and out channels: FourCastNet uses 20 input channels corresponding to 20 prognostic variables 
in_channels = np.array([0])
out_channels = np.array([0])

params['N_in_channels'] = len(in_channels)
params['N_out_channels'] = len(out_channels)
params.means = np.load(global_means_path)[0, out_channels] # for normalizing data with precomputed train stats
params.stds = np.load(global_stds_path)[0, out_channels]
params.time_means = np.load(time_means_path)[0, out_channels]

# load time means: represents climatology


# load the model
if params.nettype == 'afno':
    model = AFNONet(params, img_size=(img_shape_x, img_shape_y)).to(device)  # AFNO model 
else:
    raise Exception("not implemented")
# load saved model weights
model = load_model(model, params, model_path)
model = model.to(device)

# move normalization tensors to gpu

# means and stds over training data
means = params.means
stds = params.stds

# load climatological means
time_means = params.time_means # temporal mean (for every pixel)
m = torch.as_tensor((time_means - means)/stds)[:, 0:img_shape_x]
m = torch.unsqueeze(m, 0)
# these are needed to compute ACC and RMSE metrics
m = m.to(device, dtype=torch.float)
std = torch.as_tensor(stds[:,0,0]).to(device, dtype=torch.float)



# which field to track for visualization 
field = 'wind_10'
idx_vis = variables.index(field) # also prints out metrics for this field

# get prediction length slice from the data
print('Loading inference data')
print('Inference data from {}'.format(data_file))

data = h5py.File(data_file, 'r')['fields'][ic:(ic+prediction_length*dt):dt,in_channels,0:img_shape_x]
print(data.shape)
print("Shape of data = {}".format(data.shape))
print("Start= {}".format(ic))
# run inference
data = (data - means)/stds # standardize the data
data = torch.as_tensor(data).to(device, dtype=torch.float) # move to gpu for inference
acc_cpu, rmse_cpu, predictions_cpu, targets_cpu = inference(data, model, prediction_length, idx=idx_vis)

