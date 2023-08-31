# libraries
import os, sys, time
import numpy as np

import pandas as pd
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

            #print('Predicted timestep {} of {}. {} RMSE Error: {}, ACC: {}, MAPE: {}'.format(i, prediction_length, field, rmse[i,idx], acc[i,idx], mape[i,idx]))  
            
            pred = future_pred
            tar = future

    # copy to cpu for plotting/vis
    acc_cpu = acc.cpu().numpy()
    rmse_cpu = rmse.cpu().numpy()
    predictions_cpu = predictions.cpu().numpy()
    targets_cpu = targets.cpu().numpy()
    
    return acc_cpu, rmse_cpu, predictions_cpu, targets_cpu
##################################################################################################################

data_file = '/home/mcsp/Downloads/FourCastNet/data/test/2022.h5'
model_path = '/home/mcsp/Downloads/FourCastNet/weight/afno_backbone/TW0079/training_checkpoints/best_ckpt.tar'
time_means_path =  '/home/mcsp/Downloads/FourCastNet/stats/time_means.npy'
global_means_path = '/home/mcsp/Downloads/FourCastNet/stats/global_means.npy'
global_stds_path =  '/home/mcsp/Downloads/FourCastNet/stats/global_stds.npy'




variables = ['wind_10']

device = 'cpu'

in_channels = np.array([0])
out_channels = np.array([0])

params['N_in_channels'] = len(in_channels)
params['N_out_channels'] = len(out_channels)
params.means = np.load(global_means_path)[0, out_channels] # for normalizing data with precomputed train stats
params.stds = np.load(global_stds_path)[0, out_channels]
params.time_means = np.load(time_means_path)[0, out_channels]
# load time means: represents climatology
img_shape_x =45
img_shape_y =45

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


df1 = pd.DataFrame()
df2 = pd.DataFrame()
df3 = pd.DataFrame()
df4 = pd.DataFrame()
df5 = pd.DataFrame()
df6 = pd.DataFrame()
df7 = pd.DataFrame()
df8 = pd.DataFrame()
df9 = pd.DataFrame()
df10 = pd.DataFrame()
df11 = pd.DataFrame()
df12 = pd.DataFrame()
df13 = pd.DataFrame()
df14 = pd.DataFrame()
df15 = pd.DataFrame()
df16 = pd.DataFrame()
df17 = pd.DataFrame()
df18 = pd.DataFrame()
df19 = pd.DataFrame()
df20 = pd.DataFrame()
df21 = pd.DataFrame()
df22 = pd.DataFrame()
df23 = pd.DataFrame()
df24 = pd.DataFrame()

d = {}
for i in range(0, 720): # rolling

    dt = 1 # time step (x 1 hours)
    ic = i # start the inference from here
    prediction_length = 25 # number of steps (x 6 hours)
    field = 'wind_10'
    idx_vis = variables.index(field) # also prints out metrics for this field

    data = h5py.File(data_file, 'r')['fields'][ic:(ic+prediction_length*dt):dt,in_channels,0:img_shape_x]

    print("Start= {}".format(ic))

    # run inference
    data = (data - means)/stds # standardize the data
    data = torch.as_tensor(data).to(device, dtype=torch.float) # move to gpu for inference
    acc_cpu, rmse_cpu, predictions_cpu, targets_cpu = inference(data, model, prediction_length, idx=idx_vis)


    #TW_0097_35.652_126.194 -- 11km
    #35.49742222222223, 경도: 126.31845555555554
    latitudes = np.linspace(34.5-4.25, 37+4.25, num=45)  
    longitudes = np.linspace(125-4.25, 127.5+4.25, num=45) 

    target_lat = 35.5
    target_lon = 126.25

    # Find the index of the target latitude and longitude in the latitudes and longitudes arrays
    lat_index = np.where(np.isclose(latitudes, target_lat))[0][0]
    lon_index = np.where(np.isclose(longitudes, target_lon))[0][0]
    # means and stds over training data
    means = params.means
    stds = params.stds

    pred = stds*predictions_cpu + means
    true = stds*targets_cpu + means

    # Extract the data at the target latitude and longitude for each time step
    target_data = pred[:, 0:1, lat_index, lon_index]
    true_data = true[:, 0:1, lat_index, lon_index]
    target_data = target_data.flatten(),
    true_data =  true_data.flatten(),


    df1 = df1.append([{'target': target_data[0][1],          
                        'predict':true_data[0][1]}])
    df2 = df2.append([{'target': target_data[0][2],          
                        'predict':true_data[0][2]}])
    df3 = df3.append([{'target': target_data[0][3],          
                        'predict':true_data[0][3]}])
    df4 = df4.append([{'target': target_data[0][4],          
                        'predict':true_data[0][4]}])
    df5 = df5.append([{'target': target_data[0][5],          
                        'predict':true_data[0][5]}])
    df6 = df6.append([{'target': target_data[0][6],          
                        'predict':true_data[0][6]}])

    df7 = df7.append([{'target': target_data[0][7],          
                        'predict':true_data[0][7]}])
    df8 = df8.append([{'target': target_data[0][8],          
                        'predict':true_data[0][8]}])
    df9 = df9.append([{'target': target_data[0][9],          
                        'predict':true_data[0][9]}])
    df10 = df10.append([{'target': target_data[0][10],          
                        'predict':true_data[0][10]}])
    df11 = df11.append([{'target': target_data[0][11],          
                        'predict':true_data[0][11]}])
    df12 = df12.append([{'target': target_data[0][12],          
                        'predict':true_data[0][12]}])
    df13 = df13.append([{'target': target_data[0][13],          
                        'predict':true_data[0][13]}])
    df14 = df14.append([{'target': target_data[0][14],          
                        'predict':true_data[0][14]}])
    df15 = df15.append([{'target': target_data[0][15],          
                        'predict':true_data[0][15]}])
    df16 = df16.append([{'target': target_data[0][16],          
                        'predict':true_data[0][16]}])
    df17 = df17.append([{'target': target_data[0][17],          
                        'predict':true_data[0][17]}])
    df18 = df18.append([{'target': target_data[0][18],          
                        'predict':true_data[0][18]}])
    df19 = df19.append([{'target': target_data[0][19],          
                        'predict':true_data[0][19]}])
    df20 = df20.append([{'target': target_data[0][20],          
                        'predict':true_data[0][20]}])
    df21 = df21.append([{'target': target_data[0][21],          
                        'predict':true_data[0][21]}])
    df22 = df22.append([{'target': target_data[0][22],          
                        'predict':true_data[0][22]}])
    df23 = df23.append([{'target': target_data[0][23],          
                        'predict':true_data[0][23]}])
    df24 = df24.append([{'target': target_data[0][24],          
                        'predict':true_data[0][24]}])
    
path = '/home/mcsp/Downloads/FourCastNet/test_result/'
df1.to_excel(path + 'step1.xlsx', index=False)
df2.to_excel(path+ '/step2.xlsx', index=False)
df3.to_excel(path+ '/step3.xlsx', index=False)
df4.to_excel(path+ '/step4.xlsx', index=False)
df5.to_excel(path+ '/step5.xlsx', index=False)
df6.to_excel(path+ '/step6.xlsx', index=False)
df7.to_excel(path+ '/step7.xlsx', index=False)
df8.to_excel(path+ '/step8.xlsx', index=False)
df9.to_excel(path+ '/step9.xlsx', index=False)
df10.to_excel(path+ '/step10.xlsx', index=False)
df11.to_excel(path+ '/step11.xlsx', index=False)
df12.to_excel(path+ '/step12.xlsx', index=False)
df13.to_excel(path+ '/step13.xlsx', index=False)
df14.to_excel(path+ '/step14.xlsx', index=False)
df15.to_excel(path+ '/step15.xlsx', index=False)
df16.to_excel(path+ '/step16.xlsx', index=False)
df17.to_excel(path+ '/step17.xlsx', index=False)
df18.to_excel(path+ '/step18.xlsx', index=False)
df19.to_excel(path+ '/step19.xlsx', index=False)
df20.to_excel(path+ '/step20.xlsx', index=False)
df21.to_excel(path+ '/step21.xlsx', index=False)
df22.to_excel(path+ '/step22.xlsx', index=False)
df23.to_excel(path+ '/step23.xlsx', index=False)
df24.to_excel(path+ '/step24.xlsx', index=False)
print('Done')


lstRMSE = []
lstMAPE = []

for i in range(1, 25):
    filename = '/step'+str(i)+'.xlsx'
    # Read the data from Excel file
    data = pd.read_excel('/home/mcsp/Downloads/FourCastNet/test_result/' + filename)

    # Extract target and prediction columns from the data
    target = data['target'].values
    prediction = data['predict'].values

    # Calculate RMSE (Root Mean Squared Error)
    rmse = np.sqrt(np.mean((prediction - target) ** 2))

    # Calculate MAPE (Mean Absolute Percentage Error)
    mape = np.mean(np.abs((target - prediction) / target)) * 100
    lstRMSE.append(rmse)
    lstMAPE.append(mape)
    print("RMSE:", rmse)
    print("MAPE:", mape)


df = pd.DataFrame({
        'rmse': lstRMSE,
        'mape': lstMAPE
    })
df = df.round(2)
    # Save DataFrame to Excel
df.to_excel('eval_result.xlsx', index=False)
