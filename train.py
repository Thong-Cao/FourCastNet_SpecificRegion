
import os
import time
import numpy as np
import argparse
import h5py
import torch
import cProfile
import re
import torchvision
from torchvision.utils import save_image
import torch.nn as nn
import torch.cuda.amp as amp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
import logging
from utils import logging_utils
logging_utils.config_logger()
from utils.YParams import YParams
from utils.data_loader_multifiles import get_data_loader
from networks.afnonet import AFNONet, PrecipNet
from utils.img_utils import vis_precip
#import wandb
from utils.weighted_acc_rmse import weighted_acc, weighted_rmse, weighted_rmse_torch, unlog_tp_torch
#from apex import optimizers
from utils.darcy_loss import LpLoss
import matplotlib.pyplot as plt
from collections import OrderedDict
import pickle
DECORRELATION_TIME = 36 # 9 days
import json
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap as ruamelDict
if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("--run_num", default='00', type=str)
  parser.add_argument("--yaml_config", default='./config/AFNO.yaml', type=str)
  parser.add_argument("--config", default='afno_backbone', type=str)
  parser.add_argument("--enable_amp", action='store_true')
  parser.add_argument("--epsilon_factor", default = 0, type = float)

  args, unknown = parser.parse_known_args()

  params = YParams(os.path.abspath(args.yaml_config), args.config)
  params['epsilon_factor'] = args.epsilon_factor

  params['world_size'] = 1
  if 'WORLD_SIZE' in os.environ:
    params['world_size'] = int(os.environ['WORLD_SIZE'])

  world_rank = 0
  local_rank = 0
  if params['world_size'] > 1:
    dist.init_process_group(backend='nccl',
                            init_method='env://')
    local_rank = int(os.environ["LOCAL_RANK"])
    args.gpu = local_rank
    world_rank = dist.get_rank()
    params['global_batch_size'] = params.batch_size
    params['batch_size'] = int(params.batch_size//params['world_size'])

  torch.cuda.set_device(local_rank)
  torch.backends.cudnn.benchmark = True

  # Set up directory
  expDir = os.path.join(params.exp_dir, args.config, str(args.run_num))
  if  world_rank==0:
    if not os.path.isdir(expDir):
      os.makedirs(expDir)
      os.makedirs(os.path.join(expDir, 'training_checkpoints/'))

  params['experiment_dir'] = os.path.abspath(expDir)
  params['checkpoint_path'] = os.path.join(expDir, 'training_checkpoints/ckpt.tar')
  params['best_checkpoint_path'] = os.path.join(expDir, 'training_checkpoints/best_ckpt.tar')

  # Do not comment this line out please:
  args.resuming = True if os.path.isfile(params.checkpoint_path) else False

  params['resuming'] = args.resuming
  params['local_rank'] = local_rank
  params['enable_amp'] = args.enable_amp

  # this will be the wandb name
#  params['name'] = args.config + '_' + str(args.run_num)
#  params['group'] = "era5_wind" + args.config
  params['name'] = args.config + '_' + str(args.run_num)
  params['group'] = "era5_precip" + args.config
  params['project'] = "ERA5_precip"
  params['entity'] = "flowgan"

  if world_rank==0:
    logging_utils.log_to_file(logger_name=None, log_filename=os.path.join(expDir, 'out.log'))
    logging_utils.log_versions()
    params.log()

  #params['log_to_wandb'] = (world_rank==0) and params['log_to_wandb']
  params['log_to_screen'] = (world_rank==0) and params['log_to_screen']

  params['in_channels'] = np.array(params['in_channels'])
  params['out_channels'] = np.array(params['out_channels'])
  params['N_in_channels'] = len(params['in_channels'])
  params['N_out_channels'] = len(params['out_channels'])
    
  if world_rank == 0:
    hparams = ruamelDict()
    yaml = YAML()
    for key, value in params.params.items():
      hparams[str(key)] = str(value)
    with open(os.path.join(expDir, 'hyperparams.yaml'), 'w') as hpfile:
      yaml.dump(hparams,  hpfile )

    




  params = params
  world_rank = world_rank
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  #if params.log_to_wandb:
    #wandb.init(config=params, name=params.name, group=params.group, project=params.project, entity=params.entity)

  train_data_loader, train_dataset, train_sampler = get_data_loader(params, params.train_data_path, dist.is_initialized(), train=True)
  valid_data_loader, valid_dataset = get_data_loader(params, params.valid_data_path, dist.is_initialized(), train=False)
  loss_obj = LpLoss()


  params.crop_size_x = valid_dataset.crop_size_x
  params.crop_size_y = valid_dataset.crop_size_y
  params.img_shape_x = valid_dataset.img_shape_x
  params.img_shape_y = valid_dataset.img_shape_y

  model = AFNONet(params, img_size=(45, 45), patch_size=(50,50), in_chans=1, out_chans=1).to(device) 
  #model = torch.nn.DataParallel(model)
  optimizer = torch.optim.Adam(model.parameters(), lr = params.lr)
  gscaler = amp.GradScaler()
  scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.2, patience=5, mode='min')


  print(device)
  print("Starting Training Loop...")

  startEpoch = 0
  best_valid_loss = 1.e6
  for epoch in range(startEpoch, params.max_epochs):
      iters = 0
      start = time.time()

      tr_time = 0
      data_time = 0
      model.train()
      stop_iter = len(train_data_loader)
      for i, data in enumerate(train_data_loader, 0):
          iters += 1
          data_start = time.time()
          inp, tar = map(lambda x: x.to(device, dtype = torch.float), data)     ###
          data_time += time.time() - data_start

          tr_start = time.time()

          model.zero_grad()
          with amp.autocast(params.enable_amp):
              gen = model(inp).to(device, dtype = torch.float)
              loss = loss_obj(gen, tar)
          loss.backward()
          optimizer.step()
          tr_time += time.time() - tr_start
          if iters%200 == 0:
              percentage_complete = (iters / stop_iter) * 100
              print(f"Traning Process - Epoch: {epoch} - iters: {iters}  {iters}/{stop_iter} ({percentage_complete:.2f}%), ")
          logs = {'loss': loss}
      train_logs = logs


      ######################################################################################### VALI 1 EPOCH

      iters = 0
      model.eval()
      n_valid_batches = 20 #do validation on first 20 images, just for LR scheduler
      mult = torch.as_tensor(np.load(params.global_stds_path)[0, params.out_channels, 0, 0]).to(device)

      valid_buff = torch.zeros((3), dtype=torch.float32, device=device)
      valid_loss = valid_buff[0].view(-1)
      valid_l1 = valid_buff[1].view(-1)
      valid_steps = valid_buff[2].view(-1)
      valid_weighted_rmse = torch.zeros((params.N_out_channels), dtype=torch.float32, device=device)
      valid_weighted_acc = torch.zeros((params.N_out_channels), dtype=torch.float32, device=device)

      valid_start = time.time()

      with torch.no_grad():
          for i, data in enumerate(valid_data_loader, 0):
              iters += 1
              if i>=n_valid_batches:
                  break    
              inp, tar  = map(lambda x: x.to(device, dtype = torch.float), data)
              gen = model(inp).to(device, dtype = torch.float)
              
              valid_loss += loss_obj(gen, tar) 
              valid_l1 += nn.functional.l1_loss(gen, tar)
              valid_steps += 1
              valid_weighted_rmse += weighted_rmse_torch(gen, tar)
              if iters%50 == 0:
                  percentage_complete = (iters / stop_iter) * 100
                  print(f"Progress: {iters}/{stop_iter} ({percentage_complete:.2f}%)")
      valid_buff[0:2] = valid_buff[0:2] / valid_buff[2]
      valid_weighted_rmse = valid_weighted_rmse / valid_buff[2]
      valid_weighted_rmse *= mult
      # download buffers
      valid_buff_cpu = valid_buff.detach().cpu().numpy()
      valid_weighted_rmse_cpu = valid_weighted_rmse.detach().cpu().numpy()
      valid_time = time.time() - valid_start
      valid_weighted_rmse = mult*torch.mean(valid_weighted_rmse, axis = 0)
      try:
          os.mkdir(params['experiment_dir'] + "/" + str(i))
      except:
          pass
      save_image(torch.cat((gen[0,0], torch.zeros((valid_dataset.img_shape_x+1, 5)).to(device, dtype = torch.float), tar[0,0]), axis = 1), params['experiment_dir'] + "/" + str(i) + "/" + str(epoch) + ".png")
      
      try:
          logs = {'valid_l1': valid_buff_cpu[1], 'valid_loss': valid_buff_cpu[0], 'valid_rmse_u10': valid_weighted_rmse_cpu[0], 'valid_rmse_v10': valid_weighted_rmse_cpu[1]}
      except:
          logs = {'valid_l1': valid_buff_cpu[1], 'valid_loss': valid_buff_cpu[0], 'valid_rmse_u10': valid_weighted_rmse_cpu[0]}#, 'valid_rmse_v10': valid_weighted_rmse[1]}
      valid_logs = logs

      scheduler.step(valid_logs['valid_loss'])

      if world_rank == 0:
          torch.save({'iters': iters, 'epoch': epoch, 'model_state': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()}, params.checkpoint_path)
		
          if valid_logs['valid_loss'] <= best_valid_loss:

              #logging.info('Val loss improved from {} to {}'.format(best_valid_loss, valid_logs['valid_loss']))
              torch.save({'iters': iters, 'epoch': epoch, 'model_state': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()}, params.best_checkpoint_path)
	
              best_valid_loss = valid_logs['valid_loss']
              print('Save checkpoint at {} '.format(params.best_checkpoint_path))
	      
      print()
      print('Time taken for epoch {} is {:.2f} sec'.format(epoch , time.time()-start))
      print('train data time={:.2f}, train step time={:.2f}, valid step time={:.2f}'.format(data_time, tr_time, valid_time))
      print('Train loss: {:.3f}. Valid loss: {:.3f}'.format(train_logs['loss'], valid_logs['valid_loss']))

  print('Save checkpoint at {} '.format(params.checkpoint_path))
