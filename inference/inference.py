#!/usr/bin/python
# -*- coding: utf-8 -*-
# BSD 3-Clause License
#
# Copyright (c) 2022, FourCastNet authors
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# The code was authored by the following people:
#
# Jaideep Pathak - NVIDIA Corporation
# Shashank Subramanian - NERSC, Lawrence Berkeley National Laboratory
# Peter Harrington - NERSC, Lawrence Berkeley National Laboratory
# Sanjeev Raja - NERSC, Lawrence Berkeley National Laboratory
# Ashesh Chattopadhyay - Rice University
# Morteza Mardani - NVIDIA Corporation
# Thorsten Kurth - NVIDIA Corporation
# David Hall - NVIDIA Corporation
# Zongyi Li - California Institute of Technology, NVIDIA Corporation
# Kamyar Azizzadenesheli - Purdue University
# Pedram Hassanzadeh - Rice University
# Karthik Kashinath - NVIDIA Corporation
# Animashree Anandkumar - California Institute of Technology, NVIDIA Corporation

import argparse
import os
import sys
import time

import numpy as np

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/../')
import logging
from collections import OrderedDict

import h5py
import torch
import torch.cuda.amp as amp
import torch.distributed as dist
import torch.nn as nn
import torchvision
from numpy.core.numeric import False_
from torch.nn.parallel import DistributedDataParallel
from torchvision.utils import save_image
from utils import logging_utils
from utils.weighted_acc_rmse import (
    unweighted_acc_torch_channels,
    weighted_acc_masked_torch_channels,
    weighted_acc_torch_channels,
    weighted_rmse_torch_channels,
)

logging_utils.config_logger()
import glob
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import wandb
from networks.afnonet import AFNONet
from utils.data_loader_multifiles import get_data_loader
from utils.YParams import YParams



def get_base_year(filename):
    # extract the year using regular expressions
    year = os.path.basename(filename)[:4]
    # check if the year is a 4-digit number
    if len(year) == 4 and year.isdigit():
        print("Year:", year)
    else:
        print("Invalid filename format")
    return year


# def get_base_year(base_path):
#     year_match = re.search(r'\b(19[7-9]\d|20[0-2]\d)\b', os.path.basename(base_path))
#     if year_match:
#         return year_match.group(1)
#     else:
#         raise ValueError(f"Invalid base path format: {base_path}")



def gaussian_perturb(x, level=0.01, device=0):
    # Add Gaussian noise to the input tensor x with a specified noise level and device
    noise = level * torch.randn(x.shape).to(device, dtype=torch.float)
    return x + noise


def load_model(model, params, checkpoint_file):
    # Clear the gradients of the model
    model.zero_grad()
    
    # Load the checkpoint from the specified file
    checkpoint_fname = checkpoint_file
    checkpoint = torch.load(checkpoint_fname)
    
    try:
        # Create a new state dictionary and copy the model state from the checkpoint, excluding the 'ged' key
        new_state_dict = OrderedDict()
        for (key, val) in checkpoint['model_state'].items():
            name = key[7:]
            if name != 'ged':
                new_state_dict[name] = val
        # Load the new state dictionary into the model
        model.load_state_dict(new_state_dict)
    except:
        # If the above fails, directly load the model state from the checkpoint
        model.load_state_dict(checkpoint['model_state'])
    
    # Set the model to evaluation mode
    model.eval()
    
    # Return the loaded model
    return model




def downsample(x, scale=0.125):
    return torch.nn.functional.interpolate(x, scale_factor=scale, mode='bilinear')



def setup(params):
    # Get the device (GPU if available, else CPU)
    device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'

    # Get the data loader and dataset
    valid_data_loader, valid_dataset = get_data_loader(params, params.inf_data_path, dist.is_initialized(), train=False)

    # Get image shape from the dataset
    img_shape_x = valid_dataset.img_shape_x
    img_shape_y = valid_dataset.img_shape_y
    params.img_shape_x = img_shape_x
    params.img_shape_y = img_shape_y

    # Log loading of trained model checkpoint
    if params.log_to_screen:
        logging.info(f'Loading trained model checkpoint from {params["best_checkpoint_path"]}')

    # Get input and output channels
    in_channels = np.array(params.in_channels)
    out_channels = np.array(params.out_channels)
    n_in_channels = len(in_channels)
    n_out_channels = len(out_channels)

    # Set number of input and output channels in params
    params['N_in_channels'] = n_in_channels + 1 if params['orography'] else n_in_channels
    params['N_out_channels'] = n_out_channels

    # Load means and stds needed for standardizing wind data
    params.means = np.load(params.global_means_path)[0, out_channels]
    params.stds = np.load(params.global_stds_path)[0, out_channels]

    # Load the model based on the network type
    if params.nettype == 'afno':
        model = AFNONet(params).to(device)
    else:
        raise Exception('not implemented')

    # Load the model weights from the checkpoint file
    checkpoint_file = params['best_checkpoint_path']
    model = load_model(model, params, checkpoint_file)
    model = model.to(device)

    # Load the validation data paths
    files_paths = sorted(glob.glob(params.inf_data_path + '/*.h5'))
    logging.warning(f"Loading validation data {files_paths}")

    # Select the year for inference
    yr = 0
    if params.log_to_screen:
        logging.info(f'Loading inference data from {files_paths[yr]}')


    valid_year = get_base_year(files_paths[yr])
    # Load the validation data from the selected year
    valid_data_full = h5py.File(files_paths[yr], 'r')['fields']

    return valid_data_full, model, valid_year



def autoregressive_inference(
    params,
    ic,
    valid_data_full,
    model,
    ):
    ic = int(ic)

    # initialize global variables

    device = \
        (torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'
         )
    exp_dir = params['experiment_dir']
    dt = int(params.dt)
    prediction_length = int(params.prediction_length / dt)
    n_history = params.n_history
    img_shape_x = params.img_shape_x
    img_shape_y = params.img_shape_y
    in_channels = np.array(params.in_channels)
    out_channels = np.array(params.out_channels)
    n_in_channels = len(in_channels)
    n_out_channels = len(out_channels)
    means = params.means
    stds = params.stds

    # initialize memory for image sequences and RMSE/ACC

    valid_loss = torch.zeros((prediction_length,
                             n_out_channels)).to(device,
            dtype=torch.float)
    acc = torch.zeros((prediction_length, n_out_channels)).to(device,
            dtype=torch.float)

    # compute metrics in a coarse resolution too if params.interp is nonzero

    valid_loss_coarse = torch.zeros((prediction_length,
                                    n_out_channels)).to(device,
            dtype=torch.float)
    acc_coarse = torch.zeros((prediction_length,
                             n_out_channels)).to(device,
            dtype=torch.float)
    acc_coarse_unweighted = torch.zeros((prediction_length,
            n_out_channels)).to(device, dtype=torch.float)

    acc_unweighted = torch.zeros((prediction_length,
                                 n_out_channels)).to(device,
            dtype=torch.float)
    seq_real = torch.zeros((prediction_length, n_in_channels,
                           img_shape_x, img_shape_y)).to(device,
            dtype=torch.float)
    seq_pred = torch.zeros((prediction_length, n_in_channels,
                           img_shape_x, img_shape_y)).to(device,
            dtype=torch.float)

    acc_land = torch.zeros((prediction_length,
                           n_out_channels)).to(device,
            dtype=torch.float)
    acc_sea = torch.zeros((prediction_length,
                          n_out_channels)).to(device, dtype=torch.float)
    if params.masked_acc:
        maskarray = \
            torch.as_tensor(np.load(params.maskpath)[0:720]).to(device,
                dtype=torch.float)

    valid_data = valid_data_full[ic:ic + prediction_length * dt
                                 + n_history * dt:dt, in_channels, 0:
                                 720]  # extract valid data from first year

    # standardize

    valid_data = (valid_data - means) / stds
    valid_data = torch.as_tensor(valid_data).to(device,
            dtype=torch.float)

    # load time means

    if not params.use_daily_climatology:
        m = torch.as_tensor((np.load(params.time_means_path)[0][out_channels]- means) / stds)[:, 0:img_shape_x]  # climatology
        m = torch.unsqueeze(m, 0)
    else:

      # use daily clim like weyn et al. (different from rasp)

        dc_path = params.dc_path
        with h5py.File(dc_path, 'r') as f:
            dc = f['time_means_daily'][ic:ic + prediction_length * dt:dt]  # 1460,21,721,1440
        m = torch.as_tensor((dc[:, out_channels, 0:img_shape_x, :]- means) / stds)

    m = m.to(device, dtype=torch.float)
    if params.interp > 0:
        m_coarse = downsample(m, scale=params.interp)

    std = torch.as_tensor(stds[:, 0, 0]).to(device, dtype=torch.float)

    orography = params.orography
    orography_path = params.orography_path
    if orography:
        orog = \
            torch.as_tensor(np.expand_dims(np.expand_dims((h5py.File(orography_path,'r')['orog'])[0:720], axis=0),axis=0)).to(device, dtype=torch.float)
        logging.info('orography loaded; shape:{}'.format(orog.shape))

    # autoregressive inference

    if params.log_to_screen:
        logging.info('Begin autoregressive inference')

    with torch.no_grad():
        for i in range(valid_data.shape[0]):
            if i == 0:  # start of sequence
                first = valid_data[0:n_history + 1]
                future = valid_data[n_history + 1]
                for h in range(n_history + 1):
                    seq_real[h] = first[h * n_in_channels:(h + 1)
                        * n_in_channels][0:n_out_channels]  # extract history from 1st
                    seq_pred[h] = seq_real[h]
                if params.perturb:
                    first = gaussian_perturb(first,
                            level=params.n_level, device=device)  # perturb the ic
                if orography:
                    future_pred = model(torch.cat((first, orog),
                            axis=1))
                else:
                    future_pred = model(first)
            else:
                if i < prediction_length - 1:
                    future = valid_data[n_history + i + 1]
                if orography:
                    future_pred = model(torch.cat((future_pred, orog),
                            axis=1))  # autoregressive step
                else:
                    future_pred = model(future_pred)  # autoregressive step

            if i < prediction_length - 1:  # not on the last step
                seq_pred[n_history + i + 1] = future_pred
                seq_real[n_history + i + 1] = future
                history_stack = seq_pred[i + 1:i + 2 + n_history]

            future_pred = history_stack

        # Compute metrics

            if params.use_daily_climatology:
                clim = m[i:i + 1]
                if params.interp > 0:
                    clim_coarse = m_coarse[i:i + 1]
            else:
                clim = m
                if params.interp > 0:
                    clim_coarse = m_coarse

            pred = torch.unsqueeze(seq_pred[i], 0)
            tar = torch.unsqueeze(seq_real[i], 0)
            valid_loss[i] = weighted_rmse_torch_channels(pred, tar) \
                * std
            acc[i] = weighted_acc_torch_channels(pred - clim, tar
                    - clim)
            acc_unweighted[i] = unweighted_acc_torch_channels(pred
                    - clim, tar - clim)

            if params.masked_acc:
                acc_land[i] = weighted_acc_masked_torch_channels(pred
                        - clim, tar - clim, maskarray)
                acc_sea[i] = weighted_acc_masked_torch_channels(pred
                        - clim, tar - clim, 1 - maskarray)

            if params.interp > 0:
                pred = downsample(pred, scale=params.interp)
                tar = downsample(tar, scale=params.interp)
                valid_loss_coarse[i] = \
                    weighted_rmse_torch_channels(pred, tar) * std
                acc_coarse[i] = weighted_acc_torch_channels(pred
                        - clim_coarse, tar - clim_coarse)
                acc_coarse_unweighted[i] = \
                    unweighted_acc_torch_channels(pred - clim_coarse,
                        tar - clim_coarse)

            if params.log_to_screen:
                tmp_dict = params["idxes"]
                idx = tmp_dict[params["fld"]]
                logging.info('Predicted timestep {} of {}. {} RMS Error: {}, ACC: {}'.format(i,
                             prediction_length, args.fld, valid_loss[i,
                             idx], acc[i, idx]))
                if params.interp > 0:
                    logging.info('[COARSE] Predicted timestep {} of {}. {} RMS Error: {}, ACC: {}'.format(i,
                                 prediction_length, args.fld,
                                 valid_loss_coarse[i, idx],
                                 acc_coarse[i, idx]))

    seq_real = seq_real.cpu().numpy()
    seq_pred = seq_pred.cpu().numpy()
    valid_loss = valid_loss.cpu().numpy()
    acc = acc.cpu().numpy()
    acc_unweighted = acc_unweighted.cpu().numpy()
    acc_coarse = acc_coarse.cpu().numpy()
    acc_coarse_unweighted = acc_coarse_unweighted.cpu().numpy()
    valid_loss_coarse = valid_loss_coarse.cpu().numpy()
    acc_land = acc_land.cpu().numpy()
    acc_sea = acc_sea.cpu().numpy()

    return (
        np.expand_dims(seq_real[n_history:], 0),
        np.expand_dims(seq_pred[n_history:], 0),
        np.expand_dims(valid_loss, 0),
        np.expand_dims(acc, 0),
        np.expand_dims(acc_unweighted, 0),
        np.expand_dims(valid_loss_coarse, 0),
        np.expand_dims(acc_coarse, 0),
        np.expand_dims(acc_coarse_unweighted, 0),
        np.expand_dims(acc_land, 0),
        np.expand_dims(acc_sea, 0),
        )





def hours_to_datetime(hours, start_year, default_timedelta=6):
    """
    Convert hours to a datetime object.

    Args:
        hours (int): Number of hours since the start of the year.
        start_year (int): The starting year for the calculation.
        default_timedelta (int, optional): The default time delta in hours. Defaults to 6.

    Returns:
        datetime: The datetime object representing the calculated date and time.
    """
    total_hours = default_timedelta * 6  # Calculate the total hours based on the default time delta
    days, hours = divmod(hours, 24)  # Calculate the number of days and remaining hours

    start_date = datetime(start_year, 1, 1, 0, 0, 0)  # Create a datetime object for the start of the year
    date = start_date + timedelta(days=days, hours=hours)  # Add the calculated days and hours to the start date

    return date







if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_num', default='00', type=str)
    parser.add_argument('--yaml_config', default="/scratch/gilbreth/gupt1075/FourCastNet_gil/config/AFNO.yaml", type=str)
    
    # defaul full_field vs afno_backbone
    parser.add_argument('--config', default='full_field', type=str)
    parser.add_argument('--use_daily_climatology', action='store_true')
    parser.add_argument("--fld", default="z500", type=str )
    
    parser.add_argument('--vis', action='store_true')
    parser.add_argument('--exp_dir', default=None, type=str, help='Path to store inference outputs; must also set --weights arg')
    parser.add_argument('--interp', default=0, type=float)
    parser.add_argument('--weights', default="/scratch/gilbreth/gupt1075/model_weights/FCN_weights_v0/backbone.ckpt", type=str, help='Path to model weights, for use with exp_dir option')

    args = parser.parse_args()
    params = YParams(os.path.abspath(args.yaml_config), args.config)
    
    params["fld"] = args.fld
    params['world_size'] = 1
    params['interp'] = args.interp
    params['use_daily_climatology'] = args.use_daily_climatology
    params['global_batch_size'] = params.batch_size

    torch.cuda.set_device(0)
    torch.backends.cudnn.benchmark = True
    vis = args.vis



    # fld = 'z500'  # diff flds have diff decor times and hence differnt ics
    if args.fld == 'z500' or args.fld == '2m_temperature' or args.fld == 't850':
        params["DECORRELATION_TIME"] = 36  # 9 days (36) for z500, 2 (8 steps) days for u10, v10
    else:
        params["DECORRELATION_TIME"] = 8  # 9 days (36) for z500, 2 (8 steps) days for u10, v10
    params["idxes"] = {"u10": 0, "z500": 14, "2m_temperature": 2, "v10": 1, "t850": 5}



    # Set up directory

    if args.exp_dir is not None:
        assert args.weights is not None, \
            'Must set --weights argument if using --exp_dir'
    else:
        assert args.weights is None, \
            'Cannot use --weights argument without also using --exp_dir'
    
    expDir = os.path.join(args.exp_dir, args.config, str(args.run_num))

    if not os.path.isdir(expDir):
        os.makedirs(expDir)

    params['experiment_dir'] = os.path.abspath(expDir)
    params['best_checkpoint_path'] = (args.weights if args.exp_dir is not None else os.path.join(expDir, 'training_checkpoints/best_ckpt.tar'))
    params['resuming'] = False
    params['local_rank'] = 0

    logging_utils.log_to_file(logger_name=None, log_filename=os.path.join(expDir, 'inference_out.log'))
    logging_utils.log_versions()
    params.log()

    n_ics = params['n_initial_conditions']

    if args.fld== 'z500' or args.fld == 't850':
        n_samples_per_year = 1336
    else:
        n_samples_per_year = 1460

    if params['ics_type'] == 'default':
        num_samples = n_samples_per_year - params.prediction_length
        stop = num_samples
        ics = np.arange(0, stop, params["DECORRELATION_TIME"])
        if vis:  # visualization for just the first ic (or any ic)
            ics = [0]
        n_ics = len(ics)
        logging.warning(f" \n ICS for default: {ics} num_samples {num_samples}  prediction_lnegth: {params.prediction_length}   ")
        # logging.warning("Inference for {} initial conditions with ics_type {} : current_date {}  and hours_since_jan_01_epoch  {} ".format(n_ics, params["ics_type"],  date_strings, hours_since_jan_01_epoch ))
        # logging.warning(f"{date} {date_obj} {day_of_year} {hour_of_day} {hours_since_jan_01_epoch}")        
    
    elif params['ics_type'] == 'datetime':
        date_strings = params['date_strings']
        ics = []
        if params.perturb:  # for perturbations use a single date and create n_ics perturbations
            n_ics = params['n_perturbations']
            date = date_strings[0]
            date_obj = datetime.strptime(date, '%Y-%m-%d %H:%M:%S')
            day_of_year = date_obj.timetuple().tm_yday - 1
            hour_of_day = date_obj.timetuple().tm_hour
            hours_since_jan_01_epoch = 24 * day_of_year + hour_of_day
            for ii in range(n_ics):
                ics.append(int(hours_since_jan_01_epoch / 6))
        else:
            for date in date_strings:
                date_obj = datetime.strptime(date, '%Y-%m-%d %H:%M:%S')
                day_of_year = date_obj.timetuple().tm_yday - 1
                hour_of_day = date_obj.timetuple().tm_hour
                hours_since_jan_01_epoch = 24 * day_of_year \
                    + hour_of_day
                ics.append(int(hours_since_jan_01_epoch / 6))
        n_ics = len(ics)
        logging.warning(f" #### ICS for datetime: {ics} ")
        logging.warning("Inference for {} initial conditions with ics_type {} : current_date {}  and hours_since_jan_01_epoch  {} ".format(n_ics, params["ics_type"],  date_strings, hours_since_jan_01_epoch ))
        logging.warning(f"{date} {date_obj} {day_of_year} {hour_of_day} {hours_since_jan_01_epoch}")



    # logging.info('Inference for {} initial conditions'.format(n_ics))
    # logging.warning('\n Listed ics {} initial conditions'.format(ics))
    
    try:
        autoregressive_inference_filetag = params['inference_file_tag']
    except:
        autoregressive_inference_filetag = ''

    if params.interp > 0:
        autoregressive_inference_filetag = '_coarse'

    autoregressive_inference_filetag += '_' + args.fld + ''
    if vis:
        autoregressive_inference_filetag += '_vis'

    # get data and models

    (valid_data_full, model, valid_year) = setup(params)

    # initialize lists for image sequences and RMSE/ACC

    valid_loss = []
    valid_loss_coarse = []
    acc_unweighted = []
    acc = []
    acc_coarse = []
    acc_coarse_unweighted = []
    seq_pred = []
    seq_real = []
    acc_land = []
    acc_sea = []

    # run autoregressive inference for multiple initial conditions

    for (i, ic) in enumerate(ics):
        
        date_object = hours_to_datetime(ics[i], 2018)
        logging.warning(f"Initial condition {i+1} of {n_ics} with corresponidng time  =  {date_object}")
        
        (
            sr,
            sp,
            vl,
            a,
            au,
            vc,
            ac,
            acu,
            accland,
            accsea,
            ) = autoregressive_inference(params, ic, valid_data_full,
                    model)


        # Format the date to get the day and month
        date_string = date_object.strftime("%d_%B_%H_%Y")
        
        # with open(f"{expDir}/seq_pred_output_{i}_with_initial_condi_{date_string}.npy", 'wb') as f:
        #     np.save(f, np.squeeze(sp))
        # with open(f"{expDir}/seq_real_output_{i}_datetime_{date_string}.npy", 'wb') as f:
        #     np.save(f, np.squeeze(sr)) 

        logging.warning(f" saved real and predicted with shape {sp.shape} {sr.shape} with np_save {date_string} ")




        if i == 0 or len(valid_loss) == 0:
            seq_real = sr
            seq_pred = sp
            valid_loss = vl
            valid_loss_coarse = vc
            acc = a
            acc_coarse = ac
            acc_coarse_unweighted = acu
            acc_unweighted = au
            acc_land = accland
            acc_sea = accsea
        else:

#        seq_real = np.concatenate((seq_real, sr), 0)
#        seq_pred = np.concatenate((seq_pred, sp), 0)

            valid_loss = np.concatenate((valid_loss, vl), 0)
            valid_loss_coarse = np.concatenate((valid_loss_coarse, vc),
                    0)
            acc = np.concatenate((acc, a), 0)
            acc_coarse = np.concatenate((acc_coarse, ac), 0)
            acc_coarse_unweighted = \
                np.concatenate((acc_coarse_unweighted, acu), 0)
            acc_unweighted = np.concatenate((acc_unweighted, au), 0)
            acc_land = np.concatenate((acc_land, accland), 0)
            acc_sea = np.concatenate((acc_sea, accsea), 0)

    prediction_length = seq_real[0].shape[0]
    n_out_channels = seq_real[0].shape[1]
    img_shape_x = seq_real[0].shape[2]
    img_shape_y = seq_real[0].shape[3]

    # save predictions and loss

    if params.log_to_screen:
        logging.info('Saving files at {}'.format(os.path.join(params['experiment_dir'
                     ], 'autoregressive_predictions'
                     + autoregressive_inference_filetag + '.h5')))
    with h5py.File(os.path.join(params['experiment_dir'],
                   'autoregressive_predictions'
                   + autoregressive_inference_filetag + '.h5'), 'a') as \
        f:
        if vis:
            try:
                f.create_dataset('ground_truth', data=seq_real,
                                 shape=(n_ics, prediction_length,
                                 n_out_channels, img_shape_x,
                                 img_shape_y), dtype=np.float32)
            except:
                del f['ground_truth']
                f.create_dataset('ground_truth', data=seq_real,
                                 shape=(n_ics, prediction_length,
                                 n_out_channels, img_shape_x,
                                 img_shape_y), dtype=np.float32)
                f['ground_truth'][...] = seq_real

            try:
                f.create_dataset('predicted', data=seq_pred,
                                 shape=(n_ics, prediction_length,
                                 n_out_channels, img_shape_x,
                                 img_shape_y), dtype=np.float32)
            except:
                del f['predicted']
                f.create_dataset('predicted', data=seq_pred,
                                 shape=(n_ics, prediction_length,
                                 n_out_channels, img_shape_x,
                                 img_shape_y), dtype=np.float32)
                f['predicted'][...] = seq_pred

        if params.masked_acc:
            try:
                f.create_dataset('acc_land', data=acc_land)  # , shape = (n_ics, prediction_length, n_out_channels), dtype =np.float32)
            except:
                del f['acc_land']
                f.create_dataset('acc_land', data=acc_land)  # , shape = (n_ics, prediction_length, n_out_channels), dtype =np.float32)
                f['acc_land'][...] = acc_land

            try:
                f.create_dataset('acc_sea', data=acc_sea)  # , shape = (n_ics, prediction_length, n_out_channels), dtype =np.float32)
            except:
                del f['acc_sea']
                f.create_dataset('acc_sea', data=acc_sea)  # , shape = (n_ics, prediction_length, n_out_channels), dtype =np.float32)
                f['acc_sea'][...] = acc_sea

        try:
            f.create_dataset('rmse', data=valid_loss, shape=(n_ics,
                             prediction_length, n_out_channels),
                             dtype=np.float32)
        except:
            del f['rmse']
            f.create_dataset('rmse', data=valid_loss, shape=(n_ics,
                             prediction_length, n_out_channels),
                             dtype=np.float32)
            f['rmse'][...] = valid_loss

        try:
            f.create_dataset('acc', data=acc, shape=(n_ics,
                             prediction_length, n_out_channels),
                             dtype=np.float32)
        except:
            del f['acc']
            f.create_dataset('acc', data=acc, shape=(n_ics,
                             prediction_length, n_out_channels),
                             dtype=np.float32)
            f['acc'][...] = acc

        try:
            f.create_dataset('rmse_coarse', data=valid_loss_coarse,
                             shape=(n_ics, prediction_length,
                             n_out_channels), dtype=np.float32)
        except:
            del f['rmse_coarse']
            f.create_dataset('rmse_coarse', data=valid_loss_coarse,
                             shape=(n_ics, prediction_length,
                             n_out_channels), dtype=np.float32)
            f['rmse_coarse'][...] = valid_loss_coarse

        try:
            f.create_dataset('acc_coarse', data=acc_coarse,
                             shape=(n_ics, prediction_length,
                             n_out_channels), dtype=np.float32)
        except:
            del f['acc_coarse']
            f.create_dataset('acc_coarse', data=acc_coarse,
                             shape=(n_ics, prediction_length,
                             n_out_channels), dtype=np.float32)
            f['acc_coarse'][...] = acc_coarse

        try:
            f.create_dataset('acc_unweighted', data=acc_unweighted,
                             shape=(n_ics, prediction_length,
                             n_out_channels), dtype=np.float32)
        except:
            del f['acc_unweighted']
            f.create_dataset('acc_unweighted', data=acc_unweighted,
                             shape=(n_ics, prediction_length,
                             n_out_channels), dtype=np.float32)
            f['acc_unweighted'][...] = acc_unweighted

        try:
            f.create_dataset('acc_coarse_unweighted',
                             data=acc_coarse_unweighted, shape=(n_ics,
                             prediction_length, n_out_channels),
                             dtype=np.float32)
        except:
            del f['acc_coarse_unweighted']
            f.create_dataset('acc_coarse_unweighted',
                             data=acc_coarse_unweighted, shape=(n_ics,
                             prediction_length, n_out_channels),
                             dtype=np.float32)
            f['acc_coarse_unweighted'][...] = acc_coarse_unweighted

        f.close()
