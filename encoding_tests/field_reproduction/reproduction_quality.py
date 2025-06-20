# -*- coding: utf-8 -*-
#!/usr/bin/python
"""
Script to qualitatively compare the quality of NN outputs with their targets
"""
# Osnovno
import os
import time

import matplotlib.pyplot as plt
import matplotlib
import cartopy.crs as ccrs
import numpy as np
import torch
import torch.nn as nn

import sys
master_dir = os.getenv('UGNN3DVar_master')
sys.path.append(master_dir)
sys.path.append(master_dir + '/NNs')
import predict_1hr, predict_1hr_AE
import unet_model_7x7_4x_depth7_interdepth_padding_1_degree_v01
import unet_model_7x7_4x_depth4_interdepth_padding_1_degree_v01_stride2b_no_skip_AE


import pickle
from datetime import datetime, timedelta
import warnings



# --------------------------------------------------------------------------
# NALAGANJE POMOZNIH STVARI, KI JIH RABIMO ZA POGANJANJE NN
# --------------------------------------------------------------------------
from loading_and_plotting_data import create_lists, ERA5_loader, plotting_stuff, plot_Earth
from loading_and_plotting_data import encoded_forecast_provider, EDA_ensemble_member_load

# Ustvarim vse tabele
tabele = create_lists()

# Poti do datotek
ANNOTATIONS_NAMES = tabele["ANNOTATIONS_NAMES"]

# Scalerji
root_scaler = tabele["root_scaler"]
SCALER_NAMES = tabele["SCALER_NAMES"]
SCALER_TYPES = tabele["SCALER_TYPES"]



# Imena spremenljivk za shranjevanje in naslove slik
VARIABLES_SAVE = tabele["VARIABLES_SAVE"]

# Multiplicators for plotting the fields (and rescaling the observation increments and standard deviations)
field_adjustment_multpilicators = plotting_stuff()['FIELD_ADJUSTEMENT_MULTIPLICATION']

AE_root_model = '%s/NNs/models/autoencoder_20_12100' % os.getenv('UGNN3DVar_master')
in_out_ch = 20
AE_model = unet_model_7x7_4x_depth4_interdepth_padding_1_degree_v01_stride2b_no_skip_AE.UNet(
    in_channels=1 * in_out_ch + 3, out_channels=in_out_ch, depth=4, start_filts=50,
    up_mode='transpose', merge_mode='concat')
AE_latent_space_shape = (1, 50, 11, 22)
model_name = "trained_model_weights.pth"

# Poskrbi za ustrezno delovanje tako na CPU, kot GPU
device_ids = [0, 1]
AE_model = nn.DataParallel(AE_model, device_ids=device_ids)

# Tabela s potjo do AE modela
AE_ROOTS_MODEL = [AE_root_model]
# Tabela s AE modelom
AE_MODELS = [AE_model]
AE_predicting = predict_1hr_AE.Predicting(AE_ROOTS_MODEL, model_name)

FWD_root_model = '%s/NNs/models/U-Net_IT_ALL_numpred2_train_sumMSE_testACC_isl1_250ch_ks7x7_1hr_parallel_2024_11' % os.getenv(
        'UGNN3DVar_master')+

# Število korakov napovedi (če ni izbran samokodirnik, sicer pa število preslikav)
forecast_len = 24



FWD_start_datetime = datetime(2020, 4, 14, 00)
FWD_end_datetime = FWD_start_datetime + timedelta(hours=forecast_len)
start_datetime_str = datetime.strftime(FWD_start_datetime, '%Y-%m-%d-%H')
end_datetime_str = datetime.strftime(FWD_end_datetime, '%Y-%m-%d-%H')


ERA5_ground_truth_start_time = ERA5_loader(FWD_start_datetime.strftime('%Y-%m-%d-%H'), return_option='torch tensor').numpy() * np.expand_dims(field_adjustment_multpilicators, (1,2,3))
ERA5_ground_truth_end_time = ERA5_loader(FWD_end_datetime.strftime('%Y-%m-%d-%H'), return_option='torch tensor').numpy() * np.expand_dims(field_adjustment_multpilicators, (1,2,3))


AE_latent_vec_start_time = encoded_forecast_provider(
            FWD_root_model='persistence',
            AE_root_model=AE_root_model,
            keep_3D=False,
            initial_conditions='ERA5',
            forecast_start_datetime=FWD_start_datetime,
            forecast_end_datetime=FWD_end_datetime
        )

with torch.no_grad():
    AE_decoded_start_time = AE_predicting.decode(
                                ANNOTATIONS_NAMES,
                                AE_MODELS,
                                root_scaler,
                                SCALER_NAMES,
                                SCALER_TYPES,
                                num_of_prediction_steps=1,
                                z=torch.from_numpy(AE_latent_vec_start_time.reshape(AE_latent_space_shape))    # perturbed_AE_latent_background is a list of lists of numpy arrays
                                )   # shape (in_out_ch, 2, 180, 360), [:,0,:,:] is useless

AE_start_time = np.zeros(shape=(in_out_ch, 180, 360))
for ivar in range(in_out_ch):
    AE_start_time[ivar] = AE_decoded_start_time[ivar][1].squeeze()
AE_start_time = AE_start_time * np.expand_dims(field_adjustment_multpilicators, (1,2))


FWD_end_pred = encoded_forecast_provider(
            FWD_root_model=FWD_root_model,
            AE_root_model=AE_root_model,
            keep_3D=False,
            initial_conditions='ERA5',
            forecast_start_datetime=FWD_start_datetime,
            forecast_end_datetime=FWD_end_datetime,
            return_full_NNfwd_prediction=True
        )

FWD_end_pred = np.array(FWD_end_pred) * np.expand_dims(field_adjustment_multpilicators, (1,2,3))
FWD_end_pred = FWD_end_pred.squeeze()


# PLOTTING SETTINGS

# Which quantities do I plot?
plot_qty = ['all']  # quantities
# if 'all' in plot_qty:  # plot all in_out_ch fields
plot_qty_idx = [i for i in range(in_out_ch)]

# Plotting settings
num_fields = len(plot_qty_idx)  # Number of fields to plot
max_num_fields = in_out_ch  # Highest possible number of fields to plot
single_fig_width = 10  # Width of single figure (in inches, I think)
qty_all = VARIABLES_SAVE
# print(ED_ensemble_members_std.shape[0])
unit = [' [' + plotting_stuff()['ENOTE'][i] + ']' for i in
        range(in_out_ch)]  # How to display units

fig = plt.figure(figsize=(single_fig_width * num_fields, 60))
# matplotlib.rcParams.update({"font.size": 16})
matplotlib.rcParams.update({"font.size": 26})

projections = {'Orthographic': ccrs.Orthographic(),
               'Robinson': ccrs.Robinson(),
               'PlateCarree': ccrs.PlateCarree(),
               }
projection = 'PlateCarree'

plot_left = 0.075 / num_fields
plot_width = 0.85 / num_fields
tot_plots = 5  # Set the number of rows
height_sum = tot_plots * 0.095
dheight_plot = 0.05 / height_sum  # 0.075 / height_sum
dheight_buffer = -0.01 / height_sum  # 0.005 / height_sum
plot_bottom = 0.98  # 1 # Initial value for plot_bottom (it will gradually drop to 0)

cbar_shrink = 0.5

cmaps = plotting_stuff()['cmaps']  # Colormaps
cmaps_inc = plotting_stuff()['cmaps_inc']  # Colormaps
vminmax_std = plotting_stuff()[
    'vminmax_std']  # Ranges for all available quantities when plotting std
vminmax_std = (np.array(vminmax_std) * 0.1).tolist()
vminmax = plotting_stuff()['vminmax']




plot_bottom -= dheight_plot + dheight_buffer

ifieldplot = 0
for ifield in range(max_num_fields):
    if ifield in plot_qty_idx:
        ax_ERA5s = fig.add_axes([plot_left + ifieldplot / num_fields, plot_bottom, plot_width, dheight_plot], projection=projections[projection])
        ifieldplot += 1
        plot_Earth(
            fig=fig,
            ax=ax_ERA5s,
            field=ERA5_ground_truth_start_time[ifield, 0],
            vminmax=vminmax[ifield],
            cmap=cmaps[ifield],
            cbar_extend='both',
            cbar_shrink=cbar_shrink,
            title=r'ERA5 on %s (%s%s)' % (start_datetime_str, qty_all[ifield], unit[ifield])
        )


plot_bottom -= dheight_plot + dheight_buffer

ifieldplot = 0
for ifield in range(max_num_fields):
    if ifield in plot_qty_idx:
        ax_AE = fig.add_axes([plot_left + ifieldplot / num_fields, plot_bottom, plot_width, dheight_plot], projection=projections[projection])
        ifieldplot += 1
        plot_Earth(
            fig=fig,
            ax=ax_AE,
            field=AE_start_time[ifield],
            vminmax=vminmax[ifield],
            cmap=cmaps[ifield],
            cbar_extend='both',
            cbar_shrink=cbar_shrink,
            title=r'Reconstruction with AE (%s%s)' % (qty_all[ifield], unit[ifield])
        )

plt.savefig(f'reproduction_quality_AE_{start_datetime_str}.jpg', dpi=100)




fig = plt.figure(figsize=(single_fig_width * num_fields, 60))
matplotlib.rcParams.update({"font.size": 26})

projections = {'Orthographic': ccrs.Orthographic(),
               'Robinson': ccrs.Robinson(),
               'PlateCarree': ccrs.PlateCarree(),
               }
projection = 'PlateCarree'

plot_left = 0.075 / num_fields
plot_width = 0.85 / num_fields
tot_plots = 5  # Set the number of rows
height_sum = tot_plots * 0.095
dheight_plot = 0.05 / height_sum  # 0.075 / height_sum
dheight_buffer = -0.01 / height_sum  # 0.005 / height_sum
plot_bottom = 0.98  # 1 # Initial value for plot_bottom (it will gradually drop to 0)

cbar_shrink = 0.5

plot_bottom -= dheight_plot + dheight_buffer

ifieldplot = 0
for ifield in range(max_num_fields):
    if ifield in plot_qty_idx:
        ax_ERA5e = fig.add_axes([plot_left + ifieldplot / num_fields, plot_bottom, plot_width, dheight_plot], projection=projections[projection])
        ifieldplot += 1
        plot_Earth(
            fig=fig,
            ax=ax_ERA5e,
            field=ERA5_ground_truth_end_time[ifield, 0],
            vminmax=vminmax[ifield],
            cmap=cmaps[ifield],
            cbar_extend='both',
            cbar_shrink=cbar_shrink,
            title=r'ERA5 on %s (%s%s)' % (end_datetime_str, qty_all[ifield], unit[ifield])
        )

plot_bottom -= dheight_plot + dheight_buffer


ifieldplot = 0
for ifield in range(max_num_fields):
    if ifield in plot_qty_idx:
        ax_fwd = fig.add_axes([plot_left + ifieldplot / num_fields, plot_bottom, plot_width, dheight_plot], projection=projections[projection])
        ifieldplot += 1
        plot_Earth(
            fig=fig,
            ax=ax_fwd,
            field=FWD_end_pred[ifield],
            vminmax=vminmax[ifield],
            cmap=cmaps[ifield],
            cbar_extend='both',
            cbar_shrink=cbar_shrink,
            title=r'%dh NN forecast (%s%s)' % (forecast_len, qty_all[ifield], unit[ifield])
        )

plt.savefig(f'reproduction_quality_NNfwd_{forecast_len}h_{start_datetime_str}.jpg', dpi=100)