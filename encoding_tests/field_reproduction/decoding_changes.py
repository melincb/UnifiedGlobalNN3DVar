# -*- coding: utf-8 -*-
#!/usr/bin/python
"""
Script for checking the changes after the decoding, if we perturb a single latent vector element
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
import predict_1hr_AE
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
root_annotations = tabele["root_annotations"]
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
name_preposition = 'AE_20_12100'
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



# MAIN SETTNGS
datetime_strs = ['2020-04-15-00',  '2024-04-14-00', '2020-04-15-00', '2020-04-15-00', '2020-04-15-00']
pertudb_indices = [4786, 4786, 8878, 9626, 4233]

perturb_size = 1


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
# vminmax_std = (np.array(vminmax_std) * 0.1).tolist()
vminmax = plotting_stuff()['vminmax']
vminmax_inc = plotting_stuff()[
    'vminmax_inc']  # Ranges for all available quantities when plotting the analysis increment, if the quantity of interest is not observed






for i in range(len(datetime_strs)):
    datetime_str = datetime_strs[i]
    print(datetime_str)
    perturb_idx = pertudb_indices[i]


    if datetime_str == '2020-04-15-00':
        FWD_end_datetime = datetime.strptime(datetime_str, '%Y-%m-%d-%H')
        FWD_end_date = (FWD_end_datetime.day, FWD_end_datetime.month, FWD_end_datetime.year)
        FWD_end_time = FWD_end_datetime.hour

        FWD_start_datetime = FWD_end_datetime - timedelta(hours=24)
        FWD_start_date = (FWD_start_datetime.day, FWD_start_datetime.month, FWD_start_datetime.year)
        FWD_start_time = FWD_start_datetime.hour

        FWD_root_model = '%s/NNs/models/U-Net_IT_ALL_numpred2_train_sumMSE_testACC_isl1_250ch_ks7x7_1hr_parallel_2024_11' % os.getenv(
            'UGNN3DVar_master')

        orig_AE_latent_vec = encoded_forecast_provider(
            FWD_root_model=FWD_root_model,
            AE_root_model=AE_root_model,
            keep_3D=False,
            initial_conditions='ERA5',
            forecast_start_datetime=FWD_start_datetime,
            forecast_end_datetime=FWD_end_datetime
        )



    elif datetime_str == '2024-04-14-00':

        ensemble = 1
        included_members_lst = [1]
        EDA_ensemble_type = 'forecasts'

        if datetime_str != '2024-04-14-00':
            warnings.warn(f"WARNING: You are using True_EDA, but your observation datetime is {datetime_str} and not {'2024-04-14-00'}!")
        if ensemble != 50:
            warnings.warn(
                f"WARNING: You are using True_EDA, but your ensemble size is {ensemble} and not {50}!")

        if EDA_ensemble_type == 'forecasts':
            ensemble_main_dir_ext = 'ens_B'
        elif EDA_ensemble_type == 'analyses':
            ensemble_main_dir_ext = 'ens_an'
        else:
            print('Unsupported EDA_ensemble_type:', EDA_ensemble_type)
            raise AttributeError  # Unsupported ensemble type


        encoded_ensemble_member = [
            EDA_ensemble_member_load(
                which_ensemble=EDA_ensemble_type,
                em_idx=em_idx,
                datetime_str=datetime_str,
                return_encoded=True,
                echo=False
            )#.astype(np.float64)
            for em_idx in included_members_lst
        ][0]

        # This time we do not perturb the ensemble members, we leave them as they are!
        orig_AE_latent_vec = np.array(encoded_ensemble_member)
        print(orig_AE_latent_vec.shape)


    perturbed_AE_latent_vec = orig_AE_latent_vec.copy()
    print(f'Value of {perturb_idx} before perturbation', perturbed_AE_latent_vec[0, perturb_idx])
    perturbed_AE_latent_vec[0, perturb_idx] = perturbed_AE_latent_vec[0, perturb_idx] + perturb_size
    with torch.no_grad():
        AE_preds_orig = AE_predicting.decode(
                            ANNOTATIONS_NAMES,
                            AE_MODELS,
                            root_scaler,
                            SCALER_NAMES,
                            SCALER_TYPES,
                            num_of_prediction_steps=1,
                            input_sequence_len=1,
                            averaging_sequence_len=1,
                            z=torch.from_numpy(orig_AE_latent_vec.reshape(AE_latent_space_shape))    # perturbed_AE_latent_background is a list of lists of numpy arrays
                            )   # shape (in_out_ch, 2, 180, 360), [:,0,:,:] is useless

        ED_orig = np.zeros(shape=(in_out_ch, 180, 360))
        for ivar in range(in_out_ch):
            ED_orig[ivar] = AE_preds_orig[ivar][1].squeeze()
        ED_orig = ED_orig * np.expand_dims(field_adjustment_multpilicators, (1,2))

        AE_preds_perturb = AE_predicting.decode(
                            ANNOTATIONS_NAMES,
                            AE_MODELS,
                            root_scaler,
                            SCALER_NAMES,
                            SCALER_TYPES,
                            num_of_prediction_steps=1,
                            input_sequence_len=1,
                            averaging_sequence_len=1,
                            z=torch.from_numpy(perturbed_AE_latent_vec.reshape(AE_latent_space_shape))    # perturbed_AE_latent_background is a list of lists of numpy arrays
                            )   # shape (in_out_ch, 2, 180, 360), [:,0,:,:] is useless

        ED_perturb = np.zeros(shape=(in_out_ch, 180, 360)) * np.expand_dims(field_adjustment_multpilicators, (1,2))
        for ivar in range(in_out_ch):
            ED_perturb[ivar] = AE_preds_perturb[ivar][1].squeeze()
        ED_perturb = ED_perturb * np.expand_dims(field_adjustment_multpilicators, (1,2))



    plot_bottom -= dheight_plot + dheight_buffer

    if datetime_str == '2024-04-14-00':
        which_data = 'IFS'
    else:
        which_data = 'ERA5'

    ifieldplot = 0
    for ifield in range(max_num_fields):
        if ifield in plot_qty_idx:
            ax_diff = fig.add_axes([plot_left + ifieldplot / num_fields, plot_bottom, plot_width, dheight_plot], projection=projections[projection])
            ifieldplot += 1
            plot_Earth(
                fig=fig,
                ax=ax_diff,
                field=ED_perturb[ifield] - ED_orig[ifield],
                vminmax=vminmax_inc[ifield],
                cmap=cmaps_inc[ifield],
                cbar_extend='both',
                cbar_shrink=cbar_shrink,
                title=r'$\mathbf{z}_{%d}^{\mathrm{%s}}$' % (perturb_idx, which_data) + f' + {perturb_size}   ({qty_all[ifield]}{unit[ifield]})'
            )


plt.savefig('multpiple_options.jpg', dpi=100)