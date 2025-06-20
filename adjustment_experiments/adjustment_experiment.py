# -*- coding: utf-8 -*-
#!/usr/bin/python
"""
Script for preparing the inputs to 3D-Var algorithm

To run it please use experiments/run_3D-Var.sh!
"""

# --------------------------------------------------------------------------
# OSNOVNE KNJIZNICE
# enako kot v zagon_AE.ipynb, le s prilagojeno potjo do modulov
# --------------------------------------------------------------------------

# Osnovno
import os
import re
import time

import numpy as np
import torch
import torch.nn as nn


# Slike
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from datetime import datetime, timedelta
import matplotlib

# Moduli
import sys
master_dir = os.getenv('UGNN3DVar_master')
sys.path.append(master_dir)
sys.path.append(master_dir + '/NNs')
import predict_1hr, predict_1hr_AE
import data_preparation_1hr
import unet_model_7x7_4x_depth7_interdepth_padding_1_degree_v01
import unet_model_7x7_4x_depth4_interdepth_padding_1_degree_v01_stride2b_no_skip_AE

# --------------------------------------------------------------------------
# DODATNE KNJIZNICE + nastavitve
# --------------------------------------------------------------------------
import argparse
import pickle


idx = 1
experiment_source_general = '../algorithm/experiments/data/20ch/'
#experiment_source_detail = 'data_singobs_Ljubljana_SGD_endtime_2020-04-15-00_FWD_model=NNfwd_dt=24h_obs_qty=Z500_obs_inc=30.0_obs_std=10.0_B_type=diagonal_ens=100_init_lr=0.3_prec3DVar.pkl'
experiment_source_detail = 'data_singobs_Central_Atlantic_SGD_endtime_2020-04-15-00_FWD_model=NNfwd_dt=24h_obs_qty=TCWV_obs_inc=10.0_obs_std=3.0_B_type=diagonal_ens=100_init_lr=0.3_prec3DVar.pkl'
experiment_source = experiment_source_general + experiment_source_detail

FWD_model = 'NNfwd'
new_forecast_len = 48#72
assert FWD_model in experiment_source    # FWD models should match

datetime_str = re.search(r"endtime_(\d{4}-\d{2}-\d{2}-\d{2})", experiment_source).group(1)
FWD_end_datetime_previous = datetime.strptime(datetime_str, "%Y-%m-%d-%H")
forecast_len = int(re.search(r"dt=(\d+)h_", experiment_source).group(1))
FWD_start_datetime_previous = FWD_end_datetime_previous - timedelta(hours=forecast_len)
FWD_start_datetime_next = FWD_end_datetime_previous
FWD_end_datetime_next = FWD_start_datetime_next + timedelta(hours=24)
print(FWD_start_datetime_previous, FWD_end_datetime_next)


# --------------------------------------------------------------------------
# 1) Loading results from assimilation experiment
# --------------------------------------------------------------------------
print('Loading results from assimilation experiment')
print('Data source:', experiment_source)

dict_to_load = pickle.load(open(experiment_source, 'rb'))

# Extracting latent background and analysis
print('Extracting latent background and analysis at idx', idx)
AE_latent_analysis = dict_to_load['AE_latent_out'][idx].to(torch.float32)  # Torch tensor - Outputs in the latent space
AE_latent_background = dict_to_load['AE_latent_background'][idx]  # Torch tensor - Perturbed (transposed) latent vectors (for background)

# Converting analysis and background (of chosen idx) to gp space


# --------------------------------------------------------------------------
# NALAGANJE POMOZNIH STVARI, KI JIH RABIMO ZA POGANJANJE NN
# --------------------------------------------------------------------------
from loading_and_plotting_data import create_lists, ERA5_loader, plotting_stuff, plot_Earth
from loading_and_plotting_data import encoded_forecast_provider

# Ustvarim vse tabele
tabele = create_lists()

# Poti do datotek
root_annotations = tabele["root_annotations"]
ANNOTATIONS_NAMES = tabele["ANNOTATIONS_NAMES"]

# Scalerji
root_scaler = tabele["root_scaler"]
SCALER_NAMES = tabele["SCALER_NAMES"]
SCALER_TYPES = tabele["SCALER_TYPES"]


# Pot do statičnih polj
STATIC_FIELDS_ROOTS = tabele["STATIC_FIELDS_ROOTS"]

# Imena spremenljivk za shranjevanje in naslove slik
VARIABLES_SAVE = tabele["VARIABLES_SAVE"]

# Multiplicators for plotting the fields (and rescaling the observation increments and standard deviations)
field_adjustment_multpilicators = plotting_stuff()['FIELD_ADJUSTEMENT_MULTIPLICATION']


# --------------------------------------------------------------------------
# NALAGANJE OBEH MODELOV
# --------------------------------------------------------------------------
# # FWD
if FWD_model == 'NNfwd':
    in_out_ch = 20
    FWD_root_model = '%s/NNs/models/U-Net_IT_ALL_numpred2_train_sumMSE_testACC_isl1_250ch_ks7x7_1hr_parallel_2024_11' % os.getenv(
        'UGNN3DVar_master')
    fwd_model = unet_model_7x7_4x_depth7_interdepth_padding_1_degree_v01.UNet(
        in_channels=1 * in_out_ch + 3, out_channels=in_out_ch, depth=7, start_filts=250,
        up_mode='transpose', merge_mode='concat')

    # Tabela s potjo do  fwd modela
    FWD_ROOTS_MODEL = [FWD_root_model]
    # Tabela s fwd modelom
    FWD_MODELS = [fwd_model]
    # Ime modela - to je fiksno ("trained_model_weights.pth" vsebuje uteži z najboljšim ACC med treningom)
    model_name = "trained_model_weights.pth"

    # Število korakov napovedi (če ni izbran samokodirnik, sicer pa število preslikav)
    num_of_prediction_steps = new_forecast_len // 12  # This fwd model has 12h steps
    single_step_len = 12
    # Število vhodnih časovnih instanc - to je fiksno na 1
    input_sequence_length = 1
    # Število povprečenih časovnih instanc - to je fiksno na 1
    averaging_sequence_length = 1
else:
    print(f'Oops! I forgot to allow {FWD_model} in the part of my code where I set FWD_root_model!')
    raise AttributeError

FWD_predicting = predict_1hr.Predicting(FWD_ROOTS_MODEL, model_name)

# AE
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



# --------------------------------------------------------------------------
# Decode background and analysis
# --------------------------------------------------------------------------

with torch.no_grad():
    background_gp = np.zeros(shape=(in_out_ch, 1, 180, 360),
                                    dtype=np.float32)  # shape: (variables, 1, lats, lons)

    AE_preds = AE_predicting.decode(
        ANNOTATIONS_NAMES,
        AE_MODELS,
        root_scaler,
        SCALER_NAMES,
        SCALER_TYPES,
        num_of_prediction_steps=1,
        z=AE_latent_background.reshape(AE_latent_space_shape)    # AE_latent_background is already a torch tensor
    )  # shape (in_out_ch, 2, 180, 360), [:,0,:,:] is useless
    for ivar in range(background_gp.shape[0]):  # go through all in_out_ch variables
        background_gp[ivar] = AE_preds[ivar][1]

    print('background_gp.dtype', background_gp.dtype)

    analysis_gp = np.zeros(shape=(in_out_ch, 1, 180, 360),
                                dtype=np.float32)  # shape: (variables, 1, lats, lons)
    for izb in range(len(AE_latent_background)):
        AE_preds = AE_predicting.decode(
            ANNOTATIONS_NAMES,
            AE_MODELS,
            root_scaler,
            SCALER_NAMES,
            SCALER_TYPES,
            num_of_prediction_steps=1,
            z=AE_latent_analysis.reshape(AE_latent_space_shape)  # AE_latent_analysis is already a torch tensor
        )  # shape (in_out_ch, 2, 180, 360), [:,0,:,:] is useless
        for ivar in range(analysis_gp.shape[0]):  # go through all in_out_ch variables
            analysis_gp[ivar] = AE_preds[ivar][1]




# --------------------------------------------------------------------------
# Compute forecast using both background and analysis as initial condition
# --------------------------------------------------------------------------
with torch.no_grad():
    print('BG!')
    _, preds_bg = FWD_predicting.pred_time_series(
        root_annotations,
        ANNOTATIONS_NAMES,
        FWD_MODELS,
        STATIC_FIELDS_ROOTS,
        root_scaler,
        SCALER_NAMES,
        SCALER_TYPES,
        (FWD_start_datetime_next.day, FWD_start_datetime_next.month, FWD_start_datetime_next.year),
        num_of_prediction_steps=num_of_prediction_steps,
        input_sequence_len=1,
        averaging_sequence_len=1,
        T=FWD_start_datetime_next.hour,
        use_ERA5=False,
        list_of_input_fields=background_gp.tolist()
    )
    print('ANA!')
    _, preds_ana = FWD_predicting.pred_time_series(
        root_annotations,
        ANNOTATIONS_NAMES,
        FWD_MODELS,
        STATIC_FIELDS_ROOTS,
        root_scaler,
        SCALER_NAMES,
        SCALER_TYPES,
        (FWD_start_datetime_next.day, FWD_start_datetime_next.month, FWD_start_datetime_next.year),
        num_of_prediction_steps=num_of_prediction_steps,
        input_sequence_len=1,
        averaging_sequence_len=1,
        T=FWD_start_datetime_next.hour,
        use_ERA5=False,
        list_of_input_fields=analysis_gp.tolist()
    )


preds_bg = np.array(preds_bg)
preds_ana = np.array(preds_ana)

bg_orig = background_gp * np.expand_dims(field_adjustment_multpilicators, (1,2,3))
ana_orig = analysis_gp * np.expand_dims(field_adjustment_multpilicators, (1,2,3))
bg_prop = preds_bg * np.expand_dims(field_adjustment_multpilicators, (1,2,3))
ana_prop = preds_ana * np.expand_dims(field_adjustment_multpilicators, (1,2,3))

ana_inc_orig = ana_orig - bg_orig
ana_inc_prop = ana_prop - bg_prop

plot_qty_idx = [i for i in range(bg_orig.shape[0])]


# Plotting settings
qty_all = VARIABLES_SAVE
unit = [' [' + plotting_stuff()['ENOTE'][i] + ']' for i in range(bg_orig.shape[0])]    # How to display units


matplotlib.rcParams.update({"font.size": 16})

projections = {'Orthographic': ccrs.Orthographic(),
                   'Robinson': ccrs.Robinson(),
                   'PlateCarree': ccrs.PlateCarree()
               }
projection = 'PlateCarree'



cmaps = plotting_stuff()['cmaps'] # Colormaps
cmaps_inc = plotting_stuff()['cmaps_inc'] # Colormaps
vminmax = plotting_stuff()['vminmax'] # Ranges for all available quantities
vminmax_std = plotting_stuff()['vminmax_std'] # Ranges for all available quantities when plotting std
vminmax_inc = plotting_stuff()['vminmax_inc'] # Ranges for all available quantities when plotting the analysis increment, if the quantity of interest is not observed



if 'TCWV' in experiment_source:
    tsidx = -1
    for timestep in [0, 1, 2, 4]:  # 0h, 24h, 48h
        tsidx += 1
        diff_U200 = ana_inc_prop[qty_all.index('U200'), timestep]
        diff_V200 = ana_inc_prop[qty_all.index('V200'), timestep]
        diff_U900 = ana_inc_prop[qty_all.index('U900'), timestep]
        diff_V900 = ana_inc_prop[qty_all.index('V900'), timestep]
        diff_TCWV = ana_inc_prop[qty_all.index('TCWV'), timestep]

        corilats = [[ilat for ilon in np.arange(-180., 179. + 1e-5, step=1.)] for ilat in
                    np.arange(-89.5, 89.5 + 1e-5, step=1.)]
        coripar = 2 * 2 * np.pi / (24 * 60 * 60) * np.radians(corilats)

        dlat = 110.6e3  # 1 degree lat in meters
        a = np.cos(np.radians(corilats)) ** 2 * np.sin(np.radians(1) / 2) ** 2
        dlon = 6.357e6 * 2 * np.arctan(np.sqrt(a) / np.sqrt(1 - a))

        derivative_V900_lat = np.gradient(diff_V900, dlat)[
            0]  # Tested the correctness of index 0 on a simple case (gradient of corilats)
        derivative_U900_lon = [np.gradient(diff_U900[ilat], dlon[ilat, 0]) for ilat in
                               range(len(diff_U900))]

        div_900 = derivative_V900_lat + np.array(derivative_U900_lon)

        derivative_V200_lat = np.gradient(diff_V200, dlat)[
            0]  # Tested the correctness of index 0 on a simple case (gradient of corilats)
        derivative_U200_lon = [np.gradient(diff_U200[ilat], dlon[ilat, 0]) for ilat in
                               range(len(diff_U200))]

        div_200 = derivative_V200_lat + np.array(derivative_U200_lon)


        fig2 = plt.figure(figsize=(6*1.5*0.85, 4*1.5*0.85)) # * 2 * 0.9
        ax2 = fig2.add_subplot(1, 1, 1, projection=ccrs.Miller(central_longitude=-33))
        plot_Earth(
            fig=fig2,
            ax=ax2,
            field=diff_TCWV,
            U=bg_prop[qty_all.index('U900'), timestep],
            V=bg_prop[qty_all.index('V900'), timestep],
            quiver_scale=2.5e2,
            quiver_reduction=3,
            quiverkey_props=[1.15, 0.06, 10], #1.125
            vminmax=vminmax_inc_no_obs[qty_all.index('TCWV')],
            cmap=cmaps_inc[qty_all.index('TCWV')],
            cbar_extend='both',
            cbar_shrink=0.60,
            unit='%s%s' % ('TCWV', unit[qty_all.index('TCWV')]),
            unit_in_cbar=True,
            # obs_locs=obs_lons_lats,
            title='Backg. U900 and V900, diff. in %s' % ('TCWV'),
            # title='%s after %dh' % ('TCWV', timestep * single_step_len),
            coordinate_labels=True,
            gridline_y_distance=20
        )
        ax2.set_extent([-20, 40, -35, 35], crs=ccrs.PlateCarree())
        # print(ax2.get_xlim())
        ax2.set_xlim([-4500000, 6500000])
        # ax2.tight_layout()
        fig2.savefig(
            f'figures/{in_out_ch}ch/{experiment_source_detail[:-4]}_idx_{idx}_prop_TCWV_{timestep * single_step_len}h' + '.jpg',
            dpi=300)
        print('saved prop ZUV500', timestep * single_step_len)



        fig2 = plt.figure(figsize=(6*1.5*0.85, 4*1.5*0.85)) # * 2 * 0.9
        ax2 = fig2.add_subplot(1, 1, 1, projection=ccrs.Miller(central_longitude=-33))
        plot_Earth(
            fig=fig2,
            ax=ax2,
            field=ana_inc_prop[qty_all.index('mslp'), timestep] * 0, #div_900 * 1e6,
            U=diff_U900,
            V=diff_V900,
            quiver_scale=1.5e1,
            quiver_reduction=3,
            quiverkey_props=[1.17, 0.06, 0.5], #1.125
            vminmax=[-1, 1],
            cmap='bwr',#cmaps_inc[qty_all.index('mslp')],
            cbar=True,
            unit_in_cbar=True,
            cbar_ticks=[-1,0,1],
            # cbar_extend='both',
            cbar_shrink=0.6,
            unit='To delete',#,'%s%s' % ('MSLP', unit[qty_all.index('mslp')]),
            # unit_in_cbar=True,
            # obs_locs=obs_lons_lats,
            title='Diff. in %s, %s, and %s' % ('U900', 'V900', 'MSLP'),
            # title='%s, %s, and %s after %dh' % ('U900', 'V900', 'MSLP', timestep * single_step_len),
            coordinate_labels=True,
            gridline_y_distance=20,
            extra_contour_field=ana_inc_prop[qty_all.index('mslp'), timestep],
            extra_contour_levels=[-0.4, -0.3, -0.2, -0.1, 0.1, 0.25, 0.5],
            extra_contour_colors=['darkviolet', 'darkviolet'],
            extra_contour_linestyles=['dashed', 'solid'],
            extra_contour_linewidths=2.5,
        )
        ax2.set_extent([-20, 40, -35, 35], crs=ccrs.PlateCarree())
        # print(ax2.get_xlim())
        ax2.set_xlim([-4500000, 6500000])
        # ax2.tight_layout()
        fig2.savefig(
            f'figures/{in_out_ch}ch/{experiment_source_detail[:-4]}_idx_{idx}_prop_900_{timestep * single_step_len}h' + '.jpg',
            dpi=300)
        print('saved prop 900', timestep * single_step_len)




elif 'Ljubljana' in experiment_source:
    tsidx = 3
    projection1 = ccrs.NearsidePerspective(
        central_longitude=14.+15.,
        central_latitude=46.,
        satellite_height=4500000
    )
    for timestep in [0, 2, 4]:  # 0h, 24h, 48h
        tsidx -= 1
        diff_U500 = ana_inc_prop[qty_all.index('U500'), timestep]
        diff_V500 = ana_inc_prop[qty_all.index('V500'), timestep]
        diff_Z500 = ana_inc_prop[qty_all.index('Z500'), timestep]
        diff_T2m = ana_inc_prop[qty_all.index('T2m'), timestep]
        diff_TCWV = ana_inc_prop[qty_all.index('TCWV'), timestep]
        diff_mslp = ana_inc_prop[qty_all.index('mslp'), timestep]


        fig2 = plt.figure(figsize=(6, 4))
        ax2 = fig2.add_subplot(1, 1, 1, projection=projection1)
        plot_Earth(
            fig=fig2,
            ax=ax2,
            field=diff_Z500,
            U=diff_U500,
            V=diff_V500,
            vminmax=vminmax_inc_no_obs[qty_all.index('Z500')],
            cmap=cmaps_inc[qty_all.index('Z500')],
            cbar_extend='both',
            unit='%s%s' % ('Z500', unit[qty_all.index('Z500')]),
            unit_in_cbar=True,
            # obs_locs=obs_lons_lats,
            title='%s, %s, and %s' % ('Z500', 'U500', 'V500'),#'%s, %s, and %s after %dh' % ('Z500', 'U500', 'V500', timestep * single_step_len),
            coastlines_linewidth=0.6
        )
        fig2.savefig(f'figures/{in_out_ch}ch/{experiment_source_detail[:-4]}_idx_{idx}_prop_ZUV500_{timestep * single_step_len}h' + '.jpg', dpi=300)
        print('saved prop ZUV500', timestep * single_step_len)


        fig2 = plt.figure(figsize=(6, 4))
        ax2 = fig2.add_subplot(1, 1, 1, projection=projection1)
        plot_Earth(
            fig=fig2,
            ax=ax2,
            field=diff_TCWV,
            vminmax=vminmax_inc_no_obs[qty_all.index('TCWV')],
            cmap=cmaps_inc[qty_all.index('TCWV')],
            cbar_extend='both',
            unit='%s%s' % ('TCWV', unit[qty_all.index('TCWV')]),
            unit_in_cbar=True,
            # obs_locs=obs_lons_lats,
            title='%s and %s' % ('TCWV', 'MSLP'),#'%s and %s after %dh' % ('TCWV', 'MSLP', timestep * single_step_len),
            extra_contour_field=diff_mslp,
            extra_contour_levels=np.arange(-1.5, 1.51, step=0.5),
            extra_contour_colors=['darkviolet', 'darkviolet'],
            extra_contour_linestyles=['dashed', 'solid'],
            extra_contour_linewidths=1.5,
            coastlines_linewidth=0.6
        )
        fig2.savefig(f'figures/{in_out_ch}ch/{experiment_source_detail[:-4]}_idx_{idx}_prop_TCWV_mslp_{timestep * single_step_len}h' + '.jpg', dpi=300)
        print('saved prop TCWV_mslp', timestep * single_step_len)


        fig2 = plt.figure(figsize=(6, 4))
        ax2 = fig2.add_subplot(1, 1, 1, projection=projection1)
        plot_Earth(
            fig=fig2,
            ax=ax2,
            field=diff_T2m,
            vminmax=vminmax_inc_no_obs[qty_all.index('T2m')],
            cmap=cmaps_inc[qty_all.index('T2m')],
            cbar_extend='both',
            unit='%s%s' % ('T2m', unit[qty_all.index('T2m')]),
            unit_in_cbar=True,
            # obs_locs=obs_lons_lats,
            title='%s' % ('T2m'),#'%s after %dh' % ('T2m', timestep * single_step_len),
            coastlines_linewidth=0.6
        )
        fig2.savefig(f'figures/{in_out_ch}ch/{experiment_source_detail[:-4]}_idx_{idx}_prop_T2m_{timestep * single_step_len}h' + '.jpg', dpi=300)
        print('saved prop T2m', timestep * single_step_len)

