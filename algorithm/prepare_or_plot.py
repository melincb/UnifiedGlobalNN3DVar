# -*- coding: utf-8 -*-
#!/usr/bin/python
"""
Script for preparing the inputs to 3D-Var algorithm

To run it please use run_3D-Var_multiple_runs.sh!
"""


# Osnovno
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# Slike
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from matplotlib.figure import Figure
import matplotlib

# Moduli
import sys
master_dir = os.getenv('UGNN3DVar_master')
sys.path.append(master_dir)
sys.path.append(master_dir + '/NNs')
import predict_1hr, predict_1hr_AE
import unet_model_7x7_4x_depth7_interdepth_padding_1_degree_v01
import unet_model_7x7_4x_depth4_interdepth_padding_1_degree_v01_stride2b_no_skip_AE

# --------------------------------------------------------------------------
# DODATNE KNJIZNICE + nastavitve
# --------------------------------------------------------------------------
import argparse
import pickle
import warnings

parser = argparse.ArgumentParser()
parser.add_argument("--True_EDA", help='Use B-matrix computed from actual ensemble members from IFS ensemble forecast', required=False, default=False, action=argparse.BooleanOptionalAction)
parser.add_argument('--FWD_model', help="Which forward model do I use? Options: 'NNfwd'", type=str, required=False, default='NNfwd')
parser.add_argument("--ensemble", help="No. of ensemble members", type=int, required=False, default=100)
parser.add_argument("--obs_datetime", help="Date and hour of *observation* in format yyyy-mm-dd-hh", type=str, required=False, default='2020-04-15-00')
parser.add_argument("--forecast_len", help="Number of 1-hourly forecast steps", type=int, required=False, default=24)
parser.add_argument('--preconditioned_3D_Var', help="Preconditioned 3D-Var - minimise zeta (chi) instead of z (x)", default=True, action=argparse.BooleanOptionalAction)  #In order to minimise z: --no-precondition_3D_Var
parser.add_argument('--compute', help="Compute assimilation", default=False, action=argparse.BooleanOptionalAction)  #In order not to compute: --no-compute
parser.add_argument('--plot', help="Plot", default=False, action=argparse.BooleanOptionalAction) #In order not to plot: --no-plot
parser.add_argument('--init_lr', help='Initial learning rate for ADAM optimizer when performing 3D-Var cost function minimization in latent space', type=float, default=0.3)
parser.add_argument('--custom_addon', help="Custom addon to output filenames", type=str, default='', required=False)
parser.add_argument('--obs_inc', help="Observation increment for single observation experiment - if '0.0', regular experiment will be performed", type=str, required=False, default='0.0')
parser.add_argument('--singobs_lat', help="Latitude in case of single observation experiment", type=float, required=False, default=False)
parser.add_argument('--singobs_lon', help="Latitude in case of single observation experiment", type=float, required=False, default=False)
parser.add_argument('--B_matrix_addon', help="Custom addon for B-matrix", type=str, default='', required=False)
parser.add_argument('--plot_singles', help="Plot some figures one by one (only if --plot)", default=False, action=argparse.BooleanOptionalAction)
parser.add_argument('--obs_qty', help="Observed variable (passed e.g. as Z200,u200)", type=str, required=True)
parser.add_argument('--obs_std', help="Standard deviation of pseudo observations, arbitrary unit (passed in the same order as obs_qty, e.g. as 1.0,2.5). If set to 0.0 in a singobs experiment, the system will set obs_std to the same value as background std in physical space", type=str, required=True)
parser.add_argument('--plot_qty', help="Plotted variable (passed e.g. as Z200,u200)", type=str, default='all', required=False)
parser.add_argument('--savefig_dir', help='Directory for saving the figure (if specified, it will be saved to experiments/figures/{in_out_ch}ch/args.savefig_dir; otherwise it will be just saved to experiments/figures/{in_out_ch}ch/)', type=str, default='', required=False)
parser.add_argument('--plot_projection', help="Plotting projection (PlateCarree or NearsidePerspective)", type=str, default='PlateCarree', required=False)
parser.add_argument("--True_EDA_ensemble_type", help="'forecasts' for ensemble of backgrounds. Only effective if --True_EDA!", type=str, default='forecasts', required=False)
parser.add_argument("--True_EDA_control_or_mean", help="Is EDA B-matrix computed using control or ensemble mean as reference? Options: 'control', 'mean'. Only effective if --True_EDA!", type=str, default='control', required=False)



args = parser.parse_args()

accepted_FWD_models = ('NNfwd')
if args.FWD_model not in accepted_FWD_models:
    print('Unknown forward model:', args.FWD_model)
    print('Accepted forward models:', accepted_FWD_models)
    raise AttributeError # Unknown forward model

import gc
from datetime import datetime, timedelta

tst = datetime.now()
print('Initiated computation', tst)


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


# Pot do statičnih polj
STATIC_FIELDS_ROOTS = tabele["STATIC_FIELDS_ROOTS"]

# Imena spremenljivk za shranjevanje in naslove slik
VARIABLES_SAVE = tabele["VARIABLES_SAVE"]

# Multiplicators for plotting the fields (and rescaling the observation increments and standard deviations)
field_adjustment_multpilicators = plotting_stuff()['FIELD_ADJUSTEMENT_MULTIPLICATION']


# Get the indices of the fields corresponding to each quantity
obs_qty_indices = {VARIABLES_SAVE[idx]:idx for idx in range(len(VARIABLES_SAVE))}
# Which quantities do we observe?
obs_qty = [q for q in args.obs_qty.split(',')]  # quantities
obs_qty_idx = [obs_qty_indices[q] for q in obs_qty] # quantities corresponding incides
# Their corresponding field adjustment multiplicators (this, e.g., converts geopotential from m2/s2 to m)
field_adjustment_multpilicators_filtered = []
for i in range(len(field_adjustment_multpilicators)):
    if i in obs_qty_idx:
        field_adjustment_multpilicators_filtered.append(field_adjustment_multpilicators[i])
# Get the standard deviations of the observations (value correspond to observed quantities, all observations have same std)
obs_std = [float(s) for s in args.obs_std.split(',')]
# Convert their values to those correspondind to decoded fields (before plotting)
obs_std = [obs_std[i] / field_adjustment_multpilicators_filtered[i] for i in range(len(obs_std))]
# Get the observation departures (=observation increments) in case of single observation (location) experiment
obs_inc = [float(oi) for oi in args.obs_inc.split(',')]
if args.obs_inc != '0.0':
    # Convert their values to those correspondin to decoded fields (before plotting)
    obs_inc = [obs_inc[i] / field_adjustment_multpilicators_filtered[i] for i in range(len(obs_inc))]
obs_inc_added_zeros = np.zeros((len(list(obs_qty_indices.keys()))))
obs_inc_added_zeros[obs_qty_idx] = obs_inc  # Zero where no values are preset



# --------------------------------------------------------------------------
# NALAGANJE OBEH MODELOV
# --------------------------------------------------------------------------
# WE DON'T REALLY NEED TO LOAD FWD MODEL AS WE ALREADY LOAD THE ENCODED BACKGROUND
# # FWD
if args.FWD_model == 'NNfwd':
    in_out_ch = 20
    FWD_root_model = '%s/NNs/models/U-Net_IT_ALL_numpred2_train_sumMSE_testACC_isl1_250ch_ks7x7_1hr_parallel_2024_11' % os.getenv(
        'UGNN3DVar_master')
    # Število korakov napovedi (če ni izbran samokodirnik, sicer pa število preslikav)
    num_of_prediction_steps = args.forecast_len // 12  # This fwd model has 12h steps
else:
    print(f'Oops! I forgot to allow {args.FWD_model} in the part of my code where I set FWD_root_model!')
    raise AttributeError



# AE

AE_root_model = '%s/NNs/models/autoencoder_20_12100' % os.getenv('UGNN3DVar_master')
in_out_ch = 20
AE_model = unet_model_7x7_4x_depth4_interdepth_padding_1_degree_v01_stride2b_no_skip_AE.UNet(
    in_channels=1 * in_out_ch + 3, out_channels=in_out_ch, depth=4, start_filts=50,
    up_mode='transpose', merge_mode='concat')
name_preposition = 'AE_20_12100'
AE_latent_vec_shape = (1, 12100)
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


# ------------------------------------------------------
# PARAMETERS SETUP
# ------------------------------------------------------

# Initial learning rate for 3D-Var algorithm
init_lr = args.init_lr

# Observations points' settings
res = 4.0   # distance between observation points (or 'custom')
if (args.singobs_lon is not False) or (args.singobs_lat is not False):
    if (args.singobs_lon is not False) and (args.singobs_lat is not False):
        res = 'custom'
    else:
        raise AttributeError # You specified either singobs_lon or singobs_lat, but you should either specify both of them or none of them!
elif args.obs_inc != '0.0':
    print('You specified the observation increment, but did not set singobs_lon and singobs_lat!')
    raise AttributeError # You specified the observation increment, but did not set singobs_lon and singobs_lat!

if res == 'custom':
    pass
else:
    res_str = [s if s != '.' else '_' for s in str(res)]    # 4.0 -> '4_0'
    args.custom_addon += 'res_' + str(res)
# ------------------------------------------------------
# LOAD DATA FOR THE BEGINNING AND END OF FWD MODEL
# ------------------------------------------------------

# Determine date and time of beginning and end of FWD model
obs_datetime = [int(t) for t in args.obs_datetime.split('-')]
FWD_end_datetime = datetime(year=obs_datetime[0], month=obs_datetime[1], day=obs_datetime[2], hour=obs_datetime[3])
FWD_end_date = (FWD_end_datetime.day, FWD_end_datetime.month, FWD_end_datetime.year)
FWD_end_time = FWD_end_datetime.hour
FWD_start_datetime = FWD_end_datetime - timedelta(hours=args.forecast_len)
FWD_start_date = (FWD_start_datetime.day, FWD_start_datetime.month, FWD_start_datetime.year)
FWD_start_time = FWD_start_datetime.hour

# Load data for the beginning of FWD model

# Load data for the end of FWD model
# Load encoded background
if not args.True_EDA:
    if args.FWD_model in ('NNfwd'):
        AE_latent_background = encoded_forecast_provider(
            FWD_root_model=FWD_root_model,
            AE_root_model=AE_root_model,
            keep_3D=False,
            initial_conditions='ERA5',
            forecast_start_datetime=FWD_start_datetime,
            forecast_end_datetime=FWD_end_datetime
        )
        AE_latent_dim = AE_latent_background.shape[-1]


    else:
        print(f'Oops! I forgot to allow {args.FWD_model} in the part of my code where I load encoded background!')
        raise AttributeError



# ------------------------------------------------------
# 3D-VAR
# ------------------------------------------------------

# Manual settings for True_EDA:
included_members = 'all'

if args.compute:
    print('COMPUTING!')

    # ------------------------------------------------------
    # LOAD BACKGROUND ERROR COVARIANCE MATRIX AND ITS INVERSE
    # ------------------------------------------------------
    if args.True_EDA:
        # Manual settings:
        # included_members = 'all'

        if included_members == 'all':
            included_members_lst = [i for i in range(1, args.ensemble + 1)]
        else:
            included_members_lst = [int(s) for s in args.days_in_month.split('_')]

        if args.obs_datetime != '2024-04-14-00':
            warnings.warn(f"WARNING: You are using True_EDA, but your observation datetime is {args.obs_datetime} and not {'2024-04-14-00'}!")
        if args.ensemble != 50:
            warnings.warn(
                f"WARNING: You are using True_EDA, but your ensemble size is {args.ensemble} and not {50}!")

        if args.True_EDA_ensemble_type == 'forecasts':
            ensemble_main_dir_ext = 'ens_B'
        else:
            print('Unsupported args.True_EDA_ensemble_type:', args.True_EDA_ensemble_type)
            raise AttributeError  # Unsupported ensemble type



        B_matrix_savename = f'{AE_root_model}/EDA/B-matrices/{ensemble_main_dir_ext}' + \
                            f'/B_{args.True_EDA_control_or_mean}_{args.obs_datetime}_size_{args.ensemble}_members_{included_members}'

        if len(args.B_matrix_addon) > 0:
            B_matrix_savename += f'_{args.B_matrix_addon}'
        B_matrix_savename += '_diagonal'

        # Here we set the filename of the B-matrix, but we needn't load it in this script
        B_matrix = [] # Creating this empty list so I don't need extra if statements for deleting B-matrix in other cases


    else:
        B_matrix_savename = f'{FWD_root_model}/B-matrices/{AE_root_model[AE_root_model.rfind("/") + 1:]}' + \
                            f'/climatological_prediction_no_precondition_2015-01-01_to_2019-12-31' + \
                            f'_days_all_hrs_all_steps_{args.forecast_len}'

        if len(args.B_matrix_addon) > 0:
            B_matrix_savename += f'_{args.B_matrix_addon}'
        B_matrix_savename += '_diagonal'

        # Here I load just the B-matrix - I will only have it in memory until I generate
        # perturbed latent-background vectors and then I will delete it

        B_matrix = pickle.load(open(B_matrix_savename + '.pkl', 'rb'))

    # ------------------------------------------------------
    # PREPARE LATENT-SPACE BACKGROUND
    # ------------------------------------------------------
    print('Preparing the background vectors')
    if args.True_EDA:
        encoded_ensemble_members = [
            EDA_ensemble_member_load(
                which_ensemble=args.True_EDA_ensemble_type,
                em_idx=em_idx,
                datetime_str=args.obs_datetime,
                return_encoded=True,
                echo=False
            )
            for em_idx in included_members_lst
        ]

        # This time we do not perturb the ensemble members, we leave them as they are!
        perturbed_AE_latent_background = np.array(encoded_ensemble_members)


    else:
        perturbed_AE_latent_background = np.random.normal(size=(args.ensemble, 1, AE_latent_dim),
                                                          loc=AE_latent_background[0],
                                                          scale=np.sqrt(B_matrix)
                                                          ).astype(np.float32)


        if args.ensemble == 100 and args.obs_datetime == '2020-04-15-00':   # Let's have the same ensemble for all the experiments!
            ens_data_main_name = 'data_singobs_Ljubljana_SGD_endtime_2020-04-15-00_FWD_model=NNfwd_dt=24h_obs_qty=Z500_obs_inc=30.0_obs_std=10.0_B_type=diagonal_ens=100_init_lr=0.3_prec3DVar.pkl'
            old_data = pickle.load(open(f'experiments/data/{in_out_ch}ch/' + ens_data_main_name, 'rb'))
            perturbed_AE_latent_background = old_data['AE_latent_background'].numpy()
            print('Loaded perturbed background')
        got_perturbed_AE_latent_background = True


    # Delete the B-matrix - won't need it from here on
    del B_matrix
    gc.collect()

    # ------------------------------------------------------
    # PREPARE OBSERVATIONS LOCATIONS
    # ------------------------------------------------------

    if type(res) == float:
        # Multiple observations on global scale with 'res' degrees appart in both longitudes and latitudes
        lats = np.array([[90 - 0.75 - res * ilat for ilon in range(int(360 / res))] for ilat in range(int(180 / res))]).astype(np.float32)
        lons = np.array(
            [[-180 + 0.75 + res * ilon for ilon in range(int(360 / res))] for ilat in range(int(180 / res))]).astype(np.float32)
    else:
        # res='custom'
        # Set lats and lons for single observation experiment
        lats = np.array([[args.singobs_lat]]).astype(np.float32)
        lons = np.array([[args.singobs_lon]]).astype(np.float32)

    # Format of input and output of AE: idx ... location
    # LAT: 0 ... 89.5 degS, 179 ... 89.5 degN
    # LON: 0 ... 180 degW, 359 ... 179 degE
    # Format of input to this script (args.singobs_lat, args.singobs_lon): value ... location
    # LAT: -90 ... 90 degS, 90 ... 90 degN
    # LON: -179.99 ... 179.99 degW, 179.99 ... 179.99 degE
    # Now we need to convert the input locations to this script into their coordinates for AE input/output
    # First: latitudes: -89.5 -> 0, 89.5 -> 179
    # WARNING: Avoid using lats < -89.5 and lats > 89.5, as these cannot be interpolated correctly using F.grid_sample!
    lats_AE = lats + 89.5
    # Second: longitudes: -180 -> 0, 179 -> 359
    # WARNING: Avoid using lons > 179, as these cannot be interpolated correctly using F.grid_sample!
    lons_AE = lons + 180
    # Normalize lons_AE and lats_AE to [-1, 1]
    lats_AE_normalized = lats_AE / 179 * 2 - 1  # WARNING: lats < -89.5 and lats > 89.5 are outside of [-1, 1] range!
    lons_AE_normalized = lons_AE / 359 * 2 - 1  # WARNING: lons > 179 are outside of [-1, 1] range!

    # Stack normalized lats and lons, reshape them to (Nobs, 2), and convert them to torch tensor.
    obs_locs_torch = torch.from_numpy(np.reshape(np.stack((lats_AE_normalized, lons_AE_normalized), axis=-1), (np.shape(lats)[0] * np.shape(lats)[1], 2)))
    # After lots of testing I figured out that, even though the original script was written with (at least to my knowledge)
    # complete accordance with F.grid_sample documentation, for some reason the interpolation is performed correctly
    # only if I swap the columns of longitudes and latitudes... This part of code could definetly be written in a better way,
    # but in the end this gave correct results. The 5 strange lines are given below:
    obs_locs_torch1 = torch.from_numpy(np.zeros(shape=obs_locs_torch.shape).astype(np.float32))
    obs_locs_torch1[:, 0], obs_locs_torch1[:, 1] = obs_locs_torch[:, 1], obs_locs_torch[:, 0]
    obs_locs_torch = obs_locs_torch1
    del obs_locs_torch1
    gc.collect()

    # Reshaping locs_to_interpolate so it is really suitable for interpolation
    obs_locs_torch = obs_locs_torch.unsqueeze(0).expand(in_out_ch, -1, -1)  # Shape: (in_out_ch, Nobs, 2)
    obs_locs_torch = obs_locs_torch.unsqueeze(2)  # shape (in_out_ch, Nobs, 1, 2), suitable for interpolation

    # ------------------------------------------------------
    # PREPARE OBSERVATIONS VALUES
    # ------------------------------------------------------

    if args.obs_inc == '0.0':
        # Normal experiment: observations sampled at FWD_end_datetime
        # Here we pretend that we observe all in_out_ch variables - we will take care of this in algorithm-serial.py where we filter out non-observed quantities
        # Doesn't work if True_EDA, because the dates for True_EDA are out of the loaded ERA5 dates (thus ERA5_ground_truth_end_time is not defined in that case)!
        obs_values = F.grid_sample(ERA5_ground_truth_end_time, obs_locs_torch, mode='bilinear', align_corners=True)  # Shape: (in_out_ch, 1, Nobs, 1)

    else:
        # Single observation (location) experiment
        # First convert background from AE latent space to the gridpoint space
        # The current version of the AE can only decode fields one-by-one and its first two dimensions are flipped
        background_gp = torch.zeros(size=(args.ensemble, in_out_ch, 1, 180, 360), dtype=torch.float32) # shape: (ensemble members, variables, 1, lats, lons)
        for izb in range(len(perturbed_AE_latent_background)):
            # print('type', type(perturbed_AE_latent_background[izb]), perturbed_AE_latent_background[izb].dtype)
            AE_preds = AE_predicting.decode(
                                ANNOTATIONS_NAMES,
                                AE_MODELS,
                                root_scaler,
                                SCALER_NAMES,
                                SCALER_TYPES,
                                num_of_prediction_steps=1,
                                z=torch.from_numpy(perturbed_AE_latent_background[izb].reshape(AE_latent_space_shape))    # perturbed_AE_latent_background is a list of lists of numpy arrays
                                )   # shape (in_out_ch, 2, 180, 360), [:,0,:,:] is useless
            for ivar in range(background_gp.shape[1]):  # go through all in_out_ch variables
                background_gp[izb, ivar] = AE_preds[ivar][1]


        # Get the mean background state in the gridpoint space
        background_gp_mean = torch.mean(background_gp, dim=0)
        background_gp_std = torch.std(background_gp, dim=0)
        # Delete background_gp to recover some memory
        del background_gp, AE_preds
        gc.collect()
        # Get the exact values at the obs. locations which will be used as the means of the observations with added
        # observation departure (=observation increment)
        obs_values_before_increment = F.grid_sample(background_gp_mean, obs_locs_torch, mode='bilinear', align_corners=True)  # Shape: (in_out_ch, 1, Nobs, 1)
        # in pol pristejem se inkrement
        obs_values = obs_values_before_increment + torch.from_numpy(np.expand_dims(obs_inc_added_zeros, [1, 2, 3])) # shape (in_out_ch, 1, 1, 1), because in single obs. experiments Nobs=1

    # In the next line we prepare the (transposed) observations' vector "y"
    # Here we
    # (1) keep only the quantities that we really observe (using [obs_qty_idx, :, :, :]),
    # (2) remove the excess dimensions (using squeeze()), and
    # (3) reshape it to a (transposed) vector with dimmension (Nqties*Nobs, 1) (using view(-1, 1)).
    # If we observe 2 quantities, e.g. u10m, Z500, at 3 locations, than this vector is
    # [[u10m(loc1)], [u10m(loc2)], [u10m(loc3)], [Z500(loc1)], [Z500(loc2)], [Z500(loc3)]]
    # print(obs_values.shape)
    # print(obs_values[obs_qty_idx, :, :, :].shape)
    obs_values_filtered_vec = obs_values[obs_qty_idx, :, :, :].squeeze().view(-1, 1)
    obs_values_filtered_vec = obs_values_filtered_vec.detach().requires_grad_(False)   # So I can perturb it using np.random.normal

    # Prepare observations error covariance matrix and compute its inverse
    # There are assumed to be no correlations, so this 2D matrix only has non-zero diagonal elements)
    # Nobs = obs_values_filtered_vec_transposed.shape[1]//len(obs_qty)
    if obs_std[0] == 0.0:  # set obs_std = background_std, only works for single obs. experiments
        background_gp_std_at_obs_loc = F.grid_sample(background_gp_std, obs_locs_torch, mode='bilinear',
                                                     align_corners=True)  # Shape: (in_out_ch, 1, Nobs, 1)
        obs_std = [background_gp_std_at_obs_loc[obs_qty_idx, :, :, :].squeeze().view(-1, 1).detach().item()]

    R_matrix = np.identity(obs_values_filtered_vec.shape[0]) * np.array(
        [[ostd ** 2 for o in range(obs_values_filtered_vec.shape[1] // len(obs_qty))] for ostd in obs_std]).flatten()
    R_matrix_inv = np.linalg.inv(R_matrix).astype(np.float32)

    # Perturb the (transposed) observations vector
    print('Perturbing the observations vector')
    perturbed_obs_vec_transposed = np.float32(np.random.normal(
        size=(args.ensemble, 1, obs_values_filtered_vec.shape[0]),
        loc=np.expand_dims(np.transpose(obs_values_filtered_vec), 0),
        scale=np.expand_dims(np.sqrt(np.diagonal(R_matrix)), axis=(0, 1))
    ))


    del R_matrix
    gc.collect()

    if args.True_EDA:
        # Use the exact same observations as in the previous experiments
        if args.singobs_lat == 46.1 and args.singobs_lon == 14.5 and args.obs_qty=='Z500' and args.obs_inc == '30.0' and args.obs_std == '10.0' and args.True_EDA_control_or_mean == 'mean':
            ens_data_main_name = 'data_EDA_singobs_Ljubljana_mean_endtime_2024-04-14-00_mean_size_50_members_all_obs_qty=Z500_obs_inc=30.0_obs_std=10.0_B_type=diagonal_init_lr=0.3_prec3DVar.pkl'
            old_data = pickle.load(open(f'experiments/data/{in_out_ch}ch/' + ens_data_main_name, 'rb'))
            perturbed_obs_vec_transposed = old_data['obs_vec_transposed'].numpy()
            print('\nLoaded perturbed obs from:')
            print(ens_data_main_name)


    # ------------------------------------------------------
    # PACK ALL THE CONTENT FOR THE 3D-VAR ALGORITHM
    # ------------------------------------------------------

    # Pack content
    # This kind of content packing (list of dicts) might seem redundant for serial computing,
    # however, it is very convenient for parallel computing
    inputs_for_ensemble_3D_Var = [{'perturbed_AE_latent_background': perturbed_AE_latent_background[iens_mem],
                                   'obs_locs': [lats, lons],
                                   'perturbed_obs': perturbed_obs_vec_transposed[iens_mem],
                                   'obs_qty_idx': obs_qty_idx,
                                   'B_matrix_filename': B_matrix_savename,
                                   'R_matrix_inv': R_matrix_inv,
                                   'init_lr': init_lr,
                                   'ensemble_member_idx': iens_mem} for iens_mem in range(args.ensemble)]

    # Create unique filename for the 3D-Var outputs
    if args.True_EDA:
        file_to_dump = f'experiments/data/{in_out_ch}ch/data_EDA_{args.custom_addon}_endtime_{args.obs_datetime}_{args.True_EDA_control_or_mean}_size_{args.ensemble}_members_{included_members}_obs_qty={args.obs_qty}_obs_inc={args.obs_inc}_obs_std={args.obs_std}_B_type=diagonal'

        if args.init_lr != 0.01:
            file_to_dump += f'_init_lr={args.init_lr}'
        if len(args.B_matrix_addon) > 0:
            file_to_dump += f'_{args.B_matrix_addon}'
        if args.preconditioned_3D_Var:
            file_to_dump += f'_prec3DVar'

    else:
        file_to_dump = f'experiments/data/{in_out_ch}ch/data_{args.custom_addon}_endtime_{args.obs_datetime}_FWD_model={args.FWD_model}_dt={args.forecast_len}h_obs_qty={args.obs_qty}_obs_inc={args.obs_inc}_obs_std={args.obs_std}_B_type=diagonal_ens={args.ensemble}'
        if args.init_lr != 0.01:
            file_to_dump += f'_init_lr={args.init_lr}'
        if len(args.B_matrix_addon) > 0:
            file_to_dump += f'_{args.B_matrix_addon}'
        if args.preconditioned_3D_Var:
            file_to_dump += f'_prec3DVar'

    # Some extra stuff to store (i.e. stuff that cannot be recreated later)
    # So far we have nothing here...
    # If you do add something here, also add it to dict_to_dump and dict_to_load
    # (which is the dumped dict from ...algorithm.py, where you also have to take care of it)


    # raise AssertionError

    # Dump the inputs to 3D-Var
    dict_to_dump = {'inputs_for_ensemble_3D_Var': inputs_for_ensemble_3D_Var,
                    'name': file_to_dump,
                    'FWD_root_model': FWD_root_model,
                    'AE_root_model': AE_root_model,
                    'preconditioned_3D_Var': args.preconditioned_3D_Var}
    pickle.dump(dict_to_dump, open('experiments/data/algorithm_inputs.pkl', 'wb'))



# ------------------------------------------------------
# PLOT THE RESULTS
# ------------------------------------------------------
elif args.plot:
    print('PLOTTING!')
    start = datetime.now()
    # ------------------------------------------------------
    # LOAD THE 3D-VAR OUTPUTS
    # ------------------------------------------------------
    # Select unique filename

    if args.True_EDA:
        file_to_load = f'experiments/data/{in_out_ch}ch/data_EDA_{args.custom_addon}_endtime_{args.obs_datetime}_{args.True_EDA_control_or_mean}_size_{args.ensemble}_members_{included_members}_obs_qty={args.obs_qty}_obs_inc={args.obs_inc}_obs_std={args.obs_std}_B_type=diagonal'

        if args.init_lr != 0.01:
            file_to_load += f'_init_lr={args.init_lr}'
        if len(args.B_matrix_addon) > 0:
            file_to_load += f'_{args.B_matrix_addon}'
        if args.preconditioned_3D_Var:
            file_to_load += f'_prec3DVar'

    else:
        file_to_load = f'experiments/data/{in_out_ch}ch/data_{args.custom_addon}_endtime_{args.obs_datetime}_FWD_model={args.FWD_model}_dt={args.forecast_len}h_obs_qty={args.obs_qty}_obs_inc={args.obs_inc}_obs_std={args.obs_std}_B_type=diagonal_ens={args.ensemble}'
        if args.init_lr != 0.01:
            file_to_load += f'_init_lr={args.init_lr}'
        if len(args.B_matrix_addon) > 0:
            file_to_load += f'_{args.B_matrix_addon}'
        if args.preconditioned_3D_Var:
            file_to_load += f'_prec3DVar'


    # Load
    dict_to_load = pickle.load(open(file_to_load + '.pkl', 'rb'))
    AE_latent_analysis = dict_to_load['AE_latent_out'].to(torch.float32) # Torch tensor - Outputs in the latent space
    AE_latent_background = dict_to_load['AE_latent_background']  # Torch tensor - Perturbed (transposed) latent vectors (for background)
    obs_locs_torch = dict_to_load['obs_locs_torch']   # Observations points (torch tensor suitable for bilinear interpolation)
    obs_lons_lats = dict_to_load['obs_lons_lats']   # Observations points [lons, lats], suitable for plotting
    obs_vec_transposed = dict_to_load['obs_vec_transposed'] # Perturbed pseudo observations
    best_J = dict_to_load['best_J']   # The best value of the 3D-Var cost function for each ensemble member
    all_J = dict_to_load['all_J'] # All values of the 3D-Var cost function for each ensemble member
    all_Jo = dict_to_load['all_Jo'] # All values of the observations term of the 3D-Var cost function for each ensemble member
    all_grad_J = dict_to_load['all_grad_J']    # All gradients of the 3D-Var cost function for each ensemble member

    comment = dict_to_load['comment']   # Basically the same comments as above
    print('\nComment on contents of the algorithm output:')
    print(comment)



    # ------------------------------------------------------
    # MAIN SETTINGS FOR FIGURE NAMES
    # ------------------------------------------------------
    if len(args.savefig_dir) > 0:
        if args.True_EDA:
            relative_dir = f'experiments/figures/{in_out_ch}ch/EDA/{args.savefig_dir}'
        else:
            relative_dir = f'experiments/figures/{in_out_ch}ch/{args.savefig_dir}'
        if not os.path.exists(relative_dir):
            os.makedirs(relative_dir)
        args.savefig_dir = args.savefig_dir + '/'

    if args.True_EDA:
        savefig_name = f'experiments/figures/{in_out_ch}ch/EDA/{args.savefig_dir}/fig_{args.custom_addon}_endtime_{args.obs_datetime}_{args.True_EDA_control_or_mean}_ens={args.ensemble}_members_{included_members}_obs_qty={args.obs_qty}_obs_inc={args.obs_inc}_obs_std={args.obs_std}_B_type=diagonal'
    else:
        savefig_name = f'experiments/figures/{in_out_ch}ch/{args.savefig_dir}fig_{args.custom_addon}_endtime_{args.obs_datetime}_FWD_model={args.FWD_model}_dt={args.forecast_len}h_obs_qty={args.obs_qty}_obs_inc={args.obs_inc}_obs_std={args.obs_std}_B_type=diagonal_ens={args.ensemble}_plt_qty='
    if args.init_lr != 0.01:
        savefig_name += f'init_lr={args.init_lr}'
    if len(args.B_matrix_addon) > 0:
        savefig_name += f'_{args.B_matrix_addon}'
    if args.preconditioned_3D_Var:
        savefig_name += f'_prec3DVar'
    if args.plot_projection != 'PlateCarree':
        savefig_name += f'_{args.plot_projection}'



    # ------------------------------------------------------
    # FINAL DECODING NEEDED FOR PLOTTING
    # ------------------------------------------------------
    # Compute standard deviation of the background (propagated to the gridpoint space)
    # First convert background from AE latent space to the gridpoint space
    # The current version of the AE can only decode fields one-by-one and its first two dimensions are flipped
    background_gp = torch.zeros(size=(args.ensemble, in_out_ch, 1, 180, 360),
                                dtype=torch.float32)  # shape: (ensemble members, variables, 1, lats, lons)
    for izb in range(len(AE_latent_background)):
        AE_preds = AE_predicting.decode(
            ANNOTATIONS_NAMES,
            AE_MODELS,
            root_scaler,
            SCALER_NAMES,
            SCALER_TYPES,
            num_of_prediction_steps=1,
            z=AE_latent_background[izb].reshape(AE_latent_space_shape)    # AE_latent_background is already a torch tensor
        )  # shape (in_out_ch, 2, 180, 360), [:,0,:,:] is useless
        for ivar in range(background_gp.shape[1]):  # go through all in_out_ch variables
            background_gp[izb, ivar] = AE_preds[ivar][1]

    # Get the mean and std of background state in the gridpoint space
    background_gp_mean_npy = torch.mean(background_gp, dim=0).cpu().detach().numpy() * np.expand_dims(field_adjustment_multpilicators, (1,2,3))
    background_gp_std_npy = torch.std(background_gp, dim=0).cpu().detach().numpy() * np.expand_dims(field_adjustment_multpilicators, (1,2,3))
    background_gp_scaled = [bg.cpu().detach().numpy() * np.expand_dims(field_adjustment_multpilicators, (1, 2, 3))
                            for bg in background_gp]
    if len(obs_inc) == 1 and obs_inc[0] != 0.0:  # There is only one observation of one quantity
        background_gp_at_obs_loc = [[] for igp_field in range(background_gp.shape[1])]
        for iens_mem in range(len(background_gp)):  # Go through all ensemble members
            background_gp_interpolated_to_obs_loc = F.grid_sample(
                background_gp[iens_mem], obs_locs_torch, mode='bilinear', align_corners=True
            ).cpu().detach().numpy() * np.expand_dims(field_adjustment_multpilicators, (1, 2, 3))
            for igp_field in range(background_gp.shape[1]):
                background_gp_at_obs_loc[igp_field].append(background_gp_interpolated_to_obs_loc[igp_field].squeeze())
        background_gp_at_obs_loc_mean, background_gp_at_obs_loc_std = np.mean(background_gp_at_obs_loc, axis=1), np.std(
            background_gp_at_obs_loc, axis=1)


    del background_gp, AE_preds
    gc.collect()

    # Same procedure, but for the analysis
    analysis_gp = torch.zeros(size=(args.ensemble, in_out_ch, 1, 180, 360),
                                dtype=torch.float32)  # shape: (ensemble members, variables, 1, lats, lons)
    for izb in range(len(AE_latent_background)):
        AE_preds = AE_predicting.decode(
            ANNOTATIONS_NAMES,
            AE_MODELS,
            root_scaler,
            SCALER_NAMES,
            SCALER_TYPES,
            num_of_prediction_steps=1,
            z=AE_latent_analysis[izb].reshape(AE_latent_space_shape)  # AE_latent_analysis is already a torch tensor
        )  # shape (in_out_ch, 2, 180, 360), [:,0,:,:] is useless
        for ivar in range(analysis_gp.shape[1]):  # go through all in_out_ch variables
            analysis_gp[izb, ivar] = AE_preds[ivar][1]

    # Get the mean and std of background state in the gridpoint space
    analysis_gp_mean_npy = torch.mean(analysis_gp, dim=0).cpu().detach().numpy() * np.expand_dims(field_adjustment_multpilicators, (1,2,3))
    analysis_gp_std_npy = torch.std(analysis_gp, dim=0).cpu().detach().numpy() * np.expand_dims(field_adjustment_multpilicators, (1,2,3))
    if len(obs_inc) == 1 and obs_inc[0] != 0.0:  # There is only one observation of one quantity
        print('\nAssessing compliance of analisys increment and std with LINEARISED theory...')

        analysis_gp_at_obs_loc = [[] for igp_field in range(analysis_gp.shape[1])]
        for iens_mem in range(len(analysis_gp)):  # Go through all ensemble members
            analysis_gp_interpolated_to_obs_loc = F.grid_sample(
                analysis_gp[iens_mem], obs_locs_torch, mode='bilinear', align_corners=True
            ).cpu().detach().numpy() * np.expand_dims(field_adjustment_multpilicators, (1, 2, 3))
            for igp_field in range(analysis_gp.shape[1]):
                analysis_gp_at_obs_loc[igp_field].append(analysis_gp_interpolated_to_obs_loc[igp_field].squeeze())
        #print(np.shape(analysis_gp_at_obs_loc))
        analysis_gp_at_obs_loc_mean, analysis_gp_at_obs_loc_std = np.mean(analysis_gp_at_obs_loc, axis=1), np.std(
            analysis_gp_at_obs_loc, axis=1)
        #print(np.shape(analysis_gp_at_obs_loc_mean))
        # Finally, find mean and std of observations
        obs_mean, obs_std = np.mean(obs_vec_transposed.numpy()) * field_adjustment_multpilicators[
            obs_qty_idx[0]], np.std(obs_vec_transposed.numpy() * field_adjustment_multpilicators[obs_qty_idx[0]])
        #print(background_gp_at_obs_loc_mean)
        print('\nObservation increment', obs_mean - background_gp_at_obs_loc_mean[obs_qty_idx[0]])
        print('Observation std', obs_std)
        print('Background std', background_gp_at_obs_loc_std[obs_qty_idx[0]])
        print('\nTheoretical analysis increment', (obs_mean - background_gp_at_obs_loc_mean[obs_qty_idx[0]]) / (
                    1 + obs_std ** 2 / background_gp_at_obs_loc_std[obs_qty_idx[0]] ** 2))
        print('Analysis increment',
              analysis_gp_at_obs_loc_mean[obs_qty_idx[0]] - background_gp_at_obs_loc_mean[obs_qty_idx[0]])
        print('\nTheoretical analysis std',
              np.sqrt(1 / (1 / obs_std ** 2 + 1 / background_gp_at_obs_loc_std[obs_qty_idx[0]] ** 2)))
        print('Analysis std', analysis_gp_at_obs_loc_std[obs_qty_idx[0]])


        plot_relat_change_hist = False
        if plot_relat_change_hist:
            # Plot relative change histograms

            plt.figure(figsize=(7,6))
            ana_latent_var = np.var(AE_latent_analysis.detach().numpy(), axis=0)[0]
            bg_latent_mean = np.mean(AE_latent_background.detach().numpy(), axis=0)[0]
            ana_latent_mean = np.mean(AE_latent_analysis.detach().numpy(), axis=0)[0]
            weighted_change = np.abs(bg_latent_mean - ana_latent_mean) / np.sqrt(ana_latent_var)
            nbin = 51
            bins = np.linspace(-8, 2, nbin)
            plt.hist(np.log10(weighted_change), bins=bins, color='springgreen',
                     alpha=1, label='Relative change of\nensemble mean')
            plt.xlabel(r'$\log_{10}$(abs($\mathbf{z}$ element))')
            plt.xlim(min(bins), max(bins))
            plt.ylim(0,800)
            plt.ylabel('Number of elements')
            plt.legend(loc='upper right')
            if args.True_EDA:
                if 'CLIM_B' in args.custom_addon:
                    plt.title(f'After observing {args.obs_qty} in hybrid setup')
                else:
                    plt.title(f'After observing {args.obs_qty} in full EDA setup')
            else:
                plt.title(f'After observing {args.obs_qty}')

            plt.arrow(np.log10(weighted_change)[4786], 300, 0, -300)
            plt.text(np.log10(weighted_change)[4786], 300, 4786)#, ha='center')
            plt.arrow(np.log10(weighted_change)[8878], 500, 0, -500)
            plt.text(np.log10(weighted_change)[8878], 500, 8878)#, ha='center')
            plt.arrow(np.log10(weighted_change)[9626], 400, 0, -400)
            plt.text(np.log10(weighted_change)[9626], 400, 9626)#, ha='center')
            plt.arrow(np.log10(weighted_change)[4233], 500, 0, -500)
            plt.text(np.log10(weighted_change)[4233], 500, 4233)#, ha='center')


            # plt.savefig(savefig_name + "_latent_element_hist_solo.jpg", dpi=300)
            plt.savefig(savefig_name + "_latent_element_hist_solo.pdf")
            plt.clf()
            plt.cla()




    # Delete stuff to recover some memory
    del analysis_gp, AE_preds
    gc.collect()


    # ------------------------------------------------------
    # ACTUALLY PLOT
    # ------------------------------------------------------

    # Which quantities do I plot?
    plot_qty = [q for q in args.plot_qty.split(',')]  # quantities
    if 'all' in plot_qty:   # plot all in_out_ch fields
        plot_qty_idx = [i for i in range(background_gp_mean_npy.shape[0])]
    else:
        plot_qty_idx = [obs_qty_indices[q] for q in plot_qty]  # quantities corresponding incides

    # Plotting settings
    num_fields = len(plot_qty_idx) # Number of fields to plot
    max_num_fields = background_gp_mean_npy.shape[0] # Highest possible number of fields to plot
    single_fig_width = 10 # Width of single figure (in inches, I think)
    qty_all = VARIABLES_SAVE
    unit = [' [' + plotting_stuff()['ENOTE'][i] + ']' for i in range(background_gp_mean_npy.shape[0])]    # How to display units

    fig = plt.figure(figsize=(single_fig_width * num_fields, 60))
    matplotlib.rcParams.update({"font.size": 26})

    projections = {'Orthographic': ccrs.Orthographic(),
                   'Robinson': ccrs.Robinson(),
                   'PlateCarree': ccrs.PlateCarree(),
                   'NearsidePerspective': ccrs.NearsidePerspective(
                       central_longitude=args.singobs_lon,
                       central_latitude=args.singobs_lat,
                       satellite_height=4500000)
                   }
    projection = args.plot_projection #'platecarree'


    plot_left = 0.075 / num_fields
    plot_width = 0.85 / num_fields
    tot_plots = 5  # Set the number of rows
    height_sum = tot_plots * 0.095
    dheight_plot = 0.05 / height_sum # 0.075 / height_sum
    dheight_buffer = -0.01 / height_sum #0.005 / height_sum
    plot_bottom = 0.98 #1 # Initial value for plot_bottom (it will gradually drop to 0)

    cbar_shrink = 0.5


    cmaps = plotting_stuff()['cmaps'] # Colormaps
    cmaps_inc = plotting_stuff()['cmaps_inc'] # Colormaps
    vminmax = plotting_stuff()['vminmax'] # Ranges for all available quantities
    vminmax_std = plotting_stuff()['vminmax_std'] # Ranges for all available quantities when plotting std
    vminmax_inc = plotting_stuff()['vminmax_inc'] # Ranges for all available quantities when plotting the analysis increment, if the quantity of interest is not observed

    if args.True_EDA:
        vminmax_std = (np.array(vminmax_std) * 0.1).tolist()

    # (A) mean decoded background
    plot_bottom -= dheight_plot + dheight_buffer


    ifieldplot = 0
    for ifield in range(max_num_fields):
        if ifield in plot_qty_idx:
            ax_dfg = fig.add_axes([plot_left + ifieldplot/num_fields, plot_bottom, plot_width, dheight_plot], projection=projections[projection])
            ifieldplot += 1
            plot_Earth(
                fig=fig,
                ax=ax_dfg,
                field=background_gp_mean_npy[ifield, 0], #with clim
                vminmax=vminmax[ifield],
                cmap=cmaps[ifield],
                cbar_shrink=cbar_shrink,
                title='Decoded background (%s%s)' % (qty_all[ifield], unit[ifield])
            )



    # (B) analysis increment
    plot_bottom -= dheight_plot + dheight_buffer

    ifieldplot = 0
    for ifield in range(max_num_fields):
        if ifield in plot_qty_idx:
            ax_amb = fig.add_axes([plot_left + ifieldplot / num_fields, plot_bottom, plot_width, dheight_plot], projection=projections[projection])
            ifieldplot += 1
            plot_Earth(
                fig=fig,
                ax=ax_amb,
                field=analysis_gp_mean_npy[ifield, 0] - background_gp_mean_npy[ifield, 0],
                vminmax=vminmax_inc[ifield],
                obs_locs=obs_lons_lats,
                cmap=cmaps_inc[ifield],
                cbar_shrink=cbar_shrink,
                title='Analysis increment (%s%s)' % (qty_all[ifield], unit[ifield])
            )


    # (C) std of decoded background
    plot_bottom -= dheight_plot + dheight_buffer

    ifieldplot = 0
    for ifield in range(max_num_fields):
        if ifield in plot_qty_idx:
            ax_stdb = fig.add_axes([plot_left + ifieldplot / num_fields, plot_bottom, plot_width, dheight_plot], projection=projections[projection])
            ifieldplot += 1
            plot_Earth(
                fig=fig,
                ax=ax_stdb,
                field=background_gp_std_npy[ifield, 0],
                vminmax=vminmax_std[ifield],
                cmap='terrain_r',
                cbar_extend='max',
                cbar_shrink=cbar_shrink,
                title='Std of background (%s%s)' % (qty_all[ifield], unit[ifield])
            )


    # (D) std of analysis
    plot_bottom -= dheight_plot + dheight_buffer

    ifieldplot = 0
    for ifield in range(max_num_fields):
        if ifield in plot_qty_idx:
            ax_stda = fig.add_axes([plot_left + ifieldplot / num_fields, plot_bottom, plot_width, dheight_plot], projection=projections[projection])
            ifieldplot += 1
            plot_Earth(
                fig=fig,
                ax=ax_stda,
                field=analysis_gp_std_npy[ifield, 0],
                vminmax=vminmax_std[ifield],
                cmap='terrain_r',
                cbar_extend='max',
                cbar_shrink=cbar_shrink,
                title='Std of analysis (%s%s)' % (qty_all[ifield], unit[ifield])
            )




    # (E) ratio between abs(analysis increment) and analysis std
    plot_bottom -= dheight_plot + dheight_buffer

    ifieldplot = 0
    for ifield in range(max_num_fields):
        if ifield in plot_qty_idx:
            ax_stdrinc = fig.add_axes([plot_left + ifieldplot / num_fields, plot_bottom, plot_width, dheight_plot],
                                   projection=projections[projection])
            ifieldplot += 1
            ratio_to_plot = np.abs(analysis_gp_mean_npy[ifield, 0] - background_gp_mean_npy[ifield, 0]) / \
                            analysis_gp_std_npy[ifield, 0]

            vminmaxr = [0, np.amax(ratio_to_plot)]
            plot_Earth(
                fig=fig,
                ax=ax_stdrinc,
                field=ratio_to_plot,
                vminmax=vminmaxr,
                cmap='gnuplot2_r',
                cbar_extend='max',
                cbar_shrink=cbar_shrink,
                title='abs(ana. inc.) / ana. std. (%s)' % qty_all[ifield]
            )



    # ------------------------------------------------------
    # SAVE FIGURE AS A WHOLE
    # ------------------------------------------------------



    fig.savefig(savefig_name + ".jpg", dpi=100)#, dpi=300
    finish = datetime.now()
    print(str(finish - start))
    print('saved gigantic jpg')




    # ------------------------------------------------------
    # SPECIFIC FIGURES FOR EACH EXPERIMENT
    # ------------------------------------------------------
    if args.plot_singles:
        matplotlib.rcParams.update({"font.size": 16})
        obs_s = 120

        if 'Ljubljana' in args.custom_addon:
            # Separately plot triplets Z250-U200-V200, Z500-U500-V500, Z700-U900-V900
            analysis_increment_U500 = analysis_gp_mean_npy[qty_all.index('U500'), 0] - background_gp_mean_npy[
                qty_all.index('U500'), 0]
            analysis_increment_U700 = analysis_gp_mean_npy[qty_all.index('U700'), 0] - background_gp_mean_npy[
                qty_all.index('U700'), 0]
            analysis_increment_V500 = analysis_gp_mean_npy[qty_all.index('V500'), 0] - background_gp_mean_npy[
                qty_all.index('V500'), 0]
            analysis_increment_Z500 = analysis_gp_mean_npy[qty_all.index('Z500'), 0] - background_gp_mean_npy[
                qty_all.index('Z500'), 0]
            analysis_increment_Z700 = analysis_gp_mean_npy[qty_all.index('Z700'), 0] - background_gp_mean_npy[
                qty_all.index('Z700'), 0]

            analysis_increment_T500 = analysis_gp_mean_npy[qty_all.index('T500'), 0] - background_gp_mean_npy[
                qty_all.index('T500'), 0]
            analysis_increment_T2m = analysis_gp_mean_npy[qty_all.index('T2m'), 0] - background_gp_mean_npy[
                qty_all.index('T2m'), 0]
            analysis_increment_mslp = analysis_gp_mean_npy[qty_all.index('mslp'), 0] - background_gp_mean_npy[
                qty_all.index('mslp'), 0]


            title = 'Ana. inc. %s, %s, and %s' % ('Z500', 'U500', 'V500')
            if args.True_EDA and not 'CLIM_B' in args.custom_addon:
                fig2 = plt.figure(figsize=(6 * 1.5 * 1.06, 4 * 1.5 * 1.06))
                qs = 1e1
                qkp = [0.95, 0.05, 0.2]
                vmm = [v/10 for v in vminmax_inc[qty_all.index('Z500')]]
                # cbs = 0.55
                title += r', $\mathbf{B}_z^{\mathrm{EDA}}$'
            else:
                if 'CLIM_B' in args.custom_addon:
                    fig2 = plt.figure(figsize=(6 * 1.5 * 1.06, 4 * 1.5 * 1.06))
                    # cbs = 0.55
                    title += r', $\mathbf{B}_z^{clim}$'
                else:
                    fig2 = plt.figure(figsize=(6*2*1.12, 4*2*1.12))
                    # cbs = 0.4
                qs = 5e1
                qkp = [0.95, 0.05, 1]
                vmm = vminmax_inc[qty_all.index('Z500')]

            ax2 = fig2.add_subplot(1, 1, 1, projection=projections['NearsidePerspective'])

            plot_Earth(
                fig=fig2,
                ax=ax2,
                field=analysis_increment_Z500,
                U=analysis_increment_U500,
                V=analysis_increment_V500,
                quiver_scale=qs,
                quiverkey_props=qkp,
                vminmax=vmm,
                cmap=cmaps_inc[qty_all.index('Z500')],
                cbar_extend='both',
                # cbar_shrink=cbs,
                unit='%s%s' % ('Z500', unit[qty_all.index('Z500')]),
                unit_in_cbar=True,
                obs_locs=obs_lons_lats,
                obs_s=obs_s,
                title=title
            )
            fig2.savefig(savefig_name + '_inc_ZUV500' + '.jpg', dpi=300)
            print('saved zuv500')

            fig2 = plt.figure(figsize=(6, 4))
            ax2 = fig2.add_subplot(1, 1, 1, projection=projections['NearsidePerspective'])
            plot_Earth(
                fig=fig2,
                ax=ax2,
                field=analysis_increment_T500,
                vminmax=vminmax_inc[qty_all.index('T500')],
                cmap=cmaps_inc[qty_all.index('T500')],
                cbar_extend='both',
                unit='%s%s' % ('T500', unit[qty_all.index('T500')]),
                unit_in_cbar=True,
                obs_locs=obs_lons_lats,
                obs_s=obs_s,
                title='Ana. inc. %s' % ('T500')
            )
            fig2.savefig(savefig_name + '_inc_T500' + '.jpg', dpi=300)
            print('saved T500')


            fig2 = plt.figure(figsize=(6, 4))
            ax2 = fig2.add_subplot(1, 1, 1, projection=projections['NearsidePerspective'])
            plot_Earth(
                fig=fig2,
                ax=ax2,
                field=analysis_increment_T2m,
                vminmax=vminmax_inc[qty_all.index('T2m')],
                cmap=cmaps_inc[qty_all.index('T2m')],
                cbar_extend='both',
                unit='%s%s' % ('T2m', unit[qty_all.index('T2m')]),
                unit_in_cbar=True,
                obs_locs=obs_lons_lats,
                obs_s=obs_s,
                title='Ana. inc. %s and %s' % ('T2m', 'MSLP'),
                extra_contour_field=analysis_increment_mslp,
                extra_contour_levels=[-0.45, -0.3, -0.15, 0.15-1e-4, 0.3-1e-4, 0.45],#[-0.5, -0.25, 0.25, 0.5],
                extra_contour_colors=['darkviolet', 'darkviolet'],
                extra_contour_linestyles=['dashed', 'solid'],
                extra_contour_linewidths=2.5,
            )
            fig2.savefig(savefig_name + '_inc_T2m_mslp' + '.jpg', dpi=300)
            print('saved T2m mslp')


            # THERMAL WIND

            corilats = [[ilat for ilon in np.arange(-180., 179. + 1e-5, step=1.)] for ilat in
                        np.arange(-89.5, 89.5 + 1e-5, step=1.)]
            coripar = 2 * 2 * np.pi / (24 * 60 * 60) * np.radians(corilats)

            dlat = 110.6e3  # 1 degree lat in meters
            R = 287

            analysis_increment_dPhi_500_700 = (analysis_increment_Z500 - analysis_increment_Z700) / field_adjustment_multpilicators[
                qty_all.index('Z500')]

            derivative_in_y_500_700 = np.gradient(analysis_increment_dPhi_500_700, dlat)[
                0]  # Tested the correctness of index 0 on a simple case (gradient of corilats)
            thermal_U_phi500_700 = 1 / coripar * (- derivative_in_y_500_700)

            if args.True_EDA:
                vmmtw = [-0.15, 0.15]
                cbtickstw = [-0.1, 0, 0.1]
            else:
                vmmtw = vminmax_inc[qty_all.index('U500')]
                cbtickstw = [-1, 0, 1]

            fig2 = plt.figure(figsize=(6, 4))
            ax2 = fig2.add_subplot(1, 1, 1, projection=projections['NearsidePerspective'])
            plot_Earth(
                fig=fig2,
                ax=ax2,
                field=analysis_increment_U500 - analysis_increment_U700,
                vminmax=vmmtw,
                cmap=cmaps_inc[qty_all.index('U500')],
                cbar_extend='both',
                obs_locs=obs_lons_lats,
                obs_s=obs_s,
                title='Analysis increment',
                unit=r'$\mathrm{U500}-\mathrm{U700}$%s' % (unit[qty_all.index('U500')]),#'%s%s' % ('T500', unit[qty_all.index('T500')]),
                unit_in_cbar=True,#True,
                cbar_ticks=cbtickstw,
            )
            fig2.savefig(savefig_name + '_ana_inc_U500-U700' + '.jpg', dpi=300)
            print('saved ana inc U500-U700')


            fig2 = plt.figure(figsize=(6, 4))
            ax2 = fig2.add_subplot(1, 1, 1, projection=projections['NearsidePerspective'])
            plot_Earth(
                fig=fig2,
                ax=ax2,
                field=thermal_U_phi500_700,
                vminmax=vmmtw,
                cmap=cmaps_inc[qty_all.index('U500')],
                cbar_extend='both',
                obs_locs=obs_lons_lats,
                obs_s=obs_s,
                title='Thermal wind approx.',
                unit=r'$\mathrm{U500}-\mathrm{U700}$%s' % (unit[qty_all.index('U500')]),#'%s%s' % ('T500', unit[qty_all.index('T500')]),
                unit_in_cbar=True,#True,
                cbar_ticks=cbtickstw,
            )
            fig2.savefig(savefig_name + '_therm_U500-U700' + '.jpg', dpi=300)
            print('saved therm wind U500-U700')


            fig2 = plt.figure(figsize=(6, 4))
            ax2 = fig2.add_subplot(1, 1, 1, projection=projections['NearsidePerspective'])
            plot_Earth(
                fig=fig2,
                ax=ax2,
                field=(analysis_increment_U500 - analysis_increment_U700) - thermal_U_phi500_700,
                # field=np.abs(thermal_U_phi500_700 / (analysis_increment_U500 - analysis_increment_U700)),
                vminmax=vmmtw,
                # cmap='pink',
                cmap=cmaps_inc[qty_all.index('U500')],
                cbar_extend='both',
                obs_locs=obs_lons_lats,
                obs_s=obs_s,
                # title='ratio 500 700'
                title='Difference',
                unit=r'$\mathrm{U500}-\mathrm{U700}$%s' % (unit[qty_all.index('U500')]),#'%s%s' % ('T500', unit[qty_all.index('T500')]),
                unit_in_cbar=True,#True,
                cbar_ticks=cbtickstw,
            )
            fig2.savefig(savefig_name + '_diff_therm_inc' + '.jpg', dpi=300)
            print('saved therm wind diff')


            # Ana inc. Z500 on top of background U500 and V500
            title = 'Ana. inc. Z500'
            if args.True_EDA:
                if 'CLIM_B' in args.custom_addon:
                    title += r', $\mathbf{B}_{z}^{clim}$'
                else:
                    title += r', $\mathbf{B}_{z}^{\mathrm{EDA}}$'
            else:
                title += f', {args.obs_datetime[:-3]}'

            fig2 = plt.figure(figsize=(6 * 1.5 * 0.85, 4 * 1.5 * 0.85))  # * 2 * 0.9
            ax2 = fig2.add_subplot(1, 1, 1, projection=ccrs.Miller(central_longitude=14.5))
            if (not args.True_EDA) or ('CLIM_B' in args.custom_addon):
                ct = [-15, -10, -5, 0, 5, 10, 15]
                cl = np.arange(-15, 15.1, step=2)
                vmm = [-16, 16]
            else:
                #ct = [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5]
                ct = [-2, -1, 0, 1, 2]
                cl = np.arange(-2.2, 2.21, step=0.4)
                vmm = [-2.5, 2.5]

            plot_Earth(
                fig=fig2,
                ax=ax2,
                field=analysis_increment_Z500,
                U=background_gp_mean_npy[qty_all.index('U500'), 0],
                V=background_gp_mean_npy[qty_all.index('V500'), 0],
                plot_type='contourf',
                contourf_levels=cl,
                cbar_ticks=ct,
                quiver_scale=6e2,
                quiver_reduction=3,
                quiverkey_props=[1.15, 0.06, 20],  # 1.125
                vminmax=vmm,#vminmax_inc[qty_all.index('Z500')],
                cmap=cmaps_inc[qty_all.index('Z500')],
                cbar_extend='both',
                cbar_shrink=0.60,
                unit='%s%s' % ('Z500', unit[qty_all.index('Z500')]),
                unit_in_cbar=True,
                obs_locs=obs_lons_lats,
                obs_s=obs_s,
                title=title,
                # title='%s after %dh' % ('TCWV', timestep * single_step_len),
                coordinate_labels=True,
                gridline_y_distance=20,
                gridline_x_distance=20,
                extra_contour_field=background_gp_mean_npy[qty_all.index('Z500'), 0],
                extra_contour_levels=[4000 + 200 * i for i in range(31)],
                extra_contour_linestyles=[':',':'],
                extra_contour_colors=['C0','C0'],
            )
            ax2.set_extent([-25, 55, 25, 71], crs=ccrs.PlateCarree())
            # print(ax2.get_xlim())
            fig2.savefig(savefig_name + '_inc_Z500_bg_UV500' + '.jpg', dpi=300)
            print('saved Miller')

            ratio_to_plot = np.abs(analysis_increment_T500) / analysis_gp_std_npy[qty_all.index('T500'), 0]

            # INTERPOLATE ratio_to_plot TO OBS LOC!
            lats = np.array([[args.singobs_lat]]).astype(np.float32)
            lons = np.array([[args.singobs_lon]]).astype(np.float32)

            # Format of input and output of AE: idx ... location
            # LAT: 0 ... 89.5 degS, 179 ... 89.5 degN
            # LON: 0 ... 180 degW, 359 ... 179 degE
            # Format of input to this script (args.singobs_lat, args.singobs_lon): value ... location
            # LAT: -90 ... 90 degS, 90 ... 90 degN
            # LON: -179.99 ... 179.99 degW, 179.99 ... 179.99 degE
            # Now we need to convert the input locations to this script into their coordinates for AE input/output
            # First: latitudes: -89.5 -> 0, 89.5 -> 179
            # WARNING: Avoid using lats < -89.5 and lats > 89.5, as these cannot be interpolated correctly using F.grid_sample!
            lats_AE = lats + 89.5
            # Second: longitudes: -180 -> 0, 179 -> 359
            # WARNING: Avoid using lons > 179, as these cannot be interpolated correctly using F.grid_sample!
            lons_AE = lons + 180
            # Normalize lons_AE and lats_AE to [-1, 1]
            lats_AE_normalized = lats_AE / 179 * 2 - 1  # WARNING: lats < -89.5 and lats > 89.5 are outside of [-1, 1] range!
            lons_AE_normalized = lons_AE / 359 * 2 - 1  # WARNING: lons > 179 are outside of [-1, 1] range!

            # Stack normalized lats and lons, reshape them to (Nobs, 2), and convert them to torch tensor.
            obs_locs_torch = torch.from_numpy(np.reshape(np.stack((lats_AE_normalized, lons_AE_normalized), axis=-1),
                                                         (np.shape(lats)[0] * np.shape(lats)[1], 2)))
            # After lots of testing I figured out that, even though the original script was written with (at least to my knowledge)
            # complete accordance with F.grid_sample documentation, for some reason the interpolation is performed correctly
            # only if I swap the columns of longitudes and latitudes... This part of code could definetly be written in a better way,
            # but I started with what ChatGPT gave me and then gradually modified it until it gave correct results. The 5 strange lines are given below:
            obs_locs_torch1 = torch.from_numpy(np.zeros(shape=obs_locs_torch.shape).astype(np.float32))
            obs_locs_torch1[:, 0], obs_locs_torch1[:, 1] = obs_locs_torch[:, 1], obs_locs_torch[:, 0]
            obs_locs_torch = obs_locs_torch1
            del obs_locs_torch1
            gc.collect()

            # Reshaping locs_to_interpolate so it is really suitable for interpolation
            obs_locs_torch = obs_locs_torch.unsqueeze(0).expand(in_out_ch, -1, -1)  # Shape: (in_out_ch, Nobs, 2)
            obs_locs_torch = obs_locs_torch.unsqueeze(2)  # shape (in_out_ch, Nobs, 1, 2), suitable for interpolation

            to_be_interpolated = torch.from_numpy(
                np.array([[ratio_to_plot] for i in range(in_out_ch)]).astype(np.float32))
            # print('to_be_interpolated.shape', to_be_interpolated.shape)

            interpolated_ratio_at_obs_loc = F.grid_sample(to_be_interpolated, obs_locs_torch, mode='bilinear',
                                                          align_corners=True)  # Shape: (in_out_ch, 1, Nobs, 1)

            # print('max ratio', np.amax(ratio_to_plot / interpolated_ratio_at_obs_loc.numpy()[qty_all.index('Z500')]))



            # Ana inc. Z500 on top of background U500 and V500
            title = 'Obs. impact on Z500'
            if args.True_EDA:
                if 'CLIM_B' in args.custom_addon:
                    title += r', $\mathbf{B}_{z}^{clim}$'
                else:
                    title += r', $\mathbf{B}_{z}^{\mathrm{EDA}}$'
            else:
                title += f', {args.obs_datetime[:-3]}'

            fig2 = plt.figure(figsize=(6 * 1.5 * 0.85, 4 * 1.5 * 0.85))  # * 2 * 0.9
            ax2 = fig2.add_subplot(1, 1, 1, projection=ccrs.Miller(central_longitude=14.5))
            plot_Earth(
                fig=fig2,
                ax=ax2,
                field=ratio_to_plot / interpolated_ratio_at_obs_loc.numpy().squeeze()[qty_all.index('Z500')],
                U=background_gp_mean_npy[qty_all.index('U500'), 0],
                V=background_gp_mean_npy[qty_all.index('V500'), 0],
                plot_type='contourf',
                contourf_levels=np.arange(0.05, 0.951, step=0.1),
                cbar_ticks=[0.25, 0.5, 0.75],
                quiver_scale=5e2,
                quiver_reduction=3,
                quiverkey_props=[1.15, 0.06, 20],  # 1.125
                vminmax=[0, 1],  # vminmax_inc[qty_all.index('Z500')],
                cmap='gnuplot2_r',
                cbar_extend='both',
                cbar_shrink=0.60,
                unit='Norm. relat. impact', #'%s%s' % ('Z500', unit[qty_all.index('Z500')]),
                unit_in_cbar=True,
                obs_locs=obs_lons_lats,
                obs_s=120,
                title=title,
                coordinate_labels=True,
                gridline_y_distance=20,
                gridline_x_distance=20,
            )
            ax2.set_extent([-25, 55, 25, 71], crs=ccrs.PlateCarree())
            fig2.savefig(savefig_name + '_inc_Z500_ratio_bg_UV500' + '.jpg', dpi=300)
            print('saved Miller')




        if 'TCWV' in args.savefig_dir:
            analysis_increment_T2m = analysis_gp_mean_npy[qty_all.index('T2m'), 0] - background_gp_mean_npy[
                qty_all.index('T2m'), 0]
            analysis_increment_TCWV = analysis_gp_mean_npy[qty_all.index('TCWV'), 0] - background_gp_mean_npy[
                qty_all.index('TCWV'), 0]

            bg_U500 = background_gp_mean_npy[qty_all.index('U500'), 0] # background_gp_mean_npy_with_clim[qty_all.index('U500'), 0]
            bg_V500 = background_gp_mean_npy[qty_all.index('V500'), 0] # background_gp_mean_npy_with_clim[qty_all.index('V500'), 0]

            corilats = [[ilat for ilon in np.arange(-180., 179.+1e-5, step=1.)] for ilat in np.arange(-89.5, 89.5+1e-5, step=1.)]
            coripar = 2 * 2*np.pi / (24 * 60 * 60) * np.radians(corilats)

            dlat = 110.6e3 # 1 degree lat in meters
            a = np.cos(np.radians(corilats)) ** 2 * np.sin(np.radians(1) / 2) ** 2
            dlon = 6.357e6 * 2 * np.arctan(np.sqrt(a) / np.sqrt(1 - a))#[:,0]
            # print('dlon.shape', dlon.shape)

            analysis_increment_U900 = analysis_gp_mean_npy[qty_all.index('U900'), 0] - background_gp_mean_npy[qty_all.index('U900'), 0]
            analysis_increment_V900 = analysis_gp_mean_npy[qty_all.index('V900'), 0] - background_gp_mean_npy[qty_all.index('V900'), 0]
            derivative_V900_lat = np.gradient(analysis_increment_V900, dlat)[0]   # Tested the correctness of index 0 on a simple case (gradient of corilats)
            derivative_U900_lon = [np.gradient(analysis_increment_U900[ilat], dlon[ilat, 0]) for ilat in range(len(analysis_increment_U900))]

            div_900 = derivative_V900_lat + np.array(derivative_U900_lon)



            derivative_U900_lat = np.gradient(analysis_increment_U900, dlat)[0]
            derivative_V900_lon = [np.gradient(analysis_increment_V900[ilat], dlon[ilat, 0]) for ilat in range(len(analysis_increment_V900))]
            vort_900 = np.array(derivative_V900_lon) - derivative_U900_lat
            vort_to_div_900_ratio = vort_900 / div_900



            analysis_increment_U200 = analysis_gp_mean_npy[qty_all.index('U200'), 0] - background_gp_mean_npy[qty_all.index('U200'), 0]
            analysis_increment_V200 = analysis_gp_mean_npy[qty_all.index('V200'), 0] - background_gp_mean_npy[qty_all.index('V200'), 0]
            derivative_V200_lat = np.gradient(analysis_increment_V200, dlat)[0]   # Tested the correctness of index 0 on a simple case (gradient of corilats)
            derivative_U200_lon = [np.gradient(analysis_increment_U200[ilat], dlon[ilat, 0]) for ilat in range(len(analysis_increment_U200))]

            div_200 = derivative_V200_lat + np.array(derivative_U200_lon)


            derivative_U200_lat = np.gradient(analysis_increment_U200, dlat)[0]
            derivative_V200_lon = [np.gradient(analysis_increment_V200[ilat], dlon[ilat, 0]) for ilat in
                                   range(len(analysis_increment_V200))]
            vort_200 = np.array(derivative_V200_lon) - derivative_U200_lat



            fig2 = plt.figure(figsize=(6, 4))
            ax2 = fig2.add_subplot(1, 1, 1, projection=projections['NearsidePerspective'])
            plot_Earth(
                fig=fig2,
                ax=ax2,
                field=analysis_increment_TCWV,
                # U=bg_U500,
                # V=bg_V500,
                # quiver_color='C0',
                # quiver_reduction=3,
                # quiver_scale=2e2,
                vminmax=vminmax_inc[qty_all.index('TCWV')],
                cmap=cmaps_inc[qty_all.index('TCWV')],
                cbar_extend='both',
                unit='%s%s' % ('TCWV', unit[qty_all.index('TCWV')]),
                unit_in_cbar=True,  # True,
                obs_locs=obs_lons_lats,
                obs_s=obs_s,
                title='Ana. inc. %s' % ('TCWV')
            )
            fig2.savefig(savefig_name + '_ana_inc_TCWV' + '.jpg', dpi=300)
            print('saved ana inc tcwv')


            fig2 = plt.figure(figsize=(6*1.5*1.06, 4*1.5*1.06))
            ax2 = fig2.add_subplot(1, 1, 1, projection=projections['NearsidePerspective'])
            plot_Earth(
                fig=fig2,
                ax=ax2,
                field=div_900 * 1e6,
                U=analysis_increment_U900,
                V=analysis_increment_V900,
                quiver_scale=2e1,
                vminmax=[-3, 3],
                cmap='PuOr',
                cbar_extend='both',
                # cbar_shrink=0.55,
                unit=r'Divergence [$10^{-6}\mathrm{s}^{-1}$]',
                unit_in_cbar=True,
                obs_locs=obs_lons_lats,
                obs_s=obs_s,
                title='Ana. inc. %s and %s' % ('U900', 'V900'),
                extra_contour_field=vort_900 * 1e6,
                extra_contour_levels=[-3, -2.5, -2, -1.5, -1, -0.5, 0.5, 1, 1.5, 2, 2.5, 3],
                extra_contour_linewidths=2.5,
            )
            fig2.savefig(savefig_name + '_ana_inc_900' + '.jpg', dpi=300)
            print('saved ana inc 900')

            fig2 = plt.figure(figsize=(6 * 1.5 * 1.06, 4 * 1.5 * 1.06))
            ax2 = fig2.add_subplot(1, 1, 1, projection=projections['NearsidePerspective'])
            plot_Earth(
                fig=fig2,
                ax=ax2,
                field=div_200 * 1e6,
                U=analysis_increment_U200,
                V=analysis_increment_V200,
                quiver_scale=2e1,
                vminmax=[-3, 3],
                cmap='PuOr',
                cbar_extend='both',
                unit=r'Divergence [$10^{-6}\mathrm{s}^{-1}$]',
                unit_in_cbar=True,
                obs_locs=obs_lons_lats,
                obs_s=obs_s,
                title='Ana. inc. %s and %s' % ('U200', 'V200'),
                extra_contour_field=vort_200 * 1e6,
                extra_contour_levels=[-3, -2.5, -2, -1.5, -1, -0.5, 0.5, 1, 1.5, 2, 2.5, 3],
                extra_contour_linewidths=2.5,
            )
            fig2.savefig(savefig_name + '_ana_inc_200' + '.jpg', dpi=300)
            print('saved ana inc 900')


            fig2 = plt.figure(figsize=(6, 4))
            ax2 = fig2.add_subplot(1, 1, 1, projection=projections['NearsidePerspective'])
            plot_Earth(
                fig=fig2,
                ax=ax2,
                field=analysis_increment_T2m,
                vminmax=[-0.5, 0.5],  # vminmax_inc[qty_all.index('T500')],
                cmap=cmaps_inc[qty_all.index('T2m')],
                cbar_extend='both',
                unit='%s%s' % ('T2m', unit[qty_all.index('T2m')]),
                unit_in_cbar=True,  # True,
                obs_locs=obs_lons_lats,
                obs_s=obs_s,
                title='Ana. inc. %s' % ('T2m'),
                cbar_ticks=[-0.5,0,0.5]
            )
            fig2.savefig(savefig_name + '_ana_inc_T2m' + '.jpg', dpi=300)
            print('saved ana inc t2m')


        # --------------------------------------------
        # VERTICAL CROSS-SECTIONS
        # --------------------------------------------
        print('Vertical cross-sections')

        from scipy.ndimage import gaussian_filter

        def nice_lon_lat(input_lon_or_lat, type):
            narrow_space = '\u200A'
            if input_lon_or_lat == 0:
                return '0 °'
            if type == 'lat':
                if input_lon_or_lat > 0:
                    return f'{input_lon_or_lat}{narrow_space}°N'
                else:
                    return f'{abs(input_lon_or_lat)}' + r'$\,\degree\mathrm{S}$'
            else:
                if input_lon_or_lat < 0:
                    return f'{abs(input_lon_or_lat)}{narrow_space}°W'
                else:
                    return f'{input_lon_or_lat}{narrow_space}°E'

        def nice_xticks(input_lons_or_lats, xtype='lons', custom_dx=None):
            if not custom_dx:
                if xtype == 'lats':
                    dx = 10
                else:
                    dx = 20
            else:
                dx = custom_dx

            multiples = list(range((int(np.ceil(input_lons_or_lats[0])) + dx-1) // dx * dx,
                                   int(np.floor(input_lons_or_lats[-1])) + 1, dx))
            return_ticks = multiples
            return_labels = []
            for mt in multiples:
                if xtype == 'lons':
                    return_labels.append(nice_lon_lat(mt, type='lon'))
                else:
                    return_labels.append(nice_lon_lat(mt, type='lat'))

            return return_ticks, return_labels


        alllons = np.linspace(-180, 179, 360)
        alllats = np.linspace(-89.5, 89.5, 180)
        closest_lat_idx = (np.abs(alllats - args.singobs_lat)).argmin()
        closest_lon_idx = (np.abs(alllons - args.singobs_lon)).argmin()


        qties = {'Z': {'indices': [0, 1, 2, 3], 'height': [250, 500, 700, 850], 'unit': 'm', 'height_labels': ['250 hPa', '500 hPa', '700 hPa', '850 hPa'], 'long_name':'Geopotential height'},
                 # 'T': {'indices': [7, 8, 6], 'height': [500, 850, 1000], 'unit': r'$\degree$C'},
                 'U': {'indices': [11, 12, 13, 14, 10], 'height': [200, 500, 700, 900, 1000], 'unit': 'm/s', 'height_labels': ['200 hPa', '500 hPa', '700 hPa', '900 hPa', '10 m'], 'long_name':'Zonal wind'},
                 'V': {'indices': [16, 17, 18, 19, 15], 'height': [200, 500, 700, 900, 1000], 'unit': 'm/s', 'height_labels': ['200 hPa', '500 hPa', '700 hPa', '900 hPa', '10 m'], 'long_name':'Meridional wind'}
                 }
        if 'Ljubljana' in args.custom_addon:
            subplot_idx = 0
            fig = plt.figure(figsize=(8.5*3,6*2))
            for qty in qties.keys():
                # ANALYSIS INCREMENTS
                subplot_idx += 1
                plt.subplot(3, 2, subplot_idx)
                ana_inc_values = np.array([analysis_gp_mean_npy[idx] - background_gp_mean_npy[idx] for idx in
                                  qties[qty]['indices']]).squeeze()


                # Apply Gaussian filter to smooth contours
                ana_inc_values = np.array([gaussian_filter(ana_inc_values[ih], sigma=1, mode='wrap') for ih in range(len(ana_inc_values))])

                if qty == 'U': # cross-section in N-S
                    lats, heights = np.meshgrid(
                    np.arange(np.floor(args.singobs_lat) - 20 + 0.5, np.floor(args.singobs_lat) + 20.1 + 0.5, step=1),  # +0.5 as lats go -89.5, ... 89.5
                        np.array(qties[qty]['height'])
                    )
                    plt.contourf(lats, heights, ana_inc_values[:,closest_lat_idx-20:closest_lat_idx+20+1,closest_lon_idx], cmap=cmaps_inc[qty_all.index('U200')], levels=np.arange(-1.75,1.76, step=0.5), extend='both')

                    plt.colorbar(shrink=0.8, pad=0.05, extend='both', label=f"Analysis increment [{qties[qty]['unit']}]")

                    plt.xticks(
                        ticks=nice_xticks(alllats[closest_lat_idx-20:closest_lat_idx+20+1], xtype='lats')[0],
                        labels=nice_xticks(alllats[closest_lat_idx-20:closest_lat_idx+20+1], xtype='lats')[1],
                    )
                    # plt.title(f"{qties[qty]['long_name']} at $\lambda={alllons[closest_lon_idx]}$°")
                    plt.title(f"{qties[qty]['long_name']} at {nice_lon_lat(alllons[closest_lon_idx], 'lon')}")
                    plt.xlabel('Latitude')
                    plt.gca().invert_xaxis()
                    try:
                        plt.scatter(args.singobs_lat, int(args.obs_qty[1:]), marker='*', c='gold', edgecolors='k', s=obs_s)
                    except:
                        plt.axvline(args.singobs_lat, color='gold')

                else: # cross-section in E-W
                    extent_lon = np.degrees(3000 / (6371.0 * np.cos(np.radians(args.singobs_lat)))) # longitudinal extent that is equal to 3000km
                    lons, heights = np.meshgrid(
                    np.arange(np.floor(args.singobs_lon) - int(extent_lon)-1, np.floor(args.singobs_lon) + int(extent_lon)+1 +0.1, step=1),
                        np.array(qties[qty]['height'])
                    )
                    if qty == 'Z':
                        plt.contourf(lons, heights, ana_inc_values[:, closest_lat_idx, closest_lon_idx- int(np.floor(extent_lon))-1:closest_lon_idx+ int(np.floor(extent_lon))+1+1], cmap=cmaps_inc[qty_all.index('Z250')], levels=np.arange(-18, 18.1, step=4), extend='both')
                    elif qty == 'V':
                        plt.contourf(lons, heights, ana_inc_values[:, closest_lat_idx, closest_lon_idx- int(np.floor(extent_lon))-1:closest_lon_idx+ int(np.floor(extent_lon))+1+1], cmap=cmaps_inc[qty_all.index('V500')], levels=np.arange(-1.75,1.76, step=0.5), extend='both')

                    plt.colorbar(shrink=0.8, pad=0.05, extend='both', label=f"Analysis increment [{qties[qty]['unit']}]")

                    plt.fill_betweenx(np.linspace(1000, 850, 50), alllons[closest_lon_idx- int(np.floor(extent_lon))-1:closest_lon_idx+ int(np.floor(extent_lon))+1+1].min(), alllons[closest_lon_idx- int(np.floor(extent_lon))-1:closest_lon_idx+ int(np.floor(extent_lon))+1+1].max(), color='none', hatch='xxxx', edgecolor='gray', zorder=-1)
                    plt.fill_betweenx(np.linspace(250, 200, 50), alllons[closest_lon_idx- int(np.floor(extent_lon))-1:closest_lon_idx+ int(np.floor(extent_lon))+1+1].min(), alllons[closest_lon_idx- int(np.floor(extent_lon))-1:closest_lon_idx+ int(np.floor(extent_lon))+1+1].max(), color='none', hatch='xxxx', edgecolor='gray', zorder=-1)

                    plt.xticks(
                        ticks=nice_xticks(alllons[closest_lon_idx- int(np.floor(extent_lon))-1:closest_lon_idx+ int(np.floor(extent_lon))+1+1], xtype='lons')[0],
                        labels=nice_xticks(alllons[closest_lon_idx- int(np.floor(extent_lon))-1:closest_lon_idx+ int(np.floor(extent_lon))+1+1], xtype='lons')[1],
                    )

                    plt.title(f"{qties[qty]['long_name']} at {nice_lon_lat(alllats[closest_lat_idx], 'lat')}")
                    plt.xlim(args.singobs_lon - extent_lon, args.singobs_lon + extent_lon)
                    plt.xlabel('Longitude')
                    try:
                        plt.scatter(args.singobs_lon, int(args.obs_qty[1:]), marker='*', c='gold', edgecolors='k', s=obs_s)
                    except:
                        plt.axvline(args.singobs_lon, color='gold')


                plt.yticks(qties[qty]['height'], qties[qty]['height_labels'])
                plt.grid(axis='y')
                plt.ylim(1000, 200)
                plt.ylabel('Level')
                plt.tight_layout()


                # RELATIVE IMPACT

                subplot_idx += 1
                plt.subplot(3, 2, subplot_idx)
                ana_inc_values = np.array([analysis_gp_mean_npy[idx] - background_gp_mean_npy[idx] for idx in
                                  qties[qty]['indices']]).squeeze()
                ana_std_values = np.array([analysis_gp_std_npy[idx] for idx in
                                  qties[qty]['indices']]).squeeze()

                normalizing_value_impact = (analysis_gp_at_obs_loc_mean[obs_qty_idx[0]] - background_gp_at_obs_loc_mean[obs_qty_idx[0]]) / analysis_gp_at_obs_loc_std[obs_qty_idx[0]]


                impact = np.abs(ana_inc_values) / ana_std_values / normalizing_value_impact

                # Apply Gaussian filter to smooth contours
                impact = gaussian_filter(impact, sigma=1, mode='wrap')

                if qty == 'Z' and args.obs_qty == 'Z500':
                    normalizing_value_impact = impact[1, closest_lat_idx, closest_lon_idx]
                    impact = impact / normalizing_value_impact


                # Given that Gaussian filtering has been applied, we might get values here that are slightly higher than 1, i.e. 1 + O(0.01).
                # These values that are larger than 1 are just a consequence of data manipulation, so we reset them to 1 before plotting
                # print('max impact after filtering', np.amax(impact))
                impact[impact > 1] = 1


                import matplotlib.colors
                # levels=11
                colors = plt.cm.gnuplot2_r(np.linspace(0,1, 10))
                cmap = matplotlib.colors.ListedColormap(colors, "", len(colors))

                if qty == 'U': # cross-section in N-S
                    lats, heights = np.meshgrid(
                    np.arange(np.floor(args.singobs_lat) - 20, np.floor(args.singobs_lat) + 20.1, step=1),
                        np.array(qties[qty]['height'])
                    )
                    plt.contourf(lats, heights, impact[:,closest_lat_idx-20:closest_lat_idx+20+1,closest_lon_idx], cmap=cmap, levels=np.linspace(0,1,11))

                    plt.colorbar(shrink=0.8, pad=0.05, extend='both', label=f"Normalized relative impact")

                    plt.xticks(
                        ticks=nice_xticks(alllats[closest_lat_idx-20:closest_lat_idx+20+1], xtype='lats')[0],
                        labels=nice_xticks(alllats[closest_lat_idx-20:closest_lat_idx+20+1], xtype='lats')[1],
                    )
                    plt.title(f"{qties[qty]['long_name']} at {nice_lon_lat(alllons[closest_lon_idx], 'lon')}")
                    plt.xlabel('Latitude')
                    plt.gca().invert_xaxis()
                    try:
                        plt.scatter(args.singobs_lat, int(args.obs_qty[1:]), marker='*', c='gold', edgecolors='k', s=obs_s)
                    except:
                        plt.axvline(args.singobs_lat, color='gold')

                else: # cross-section in E-W
                    extent_lon = 40
                    lons, heights = np.meshgrid(
                    np.arange(np.floor(args.singobs_lon) - extent_lon, np.floor(args.singobs_lon) + extent_lon +0.1, step=1),
                        np.array(qties[qty]['height'])
                    )
                    if qty == 'Z':
                        plt.contourf(lons, heights, impact[:, closest_lat_idx, closest_lon_idx-extent_lon:closest_lon_idx+extent_lon+1], cmap=cmap, levels=np.linspace(0,1,11))
                    elif qty == 'V':
                        plt.contourf(lons, heights, impact[:, closest_lat_idx, closest_lon_idx-extent_lon:closest_lon_idx+extent_lon+1], cmap=cmap, levels=np.linspace(0,1,11))

                    plt.colorbar(shrink=0.8, pad=0.05, extend='both', label=f"Normalized relative impact")

                    plt.fill_betweenx(np.linspace(1000, 850, 50), alllons[closest_lon_idx-extent_lon:closest_lon_idx+extent_lon+1].min(), alllons[closest_lon_idx-extent_lon:closest_lon_idx+extent_lon+1].max(), color='none', hatch='xxxx', edgecolor='gray', zorder=-1)
                    plt.fill_betweenx(np.linspace(250, 200, 50), alllons[closest_lon_idx-extent_lon:closest_lon_idx+extent_lon+1].min(), alllons[closest_lon_idx-extent_lon:closest_lon_idx+extent_lon+1].max(), color='none', hatch='xxxx', edgecolor='gray', zorder=-1)

                    plt.xticks(
                        ticks=nice_xticks(alllons[closest_lon_idx-extent_lon:closest_lon_idx+extent_lon+1], xtype='lons')[0],
                        labels=nice_xticks(alllons[closest_lon_idx-extent_lon:closest_lon_idx+extent_lon+1], xtype='lons')[1],
                    )
                    plt.title(f"{qties[qty]['long_name']} at {nice_lon_lat(alllats[closest_lat_idx], 'lat')}")
                    plt.xlabel('Longitude')
                    try:
                        plt.scatter(args.singobs_lon, int(args.obs_qty[1:]), marker='*', c='gold', edgecolors='k', s=obs_s)
                    except:
                        plt.axvline(args.singobs_lon, color='gold')


                plt.yticks(qties[qty]['height'], qties[qty]['height_labels'])
                plt.grid(axis='y')
                plt.ylim(1000, 200)
                plt.ylabel('Level')
                plt.tight_layout()



            fig.savefig(savefig_name + "_crosssection_ZUV_smoothed.pdf", dpi=300)

        elif 'TCWV' in args.savefig_dir:
            # print('declaring figure')
            fig = plt.figure(figsize=(6*15/18*121/127, 4))#figsize=(8,6))

            divs = []
            divs_at_obs_loc = []

            for iheight in range(len(qties['U']['height'])):
                print(iheight)
                analysis_increment_U = analysis_gp_mean_npy[qties['U']['indices'][iheight], 0] - background_gp_mean_npy[
                    qties['U']['indices'][iheight], 0]
                analysis_increment_V = analysis_gp_mean_npy[qties['V']['indices'][iheight], 0] - background_gp_mean_npy[
                    qties['V']['indices'][iheight], 0]

                corilats = [[ilat for ilon in np.arange(-180., 179. + 1e-5, step=1.)] for ilat in
                            np.arange(-89.5, 89.5 + 1e-5, step=1.)]
                coripar = 2 * 2 * np.pi / (24 * 60 * 60) * np.radians(corilats)

                dlat = 110.6e3  # 1 degree lat in meters
                a = np.cos(np.radians(corilats)) ** 2 * np.sin(np.radians(1) / 2) ** 2
                dlon = 6.357e6 * 2 * np.arctan(np.sqrt(a) / np.sqrt(1 - a))  # [:,0]

                derivative_V_lat = np.gradient(analysis_increment_V, dlat)[
                    0]  # Tested the correctness of index 0 on a simple case (gradient of corilats)
                derivative_U_lon = [np.gradient(analysis_increment_U[ilat], dlon[ilat, 0]) for ilat in
                                       range(len(analysis_increment_U))]

                div = derivative_V_lat + np.array(derivative_U_lon)
                # print('filtering')

                divs.append(gaussian_filter(div * 1e6, sigma=1, mode='wrap'))

                div_for_interpolation = torch.from_numpy(np.array([[div] for i in range(in_out_ch)])).to(torch.float32)
                div_interpolated_to_obs_loc = F.grid_sample(
                    div_for_interpolation, obs_locs_torch, mode='bilinear', align_corners=True
                ).numpy().squeeze()[0]
                divs_at_obs_loc.append(div_interpolated_to_obs_loc * 1e6)

            extent_lon = 15#20
            qty = 'V'
            lons, heights = np.meshgrid(
                np.arange(np.floor(args.singobs_lon) - extent_lon, np.floor(args.singobs_lon) + extent_lon + 0.1, step=1),
                np.array(qties[qty]['height'])
            )

            plt.contourf(lons, heights,
                         np.array(divs)[:, closest_lat_idx, closest_lon_idx - extent_lon:closest_lon_idx + extent_lon + 1],
                         cmap='PuOr', levels=np.arange(-3,3.01, step=0.4), extend='both')

            plt.colorbar(shrink=0.8, pad=0.05, extend='both', label=r'Divergence [$10^{-6}\mathrm{s}^{-1}$]', ticks=[-3, -2, -1, 0, 1, 2, 3])


            plt.xticks(
                ticks=nice_xticks(alllons[closest_lon_idx - extent_lon:closest_lon_idx + extent_lon + 1], xtype='lons', custom_dx=10)[0],
                labels=nice_xticks(alllons[closest_lon_idx - extent_lon:closest_lon_idx + extent_lon + 1], xtype='lons', custom_dx=10)[1],
            )
            # plt.title(f"Divergence at $\phi={alllats[closest_lat_idx]}$°")
            # plt.title(f"Divergence at {nice_lon_lat(alllats[closest_lat_idx], 'lat')}")
            plt.title(f"Ana. inc. at {nice_lon_lat(alllats[closest_lat_idx], 'lat')}", y=1.06)
            # plt.xlabel('Longitude')
            try:
                plt.scatter(args.singobs_lon, int(args.obs_qty[1:]), marker='*', c='gold', edgecolors='k', s=obs_s)
            except:
                plt.axvline(args.singobs_lon, color='gold')

            plt.yticks(qties[qty]['height'], qties[qty]['height_labels'])
            plt.grid(axis='y')
            plt.ylim(1000, 200)
            # plt.ylabel('Level')
            plt.tight_layout()

            fig.savefig(savefig_name + "_crosssection_divergence_smoothed.jpg", dpi=300)

            plt.clf()
            plt.cla()
            plt.plot(divs_at_obs_loc, qties[qty]['height'], 'ko-', clip_on=False)
            plt.yticks(qties[qty]['height'], qties[qty]['height_labels'])
            plt.grid(axis='y')
            plt.axvline(0, color='grey', linewidth=1)
            plt.ylim(1000, 200)
            plt.ylabel('Level')
            plt.title(f"Divergence at observation location")
            plt.xlabel(r'Divergence [$10^{-6}\mathrm{s}^{-1}$]')
            plt.savefig(savefig_name + "_div_at_obs_loc.jpg", dpi=300)



