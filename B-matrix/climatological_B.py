# -*- coding: utf-8 -*-
#!/usr/bin/python
"""
Script for computing the climatological B-matrix
"""

# --------------------------------------------------------------------------
# OSNOVNE KNJIZNICE
# --------------------------------------------------------------------------

# Osnovno
import os
import numpy as np
import torch
import torch.nn as nn


# Slike
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from datetime import *


# Moduli
import sys
master_dir = os.getenv('UGNN3DVar_master')
sys.path.append(master_dir)
sys.path.append(master_dir + '/NNs')
import unet_model_7x7_4x_depth7_interdepth_padding_1_degree_v01
import unet_model_7x7_4x_depth4_interdepth_padding_1_degree_v01_stride2b_no_skip_AE

# --------------------------------------------------------------------------
# DODATNE KNJIZNICE
# --------------------------------------------------------------------------
import argparse
import pickle

parser = argparse.ArgumentParser()
parser.add_argument("--start_date", help="Start date in format yyyy-mm-dd", type=str, required=False, default='2015-01-01')
parser.add_argument("--end_date", help="End date in format yyyy-mm-dd", type=str, required=False, default='2019-12-31')
parser.add_argument("--update_interval", help="Interval for saving recomputed B-matrix (in days)", type=int, default=30, required=False)
parser.add_argument("--restart_computation", help='True: restart computation for the entire interval. False: upload the B-matrix from the previous update and continue from there.', default=True, action=argparse.BooleanOptionalAction)
parser.add_argument("--days_in_month", help="Days in month used for B-matrix computation, e.g. 1_2_3 means 1st, 2nd, and 3rd day of each month will be used", type=str, required=False, default='1_6_11_16_21_26')
parser.add_argument("--hours_in_day", help="Hours in day used for B-matrix computation, e.g. 0_12 means that the forecast was initiated at 0UTC and 12UTC", type=str, required=False, default='0_12')
parser.add_argument("--forecast_len", help="Number of 1-hourly forecast steps", type=int, required=False, default=12)
parser.add_argument("--plot", help='Plot B matrix', default=False, action=argparse.BooleanOptionalAction)
parser.add_argument("--compute", help='Compute B matrix', default=False, action=argparse.BooleanOptionalAction)
parser.add_argument("--keep_3D", help='True: 3D-dimmensional latent space; False: flattened latent space', default=False, action=argparse.BooleanOptionalAction)

args = parser.parse_args()

if args.days_in_month == 'all':
    days_in_month = [i for i in range(1, 31+1)] # no problem if day in month doesn't exist
else:
    days_in_month = [int(s) for s in args.days_in_month.split('_')]
if args.hours_in_day == 'all':
    hours_in_day = [i for i in range(0, 23+1)]
else:
    hours_in_day = [int(s) for s in args.hours_in_day.split('_')]

import gc
from datetime import datetime, timedelta

tst = datetime.now()


# --------------------------------------------------------------------------
# NALAGANJE POMOZNIH STVARI, KI JIH RABIMO ZA POGANJANJE NN
# --------------------------------------------------------------------------
from loading_and_plotting_data import create_lists, encoded_forecast_provider

# Ustvarim vse tabele
tabele = create_lists()



# --------------------------------------------------------------------------
# NALAGANJE OBEH MODELOV
# --------------------------------------------------------------------------

AE_root_model = '%s/NNs/models/autoencoder_20_12100' % os.getenv('UGNN3DVar_master')
AE_latent_vec_shape = (1, 12100)



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
num_of_prediction_steps = args.forecast_len // 12  # This fwd model has 12h steps
# Število vhodnih časovnih instanc - to je fiksno na 1
input_sequence_length = 1
# Število povprečenih časovnih instanc - to je fiksno na 1
averaging_sequence_length = 1

# --------------------------------------------------------------------------
# NASTAVITVE ZA SHRANJEVANJE
# --------------------------------------------------------------------------

computation_start_date = datetime.strptime(args.start_date, '%Y-%m-%d')
computation_end_date = datetime.strptime(args.end_date, '%Y-%m-%d')

precondition_add = '_no_precondition'

B_matrix_savename = f'{FWD_root_model}/B-matrices/{AE_root_model[AE_root_model.rfind("/")+1:]}' +\
                    f'/climatological_prediction{precondition_add}_{args.start_date}_to_{args.end_date}' +\
                    f'_days_{args.days_in_month}_hrs_{args.hours_in_day}_steps_{args.forecast_len}'#_OUT'
B_matrix_savename_intermediate = f'{B_matrix_savename}_intermediate.pkl'

if not args.restart_computation:
    B_matrix, B_matrix_len, last_date = pickle.load(open(B_matrix_savename_intermediate, 'rb'))
    computation_start_date = last_date
    print('Uploaded B up to', last_date)
else:
    starting_vector = 0 * np.random.normal(size=AE_latent_vec_shape)#.astype(np.float32)
    B_matrix = np.outer(starting_vector, starting_vector)
    B_matrix_len = 0
    del starting_vector
    gc.collect()

intermediate_dates = [] # Checkpointi za B, ki jih shranjujemo (in sproti povozimo)
next_start_date = computation_start_date
while next_start_date < computation_end_date:
    next_start_date = next_start_date + timedelta(days=args.update_interval)
    if next_start_date < computation_end_date:
        intermediate_dates.append(next_start_date)


if args.compute:
    # --------------------------------------------------------------------------
    # IZRACUN
    # --------------------------------------------------------------------------

    day_count = (computation_end_date - computation_start_date).days + 1

    for single_date in (computation_start_date + timedelta(days=n) for n in range(day_count)):
        if single_date.day in days_in_month:    # e.g. days_in_month=[1,6,11,16,21,26]
            for FWD_start_time in hours_in_day: #(00UTC to 12UTC and 12UTC to 00UTC)
                print()
                print(single_date, FWD_start_time)
                FWD_start_date = (single_date.day, single_date.month, single_date.year)
                FWD_start_datetime = datetime(year=single_date.year, month=single_date.month, day=single_date.day, hour=FWD_start_time)
                FWD_end_datetime = FWD_start_datetime + timedelta(hours=args.forecast_len)

                AE_encoded_forecast = encoded_forecast_provider(
                    FWD_root_model=FWD_root_model,
                    AE_root_model=AE_root_model,
                    initial_conditions='ERA5',
                    keep_3D=args.keep_3D,
                    echo=False,
                    forecast_start_datetime=FWD_start_datetime,
                    forecast_end_datetime=FWD_end_datetime
                )
                AE_truth = encoded_forecast_provider(
                    FWD_root_model='persistence',
                    AE_root_model=AE_root_model,
                    initial_conditions='ERA5',
                    keep_3D=args.keep_3D,
                    echo=False,
                    forecast_start_datetime=FWD_end_datetime,
                    forecast_end_datetime=FWD_end_datetime
                )

                # Compute the difference and the outer product
                if args.latent_precondition:
                    background_error = ((np.float64(AE_encoded_forecast) - np.float64(latent_precondition_mean_std[0])) / np.float64(latent_precondition_mean_std[1])
                                        - (np.float64(AE_truth) - np.float64(latent_precondition_mean_std[0])) / np.float64(latent_precondition_mean_std[1]))
                else:
                    background_error = np.float64(AE_encoded_forecast) - np.float64(AE_truth)   # shape (1, 12100), this is actually e.T


                st = datetime.now()
                for irow in range(len(B_matrix)):
                    single_B_row = background_error[0, irow] * background_error[0]
                    B_matrix[irow] = B_matrix[irow] * (B_matrix_len / (B_matrix_len + 1)) + single_B_row * (1 / (B_matrix_len + 1))
                # The upper 3 rows are equivalent to this (but much faster):
                # B_matrix = B_matrix * (B_matrix_len / (B_matrix_len + 1)) + np.outer(background_error.T, background_error.T) * (1 / (B_matrix_len + 1))
                et = datetime.now()
                print('time updating B for loop', et - st)
                B_matrix_len += 1

        if single_date in intermediate_dates:
            last_date = single_date
            pickle.dump([B_matrix, B_matrix_len, last_date], open(B_matrix_savename_intermediate, 'wb'))

    pickle.dump(B_matrix, open(B_matrix_savename + '.pkl', 'wb'))
    tet = datetime.now()
    print('Computation ended on', tet)
    print('Running time:', tet - tst)

if args.plot:
    B_matrix = pickle.load(open(B_matrix_savename + '.pkl', 'rb'))
    log_diagonals = np.zeros(shape=(len(B_matrix)))
    log_off_diagonals = np.zeros(shape=((len(B_matrix) - 1) * len(B_matrix) // 2))
    off_diags_end_idx = 0

    greater_than_1 = 0
    for irow in range(len(B_matrix) - 1):
        # print(irow, B_matrix[irow])
        log_diagonals[irow] = np.log10(B_matrix[irow][irow])
        off_diags_start_idx = off_diags_end_idx
        off_diags_end_idx += len(B_matrix) - irow - 1
        log_off_diagonals[off_diags_start_idx:off_diags_end_idx] = np.log10(np.abs(B_matrix[irow][irow + 1:]))

    log_diagonals[-1] = np.log10(B_matrix[-1][-1])


    import matplotlib
    matplotlib.rcParams.update({"font.size": 16})

    nbin = 31
    bins = np.linspace(-6, 0, nbin)
    plt.hist(np.log10(np.diagonal(B_matrix)), bins=bins, density=True, alpha=0.8, label='Diagonal elements')
    plt.hist(np.log10(np.abs(B_matrix - B_matrix * np.identity(len(B_matrix))).flatten()), bins=bins, density=True,
             alpha=0.5, label='Off-diagonal elements')
    plt.xlabel(r'$\log_{10}$(abs($\mathbf{B}_z^{clim}$ element))')
    plt.xlim(min(bins), max(bins))
    plt.ylabel('Share')
    plt.legend(loc='upper left')
    plt.title(r'Distribution of $\mathbf{B}_z^{clim}$ elements')
    plt.yticks(ticks=nbin / (max(bins) - min(bins)) * np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5]),
               labels=[r'$0\,\%$', r'$10\,\%$', r'$20\,\%$', r'$30\,\%$', r'$40\,\%$', r'$50\,\%$'])  # to mapiranje postudiraj
    plt.tight_layout()

    plt.savefig(
        'figures/' + f'hist_{AE_root_model[AE_root_model.rfind("/") + 1:]}' + \
        f'_climatological_prediction_{args.start_date}_to_{args.end_date}' + \
        f'_days_{args.days_in_month}_hrs_{args.hours_in_day}_steps_{args.forecast_len}' + \
        f'{addname}_share.jpg', dpi=300)
