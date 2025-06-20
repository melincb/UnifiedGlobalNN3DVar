# -*- coding: utf-8 -*-
#!/usr/bin/python
"""
Script for computing the climatological B-matrix
"""

# --------------------------------------------------------------------------
# OSNOVNE KNJIZNICE
# enako kot v zagon_AE.ipynb, le s prilagojeno potjo do modulov
# --------------------------------------------------------------------------

# Osnovno
import os
import numpy as np
import torch
import torch.nn as nn


# Slike
import matplotlib.pyplot as plt

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
parser.add_argument("--ensemble_type", help="'analyses' for ensemble of analyses, 'forecasts' for ensemble of forecasts'", type=str, default='forecasts', required=False)
parser.add_argument("--control_or_mean", help="Compute background error vs control or vs ensemble mean; options: 'control', 'mean'", type=str, default='mean', required=False)
parser.add_argument("--datetime_str", help="Date and time when the ensemble is valid in format yyyy-mm-dd-hh", type=str, required=False, default='2024-04-14-00')
parser.add_argument("--ensemble_size", help="Number of ensemble members", type=int, default=50, required=False)
parser.add_argument("--included_members", help="Included ensemble members, e.g. 1_10 means that we use ensemble members with indices 1 and 10 (WARNING: idx 0 stands for control run!)", type=str, required=False, default='all')
parser.add_argument("--plot", help='Plot B matrix', default=False, action=argparse.BooleanOptionalAction)
parser.add_argument("--compute", help='Compute B matrix', default=False, action=argparse.BooleanOptionalAction)
parser.add_argument("--store_only_diagonal", help='Store only diagonal of B', default=True, action=argparse.BooleanOptionalAction)

args = parser.parse_args()

if args.included_members == 'all':
    included_members = [i for i in range(1, args.ensemble_size + 1)]
else:
    included_members = [int(s) for s in args.days_in_month.split('_')]

assert args.ensemble_type in ('forecasts')
assert args.control_or_mean in ('control', 'mean')

import gc
from datetime import datetime, timedelta

tst = datetime.now()


# --------------------------------------------------------------------------
# NALAGANJE POMOZNIH STVARI, KI JIH RABIMO ZA POGANJANJE NN
# --------------------------------------------------------------------------
from loading_and_plotting_data import create_lists, encoded_forecast_provider, EDA_ensemble_member_load

# Ustvarim vse tabele
tabele = create_lists()


# IMPORT AE

AE_root_model = '%s/NNs/models/autoencoder_20_12100' % os.getenv('UGNN3DVar_master')
AE_latent_vec_shape = (1, 12100)


# --------------------------------------------------------------------------
# NASTAVITVE ZA SHRANJEVANJE
# --------------------------------------------------------------------------

if args.ensemble_type == 'forecasts':
    ensemble_main_dir_ext = 'ens_B'
else:
    print('Unsupported args.ensemble_type:', args.ensemble_type)
    raise AttributeError  # Unsupported ensemble type



B_matrix_savename = f'{AE_root_model}/EDA/B-matrices/{ensemble_main_dir_ext}' +\
                    f'/B_{args.control_or_mean}_{args.datetime_str}_size_{args.ensemble_size}_members_{args.included_members}'







if args.compute:
    # --------------------------------------------------------------------------
    # IZRACUN
    # --------------------------------------------------------------------------

    starting_vector = 0 * np.random.normal(size=AE_latent_vec_shape).astype(np.float64)
    B_matrix = np.outer(starting_vector, starting_vector)
    B_matrix_len = 0
    del starting_vector
    gc.collect()

    encoded_ensemble_members = [
        EDA_ensemble_member_load(
            which_ensemble=args.ensemble_type,
            em_idx=em_idx,
            datetime_str=args.datetime_str,
            return_encoded=True,
            echo=False
        ).astype(np.float64)
        for em_idx in included_members
    ]

    if args.control_or_mean == 'control':
        # Use control run as reference for computing background errors
        reference_vector = EDA_ensemble_member_load(
            which_ensemble=args.ensemble_type,
            em_idx=0,
            datetime_str=args.datetime_str,
            return_encoded=True,
            echo=False
        ).astype(np.float64)
    elif args.control_or_mean == 'mean':
        # Use ensemble as reference for computing background errors
        reference_vector = np.mean(encoded_ensemble_members, axis=0)
    else:
        print(f'Oops! Forgot to include {args.control_or_mean} as an option when setting the reference vector for computing background errors.')
        raise AttributeError



    for em in encoded_ensemble_members:
        background_error = em.T - reference_vector.T
        st = datetime.now()
        B_matrix = B_matrix * (B_matrix_len / (B_matrix_len + 1)) + np.outer(background_error, background_error) * (
                    1 / (B_matrix_len + 1))
        et = datetime.now()
        print('time updating B em', et - st)
        B_matrix_len += 1

    if args.control_or_mean == 'mean':
        B_matrix = B_matrix * B_matrix_len / (B_matrix_len - 1)

    if args.store_only_diagonal:
        pickle.dump(np.diagonal(B_matrix), open(B_matrix_savename + '_diagonal.pkl', 'wb'))
    else:
        pickle.dump(B_matrix, open(B_matrix_savename + '.pkl', 'wb'))
    tet = datetime.now()
    print('Computation ended on', tet)
    print('Running time:', tet - tst)

if args.plot:   # DO TU
    if args.store_only_diagonal:
        raise AttributeError # Can't plot offdiagonals if only diagonals are storred
    B_matrix = pickle.load(open(B_matrix_savename + '.pkl', 'rb'))
    log_diagonals = np.zeros(shape=(len(B_matrix)))
    log_off_diagonals = np.zeros(shape=((len(B_matrix) - 1) * len(B_matrix) // 2))
    off_diags_end_idx = 0
    for irow in range(len(B_matrix) - 1):
        # print(irow, B_matrix[irow])
        log_diagonals[irow] = np.log10(B_matrix[irow][irow])
        off_diags_start_idx = off_diags_end_idx
        off_diags_end_idx += len(B_matrix) - irow - 1
        log_off_diagonals[off_diags_start_idx:off_diags_end_idx] = np.log10(np.abs(B_matrix[irow][irow + 1:]))

    log_diagonals[-1] = np.log10(B_matrix[-1][-1])


    plt.figure(figsize=(7, 6))
    import matplotlib
    matplotlib.rcParams.update({"font.size": 16})

    nbin = 31
    bins = np.linspace(-6, 0, nbin)
    plt.hist(np.log10(np.diagonal(B_matrix)), bins=bins, density=True, alpha=0.8, label='Diagonal elements')
    plt.hist(np.log10(np.abs(B_matrix - B_matrix * np.identity(len(B_matrix))).flatten()), bins=bins, density=True,
             alpha=0.5, label='Off-diagonal elements')
    plt.xlabel(r'$\log_{10}$(abs($\mathbf{B}_z^{\mathrm{EDA}}$ element))')
    plt.xlim(min(bins), max(bins))
    plt.ylabel('Share')
    plt.legend(loc='upper left')
    plt.title(r'Distribution of $\mathbf{B}_z^{\mathrm{EDA}}$ elements')
    plt.yticks(ticks=nbin / (max(bins) - min(bins)) * np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5]),
               labels=[r'$0\,\%$', r'$10\,\%$', r'$20\,\%$', r'$30\,\%$', r'$40\,\%$', r'$50\,\%$'])  # to mapiranje postudiraj
    plt.tight_layout()
    plt.savefig(f'figures/EDA_hist_{AE_root_model[AE_root_model.rfind("/")+1:]}' +\
                f'_{ensemble_main_dir_ext}_{args.control_or_mean}_{args.datetime_str}_size_{args.ensemble_size}_members_{args.included_members}_share.jpg', dpi=300)