# -*- coding: utf-8 -*-
#!/usr/bin/python
"""
Script for computing the forecast biases
"""


# Osnovno
import os
import numpy as np
import torch
import torch.nn as nn


# Slike
import matplotlib.pyplot as plt
from datetime import *


# Moduli
import sys
master_dir = os.getenv('UGNN3DVar_master')
sys.path.append(master_dir)
sys.path.append(master_dir + '/NNs')

from loading_and_plotting_data import encoded_forecast_provider

# --------------------------------------------------------------------------
# DODATNE KNJIZNICE + nastavitve
# --------------------------------------------------------------------------
import argparse
import pickle

parser = argparse.ArgumentParser()
parser.add_argument("--start_date", help="Start date in format yyyy-mm-dd", type=str, required=False, default='2015-01-01')
parser.add_argument("--end_date", help="End date in format yyyy-mm-dd", type=str, required=False, default='2019-12-31')
parser.add_argument("--days_in_month", help="Days in month used for B-matrix computation, e.g. 1_2_3 means 1st, 2nd, and 3rd day of each month will be used", type=str, required=False, default='1_6_11_16_21_26')
parser.add_argument("--hours_in_day", help="Hours in day used for B-matrix computation, e.g. 0_12 means that the forecast was initiated at 0UTC and 12UTC", type=str, required=False, default='0_12')
parser.add_argument("--forecast_len", help="Forecast length (in hours; this used to be forecast_steps, when 1h fwd model was used)", type=int, required=False, default=12)
parser.add_argument('--FWD_model', help="Which forward model do I use? Options: 'NNfwd'", type=str, required=True)
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
print('Initiated computation', tst)


# --------------------------------------------------------------------------
# NALAGANJE OBEH MODELOV
# --------------------------------------------------------------------------

if args.FWD_model == 'NNfwd':
    in_out_ch = 20
    FWD_root_model = '%s/NNs/models/U-Net_IT_ALL_numpred2_train_sumMSE_testACC_isl1_250ch_ks7x7_1hr_parallel_2024_11' % os.getenv(
        'UGNN3DVar_master')

else:
    print(f'Oops! I forgot to allow {args.FWD_model} in the part of my code where I set FWD_root_model!')
    raise AttributeError


AE_root_model = '%s/NNs/models/autoencoder_20_12100' % os.getenv('UGNN3DVar_master')
name_preposition = 'AE_20_12100'



AE_latent_errors = []

# --------------------------------------------------------------------------
# IZRACUN
# --------------------------------------------------------------------------
computation_start_date = datetime.strptime(args.start_date, '%Y-%m-%d')
computation_end_date = datetime.strptime(args.end_date, '%Y-%m-%d')
day_count = (computation_end_date - computation_start_date).days + 1
for single_date in (computation_start_date + timedelta(days=n) for n in range(day_count)):
    if single_date.day in days_in_month:  # e.g. days_in_month=[1,6,11,16,21,26]
        for FWD_start_time in hours_in_day:  # (00UTC to 12UTC and 12UTC to 00UTC)
            print()
            print(single_date, FWD_start_time)
            FWD_start_datetime = datetime(year=single_date.year, month=single_date.month, day=single_date.day, hour=FWD_start_time)
            FWD_end_datetime =FWD_start_datetime + timedelta(hours=args.forecast_len)

            AE__datetime = FWD_end_datetime
            AE_date = (AE__datetime.day, AE__datetime.month, AE__datetime.year)
            AE_time = AE__datetime.hour

            # Load encoded background
            if args.FWD_model in ('NNfwd'):
                AE_latent_prediction = encoded_forecast_provider(
                    FWD_root_model=FWD_root_model,
                    AE_root_model=AE_root_model,
                    initial_conditions='ERA5',
                    forecast_start_datetime=FWD_start_datetime,
                    forecast_end_datetime=FWD_end_datetime
                )
                AE_latent_truth = encoded_forecast_provider(
                    FWD_root_model='persistence',
                    AE_root_model=AE_root_model,
                    initial_conditions='ERA5',
                    forecast_start_datetime=FWD_end_datetime,
                    forecast_end_datetime=FWD_end_datetime
                )
            else:
                print(
                    f'Oops! I forgot to allow {args.FWD_model} in the part of my code where I load encoded background!')
                raise AttributeError

            AE_latent_errors.append(AE_latent_prediction[0] - AE_latent_truth[0])


plt.figure(figsize=(7, 6))
import matplotlib
matplotlib.rcParams.update({"font.size": 16})

elementwise_std = np.std(AE_latent_errors, axis=0)
elementwise_mean = np.mean(AE_latent_errors, axis=0)

plt.scatter(np.abs(elementwise_mean), elementwise_std, alpha=0.1)
# plt.plot([np.amin(elementwise_std), np.amax(elementwise_std)], [np.amin(elementwise_std), np.amax(elementwise_std)], 'k--')
plt.plot([1e-1, 4e-1], [1e-1, 4e-1], 'k--')
plt.yscale('log')
plt.xscale('log')
plt.xlabel('Absolute value of mean forecast error')
plt.ylabel('Standard deviation of forecast error')
plt.ylim(1.4e-1, 4e-1)
plt.title('Assessing forecast bias')
plt.tight_layout()
plt.savefig(f'{name_preposition}_forecast_error_{AE_root_model[AE_root_model.rfind("/") + 1:]}' + \
        f'_{args.FWD_model}' + \
        f'_{args.start_date}_to_{args.end_date}' + \
        f'_days_{args.days_in_month}_hrs_{args.hours_in_day}_steps_{args.forecast_len}' + \
        '_mean_std.jpg', dpi=300)


