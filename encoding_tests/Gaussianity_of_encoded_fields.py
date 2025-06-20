# -*- coding: utf-8 -*-
#!/usr/bin/python
"""
Script for assessing the Gausianity of the encoded fields
"""

# Osnovno
import os
import numpy as np
import torch
import torch.nn as nn
import scipy


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
parser.add_argument("--days_in_month", help="Days in month used for B-matrix computation, e.g. 1_2_3 means 1st, 2nd, and 3rd day of each month will be used", type=str, required=False, default='all')
parser.add_argument("--hours_in_day", help="Hours in day used for B-matrix computation, e.g. 0_12 means that the forecast was initiated at 0UTC and 12UTC", type=str, required=False, default='0_12')
parser.add_argument("--forecast_steps", help="Number of 1-hourly forecast steps", type=int, required=False, default=12)
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



AE_root_model = '%s/NNs/models/autoencoder_20_12100' % os.getenv('UGNN3DVar_master')
name_preposition = 'AE_20_12100'




AE_latent_predictions = []

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
            FWD_end_datetime =FWD_start_datetime + timedelta(hours=num_of_prediction_steps)

            AE__datetime = FWD_end_datetime
            AE_date = (AE__datetime.day, AE__datetime.month, AE__datetime.year)
            AE_time = AE__datetime.hour

            # Load encoded fields (i.e. "persistence" forecast)
            AE_latent_prediction = encoded_forecast_provider(
                FWD_root_model='persistence',
                AE_root_model=AE_root_model,
                initial_conditions='ERA5',
                forecast_start_datetime=FWD_start_datetime,
                forecast_end_datetime=FWD_end_datetime
            )

            AE_latent_predictions.append(AE_latent_prediction[0])


all_skewness_latent = []
all_kurt_latent = []
AE_latent_predictions = np.array(AE_latent_predictions)
for i in range(np.shape(AE_latent_predictions)[1]): # AE_latent_predictions.shape = (N_predicitions, AE_latent_shape[1])
    all_skewness_latent.append(scipy.stats.skew(AE_latent_predictions[:,i]))
    all_kurt_latent.append(scipy.stats.kurtosis(AE_latent_predictions[:,i]))
print(np.mean(np.abs(all_skewness_latent)))
print(np.amax(np.abs(all_skewness_latent)))
print(np.mean(np.abs(all_kurt_latent)))
print(np.amax(np.abs(all_kurt_latent)))


