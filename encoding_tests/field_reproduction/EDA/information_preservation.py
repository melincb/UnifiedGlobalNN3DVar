
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

AE_root_model = '%s/NNs/models/autoencoder_20_12100' % os.getenv('UGNN3DVar_master')in_out_ch = 20
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




ensemble = 50
datetime_str = '2024-04-14-00'
EDA_ensemble_type = 'forecasts' # 'forecasts' for ensemble of backgrounds
included_members = 'all'

if included_members == 'all':
    included_members_lst = [i for i in range(1, ensemble + 1)]
else:
    included_members_lst = [int(s) for s in included_members.split('_')]

if datetime_str != '2024-04-14-00':
    warnings.warn(f"WARNING: You are using True_EDA, but your observation datetime is {datetime_str} and not {'2024-04-14-00'}!")
if ensemble != 50:
    warnings.warn(
        f"WARNING: You are using True_EDA, but your ensemble size is {ensemble} and not {50}!")

if EDA_ensemble_type == 'forecasts':
    ensemble_main_dir_ext = 'ens_B'
else:
    print('Unsupported EDA_ensemble_type:', EDA_ensemble_type)
    raise AttributeError  # Unsupported ensemble type


encoded_ensemble_members = [
    EDA_ensemble_member_load(
        which_ensemble=EDA_ensemble_type,
        em_idx=em_idx,
        datetime_str=datetime_str,
        return_encoded=True,
        echo=False
    )#.astype(np.float64)
    for em_idx in included_members_lst
]

# This time we do not perturb the ensemble members, we leave them as they are!
perturbed_AE_latent_background = np.array(encoded_ensemble_members)


ED_ensemble_members = torch.zeros(size=(ensemble, in_out_ch, 1, 180, 360), dtype=torch.float32) # shape: (ensemble members, variables, 1, lats, lons)
for izb in range(len(perturbed_AE_latent_background)):
    AE_preds = AE_predicting.decode(
                        ANNOTATIONS_NAMES,
                        AE_MODELS,
                        root_scaler,
                        SCALER_NAMES,
                        SCALER_TYPES,
                        num_of_prediction_steps=1,
                        input_sequence_len=1,
                        averaging_sequence_len=1,
                        z=torch.from_numpy(perturbed_AE_latent_background[izb].reshape(AE_latent_space_shape))    # perturbed_AE_latent_background is a list of lists of numpy arrays
                        )   # shape (in_out_ch, 2, 180, 360), [:,0,:,:] is useless
    for ivar in range(ED_ensemble_members.shape[1]):  # go through all in_out_ch variables
        ED_ensemble_members[izb, ivar] = AE_preds[ivar][1]


ED_ensemble_members_std = torch.std(ED_ensemble_members, dim=0).detach().numpy() * np.expand_dims(field_adjustment_multpilicators, (1,2,3))
ED_ensemble_members_std = ED_ensemble_members_std.squeeze()

raw_ensemble_members = torch.Tensor([
    EDA_ensemble_member_load(
        which_ensemble=EDA_ensemble_type,
        em_idx=em_idx,
        datetime_str=datetime_str,
        return_encoded=False,
        echo=False
    )#.astype(np.float64)
    for em_idx in included_members_lst
])


raw_ensemble_members_std = torch.std(raw_ensemble_members, dim=0).detach().numpy() * np.expand_dims(field_adjustment_multpilicators, (1,2))






# PLOTTING

# Which quantities do I plot?
plot_qty = ['all']  # quantities
# if 'all' in plot_qty:  # plot all in_out_ch fields
plot_qty_idx = [i for i in range(ED_ensemble_members_std.shape[0])]

# Plotting settings
num_fields = len(plot_qty_idx)  # Number of fields to plot
max_num_fields = ED_ensemble_members_std.shape[0]  # Highest possible number of fields to plot
single_fig_width = 10  # Width of single figure (in inches, I think)
qty_all = VARIABLES_SAVE
unit = [' [' + plotting_stuff()['ENOTE'][i] + ']' for i in
        range(ED_ensemble_members_std.shape[0])]  # How to display units

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

cmaps = plotting_stuff()['cmaps']  # Colormaps
vminmax_std_EDA = plotting_stuff()[
    'vminmax_std_EDA']  # Ranges for all available quantities when plotting std
# vminmax_std = (np.array(vminmax_std) * 0.1).tolist()



plot_bottom -= dheight_plot + dheight_buffer

ifieldplot = 0
for ifield in range(max_num_fields):
    if ifield in plot_qty_idx:
        ax_stdraw = fig.add_axes([plot_left + ifieldplot / num_fields, plot_bottom, plot_width, dheight_plot], projection=projections[projection])
        ifieldplot += 1
        plot_Earth(
            fig=fig,
            ax=ax_stdraw,
            field=raw_ensemble_members_std[ifield],
            vminmax=vminmax_std_EDA[ifield],
            cmap='terrain_r',
            cbar_extend='max',
            cbar_shrink=cbar_shrink,
            title='Std of $\mathbf{x}_{ens}^{\mathrm{IFS}}$ (%s%s)' % (qty_all[ifield], unit[ifield])
        )


plot_bottom -= dheight_plot + dheight_buffer

ifieldplot = 0
for ifield in range(max_num_fields):
    if ifield in plot_qty_idx:
        ax_stdED = fig.add_axes([plot_left + ifieldplot / num_fields, plot_bottom, plot_width, dheight_plot], projection=projections[projection])
        ifieldplot += 1
        plot_Earth(
            fig=fig,
            ax=ax_stdED,
            field=ED_ensemble_members_std[ifield],
            vminmax=vminmax_std_EDA[ifield],
            cmap='terrain_r',
            cbar_extend='max',
            cbar_shrink=cbar_shrink,
            title=r'Std of D(E($\mathbf{x}_{ens}^{\mathrm{IFS}}$)) (%s%s)' % (qty_all[ifield], unit[ifield])
        )






fig.savefig(f'information_preservation_{EDA_ensemble_type}_{datetime_str}_ensemble_{ensemble}_members_{included_members}.jpg', dpi=100)
