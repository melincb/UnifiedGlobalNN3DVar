# -*- coding: utf-8 -*-
#!/usr/bin/python
"""
Script for executing the 3D-Var algorithm
"""

# --------------------------------------------------------------------------
# OSNOVNE KNJIZNICE
# enako kot v zagon_AE.ipynb, le s prilagojeno potjo do modulov
# --------------------------------------------------------------------------

# Osnovno
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Slike
from datetime import *


# Moduli
import sys
master_dir = os.getenv('UGNN3DVar_master')
sys.path.append(master_dir)
sys.path.append(master_dir + '/NNs')
import predict_1hr, predict_1hr_AE
import unet_model_7x7_4x_depth4_interdepth_padding_1_degree_v01_stride2b_no_skip_AE

# --------------------------------------------------------------------------
# DODATNE KNJIZNICE
# --------------------------------------------------------------------------
import argparse
import pickle

import gc
from datetime import datetime, timedelta




# --------------------------------------------------------------------------
# NALAGANJE POMOZNIH STVARI, KI JIH RABIMO ZA POGANJANJE NN
# --------------------------------------------------------------------------
from loading_and_plotting_data import create_lists, ERA5_loader


# ------------------------------------------------------
# LOAD ALGORITHM INPUTS
# ------------------------------------------------------

inputs = pickle.load(open('experiments/data/algorithm_inputs.pkl', 'rb'))
inputs_for_ensemble_3D_Var = inputs['inputs_for_ensemble_3D_Var'] # NO TF-RELATED STUFF ALLOWED!
# parallel_processes = inputs['parallel_processes']
file_to_dump = inputs['name']
FWD_root_model = inputs['FWD_root_model']
AE_root_model = inputs['AE_root_model']
preconditioned_3D_Var = inputs['preconditioned_3D_Var']


# --------------------------------------------------------------------------
# LOADING AUTOENCODER
# --------------------------------------------------------------------------
# WE DON'T REALLY NEED TO LOAD FWD MODEL AS WE ALREADY LOAD THE ENCODED BACKGROUND
# The specifications for the fwd model are, however, given by algorithm inputs ['FWD_root_model']



if AE_root_model == '%s/NNs/models/autoencoder_20_12100' % os.getenv('UGNN3DVar_master'):
    in_out_ch = 20
    AE_model = unet_model_7x7_4x_depth4_interdepth_padding_1_degree_v01_stride2b_no_skip_AE.UNet(
       in_channels=1 * in_out_ch + 3, out_channels=in_out_ch, depth=4, start_filts=50,
       up_mode='transpose', merge_mode='concat')
    name_preposition = 'AE_20_12100'
    # Ustvarim vse tabele
    tabele = create_lists()
else:
    raise AttributeError # invalid AE_root model (maybe just not correctly included in algorithm_serial.py)
AE_latent_vec_shape = (1, 12100)
AE_latent_space_shape = (1, 50, 11, 22)
# Poskrbi za ustrezno delovanje tako na CPU, kot GPU
device_ids = [0, 1]
AE_model = nn.DataParallel(AE_model, device_ids=device_ids)

# Tabela s potjo do AE modela
AE_ROOTS_MODEL = [AE_root_model]
# Tabela s AE modelom
AE_MODELS = [AE_model]
model_name = "trained_model_weights.pth"
AE_predicting = predict_1hr_AE.Predicting(AE_ROOTS_MODEL, model_name)


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


# ------------------------------------------------------
# SET THE CONTENT THAT IS THE SAME FOR ALL THE ENSEMBLE MEMBERS
# ------------------------------------------------------
# Also convert most of the content from numpy to torch.
# These two steps are not possible in algorithm-parallel.py
# due to multiprocessing library's requirements!

# We can already load B-matrix inverse here as the ensemble members need the same B-matrix inverse
B_matrix_filename = inputs_for_ensemble_3D_Var[0]['B_matrix_filename']
# print('B_matrix_filename', B_matrix_filename)


if not preconditioned_3D_Var:
    B_matrix = pickle.load(open(B_matrix_filename + '.pkl', 'rb'))  # type numpy array
    B_matrix_inv = torch.from_numpy(np.eye(len(B_matrix)).astype(B_matrix.dtype) * (1 / np.diagonal(B_matrix)))


else:
    print('SETTING SQRT AND SQRT INV FOR DIAGONAL VERSION OF B')
    # TO RUN THE EXPERIMENT WITH EDA BACKGROUNDS, BUT CLIM_B, UNCOMMENT THE LINES BETWEEN # *-----* and # *|||||* !
    # *-----*
    # print('\nTHIS IS A SPECIAL EXPERIMENT WHERE WE USE THE ENSEMBLE SUITABLE FOR THE B-MATRIX FROM ABOVE, BUT WE PERFORM 3D-Var WITH THE FOLLOWING B:')
    # B_matrix_filename = (master_dir + '/NNs/' +
    #                      'models/U-Net_IT_ALL_numpred2_train_sumMSE_testACC_isl1_250ch_ks7x7_1hr_parallel_2024_11/B-matrices/autoencoder_20_12100/' +
    #                     'climatological_prediction_no_precondition_2015-01-01_to_2019-12-31_days_all_hrs_all_steps_24_diagonal')
    # print('\nUsed B_matrix_filename', B_matrix_filename)
    # from time import sleep
    # sleep(5)
    # *|||||*

    B_matrix = pickle.load(open(B_matrix_filename + '.pkl', 'rb')) # type numpy array
    if B_matrix.dtype != np.float32:
        B_matrix = B_matrix.astype(np.float32)
    B_matrix_sqrt = torch.from_numpy(np.eye(len(B_matrix)).astype(B_matrix.dtype) * np.sqrt(B_matrix))
    B_matrix_sqrt_inv = torch.from_numpy(np.eye(len(B_matrix)).astype(B_matrix.dtype) * (1/np.sqrt(B_matrix)))



# We are also always observing the same quantities...
obs_qty_idx = inputs_for_ensemble_3D_Var[0]['obs_qty_idx']

# Always the same R-matrix ...
R_matrix_inv = torch.from_numpy(inputs_for_ensemble_3D_Var[0]['R_matrix_inv'])


# And the initial learning rate ...
init_lr = inputs_for_ensemble_3D_Var[0]['init_lr']

# We can also already set the torch version of observation locations here as
# all the ensemble members need the same observation locations
lats, lons = inputs_for_ensemble_3D_Var[0]['obs_locs']

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
# but I started with what ChatGPT gave me and then gradually modified it until it gave correct results. The 5 strange lines are given below:
obs_locs_torch1 = torch.from_numpy(np.zeros(shape=obs_locs_torch.shape).astype(np.float32))
obs_locs_torch1[:, 0], obs_locs_torch1[:, 1] = obs_locs_torch[:, 1], obs_locs_torch[:, 0]
obs_locs_torch = obs_locs_torch1
del obs_locs_torch1
gc.collect()

# Reshaping locs_to_interpolate so it is really suitable for interpolation
obs_locs_torch = obs_locs_torch.unsqueeze(0).expand(in_out_ch, -1, -1)  # Shape: (in_out_ch, Nobs, 2)
obs_locs_torch = obs_locs_torch.unsqueeze(2)  # shape (in_out_ch, Nobs, 1, 2), suitable for interpolation



# ------------------------------------------------------
# DEFINE A FUNCTION THAT FINDS A LATENT STATE WHICH MINIMISES THE 3D-VAR COST FUNCTION
# ------------------------------------------------------


def findLatent3DVar_preconditioned(
        AE_latent_background_vec_transposed,  # np array, vector 1 x autoencoder_latent_dim (already perturbed)
        obs_vec_transposed,  # np.array, vector 1 x number of obs. (already perturbed)
        ensemble_member_idx, # Just to print
        B_matrix_sqrt,
        B_matrix_sqrt_inv,
        obs_locs_torch=obs_locs_torch,
        obs_qty_idx=obs_qty_idx,
        R_matrix_inv=R_matrix_inv,
        init_lr=init_lr,  # Define your initial learning rate
        max_num_steps=50,
        factor_lr=0.5,  # Factor by which the learning rate is reduced
        rtol_stop=0.01,   # Relative tolerance for convergence criterion
        minimum_lr = 1e-4
):
    # ------------------------------------------------------
    # CONVERT THE REMAINING STUFF TO TORCH
    # ------------------------------------------------------
    # Most variables have already been converted to torch earlier -
    # this won't be possible in algorithm-parallel.py, where everything will be
    # converted to torch in this section.

    # Transpose the input 'vectors' so they indeed become Nx1 vectors
    AE_latent_background_vec = torch.transpose(torch.from_numpy(AE_latent_background_vec_transposed), 0, 1)
    obs_vec = torch.transpose(torch.from_numpy(obs_vec_transposed), 0, 1)

    if B_matrix_sqrt.dtype == torch.float64:
        # Since in this B_matrix_inv has dtype float64, we also need to convert everything else to float64
        AE_latent_background_vec = AE_latent_background_vec.to(torch.float64)
        obs_vec = obs_vec.to(torch.float64)
        R_matrix_inv = R_matrix_inv.to(torch.float64)
        obs_locs_torch = obs_locs_torch.to(torch.float64)


    # Define a vector which will be corrected throughout the process
    corrected_zeta_vec = B_matrix_sqrt_inv @ (AE_latent_background_vec - AE_latent_background_vec).clone()  # The first guess for zeta is zero vector
    corrected_zeta_vec.requires_grad = True  # So we can compute the gradient and apply it


    # ------------------------------------------------------
    # PREPARE MINIMISATION TRACKERS AND SETTINGS
    # ------------------------------------------------------

    # Setting the optimizer for stochastic gradient descend during 3D-Var
    optimizer = torch.optim.SGD([corrected_zeta_vec], lr=init_lr)
    # Variable to keep track of the best loss
    best_J = float('inf')
    # Minimisation step with the best loss
    best_J_step = 0
    # Latent state at the best step
    best_AE_latent_vec = (B_matrix_sqrt @ corrected_zeta_vec.clone().detach() + AE_latent_background_vec)   # Basically, this is background


    all_J = [] # Store the values of cost function at each minimisation step
    all_Jo = [] # Store the values of observation term of cost function at each minimisation step
    all_grad_J = [] # Store the values of the cost function gradient's Euclidean norm at each minimisation step

    # Ending step of the minimisation process (retains this value if the convergence is not reached before that)
    ending_step = max_num_steps

    # ------------------------------------------------------
    # 3D-VAR COST FUNCTION (Preconditioned version)
    # ------------------------------------------------------
    def latent3DVar():
        '''Compute 3D-Var cost function as a function of zeta (chi) instead of z (x)'''

        # ------------------------------------------------------
        # BACKGROUND TERM
        # ------------------------------------------------------

        # Compute the background term, J_b = 1/2 * zeta**T @ zeta
        J_b = 1 / 2 * (corrected_zeta_vec.T @ corrected_zeta_vec)

        # ------------------------------------------------------
        # OBSERVATION TERM
        # ------------------------------------------------------

        # Decode the latent vector, the output is D(z)
        # First we use the function which returns the output with shape (in_out_ch, 2, 180, 360) -
        # here [:,0,:,:] is zeros that we don't need, whereas [:,1,:,:] is the actual decoded field
        decoded_redundant = AE_predicting.decode(
            ANNOTATIONS_NAMES,
            AE_MODELS,
            root_scaler,
            SCALER_NAMES,
            SCALER_TYPES,
            num_of_prediction_steps=1,
            z=torch.reshape((AE_latent_background_vec + B_matrix_sqrt @ corrected_zeta_vec).to(torch.float32), AE_latent_space_shape))  # (in_out_ch, 2, 180, 360)
            # z=torch.reshape(AE_latent_background_vec + B_matrix_sqrt @ corrected_zeta_vec, AE_latent_space_shape))  # (in_out_ch, 2, 180, 360)
            # to(torch.float32) has no effect if B already has this dtype


        # Now only get the real decoded stuff
        decoded = torch.zeros(size=(in_out_ch, 1, 180, 360), dtype=torch.float32)
        for ivar in range(decoded.shape[0]):
            decoded[ivar] = decoded_redundant[ivar][1]

        if B_matrix_sqrt.dtype == torch.float64:
            decoded = decoded.to(torch.float64)

        # Now use D(z) to get H(D(z)):
        # (1) Interpolate decoded fields to observation locations
        decoded_obs_locs = F.grid_sample(decoded, obs_locs_torch, mode='bilinear',
                                         align_corners=True)  # Shape: (in_out_ch, 1, Nobs, 1)

        # (2) keep only the quantities that we really observe (using [obs_qty_idx, :, :, :]),
        # (3) remove the excess dimensions (using squeeze()), and
        # (4) reshape it to a vector with dimmension (1, Nqties*Nobs) (using view(-1, 1)).
        H_of_D_of_z = decoded_obs_locs[obs_qty_idx, :, :, :].squeeze().view(-1, 1)

        # Get the difference between the observations values and the extracted values, y - H(D(z)) = y - H(D(z_b + B_sqrt @ zeta))
        y_minus_H_of_D_of_z = torch.subtract(obs_vec, H_of_D_of_z)

        # Compute the observations term, J_o = 1/2 * (y - H(D(z)))**T R**-1 (y - H(D(z)))
        J_o = 1 / 2 * torch.matmul(torch.matmul(torch.transpose(y_minus_H_of_D_of_z, 0, 1), R_matrix_inv),
                                   y_minus_H_of_D_of_z)

        # ------------------------------------------------------
        # SUM BACKGROUND AND OBSERVATION TERMS
        # ------------------------------------------------------

        J = torch.add(J_b, J_o)

        return torch.squeeze(J), torch.squeeze(J_o)  # Return them as scalars


    # ------------------------------------------------------
    # MINIMISATION ALGORITHM
    # ------------------------------------------------------
    for step in range(1, max_num_steps + 1):
        J, Jo = latent3DVar()  # Get current values of the cost function

        all_J.append(J.item())  # Store current value of the cost function
        all_Jo.append(Jo.item())  # Store current value of the observation term
        previous_zeta_vec = corrected_zeta_vec.clone().detach()  # Store current latent vector
        J.backward()  # Compute current gradient
        grad_J = corrected_zeta_vec.grad  # Get the gradient of the latent vector
        norm_grad_J = torch.norm(grad_J, p=2)
        all_grad_J.append(norm_grad_J)  # Store the Euclidean norm of current gradient

        optimizer.step()  # Change the latent vector according to its current gradient
        optimizer.zero_grad()  # Clear the gradients for the next iteration

        # Check for improvement in loss (for learning rate)
        if J < best_J:
            best_J = J
            best_grad_J = norm_grad_J
            best_J_step = step
            best_AE_latent_vec = (B_matrix_sqrt @ previous_zeta_vec.clone() + AE_latent_background_vec)
        else:
            current_lr = optimizer.param_groups[0]['lr']
            new_lr = max(current_lr * factor_lr, minimum_lr)
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lr

        # Check for improvement in gradient (for stopping criterion)
        if step >= 2:
            if all_grad_J[-1] / all_grad_J[0] < rtol_stop:
                ending_step = step
                break


        # Monitor the minimisation procedure
        # We print ensemble member index, minimisation step, cost function value,
        # the ratio between the cost function in this and the previous step, number of steps after the last update of the best latent vector
        if step == 1:
            print('\nEns. member', ensemble_member_idx, 'initial J', J, 'initial grad J', all_grad_J[0])
        if step == 2 or step % 10 == 0:
            print('Ens. member', ensemble_member_idx, 'step', step, 'J', J, 'ratio', all_J[-1] / all_J[-2], 'grad J',
                  all_grad_J[-1], 'ratio', all_grad_J[-1] / all_grad_J[0])

    print('Ens. member', ensemble_member_idx, 'ending step', ending_step, 'ending J', J, 'best step', best_J_step,
          'best J', best_J, 'best grad J ratio', best_grad_J / all_grad_J[0])

    # This kind of output may be a bit clumsy, however, it has to be done this way in case of parallelization,
    # so we decided to do it the same way here for the sake of universality.
    return {'best_AE_latent': torch.transpose(best_AE_latent_vec, 0, 1), 'best_J': best_J, 'all_J': all_J,
            'all_Jo': all_Jo, 'all_grad_J': all_grad_J}



# ------------------------------------------------------
# RUN THE FUNCTION THAT FINDS A LATENT STATE WHICH MINIMISES THE 3D-VAR COST FUNCTION
# ------------------------------------------------------
# Run it in serial - one-by-one ensemble member

results = []
# Manually set the minimization criteria
max_num_steps = 50
factor_lr = 0.5  # Factor by which the learning rate is reduced
patience_lr = 3  # Number of epochs with no improvement after which learning rate is reduced
rtol_stop = 0.01   # Relative tolerance for convergence criterion
patience_stop = 5
minimum_lr = 1e-4

if preconditioned_3D_Var:
    for ens_member in inputs_for_ensemble_3D_Var:
        results.append(findLatent3DVar_preconditioned(
            AE_latent_background_vec_transposed=ens_member['perturbed_AE_latent_background'],
            obs_vec_transposed=ens_member['perturbed_obs'],
            ensemble_member_idx=ens_member['ensemble_member_idx'],
            B_matrix_sqrt=B_matrix_sqrt,
            B_matrix_sqrt_inv=B_matrix_sqrt_inv,
            max_num_steps=max_num_steps,
            factor_lr=factor_lr,
            rtol_stop=rtol_stop,
            minimum_lr=minimum_lr
        ))

    del B_matrix_sqrt, B_matrix_sqrt_inv

else:
    raise AttributeError    # Not implemented in the final version




# ------------------------------------------------------
# PREPARE ALGORITHM OUTPUTS TO DUMP THEM
# ------------------------------------------------------


del R_matrix_inv
gc.collect()

# This process may be a bit clumsy, however, it has to be done this way in algorithm-parallel.py,
# so we decided to do it the same way here for the sake of universality.

AE_latent_out = torch.stack([results[iens_mem]['best_AE_latent'] for iens_mem in range(len(results))]) # shape (ens, 1, 49500)
best_J = [results[iens_mem]['best_J'] for iens_mem in range(len(results))]
all_J = [results[iens_mem]['all_J'] for iens_mem in range(len(results))]
all_Jo = [results[iens_mem]['all_Jo'] for iens_mem in range(len(results))]
all_grad_J = [results[iens_mem]['all_grad_J'] for iens_mem in range(len(results))]


# At first I had a list of np arrays here and directly converted them using torch.tensor,
# however I got a user warning that it is much faster if I convert list of arrays to array of arrays and then use torch.from_numpy()
AE_latent_background = torch.from_numpy(np.array([inputs_for_ensemble_3D_Var[iens_mem]['perturbed_AE_latent_background'] for iens_mem in range(len(inputs_for_ensemble_3D_Var))]))
obs_vec_transposed = torch.from_numpy(np.array([inputs_for_ensemble_3D_Var[iens_mem]['perturbed_obs'] for iens_mem in range(len(inputs_for_ensemble_3D_Var))]))


# Write a short comment where you describe the dumped stuff!
comment = "AE_latent_out: latent element values after fit\n" + \
          "AE_latent_background: latent background samples, used also as initial guess for fit\n" + \
          "obs_locs_torch: observations' locations (pytorch tensor suitable for bilinear interpolation)\n" + \
          "obs_lons_lats: observations' locations ([lons, lats])\n" + \
          "obs_vec_transposed: transposed observation vectors in fit\n" + \
          "obs_qty_idx: indices of the observed quantities (see prepare_or_plot.py for details)\n" + \
          "best_J: best J value for each ensemble member\n" + \
          "all_J: all J values for each ensemble member\n" + \
          "all_Jo: all Jo values for each ensemble member (observation term only - how to get the background term: Jb = J - Jo)\n" + \
          "all_grad_J: all gradients of J values for each ensemble member (Euclidian norm, i.e. p=2)\n" + \
          "FWD_root_model: root directory of the forward model\n" + \
          "AE_root_model: root directory of the autoencoder\n"

dict_to_dump = {'AE_latent_out': AE_latent_out,
                'AE_latent_background': AE_latent_background,
                'obs_locs_torch': obs_locs_torch, 'obs_lons_lats': [lons, lats],
                'obs_vec_transposed': obs_vec_transposed, 'obs_qty_idx': obs_qty_idx,
                'best_J': best_J, 'all_J': all_J, 'all_Jo':all_Jo, 'all_grad_J':all_grad_J,
                'FWD_root_model': FWD_root_model, 'AE_root_model': AE_root_model,
                'comment': comment}

print('Dumping outputs to', file_to_dump)

# Save the outputs
pickle.dump(dict_to_dump, open(file_to_dump + '.pkl', 'wb'))