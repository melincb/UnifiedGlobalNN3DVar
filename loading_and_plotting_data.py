# -*- coding: utf-8 -*-
#!/usr/bin/python
"""
Script with data required for data standardisation and for plotting
"""
import os
import time

# --------------------------------------------------------------------------
# TABELE
# Tabele vsebujejo informacije o spremenljivkah, vedno v istem vrstnem redu!
# --------------------------------------------------------------------------

# Te funkcije ni treba spreminjati
def create_lists():
    root_annotations = ''   # SET PATH TO YOUR ROOT ANNOTATIONS!

    # Imena datotek s potmi do podatkov
    ANNOTATIONS_NAMES = [
        "geopotential_250_annotation_file.txt",
        "geopotential_500_annotation_file.txt",
        "geopotential_700_annotation_file.txt",
        "geopotential_850_annotation_file.txt",
        "mslp_annotation_file.txt",
        "surface_temperature_annotation_file.txt",
        "t2m_annotation_file.txt",
        "t500_annotation_file.txt",
        "t850_annotation_file.txt",
        "twv_annotation_file.txt",
        "u_10m_annotation_file.txt",
        "u_200_annotation_file.txt",
        "u_500_annotation_file.txt",
        "u_700_annotation_file.txt",
        "u_900_annotation_file.txt",
        "v_10m_annotation_file.txt",
        "v_200_annotation_file.txt",
        "v_500_annotation_file.txt",
        "v_700_annotation_file.txt",
        "v_900_annotation_file.txt",
    ]

    # Pot do direktorije s shranjenimi scalerji
    root_scaler = '%s/NNs/Scalers_1hr' % os.getenv('UGNN3DVar_master')

    # Imena scalerjev
    SCALER_NAMES = [
        "StandardScaler_1degree_Geopotential_250_1hr_1970-2014.pt",
        "StandardScaler_1degree_Geopotential_500_1hr_1970-2014.pt",
        "StandardScaler_1degree_Geopotential_700_1hr_1970-2014.pt",
        "StandardScaler_1degree_Geopotential_850_1hr_1970-2014.pt",
        "StandardScaler_1degree_Mslp_1hr_1970-2014.pt",
        "StandardScaler_1degree_Surface_Temperature_1hr_1970-2014.pt",
        "StandardScaler_1degree_Temperature_2m_1hr_1970-2014.pt",
        "StandardScaler_1degree_Temperature_500_1hr_1970-2014.pt",
        "StandardScaler_1degree_Temperature_850_1hr_1970-2014.pt",
        "StandardScaler_1degree_Total_Column_Water_Vapour_1hr_1970-2014.pt",
        "StandardScaler_1degree_Wind_U_10m_1hr_1970-2014.pt",
        "StandardScaler_1degree_Wind_U_200_1hr_1970-2014.pt",
        "StandardScaler_1degree_Wind_U_500_1hr_1970-2014.pt",
        "StandardScaler_1degree_Wind_U_700_1hr_1970-2014.pt",
        "StandardScaler_1degree_Wind_U_900_1hr_1970-2014.pt",
        "StandardScaler_1degree_Wind_V_10m_1hr_1970-2014.pt",
        "StandardScaler_1degree_Wind_V_200_1hr_1970-2014.pt",
        "StandardScaler_1degree_Wind_V_500_1hr_1970-2014.pt",
        "StandardScaler_1degree_Wind_V_700_1hr_1970-2014.pt",
        "StandardScaler_1degree_Wind_V_900_1hr_1970-2014.pt",
    ]

    # Tip scalerjev (prostorsko odvisna časovna standardna deviacija)
    scaler_type = "StandardScaler"
    SCALER_TYPES = [scaler_type for _ in range(len(SCALER_NAMES))]

    # Poti do statičnih polj (ta so že standardizirana, zato ne potrebujejo svojega scalerja)
    STATIC_FIELDS_ROOTS = [
        "%s/NNs/Static_Fields/latitudes.npy" % os.getenv('UGNN3DVar_master'),
        "%s/NNs/Static_Fields/land_sea_mask.npy" % os.getenv('UGNN3DVar_master'),
        "%s/NNs/Static_Fields/surface_topography.npy" % os.getenv('UGNN3DVar_master'),
    ]

    # Imena spremenljivk, uporabljena pri shranjevanju slik
    VARIABLES_SAVE = [
        "Z250",
        "Z500",
        "Z700",
        "Z850",
        "mslp",
        "ST",
        "T2m",
        "T500",
        "T850",
        "TCWV",
        "U10m",
        "U200",
        "U500",
        "U700",
        "U900",
        "V10m",
        "V200",
        "V500",
        "V700",
        "V900",
    ]


    # Slovar vseh tabel in poti
    lists = {"root_annotations" : root_annotations,
             "ANNOTATIONS_NAMES" : ANNOTATIONS_NAMES,
             "root_scaler" : root_scaler,
             "SCALER_NAMES" : SCALER_NAMES,
             "SCALER_TYPES" : SCALER_TYPES,
             "STATIC_FIELDS_ROOTS" : STATIC_FIELDS_ROOTS,
             "VARIABLES_SAVE" : VARIABLES_SAVE,
             }

    return lists



def ERA5_loader(datetime_str, return_option='torch tensor'):
    '''datetime_str should be in format "yyyy-mm-dd-hh"
    return_option:
        1) 'torch tensor' -> torch tensor with shape (24, 1, 180, 360)
        2) 'list of tensors' -> list of lists of 2D torch tensors
        3) 'list of np arrays' -> list of lists of 2D numpy arrays
    '''
    in_out_ch = 20

    # Ustvarim vse tabele
    tabele = create_lists()

    # Poti do datotek
    root_annotations = tabele["root_annotations"]
    ANNOTATIONS_NAMES = tabele["ANNOTATIONS_NAMES"]


    import data_preparation_1hr
    import numpy as np
    import torch


    date_to_day = data_preparation_1hr.DateToConsecutiveDay()

    # --> Get index for annotation_file
    year, month, day, hour = [int(s) for s in datetime_str.split('-')]
    index = date_to_day.num_of_samples(
        year, month, day, delta_t=1/24) + hour


    # --> Load fields annotations (path to dataset fields)
    ANNOTATION_FILES = []
    for i in range(len(ANNOTATIONS_NAMES)):
        annotation_file = np.loadtxt(
            os.path.join(root_annotations, ANNOTATIONS_NAMES[i]), dtype=np.str_)
        ANNOTATION_FILES.append(annotation_file)

    # Truth fields
    FIELDS = []
    for variable_index in range(len(ANNOTATION_FILES)):
        path = ANNOTATION_FILES[variable_index][index]
        if return_option in ('torch tensor', 'list of torch tensors'):
            field = torch.from_numpy(np.load(path))
        else:
            field = np.load(path)
        FIELDS.append([field])#s)

    if return_option == 'torch tensor':
        total_tensor = torch.zeros(size=(in_out_ch, 1, 180, 360), dtype=torch.float32)
        for ivar in range(total_tensor.shape[0]):
            total_tensor[ivar] = FIELDS[ivar][0]
        return total_tensor
    else:
        return FIELDS

def plotting_stuff():

    # Units
    ENOTE = [
        'm',  # Z250
        'm',  # Z500
        'm',  # Z700
        'm',  # Z850
        'hPa',  # mslp
        'K',  # ST
        'K',  # T2m
        'K',  # T500
        'K',  # T850
        r'$\mathrm{kg}/\mathrm{m}^2$',  # TCWV
        "m/s",  # U10m
        "m/s",  # U200
        "m/s",  # U500
        "m/s",  # U700
        "m/s",  # U900
        "m/s",  # V10m
        "m/s",  # V200
        "m/s",  # V500
        "m/s",  # V700
        "m/s",  # V900
    ]

    # S čim pomnožimo vrednosti polj
    FIELD_ADJUSTEMENT_MULTIPLICATION = [
        1. / 9.81,  # Z250
        1. / 9.81,  # Z500
        1. / 9.81,  # Z700
        1. / 9.81,  # Z850
        1. / 100,  # mslp
        1.,  # ST
        1.,  # T2m
        1.,  # T500
        1.,  # T850
        1.,  # TCWV
        1.,  # U10m
        1.,  # U200
        1.,  # U500
        1.,  # U700
        1.,  # U900
        1.,  # V10m
        1.,  # V200
        1.,  # V500
        1.,  # V700
        1.,  # V900
    ]

    cmaps = [
        'jet',#'seismic',  # Z250
        'jet',#'seismic',  # Z500
        'jet',#'seismic',  # Z700
        'jet',#'seismic',  # Z850
        'spring',#'RdGy',  # mslp
        'jet',#'seismic',  # ST
        'jet',#'seismic',  # T2m
        'jet',#'seismic',  # T500
        'jet',#'seismic',  # T850
        'gist_stern_r',#'RdGy',  # TCWV
        'RdGy',  # U10m
        'RdGy',  # U200
        'RdGy',  # U500
        'RdGy',  # U700
        'RdGy',  # U900
        'RdGy',  # V10m
        'RdGy',  # V200
        'RdGy',  # V500
        'RdGy',  # V700
        'RdGy',  # V900
    ]

    cmaps_inc = [
        'seismic',  # Z250
        'seismic',  # Z500
        'seismic',  # Z700
        'seismic',  # Z850
        'RdGy',  # mslp
        'seismic',  # ST
        'seismic',  # T2m
        'seismic',  # T500
        'seismic',  # T850
        'RdGy',  # TCWV
        'RdGy',  # U10m
        'RdGy',  # U200
        'RdGy',  # U500
        'RdGy',  # U700
        'RdGy',  # U900
        'RdGy',  # V10m
        'RdGy',  # V200
        'RdGy',  # V500
        'RdGy',  # V700
        'RdGy',  # V900
    ]

    vminmax = [  # Plotting ranges for 'full' fields, e.g. truth, analysis, background, etc
        (9000, 11000),  # (-500, 500),    # Z250
        (4500, 6000),  # (-500, 500),    # Z500
        (2500, 3200),  # (-500, 500),    # Z700
        (950, 1700),  # (-500, 500),    # Z850
        (970, 1030),  # (-30, 30),    # mslp
        (240, 310),  #(-20, 20),  # ST
        (240, 310),# (-20, 20),  # T2m
        (230, 270), #(-40, 40),  # T500
        (240, 300), #(-20, 20),  # T850
        (5, 70), #(-40, 40),  # TCVW
        (-20, 20),  # U10m
        (-100, 100),  # U200
        (-50, 50),  # U500
        (-30, 30),  # U700
        (-20, 20),#(-20, 20),  # U900
        (-20, 20),  # V10m
        (-100, 100),  # V200
        (-50, 50),  # V500
        (-30, 30),  # V700
        (-20, 20),  # V900
    ]

    vminmax_inc = [  # Plotting ranges for analysis increments
        (-30, 30),  # Z250
        (-30, 30),  # Z500
        (-30, 30),  # Z700
        (-30, 30),  # Z850
        (-1, 1),    # mslp
        (-1, 1),    # ST
        (-1, 1),    # T2m
        (-1, 1),    # T500
        (-1, 1),    # T850
        (-3, 3),    # TCVW
        (-0.5, 0.5),    # U10m
        (-1, 1),    # U200
        (-1, 1),    # U500
        (-1, 1),    # U700
        (-1, 1),    # U900
        (-0.5, 0.5),    # V10m
        (-1, 1),    # V200
        (-1, 1),    # V500
        (-1, 1),    # V700
        (-1, 1),    # V900
    ]

    vminmax_std = [  # Plotting ranges for standard deviations
        (0, 30),    # Z250
        (0, 20),    # Z500
        (0, 20),    # Z700
        (0, 20),    # Z850
        (0, 2), # mslp
        (0, 2), # ST
        (0, 2), # T2m
        (0, 2), # T500
        (0, 2), # T850
        (0, 5), # TCVW
        (0, 2), # U10m
        (0, 5), # U200
        (0, 5), # U500
        (0, 2), # U700
        (0, 2), # U900
        (0, 2), # V10m
        (0, 5), # V200
        (0, 5), # V500
        (0, 2), # V700
        (0, 2), # V900
    ]

    vminmax_std_EDA = [ # Plotting ranges for standard deviations
        (0, 5),  # Z250
        (0, 5),  # Z500
        (0, 5),  # Z700
        (0, 5),  # Z850
        (0, 0.5),  # mslp
        (0, 1),  # ST
        (0, 1),  # T2m
        (0, 0.5),  # T500
        (0, 1),  # T850
        (0, 2),  # TCVW
        (0, 1),  # U10m
        (0, 2),  # U200
        (0, 2),  # U500
        (0, 2),  # U700
        (0, 2),  # U900
        (0, 1),  # V10m
        (0, 2),  # V200
        (0, 2),  # V500
        (0, 2),  # V700
        (0, 2),  # V900
    ]
    return {'ENOTE':ENOTE, 'FIELD_ADJUSTEMENT_MULTIPLICATION':FIELD_ADJUSTEMENT_MULTIPLICATION,
            'vminmax':vminmax, 'vminmax_inc':vminmax_inc, 'vminmax_std':vminmax_std, 'vminmax_std_EDA':vminmax_std_EDA,
            'cmaps':cmaps, 'cmaps_inc':cmaps_inc}


def plot_Earth(fig,
               ax,
               field,
               vminmax,
               obs_locs=[], # The input should be a list with shape (2, 1, Nobs) with [0,0,:] being lons and [1,0,:] being lats
               obs_s=30,
               cmap='bwr',
               projection='robinson',
               cbar=True,
               cbar_extend='both',
               cbar_ticks=[],
               title='',
               unit='',
               title_loc='top',    # 'left', 'top', None
               unit_in_cbar=False,
               U=[],    # If not empty, displays arrows
               V=[],    # If not empty, displays arrows
               quiver_scale=5e1,
               quiver_reduction=2,
               quiver_color='k',
               quiverkey_props=[0.95, 0.05, 1],
               gridline_x_distance=None,
               gridline_y_distance=None,
               cbar_shrink=0.8,
               extra_contour_field=[],
               extra_contour_levels=[],
               extra_contour_colors=['b', 'r'],
               extra_contour_linestyles=['solid', 'solid'],
               extra_contour_linewidths=1.5,
               coastlines_linewidth=None,
               coordinate_labels=False,
               plot_type='pcolormesh',
               contourf_levels=10,
               coastlines_color='k'
               ):
    import numpy as np
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs

    lons = np.linspace(-180, 179, 360)
    lats = np.linspace(-89.5, 89.5, 180)
    lons, lats = np.meshgrid(lons, lats)
    if plot_type == 'pcolormesh':
        mesh = ax.pcolormesh(lons, lats, field, cmap=cmap, vmin=vminmax[0], vmax=vminmax[1], transform=ccrs.PlateCarree())
    elif plot_type == 'contourf':
        mesh = ax.contourf(lons, lats, field, cmap=cmap, vmin=vminmax[0], vmax=vminmax[1], extend='both', transform=ccrs.PlateCarree(), levels=contourf_levels)
    else:
        raise AttributeError    # plot_type not supported
    if len(obs_locs) > 0:
        ax.scatter(obs_locs[0][0], obs_locs[1][0], c='gold', edgecolor='k', s=obs_s, marker='*', transform=ccrs.PlateCarree(), zorder=1000)
    ax.set_global()
    if coastlines_linewidth:
        ax.coastlines(linewidth=coastlines_linewidth, color=coastlines_color)
    else:
        ax.coastlines(color=coastlines_color)
    #if not coordinate_labels:
    gl = ax.gridlines()
    if coordinate_labels:
        gl.xlabels_bottom = True
        gl.ylabels_left = True

    if gridline_x_distance:
        import matplotlib.ticker as mticker
        gl.xlocator = mticker.FixedLocator(np.arange(-180, 180, step=gridline_x_distance))
    if gridline_y_distance:
        import matplotlib.ticker as mticker
        gl.ylocator = mticker.FixedLocator(np.arange(-90, 90.01, step=gridline_y_distance))

    if len(U) > 0 and len(V) > 0:
        Q = ax.quiver(lons[::quiver_reduction, ::quiver_reduction], lats[::quiver_reduction, ::quiver_reduction],
                      U[::quiver_reduction, ::quiver_reduction], V[::quiver_reduction, ::quiver_reduction],
                      scale=quiver_scale, color=quiver_color,
                      transform=ccrs.PlateCarree())
        ax.quiverkey(Q, quiverkey_props[0], quiverkey_props[1], quiverkey_props[2], str(quiverkey_props[2])+r'$\,\frac{\mathrm{m}}{\mathrm{s}}$', labelpos='W')
        # if cbar:
        #     if cbar_shrink == 0.8:
        #         ax.quiverkey(Q, 0.95, 0.05, 1, r'$1\,\frac{\mathrm{m}}{\mathrm{s}}$', labelpos='W')
        #     else:
        #         ax.quiverkey(Q, 1.125, 0.07, 0.5, r'$0.5\,\frac{\mathrm{m}}{\mathrm{s}}$', labelpos='W')
        # else:
        #     ax.quiverkey(Q, 0.97, -0.10, 10, r'$10\,\frac{\mathrm{m}}{\mathrm{s}}$', labelpos='W')

    if len(extra_contour_field) > 0:
        if min(extra_contour_levels) < 0:
            ax.contour(lons, lats, extra_contour_field, levels=[i for i in extra_contour_levels if i < 0],
                       colors=[extra_contour_colors[0] for i in extra_contour_levels if i < 0],
                       linestyles=extra_contour_linestyles[0], linewidths=extra_contour_linewidths,
                       transform=ccrs.PlateCarree())
        if max(extra_contour_levels) > 0:
            ax.contour(lons, lats, extra_contour_field, levels=[i for i in extra_contour_levels if i > 0],
                       colors=[extra_contour_colors[1] for i in extra_contour_levels if i > 0],
                       linestyles=extra_contour_linestyles[1], linewidths=extra_contour_linewidths,
                       transform=ccrs.PlateCarree())

    if title_loc:
        if title_loc == 'left':
            ax.set_ylabel(title)
        elif title_loc == 'top':
            if len(obs_locs) > 0:
                ax.set_title(title, y=1.02)
            else:
                ax.set_title(title, y=1.02) # set , y=1.02 manually for adjustment experiments

    if cbar:
        if unit_in_cbar:
            cb = fig.colorbar(mesh, ax=ax, shrink=cbar_shrink, pad=0.05, extend=cbar_extend, label=unit)
        else:
            cb = fig.colorbar(mesh, ax=ax, shrink=cbar_shrink, pad=0.05, extend=cbar_extend)
        if len(cbar_ticks) > 0:
            cb.set_ticks(cbar_ticks)


def encoded_forecast_provider(
    FWD_root_model,
    AE_root_model,
    initial_conditions,
    forecast_start_datetime=None,
    forecast_end_datetime=None,
    forecast_len=None,
    keep_3D=False,
    echo=True,
    em=None,    # ensemble member index for full EDA
    return_full_NNfwd_prediction=False
):
    '''
    Provides encoded forecast for a chosen model.

    If initial_conditions is 'ERA5':
        If the encoded forecast is already in the database, it just loads it.
        Otherwise it computes it (and stores it in the database).

        You need to provide two of the following time-related arguments:
        forecast_start_time (datetime.datetime),
        forecast_end_time (datetime.datetime),
        forecast_len (int - hours)
        Produces an error if forecast_start_time - forecast_end_time != forecast_len * 1h.

        In case of persistence forecast, FWD_root_model='persistence' is sufficient

    Otherwise em should be provided.

    keep_3D:
        - True: Stores a 3-dimmensional output, suitable for Multiscale NFs from torchflow library
                Example: if AE_latent_space_shape is (1, 50, 11, 22), the shape of the saved output will be (50, 11, 22)
        - False: The latent space is saved as a transposed vector (1 x flattened_latent_dim)
                Example: if AE_latent_space_shape is (1, 50, 11, 22), the shape of the saved output will be (1, 12100)
    '''
    import os, sys
    master_dir = os.getenv('UGNN3DVar_master')
    sys.path.append(master_dir)
    sys.path.append(master_dir + '/NNs')
    sys.path.append(master_dir + '/NNs/models')

    if type(initial_conditions) == str:
        if initial_conditions != 'ERA5':
            raise AttributeError    # Spurious initial conditions

        from datetime import datetime, timedelta

        # ------------------------------------------------
        # Check if there are enough arguments provided and if they add up
        # ------------------------------------------------
        if not forecast_start_datetime:
            if (not forecast_end_datetime) or (forecast_len == None):
                raise AttributeError  # Not enought time-related arguments provided
            else:
                forecast_start_datetime = forecast_end_datetime - timedelta(hours=forecast_len)

        elif not forecast_end_datetime:
            if (not forecast_start_datetime) or (forecast_len == None):
                raise AttributeError  # Not enought time-related arguments provided
            else:
                forecast_end_datetime = forecast_start_datetime + timedelta(hours=forecast_len)

        elif forecast_len == None:
            forecast_len = int((forecast_end_datetime - forecast_start_datetime).total_seconds()) // 3600

        else:
            # Check if given arguments add up
            if forecast_start_datetime + timedelta(hours=forecast_len) != forecast_end_datetime:
                raise AttributeError  # forecast_start_time + timedelta(hours=forecast_len) != forecast_end_time

        # ------------------------------------------------
        # Dedicate a filename
        # ------------------------------------------------
        FWD_start_datetime = forecast_start_datetime
        FWD_end_datetime = forecast_end_datetime

        AE__datetime = FWD_end_datetime
        AE_date = (AE__datetime.day, AE__datetime.month, AE__datetime.year)
        AE_time = AE__datetime.hour
        FWD_start_date = (FWD_start_datetime.day, FWD_start_datetime.month, FWD_start_datetime.year)
        FWD_start_time = FWD_start_datetime.hour


        if AE_root_model == '%s/NNs/models/autoencoder_20_12100' % os.getenv('UGNN3DVar_master'):
            AE_latent_vec_shape = (1, 12100)
            AE_latent_space_shape = (1, 50, 11, 22)

        else:
            print('UNKNOWN AE_root_model')
            raise AttributeError    # unknown AE_root_model, maybe you just need to add it in loading_and_plotting_data.py


        shape_addon = ''
        alter_shape_addon = ''
        if keep_3D == True:
            reshaped_shape = AE_latent_space_shape[-3:]
            shape_addon += '_3D'
        else:
            reshaped_shape =AE_latent_vec_shape
            alter_shape_addon += '_3D'



        encoded_prediction_filename = ('%s/saved_encodings/%s/encoded_pred_%04d_%02d_%02d_%02d_to_%04d_%02d_%02d_%02d%s.pkl') % (
                              AE_root_model, FWD_root_model[FWD_root_model.rfind("/") + 1:],
                              FWD_start_datetime.year, FWD_start_datetime.month, FWD_start_datetime.day,
                              FWD_start_datetime.hour,
                              FWD_end_datetime.year, FWD_end_datetime.month, FWD_end_datetime.day,
                              FWD_end_datetime.hour,
                              shape_addon
                            )

        encoded_prediction_filename_alter_shape = ('%s/saved_encodings/%s/encoded_pred_%04d_%02d_%02d_%02d_to_%04d_%02d_%02d_%02d%s.pkl') % (
                              AE_root_model, FWD_root_model[FWD_root_model.rfind("/") + 1:],
                              FWD_start_datetime.year, FWD_start_datetime.month, FWD_start_datetime.day,
                              FWD_start_datetime.hour,
                              FWD_end_datetime.year, FWD_end_datetime.month, FWD_end_datetime.day,
                              FWD_end_datetime.hour,
                              alter_shape_addon
                            )

        # ------------------------------------------------
        # Check if the encoded forecast already exists
        # ------------------------------------------------

        import os, pickle

        if not return_full_NNfwd_prediction:
            if os.path.isfile(encoded_prediction_filename):
                if echo:
                    print('\nEncoded forecast already in the database!')
                encoded_forecast = pickle.load(open(encoded_prediction_filename, 'rb')) # shape e.g. (1, 12100) - has to be (1, something)
                if echo:
                    print('\nLoaded encoded prediction from', encoded_prediction_filename)
                return encoded_forecast
            elif os.path.isfile(encoded_prediction_filename_alter_shape):
                if echo:
                    print('\nEncoded forecast with alternative shape already in the database!')
                encoded_forecast = pickle.load(open(encoded_prediction_filename_alter_shape, 'rb')).reshape(reshaped_shape) # shape e.g. (1, 49500) - has to be (1, something)
                if echo:
                    print('\nLoaded encoded prediction with alternative shape from', encoded_prediction_filename_alter_shape)
                return encoded_forecast
            else:
                if echo:
                    print('\nENCODED FORECAST NOT YET IN THE DATABASE!')


        # ------------------------------------------------
        # Computing the forecast and storing it
        # ------------------------------------------------


        import predict_1hr, predict_1hr_AE
        import data_preparation_1hr
        import unet_model_7x7_4x_depth7_interdepth_padding_1_degree_v01
        import unet_model_7x7_4x_depth4_interdepth_padding_1_degree_v01_stride2b_no_skip_AE
        import pickle
        import numpy as np
        import torch
        import torch.nn as nn

        # Ustvarim vse tabele
        tabele = create_lists()

        # Poti do datotek
        root_annotations = tabele["root_annotations"]
        ANNOTATIONS_NAMES = tabele["ANNOTATIONS_NAMES"]

        # Scalerji
        root_scaler = tabele["root_scaler"]
        SCALER_NAMES = tabele["SCALER_NAMES"]
        SCALER_TYPES = tabele["SCALER_TYPES"]

        # Poti do klimatoloških polj
        # ROOTS_SAVE_CLIMA = tabele["ROOTS_SAVE_CLIMA"]

        # Pot do statičnih polj
        STATIC_FIELDS_ROOTS = tabele["STATIC_FIELDS_ROOTS"]



        fwd_model = unet_model_7x7_4x_depth7_interdepth_padding_1_degree_v01.UNet(
                in_channels=1*20+3, out_channels=20, depth=7, start_filts=250,
                up_mode='transpose', merge_mode='concat')
        num_prediction_steps = forecast_len // 12


        # Tabela s potjo do  fwd modela
        FWD_ROOTS_MODEL = [FWD_root_model]
        # Tabela s fwd modelom
        FWD_MODELS = [fwd_model]
        # Ime modela - to je fiksno ("trained_model_weights.pth" vsebuje uteži z najboljšim ACC med treningom)
        model_name="trained_model_weights.pth"

        # IMPORT AE
        if AE_root_model == '%s/NNs/models/autoencoder_20_12100' % os.getenv('UGNN3DVar_master'):
            AE_model = unet_model_7x7_4x_depth4_interdepth_padding_1_degree_v01_stride2b_no_skip_AE.UNet(
                in_channels=1 * 20 + 3, out_channels=20, depth=4, start_filts=50,
                up_mode='transpose', merge_mode='concat')

        else:
            print('UNKNOWN AE_root_model')
            raise AttributeError    # unknown AE_root_model, maybe you just need to add it in loading_and_plotting_data.py

        # Poskrbi za ustrezno delovanje tako na CPU, kot GPU
        device_ids = [0, 1]
        AE_model = nn.DataParallel(AE_model, device_ids=device_ids)

        # Tabela s potjo do AE modela
        AE_ROOTS_MODEL = [AE_root_model]
        # Tabela s AE modelom
        AE_MODELS = [AE_model]
        AE_predicting = predict_1hr_AE.Predicting(AE_ROOTS_MODEL, model_name)



        if FWD_root_model[-len('persistence'):] == 'persistence':
            st = datetime.now()
            AE_latent = AE_predicting.encode_decode(
                root_annotations,
                ANNOTATIONS_NAMES,
                AE_MODELS,
                STATIC_FIELDS_ROOTS,
                root_scaler,
                SCALER_NAMES,
                SCALER_TYPES,
                FWD_start_date,
                num_of_prediction_steps=1,  # num_of_prediction_steps, ampak dela samo za 1
                action="encode",
                T=FWD_start_time,
                use_ERA5=True,
                list_of_input_fields=None  #
            ).reshape(reshaped_shape)
            et = datetime.now()
            if echo:
                print('time for encoding ground truth', et - st)
            pickle.dump(AE_latent, open(encoded_prediction_filename, 'wb'))

        else:
            prediction_filename = ('%s/saved_predictions/pred_%04d_%02d_%02d_%02d_to_%04d_%02d_%02d_%02d.pkl') % (
                FWD_root_model,
                FWD_start_datetime.year, FWD_start_datetime.month, FWD_start_datetime.day, FWD_start_datetime.hour,
                FWD_end_datetime.year, FWD_end_datetime.month, FWD_end_datetime.day, FWD_end_datetime.hour
            )
            if os.path.isfile(prediction_filename):
                # The forecast for chosen time instances already exists, we only need to encode it
                if echo:
                    print('\nForecast %04d-%02d-%02d-%02d to %04d-%02d-%02d-%02d already exists' % (
                        FWD_start_datetime.year, FWD_start_datetime.month, FWD_start_datetime.day, FWD_start_datetime.hour,
                        FWD_end_datetime.year, FWD_end_datetime.month, FWD_end_datetime.day, FWD_end_datetime.hour
                    ))
                final_pred = pickle.load(open(prediction_filename, 'rb'))
            else:
                # Compute forecast_timesteps 1-hour forecasts
                if echo:
                    print('\nComputing forecast')
                FWD_predicting = predict_1hr.Predicting(FWD_ROOTS_MODEL, model_name)
                # Napoved vrne tabelo "resnic = loading ERA5 reanaliz" in tabelo napovedi modela.
                # V primeru samokodirnika je preds[0]=ERA5, preds[1]=avtodekodirano polje
                st = datetime.now()
                if echo:
                    print('start time', datetime.now())
                truths, preds = FWD_predicting.pred_time_series(    # truths would only indeed be "true" if time stepping was 1hr, but it is 12hr, so this is redundant
                    root_annotations,
                    ANNOTATIONS_NAMES,
                    FWD_MODELS,
                    STATIC_FIELDS_ROOTS,
                    root_scaler,
                    SCALER_NAMES,
                    SCALER_TYPES,
                    (FWD_start_datetime.day, FWD_start_datetime.month, FWD_start_datetime.year),
                    num_of_prediction_steps=num_prediction_steps,
                    T=FWD_start_datetime.hour)

                et = datetime.now()
                if echo:
                    print('time for prediction', et - st)
                truths, preds = np.array(truths).astype(np.float32), np.array(preds).astype(
                    np.float32)  # shape=(24 or 20, num_prediction_steps+1, 180, 360)
                final_pred = list(np.expand_dims(preds[:, -1, :, :], axis=1))
                for i in range(len(final_pred)):
                    final_pred[i] = list(final_pred[i])
                
                if return_full_NNfwd_prediction:
                    print('RETURNING FULL NNfwd prediction!')
                    return final_pred


            # Encode the forecast
            if echo:
                print('\nEncoding forecast')
            st = datetime.now()
            AE_latent = AE_predicting.encode_decode(
                root_annotations,
                ANNOTATIONS_NAMES,
                AE_MODELS,
                STATIC_FIELDS_ROOTS,
                root_scaler,
                SCALER_NAMES,
                SCALER_TYPES,
                AE_date,
                num_of_prediction_steps=1,  # num_of_prediction_steps, ampak dela samo za 1
                action="encode",
                T=AE_time,
                use_ERA5=False,
                list_of_input_fields=final_pred  #
            ).reshape(reshaped_shape)
            et = datetime.now()
            if echo:
                print('time for encoding forecast', et - st)

            pickle.dump(AE_latent, open(encoded_prediction_filename, 'wb'))

        return AE_latent

    else:
        if FWD_root_model == 'ens_B':   # ensemble members
            # ------------------------------------------------
            # Dedicate a filename
            # ------------------------------------------------
            FWD_start_datetime = forecast_start_datetime
            FWD_end_datetime = forecast_end_datetime

            AE__datetime = FWD_end_datetime
            AE_date = (AE__datetime.day, AE__datetime.month, AE__datetime.year)
            AE_time = AE__datetime.hour

            if AE_root_model == '%s/NNs/models/autoencoder_20_12100' % os.getenv('UGNN3DVar_master'):
                AE_latent_vec_shape = (1, 12100)
                AE_latent_space_shape = (1, 50, 11, 22)

            else:
                print('UNKNOWN AE_root_model')
                raise AttributeError  # unknown AE_root_model, maybe you just need to add it in loading_and_plotting_data.py

            shape_addon = ''
            alter_shape_addon = ''
            if keep_3D == True:
                reshaped_shape = AE_latent_space_shape[-3:]
                shape_addon += '_3D'
            else:
                reshaped_shape = AE_latent_vec_shape
                alter_shape_addon += '_3D'


            encoded_em_filename = ('%s/saved_encodings/%s/encoded_%04d_%02d_%02d_%02d_em%02d%s.pkl') % (
                AE_root_model, FWD_root_model,
                FWD_start_datetime.year, FWD_start_datetime.month, FWD_start_datetime.day, FWD_start_datetime.hour, em,
                shape_addon
            )
            encoded_em_filename_alter_shape = ('%s/saved_encodings/%s/encoded_%04d_%02d_%02d_%02d_em%02d%s.pkl') % (
                AE_root_model, FWD_root_model,
                FWD_start_datetime.year, FWD_start_datetime.month, FWD_start_datetime.day, FWD_start_datetime.hour, em,
                alter_shape_addon
            )

            # ------------------------------------------------
            # Check if the encoded ensemble member already exists
            # ------------------------------------------------

            import os, pickle

            if os.path.isfile(encoded_em_filename):
                if echo:
                    print(f'\nEncoded ensemble member {em} already in the database!')
                encoded_forecast = pickle.load(open(encoded_em_filename, 'rb'))
                if echo:
                    print('\nLoaded encoded ensemble member from', encoded_em_filename)
                return encoded_forecast
            elif os.path.isfile(encoded_em_filename_alter_shape):
                if echo:
                    print(f'\nEncoded ensemble member {em} with alternative shape already in the database!')
                encoded_forecast = pickle.load(open(encoded_em_filename_alter_shape, 'rb')).reshape(
                    reshaped_shape)
                if echo:
                    print('\nLoaded encoded ensemble member with alternative shape from',
                          encoded_em_filename_alter_shape)
                return encoded_forecast
            else:
                if echo:
                    print(f'\nENCODED ENSEMBLE MEMBER {em} NOT YET IN THE DATABASE!')

                # ------------------------------------------------
                # Encoding the ensemble member
                # ------------------------------------------------

                import predict_1hr, predict_1hr_AE
                import data_preparation_1hr
                import unet_model_7x7_4x_depth7_interdepth_padding_1_degree_v01
                import unet_model_7x7_4x_depth4_interdepth_padding_1_degree_v01_stride2b_no_skip_AE
                import pickle
                import numpy as np
                import torch
                import torch.nn as nn
                from datetime import datetime

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


                # IMPORT AE
                model_name = "trained_model_weights.pth"

                if AE_root_model == '%s/NNs/models/autoencoder_20_12100' % os.getenv('UGNN3DVar_master'):
                    AE_model = unet_model_7x7_4x_depth4_interdepth_padding_1_degree_v01_stride2b_no_skip_AE.UNet(
                        in_channels=1 * 20 + 3, out_channels=20, depth=4, start_filts=50,
                        up_mode='transpose', merge_mode='concat')

                else:
                    print('UNKNOWN AE_root_model')
                    raise AttributeError  # unknown AE_root_model, maybe you just need to add it in loading_and_plotting_data.py

                # Poskrbi za ustrezno delovanje tako na CPU, kot GPU
                device_ids = [0, 1]
                AE_model = nn.DataParallel(AE_model, device_ids=device_ids)

                # Tabela s potjo do AE modela
                AE_ROOTS_MODEL = [AE_root_model]
                # Tabela s AE modelom
                AE_MODELS = [AE_model]
                AE_predicting = predict_1hr_AE.Predicting(AE_ROOTS_MODEL, model_name)

                if echo:
                    print('\nEncoding')
                st = datetime.now()
                AE_latent = AE_predicting.encode_decode(
                    root_annotations,
                    ANNOTATIONS_NAMES,
                    AE_MODELS,
                    STATIC_FIELDS_ROOTS,
                    root_scaler,
                    SCALER_NAMES,
                    SCALER_TYPES,
                    AE_date,
                    num_of_prediction_steps=1,  # num_of_prediction_steps, ampak dela samo za 1
                    action="encode",
                    T=AE_time,
                    use_ERA5=False,
                    list_of_input_fields=initial_conditions #
                ).reshape(reshaped_shape)
                et = datetime.now()
                if echo:
                    print('time for encoding', et - st)

                pickle.dump(AE_latent, open(encoded_em_filename, 'wb'))

                return AE_latent

        else:
            print('\n\nNNfwd MODEL ONLY TAKES ERA5 AS INPUT!')

            raise AttributeError # NNfwd only takes ERA5 as input


def EDA_ensemble_member_load(
        which_ensemble='forecasts',
        em_idx=1,
        datetime_str="2024-04-14-00",
        return_encoded=True,
        echo=False
):
    '''
    which_ensemble: "forecasts" for members ensemble of backgrounds
    datetime_str should be in format "yyyy-mm-dd-hh"

    As we have no permission to share the IFS ensemble members, we only provide their encoded versions.
    Code below #*****# serves just as demonstration, how we treated them in our study to encode them
    '''

    import os

    if which_ensemble == 'forecasts':
        FWD_root_model = 'ens_B'
    else:
        raise AttributeError # Unrecognised which_ensemble

    import datetime
    encoded_em = encoded_forecast_provider(
        FWD_root_model=FWD_root_model,
        AE_root_model='%s/NNs/models/autoencoder_20_12100' % os.getenv('UGNN3DVar_master'),
        initial_conditions=[1,2,3],
        forecast_start_datetime=datetime.datetime.strptime(datetime_str, '%Y-%m-%d-%H'),
        # has no effect, but has to be set
        forecast_end_datetime=datetime.datetime.strptime(datetime_str, '%Y-%m-%d-%H'),
        # has no effect, but has to be set
        keep_3D=False,
        echo=echo,
        em=em_idx
    )

    return encoded_em

    #*****#

    # ---------------------------------------------
    # Set the main direcrory for data
    # ---------------------------------------------

    if which_ensemble == 'forecasts':
        ensemble_main_dir = '' # path do data
    else:
        print('Unsupported ensemble type (which_ensemble):', which_ensemble)
        raise AttributeError # Unsupported ensemble type


    # ---------------------------------------------
    # Load the ensemble member for each respective field and stack them to a list of np arrays, as suitable to enter the encoded_forecast_provider
    # ---------------------------------------------
    import numpy as np
    from datetime import datetime

    # This loading might seem redundant if the encoded version is already stored, but it only takes less than 0.01 seconds...
    all_fields = np.array([
        np.load(f'{ensemble_main_dir}/Geopotential_250/z_{datetime_str}-em{em_idx:02d}.npy'),
        np.load(f'{ensemble_main_dir}/Geopotential_500/z_{datetime_str}-em{em_idx:02d}.npy'),
        np.load(f'{ensemble_main_dir}/Geopotential_700/z_{datetime_str}-em{em_idx:02d}.npy'),
        np.load(f'{ensemble_main_dir}/Geopotential_850/z_{datetime_str}-em{em_idx:02d}.npy'),
        np.load(f'{ensemble_main_dir}/Mslp/msl_{datetime_str}-em{em_idx:02d}.npy'),
        np.load(f'{ensemble_main_dir}/SurfaceTemperature/stl1_{datetime_str}-em{em_idx:02d}.npy'),
        np.load(f'{ensemble_main_dir}/Temperature_2m/t2m_{datetime_str}-em{em_idx:02d}.npy'),
        np.load(f'{ensemble_main_dir}/Temperature_500/t_{datetime_str}-em{em_idx:02d}.npy'),
        np.load(f'{ensemble_main_dir}/Temperature_850/t_{datetime_str}-em{em_idx:02d}.npy'),
        np.load(f'{ensemble_main_dir}/Total_column_water_vapour/tcwv_{datetime_str}-em{em_idx:02d}.npy'),
        np.load(f'{ensemble_main_dir}/Wind_U_10m/u10_{datetime_str}-em{em_idx:02d}.npy'),
        np.load(f'{ensemble_main_dir}/Wind_U_200/u_{datetime_str}-em{em_idx:02d}.npy'),
        np.load(f'{ensemble_main_dir}/Wind_U_500/u_{datetime_str}-em{em_idx:02d}.npy'),
        np.load(f'{ensemble_main_dir}/Wind_U_700/u_{datetime_str}-em{em_idx:02d}.npy'),
        np.load(f'{ensemble_main_dir}/Wind_U_900/u_{datetime_str}-em{em_idx:02d}.npy'),
        np.load(f'{ensemble_main_dir}/Wind_V_10m/v10_{datetime_str}-em{em_idx:02d}.npy'),
        np.load(f'{ensemble_main_dir}/Wind_V_200/v_{datetime_str}-em{em_idx:02d}.npy'),
        np.load(f'{ensemble_main_dir}/Wind_V_500/v_{datetime_str}-em{em_idx:02d}.npy'),
        np.load(f'{ensemble_main_dir}/Wind_V_700/v_{datetime_str}-em{em_idx:02d}.npy'),
        np.load(f'{ensemble_main_dir}/Wind_V_900/v_{datetime_str}-em{em_idx:02d}.npy'),
    ])


    # Do I return the fields in their physical form, or do I encode them?
    if not return_encoded:
        return  all_fields

    else:
        # To make it suitable for entering the encoder, reshape all_fields to (20, 1, 180, 360) and make it a list of numpy arrays
        all_fields = np.expand_dims(all_fields, 1)
        list_of_input_fields = list(all_fields)

        import datetime
        encoded_em = encoded_forecast_provider(
            FWD_root_model=ensemble_main_dir.split('/')[-1],
            AE_root_model='%s/NNs/models/autoencoder_20_12100' % os.getenv('UGNN3DVar_master'),
            initial_conditions=list_of_input_fields,
            forecast_start_datetime=datetime.datetime.strptime(datetime_str, '%Y-%m-%d-%H'),    # has no effect, but has to be set
            forecast_end_datetime=datetime.datetime.strptime(datetime_str, '%Y-%m-%d-%H'),      # has no effect, but has to be set
            keep_3D=False,
            echo=echo,
            em=em_idx
        )

        return encoded_em



