#!/bin/bash



bash run_3D-Var.sh y y "--obs_qty=Z500 --obs_inc=30.0 --obs_std=10.0 --custom_addon=singobs_Ljubljana_SGD --singobs_lat=46.1 --singobs_lon=14.5 --savefig_dir=Ljubljana --plot_singles --ensemble=5 --init_lr=0.3 --preconditioned_3D_Var"

#bash run_3D-Var.sh n y "--obs_qty=Z500 --obs_inc=30.0 --obs_std=10.0 --custom_addon=singobs_Ljubljana_SGD --singobs_lat=46.1 --singobs_lon=14.5 --savefig_dir=Ljubljana --plot_singles --ensemble=100 --init_lr=0.3 --preconditioned_3D_Var --obs_datetime=2020-01-01-00"
#bash run_3D-Var.sh n y "--obs_qty=Z500 --obs_inc=30.0 --obs_std=10.0 --custom_addon=singobs_Ljubljana_SGD --singobs_lat=46.1 --singobs_lon=14.5 --savefig_dir=Ljubljana --plot_singles --ensemble=100 --init_lr=0.3 --preconditioned_3D_Var --obs_datetime=2020-01-08-00"


#bash run_3D-Var.sh n y "--True_EDA --obs_datetime=2024-04-14-00 --obs_qty=Z500 --obs_inc=30.0 --obs_std=10.0 --custom_addon=singobs_Ljubljana_mean --singobs_lat=46.1 --singobs_lon=14.5 --savefig_dir=Ljubljana --plot_singles --ensemble=50 --init_lr=0.3 --preconditioned_3D_Var --True_EDA_control_or_mean=mean"



#bash run_3D-Var.sh n y "--True_EDA --obs_datetime=2024-04-14-00 --obs_qty=Z500 --obs_inc=30.0 --obs_std=10.0 --custom_addon=singobs_Ljubljana_mean_CLIM_B --singobs_lat=46.1 --singobs_lon=14.5 --savefig_dir=Ljubljana --plot_singles --ensemble=50 --init_lr=0.3 --preconditioned_3D_Var --True_EDA_control_or_mean=mean"
# WARNING: To use climatological B with operational EDA backgrounds, you need to set also set this manually in algorithm-serial.py (search for *-----*)

#bash run_3D-Var.sh n y "--obs_qty=TCWV --obs_inc=10.0 --custom_addon=singobs_Central_Atlantic_SGD --obs_std=3.0 --singobs_lon=-33.0 --singobs_lat=0.0 --plot_qty=all --ensemble=100 --init_lr=0.3 --savefig_dir=TCWV --preconditioned_3D_Var --plot_singles" 


