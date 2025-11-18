The trained weights for the neural networks are available at: https://zenodo.org/records/15706040
Download them and store the pth files to:
- NNs/models/U-Net_IT_ALL_numpred2_train_sumMSE_testACC_isl1_250ch_ks7x7_1hr_parallel_2024_11 (for U-Net)
- NNs/models/autoencoder_20_12100 (for AE)

The main code for the experiments is in algorithm folder. Use run_3D-Var_multiple_runs.sh to run them.

Before running the experiments, you need to store the "UGNN3DVar_master" variable in your python environment, which points to the location of the master directory related to this repository in your local system.

The perturbed backgrounds and the observations, used in the paper, are available in algorithm/experiments/data/20ch.

For copyright reasons we don't provide full ERA5 fields and IFS EDA ensemble of backgrounds, but only their encoded counterparts in:
- NNs/models/autoencoder_20_12100/saved_encodings/U-Net_IT_ALL_numpred2_train_sumMSE_testACC_isl1_250ch_ks7x7_1hr_parallel_2024_11
- NNs/models/autoencoder_20_12100/saved_encodings/ens_B

The B-matrices may be recomputed using the scripts in B-matrix folder. Their diagonal elements, which are used for running the assimilation experiments are storred in:
- NNs/models/U-Net_IT_ALL_numpred2_train_sumMSE_testACC_isl1_250ch_ks7x7_1hr_parallel_2024_11/B-matrices/autoencoder_20_12100
- NNs/models/autoencoder_20_12100/EDA/B-matrices/ens_B

The yaml file for conda vitual environment that we used for our research is given in UGNN3DVar_conda_virtualenv.yml.
