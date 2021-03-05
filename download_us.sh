#!/bin/bash

# Download LPD model
mkdir -p ./models/LP/LPD
cd ./models/LP/LPD
wget https://api.ngc.nvidia.com/v2/models/nvidia/tlt_lpdnet/versions/pruned_v1.0/files/usa_pruned.etlt
wget https://api.ngc.nvidia.com/v2/models/nvidia/tlt_lpdnet/versions/pruned_v1.0/files/usa_lpd_cal.bin
wget https://api.ngc.nvidia.com/v2/models/nvidia/tlt_lpdnet/versions/pruned_v1.0/files/usa_lpd_label.txt
cd -

# Download LPR model
mkdir -p ./models/LP/LPR
cd ./models/LP/LPR
wget https://api.ngc.nvidia.com/v2/models/nvidia/tlt_lprnet/versions/deployable_v1.0/files/us_lprnet_baseline18_deployable.etlt
touch labels_us.txt
cd -