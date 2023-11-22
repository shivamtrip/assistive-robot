!usr/bin/env bash

# Convert data to vlmaps format
!python ../scripts/save_data_rtabmap_test.py --fps 10

# Create vlmaps
!python ../scripts/main.py --use_self_built_map True --depth_sample_rate 30 --cs 0.05 --gs 1000

# Inference vlmaps
!python ../scripts/main.py --use_self_built_map True --depth_sample_rate 30 --cs 0.05 --gs 1000 --inference True