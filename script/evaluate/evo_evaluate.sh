#!/bin/bash
###
 # @Author: pengen.gao gaope.hb@gmail.com
 # @Date: 2025-12-25 00:49:57
 # Copyright (c) 2025 by gaope.hb@gmail.com, All Rights Reserved.
###

# Check if two arguments are provided
# Default file paths (can be configured here)
DATA_CASE="MH_01_easy"
GT_FILE="$HOME/ws/catkin_ws/src/vio_backend/data/euroc/ground_truth/$DATA_CASE/${DATA_CASE}_groundtruth.csv"
# RESULT_FILE="/home/gao/ws/catkin_ws/src/vio_backend/data/euroc/openvins/MH_01_easy.csv"
RESULT_FILE="/home/gao/ws/catkin_ws/src/vio_backend/data/euroc/openvins/MH_01_easy_slam.csv"

# Use command line arguments if provided, otherwise use defaults
if [ $# -eq 2 ]; then
    A_FILE=$1
    B_FILE=$2
elif [ $# -eq 0 ]; then
    A_FILE=$GT_FILE
    B_FILE=$RESULT_FILE
    echo "Using default files:"
    echo "  A: $A_FILE"
    echo "  B: $B_FILE"
else
    echo "Usage: $0 [<A_file_path> <B_file_path>]"
    echo "Or configure default paths in the script."
    exit 1
fi

# Check if files exist
if [ ! -f "$A_FILE" ]; then
    echo "Error: File A '$A_FILE' does not exist."
    exit 1
fi

if [ ! -f "$B_FILE" ]; then
    echo "Error: File B '$B_FILE' does not exist."
    exit 1
fi

# Run evo_ape command
evo_ape tum "$A_FILE" "$B_FILE" -va