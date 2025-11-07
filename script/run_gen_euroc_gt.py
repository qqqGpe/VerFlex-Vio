'''
Author: pengen.gao gaope.hb@gmail.com
Date: 2025-11-08 01:08:58
Copyright (c) 2025 by gaope.hb@gmail.com, All Rights Reserved.
'''
import os
import numpy as np

DATASET_DIR = "/home/gao/dataset/euroc_mav"
CASE_LIST = ["MH_01_easy", "MH_02_easy", "MH_03_medium", "MH_04_difficult", "MH_05_difficult",
             "V1_01_easy", "V1_02_medium", "V1_03_difficult", "V2_01_easy", "V2_02_medium", "V2_03_difficult"]
DEST_DIR = "/home/gao/ws/catkin_ws/src/vio_backend/data/euroc"

def load_euroc_groundtruth():
    """
    Load EuRoC groundtruth data from each case in CASE_LIST
    Returns a dictionary with case names as keys and data as values
    """
    gt_data = {}

    for case in CASE_LIST:
        gt_file = os.path.join(DATASET_DIR, case, "mav0", "state_groundtruth_estimate0", "data.csv")

        if os.path.exists(gt_file):
            print(f"Loading groundtruth for {case}...")
            # Read CSV file, skip header if exists
            data = np.loadtxt(gt_file, delimiter=',', skiprows=1)
            gt_data[case] = data
            print(f"Loaded {len(data)} groundtruth entries for {case}")
        else:
            print(f"Warning: Groundtruth file not found for {case}: {gt_file}")

    return gt_data

def process_groundtruth_data(gt_data):
    """
    Process groundtruth data - placeholder for future operations
    """
    for case, data in gt_data.items():
        # Extract columns 1 -> 7
        selected_cols = data[:, 0:8]

        # Swap second column to last: [0, 4, 5, 6, 7] -> [0, 5, 6, 7, 4]
        reordered_data = selected_cols[:, [0, 1, 2, 3, 5, 6, 7, 4]]
        reordered_data[:, 0] = reordered_data[:, 0] * 1e-9

        # Create destination directory for this case
        case_dest_dir = os.path.join(DEST_DIR, case)
        os.makedirs(case_dest_dir, exist_ok=True)

        # Save to CSV file with header
        output_file = os.path.join(case_dest_dir, f"{case}_groundtruth.csv")
        header = "#timestamp tx ty tz qx qy qz qw"

        # Write header manually to avoid extra newline at the end
        with open(output_file, 'w') as file:
            file.write(header + '\n')
            np.savetxt(file, reordered_data, delimiter=' ', fmt='%.9f')

        print(f"Saved processed groundtruth for {case} to {output_file}")

if __name__ == "__main__":
    # Load all groundtruth data
    groundtruth_data = load_euroc_groundtruth()

    # Process the data (placeholder for future operations)
    process_groundtruth_data(groundtruth_data)

