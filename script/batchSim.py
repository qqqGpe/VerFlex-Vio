'''
Author: pengen.gao gaope.hb@gmail.com
Date: 2025-04-07 23:31:39
Copyright (c) 2025 by gaope.hb@gmail.com, All Rights Reserved.
'''
#!/usr/bin/env python3
import os
import subprocess
import datetime
import time
import psutil
import logging
import argparse
import csv
from tabulate import tabulate

# Substitute with your dataset path
DATA_DIR = os.path.join(os.path.expanduser("~"), "dataset/euroc_mav")
WS_DIR = os.path.join(os.path.expanduser("~"), "ws/catkin_ws")
LOG_DIR = os.path.join(WS_DIR, "src/vio_backend/log/vio_sim_" + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M"))
CODEBASE_DIR = os.path.join(WS_DIR, "src/vio_backend")
# Define the list of cases to process
CASE_LIST = ["MH_01_easy", "MH_02_easy"]

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.StreamHandler()  # Log to console
    ]
)
logger = logging.getLogger(__name__)


def is_process_running(process):
    try:
        return process.is_running() and process.status() != psutil.STATUS_ZOMBIE
    except (psutil.NoSuchProcess, psutil.ZombieProcess, psutil.AccessDenied):
        return False


def run_command(command):
    process = subprocess.Popen(command, shell=True, executable="/bin/zsh")
    process_state = psutil.Process(process.pid)
    time.sleep(5)
    while is_process_running(process_state):
        try:
            if process_state.status() == psutil.STATUS_RUNNING:
                time.sleep(1)
        except psutil.NoSuchProcess:
            logger.info("SLAM system has terminated.")
            break

    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()


def evo_ape_cmd(log_dir, groundtruth_file, simulate_file, case_name):
    ape_output_file = os.path.join(log_dir, case_name + '_ape.zip')
    cmd_ape = f"evo_ape tum {groundtruth_file} {simulate_file} -va --save_results {ape_output_file}"
    logger.info(f"Running EVO APE command for case: {case_name}")
    run_command(cmd_ape)


def evo_res_cmd(log_dir, groundtruth_file, simulate_file, case_name):
    ape_output_file = os.path.join(log_dir, case_name + '_ape.zip')
    res_output_file = os.path.join(log_dir, case_name + '_res.csv')
    cmd_res = f"evo_res {ape_output_file} --save_table {res_output_file}"
    logger.info(f"Running EVO RES command for case: {case_name}")
    run_command(cmd_res)


def extract_results(log_dir, case_name):
    res_output_file = os.path.join(log_dir, case_name + '_res.csv')
    if os.path.exists(res_output_file):
        try:
            with open(res_output_file, 'r') as f:
                lines = f.readlines()
                if len(lines) >= 2:
                    # Skip first line (header) and read second line
                    values = lines[1].strip().split(',')
                    if len(values) >= 8:
                        result_dict = {
                            'dataset_name': case_name,
                            'rmse': float(values[1]),
                            'mean': float(values[2]),
                            'median': float(values[3]),
                            'std': float(values[4]),
                            'min': float(values[5]),
                            'max': float(values[6]),
                            'sse': float(values[7])
                        }
                        logger.info(f"Extracted results for {case_name}: {result_dict}")
                        return result_dict
                    else:
                        logger.error(f"Insufficient columns in results file {res_output_file}")
                        return None
                else:
                    logger.error(f"Results file {res_output_file} has insufficient lines")
                    return None
        except Exception as e:
            logger.error(f"Error reading results file {res_output_file}: {e}")
            return None
    else:
        logger.warning(f"Results file not found: {res_output_file}")
        return None


def evaluate_results(log_dir, case_name):
    logger.info(f"Evaluating results for case: {case_name}")
    # Find groundtruth file in CODEBASE_DIR/data/euroc
    ground_truth_data_dir = os.path.join(CODEBASE_DIR, "data/euroc")
    groundtruth_file = None

    # Look for matching case folder in ground truth data directory
    for item in os.listdir(ground_truth_data_dir):
        item_path = os.path.join(ground_truth_data_dir, item)
        if os.path.isdir(item_path) and case_name in item:
            # Find groundtruth file in the matching folder
            for file in os.listdir(item_path):
                if "groundtruth" in file.lower() or "gt" in file.lower():
                    groundtruth_file = os.path.join(item_path, file)
                    break
            if groundtruth_file:
                break

    if groundtruth_file:
        logger.info(f"Found groundtruth file: {groundtruth_file}")
    else:
        logger.warning(f"No groundtruth file found for case: {case_name}")

    # Look for simulate data file in LOG_DIR
    simulate_file = None
    for file in os.listdir(log_dir):
        if case_name in file and file.endswith('.csv'):
            simulate_file = os.path.join(log_dir, file)
            break

    if simulate_file:
        logger.info(f"Found simulate file: {simulate_file}")
    else:
        logger.warning(f"No simulate file found for case: {case_name}")

    # Evaluate vio running results
    if groundtruth_file and simulate_file:
        evo_ape_cmd(log_dir, groundtruth_file, simulate_file, case_name)
        evo_res_cmd(log_dir, groundtruth_file, simulate_file, case_name)
        result = extract_results(log_dir, case_name)
        return result
    else:
        logger.warning(f"Skipping evaluation for case: {case_name} due to missing files.")
        return None


def save_and_show_results(log_dir, result_list):
    # Filter out None results
    valid_results = [result for result in result_list if result is not None]

    if not valid_results:
        logger.warning("No valid results to save and display.")
        return

    # Create summary table filename
    summary_file = os.path.join(log_dir, "summary_results.csv")

    # Write results to CSV file
    try:
        fieldnames = ['dataset_name', 'rmse', 'mean', 'median', 'std', 'min', 'max', 'sse']
        with open(summary_file, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for result in valid_results:
                writer.writerow(result)
        logger.info(f"Results saved to: {summary_file}")
    except Exception as e:
        logger.error(f"Error saving results to CSV: {e}")

    # Display results using tabulate
    try:

        # Prepare data for tabulate
        headers = ['Dataset', 'RMSE', 'Mean', 'Median', 'Std', 'Min', 'Max', 'SSE']
        table_data = []
        for result in valid_results:
            row = [
                result['dataset_name'],
                f"{result['rmse']:.6f}",
                f"{result['mean']:.6f}",
                f"{result['median']:.6f}",
                f"{result['std']:.6f}",
                f"{result['min']:.6f}",
                f"{result['max']:.6f}",
                f"{result['sse']:.6f}"
            ]
            table_data.append(row)

        print("\n" + "="*80)
        print("VIO SIMULATION RESULTS SUMMARY")
        print("="*80)
        print(tabulate(table_data, headers=headers, tablefmt="grid"))
        print("="*80 + "\n")

    except ImportError:
        logger.warning("tabulate not installed. Install with: pip install tabulate")
        # Fallback simple table display
        print("\n" + "="*80)
        print("VIO SIMULATION RESULTS SUMMARY")
        print("="*80)
        for result in valid_results:
            print(f"Dataset: {result['dataset_name']}")
            print(f"  RMSE: {result['rmse']:.6f}")
            print(f"  Mean: {result['mean']:.6f}")
            print(f"  Median: {result['median']:.6f}")
            print(f"  Std: {result['std']:.6f}")
            print(f"  Min: {result['min']:.6f}")
            print(f"  Max: {result['max']:.6f}")
            print(f"  SSE: {result['sse']:.6f}")
            print("-" * 40)
        print("="*80 + "\n")


def run_slam_and_rosbag(data_dir, case_name, ros_node_name, roslaunch_name, log_dir):
    logger.info(f"Launching SLAM system: {roslaunch_name}")
    dataset_name = os.path.splitext(os.path.basename(rosbag_file))[0]
    cmd_disbale_rviz = "use_rviz:=false"
    cmd_disable_full_log = "save_full_log:=false"
    cmd_set_bag_path = f"bag_path:={os.path.join(data_dir, case_name + '.bag')}"
    cmd_set_log_path = f"log_path:={log_dir}"
    cmd_set_dataset = f"dataset:={case_name}"
    cmd_use_limit = f"use_rate_limit:={False}"
    command_list = ' '.join(["roslaunch", ros_node_name, roslaunch_name, cmd_disbale_rviz, cmd_disable_full_log, cmd_set_bag_path, cmd_set_log_path, cmd_set_dataset, cmd_use_limit])
    run_command(command_list)
    result = evaluate_results(log_dir, case_name)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch VIO Simulation Script")
    parser.add_argument("--ros_node", type=str, default="vio", help="Name of the ROS node")
    parser.add_argument("--launch_file", type=str, default="euroc_serial_backend.launch", help="ROS launch file to use")
    args = parser.parse_args()

    logger.info(f"Found {len(CASE_LIST)} rosbag files for simulation:")
    for i, rosbag_file in enumerate(CASE_LIST, 1):
        logger.info(f"{i}. {rosbag_file}")

    os.chdir(WS_DIR)
    os.makedirs(LOG_DIR, exist_ok=True)

    # Process each rosbag file
    result_list = list()
    for case_name in CASE_LIST:
        logger.info(f"\n=== Processing rosbag: {case_name} ===")
        result = run_slam_and_rosbag(DATA_DIR, case_name, args.ros_node, args.launch_file, LOG_DIR)
        result_list.append(result)
        logger.info(f"=== Finished processing {case_name} ===\n")
        # Short pause to ensure system is fully cleaned up
        time.sleep(1)

    save_and_show_results(LOG_DIR, result_list)


