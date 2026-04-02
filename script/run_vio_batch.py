#!/usr/bin/env python3
'''
Author: pengen.gao gaope.hb@gmail.com
Date: 2025-04-07
Copyright (c) 2025 by gaope.hb@gmail.com, All Rights Reserved.

Batch VIO Processing Script with evo API evaluation
'''
import os
import sys
import subprocess
import datetime
import time
import psutil
import logging
import argparse
import csv

# Auto-detect workspace from script location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PACKAGE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
DEFAULT_WS_DIR = os.path.abspath(os.path.join(PACKAGE_DIR, "..", ".."))
DEFAULT_PKG_NAME = os.path.basename(PACKAGE_DIR)

# Add third_party to path for evo
THIRD_PARTY_DIR = os.path.join(PACKAGE_DIR, "third_party")
EVO_DIR = os.path.join(THIRD_PARTY_DIR, "evo")

ALL_CASES = ["MH_01_easy", "MH_02_easy", "MH_03_medium", "V1_01_easy", "V1_02_medium", "V2_01_easy", "V2_02_medium"]

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


def is_process_running(process):
    try:
        return process.is_running() and process.status() != psutil.STATUS_ZOMBIE
    except (psutil.NoSuchProcess, psutil.ZombieProcess, psutil.AccessDenied):
        return False


def run_command(command, shell=True, executable="/bin/zsh"):
    process = subprocess.Popen(command, shell=shell, executable=executable)
    process_state = psutil.Process(process.pid)
    time.sleep(5)

    while is_process_running(process_state):
        try:
            if process_state.status() == psutil.STATUS_RUNNING:
                time.sleep(1)
        except psutil.NoSuchProcess:
            logger.info("VIO process terminated.")
            break

    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()


def run_vio_case(data_dir, case_name, ros_node_name, roslaunch_name, log_dir):
    logger.info(f"Launching VIO for case: {case_name}")

    cmd_disable_rviz = "use_rviz:=false"
    cmd_disable_full_log = "save_full_log:=false"
    cmd_set_bag_path = f"bag_path:={os.path.join(data_dir, case_name + '.bag')}"
    cmd_set_log_path = f"log_path:={log_dir}"
    cmd_set_dataset = f"dataset:={case_name}"
    cmd_use_limit = "use_rate_limit:=false"

    command = ' '.join([
        "roslaunch", ros_node_name, roslaunch_name,
        cmd_disable_rviz, cmd_disable_full_log,
        cmd_set_bag_path, cmd_set_log_path,
        cmd_set_dataset, cmd_use_limit
    ])

    logger.info(f"Running: {command}")
    run_command(command)

    output_csv = None
    for file in os.listdir(log_dir):
        if case_name in file and file.endswith('.csv'):
            output_csv = os.path.join(log_dir, file)
            break

    if output_csv:
        logger.info(f"Output file: {output_csv}")
    else:
        logger.warning(f"No output CSV found for case: {case_name}")

    return output_csv


def evaluate_single_case(log_dir, case_name, codebase_dir, align=False):
    # Import evo from third_party
    sys.path.insert(0, EVO_DIR)

    from evo.tools import file_interface
    from evo.core.metrics import PoseRelation
    from evo.main_ape import ape
    from evo.core import sync

    logger.info(f"Evaluating case: {case_name}")

    ground_truth_data_dir = os.path.join(codebase_dir, "data/euroc/ground_truth")
    groundtruth_file = None

    if os.path.exists(ground_truth_data_dir):
        for item in os.listdir(ground_truth_data_dir):
            item_path = os.path.join(ground_truth_data_dir, item)
            if os.path.isdir(item_path) and case_name in item:
                for file in os.listdir(item_path):
                    if "groundtruth" in file.lower() or "gt" in file.lower():
                        groundtruth_file = os.path.join(item_path, file)
                        break
                if groundtruth_file:
                    break

    if not groundtruth_file:
        logger.warning(f"No groundtruth file for case: {case_name}")
        return None

    estimate_file = None
    for file in os.listdir(log_dir):
        if case_name in file and file.endswith('.csv'):
            estimate_file = os.path.join(log_dir, file)
            break

    if not estimate_file:
        logger.warning(f"No estimate file for case: {case_name}")
        return None

    try:
        traj_ref = file_interface.read_tum_trajectory_file(groundtruth_file)
        traj_est = file_interface.read_tum_trajectory_file(estimate_file)
        logger.info(f"Loaded: ref={traj_ref.num_poses}, est={traj_est.num_poses}")

        traj_ref, traj_est = sync.associate_trajectories(traj_ref, traj_est, 0.01, 0.0)

        result = ape(
            traj_ref=traj_ref,
            traj_est=traj_est,
            pose_relation=PoseRelation.translation_part,
            align=align,
            correct_scale=False,
            ref_name="reference",
            est_name="estimate"
        )

        stats = {
            'dataset_name': case_name,
            'ape_rmse': result.stats.get('rmse', 0.0),
            'ape_mean': result.stats.get('mean', 0.0),
            'ape_median': result.stats.get('median', 0.0),
            'ape_std': result.stats.get('std', 0.0),
            'ape_min': result.stats.get('min', 0.0),
            'ape_max': result.stats.get('max', 0.0),
            'ape_sse': result.stats.get('sse', 0.0)
        }

        logger.info(f"APE RMSE: {stats['ape_rmse']:.6f}")
        return stats

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        return None


def save_summary_results(log_dir, result_list):
    try:
        from tabulate import tabulate
        has_tabulate = True
    except ImportError:
        has_tabulate = False

    valid_results = [r for r in result_list if r is not None]

    if not valid_results:
        logger.warning("No valid results.")
        return

    fieldnames = ['dataset_name', 'ape_rmse', 'ape_mean', 'ape_median', 'ape_std', 'ape_min', 'ape_max', 'ape_sse']
    summary_file = os.path.join(log_dir, "vio_batch_results.csv")

    try:
        with open(summary_file, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for result in valid_results:
                writer.writerow(result)
        logger.info(f"Results saved to: {summary_file}")
    except Exception as e:
        logger.error(f"Error saving: {e}")

    if has_tabulate:
        headers = ['Dataset', 'APE RMSE', 'APE Mean', 'APE Median', 'APE Std', 'APE Min', 'APE Max', 'APE SSE']
        table_data = []
        for result in valid_results:
            table_data.append([
                result['dataset_name'],
                f"{result['ape_rmse']:.6f}",
                f"{result['ape_mean']:.6f}",
                f"{result['ape_median']:.6f}",
                f"{result['ape_std']:.6f}",
                f"{result['ape_min']:.6f}",
                f"{result['ape_max']:.6f}",
                f"{result['ape_sse']:.6f}"
            ])

        print("\n" + "=" * 100)
        print("VIO BATCH PROCESSING + EVALUATION RESULTS")
        print("=" * 100)
        print(tabulate(table_data, headers=headers, tablefmt="grid"))
        print("=" * 100 + "\n")
    else:
        print("\n" + "=" * 80)
        print("VIO BATCH RESULTS")
        print("=" * 80)
        for result in valid_results:
            print(f"{result['dataset_name']}: RMSE={result['ape_rmse']:.6f}")
        print("=" * 80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch VIO with evo API")
    parser.add_argument("--ws_dir", type=str, default="", help="Catkin workspace")
    parser.add_argument("--dataset_dir", type=str, required=True, help="EuRoC dataset dir")
    parser.add_argument("--cases", type=str, default="MH_01_easy", help="Case names or 'all'")
    parser.add_argument("--ros_node", type=str, default="vio", help="ROS node name")
    parser.add_argument("--launch_file", type=str, default="euroc_serial_backend.launch")
    parser.add_argument("--align", action="store_true", help="Align trajectories")
    parser.add_argument("--no_eval", action="store_true", help="Skip evaluation")

    args = parser.parse_args()

    # Resolve workspace
    if args.ws_dir:
        ws_dir = os.path.abspath(os.path.expanduser(args.ws_dir))
    else:
        cwd = os.getcwd()
        if os.path.isfile(os.path.join(cwd, "src", "CMakeLists.txt")):
            ws_dir = cwd
        else:
            ws_dir = DEFAULT_WS_DIR
    logger.info(f"Workspace: {ws_dir}")

    codebase_dir = os.path.join(ws_dir, "src", DEFAULT_PKG_NAME)
    if not os.path.isdir(codebase_dir):
        codebase_dir = PACKAGE_DIR
    logger.info(f"Package dir: {codebase_dir}")

    data_dir = os.path.abspath(os.path.expanduser(args.dataset_dir))
    if not os.path.isdir(data_dir):
        logger.error(f"Dataset dir not exist: {data_dir}")
        sys.exit(1)
    logger.info(f"Dataset dir: {data_dir}")

    if args.cases.lower() == "all":
        case_list = ALL_CASES
    else:
        case_list = [c.strip() for c in args.cases.split(",") if c.strip()]

    log_dir = os.path.join(codebase_dir, "log", "vio_batch_" + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M"))

    logger.info(f"Cases: {case_list}")
    logger.info(f"Log dir: {log_dir}")

    os.makedirs(log_dir, exist_ok=True)
    os.chdir(ws_dir)

    result_list = []
    for case_name in case_list:
        logger.info(f"\n=== Processing: {case_name} ===")

        output_file = run_vio_case(data_dir, case_name, args.ros_node, args.launch_file, log_dir)

        if not args.no_eval:
            result = evaluate_single_case(log_dir, case_name, codebase_dir, align=args.align)
            result_list.append(result)
            if result:
                logger.info(f"APE RMSE = {result['ape_rmse']:.6f}")
        else:
            result_list.append(None)

        time.sleep(2)

    if not args.no_eval:
        save_summary_results(log_dir, result_list)
    else:
        logger.info(f"\nVIO done. Results: {log_dir}")

    logger.info(f"\nComplete! Results: {log_dir}")
