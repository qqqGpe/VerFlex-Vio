#!/usr/bin/env python3
import os
import subprocess
import datetime
import time
import psutil
import logging
import argparse

# Substitute with your dataset path
DATA_DIR = os.path.join(os.path.expanduser("~"), "dataset/euroc_mav")
VIO_DIR = os.path.join(os.path.expanduser("~"), "ws/catkin_ws")
LOG_DIR = os.path.join(VIO_DIR, "src/vio_backend/log/vio_sim_" + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M"))

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

def run_slam_and_rosbag(data_dir, case_name, ros_node_name, roslaunch_name, log_dir):
    logger.info(f"Launching SLAM system: {roslaunch_name}")
    dataset_name = os.path.splitext(os.path.basename(rosbag_file))[0]
    cmd_disbale_rviz = "use_rviz:=false"
    cmd_disable_full_log = "save_full_log:=false"
    cmd_set_bag_path = "bag_path:={}".format(os.path.join(data_dir, case_name + ".bag"))
    cmd_set_log_path = "log_path:={}".format(log_dir)
    cmd_set_dataset = "dataset:={}".format(case_name)
    cmd_use_limit = "use_rate_limit:={}".format("false")
    vio_process = subprocess.Popen(
        [
            "roslaunch",
            ros_node_name,
            roslaunch_name,
            cmd_disbale_rviz,
            cmd_disable_full_log,
            cmd_set_bag_path,
            cmd_set_dataset,
            cmd_set_log_path,
            cmd_use_limit
        ]
    )
    process = psutil.Process(vio_process.pid)
    time.sleep(5)

    while is_process_running(process):
        try:
            if process.status() == psutil.STATUS_RUNNING:
                time.sleep(1)
        except psutil.NoSuchProcess:
            logger.info("SLAM system has terminated.")
            break

    # Terminate SLAM process if still running
    logger.info("Finish batch simulation, terminating SLAM system...")
    vio_process.terminate()
    try:
        vio_process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        vio_process.kill()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Batch VIO Simulation Script")
    parser.add_argument("--ros_node", type=str, default="vio", help="Name of the ROS node")
    parser.add_argument("--launch_file", type=str, default="euroc_serial_backend.launch", help="ROS launch file to use")
    args = parser.parse_args()

    # Define the list of cases to process
    CASE_LIST = ["MH_01_easy", "MH_02_easy"]

    logger.info(f"Found {len(CASE_LIST)} rosbag files for simulation:")
    for i, rosbag_file in enumerate(CASE_LIST, 1):
        logger.info(f"{i}. {rosbag_file}")

    os.makedirs(LOG_DIR, exist_ok=True)
    source_cmd = "source " + os.path.join(VIO_DIR, "devel/setup.zsh")
    subprocess.run(source_cmd, shell=True, executable="/bin/zsh")

    # Process each rosbag file
    for case_name in CASE_LIST:
        logger.info(f"\n=== Processing rosbag: {case_name} ===")
        run_slam_and_rosbag(DATA_DIR, case_name, args.ros_node, args.launch_file, LOG_DIR)
        logger.info(f"=== Finished processing {case_name} ===\n")
        # Short pause to ensure system is fully cleaned up
        time.sleep(2)
