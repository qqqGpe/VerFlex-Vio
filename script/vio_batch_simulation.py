#!/usr/bin/env python3
import os
import subprocess
import datetime
import time
import psutil
import rospy
import rosbag


def find_euroc_rosbags(directory):
    rosbags = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(".bag"):
                rosbags.append(os.path.join(root, file))
    return sorted(rosbags)


def is_process_running(process):
    try:
        return process.is_running() and process.status() != psutil.STATUS_ZOMBIE
    except (psutil.NoSuchProcess, psutil.ZombieProcess, psutil.AccessDenied):
        return False


def run_slam_and_rosbag(rosbag_file, ros_node_name, roslaunch_name, log_dir):

    print(f"Launching SLAM system: {roslaunch_name}")
    dataset_name = os.path.splittext(os.path.basename(rosbag_file))[0]
    cmd_disbale_rviz = "use_rviz:=false"
    cmd_disable_full_log = "save_full_log:=false"
    cmd_set_bag_path = "bag_path:=" + rosbag_file
    cmd_set_log_path = "log_path:={}".format(log_dir)
    cmd_set_dataset = "dataset:=" + dataset_name
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
        ]
    )
    process = psutil.Process(vio_process.pid)
    time.sleep(5)

    while is_process_running(process):
        try:
            if process.status() == psutil.STATUS_RUNNING:
                time.sleep(1)
        except psutil.NoSuchProcess:
            print("SLAM system has terminated.")
            break

    # # 终止SLAM系统
    print("Finish batch simulation, terminating SLAM system...")
    vio_process.terminate()
    try:
        vio_process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        vio_process.kill()


if __name__ == "__main__":

    ros_node_name = "vio"
    roslaunch_name = "euroc_serial_backend.launch"

    dataset_dir = os.path.join(os.path.expanduser("~"), "dataset/euroc_mav")  # 替换为实际的Euroc数据集路径
    vio_dir = os.path.join(os.path.expanduser("~"), "ws/catkin_ws")
    log_dir = os.path.join(vio_dir, "src/vio_backend/log/vio_sim_" + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M"))

    rosbags = find_euroc_rosbags(dataset_dir)
    if not rosbags:
        print("No rosbag files found in the specified directory.")
        exit(1)

    print("Found {} rosbag files for simulation:".format(len(rosbags)))
    for i, rosbag_file in enumerate(rosbags, 1):
        print("{}. {}".format(i, rosbag_file))

    os.makedirs(log_dir, exist_ok=True)

    source_cmd = "source " + os.path.join(vio_dir, "devel/setup.zsh")
    subprocess.run(source_cmd, shell=True, executable="/bin/zsh")

    # # 依次处理每个rosbag
    for rosbag_file in rosbags:
        print(f"\n=== Processing rosbag: {rosbag_file} ===")
        run_slam_and_rosbag(rosbag_file, ros_node_name, roslaunch_name, log_dir)
        print(f"=== Finished processing {rosbag_file} ===\n")

        #     # 短暂暂停，确保系统完全清理
        time.sleep(2)
