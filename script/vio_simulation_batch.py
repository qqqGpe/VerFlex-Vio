#!/usr/bin/env python3
import subprocess
import os
import roslaunch
from evo.tools import file_interface
from evo.core import metrics, sync
from evo.main_ape import ape
from evo.tools.settings import SETTINGS

def run_slam_program():
    # 替换为你的SLAM程序的launch文件路径
    slam_launch_file = "/home/gao/ws/catkin_ws/src/vio_backend/launch/euroc_serial_backend.launch"

    # 运行roslaunch命令
    print("启动SLAM程序...")
    slam_process = subprocess.Popen(["roslaunch", slam_launch_file])

    # 等待SLAM程序完成（或者你可以设置其他终止条件）
    try:
        slam_process.wait()
    except KeyboardInterrupt:
        slam_process.terminate()
        print("\nSLAM程序已终止")

    return True

def calculate_ape():
    # 替换为你的真实轨迹和估计轨迹文件路径
    # 通常SLAM程序会输出估计轨迹，你需要提供真实轨迹作为参考
    ref_file = "path_to_ground_truth/ground_truth.txt"  # 真实轨迹
    est_file = "path_to_estimated/estimated_trajectory.txt"  # 估计轨迹

    # 加载轨迹数据
    traj_ref = file_interface.read_tum_trajectory_file(ref_file)
    traj_est = file_interface.read_tum_trajectory_file(est_file)

    # 同步和关联轨迹数据
    traj_ref, traj_est = sync.associate_trajectories(traj_ref, traj_est)

    # 计算APE
    ape_metric = metrics.APE(metrics.POSE_RATION.absolute_translation_part)
    ape_metric.process_data((traj_ref, traj_est))

    # 打印APE统计信息
    print("\nAPE统计结果:")
    print(f"最大误差: {ape_metric.stats['max']} 米")
    print(f"平均误差: {ape_metric.stats['mean']} 米")
    print(f"均方根误差: {ape_metric.stats['rmse']} 米")
    print(f"中位数误差: {ape_metric.stats['median']} 米")
    print(f"标准差: {ape_metric.stats['std']} 米")

    # 可选：保存结果到文件
    result_file = "ape_results.zip"
    ape_metric.save_results(result_file)
    print(f"\nAPE结果已保存到: {result_file}")

if __name__ == "__main__":

    uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
    roslaunch.configure_logging(uuid)
    tracking_launch = roslaunch.parent.ROSLaunchParent(
        uuid, ["/home/gao/ws/catkin_ws/src/vio_backend/launch/euroc_serial_backend.launch"])
    tracking_launch.start()

    try:
        tracking_launch.spin()
    except KeyboardInterrupt:
        tracking_launch.shutdown()