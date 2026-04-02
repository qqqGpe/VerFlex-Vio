#!/usr/bin/env python3
'''
Author: pengen.gao gaope.hb@gmail.com
Date: 2025-04-07
Copyright (c) 2025 by gaope.hb@gmail.com, All Rights Reserved.

Batch VIO Evaluation Script using evo API (evo_ape and evo_rpe)
Similar to run_batch_sim.py but uses evo Python API instead of CLI commands.
'''
import os
import sys
import argparse
import logging
import datetime
import csv

# Add third_party to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PACKAGE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
THIRD_PARTY_DIR = os.path.join(PACKAGE_DIR, "third_party")
EVO_DIR = os.path.join(THIRD_PARTY_DIR, "evo")

# Add evo to Python path
if os.path.exists(EVO_DIR):
    sys.path.insert(0, EVO_DIR)

from tabulate import tabulate

# Import evo modules
from evo.tools import file_interface
from evo.core.metrics import PoseRelation, Unit
from evo.main_ape import ape
from evo.main_rpe import rpe
from evo.core import sync

ALL_CASES = ["MH_01_easy", "MH_02_easy", "MH_03_medium", "V1_01_easy", "V1_02_medium", "V2_01_easy", "V2_02_medium"]

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


def load_trajectories(groundtruth_file, estimate_file):
    """
    Load ground truth and estimate trajectories using evo API.
    
    Args:
        groundtruth_file: Path to ground truth trajectory file (TUM format)
        estimate_file: Path to estimated trajectory file (TUM format)
    
    Returns:
        tuple: (traj_ref, traj_est)
    """
    try:
        traj_ref = file_interface.read_tum_trajectory_file(groundtruth_file)
        traj_est = file_interface.read_tum_trajectory_file(estimate_file)
        logger.info(f"Loaded trajectories: ref={traj_ref.num_poses} poses, est={traj_est.num_poses} poses")
        return traj_ref, traj_est
    except Exception as e:
        logger.error(f"Failed to load trajectories: {e}")
        return None, None


def synchronize_trajectories(traj_ref, traj_est, t_max_diff=0.01):
    """
    Synchronize two trajectories by timestamp association.
    
    Args:
        traj_ref: Reference trajectory
        traj_est: Estimate trajectory
        t_max_diff: Maximum time difference for matching (seconds)
    
    Returns:
        tuple: (synced_ref, synced_est)
    """
    try:
        traj_ref_sync, traj_est_sync = sync.associate_trajectories(
            traj_ref, traj_est, t_max_diff, 0.0
        )
        logger.info(f"Synchronized trajectories: ref={traj_ref_sync.num_poses}, est={traj_est_sync.num_poses}")
        return traj_ref_sync, traj_est_sync
    except Exception as e:
        logger.warning(f"Synchronization failed: {e}, using original trajectories")
        return traj_ref, traj_est


def compute_ape(traj_ref, traj_est, align=False, correct_scale=False, pose_relation=PoseRelation.translation_part):
    """
    Compute Absolute Pose Error (APE) using evo API.
    
    Args:
        traj_ref: Reference trajectory
        traj_est: Estimate trajectory
        align: Whether to align trajectories (SE(3))
        correct_scale: Whether to correct scale (Sim(3))
        pose_relation: Which pose component to evaluate
    
    Returns:
        dict: Statistics including rmse, mean, median, std, min, max, sse
    """
    try:
        result = ape(
            traj_ref=traj_ref,
            traj_est=traj_est,
            pose_relation=pose_relation,
            align=align,
            correct_scale=correct_scale,
            n_to_align=-1,
            align_origin=False,
            ref_name="reference",
            est_name="estimate"
        )
        
        stats = {
            'rmse': result.stats.get('rmse', 0.0),
            'mean': result.stats.get('mean', 0.0),
            'median': result.stats.get('median', 0.0),
            'std': result.stats.get('std', 0.0),
            'min': result.stats.get('min', 0.0),
            'max': result.stats.get('max', 0.0),
            'sse': result.stats.get('sse', 0.0)
        }
        return stats
    except Exception as e:
        logger.error(f"APE computation failed: {e}")
        return None


def compute_rpe(traj_ref, traj_est, delta=1.0, delta_unit=Unit.frames, 
                pose_relation=PoseRelation.translation_part, align=False):
    """
    Compute Relative Pose Error (RPE) using evo API.
    
    Args:
        traj_ref: Reference trajectory
        traj_est: Estimate trajectory
        delta: Delta value for RPE computation
        delta_unit: Unit for delta (frames, meters, etc.)
        pose_relation: Which pose component to evaluate
        align: Whether to align trajectories
    
    Returns:
        dict: Statistics including rmse, mean, median, std, min, max, sse
    """
    try:
        result = rpe(
            traj_ref=traj_ref,
            traj_est=traj_est,
            pose_relation=pose_relation,
            delta=delta,
            delta_unit=delta_unit,
            rel_delta_tol=0.1,
            all_pairs=False,
            pairs_from_reference=False,
            align=align,
            correct_scale=False,
            n_to_align=-1,
            align_origin=False,
            ref_name="reference",
            est_name="estimate"
        )
        
        stats = {
            'rmse': result.stats.get('rmse', 0.0),
            'mean': result.stats.get('mean', 0.0),
            'median': result.stats.get('median', 0.0),
            'std': result.stats.get('std', 0.0),
            'min': result.stats.get('min', 0.0),
            'max': result.stats.get('max', 0.0),
            'sse': result.stats.get('sse', 0.0)
        }
        return stats
    except Exception as e:
        logger.error(f"RPE computation failed: {e}")
        return None


def evaluate_case(log_dir, case_name, codebase_dir, align=False, metric='both'):
    """
    Evaluate a single VIO case.
    
    Args:
        log_dir: Directory containing VIO output files
        case_name: Name of the test case
        codebase_dir: Path to the vio_backend package
        align: Whether to align trajectories before evaluation
        metric: 'ape', 'rpe', or 'both'
    
    Returns:
        dict: Evaluation results with APE and RPE metrics
    """
    logger.info(f"Evaluating case: {case_name}")
    
    # Find groundtruth file
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
        logger.warning(f"No groundtruth file found for case: {case_name}")
        return None
    
    logger.info(f"Groundtruth file: {groundtruth_file}")
    
    # Find estimate file (VIO output)
    estimate_file = None
    for file in os.listdir(log_dir):
        if case_name in file and file.endswith('.csv'):
            estimate_file = os.path.join(log_dir, file)
            break
    
    if not estimate_file:
        logger.warning(f"No estimate file found for case: {case_name}")
        return None
    
    logger.info(f"Estimate file: {estimate_file}")
    
    # Load trajectories
    traj_ref, traj_est = load_trajectories(groundtruth_file, estimate_file)
    if traj_ref is None or traj_est is None:
        return None
    
    # Synchronize trajectories
    traj_ref, traj_est = synchronize_trajectories(traj_ref, traj_est)
    
    result = {'dataset_name': case_name}
    
    # Compute APE
    if metric in ('ape', 'both'):
        ape_stats = compute_ape(traj_ref, traj_est, align=align, pose_relation=PoseRelation.translation_part)
        if ape_stats:
            result.update({
                'ape_rmse': ape_stats['rmse'],
                'ape_mean': ape_stats['mean'],
                'ape_median': ape_stats['median'],
                'ape_std': ape_stats['std'],
                'ape_min': ape_stats['min'],
                'ape_max': ape_stats['max'],
                'ape_sse': ape_stats['sse']
            })
    
    # Compute RPE
    if metric in ('rpe', 'both'):
        rpe_stats = compute_rpe(traj_ref, traj_est, delta=1.0, delta_unit=Unit.frames, 
                               pose_relation=PoseRelation.translation_part, align=align)
        if rpe_stats:
            result.update({
                'rpe_rmse': rpe_stats['rmse'],
                'rpe_mean': rpe_stats['mean'],
                'rpe_median': rpe_stats['median'],
                'rpe_std': rpe_stats['std'],
                'rpe_min': rpe_stats['min'],
                'rpe_max': rpe_stats['max'],
                'rpe_sse': rpe_stats['sse']
            })
    
    if 'ape_rmse' in result or 'rpe_rmse' in result:
        logger.info(f"Results for {case_name}: APE RMSE={result.get('ape_rmse', 'N/A')}, RPE RMSE={result.get('rpe_rmse', 'N/A')}")
        return result
    else:
        return None


def save_results(log_dir, result_list, metric_type='both'):
    """
    Save evaluation results to CSV and display summary table.
    
    Args:
        log_dir: Directory to save results
        result_list: List of result dictionaries
        metric_type: 'ape', 'rpe', or 'both'
    """
    valid_results = [r for r in result_list if r is not None]
    
    if not valid_results:
        logger.warning("No valid results to save.")
        return
    
    # Determine which metrics to include
    if metric_type == 'ape':
        fieldnames = ['dataset_name', 'ape_rmse', 'ape_mean', 'ape_median', 'ape_std', 'ape_min', 'ape_max', 'ape_sse']
        headers = ['Dataset', 'APE RMSE', 'APE Mean', 'APE Median', 'APE Std', 'APE Min', 'APE Max', 'APE SSE']
    elif metric_type == 'rpe':
        fieldnames = ['dataset_name', 'rpe_rmse', 'rpe_mean', 'rpe_median', 'rpe_std', 'rpe_min', 'rpe_max', 'rpe_sse']
        headers = ['Dataset', 'RPE RMSE', 'RPE Mean', 'RPE Median', 'RPE Std', 'RPE Min', 'RPE Max', 'RPE SSE']
    else:  # both
        fieldnames = ['dataset_name', 'ape_rmse', 'ape_mean', 'ape_median', 'ape_std', 
                      'rpe_rmse', 'rpe_mean', 'rpe_median', 'rpe_std']
        headers = ['Dataset', 'APE RMSE', 'APE Mean', 'APE Median', 'APE Std', 
                   'RPE RMSE', 'RPE Mean', 'RPE Median', 'RPE Std']
    
    # Save to CSV
    if metric_type == 'ape':
        summary_file = os.path.join(log_dir, "ape_results.csv")
    elif metric_type == 'rpe':
        summary_file = os.path.join(log_dir, "rpe_results.csv")
    else:
        summary_file = os.path.join(log_dir, "evo_results_summary.csv")
    
    try:
        with open(summary_file, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for result in valid_results:
                row = {k: result.get(k, '') for k in fieldnames}
                writer.writerow(row)
        logger.info(f"Results saved to: {summary_file}")
    except Exception as e:
        logger.error(f"Error saving results: {e}")
    
    # Display table
    try:
        table_data = []
        for result in valid_results:
            if metric_type == 'ape':
                row = [
                    result['dataset_name'],
                    f"{result.get('ape_rmse', 'N/A'):.6f}" if result.get('ape_rmse') else 'N/A',
                    f"{result.get('ape_mean', 'N/A'):.6f}" if result.get('ape_mean') else 'N/A',
                    f"{result.get('ape_median', 'N/A'):.6f}" if result.get('ape_median') else 'N/A',
                    f"{result.get('ape_std', 'N/A'):.6f}" if result.get('ape_std') else 'N/A',
                    f"{result.get('ape_min', 'N/A'):.6f}" if result.get('ape_min') else 'N/A',
                    f"{result.get('ape_max', 'N/A'):.6f}" if result.get('ape_max') else 'N/A',
                    f"{result.get('ape_sse', 'N/A'):.6f}" if result.get('ape_sse') else 'N/A'
                ]
            elif metric_type == 'rpe':
                row = [
                    result['dataset_name'],
                    f"{result.get('rpe_rmse', 'N/A'):.6f}" if result.get('rpe_rmse') else 'N/A',
                    f"{result.get('rpe_mean', 'N/A'):.6f}" if result.get('rpe_mean') else 'N/A',
                    f"{result.get('rpe_median', 'N/A'):.6f}" if result.get('rpe_median') else 'N/A',
                    f"{result.get('rpe_std', 'N/A'):.6f}" if result.get('rpe_std') else 'N/A',
                    f"{result.get('rpe_min', 'N/A'):.6f}" if result.get('rpe_min') else 'N/A',
                    f"{result.get('rpe_max', 'N/A'):.6f}" if result.get('rpe_max') else 'N/A',
                    f"{result.get('rpe_sse', 'N/A'):.6f}" if result.get('rpe_sse') else 'N/A'
                ]
            else:
                row = [
                    result['dataset_name'],
                    f"{result.get('ape_rmse', 'N/A'):.6f}" if result.get('ape_rmse') else 'N/A',
                    f"{result.get('ape_mean', 'N/A'):.6f}" if result.get('ape_mean') else 'N/A',
                    f"{result.get('ape_median', 'N/A'):.6f}" if result.get('ape_median') else 'N/A',
                    f"{result.get('ape_std', 'N/A'):.6f}" if result.get('ape_std') else 'N/A',
                    f"{result.get('rpe_rmse', 'N/A'):.6f}" if result.get('rpe_rmse') else 'N/A',
                    f"{result.get('rpe_mean', 'N/A'):.6f}" if result.get('rpe_mean') else 'N/A',
                    f"{result.get('rpe_median', 'N/A'):.6f}" if result.get('rpe_median') else 'N/A',
                    f"{result.get('rpe_std', 'N/A'):.6f}" if result.get('rpe_std') else 'N/A'
                ]
            table_data.append(row)
        
        print("\n" + "=" * 100)
        print("VIO EVALUATION RESULTS (using evo API)")
        print("=" * 100)
        print(tabulate(table_data, headers=headers, tablefmt="grid"))
        print("=" * 100 + "\n")
        
    except ImportError:
        logger.warning("tabulate not installed. Using simple output.")
        print("\n" + "=" * 80)
        print("VIO EVALUATION RESULTS")
        print("=" * 80)
        for result in valid_results:
            print(f"\nDataset: {result['dataset_name']}")
            if 'ape_rmse' in result:
                print(f"  APE - RMSE: {result['ape_rmse']:.6f}, Mean: {result['ape_mean']:.6f}, Median: {result['ape_median']:.6f}")
            if 'rpe_rmse' in result:
                print(f"  RPE - RMSE: {result['rpe_rmse']:.6f}, Mean: {result['rpe_mean']:.6f}, Median: {result['rpe_median']:.6f}")
        print("=" * 80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Batch VIO Evaluation using evo API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  # Run evaluation on existing log directory:
  python3 run_evo_batch.py --log_dir /path/to/vio/logs
  
  # Evaluate specific cases:
  python3 run_evo_batch.py --log_dir /path/to/logs --cases MH_01_easy,MH_02_easy
  
  # Evaluate all EuRoC cases:
  python3 run_evo_batch.py --log_dir /path/to/logs --cases all
  
  # With trajectory alignment:
  python3 run_evo_batch.py --log_dir /path/to/logs --align
  
  # Only compute APE:
  python3 run_evo_batch.py --log_dir /path/to/logs --metric ape
""")
    parser.add_argument("--log_dir", type=str, required=True,
                        help="Directory containing VIO output CSV files")
    parser.add_argument("--codebase_dir", type=str, 
                        default=PACKAGE_DIR,
                        help="Path to vio_backend package (for finding groundtruth)")
    parser.add_argument("--cases", type=str, default="MH_01_easy",
                        help="Comma-separated case names, or 'all' for all cases")
    parser.add_argument("--align", action="store_true",
                        help="Align trajectories before evaluation (SE(3))")
    parser.add_argument("--metric", type=str, default="both", choices=["ape", "rpe", "both"],
                        help="Metric to compute: ape, rpe, or both")
    
    args = parser.parse_args()
    
    # Resolve case list
    if args.cases.lower() == "all":
        case_list = ALL_CASES
    else:
        case_list = [c.strip() for c in args.cases.split(",") if c.strip()]
    
    # Resolve directories
    log_dir = os.path.abspath(args.log_dir)
    if not os.path.isdir(log_dir):
        logger.error(f"Log directory does not exist: {log_dir}")
        sys.exit(1)
    
    codebase_dir = os.path.abspath(args.codebase_dir)
    
    logger.info(f"Log dir: {log_dir}")
    logger.info(f"Codebase dir: {codebase_dir}")
    logger.info(f"Cases to evaluate ({len(case_list)}): {case_list}")
    logger.info(f"Align trajectories: {args.align}")
    logger.info(f"Metrics: {args.metric}")
    
    # Evaluate each case
    result_list = []
    for case_name in case_list:
        logger.info(f"\n=== Evaluating: {case_name} ===")
        result = evaluate_case(log_dir, case_name, codebase_dir, align=args.align, metric=args.metric)
        result_list.append(result)
    
    # Save and display results
    save_results(log_dir, result_list, metric_type=args.metric)
