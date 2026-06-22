#!/home/gao/miniconda3/bin/python3
'''
Author: pengen.gao gaope.hb@gmail.com
Batch simulation + evo evaluation for the standalone vio_offline binary (no ROS).

Usage examples:
  # Run all EuRoC cases with default config:
  python3 run_offline_batch.py --binary build/vio_offline \
      --config config/euroc_stereo.yaml \
      --dataset_root ~/dataset/euroc_mav

  # Specific cases, with alignment, 4 parallel jobs:
  python3 run_offline_batch.py --binary build/vio_offline \
      --config config/euroc_stereo.yaml \
      --dataset_root ~/dataset/euroc_mav \
      --cases MH_01_easy,MH_02_easy \
      --align --jobs 4
'''

import os
import sys
import argparse
import logging
import subprocess
import tempfile
import glob
import copy
import csv
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = SCRIPT_DIR.parent

from tabulate import tabulate
from evo.tools import file_interface
from evo.core.metrics import PoseRelation, Unit
from evo.main_ape import ape
from evo.main_rpe import rpe
from evo.core import sync

ALL_CASES = ["MH_01_easy", "MH_02_easy", "MH_03_medium", "MH_04_difficult", "MH_05_difficult", "V1_01_easy", "V1_02_medium", "V1_03_difficult", "V2_01_easy", "V2_02_medium", "V2_03_difficult"]

GT_DIR = PACKAGE_DIR / "data" / "euroc" / "ground_truth"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def load_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def write_temp_config(base_cfg: dict, case_name: str, dataset_root: str, log_dir: str) -> str:
    """Write a per-case temporary YAML config and return its path."""
    cfg = copy.deepcopy(base_cfg)
    cfg["dataset_dir"] = os.path.join(dataset_root, case_name)
    cfg["log_path"] = log_dir
    cfg["bag_name"] = case_name
    # Batch runs are headless — never pop the Pangolin window, even if the
    # base config has it enabled (also disabled when running in parallel jobs).
    cfg["enable_pangolin_viewer"] = False

    fd, tmp_path = tempfile.mkstemp(suffix=".yaml", prefix=f"vio_{case_name}_")
    os.close(fd)
    with open(tmp_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)
    return tmp_path


# ---------------------------------------------------------------------------
# Running the binary
# ---------------------------------------------------------------------------

def run_vio(binary: str, config_path: str, log_dir: str, timeout: int = 600) -> bool:
    """Run vio_offline, tee stdout/stderr to <log_dir>/vio_run.log, return True on success."""
    cmd = [binary, "--config", config_path]
    log_file = os.path.join(log_dir, "vio_run.log")
    logger.info("Running: %s  (log -> %s)", " ".join(cmd), log_file)
    try:
        with open(log_file, "w") as lf:
            result = subprocess.run(cmd, timeout=timeout, stdout=lf, stderr=lf)
        if result.returncode != 0:
            logger.error("vio_offline exited with code %d — see %s", result.returncode, log_file)
            return False
        return True
    except subprocess.TimeoutExpired:
        logger.error("vio_offline timed out after %ds — see %s", timeout, log_file)
        return False
    except FileNotFoundError:
        logger.error("Binary not found: %s", binary)
        return False


# ---------------------------------------------------------------------------
# Finding output files
# ---------------------------------------------------------------------------

def find_tum_output(log_dir: str, case_name: str) -> str | None:
    """Return the most recently modified *_tum.csv for this case."""
    pattern = os.path.join(log_dir, f"{case_name}*_tum.csv")
    matches = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
    return matches[0] if matches else None


def find_groundtruth(case_name: str) -> str | None:
    """Return TUM-format ground truth file for the given case."""
    case_dir = GT_DIR / case_name
    if not case_dir.exists():
        return None
    for f in case_dir.iterdir():
        if f.suffix == ".csv":
            return str(f)
    return None


# ---------------------------------------------------------------------------
# evo evaluation (reused from run_evo_batch.py)
# ---------------------------------------------------------------------------

def _compute_ape(traj_ref, traj_est, align: bool):
    result = ape(
        traj_ref=traj_ref, traj_est=traj_est,
        pose_relation=PoseRelation.translation_part,
        align=align, correct_scale=False,
        n_to_align=-1, align_origin=False,
        ref_name="reference", est_name="estimate",
    )
    return result.stats


def _compute_rpe(traj_ref, traj_est, align: bool):
    result = rpe(
        traj_ref=traj_ref, traj_est=traj_est,
        pose_relation=PoseRelation.translation_part,
        delta=1.0, delta_unit=Unit.frames,
        rel_delta_tol=0.1, all_pairs=False,
        pairs_from_reference=False, align=align,
        correct_scale=False, n_to_align=-1, align_origin=False,
        ref_name="reference", est_name="estimate",
    )
    return result.stats


def evaluate(gt_file: str, est_file: str, align: bool, metric: str) -> dict | None:
    try:
        traj_ref = file_interface.read_tum_trajectory_file(gt_file)
        traj_est = file_interface.read_tum_trajectory_file(est_file)
    except Exception as e:
        logger.error("Failed to load trajectories: %s", e)
        return None

    try:
        traj_ref, traj_est = sync.associate_trajectories(traj_ref, traj_est, 0.01, 0.0)
    except Exception as e:
        logger.warning("Trajectory sync failed (%s), continuing without sync", e)

    out = {}
    if metric in ("ape", "both"):
        try:
            s = _compute_ape(traj_ref, traj_est, align)
            out.update({f"ape_{k}": v for k, v in s.items()})
        except Exception as e:
            logger.error("APE failed: %s", e)

    if metric in ("rpe", "both"):
        try:
            s = _compute_rpe(traj_ref, traj_est, align)
            out.update({f"rpe_{k}": v for k, v in s.items()})
        except Exception as e:
            logger.error("RPE failed: %s", e)

    return out if out else None


# ---------------------------------------------------------------------------
# Per-case pipeline
# ---------------------------------------------------------------------------

def run_case(case_name: str, binary: str, base_cfg: dict, dataset_root: str,
             log_root: str, align: bool, metric: str, timeout: int) -> dict:
    """Always returns a dict with at least 'dataset_name' and 'status'."""
    result = {"dataset_name": case_name, "status": "unknown"}
    log_dir = os.path.join(log_root, case_name)
    os.makedirs(log_dir, exist_ok=True)

    dataset_path = os.path.join(dataset_root, case_name)
    if not os.path.isdir(dataset_path):
        logger.warning("[%s] Dataset not found: %s", case_name, dataset_path)
        result["status"] = "no_dataset"
        return result

    gt_file = find_groundtruth(case_name)
    if gt_file is None:
        logger.warning("[%s] No ground truth file found.", case_name)
        result["status"] = "no_groundtruth"
        return result

    tmp_cfg = write_temp_config(base_cfg, case_name, dataset_root, log_dir)
    try:
        ok = run_vio(binary, tmp_cfg, log_dir, timeout=timeout)
    finally:
        os.unlink(tmp_cfg)

    if not ok:
        result["status"] = "vio_failed"
        return result

    est_file = find_tum_output(log_dir, case_name)
    if est_file is None:
        logger.error("[%s] No TUM output file found in %s.", case_name, log_dir)
        result["status"] = "no_output"
        return result

    logger.info("[%s] Evaluating: est=%s", case_name, est_file)
    stats = evaluate(gt_file, est_file, align, metric)
    if stats is None:
        result["status"] = "eval_failed"
        return result

    result["status"] = "ok"
    result.update(stats)
    return result


# ---------------------------------------------------------------------------
# Results display and save
# ---------------------------------------------------------------------------

def save_and_print(results: list[dict], log_root: str, metric: str):
    if not results:
        logger.warning("No results.")
        return

    # Determine metric columns from successful runs
    sample_ok = next((r for r in results if r.get("status") == "ok"), None)
    ape_cols = [k for k in (sample_ok or {}) if k.startswith("ape_")] if sample_ok else []
    rpe_cols = [k for k in (sample_ok or {}) if k.startswith("rpe_")] if sample_ok else []
    metric_cols = (ape_cols if metric in ("ape", "both") else []) + \
                  (rpe_cols if metric in ("rpe", "both") else [])
    show_cols = ["dataset_name", "status"] + metric_cols

    headers = [c.replace("_", " ").upper() for c in show_cols]
    rows = []
    for r in results:
        row = []
        for c in show_cols:
            v = r.get(c, "")
            row.append(f"{v:.4f}" if isinstance(v, float) else str(v))
        rows.append(row)

    print("\n" + "=" * 120)
    print("  VIO OFFLINE BATCH RESULTS")
    print("=" * 120)
    print(tabulate(rows, headers=headers, tablefmt="grid"))

    failed = [r for r in results if r.get("status") != "ok"]
    if failed:
        print("\nFailed / skipped cases:")
        for r in failed:
            log_hint = os.path.join(log_root, r["dataset_name"], "vio_run.log")
            print(f"  [{r['status']:>15}]  {r['dataset_name']:<25}  log: {log_hint}")
    print("=" * 120 + "\n")

    # CSV summary
    summary_path = os.path.join(log_root, f"batch_results_{metric}.csv")
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=show_cols, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    logger.info("Summary saved to %s", summary_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Batch offline VIO simulation + evo evaluation (no ROS)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--binary", required=True,
                        help="Path to vio_offline executable")
    parser.add_argument("--config", required=True,
                        help="Base YAML config file (dataset_dir/log_path/bag_name will be overridden)")
    parser.add_argument("--dataset_root", required=True,
                        help="Root directory containing EuRoC sequence folders")
    parser.add_argument("--cases", default="MH_01_easy",
                        help="Comma-separated case names, or 'all'")
    parser.add_argument("--log_root", default=None,
                        help="Root directory for per-case logs (default: <config_dir>/batch_logs)")
    parser.add_argument("--align", action="store_true",
                        help="SE(3) align trajectories before evaluation")
    parser.add_argument("--metric", default="both", choices=["ape", "rpe", "both"],
                        help="Metrics to compute")
    parser.add_argument("--timeout", type=int, default=600,
                        help="Per-case timeout in seconds (default: 600)")
    parser.add_argument("--jobs", type=int, default=1,
                        help="Number of parallel jobs (default: 1)")
    args = parser.parse_args()

    # Resolve paths
    binary = os.path.abspath(args.binary)
    if not os.path.isfile(binary):
        logger.error("Binary not found: %s", binary)
        sys.exit(1)

    config_path = os.path.abspath(args.config)
    if not os.path.isfile(config_path):
        logger.error("Config not found: %s", config_path)
        sys.exit(1)

    dataset_root = os.path.expanduser(args.dataset_root)
    log_root = args.log_root or os.path.join(os.path.dirname(config_path), "batch_logs")
    os.makedirs(log_root, exist_ok=True)

    case_list = ALL_CASES if args.cases.lower() == "all" else \
        [c.strip() for c in args.cases.split(",") if c.strip()]

    base_cfg = load_yaml(config_path)

    logger.info("Binary    : %s", binary)
    logger.info("Config    : %s", config_path)
    logger.info("Datasets  : %s", dataset_root)
    logger.info("Log root  : %s", log_root)
    logger.info("Cases (%d): %s", len(case_list), case_list)
    logger.info("Align     : %s  |  Metric: %s  |  Jobs: %d", args.align, args.metric, args.jobs)

    results_map = {}
    with ThreadPoolExecutor(max_workers=args.jobs) as executor:
        futures = {
            executor.submit(run_case, case, binary, base_cfg, dataset_root,
                            log_root, args.align, args.metric, args.timeout): case
            for case in case_list
        }
        completed = 0
        for f in as_completed(futures):
            case = futures[f]
            completed += 1
            r = f.result()
            results_map[case] = r
            logger.info("[%d/%d] [%s] status: %s", completed, len(case_list), case, r.get("status"))

    results = [results_map[case] for case in case_list]

    save_and_print(results, log_root, args.metric)


if __name__ == "__main__":
    main()
