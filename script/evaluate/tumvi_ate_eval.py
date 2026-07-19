#!/usr/bin/env python3
"""ATE evaluation: Umeyama SE(3) align estimated trajectory to GT, report RMSE.
Usage: ate_eval.py <est_tum.csv> <gt_tum.txt>
Both files: TUM format (ts tx ty tz qx qy qz qw), '#' comment lines skipped.
"""
import sys
import numpy as np


def load_traj(path):
    ts, xyz = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            p = line.split()
            ts.append(float(p[0]))
            xyz.append([float(p[1]), float(p[2]), float(p[3])])
    return np.array(ts), np.array(xyz)


def interp_gt(gt_ts, gt_xyz, query_ts):
    out = np.empty((len(query_ts), 3))
    for i, t in enumerate(query_ts):
        idx = np.searchsorted(gt_ts, t)
        if idx <= 0:
            out[i] = gt_xyz[0]
        elif idx >= len(gt_ts):
            out[i] = gt_xyz[-1]
        else:
            t0, t1 = gt_ts[idx - 1], gt_ts[idx]
            a = (t - t0) / (t1 - t0)
            out[i] = (1 - a) * gt_xyz[idx - 1] + a * gt_xyz[idx]
    return out


def umeyama_align(model, data):
    """Rigid (no scale) SE(3) align model -> data. Returns aligned model."""
    mu_m = model.mean(0)
    mu_d = data.mean(0)
    mc = model - mu_m
    dc = data - mu_d
    W = dc.T @ mc / model.shape[0]
    U, _, Vt = np.linalg.svd(W)
    diag = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        diag[2, 2] = -1
    R = U @ diag @ Vt
    return (R @ mc.T).T + mu_d


def main():
    est_ts, est_xyz = load_traj(sys.argv[1])
    gt_ts, gt_xyz = load_traj(sys.argv[2])
    gt_interp = interp_gt(gt_ts, gt_xyz, est_ts)
    aligned = umeyama_align(est_xyz, gt_interp)
    err = np.linalg.norm(aligned - gt_interp, axis=1)
    rmse = np.sqrt((err ** 2).mean())
    # Trajectory length (for context)
    seg = np.diff(est_xyz, axis=0)
    path_len = np.sum(np.linalg.norm(seg, axis=1))
    print(f"poses: {len(est_ts)}   path length: {path_len:.1f} m")
    print(f"ATE RMSE: {rmse:.4f} m")
    print(f"  mean: {err.mean():.4f}   median: {np.median(err):.4f}   max: {err.max():.4f}")


if __name__ == "__main__":
    main()
