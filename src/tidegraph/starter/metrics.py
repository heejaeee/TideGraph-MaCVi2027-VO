#!/usr/bin/env python3
"""Position-only trajectory metrics for the Maritime Visual Odometry challenge.

The prediction is interpolated onto reference times within their common window,
then aligned once using a global similarity transform (rotation, translation and
scale). Drift compares relative displacements over eligible 100–800 m reference
path segments; no additional per-segment alignment is fitted. Absolute trajectory
error is reported separately.

This measures globally scale-aligned position/shape accuracy. It is inspired by
KITTI but is not the full orientation-dependent KITTI pose metric, and scores are
not directly interchangeable. Orientation and recovered metric scale are not
scored. Always report scored/unscored counts and temporal coverage with results.
"""
from __future__ import annotations
import numpy as np

__all__ = ["SEGMENT_LENGTHS_M", "umeyama", "align_sim3", "cumulative_length",
           "drift_percent", "ate_rmse", "resample_to", "score_trajectory"]

# Segment lengths follow KITTI; the position-only metric and alignment differ.
SEGMENT_LENGTHS_M = (100, 200, 300, 400, 500, 600, 700, 800)


def umeyama(src: np.ndarray, dst: np.ndarray, with_scale: bool):
    """Least-squares transform taking `src` onto `dst` (Umeyama 1991).

    Args:
        src, dst: (N, D) arrays of corresponding points.
        with_scale: fit a single global scale as well as rotation and translation.

    Returns:
        (scale, R, t) such that ``dst ~= scale * (R @ src.T).T + t``, or None if the fit is
        degenerate (co-located points, non-finite input, SVD failure).
    """
    if src.shape != dst.shape or src.shape[0] < 2:
        return None
    if not (np.all(np.isfinite(src)) and np.all(np.isfinite(dst))):
        return None
    n, dim = src.shape
    mu_s, mu_d = src.mean(axis=0), dst.mean(axis=0)
    sc, dc = src - mu_s, dst - mu_d
    cov = (dc.T @ sc) / n
    if not np.all(np.isfinite(cov)):
        return None
    try:
        U, D, Vt = np.linalg.svd(cov)
    except np.linalg.LinAlgError:
        return None
    # Reflection guard: without it a mirrored trajectory can score as a good fit.
    S = np.eye(dim)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[dim - 1, dim - 1] = -1.0
    R = U @ S @ Vt
    if with_scale:
        var_s = (sc ** 2).sum() / n
        if var_s <= 0:
            return None
        scale = float((D * np.diag(S)).sum() / var_s)
    else:
        scale = 1.0
    t = mu_d - scale * (R @ mu_s)
    return scale, R, t


def align_sim3(estimate: np.ndarray, reference: np.ndarray):
    """Global Sim(3) alignment of a monocular estimate onto the reference.

    Returns (aligned_estimate, scale) or (None, None) if the fit is degenerate.
    """
    fit = umeyama(estimate, reference, with_scale=True)
    if fit is None:
        return None, None
    s, R, t = fit
    return (s * (R @ estimate.T).T + t), s


def cumulative_length(points: np.ndarray) -> np.ndarray:
    """Arc length along a polyline, starting at 0. Shape (N,) for an (N, D) input."""
    steps = np.linalg.norm(np.diff(points, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(steps)])


def drift_percent(reference: np.ndarray, estimate: np.ndarray,
                  segment_lengths=SEGMENT_LENGTHS_M, start_stride: int = 10,
                  min_points: int = 3, return_segments: bool = False):
    """Relative-motion translational drift, as a percentage (KITTI-style, position-only).

    For each sub-sequence length L and each start i, take the end index j where the reference
    arc length from i first reaches L, and compare the RELATIVE DISPLACEMENT over that segment
    in the two trajectories:

        error = || (estimate[j] - estimate[i]) - (reference[j] - reference[i]) ||  /  L

    averaged over all (L, i), times 100.

    This is the relative-motion form the KITTI dev kit uses (compare relative poses between
    segment endpoints), NOT a per-segment fit. An earlier version fitted a fresh rigid
    transform over each whole segment and measured its residual; that extra fit ABSORBS drift
    (on a 1 km trajectory with different local scale in each half it under-reports ~2x), so it
    is removed here.

    Scope note — this is POSITION-ONLY. The full KITTI metric forms relative poses
    `inv(T_i) * T_j` using each trajectory's orientation at i; our reference provides position
    and heading, not full 3-D orientation, so we compare relative displacements in the single
    globally-aligned frame (equivalent to KITTI's translational term with identity local
    rotation). The estimate must already be globally Sim(3)-aligned to the reference. A full
    6-DoF RPE would require saved per-frame orientations for both trajectories.
    """
    if len(reference) != len(estimate) or len(reference) < 2:
        return (float('nan'), []) if return_segments else float('nan')
    L = cumulative_length(reference)
    total = L[-1]
    errors, records = [], []
    for target in segment_lengths:
        if target > total:
            break
        for i in range(0, len(reference) - 1, start_stride):
            end_length = L[i] + target
            if end_length > total:
                break
            j = int(np.searchsorted(L, end_length))
            if j >= len(reference) or (j - i) < min_points:
                break
            rel_ref = reference[j] - reference[i]
            rel_est = estimate[j] - estimate[i]
            value = float(np.linalg.norm(rel_est - rel_ref) / target * 100.0)
            if np.isfinite(value):
                errors.append(value)
                if return_segments:
                    records.append(dict(segment_m=target, start=i, end=j, drift_pct=value))
    mean = float(np.mean(errors)) if errors else float('nan')
    return (mean, records) if return_segments else mean


def ate_rmse(reference: np.ndarray, aligned_estimate: np.ndarray) -> float:
    """Root-mean-square absolute trajectory error, in metres, after global alignment.

    Reported alongside drift % for context only. It is not the ranking metric.
    """
    if len(reference) != len(aligned_estimate) or len(reference) == 0:
        return float('nan')
    err = np.linalg.norm(aligned_estimate - reference, axis=1)
    return float(np.sqrt((err ** 2).mean()))


def resample_to(sample_times: np.ndarray, sample_values: np.ndarray,
                query_times: np.ndarray) -> np.ndarray:
    """Linearly interpolate a trajectory onto a different time base, per axis."""
    return np.column_stack([np.interp(query_times, sample_times, sample_values[:, k])
                            for k in range(sample_values.shape[1])])


def overlap_window(a_times: np.ndarray, b_times: np.ndarray):
    """Boolean mask selecting the entries of `b_times` covered by `a_times`.

    Scoring is restricted to the window both trajectories cover, so a tracker that stops early
    is not credited (or penalised) for time it never saw.
    """
    lo = max(a_times.min(), b_times.min())
    hi = min(a_times.max(), b_times.max())
    return (b_times >= lo) & (b_times <= hi)


def score_trajectory(estimate_times, estimate_xyz, reference_times, reference_xyz):
    """Sim(3)-align an estimate to the reference over their shared window and score it."""
    mask = overlap_window(estimate_times, reference_times)
    if mask.sum() < 10:
        return None
    ref_t = reference_times[mask]
    ref = reference_xyz[mask]
    est = resample_to(estimate_times, estimate_xyz, ref_t)
    if not np.all(np.isfinite(est)):
        return None
    aligned, scale = align_sim3(est, ref)
    if aligned is None:
        return None
    return dict(drift_pct=drift_percent(ref, aligned),
                ate_m=ate_rmse(ref, aligned),
                scale=float(scale),
                path_length_m=float(cumulative_length(ref)[-1]),
                reference=ref[:, :2], aligned=aligned[:, :2], times=ref_t)
