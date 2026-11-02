"""Assemble the final-entry trajectories from per-clip heading and speed signals.

Per camera mode (configs/final_entry.json):
  wide      Omega window rotations -> median adjacent relative rotations -> planar yaw
            -> Gaussian 2 s; yaw gain about the initial heading (0.95, clip_092/103: 1.30);
            retained legacy water-flow speed -> uniform 2 Hz, mean 1 -> TV (lambda 32).
  flir-VIS  Omega window rotations -> all-pairs in-window yaw graph (Huber 3 deg)
            -> Gaussian 2 s; group gain A 1.20 / B 1.10; retained speed.
  flir-IR   retained legacy yaw and constant speed.
Positions are the trapezoidal integral of speed * (sin yaw, 0, cos yaw).

mode='replay' takes the 19226 heading/speed signals from data/final_signals (exact
bytes of the uploaded file); mode='rebuild' recomputes the headings from Omega
windows (data/omega, or windows regenerated from video). The two differ by ~1e-12 m.
"""
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from . import core
from .starter.public_dataset import load_manifest, load_timestamps, validate_prediction

MAX_GAP_S = 1.0
MIN_COVERAGE = 0.95


def read(path):
    return json.loads(Path(path).read_text())


def gap_aware_coverage(t, released):
    """Fraction of released frame times covered by a sample or a <=1 s interpolation interval."""
    inside = (released >= t[0]) & (released <= t[-1])
    k = np.clip(np.searchsorted(t, released, side='right') - 1, 0, len(t) - 2)
    on_sample = np.isclose(released, t[k]) | np.isclose(released, t[k + 1])
    short = (t[k + 1] - t[k]) <= MAX_GAP_S
    return float(np.mean(inside & (on_sample | short)))


def clip_signals(key, mode_name, t, data_dir, omega_dir, cfg, mode):
    """Return (yaw, speed, diagnostics) for one clip."""
    n = len(t)
    with np.load(data_dir / 'inputs' / f'{key}.npz', allow_pickle=False) as z:
        np.testing.assert_array_equal(t, z['times'])
        anchor = float(z['initial_yaw'])
        legacy_speed = z['speed']
        legacy_yaw = z['legacy_yaw'] if mode_name == 'flir-IR' else None
    final = None
    if mode == 'replay':
        with np.load(data_dir / 'final_signals' / f'{key}.npz', allow_pickle=False) as z:
            np.testing.assert_array_equal(t, z['times'])
            final = dict(yaw=z['yaw'], speed=z['speed'])
    detail = {}
    if mode_name == 'wide':
        if final is not None:
            yaw = final['yaw']
        else:
            raw = core.relative_window_yaw(omega_dir, key, n)
            yaw = core.gaussian_filter1d(raw, cfg['yaw_sigma_seconds'] / np.median(np.diff(t)))
            yaw = yaw - yaw[0] + anchor
        gain = cfg['wide_gain_overrides'].get(key, cfg['wide_gain'])
        yaw = anchor + gain * (yaw - anchor)
        speed, detail = core.apply(t, legacy_speed, 'tv', cfg['wide_tv_lambda'])
        detail = dict(detail, yaw_gain=gain)
    elif mode_name == 'flir-VIS':
        if final is not None:
            yaw, speed = final['yaw'], final['speed']
        else:
            raw, i, j, z, _ = core.chain_and_edges(omega_dir, key, n)
            raw, detail = core.solve_yaw(n, i, j, z, raw, delta_deg=cfg['graph_huber_degrees'])
            if not detail['converged']:
                raise RuntimeError('Yaw solver did not converge: ' + key)
            yaw = core.gaussian_filter1d(raw, cfg['yaw_sigma_seconds'] / np.median(np.diff(t)))
            yaw += anchor - yaw[0]
            group = 'A' if key in cfg['flir_groups']['A'] else 'B'
            yaw = anchor + cfg['flir_gains'][group] * (yaw - anchor)
            speed = legacy_speed
            detail = dict(detail, group=group)
    elif mode_name == 'flir-IR':
        yaw, speed = (final['yaw'], final['speed']) if final is not None else (legacy_yaw, legacy_speed)
    else:
        raise ValueError(mode_name)
    return yaw, speed, detail


def build(input_dir, data_dir, cfg, mode, omega_dir=None, log=print):
    """Return (bundle, per-clip checks). input_dir is the released test package root."""
    input_dir, data_dir = Path(input_dir), Path(data_dir)
    omega_dir = Path(omega_dir) if omega_dir else data_dir / 'omega'
    meta = read(data_dir / 'metadata' / 'clip_index.json')
    clips = {c['key']: c for c in load_manifest(input_dir / 'release_manifest.json')['clips']}
    bundle, checks = {}, []
    for key in cfg['clip_order']:
        m = meta[key]
        t = np.array(m['times'])
        mode_name = clips[key]['camera_mode']
        if mode_name != m['camera_mode']:
            raise ValueError(f'camera mode mismatch for {key}')
        yaw, speed, detail = clip_signals(key, mode_name, t, data_dir, omega_dir, cfg, mode)
        positions = core.integrate(t, yaw, speed)
        pred = dict(key=key, frame_indices=m['frame_indices'], times=m['times'], positions=positions.tolist())
        released = load_timestamps(input_dir, clips[key])
        validate_prediction(pred, key, released)
        coverage = gap_aware_coverage(t, released)
        max_gap = float(np.diff(t).max())
        if coverage < MIN_COVERAGE or max_gap > MAX_GAP_S:
            raise ValueError(f'coverage/gap gate failed for {key}: {coverage:.3f}, {max_gap:.2f} s')
        bundle[key] = pred
        checks.append(dict(key=key, mode=mode_name, samples=len(t), coverage=coverage,
                           max_gap_s=max_gap, solver=detail))
        log(f'{mode} {key} {mode_name} samples={len(t)} coverage={coverage:.4f}')
    return bundle, checks


def encode(bundle):
    """Byte layout of the uploaded file: compact JSON, clip_order keys, trailing newline."""
    return (json.dumps(bundle, separators=(',', ':'), allow_nan=False) + '\n').encode()


def compare(generated_dir, reference_dir, keys):
    """Max geodesic rotation difference (deg) per clip between two window sets."""
    out = {}
    for key in keys:
        worst = 0.
        for f in sorted((Path(reference_dir) / 'windows' / key).glob('*.npz')):
            with np.load(f) as a, np.load(Path(generated_dir) / 'windows' / key / f.name) as b:
                d = Rotation.from_matrix(a['c2w'][:, :3, :3].transpose(0, 2, 1) @ b['c2w'][:, :3, :3]).magnitude()
            worst = max(worst, float(d.max()))
        out[key] = float(np.rad2deg(worst))
    return out
