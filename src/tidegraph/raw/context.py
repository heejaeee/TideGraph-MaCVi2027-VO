"""Shared paths and frozen primitives for the raw-video workers.

ROOT is a new output workspace, never the historical experiment directory.
The only incoming files are released videos/timestamps/calibration and weights.
"""
import json
import os
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation

from ..core import integrate as integrate_velocity
from ..starter.public_dataset import load_manifest, load_timestamps, write_json
from .axes import frame_axes, wrap
from . import motion
from .legacy.fuse_headings import planar_yaw

motion.planar_yaw = planar_yaw

ROOT = Path(os.environ['TIDEGRAPH_RAW_WORK']).resolve()
DATA = ROOT / 'input'
CLIPS = {c['key']: c for c in load_manifest(DATA/'release_manifest.json')['clips']}
KEYS = [k for k, c in CLIPS.items() if c['camera_mode'] == 'flir-VIS']
if os.environ.get('TIDEGRAPH_RAW_KEYS'):
    KEYS = [k for k in KEYS if k in os.environ['TIDEGRAPH_RAW_KEYS'].split(',')]
SEEDS = [0, 104729, 209759]


def save_json(path, value):
    path = Path(path)
    temp = path.with_name(path.name + '.tmp')
    write_json(temp, value)
    temp.replace(path)


def frame_root(split):
    if split != 'test':
        raise ValueError('This entry reproducer does not train or read TRAIN/VAL labels')
    return ROOT/'runs/domain_v6_test'


def frames(key):
    folder = frame_root('test')/'frames'/key
    return folder, json.loads((folder/'index.json').read_text())


def window_starts(n):
    starts = list(range(0, max(1, n-31), 16))
    if starts[-1] != max(0, n-32):
        starts.append(max(0, n-32))
    return starts


def mask_path(key, i, split):
    paths = [frame_root(split)/f'segmentation/{key}/{i:04d}.png',
             ROOT/f'runs/domain_v16/{split}/extra_masks/{key}/{i:04d}.png',
             ROOT/f'runs/domain_v65_submission/{split}/masks/{key}/{i:04d}.png']
    return next((p for p in paths if p.exists()), paths[-1])


def pose_path(key, split, seed):
    if split != 'test':
        raise ValueError(split)
    return ROOT/f'runs/domain_v65_submission/test/poses/slam_s{seed}/{key}.npz'


def envelope(ds):
    lo, hi = ds.min(0), ds.max(0)
    return np.where(lo > 0, lo, np.where(hi < 0, hi, 0.))


def provider(key, split):
    if split != 'test':
        raise ValueError(split)
    base = ROOT/'runs/submission_test_v1'
    pred = json.loads((base/f'r3/predictions/{key}.json').read_text())
    t = np.asarray(pred['times'])
    rp = np.load(base/f'r3/diagnostics/{key}.poses.npz')['camera_to_world']
    k = np.load(ROOT/f'runs/domain_v4/calibrated_rotation_test/{key}.npz')['learned_k']
    _, meta = frames(key)
    return dict(pred=pred, t=t), None, None, rp, k, meta


def guard():
    # Process-wide filesystem/network audit is installed by worker.py.
    pass


core = SimpleNamespace(bg=SimpleNamespace(guard=guard, wrap=wrap, frame_axes=frame_axes),
                       g=SimpleNamespace(provider=provider),
                       local=SimpleNamespace(motion=motion),
                       integrate_velocity=integrate_velocity, Rotation=Rotation)
v = SimpleNamespace(core=core, ROOT=ROOT, np=np, json=json, cv2=cv2,
                    save_json=save_json, TRAIN=[], CLIPS=CLIPS)
test = SimpleNamespace(KEYS=KEYS, CLIPS=CLIPS)
