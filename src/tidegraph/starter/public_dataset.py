"""Public v0.1 dataset/prediction contract (no private manifest or navigation inputs)."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np

SCHEMA_VERSION = 1
CALIBRATION_FIELDS = {
    'fx', 'fy', 'cx', 'cy', 'width', 'height', 'container_width', 'container_height',
    'active_rect', 'provenance',
}
CLIP_FIELDS = {'key', 'split', 'camera_mode', 'calibration', 'video', 'timestamps', 'reference'}
MANIFEST_FIELDS = {'schema_version', 'version', 'counts', 'clips'}
KEY_PATTERN = re.compile(r'clip_[0-9]{3,}')


def read_json(path):
    def reject_constant(value):
        raise ValueError(f'invalid JSON constant: {value}')
    with open(path) as fh:
        return json.load(fh, parse_constant=reject_constant)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def fields(value, allowed, label):
    if not isinstance(value, dict) or set(value) != allowed:
        raise ValueError(f'{label}: expected exactly {sorted(allowed)}')


def finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and np.isfinite(value)


def validate_calibration(cal):
    fields(cal, CALIBRATION_FIELDS, 'calibration')
    for name in ('fx', 'fy', 'cx', 'cy'):
        if not finite_number(cal[name]) or (name in ('fx', 'fy') and cal[name] <= 0):
            raise ValueError(f'invalid calibration {name}')
    for name in ('width', 'height', 'container_width', 'container_height'):
        if type(cal[name]) is not int or cal[name] <= 0:
            raise ValueError(f'invalid calibration {name}')
    rect = cal['active_rect']
    if not isinstance(rect, list) or len(rect) != 4 or any(type(v) is not int for v in rect):
        raise ValueError('active_rect must be integer [x, y, width, height]')
    x, y, w, h = rect
    if (x < 0 or y < 0 or w != cal['width'] or h != cal['height'] or
            x + w > cal['container_width'] or y + h > cal['container_height']):
        raise ValueError('active_rect does not fit the encoded image/calibration')
    if not 0 <= cal['cx'] <= w or not 0 <= cal['cy'] <= h:
        raise ValueError('principal point is outside the active image')
    if cal['provenance'] not in ('nominal-fov-derived', 'measured', 'estimated'):
        raise ValueError('unknown calibration provenance')
    return cal


def calibration_from_private(record):
    active = record['active']
    x0, y0, x1, y1 = record['active_rect']  # private resolver uses inclusive corners
    cal = {name: active[name] for name in ('fx', 'fy', 'cx', 'cy', 'width', 'height')}
    cal.update(container_width=record['container'][0], container_height=record['container'][1],
               active_rect=[x0, y0, x1 - x0 + 1, y1 - y0 + 1],
               provenance=record['provenance'])
    return validate_calibration(cal)


def asset_path(root, relative):
    if not isinstance(relative, str) or Path(relative).is_absolute() or '..' in Path(relative).parts:
        raise ValueError('dataset assets must have relative paths inside the package')
    root = Path(root).resolve()
    result = (root / relative).resolve()
    if not result.is_relative_to(root):
        raise ValueError('dataset asset escapes the package')
    return result


def load_manifest(path):
    data = read_json(path)
    fields(data, MANIFEST_FIELDS, 'release manifest')
    if data['schema_version'] != SCHEMA_VERSION or not isinstance(data['version'], str):
        raise ValueError('unsupported release manifest version')
    if not isinstance(data['clips'], list) or not data['clips']:
        raise ValueError('empty release manifest')
    seen = set()
    counts = {'train': 0, 'val': 0, 'test': 0}
    for clip in data['clips']:
        fields(clip, CLIP_FIELDS, 'clip')
        key = clip['key']
        if not isinstance(key, str) or not KEY_PATTERN.fullmatch(key) or key in seen:
            raise ValueError('clip IDs must be unique anonymous clip_NNN identifiers')
        seen.add(key)
        if clip['split'] not in counts:
            raise ValueError(f'{key}: invalid split')
        counts[clip['split']] += 1
        if not isinstance(clip['camera_mode'], str) or not clip['camera_mode']:
            raise ValueError(f'{key}: camera_mode is required')
        validate_calibration(clip['calibration'])
        expected = {'video': f'videos/{key}.mp4', 'timestamps': f'timing/{key}.json',
                    'reference': None if clip['split'] == 'test' else f'references/{key}.json'}
        for name, value in expected.items():
            if clip[name] != value:
                raise ValueError(f'{key}: invalid {name} path')
    expected_counts = {**counts, 'total': len(seen)}
    if data['counts'] != expected_counts:
        raise ValueError('manifest counts do not match the clip list')
    return data


def select_clips(manifest, split='val', clips=None):
    if clips is not None:
        wanted = set(clips.split(','))
        selected = [c for c in manifest['clips'] if c['key'] in wanted]
        unknown = wanted - {c['key'] for c in selected}
        if unknown:
            raise ValueError(f'unknown clip IDs: {sorted(unknown)}')
    else:
        wanted = {'train', 'val', 'test'} if split == 'all' else set(split.split(','))
        if not wanted <= {'train', 'val', 'test'}:
            raise ValueError(f'unknown splits: {sorted(wanted)}')
        selected = [c for c in manifest['clips'] if c['split'] in wanted]
    if not selected:
        raise ValueError('selection contains no clips')
    return selected


def validate_times(values, label='times', minimum=2):
    if not isinstance(values, list) or any(not finite_number(v) for v in values):
        raise ValueError(f'{label}: expected finite numeric times')
    times = np.asarray(values, dtype=float)
    if times.ndim != 1 or len(times) < minimum or not np.all(np.diff(times) > 0):
        raise ValueError(f'{label}: expected at least {minimum} strictly increasing times')
    return times


def load_timestamps(root, clip):
    data = read_json(asset_path(root, clip['timestamps']))
    fields(data, {'key', 'times'}, 'timestamps')
    if data['key'] != clip['key']:
        raise ValueError('timestamps key does not match manifest')
    return validate_times(data['times'], 'frame presentation times')


def load_reference(root, clip):
    data = read_json(asset_path(root, clip['reference']))
    fields(data, {'key', 'times', 'positions'}, 'reference')
    if data['key'] != clip['key']:
        raise ValueError('reference key does not match manifest')
    times = validate_times(data['times'], 'reference times', minimum=10)
    positions = np.asarray(data['positions'], dtype=float)
    if positions.shape != (len(times), 3) or not np.isfinite(positions).all():
        raise ValueError('reference positions must be finite Nx3 coordinates')
    return times, positions


def validate_prediction(data, key, frame_times):
    # The minimal format is model-independent; metadata may be added by other estimators.
    if not isinstance(data, dict) or data.get('key') != key:
        raise ValueError('prediction key does not match filename/manifest')
    times = validate_times(data['times'], 'prediction times')
    xyz = np.asarray(data['positions'], dtype=float)
    if xyz.shape != (len(times), 3):
        raise ValueError('positions must be Nx3, one per prediction time')
    if not np.isfinite(xyz).all():
        raise FloatingPointError('prediction positions contain non-finite values')
    indices = data['frame_indices']
    if (not isinstance(indices, list) or len(indices) != len(times) or
            any(type(i) is not int for i in indices) or
            indices[0] < 0 or indices[-1] >= len(frame_times) or
            any(b <= a for a, b in zip(indices, indices[1:]))):
        raise ValueError('frame_indices must be increasing decoded source-frame indices')
    if not np.allclose(times, frame_times[indices], rtol=0, atol=1e-6):
        raise ValueError('prediction times do not match the released source-frame PTS')
    return times, xyz
