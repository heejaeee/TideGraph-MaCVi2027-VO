"""Reproduce entry #19406 from released raw videos and frozen model assets.

This path does not fetch or read the estimated_signals artifact. See RAW_REPRODUCTION.md.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

PACKAGE = Path(__file__).resolve().parent
SEEDS = (0, 104729, 209759)


def link(source, target):
    source, target = Path(source).resolve(), Path(target)
    if not source.exists():
        raise FileNotFoundError(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        if target.resolve() != source:
            raise ValueError('Workspace asset changed: '+str(target))
    else:
        target.symlink_to(source, target_is_directory=source.is_dir())


def prepare(input_dir, work, assets):
    """Mount only code, weights, released TEST inputs and fixed training metadata."""
    input_dir = input_dir.resolve()
    manifest = json.loads((input_dir/'release_manifest.json').read_text())
    clips = manifest['clips']
    cfg = json.loads((PACKAGE/'configs/final_entry.json').read_text())
    if {c['key'] for c in clips} != set(cfg['clip_order']):
        raise ValueError('Expected the official 20-clip TEST release for this final entry')
    if any(c['split'] != 'test' or c.get('reference') is not None for c in clips):
        raise ValueError('Raw final-entry inference accepts only the TEST release without reference labels')
    work.mkdir(parents=True, exist_ok=True)
    contract = dict(input_manifest_sha256=hashlib.sha256((input_dir/'release_manifest.json').read_bytes()).hexdigest(),
                    final_config_sha256=hashlib.sha256((PACKAGE/'configs/final_entry.json').read_bytes()).hexdigest(),
                    assets=assets)
    lock = work/'run_contract.json'
    if lock.exists() and json.loads(lock.read_text()) != contract:
        raise ValueError('Input/configuration changed: use a new output directory')
    lock.write_text(json.dumps(contract, indent=2)+'\n')
    link(input_dir, work/'input')
    for name, path in assets['repositories'].items():
        link(path, work/'repos'/name)
    link(PACKAGE/'src/tidegraph/starter', work/'repos/lookout-vo-challenge/starter')
    for name, path in assets['files'].items():
        link(path, work/name)
    # Upstream models are loaded locally; no automatic network download in workers.
    info = json.loads((PACKAGE/'configs/segmentation.json').read_text())
    info['model'] = str(Path(assets['segmentation_model']).resolve())
    for folder in ('runs/domain_v5/segmentation', 'runs/domain_v6_test/segmentation'):
        p = work/folder/'model.json'; p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(info, indent=2)+'\n')
    p = work/'reports/domain_v3/pi3x_source.json'; p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(dict(model='yyfz233/Pi3X', revision='bb1deea4d7423de5b30691739cb451a3f57dc1d5',
        checkpoint=str((work/'models/pi3x.safetensors').resolve())), indent=2)+'\n')
    return clips


def tasks(clips, lane):
    keys = [c['key'] for c in clips]
    flir = [c['key'] for c in clips if c['camera_mode'] == 'flir-VIS']
    if lane in ('all', 'baseline-cpu'):
        for key in keys:
            for stage in ('klt', 'water', 'features'):
                yield stage, key, None
    if lane in ('all', 'baseline-gpu'):
        for key in keys:
            for stage in ('r3', 'vggt'):
                yield stage, key, None
    if lane in ('all', 'geometry'):
        for stage in ('frames', 'fusion', 'calibration', 'calibrated_rotation', 'masks', 'tracks', 'endpoint_masks', 'complete_masks', 'pi3x'):
            yield stage, None, None
        for key in flir:
            for seed in SEEDS:
                yield 'dpvo', key, seed
                yield 'slam', key, seed
        for stage in ('v24', 'courses', 'speeds'):
            yield stage, None, None
    if lane in ('all', 'omega'):
        # Reuse one frozen model across clips; generate() resumes completed clips.
        yield 'omega', None, None
    if lane in ('all', 'assemble'):
        yield 'assemble', None, None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input', required=True, type=Path)
    ap.add_argument('--output', required=True, type=Path)
    ap.add_argument('--assets', required=True, type=Path)
    ap.add_argument('--lane', choices=['all', 'baseline-cpu', 'baseline-gpu', 'geometry', 'omega', 'assemble'], default='all')
    ap.add_argument('--keys', help='Diagnostic subset of per-clip stages; not a complete submission')
    ap.add_argument('--stage', help='Run one named stage from this lane')
    ap.add_argument('--plan-only', action='store_true')
    a = ap.parse_args(); work = a.output.resolve()
    assets = json.loads(a.assets.read_text())
    clips = prepare(a.input, work, assets)
    env = os.environ.copy()
    env.update(TIDEGRAPH_RAW_WORK=str(work), PYTHONPATH=str(PACKAGE/'src'),
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONNOUSERSITE='1',
               OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', PYTHONUNBUFFERED='1')
    if assets.get('extra_pythonpath'):
        env['PYTHONPATH'] += ':'+':'.join(assets['extra_pythonpath'])
    if assets.get('dpvo_site_packages'):
        env['TIDEGRAPH_DPVO_SITE_PACKAGES'] = assets['dpvo_site_packages']
    keys = set(a.keys.split(',')) if a.keys else None
    if keys:
        env['TIDEGRAPH_RAW_KEYS'] = ','.join(sorted(keys))
    else:
        env.pop('TIDEGRAPH_RAW_KEYS', None)
    logs = work/'logs'; logs.mkdir(exist_ok=True)
    for stage, key, seed in tasks(clips, a.lane):
        if a.stage and stage != a.stage: continue
        if keys and key and key not in keys: continue
        name = '_'.join(str(x) for x in (stage, key, seed) if x is not None)
        if keys and key is None:
            name += '_subset_' + '_'.join(sorted(keys))
        marker = logs/(name+'.done.json')
        if marker.exists():
            print('RESUME', name, flush=True); continue
        command = [sys.executable, '-m', 'tidegraph.raw.worker', '--work', str(work), '--stage', stage]
        if key: command += ['--key', key]
        if seed is not None: command += ['--seed', str(seed)]
        print('RUN', name, flush=True)
        if a.plan_only: continue
        began = time.perf_counter()
        with (logs/(name+'.log')).open('a') as log:
            result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
        record = dict(stage=stage, key=key, seed=seed, returncode=result.returncode,
                      elapsed_seconds=time.perf_counter()-began)
        if result.returncode:
            (logs/(name+'.failed.json')).write_text(json.dumps(record, indent=2)+'\n')
            raise RuntimeError('Raw stage failed; see '+str(logs/(name+'.log')))
        marker.write_text(json.dumps(record, indent=2)+'\n')
        print('DONE', name, round(record['elapsed_seconds'], 1), flush=True)


if __name__ == '__main__':
    main()
