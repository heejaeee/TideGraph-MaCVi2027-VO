"""One isolated raw inference stage; invoked by raw_run.py.

Network connections and reads of packaged prediction caches/reference labels
are forbidden. Every estimated intermediate must live in this run's workspace.
"""
import argparse
import json
import os
from pathlib import Path
import runpy
import sys
import time


def install_guard(work, package):
    denied = [package/'data', package/'submission', package/'legacy']
    historical = os.environ.get('TIDEGRAPH_FORBID_HISTORY')
    if historical:
        denied += [Path(historical)/n for n in ('runs', 'submissions', 'data/features')]

    def audit(event, args):
        if event in ('socket.connect', 'socket.getaddrinfo'):
            raise RuntimeError('Raw inference is offline; prepare weights before running')
        if event == 'open' and isinstance(args[0], (str, bytes, os.PathLike)):
            p = Path(os.fsdecode(args[0])).resolve()
            in_work = p == work or work in p.parents
            if not in_work and any(p == d or d in p.parents for d in denied):
                raise RuntimeError('Raw inference attempted to read an archived prediction: '+str(p))
            if p.suffix in ('.csv', '.npz', '.npy', '.json') and any(part in ('reference', 'references', 'ground_truth') for part in p.parts):
                raise RuntimeError('Reference labels are not inference inputs: '+str(p))
    sys.addaudithook(audit)


def baseline(component, key):
    from .context import ROOT, DATA
    names = dict(r3='submission_r3_infer_clip', vggt='submission_vggt_infer_clip',
                 klt='submission_rotation_infer_clip', water='submission_water_infer_clip',
                 features='submission_extract_features')
    dest = ROOT/'runs/submission_test_v1'/component
    dest.mkdir(parents=True, exist_ok=True)
    script = Path(__file__).parent/'legacy'/f'{names[component]}.py'
    sys.argv = [str(script), '--dataset', str(DATA/'release_manifest.json'),
                '--key', key, '--output', str(dest)]
    runpy.run_path(str(script), run_name='__main__')


def frames_all():
    from .context import ROOT, DATA, CLIPS, KEYS, save_json
    from ..omega_windows import frame_ids, frames
    from ..starter.public_dataset import load_timestamps
    import cv2
    meta = {}
    for key, c in CLIPS.items():
        ts = load_timestamps(DATA, c)
        ids = frame_ids(ts)
        rec = dict(camera_mode=c['camera_mode'], times=ts[ids].tolist(),
                   frame_indices=ids.tolist())
        meta[key] = rec
        if key in KEYS:
            folder = ROOT/'runs/domain_v6_test/frames'/key
            folder.mkdir(parents=True, exist_ok=True)
            if (folder/'index.json').exists():
                continue
            images = frames(DATA, c, ids)
            for i, rgb in enumerate(images):
                if not cv2.imwrite(str(folder/f'{i:04d}.png'), rgb[:, :, ::-1],
                                   [cv2.IMWRITE_PNG_COMPRESSION, 1]):
                    raise OSError(key)
            save_json(folder/'index.json', dict(rec, key=key, width=960,
                      height=images[0].shape[0], source_active_rect=c['calibration']['active_rect']))
            print('FRAMES', key, len(ids), flush=True)
    save_json(ROOT/'generated/metadata/clip_index.json', meta)


def legacy_cli(name, args):
    script = Path(__file__).parent/'legacy'/f'{name}.py'
    sys.argv = [str(script), *map(str, args)]
    runpy.run_path(str(script), run_name='__main__')


def fusion():
    from .context import ROOT, DATA
    base = ROOT/'runs/submission_test_v1'
    args = ['--dataset', DATA/'release_manifest.json', '--split', 'test',
            '--model', ROOT/'models/flow_speed_v3', '--output', base/'fused']
    for name in ('r3', 'vggt', 'klt', 'water', 'features'):
        args += ['--'+name, base/name]
    if os.environ.get('TIDEGRAPH_RAW_KEYS'):
        args += ['--keys', os.environ['TIDEGRAPH_RAW_KEYS']]
    legacy_cli('submission_fuse', args)


def v24():
    from .context import ROOT, KEYS, SEEDS, save_json, planar_yaw, gaussian_filter1d
    from ..core import relative_window_yaw
    import numpy as np
    base = ROOT/'runs/submission_test_v1'
    records = {r['key']: r for r in json.loads((base/'fused/inference_manifest.json').read_text())['clips']}
    for key in KEYS:
        pred = json.loads((base/f'r3/predictions/{key}.json').read_text())
        t = np.asarray(pred['times']); sigma = 2/np.median(np.diff(t))
        rp = np.load(base/f'r3/diagnostics/{key}.poses.npz')['camera_to_world']
        # Historical np.interp casts the float32 R3 yaw to float64 before smoothing.
        r = gaussian_filter1d(np.interp(t, t, planar_yaw(rp)), sigma)
        s = np.load(base/f'fused/diagnostics/{key}.fusion.npz')['speed'].copy()
        if records[key]['speed_correlation'] < .7:
            s = np.ones(len(t))
        raw = []
        for seed in SEEDS:
            z = np.load(ROOT/f'runs/domain_v6_test/dpvo/random_s{seed}/{key}.npz')
            raw.append(gaussian_filter1d(np.interp(t, z['times'], planar_yaw(z['camera_to_world'])), sigma))
        vg = gaussian_filter1d(relative_window_yaw(base/'vggt', key, len(t)), sigma)
        pi = gaussian_filter1d(relative_window_yaw(ROOT/'runs/domain_v13/pi3x_test', key, len(t)), sigma)
        family = np.array([a-a[0]+r[0] for a in (np.median(raw, axis=0), vg, pi)])
        path = ROOT/f'runs/domain_v24_submission/test/signals/{key}.npz'
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, times=t, speed=s, yaw=np.median(family, axis=0),
                            initial_yaw=r[0], family_yaws=family, dpvo_yaws=raw)


def courses():
    from .context import ROOT, KEYS, core, save_json
    from . import course, consensus, multiview
    from types import SimpleNamespace
    course.core = core
    course.audit = SimpleNamespace(initial=None)
    course.inputs = lambda k: consensus.inputs(k, 'test')
    multiview.OUT = ROOT/'runs/domain_v65_submission/test/course'
    multiview.save_json = save_json
    multiview.prepare = lambda k: consensus.prepare(k, 'test')
    for fold, seed in enumerate((0, 104729, 314159, 419431, 523717)):
        name = f'fold{fold}'
        multiview.SCENARIOS = {name: dict(fold=fold, thin=1, focal=1., seed=seed)}
        multiview.extract(name, KEYS)


def speeds():
    from . import consensus, old_speed, speed, support
    from .context import ROOT, KEYS
    import numpy as np
    consensus.infer('test')
    old_speed.infer('test')
    speed.extract('test')
    for key in KEYS:
        raw = np.load(speed.OUT/f'test/raw/{key}.npz')
        old = np.load(old_speed.OUT/f'test/signals/{key}.npz')['log_speed_delta']
        dest = speed.OUT/f'test/base/signals/{key}.npz'
        dest.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(dest, times=raw['times'], yaw=raw['yaw'], speed=raw['speed'],
                            v65_control=np.zeros(len(old)), v73_control=old, **speed.signals(raw, 'base'))
    support.infer('test')


def assemble():
    from .context import ROOT, CLIPS
    from .. import pipeline
    import numpy as np
    dest = ROOT/'generated/inputs'; dest.mkdir(parents=True, exist_ok=True)
    selected = os.environ.get('TIDEGRAPH_RAW_KEYS', '').split(',')
    for key, c in CLIPS.items():
        if selected != [''] and key not in selected:
            continue
        with np.load(ROOT/f'runs/submission_test_v1/fused/diagnostics/{key}.fusion.npz') as z:
            t, y, s = z['times'], z['yaw'], z['speed']
        if c['camera_mode'] == 'flir-VIS':
            with np.load(ROOT/f'runs/domain_v76_direction_guarded_speed/test/base/signals/{key}.npz') as z:
                t, y = z['times'], z['yaw']
                s = z['speed']*np.exp(z['direction_guarded_quarter'])
        if c['camera_mode'] == 'flir-IR':
            s = np.ones(len(t))
        np.savez_compressed(dest/f'{key}.npz', times=t, initial_yaw=y[0], speed=s,
                            **({'legacy_yaw': y} if c['camera_mode'] == 'flir-IR' else {}))
    cfg = pipeline.read(Path(__file__).resolve().parents[3]/'configs/final_entry.json')
    subset = selected != ['']
    if subset:
        cfg['clip_order'] = [k for k in cfg['clip_order'] if k in selected]
    bundle, checks = pipeline.build(ROOT/'input', ROOT/'generated', cfg, 'rebuild', ROOT/'omega')
    encoded = pipeline.encode(bundle)
    name = 'diagnostic_subset' if subset else 'submission'
    (ROOT/(name+'.json')).write_bytes(encoded)
    import hashlib
    (ROOT/('diagnostic_subset_report.json' if subset else 'raw_report.json')).write_text(json.dumps(dict(clips=len(bundle), checks=checks,
        sha256=hashlib.sha256(encoded).hexdigest(), expected_sha256=cfg['expected_sha256'],
        archived_estimates_read=False, source='raw released videos and frozen weights'), indent=2)+'\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--work', required=True, type=Path)
    ap.add_argument('--stage', required=True)
    ap.add_argument('--key'); ap.add_argument('--seed', type=int)
    a = ap.parse_args(); work = a.work.resolve()
    os.environ['TIDEGRAPH_RAW_WORK'] = str(work)
    package = Path(__file__).resolve().parents[3]
    legacy = Path(__file__).parent/'legacy'
    sys.path[:0] = [str(legacy), str(package/'src/tidegraph/starter')]
    for name in ('R3', 'vggt', 'Pi3', 'AnyCalib', 'DPVO', 'co-tracker'):
        sys.path.insert(0, str(work/'repos'/name))
    if a.stage in ('dpvo', 'slam') and os.environ.get('TIDEGRAPH_DPVO_SITE_PACKAGES'):
        sys.path.append(os.environ['TIDEGRAPH_DPVO_SITE_PACKAGES'])
    sys.path.insert(0, str(work/'repos/DPVO/dpvo/lietorch'))
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                      TORCH_HOME=str(work/'torch'), HF_HOME=str(work/'huggingface'))
    install_guard(work, package)
    from .context import ROOT, DATA, KEYS
    import cv2
    cv2.setNumThreads(1)
    manifest = DATA/'release_manifest.json'
    started = time.perf_counter()
    if a.stage in ('r3', 'vggt', 'klt', 'water', 'features'):
        baseline(a.stage, a.key)
    elif a.stage == 'frames': frames_all()
    elif a.stage == 'fusion': fusion()
    elif a.stage == 'calibration':
        legacy_cli('calibrate_domain_v4', ['--dataset', manifest, '--keys', ','.join(KEYS),
                   '--output', ROOT/'runs/domain_v4/anycalib_test'])
    elif a.stage == 'calibrated_rotation':
        legacy_cli('calibrated_rotation_v4', ['--dataset', manifest, '--keys', ','.join(KEYS),
                   '--calibration', ROOT/'runs/domain_v4/anycalib_test',
                   '--output', ROOT/'runs/domain_v4/calibrated_rotation_test'])
    elif a.stage == 'pi3x':
        legacy_cli('pi3x_domain_infer', ['--dataset', manifest, '--keys', ','.join(KEYS),
                   '--output', ROOT/'runs/domain_v13/pi3x_test'])
    elif a.stage in ('masks', 'tracks'):
        import torch
        from . import features
        torch.manual_seed(0); torch.set_num_threads(2)
        getattr(features, a.stage)()
    elif a.stage == 'complete_masks':
        from .complete_masks import main as run
        run()
    elif a.stage == 'endpoint_masks':
        from .endpoint_masks import main as run
        run()
    elif a.stage == 'dpvo':
        from .dpvo import work as run
        run(a.key, a.seed)
    elif a.stage == 'slam':
        from .slam import run
        run(a.key, 'slam', a.seed)
    elif a.stage == 'v24': v24()
    elif a.stage == 'courses': courses()
    elif a.stage == 'speeds': speeds()
    elif a.stage == 'assemble': assemble()
    elif a.stage == 'omega':
        from ..omega_windows import generate
        generate(DATA, ROOT/'omega', ROOT/'models/omega.pt', ROOT/'repos/vggt-omega',
                 keys=[a.key] if a.key else (sorted(os.environ['TIDEGRAPH_RAW_KEYS'].split(','))
                                             if os.environ.get('TIDEGRAPH_RAW_KEYS') else None))
    else: raise ValueError(a.stage)
    print('RAW_STAGE_COMPLETE', a.stage, a.key, time.perf_counter()-started, flush=True)


if __name__ == '__main__':
    main()
