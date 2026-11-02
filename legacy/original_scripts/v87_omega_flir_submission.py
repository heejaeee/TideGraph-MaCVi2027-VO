"""Deploy the selected Omega16-only FLIR heading ablation, with V83 speed.

No label loading, per-clip tuning, training, or challenge portal upload.
Wide and infrared trajectories are exact copies of the official 12.68% V83.
"""
import argparse
import concurrent.futures
import gc
import platform
import shutil
import subprocess
import sys
import time

import cv2
import numpy as np

import v81_omega_pipeline as omega
import v82_omega_val as frame_io
import v84_flir_dvlt as development
from v83_omega_wide_submission import direction_prediction
from v78_external_data import sha, save

ROOT = omega.ROOT
DATA = ROOT / 'data/lookout-vo-test-v0.1'
MANIFEST = DATA / 'release_manifest.json'
OUT = ROOT / 'runs/domain_v87_omega_flir_test'
REP = ROOT / 'reports/domain_v87_omega_flir_submission'
DEST = ROOT / 'submissions/macvi27_vo_v87_omega_visible'
BASE = ROOT / 'submissions/macvi27_vo_v83_omega_wide/submission.json'
STATE = ROOT / 'runs/domain_v76_direction_guarded_speed/test/base/signals'
read, now, guard = omega.read, development.now, development.guard
CLIPS = {c['key']: c for c in read(MANIFEST)['clips']}
FLIR = {k: c for k, c in CLIPS.items() if c['camera_mode'] == 'flir-VIS'}


def test_case(key):
    assert key in FLIR
    pred = read(OUT / f'baseline/predictions/{key}.json')
    with np.load(STATE / f'{key}.npz') as z:
        t, yaw = z['times'], z['yaw']
        speed = z['speed'] * np.exp(z['direction_guarded_quarter'])
    assert np.array_equal(t, pred['times'])
    assert np.array_equal(omega.integrate(t, yaw, speed), pred['positions']), key
    return CLIPS[key], pred, t, yaw, speed, None


def verify_dependencies(plan):
    for path, digest in plan['dependencies'].items():
        assert sha(ROOT / path) == digest, path


def freeze():
    guard()
    if (REP / 'plan.json').exists():
        verify_dependencies(read(REP / 'plan.json'))
        return
    assert len(CLIPS) == 20 and len(FLIR) == 11
    assert all(c['reference'] is None and c['split'] == 'test' for c in CLIPS.values())
    assert sha(BASE) == '41140197649a837582b0327de3e78c6cabbc977e6dc235bc41441156a733ceab'
    base = read(BASE)
    assert set(base) == set(CLIPS)
    for key, pred in base.items():
        save(OUT / f'baseline/predictions/{key}.json', pred)
    test_parity = []
    for key in FLIR:
        _, _, t, _, _, _ = test_case(key)
        test_parity.append(dict(key=key, poses=len(t), max_position_error_m=0.0))
    files = [BASE, MANIFEST, ROOT / 'scripts/v87_omega_flir_submission.py',
             ROOT / 'scripts/v83_omega_wide_submission.py', ROOT / 'scripts/v84_flir_dvlt.py',
             ROOT / 'scripts/v82_omega_val.py', ROOT / 'scripts/v81_omega_pipeline.py',
             ROOT / 'scripts/domain_fusion_v2.py', ROOT / 'scripts/fuse_headings.py',
             ROOT / 'repos/lookout-vo-challenge/starter/pack_submission.py',
             ROOT / 'repos/lookout-vo-challenge/starter/public_dataset.py',
             development.REP / 'results.json', omega.REP / 'checkpoint_access.json']
    files += [STATE / f'{key}.npz' for key in FLIR]
    files += [DATA / c['timestamps'] for c in CLIPS.values()]
    # Verify deployment is exactly the evaluated Omega16_only ablation, without GT.
    parity = []
    for key in development.KEYS:
        case = development.case(key)
        source = omega.OUT / f'omega_w16/{case["group"]}'
        actual, yaw = direction_prediction(case['base'], case['t'], case['yaw'], case['speed'], source, key)
        expected_path = development.OUT / f'predictions/Omega16_only/{key}.json'
        expected = read(expected_path)
        assert actual == expected, key
        with np.load(development.OUT / f'signals/Omega16_only/{key}.npz') as z:
            assert np.array_equal(yaw, z['yaw']) and np.array_equal(case['speed'], z['speed'])
        files += [expected_path, case['state_path'], case['base_path']]
        parity.append(dict(key=key, group=case['group'], max_position_error_m=0.0))
    results = read(development.REP / 'results.json')
    comparison = {group: {method: values[method] for method in ['V83_FLIR_baseline', 'Omega16_only']}
                  for group, values in results['groups'].items()}
    model = read(ROOT / 'reports/domain_v82_omega_val/plan.json')['model']
    save(REP / 'plan.json', dict(
        created_utc=now(), version='V87', base_submission=str(BASE.relative_to(ROOT)),
        base_sha256=sha(BASE), base_official_test_drift_pct=12.68,
        rule='All FLIR-VIS: Omega416 W16/O8 relative rotation heading, Gaussian2seconds, initial heading alignment, exact V83 speed. Wide8 and infrared1 copied exactly from V83.',
        changed_keys=list(FLIR), preserved_keys=[k for k in CLIPS if k not in FLIR],
        model=model, precision='Existing V81/V83 model default; no additional autocast',
        preprocessing='Original baseline frame indices and timestamps, active rectangle, width960 INTER_AREA PNG, official balanced416 preprocessing. No CLAHE on Omega input.',
        selection='User-approved Omega16_only after reused TRAIN8 and CANOE_RGB2 comparisons; no independent FLIR-VIS validation set and no known candidate TEST score.',
        development_comparison=comparison, frozen_before_test_inference=True,
        test_ground_truth_used=False, reference_access_blocked=True,
        training=False, new_speed_estimation=False, test_specific_tuning=False,
        dependencies={str(p.relative_to(ROOT)): sha(p) for p in files},
        preserved_submissions={str(p.relative_to(ROOT)): sha(p) for p in sorted((ROOT / 'submissions').glob('*/submission.json'))}))
    save(REP / 'adapter_parity.json', dict(passed=True, test_baseline=test_parity,
         development_candidate=parity, exact_test_baseline_clips=11,
         exact_development_candidate_clips=10, reference_labels_used=False))
    save(REP / 'development_comparison.json', dict(groups=comparison,
         scope='Previously reused development clips, not blind validation or a predicted TEST score.'))
    print('V87_FROZEN FLIR11 changed, wide8+IR1 preserved; development adapter exact10', flush=True)


def prepare(key):
    guard()
    assert key in FLIR
    verify_dependencies(read(REP / 'plan.json'))
    cv2.setNumThreads(1)
    frame_io.DATA = DATA
    frame_io.OUT = OUT
    frame_io.BASE = OUT / 'baseline'
    frame_io.CLIPS = FLIR
    frame_io.case = test_case
    folder, meta = frame_io.frames(key)
    assert meta['source_active_rect'] == CLIPS[key]['calibration']['active_rect']
    paths = [folder / f'{i:04d}.png' for i in range(len(meta['times']))]
    assert all(p.is_file() for p in paths)
    save(REP / f'inputs/{key}.json', dict(key=key, frames=len(paths),
         index=str((folder / 'index.json').relative_to(ROOT)), index_sha256=sha(folder / 'index.json'),
         frame_sha256={p.name: sha(p) for p in paths}))
    print('FRAMES_READY', key, len(paths), flush=True)


def infer():
    guard()
    plan = read(REP / 'plan.json')
    verify_dependencies(plan)
    torch, model, load, decode = omega.model_and_io()
    repo_commit = subprocess.check_output(['git', '-C', str(omega.REPO), 'rev-parse', 'HEAD'], text=True).strip()
    assert repo_commit == '48b23c8ce72c9a8fcf987f0d7f43d8e4870759f6'
    assert not subprocess.check_output(['git', '-C', str(omega.REPO), 'status', '--porcelain', '--untracked-files=no'], text=True).strip()
    save(REP / 'environment.json', dict(python=platform.python_version(), torch=torch.__version__,
         numpy=np.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(),
         omega_repository_commit=repo_commit, omega_tracked_sources_clean=True))
    for number, key in enumerate(FLIR, 1):
        done = OUT / f'omega/diagnostics/{key}.json'
        if done.exists():
            continue
        inp = read(REP / f'inputs/{key}.json')
        index = ROOT / inp['index']
        assert sha(index) == inp['index_sha256']
        meta = read(index)
        n = len(meta['times'])
        paths = [index.parent / f'{i:04d}.png' for i in range(n)]
        for path in paths:
            assert sha(path) == inp['frame_sha256'][path.name], path
        begin = time.perf_counter()
        images = load([str(p) for p in paths], image_resolution=416, mode='balanced')
        assert len(images) == n
        torch.cuda.reset_peak_memory_stats()
        dest = OUT / f'omega/windows/{key}'
        dest.mkdir(parents=True, exist_ok=True)
        for start in range(0, n, 8):
            end = min(start + 16, n)
            path = dest / f'{start:06d}.npz'
            if not path.exists():
                with torch.inference_mode():
                    out = model(images[start:end].cuda())
                    ex, ki = decode(out['pose_enc'].float(), images.shape[-2:])
                ex = ex[0].cpu().numpy()
                c2w = np.tile(np.eye(4), (len(ex), 1, 1))
                c2w[:, :3, :3] = ex[:, :, :3].transpose(0, 2, 1)
                c2w[:, :3, 3] = -np.einsum('nij,nj->ni', c2w[:, :3, :3], ex[:, :, 3])
                intrinsics = ki[0].cpu().numpy()
                assert np.isfinite(c2w).all() and np.isfinite(intrinsics).all()
                assert np.max(abs(np.linalg.det(c2w[:, :3, :3]) - 1)) < 1e-3
                np.savez_compressed(path, start=np.array(start), c2w=c2w, intrinsics=intrinsics)
                del out, ex, ki
            if start % 160 == 0:
                print('OMEGA_WINDOW', number, len(FLIR), key, start, end, n, flush=True)
            if end == n:
                break
        assert np.isfinite(omega.relative_window_yaw(OUT / 'omega', key, n)).all()
        save(done, dict(key=key, frames=n, window=16, overlap=8, shape=list(images.shape),
             seconds=time.perf_counter() - begin, peak_cuda_gib=torch.cuda.max_memory_allocated() / 2**30,
             checkpoint_sha256=plan['model']['sha256'], reference_access_blocked=True,
             windows={p.name: sha(p) for p in sorted(dest.glob('*.npz'))}))
        del images
        gc.collect()
        torch.cuda.empty_cache()
        print('OMEGA_CLIP_DONE', number, len(FLIR), key, flush=True)


def pack():
    from public_dataset import validate_prediction, load_timestamps
    from pack_submission import pack as official_pack
    guard()
    plan = read(REP / 'plan.json')
    verify_dependencies(plan)
    base = read(BASE)
    payload, checks = {}, []
    for key, clip in CLIPS.items():
        pred = base[key]
        if key in FLIR:
            diag = read(OUT / f'omega/diagnostics/{key}.json')
            for name, digest in diag['windows'].items():
                assert sha(OUT / f'omega/windows/{key}' / name) == digest
            _, old, t, base_yaw, speed, _ = test_case(key)
            pred, yaw = direction_prediction(old, t, base_yaw, speed, OUT / 'omega', key)
            path = OUT / f'signals/{key}.npz'
            path.parent.mkdir(exist_ok=True)
            np.savez_compressed(path, times=t, yaw=yaw, speed=speed, baseline_yaw=base_yaw)
            with np.load(STATE / f'{key}.npz') as z:
                assert np.array_equal(speed, z['speed'] * np.exp(z['direction_guarded_quarter']))
        else:
            assert pred == base[key]
        ts = load_timestamps(DATA, clip)
        t, positions = validate_prediction(pred, key, ts)
        assert pred['times'] == base[key]['times'] and pred['frame_indices'] == base[key]['frame_indices']
        assert t[0] == ts[0] and t[-1] == ts[-1] and np.diff(t).max() <= 1.000001
        assert np.isfinite(positions).all() and np.ptp(positions, axis=0).max() > 1e-6
        hi = np.searchsorted(t, ts)
        ix, lo = np.clip(hi, 0, len(t)-1), np.clip(hi-1, 0, len(t)-1)
        support = np.isclose(ts, t[ix], rtol=0, atol=1e-8) | ((hi > 0) & (hi < len(t)) & (t[ix]-t[lo] <= 1.+1e-8))
        assert support.all(), key
        payload[key] = pred
        save(OUT / f'final/predictions/{key}.json', pred)
        checks.append(dict(key=key, camera_mode=clip['camera_mode'], poses=len(t), changed=pred != base[key],
             speed_preserved=True, max_gap_seconds=float(np.diff(t).max()),
             released_timestamp_coverage=float(support.mean()),
             max_position_change_from_v83=float(np.max(abs(positions-np.array(base[key]['positions']))))))
    assert {c['key'] for c in checks if c['changed']} == set(FLIR)
    DEST.mkdir(parents=True, exist_ok=True)
    bundle = DEST / 'submission.json'
    if bundle.exists():
        assert read(bundle) == payload
    else:
        official_pack(MANIFEST, OUT / 'final', bundle, 'test')
    assert read(bundle) == payload and set(payload) == set(CLIPS)
    assert bundle.stat().st_size < 64 * 1024 * 1024
    for path, digest in plan['preserved_submissions'].items():
        assert sha(ROOT / path) == digest, path
    validation = dict(passed=True, clips=20, changed_flir_visible=11, preserved_wide=8, preserved_infrared=1,
         checks=checks, official_packer_equal=True, all_speeds_preserved=True, all_timestamps_preserved=True,
         released_timestamp_coverage=1., hidden_reference_coverage_unknown=True,
         development_adapter_exact_clips=10, previous_submissions_preserved=len(plan['preserved_submissions']))
    save(DEST / 'validation.json', validation)
    save(REP / 'validation.json', validation)
    provenance = dict(created_utc=now(), version='V87', method='Omega16 FLIR-VIS heading, exact V83 speed',
         base=str(BASE.relative_to(ROOT)), base_sha256=sha(BASE), sha256=sha(bundle), bytes=bundle.stat().st_size,
         changed_keys=list(FLIR), preserved_keys=plan['preserved_keys'], checkpoint=plan['model'],
         development_comparison=plan['development_comparison'], reused_development=True,
         official_test_score=None, test_ground_truth_used=False, challenge_uploaded=False,
         source_predictions={k: sha(OUT / f'final/predictions/{k}.json') for k in CLIPS})
    save(DEST / 'provenance.json', provenance)
    save(REP / 'completion.json', provenance)
    (DEST / 'SHA256SUMS').write_text(sha(bundle) + '  submission.json\n')
    description = ('Camera-mode routed maritime visual odometry using frozen pretrained models. Wide and FLIR visible clips use VGGT-Omega 416 rotations in 16-frame windows with 8-frame overlap, relative rotation fusion and 2-second heading smoothing. Headings are integrated with water-flow relative speed on wide clips and retained R3 and static DPV-SLAM relative-speed estimates with geometric support gating on FLIR visible clips. Infrared retains the baseline heading and constant speed. Released calibration and timestamps and external pretrained weights are used. Settings are selected on public development data. No training is performed on validation or test clips.')
    allowed = set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 -:,.()[]_/\\+*%$§!?')
    assert set(description) <= allowed and '\n' not in description and '\r' not in description
    (DEST / 'description.txt').write_text(description)
    notes = ('Upload only submission.json.\nModel name: TideGraph-OmegaV\nHardware: NVIDIA RTX 5080\n'
             'Meta data used: Yes - released calibration and timestamps.\n'
             'Additional datasets used: Yes - external pretrained model weights.\n'
             'FPS: Full end-to-end runtime has not been measured. Cached-stage runtime is not full-pipeline FPS.\n\n'
             'V83 official best: 12.68 percent. V87 official score: unknown until submission.\n'
             'Reused FLIR TRAIN8: 8.643620098453221 -> 5.930328637395857 percent, 8/8 improved.\n'
             'Reused external CANOE RGB2: 6.825081654927551 -> 2.9458476524855413 percent, 2/2 improved.\n'
             'These are development results, not independent validation or a prediction of TEST score.\n'
             '11 FLIR visible headings changed. All20 relative-speed sequences retained. Wide8 and infrared1 exact V83 copies.\n'
             'One global alignment per complete trajectory for reported development metrics.\n\nDescription:\n' + description + '\n')
    (DEST / 'submission_notes.txt').write_text(notes)
    code = DEST / 'provenance_code'
    code.mkdir(exist_ok=True)
    for name in ['v87_omega_flir_submission.py', 'v83_omega_wide_submission.py', 'v82_omega_val.py',
                 'v81_omega_pipeline.py', 'domain_fusion_v2.py', 'fuse_headings.py']:
        shutil.copy2(ROOT / 'scripts' / name, code / name)
    for name in ['plan.json', 'adapter_parity.json', 'development_comparison.json', 'environment.json']:
        shutil.copy2(REP / name, DEST / name)
    print('SUBMISSION_READY', bundle, bundle.stat().st_size, sha(bundle), flush=True)


def call(action, key=None):
    name = action if key is None else action + '_' + key
    folder = REP / 'logs'
    folder.mkdir(exist_ok=True)
    args = [sys.executable, __file__, action] + (['--key', key] if key else [])
    print('START', name, flush=True)
    with (folder / f'{name}.log').open('a') as f:
        subprocess.run(args, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, check=True)
    print('DONE', name, flush=True)


def run():
    freeze()
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        jobs = [pool.submit(call, 'prepare', key) for key in FLIR if not (REP / f'inputs/{key}.json').exists()]
        for future in concurrent.futures.as_completed(jobs):
            future.result()
    call('infer')
    call('pack')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['freeze', 'prepare', 'infer', 'pack', 'run'])
    parser.add_argument('--key')
    args = parser.parse_args()
    if args.action == 'prepare':
        prepare(args.key)
    else:
        globals()[args.action]()
