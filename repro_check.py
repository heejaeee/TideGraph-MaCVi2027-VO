"""Hand-in checker for the TideGraph Maritime Visual Odometry release.

Modelled on the MaCVi 2027 reproduction template checker, adapted to a trajectory
task (the output is one submission.json bundle, not per-image files).

    python repro_check.py .                                   layout, repro.yaml, weights link, run.py signature,
                                                              submitted bundle (hash, 20 clips, format, coverage gate)
    python repro_check.py . --run --input <test package>      also runs main() (default mode replay) and compares
    python repro_check.py . --run --input <pkg> --mode rebuild
    python repro_check.py . --download-weights                fetch the Omega checkpoint (gated: needs HF_TOKEN)
"""
import argparse
import ast
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
import urllib.request
from pathlib import Path

REQUIRED = ['README.md', 'METHOD.md', 'repro.yaml', 'requirements.txt', 'run.py', 'repro_check.py', 'LICENSE',
            'configs/final_entry.json', 'submission/submission.json', 'weights/weights_link.yaml',
            'src/tidegraph/core.py', 'src/tidegraph/pipeline.py', 'src/tidegraph/omega_windows.py',
            'src/tidegraph/starter/public_dataset.py', 'data/metadata/clip_index.json',
            'data/metadata/release_manifest.json', 'docker/Dockerfile']
DATA_DIRS = ['data/omega/windows', 'data/inputs', 'data/final_signals', 'data/metadata/timing']
REPRO_KEYS = ['team', 'track', 'submission_id', 'contact']
MAX_BUNDLE = 64 * 2**20
problems = 0


def ok(msg):
    print('  OK   ' + msg)


def warn(msg):
    print('  WARN ' + msg)


def fail(msg):
    global problems
    problems += 1
    print('  FAIL ' + msg)


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def flat_yaml(path):
    """key: value pairs of a flat YAML file (comments stripped); enough for repro.yaml / weights_link.yaml."""
    out = {}
    for line in Path(path).read_text().splitlines():
        line = line.split(' #', 1)[0].rstrip()
        if not line or line.lstrip().startswith('#') or ':' not in line:
            continue
        k, v = line.split(':', 1)
        out[k.strip()] = v.strip().strip('"\'')
    return out


def check_layout(root):
    print('layout')
    for name in REQUIRED:
        (ok if (root / name).is_file() else fail)(name)
    for name in DATA_DIRS:
        d = root / name
        (ok if d.is_dir() and any(d.iterdir()) else fail)(name + '/')
    bad = []
    for path in [root / 'run.py', *(root / 'src').rglob('*.py')]:
        if re.search(r"['\"](/media/|/home/|/Users/|[A-Za-z]:\\\\)", path.read_text()):
            bad.append(str(path.relative_to(root)))
    (fail(f'absolute paths in {bad}') if bad else ok('no absolute paths in run.py / src/'))


def check_repro_yaml(root):
    print('repro.yaml')
    y = flat_yaml(root / 'repro.yaml')
    for k in REPRO_KEYS:
        if not y.get(k):
            fail(f'{k} missing')
        elif 'TODO' in y[k]:
            warn(f'{k} still a placeholder: {y[k]}')
        else:
            ok(f'{k}: {y[k]}')


def check_weights(root):
    print('weights')
    link = root / 'weights' / 'weights_link.yaml'
    y = flat_yaml(link)
    for k in ('url', 'sha256', 'filename'):
        (ok if y.get(k) else fail)(f'weights_link.yaml {k}')
    if y.get('sha256') and not re.fullmatch(r'[0-9a-f]{64}', y['sha256']):
        fail('weights_link.yaml sha256 is not 64 hex characters')
    local = root / 'weights' / y.get('filename', '')
    if y.get('filename') and local.is_file():
        (ok if sha(local) == y['sha256'] else fail)(f'downloaded {local.name} matches sha256')
    else:
        ok('checkpoint not downloaded (only mode=video needs it)')


def check_signature(root):
    print('run.py')
    tree = ast.parse((root / 'run.py').read_text())
    fn = next((n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main'), None)
    if fn is None:
        return fail('run.py has no top-level main()')
    args = [a.arg for a in fn.args.args][:4]
    (ok if args == ['model_dir', 'input_dir', 'output_dir', 'device'] else fail)(f'main({", ".join(args)}, ...)')


def check_bundle(root, path, label):
    import numpy as np
    sys.path.insert(0, str(root / 'src'))
    from tidegraph.pipeline import gap_aware_coverage, MIN_COVERAGE, MAX_GAP_S
    cfg = json.loads((root / 'configs/final_entry.json').read_text())
    meta = json.loads((root / 'data/metadata/clip_index.json').read_text())
    size = path.stat().st_size
    (ok if size <= MAX_BUNDLE else fail)(f'{label}: {size / 2**20:.2f} MiB (limit 64 MiB)')
    bundle = json.loads(path.read_text())
    (ok if list(bundle) == cfg['clip_order'] else fail)(f'{label}: {len(bundle)} clips in clip_order')
    worst = 1.
    for key in cfg['clip_order']:
        p = bundle.get(key)
        if p is None:
            continue
        t = np.asarray(p['times'], float)
        xyz = np.asarray(p['positions'], float)
        released = np.asarray(json.loads((root / 'data/metadata/timing' / f'{key}.json').read_text())['times'])
        good = (p['key'] == key and xyz.shape == (len(t), 3) and np.isfinite(xyz).all()
                and np.all(np.diff(t) > 0) and p['frame_indices'] == meta[key]['frame_indices']
                and np.allclose(released[p['frame_indices']], t, rtol=0, atol=1e-6))
        cov = gap_aware_coverage(t, released)
        worst = min(worst, cov)
        if not good or cov < MIN_COVERAGE or np.diff(t).max() > MAX_GAP_S:
            fail(f'{label}: {key} format/coverage (coverage {cov:.3f}, max gap {np.diff(t).max():.2f} s)')
    ok(f'{label}: minimum gap-aware coverage {worst:.4f} (gate {MIN_COVERAGE}, gaps <= {MAX_GAP_S:g} s)')
    return sha(path)


def check_submission(root):
    print('submission/submission.json')
    cfg = json.loads((root / 'configs/final_entry.json').read_text())
    digest = check_bundle(root, root / 'submission/submission.json', 'uploaded bundle')
    (ok if digest == cfg['expected_sha256'] else fail)(f'sha256 {digest}')


def run(root, input_dir, output, mode, device):
    print(f'run.main (mode={mode})')
    sys.path.insert(0, str(root))
    import run as entry
    report = entry.main(root / 'weights', input_dir, output, device, mode)
    digest = check_bundle(root, Path(output) / 'submission.json', 'regenerated bundle')
    if report['byte_identical']:
        ok(f'regenerated bundle is byte-identical to the upload ({digest[:16]}...)')
    elif report['max_position_diff_m'] < 1e-6:
        ok(f'regenerated bundle matches numerically (max {report["max_position_diff_m"]:.2e} m), bytes differ')
    else:
        warn(f'regenerated bundle differs by up to {report["max_position_diff_m"]:.3f} m; see report.json')


def download(root):
    y = flat_yaml(root / 'weights' / 'weights_link.yaml')
    dest = root / 'weights' / y['filename']
    if dest.is_file() and sha(dest) == y['sha256']:
        return print(f'{dest} already present')
    req = urllib.request.Request(y['url'])
    if os.environ.get('HF_TOKEN'):
        req.add_header('Authorization', 'Bearer ' + os.environ['HF_TOKEN'])
    tmp = dest.with_suffix('.part')
    with urllib.request.urlopen(req) as r, open(tmp, 'wb') as f:
        shutil.copyfileobj(r, f, 1 << 24)
    if sha(tmp) != y['sha256']:
        tmp.unlink()
        raise SystemExit('sha256 mismatch after download')
    tmp.rename(dest)
    (root / 'weights' / '.downloaded_by_repro_check').write_text(y['filename'] + '\n')
    print(f'saved {dest}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('root', type=Path)
    ap.add_argument('--run', action='store_true')
    ap.add_argument('--input', type=Path, help='released test package root (for --run)')
    ap.add_argument('--output', type=Path)
    ap.add_argument('--mode', default='replay', choices=['replay', 'rebuild', 'video'])
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--download-weights', action='store_true')
    a = ap.parse_args()
    root = a.root.resolve()
    if a.download_weights:
        download(root)
    check_layout(root)
    check_repro_yaml(root)
    check_weights(root)
    check_signature(root)
    check_submission(root)
    if a.run:
        if a.input is None:
            ap.error('--run needs --input')
        run(root, a.input.resolve(), a.output or Path(tempfile.mkdtemp(prefix='tidegraph_')), a.mode, a.device)
    print(f'\n{"ALL CHECKS PASSED" if problems == 0 else f"{problems} PROBLEM(S)"}')
    sys.exit(1 if problems else 0)
