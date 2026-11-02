"""TideGraph inference entry point for the Maritime Visual Odometry final entry.

    main(model_dir, input_dir, output_dir, device, mode=None)

  model_dir   raw assets directory for 'raw'; Omega weights directory for 'video'
  input_dir   released test package root (release_manifest.json, timing/, videos/)
  output_dir  receives submission.json and validation reports
  device      CUDA required for raw inference; replay/rebuild run on CPU
  mode        'replay'  archived heading/speed -> final settings
              'rebuild' archived Omega windows -> yaw graph / gains / TV
              'video'   raw video -> Omega windows, with archived legacy signals
              'raw'     raw video and frozen weights -> all motion signals
              default: $TIDEGRAPH_MODE or 'replay'

Only 'raw' regenerates every motion input. See RAW_REPRODUCTION.md for its
verification status and environment requirements. Fresh GPU inference may differ
numerically from archived estimates. Inference is offline and does not train.
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))

CONFIG = ROOT / 'configs' / 'final_entry.json'
DATA = ROOT / 'data'
SUBMITTED = ROOT / 'submission' / 'submission.json'
MODES = ('replay', 'rebuild', 'video', 'raw')


def omega_stage(model_dir, input_dir, output_dir, device):
    from tidegraph import omega_windows
    weights = Path(model_dir) / omega_windows.WEIGHTS_NAME
    source = Path(os.environ.get('VGGT_OMEGA_SRC', ROOT / 'third_party' / 'vggt-omega'))
    if not weights.is_file():
        raise FileNotFoundError(f'{weights} missing: see weights/weights_link.yaml')
    if not (source / 'vggt_omega').is_dir():
        raise FileNotFoundError(f'{source} missing: run scripts/setup_vggt_omega.sh or set VGGT_OMEGA_SRC')
    out = Path(output_dir) / 'omega'
    omega_windows.generate(input_dir, out, weights, source, device)
    return out


def main(model_dir, input_dir, output_dir, device, mode=None, omega_dir=None, omega_only=False):
    import numpy as np
    mode = mode or os.environ.get('TIDEGRAPH_MODE', 'replay')
    if mode not in MODES:
        raise ValueError(f'mode must be one of {MODES}')
    if mode == 'raw':
        if not device.startswith('cuda'):
            raise ValueError('Full raw inference requires a CUDA GPU')
        import subprocess
        assets = Path(model_dir) / 'assets.json'
        subprocess.run([sys.executable, str(ROOT/'raw_run.py'), '--input', str(input_dir),
                        '--output', str(output_dir), '--assets', str(assets)], check=True)
        return json.loads((Path(output_dir)/'raw_report.json').read_text())
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tic = time.perf_counter()
    if mode == 'video' and omega_dir is None:
        omega_dir = omega_stage(model_dir, input_dir, output_dir, device)
        if omega_only:
            return None
    from tidegraph import pipeline
    cfg = pipeline.read(CONFIG)
    bundle, checks = pipeline.build(input_dir, DATA, cfg, 'replay' if mode == 'replay' else 'rebuild', omega_dir)
    encoded = pipeline.encode(bundle)
    (output_dir / 'submission.json').write_bytes(encoded)

    submitted = json.loads(SUBMITTED.read_text())
    parity = {k: float(np.max(np.abs(np.asarray(v['positions']) - np.asarray(submitted[k]['positions']))))
              for k, v in bundle.items()}
    sha = hashlib.sha256(encoded).hexdigest()
    report = dict(mode=mode, clips=len(bundle), sha256=sha, expected_sha256=cfg['expected_sha256'],
                  byte_identical=sha == cfg['expected_sha256'], max_position_diff_m=max(parity.values()),
                  per_clip_max_position_diff_m=parity, elapsed_seconds=time.perf_counter() - tic, checks=checks)
    if mode == 'video' or omega_dir is not None:
        keys = [c['key'] for c in checks if c['mode'] != 'flir-IR']
        report['omega_max_rotation_diff_deg_vs_data'] = pipeline.compare(omega_dir, DATA / 'omega', keys)
    (output_dir / 'report.json').write_text(json.dumps(report, indent=1) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k in
                      ('mode', 'clips', 'sha256', 'byte_identical', 'max_position_diff_m', 'elapsed_seconds')}, indent=1))
    return report


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--input', required=True, type=Path, help='released test package root')
    ap.add_argument('--output', required=True, type=Path)
    ap.add_argument('--mode', choices=MODES)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--model-dir', type=Path, default=ROOT / 'weights')
    ap.add_argument('--omega-dir', type=Path, help="rebuild from these windows instead of data/omega (e.g. <output>/omega)")
    ap.add_argument('--omega-only', action='store_true', help="mode=video: stop after writing <output>/omega")
    a = ap.parse_args()
    main(a.model_dir, a.input, a.output, a.device, a.mode, a.omega_dir, a.omega_only)
