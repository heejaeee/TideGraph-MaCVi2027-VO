"""Frozen single-image calibration; no trajectory references are accessed."""
import argparse, hashlib, json, subprocess, sys, time
from pathlib import Path
import cv2
import numpy as np
import torch

ROOT = Path(__import__('os').environ['TIDEGRAPH_RAW_WORK'])
sys.path.insert(0, str(ROOT / 'repos/lookout-vo-challenge/starter'))
from public_dataset import load_manifest, select_clips, load_timestamps, write_json
from anycalib import AnyCalib


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--keys', required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--dataset', type=Path, default=ROOT/'data/release/release_manifest.json')
    a = ap.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    cv2.setNumThreads(2)
    model = AnyCalib(model_id='anycalib_pinhole').eval().cuda()
    checkpoint = Path(torch.hub.get_dir()) / 'anycalib/anycalib_pinhole.pt'
    write_json(a.output / 'provenance.json', dict(
        model='anycalib_pinhole', repository='https://github.com/javrtg/AnyCalib',
        commit=subprocess.check_output(['git', '-C', str(ROOT/'repos/AnyCalib'), 'rev-parse', 'HEAD'], text=True).strip(),
        checkpoint=str(checkpoint), sha256=hashlib.file_digest(checkpoint.open('rb'), 'sha256').hexdigest()
        if hasattr(hashlib, 'file_digest') else hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        settings='9 uniformly spaced frames, pinhole fx/fy/cx/cy, coordinate-wise median; no labels'))
    data = a.dataset.parent
    for c in select_clips(load_manifest(a.dataset), clips=a.keys):
        dest = a.output / (c['key']+'.json')
        if dest.exists():
            continue
        started = time.perf_counter()
        ts = load_timestamps(data, c)
        ids = np.unique(np.linspace(0, len(ts)-1, 9).astype(int))
        cap = cv2.VideoCapture(str(data/c['video']))
        x,y,w,h = c['calibration']['active_rect']
        values = []
        wanted = set(ids)
        for idx in range(int(ids[-1])+1):
            if not cap.grab():
                raise RuntimeError((c['key'], int(idx)))
            if idx not in wanted:
                continue
            ok, bgr = cap.retrieve()
            if not ok:
                raise RuntimeError((c['key'], int(idx)))
            rgb = cv2.cvtColor(bgr[y:y+h,x:x+w], cv2.COLOR_BGR2RGB)
            im = torch.from_numpy(rgb.copy()).permute(2,0,1).float().cuda()/255.
            out = model.predict(im, cam_id='pinhole')
            values.append(out['intrinsics'].float().cpu().numpy())
            del out, im
        cap.release()
        values = np.array(values)
        if not np.isfinite(values).all() or not (values[:,:2]>0).all():
            raise ValueError('Invalid predicted intrinsics')
        write_json(dest, dict(key=c['key'], frame_indices=ids.tolist(),
            intrinsics_per_frame=values.tolist(), median=np.median(values,axis=0).tolist(),
            relative_focal_mad=(np.median(abs(values[:,:2]-np.median(values[:,:2],axis=0)),axis=0)/np.median(values[:,:2],axis=0)).tolist(),
            nominal=[c['calibration'][k] for k in ['fx','fy','cx','cy']],
            active_rect=[x,y,w,h], elapsed_seconds=time.perf_counter()-started))
        print('CALIBRATED', c['key'], np.median(values,axis=0).round(2).tolist(), flush=True)


if __name__ == '__main__':
    main()
