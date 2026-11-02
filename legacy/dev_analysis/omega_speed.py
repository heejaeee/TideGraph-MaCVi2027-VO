"""Relative speed series from Omega window translations (CPU, existing windows only).

Each W16 window has its own arbitrary scale. Per-step camera-centre distances
|C[k+1]-C[k]| are chained across windows: each new window is rescaled by the
median ratio of its steps to the running estimate on the 7 shared steps, then
overlapping estimates are combined by median. Output speed = step/dt, mean 1.
Writes <out>/<key>.json: key, mode, times (step mid-times), speed, step_windows.

usage: ./.venv/bin/python tools/omega_speed.py --windows snapshot/omega --data test --out outputs/omega_speed/test
       ./.venv/bin/python tools/omega_speed.py --windows outputs/val_omega --data dev --out outputs/omega_speed/dev
"""
import argparse,json,sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
DATA={'test':ROOT/'lookout-vo-test-v0.1','dev':ROOT/'lookout-vo-dev-v0.1/release'}

def chained_steps(win_dir,n):
    est=[[] for _ in range(n-1)]
    for p in sorted(win_dir.glob('*.npz')):
        d=np.load(p);s=int(d['start']);C=d['c2w'][:,:3,3];step=np.linalg.norm(np.diff(C,axis=0),axis=1)
        prev=[np.median(est[s+k]) if est[s+k] else np.nan for k in range(len(step))]
        r=np.array(prev)/np.maximum(step,1e-12);r=r[np.isfinite(r)]
        scale=float(np.median(r)) if len(r) else 1.
        for k,v in enumerate(step):est[s+k].append(v*scale)
    if any(not e for e in est):raise ValueError('incomplete windows: '+str(win_dir))
    return np.array([np.median(e) for e in est]),np.array([len(e) for e in est])

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--windows',type=Path,required=True);ap.add_argument('--data',choices=list(DATA),required=True)
    ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    sys.path.insert(0,str(DATA['dev']/'starter'))
    from public_dataset import load_manifest,load_timestamps
    frame_ids=lambda ts:np.unique(np.r_[np.searchsorted(ts,np.arange(ts[0],ts[-1],.5)),len(ts)-1])  # = omega_windows.frame_ids
    root=DATA[a.data];man=root/'release_manifest.json';clips={c['key']:c for c in load_manifest(man)['clips']}
    a.out.mkdir(parents=True,exist_ok=True)
    for wd in sorted((a.windows/'windows').iterdir()):
        clip=clips[wd.name];t=load_timestamps(root,clip)[frame_ids(load_timestamps(root,clip))]
        step,cover=chained_steps(wd,len(t));v=step/np.diff(t);v=v/v.mean()
        rec=dict(key=wd.name,mode=clip['camera_mode'],times=(.5*(t[1:]+t[:-1])).tolist(),speed=v.tolist(),step_windows=cover.tolist())
        (a.out/f'{wd.name}.json').write_text(json.dumps(rec))
        print(wd.name,clip['camera_mode'],'steps',len(v),'cv %.2f'%v.std(),flush=True)

if __name__=='__main__':main()
