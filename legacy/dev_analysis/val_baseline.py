"""19226 post-processing on regenerated validation windows, scored on dev references.

Yaw exactly as team.py: wide = relative_window_yaw; flir-VIS = chain_and_edges +
Huber-3deg solve_yaw; both Gaussian sigma 2 s; FLIR yaw scaled by a gain.
Validation clips have no A/B land grouping, so FLIR gains are a sweep.
Speed variants (19226 wide speed cannot be rebuilt for dev clips):
  const  - speed 1 everywhere (19226 FLIR rule; deployable)
  oracle - reference speed (DIAGNOSTIC ONLY: isolates yaw error, not deployable)
Writes <out>/<variant>/predictions/*.json (starter format) and <out>/scores.json.

usage: ./.venv/bin/python tools/val_baseline.py --windows outputs/val_omega --out outputs/val_baseline
"""
import argparse,json,sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));import core
DEV=ROOT/'lookout-vo-dev-v0.1/release';sys.path.insert(0,str(DEV/'starter'))
import metrics
from public_dataset import load_manifest,load_reference,load_timestamps,validate_prediction

def raw_yaw(windows,key,mode,n,huber):
    if mode=='wide':return core.relative_window_yaw(windows,key,n),{}
    raw,i,j,z,w=core.chain_and_edges(windows,key,n);raw,d=core.solve_yaw(n,i,j,z,raw,delta_deg=huber)
    if not d['converged']:raise RuntimeError('solve_yaw did not converge: '+key)
    return raw,d

def oracle_speed(t,rt,rP):
    v=np.linalg.norm(np.diff(rP,axis=0),axis=1)/np.diff(rt)
    s=np.interp(t,.5*(rt[1:]+rt[:-1]),core.gaussian_filter1d(v,5))
    return s/max(s.mean(),1e-9)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--windows',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--gains',default='1.0,1.1,1.2');ap.add_argument('--sigma',type=float,default=2.);ap.add_argument('--huber',type=float,default=3.)
    a=ap.parse_args();gains=[float(g) for g in a.gains.split(',')]
    clips={c['key']:c for c in load_manifest(DEV/'release_manifest.json')['clips']};rows=[]
    for info in sorted((a.windows/'clips').glob('*.json')):
        m=json.loads(info.read_text());key=m['key'];mode=m['mode'];clip=clips[key]
        t=np.array(m['times']);n=len(t);rt,rP=load_reference(DEV,clip)
        raw,diag=raw_yaw(a.windows,key,mode,n,a.huber)
        y0=core.gaussian_filter1d(raw,a.sigma/np.median(np.diff(t)));y0=y0-y0[0]
        speeds=dict(const=np.ones(n),oracle=oracle_speed(t,rt,rP))
        row=dict(key=key,mode=mode,frames=n,turn_deg=float(np.rad2deg(y0[-1])),
                 yaw_range_deg=float(np.rad2deg(np.ptp(y0))),graph=diag,scores={})
        for g in (gains if mode=='flir-VIS' else [1.0]):
            yaw=g*y0
            for sname,s in speeds.items():
                P=core.integrate(t,yaw,s);variant=f'{sname}_g{g:.2f}'
                pred=dict(key=key,frame_indices=m['frame_indices'],times=m['times'],positions=P.tolist())
                validate_prediction(pred,key,load_timestamps(DEV,clip))
                d=a.out/variant/'predictions';d.mkdir(parents=True,exist_ok=True)
                (d/f'{key}.json').write_text(json.dumps(pred,separators=(',',':')))
                vo=metrics.score_trajectory(t,P,rt,rP)
                row['scores'][variant]=None if vo is None else float(vo['drift_pct'])
        rows.append(row)
        print(key,mode,'turn %+7.1f deg'%row['turn_deg'],' '.join(f'{k}={v:.2f}' for k,v in row['scores'].items() if v is not None),flush=True)
    summary={}
    for mode in sorted({r['mode'] for r in rows}):
        rs=[r for r in rows if r['mode']==mode]
        for v in sorted({k for r in rs for k in r['scores']}):
            vals=[r['scores'][v] for r in rs if r['scores'].get(v) is not None]
            summary[f'{mode}/{v}']=dict(clips=len(vals),mean=float(np.mean(vals)),median=float(np.median(vals)))
    a.out.mkdir(parents=True,exist_ok=True)
    (a.out/'scores.json').write_text(json.dumps(dict(rows=rows,summary=summary,windows=str(a.windows),
        sigma=a.sigma,huber=a.huber,note='const = deployable rule; oracle speed is diagnostic only'),indent=1))
    for k,v in summary.items():print(f'{k:28s} clips {v["clips"]:2d}  mean {v["mean"]:6.2f}%  median {v["median"]:6.2f}%')

if __name__=='__main__':main()
