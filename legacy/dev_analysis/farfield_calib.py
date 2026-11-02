#!/usr/bin/env python3
"""Far-field rotation and focal-length self-calibration from video.

Distant features (shoreline, buildings, horizon) obey the rotation-only model
x' ~ K R K^-1 x even while the boat translates, so fitting it over many frame
pairs gives the camera rotation per step and, from the perspective terms
(off-centre points shift more than central ones), the clip's focal length.

  calib --clips clip_075 ...          track + fit; writes outputs/farfield/<key>.npz
  eval  --set flir_vis_val            score yaw variants against dev references

Needs OpenCV (cv2). Kept out of the reproduction venv: put it on PYTHONPATH.
"""
import argparse,json,sys,time
from pathlib import Path
import numpy as np
from scipy.optimize import minimize_scalar

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'vendor/lookout_starter'))
import metrics

DEV=ROOT/'lookout-vo-dev-v0.1/release'
OUT=ROOT/'outputs/farfield'
HZ=4.            # trajectory sampling rate
LONG=4           # long pairs span LONG samples (1 s) for the focal fit
SCALE=.5         # decode at half resolution
F_GRID=np.linspace(.55,1.45,46)   # focal / nominal fx


def manifest():return {c['key']:c for c in json.loads((DEV/'release_manifest.json').read_text())['clips']}


# ---------------------------------------------------------------- tracking

def sample_frames(clip):
    import cv2
    times=np.asarray(json.loads((DEV/clip['timestamps']).read_text())['times'])
    grid=np.arange(times[0],times[-1]+1e-9,1/HZ)
    want=np.unique(np.searchsorted(times,grid).clip(0,len(times)-1))
    cap=cv2.VideoCapture(str(DEV/clip['video']));frames=[];ts=[];k=0;j=0
    while j<len(want):
        ok=cap.grab()
        if not ok:break
        if k==want[j]:
            _,img=cap.retrieve();g=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            frames.append(cv2.resize(g,None,fx=SCALE,fy=SCALE,interpolation=cv2.INTER_AREA));ts.append(times[k]);j+=1
        k+=1
    cap.release();return np.asarray(ts),frames


def track(a,b):
    """Corners in a tracked into b, kept only if forward-backward consistent."""
    import cv2
    p=cv2.goodFeaturesToTrack(a,maxCorners=800,qualityLevel=.005,minDistance=7,blockSize=7)
    if p is None or len(p)<8:return np.zeros((0,2)),np.zeros((0,2))
    lk=dict(winSize=(21,21),maxLevel=4,criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,30,.01))
    q,s1,_=cv2.calcOpticalFlowPyrLK(a,b,p,None,**lk);r,s2,_=cv2.calcOpticalFlowPyrLK(b,a,q,None,**lk)
    ok=(s1[:,0]==1)&(s2[:,0]==1)&(np.linalg.norm((p-r)[:,0],axis=1)<.5)
    return p[ok,0].astype(float),q[ok,0].astype(float)


def static_cells(pairs,shape,cell=16,still=.15,frac=.7):
    """Image cells whose features almost never move: overlays, the own bow, rails."""
    h,w=shape;H,W=h//cell+1,w//cell+1;tot=np.zeros((H,W));zero=np.zeros((H,W))
    for p,q in pairs:
        if not len(p):continue
        i=(p[:,1]//cell).astype(int);j=(p[:,0]//cell).astype(int);np.add.at(tot,(i,j),1)
        np.add.at(zero,(i,j),(np.linalg.norm(q-p,axis=1)<still).astype(float))
    return (tot>=15)&(zero>=frac*np.maximum(tot,1))


def keep(p,q,mask,cell=16):
    if not len(p):return p,q
    ok=~mask[(p[:,1]//cell).astype(int),(p[:,0]//cell).astype(int)];return p[ok],q[ok]


# ---------------------------------------------------------------- rotation geometry

def rays(p,f,c):
    v=np.c_[(p[:,0]-c[0])/f,(p[:,1]-c[1])/f,np.ones(len(p))];return v/np.linalg.norm(v,axis=1,keepdims=True)


def kabsch(a,b,w=None):
    """R minimising sum w|b - R a|^2 over unit rays."""
    w=np.ones(len(a)) if w is None else w;U,_,Vt=np.linalg.svd((a*w[:,None]).T@b)
    D=np.diag([1,1,np.sign(np.linalg.det(Vt.T@U.T))]);return Vt.T@D@U.T


def project(v,f,c):return np.c_[f*v[:,0]/v[:,2]+c[0],f*v[:,1]/v[:,2]+c[1]]


def residual(p,q,R,f,c):
    v=rays(p,f,c)@R.T;r=np.full(len(p),1e3);ok=v[:,2]>.1
    r[ok]=np.linalg.norm(project(v[ok],f,c)-q[ok],axis=1);return r


def ransac_rotation(p,q,f,c,thr,iters=150,rng=np.random.default_rng(0)):
    if len(p)<6:return np.eye(3),np.zeros(len(p),bool)
    a,b=rays(p,f,c),rays(q,f,c);best=np.zeros(len(p),bool)
    for _ in range(iters):
        idx=rng.choice(len(p),2,replace=False)
        if abs(a[idx[0]]@a[idx[1]])>.99999:continue
        inl=residual(p,q,kabsch(a[idx],b[idx]),f,c)<thr
        if inl.sum()>best.sum():best=inl
    R=kabsch(a[best],b[best]) if best.sum()>=3 else np.eye(3)
    best=residual(p,q,R,f,c)<thr
    return (kabsch(a[best],b[best]) if best.sum()>=3 else R),best


def fit_rotation(p,q,f,c,tau=2.):
    """Rotation at focal f: Kabsch, re-weighted to the points it explains (two passes)."""
    if len(p)<6:return np.eye(3),np.inf
    a,b=rays(p,f,c),rays(q,f,c);R=kabsch(a,b)
    for _ in range(3):
        r=residual(p,q,R,f,c);inl=r<3*tau
        if inl.sum()<6:break
        R=kabsch(a[inl],b[inl])
    r=residual(p,q,R,f,c);return R,float(np.sum(np.minimum(r,tau)**2))


def angle(R):return float(np.degrees(np.arccos(np.clip((np.trace(R)-1)/2,-1,1))))


# ---------------------------------------------------------------- focal estimate

def focal_curve(long_pairs,f0,c,min_deg=.4):
    """Truncated reprojection loss per focal, summed over 1-s pairs that rotate enough."""
    use=[];
    for p,q in long_pairs:
        R,_=ransac_rotation(p,q,f0,c,thr=3.)
        if angle(R)>=min_deg:use.append((p,q))
    if len(use)<10:return None
    L=np.array([[fit_rotation(p,q,s*f0,c)[1] for p,q in use] for s in F_GRID])  # grid x pairs
    return L,len(use)


def best_scale(L):
    tot=L.sum(1);k=int(np.argmin(tot))
    if 0<k<len(F_GRID)-1:  # parabola through the minimum and its neighbours
        x=F_GRID[k-1:k+2];y=tot[k-1:k+2];a,b,_=np.polyfit(x,y,2)
        if a>0:return float(np.clip(-b/(2*a),x[0],x[-1]))
    return float(F_GRID[k])


def calib(key,clips):
    clip=clips[key];K=clip['calibration'];f0=K['fx']*SCALE;c=(K['cx']*SCALE,K['cy']*SCALE)
    tic=time.perf_counter();ts,frames=sample_frames(clip)
    step=[track(frames[i],frames[i+1]) for i in range(len(frames)-1)]
    longp=[track(frames[i],frames[i+LONG]) for i in range(0,len(frames)-LONG,LONG)]
    mask=static_cells(step,frames[0].shape)
    step=[keep(p,q,mask) for p,q in step];longp=[keep(p,q,mask) for p,q in longp]
    fc=focal_curve(longp,f0,c)
    if fc is None:s_hat=np.nan;ci=(np.nan,np.nan);n_used=0;curve=np.full(len(F_GRID),np.nan)
    else:
        L,n_used=fc;s_hat=best_scale(L);curve=L.sum(1)
        rng=np.random.default_rng(1);boots=[best_scale(L[:,rng.integers(0,L.shape[1],L.shape[1])]) for _ in range(200)]
        ci=tuple(np.percentile(boots,[5,95]))
    inliers=[];
    for p,q in step:
        _,inl=ransac_rotation(p,q,f0,c,thr=1.5);inliers.append(inl)
    np.savez_compressed(OUT/f'{key}.npz',times=ts,f0=f0,cx=c[0],cy=c[1],scale_hat=s_hat,ci=np.asarray(ci),curve=curve,grid=F_GRID,
        n_long=n_used,mask=mask,
        **{f'p{i}':p for i,(p,q) in enumerate(step)},**{f'q{i}':q for i,(p,q) in enumerate(step)},
        **{f'i{i}':m for i,m in enumerate(inliers)},n_steps=len(step),
        **{f'lp{i}':p for i,(p,q) in enumerate(longp)},**{f'lq{i}':q for i,(p,q) in enumerate(longp)},n_longp=len(longp))
    info=dict(key=key,mode=clip['camera_mode'],f_nominal=K['fx'],fy_nominal=K['fy'],scale_hat=s_hat,ci=list(ci),f_hat=s_hat*K['fx'],
              long_pairs_used=n_used,static_cells=int(mask.sum()),median_tracks=float(np.median([len(p) for p,_ in step])),
              median_inlier_frac=float(np.median([m.mean() if len(m) else 0 for m in inliers])),seconds=round(time.perf_counter()-tic,1))
    print(json.dumps(info),flush=True);return info


# ---------------------------------------------------------------- yaw + evaluation

def yaw_series(z,scale):
    """Camera yaw (rad, clockwise from initial forward) by chaining step rotations at focal scale*f0."""
    f=float(z['f0'])*scale;c=(float(z['cx']),float(z['cy']));C=np.eye(3);yaw=[0.]
    for i in range(int(z['n_steps'])):
        p,q,inl=z[f'p{i}'],z[f'q{i}'],z[f'i{i}']
        if inl.sum()>=6:
            R,_=fit_rotation(p[inl],q[inl],f,c);C=R@C
        fw=C.T@np.array([0.,0.,1.]);yaw.append(np.arctan2(fw[0],fw[2]))
    return np.unwrap(np.asarray(yaw))


def yaw_series_essential(z):
    """Yaw from full two-view geometry (rotation + translation) on the 1-s pairs, chained at 1 Hz.

    Fitting translation explicitly keeps sideways parallax (camera not looking along the
    direction of travel) out of the rotation, unlike the rotation-only fit.
    """
    import cv2
    f=float(z['f0']);K=np.array([[f,0,float(z['cx'])],[0,f,float(z['cy'])],[0,0,1.]]);C=np.eye(3);yaw=[0.]
    for i in range(int(z['n_longp'])):
        p,q=z[f'lp{i}'],z[f'lq{i}']
        if len(p)>=12:
            E,m=cv2.findEssentialMat(p,q,K,method=cv2.RANSAC,prob=.999,threshold=1.)
            if E is not None and E.shape==(3,3):
                _,R,_,_=cv2.recoverPose(E,p,q,K,mask=m);C=R@C
        fw=C.T@np.array([0.,0.,1.]);yaw.append(np.arctan2(fw[0],fw[2]))
    t=z['times'][::LONG][:len(yaw)];return t,np.unwrap(np.asarray(yaw))


def integrate(t,yaw):
    v=np.c_[np.sin(yaw),np.zeros(len(yaw)),np.cos(yaw)]
    return np.vstack([np.zeros(3),np.cumsum(.5*(v[:-1]+v[1:])*np.diff(t)[:,None],axis=0)])


def drift(t,yaw,rt,rP):
    out=metrics.score_trajectory(t,integrate(t,yaw),rt,rP);return out['drift_pct'] if out else np.nan


def evaluate(keys,clips):
    rows=[]
    for key in keys:
        f=OUT/f'{key}.npz'
        if not f.exists():print('missing',key);continue
        z=np.load(f);t=z['times'];ref=json.loads((DEV/clips[key]['reference']).read_text())
        rt=np.asarray(ref['times']);rP=np.asarray(ref['positions'],float);s_hat=float(z['scale_hat'])
        y0=yaw_series(z,1.)                       # nominal focal
        a=y0-y0[0]
        best=minimize_scalar(lambda g:drift(t,g*a,rt,rP),bounds=(.05,3.),method='bounded')
        rate=minimize_scalar(lambda b:drift(t,a-b*(t-t[0]),rt,rP),bounds=(-.02,.02),method='bounded')  # rad/s
        te,ye=yaw_series_essential(z) if 'n_longp' in z else (None,None)
        row=dict(key=key,mode=clips[key]['camera_mode'],scale_hat=s_hat,ci=z['ci'].tolist(),
                 drift_essential=(drift(te,ye-ye[0],rt,rP) if te is not None else np.nan),
                 oracle_rate_degmin=float(np.degrees(rate.x)*60),drift_oracle_rate=float(rate.fun),
                 drift_nominal=drift(t,a,rt,rP),
                 drift_gain115=drift(t,1.15*a,rt,rP),
                 drift_selfcal=(drift(t,(lambda y:y-y[0])(yaw_series(z,s_hat)),rt,rP) if np.isfinite(s_hat) else np.nan),
                 oracle_gain=float(best.x),drift_oracle_gain=float(best.fun),
                 drift_mirrored=drift(t,-a,rt,rP))
        rows.append(row);print(json.dumps({k:(round(v,3) if isinstance(v,float) else v) for k,v in row.items()}),flush=True)
    return rows


def main():
    ap=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter);sub=ap.add_subparsers(dest='cmd',required=True)
    c=sub.add_parser('calib');c.add_argument('--clips',nargs='*');c.add_argument('--set')
    e=sub.add_parser('eval');e.add_argument('--clips',nargs='*');e.add_argument('--set')
    a=ap.parse_args();clips=manifest();OUT.mkdir(parents=True,exist_ok=True)
    keys=a.clips or json.loads((ROOT/'configs/dev_split.json').read_text())[a.set]
    if a.cmd=='calib':
        res=[calib(k,clips) for k in keys];(OUT/f'calib_{a.set or "clips"}.json').write_text(json.dumps(res,indent=1))
    else:
        rows=evaluate(keys,clips);(OUT/f'eval_{a.set or "clips"}.json').write_text(json.dumps(rows,indent=1))
        if rows:
            for k in ('drift_nominal','drift_gain115','drift_selfcal','drift_essential','drift_oracle_gain','drift_oracle_rate'):
                v=[r[k] for r in rows if np.isfinite(r[k])];print(f'{k:18s} mean {np.mean(v):6.2f}%  (n={len(v)})')


if __name__=='__main__':main()
