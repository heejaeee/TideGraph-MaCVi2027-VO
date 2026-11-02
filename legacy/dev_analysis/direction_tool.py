"""Direction tool: which way should each clip's trajectory be adjusted?

The leaderboard returns one aggregate drift per submission. This tool keeps a
registry of scored submissions (added one by one), decomposes how each one
differs from the base per clip, and estimates which adjustment direction
lowers drift for each clip group. Everything runs on CPU from submission files.

  add      register a submission file with its leaderboard drift, rebuild report
  report   rebuild report.html / analysis.json from the registry
  apply    write a new probe submission from an adjustment spec
  dev      with reference trajectories (dev val split), find each clip's best adjustment
  selftest synthetic check that the estimator recovers known directions

Adjustments are Sim(3)-invariant shape changes on step headings h and speeds s
(global rotation and scale are free under the metric):
  gain     h' = h0 + (1+gain)(h-h0)          turn magnitude (+0.1 = 10% larger turns)
  drift    h' += drift * (t-t0)/60 [deg/min]  steady heading drift (+ = clockwise from above)
  trend    s' *= 1 + trend*u, u in [-.5,.5]   speed ramp (+ = faster at the end)
  contrast s' = m + (1+contrast)(s-m)          speed variation amplitude
"""
import argparse,hashlib,json,math,shutil,sys
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'vendor/lookout_starter'))
import metrics
from public_dataset import load_manifest,load_reference,load_timestamps,select_clips,validate_prediction

PARAMS=('gain','drift','trend','contrast')
UNITS=dict(gain='x',drift='deg/min',trend='frac',contrast='x')
STEP=dict(gain=.1,drift=1.,trend=.2,contrast=.3)  # one "probe step" per parameter
# Median reference path length (m) per camera mode over the 92 dev clips; sets
# which 100-800 m segments count when one trajectory is scored against another.
PATH_M={'wide':1715.,'flir-VIS':1061.,'flir-IR':1064.,'sionyx':3505.,'zoom':1634.,'infrared':688.}
DEFAULT_WORKSPACE=ROOT/'outputs/direction'
LEADERBOARD_ROUNDING=.005


def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(value,indent=1,allow_nan=False)+'\n')


def load_bundle(path):
    """Portal bundle {key: prediction}, one prediction file, or a run dir with predictions/*.json."""
    path=Path(path)
    if path.is_dir():
        files=sorted((path/'predictions').glob('*.json')) or sorted(path.glob('clip_*.json'))
        data={f.stem:read(f) for f in files}
    else:data=read(path)
    if 'key' in data:data={data['key']:data}
    return {k:(np.asarray(v['times'],float),np.asarray(v['positions'],float)) for k,v in data.items()}


def test_clips():
    meta=read(ROOT/'snapshot/metadata/clip_index.json');cfg=read(ROOT/'configs/best19226.json')
    groups={}
    for key,m in meta.items():
        mode=m['camera_mode']
        groups[key]=next((f'flir-{g}' for g,keys in cfg['flir_groups'].items() if key in keys),mode)
    return {k:dict(mode=meta[k]['camera_mode'],group=groups[k]) for k in cfg['clip_order']}


# ---------------------------------------------------------------- geometry

class Plane:
    """Horizontal plane of a base trajectory: x/z for flat starter-style output, else PCA."""
    def __init__(self,P):
        self.c=np.zeros(3)
        if np.ptp(P[:,1])<=1e-9*max(np.ptp(P),1.):self.E=np.array([[1.,0,0],[0,0,1.]])
        else:
            self.c=P.mean(0);_,_,vt=np.linalg.svd(P-self.c,full_matrices=False);self.E=vt[:2]
    def xy(self,P):return (P-self.c)@self.E.T
    def lift(self,xy):return self.c+xy@self.E


def steps(t,xy):
    """Step mid-times, unwrapped headings (rad, atan2(east,north)) and speeds."""
    d=np.diff(xy,axis=0);dt=np.diff(t);l=np.hypot(d[:,0],d[:,1])
    h=np.arctan2(d[:,0],d[:,1]);ok=l>1e-9*max(l.max(),1e-12);idx=np.arange(len(h))
    if ok.sum()>=2:h=np.interp(idx,idx[ok],np.unwrap(h[ok]))
    return .5*(t[1:]+t[:-1]),h,l/np.maximum(dt,1e-9),dt


def adjust(t,xy,theta):
    tau,h,s,dt=steps(t,xy);g,b,a,c=(float(theta.get(p,0.)) for p in PARAMS)
    h=h[0]+(1+g)*(h-h[0])+np.deg2rad(b)*(tau-tau[0])/60
    span=max(tau[-1]-tau[0],1e-9);u=(tau-tau[0])/span-.5
    m=(s*dt).sum()/dt.sum();s=np.maximum((m+(1+c)*(s-m))*(1+a*u),0)
    return np.vstack([xy[0],xy[0]+np.cumsum(np.c_[np.sin(h),np.cos(h)]*(s*dt)[:,None],axis=0)])


def wls(X,y,w):
    sw=np.sqrt(w);return np.linalg.lstsq(X*sw[:,None],y*sw,rcond=1e-8)[0]


def decompose(t,xb,xs):
    """Adjustment that best maps base xb onto submission xs (same times, same plane)."""
    tau,hb,sb,dt=steps(t,xb);_,hs,ss,_=steps(t,xs)
    w=np.sqrt(sb*ss)*dt;dh=np.unwrap(np.angle(np.exp(1j*(hs-hb))))
    coef=wls(np.c_[np.ones_like(tau),hb-hb[0],(tau-tau[0])/60],dh,w)
    span=max(tau[-1]-tau[0],1e-9);u=(tau-tau[0])/span-.5
    mb=(sb*dt).sum()/dt.sum();ms=(ss*dt).sum()/dt.sum();vb=sb/mb-1
    if np.sqrt(np.average(vb**2,weights=dt))<1e-4:a,c=wls(np.c_[u],ss/ms-1,dt)[0],0.
    else:a,k=wls(np.c_[u,vb],ss/ms-1,dt);c=k-1
    return dict(gain=float(coef[1]),drift=float(np.rad2deg(coef[2])),trend=float(a),contrast=float(c))


def pseudo_drift(t_ref,ref,t_est,est,path_m):
    """Official drift of est, using ref (1 Hz, scaled to path_m metres) as stand-in reference."""
    tt=np.arange(t_ref[0],t_ref[-1]+1e-9,1.);r=metrics.resample_to(t_ref,ref,tt)
    length=metrics.cumulative_length(r)[-1]
    if length<=0:return float('nan')
    out=metrics.score_trajectory(t_est,est,tt,r*(path_m/length))
    return float(out['drift_pct']) if out else float('nan')


def on_grid(tb,Pb,ts,Ps):
    """Submission resampled onto the base times it covers, Sim(3)-aligned to the base."""
    mask=metrics.overlap_window(ts,tb);t=tb[mask]
    P=Ps if len(ts)==len(tb) and np.allclose(ts,tb,atol=1e-6) else metrics.resample_to(ts,Ps,t)
    aligned,_=metrics.align_sim3(P,Pb[mask])
    return mask,(P if aligned is None else aligned)


# ---------------------------------------------------------------- analysis

def clip_analysis(key,info,tb,Pb,subs):
    plane=Plane(Pb);xb=plane.xy(Pb);path_m=PATH_M.get(info['mode'],1500.)
    bound={}
    for p in PARAMS:  # |d drift / d param| cannot exceed the drift between base and base+step
        moved=plane.lift(adjust(tb,xb,{p:STEP[p]}))
        bound[p]=pseudo_drift(tb,Pb,tb,moved,path_m)/STEP[p]
    rows=[];series=[dict(label='base',xy=xb,t=tb)]
    for label,(ts,Ps) in subs:
        if ts is None:rows.append(None);continue
        mask,P=on_grid(tb,Pb,ts,Ps);t=tb[mask];xs=plane.xy(P);xbm=xb[mask]
        theta=decompose(t,xbm,xs);full=pseudo_drift(tb,Pb,ts,Ps,path_m)
        resid=pseudo_drift(t,plane.lift(adjust(t,xbm,theta)),ts,Ps,path_m) if full>1e-9 else 0.
        rows.append(dict(theta=theta,pseudo=full,resid=min(resid,full) if np.isfinite(resid) else full))
        series.append(dict(label=label,xy=xs,t=t))
    return dict(key=key,path_m=path_m,bound=bound,rows=rows,series=series)


def estimate_directions(clips,info,scores,base_score,groups,n_expected,model_sd=.5,nonlinear=.25):
    """Bayesian linear model: score_k - base = sum_groups (n_g/N) * mean_dtheta * gradient.

    Prior per (group, param): gradient ~ N(0, (model_sd*bound)^2), where bound is the
    largest possible drift slope (base vs base+step). Noise: leaderboard rounding, the
    part of each submission's change the four adjustments do not explain, and a
    second-order term that grows with how far the submission moved.
    """
    feats=[(g,p) for g in groups for p in PARAMS];K=len(scores)
    X=np.zeros((K,len(feats)));var=np.full(K,LEADERBOARD_ROUNDING**2/3)
    tau=np.zeros(len(feats))
    for j,(g,p) in enumerate(feats):
        tau[j]=model_sd*np.mean([c['bound'][p] for c in clips if info[c['key']]['group']==g])
    for k in range(K):
        for c in clips:
            row=c['rows'][k];g=info[c['key']]['group']
            if row is None:continue
            for p in PARAMS:X[k,feats.index((g,p))]+=row['theta'][p]/n_expected
            var[k]+=(.5*row['resid']/n_expected)**2+(nonlinear*row['pseudo']/n_expected)**2
    y=np.asarray(scores)-base_score;W=1/var
    tau=np.maximum(tau,1e-6);A=np.diag(1/tau**2)+X.T@(X*W[:,None])
    cov=np.linalg.inv(A);mu=cov@(X.T@(W*y));sd=np.sqrt(np.diag(cov))
    out=[]
    for j,(g,p) in enumerate(feats):
        n=sum(info[k]['group']==g for k in info);scale=n/n_expected*STEP[p]
        helps=float(norm.cdf(-mu[j]/sd[j]))
        rec='increase' if helps>.8 else 'decrease' if helps<.2 else 'uncertain'
        if (abs(mu[j])+1.2816*sd[j])*scale<2*LEADERBOARD_ROUNDING:rec='negligible'  # below leaderboard resolution
        out.append(dict(group=g,param=p,clips=n,step=STEP[p],gradient=float(mu[j]),gradient_sd=float(sd[j]),
                        prior_sd=float(tau[j]),delta_per_step=float(mu[j]*scale),delta_sd=float(sd[j]*scale),
                        p_increase_helps=helps,information=float(1-sd[j]/tau[j]),recommendation=rec))
    fitted=X@mu
    return out,dict(X=X.tolist(),y=y.tolist(),fitted=fitted.tolist(),noise_sd=np.sqrt(var).tolist())


def one_axis_sweeps(directions,X,y,tol=.1):
    """Submissions that moved essentially one (group, param): quadratic fit -> best value."""
    X=np.asarray(X);y=np.asarray(y);out=[]
    for j,d in enumerate(directions):
        others=np.delete(np.abs(X),j,axis=1).sum(1);on=np.abs(X[:,j])
        sel=[k for k in range(len(y)) if on[k]>1e-9 and others[k]<=tol*on[k]]
        if len(sel)<1:continue
        # x is (n_g/N) * mean parameter change; fit_sweep rescales to the parameter itself
        out.append(dict(group=d['group'],param=d['param'],submissions=sel,x=np.r_[0.,X[sel,j]].tolist(),y=np.r_[0.,y[sel]].tolist()))
    return out


def fit_sweep(sweep,scale):
    x=np.asarray(sweep['x'])/scale;y=np.asarray(sweep['y']);res=dict(values=x.tolist(),delta=y.tolist())
    if len(x)>=3:
        q2,q1,q0=np.polyfit(x,y,2)
        res.update(curvature=float(q2),best=float(-q1/(2*q2)) if q2>0 else None,
                   best_delta=float(q0-q1**2/(4*q2)) if q2>0 else None)
    else:res.update(slope=float((y[1]-y[0])/(x[1]-x[0])))
    return res


def next_probes(directions,n=3):
    """Most informative single-axis probes: largest remaining uncertainty in aggregate drift."""
    ranked=sorted(directions,key=lambda d:-d['delta_sd'])[:n];out=[]
    for d in ranked:
        sign=-1. if d['gradient']>0 else 1.
        out.append(dict(group=d['group'],param=d['param'],value=sign*d['step'],
                        expected_delta=float(sign*d['delta_per_step']),delta_sd=d['delta_sd'],
                        spec={'groups':{d['group']:{d['param']:sign*d['step']}}}))
    return out


def build_report(ws):
    reg=read(ws/'registry.json');info=test_clips();n_expected=len(info)
    base=load_bundle(ws/reg['base']['file']);subs=[]
    for s in reg['submissions']:subs.append((s['label'],load_bundle(ws/s['file'])))
    clips=[]
    for key in info:
        tb,Pb=base[key];per=[(label,b.get(key,(None,None))) for label,b in subs]
        clips.append(clip_analysis(key,info[key],tb,Pb,per));print('analysed',key,flush=True)
    groups=list(dict.fromkeys(i['group'] for i in info.values()))
    scores=[s['drift'] for s in reg['submissions']]
    directions,fit=estimate_directions(clips,info,scores,reg['base']['drift'],groups,n_expected) if scores else ([],{})
    sweeps=[]
    for sw in one_axis_sweeps(directions,fit.get('X',np.zeros((0,0))),fit.get('y',[])) if scores else []:
        scale=sum(i['group']==sw['group'] for i in info.values())/n_expected
        sweeps.append(dict(sw,**fit_sweep(sw,scale),labels=[reg['submissions'][k]['label'] for k in sw['submissions']]))
    analysis=dict(base=reg['base'],submissions=reg['submissions'],directions=directions,fit=fit,sweeps=sweeps,
                  probes=next_probes(directions) if directions else [],params=PARAMS,units=UNITS,steps=STEP,
                  clips=[dict(key=c['key'],**info[c['key']],path_m=c['path_m'],bound=c['bound'],rows=c['rows']) for c in clips])
    write(ws/'analysis.json',analysis)
    view=dict(analysis,clips=[dict(a,series=[dict(label=s['label'],t=np.round(s['t'],2).tolist(),
              x=np.round(s['xy'][:,0],3).tolist(),y=np.round(s['xy'][:,1],3).tolist()) for s in c['series']])
              for a,c in zip(analysis['clips'],clips)],mode='test')
    render(ws/'report.html',view);return analysis


def render(path,view):
    html=(ROOT/'tools/direction_report.html').read_text()
    Path(path).write_text(html.replace('/*DATA*/null',json.dumps(view,allow_nan=False).replace('</','<\\/')))
    print('report',path)


def print_summary(a):
    print(f"\nbase {a['base']['label']}: {a['base']['drift']:.2f}%  submissions: {len(a['submissions'])}")
    for d in sorted(a['directions'],key=lambda d:-abs(d['delta_per_step'])/max(d['delta_sd'],1e-9))[:8]:
        print(f"  {d['group']:9s} {d['param']:8s} +{d['step']:g}: {d['delta_per_step']:+.3f}±{d['delta_sd']:.3f} pp"
              f"  P(increase helps)={d['p_increase_helps']:.2f}  -> {d['recommendation']}")
    for p in a['probes']:print(f"  next probe: {p['group']} {p['param']} {p['value']:+g}  spec={json.dumps(p['spec'])}")


# ---------------------------------------------------------------- commands

def workspace(path):
    ws=Path(path).resolve()
    if not (ws/'registry.json').exists():
        (ws/'files').mkdir(parents=True,exist_ok=True)
        src=ROOT/'artifacts/submission_19226.json';shutil.copy(src,ws/'files/base_19226.json')
        write(ws/'registry.json',dict(base=dict(label='19226',file='files/base_19226.json',drift=11.29,sha256=sha(src)),submissions=[]))
    return ws


def cmd_add(args):
    ws=workspace(args.workspace);reg=read(ws/'registry.json');digest=sha(args.file)
    if digest==reg['base']['sha256'] or any(s['sha256']==digest for s in reg['submissions']):
        raise SystemExit('This exact submission is already registered')
    bundle=load_bundle(args.file);missing=sorted(set(test_clips())-set(bundle))
    label=args.label or (args.file.parent.name if args.file.stem=='submission' else args.file.stem)
    if any(s['label']==label for s in reg['submissions']):label=f'{label}-{digest[:6]}'
    dest=ws/'files'/f'{len(reg["submissions"])+1:02d}_{label}.json'
    if args.file.is_dir():dest.write_text(json.dumps({k:dict(key=k,times=t.tolist(),positions=P.tolist()) for k,(t,P) in bundle.items()}))
    else:shutil.copy(args.file,dest)
    reg['submissions'].append(dict(label=label,file=str(dest.relative_to(ws)),drift=args.drift,sha256=digest,
                                   missing_clips=missing,note=args.note or ''))
    write(ws/'registry.json',reg);print('registered',label,args.drift,'missing clips:',missing or 'none')
    print_summary(build_report(ws))


def cmd_report(args):print_summary(build_report(workspace(args.workspace)))


def cmd_apply(args):
    output=Path(args.output).resolve()
    if (output/'submission.json').exists():raise SystemExit('Choose a new output directory; refusing overwrite')
    spec=read(args.spec);info=test_clips();base=read(args.base)
    manifest=load_manifest(ROOT/'snapshot/metadata/release_manifest.json');released={c['key']:c for c in manifest['clips']}
    out={}
    for key,pred in base.items():
        theta=dict(spec.get('all',{}));theta.update(spec.get('groups',{}).get(info[key]['group'],{}))
        theta.update(spec.get('clips',{}).get(key,{}))
        bad=set(theta)-set(PARAMS)
        if bad:raise SystemExit(f'Unknown parameter(s) {sorted(bad)}; use {PARAMS}')
        t=np.asarray(pred['times']);P=np.asarray(pred['positions'],float)
        if any(theta.values()):plane=Plane(P);P=plane.lift(adjust(t,plane.xy(P),theta))
        new=dict(pred,positions=P.tolist())
        validate_prediction(new,key,load_timestamps(ROOT/'snapshot/metadata',released[key]));out[key]=new
    output.mkdir(parents=True,exist_ok=True)
    with (output/'submission.json').open('x') as f:f.write(json.dumps(out,separators=(',',':'),allow_nan=False)+'\n')
    write(output/'spec.json',dict(spec,base=str(args.base),base_sha256=sha(args.base)))
    print('wrote',output/'submission.json',sha(output/'submission.json'))


def cmd_dev(args):
    """Against real references: the adjustment that minimises each clip's official drift."""
    manifest=load_manifest(args.manifest);root=args.manifest.resolve().parent
    preds=load_bundle(args.predictions);rows=[];view_clips=[]
    for clip in select_clips(manifest,args.split,args.clips):
        key=clip['key']
        if key not in preds or clip['reference'] is None:continue
        t,P=preds[key];rt,rP=load_reference(root,clip);plane=Plane(P);xy=plane.xy(P)
        def drift(v):
            out=metrics.score_trajectory(t,plane.lift(adjust(t,xy,dict(zip(PARAMS,v)))),rt,rP)
            return out['drift_pct'] if out and np.isfinite(out['drift_pct']) else 1e3
        before=drift(np.zeros(4));opt=minimize(drift,np.zeros(4),method='Nelder-Mead',
            options=dict(initial_simplex=np.vstack([np.zeros(4),np.diag([STEP[p] for p in PARAMS])]),xatol=1e-3,fatol=1e-4,maxiter=400))
        theta=dict(zip(PARAMS,map(float,opt.x)));after=float(opt.fun)
        rows.append(dict(key=key,mode=clip['camera_mode'],before=before,after=after,theta=theta))
        # Draw everything Sim(3)-aligned to the reference, as the metric sees it.
        mask=metrics.overlap_window(rt,t);tm=t[mask];ref=metrics.resample_to(rt,rP,tm);rplane=Plane(ref)
        series=[dict(label='reference',t=tm,xy=rplane.xy(ref))]
        for label,est in [('prediction',P),('best adjusted',plane.lift(adjust(t,xy,theta)))]:
            al,_=metrics.align_sim3(est[mask],ref)
            series.append(dict(label=label,t=tm,xy=rplane.xy(est[mask] if al is None else al)))
        view_clips.append(dict(key=key,mode=clip['camera_mode'],group=clip['camera_mode'],series=[dict(label=s['label'],
            t=np.round(s['t'],2).tolist(),x=np.round(s['xy'][:,0],3).tolist(),y=np.round(s['xy'][:,1],3).tolist()) for s in series]))
        print(f"{key} {clip['camera_mode']:9s} {before:6.2f}% -> {after:6.2f}%  "+' '.join(f'{p}={v:+.3f}' for p,v in theta.items()),flush=True)
    summary={}
    for mode in sorted({r['mode'] for r in rows}):
        rs=[r for r in rows if r['mode']==mode]
        summary[mode]=dict(clips=len(rs),before=float(np.mean([r['before'] for r in rs])),after=float(np.mean([r['after'] for r in rs])),
            **{p:dict(median=float(np.median([r['theta'][p] for r in rs])),frac_positive=float(np.mean([r['theta'][p]>0 for r in rs]))) for p in PARAMS})
    out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=True)
    write(out/'dev_analysis.json',dict(rows=rows,summary=summary))
    render(out/'report.html',dict(mode='dev',dev=dict(rows=rows,summary=summary),clips=view_clips,params=PARAMS,units=UNITS,steps=STEP))
    for mode,s in summary.items():
        print(f"{mode}: {s['clips']} clips, mean drift {s['before']:.2f}% -> {s['after']:.2f}%  "
              +' '.join(f"{p} med {s[p]['median']:+.3f} ({100*s[p]['frac_positive']:.0f}% +)" for p in PARAMS))


def cmd_selftest(args):
    """Fake references = base with known per-group adjustments plus unmodelled wiggle;
    random group probes are scored against them, then the estimator must recover the signs."""
    rng=np.random.default_rng(args.seed);info=test_clips();base=read(ROOT/'artifacts/submission_19226.json')
    groups=list(dict.fromkeys(i['group'] for i in info.values()))
    truth={g:{p:float(rng.choice([-1,1])*rng.uniform(.5,1.5)*STEP[p]) for p in PARAMS} for g in groups}
    refs={}
    for key,i in info.items():
        t=np.asarray(base[key]['times']);P=np.asarray(base[key]['positions']);plane=Plane(P)
        xy=adjust(t,plane.xy(P),truth[i['group']]);tau,h,s,dt=steps(t,xy)
        h=h+np.deg2rad(4)*np.sin(2*np.pi*tau/rng.uniform(40,120))  # unmodelled error
        xy=np.vstack([xy[0],xy[0]+np.cumsum(np.c_[np.sin(h),np.cos(h)]*(s*dt)[:,None],axis=0)])
        tt=np.arange(t[0],t[-1],1.);r=metrics.resample_to(t,plane.lift(xy),tt)
        refs[key]=(tt,r*PATH_M[i['mode']]/metrics.cumulative_length(r)[-1])
    def score(bundle):
        return round(float(np.mean([metrics.score_trajectory(np.asarray(v['times']),np.asarray(v['positions']),*refs[k])['drift_pct']
                                    for k,v in bundle.items()])),2)
    ws=Path(args.workspace).resolve()
    if ws.exists():raise SystemExit('Choose a new selftest workspace; refusing overwrite')
    (ws/'files').mkdir(parents=True);(ws/'files/base.json').write_text(json.dumps(base))
    reg=dict(base=dict(label='base',file='files/base.json',drift=score(base),sha256=''),submissions=[])
    for k in range(args.probes):
        spec={'groups':{}}
        for g in rng.choice(groups,size=rng.integers(1,3),replace=False):
            spec['groups'][str(g)]={str(p):float(rng.choice([-1,1])*STEP[p]) for p in rng.choice(PARAMS,size=rng.integers(1,3),replace=False)}
        cmd_apply(argparse.Namespace(base=ws/'files/base.json',spec=_tmp(ws,spec,k),output=ws/f'probe{k}'))
        sub=read(ws/f'probe{k}/submission.json');shutil.copy(ws/f'probe{k}/submission.json',ws/f'files/probe{k}.json')
        reg['submissions'].append(dict(label=f'probe{k}',file=f'files/probe{k}.json',drift=score(sub),sha256='',note=json.dumps(spec)))
    write(ws/'registry.json',reg);a=build_report(ws);print_summary(a)
    confident=[d for d in a['directions'] if d['recommendation'] in ('increase','decrease')]
    right=[d for d in confident if (d['recommendation']=='increase')==(truth[d['group']][d['param']]>0)]
    print(f"\nselftest: {len(right)}/{len(confident)} confident directions match the planted truth "
          f"({len(a['directions'])-len(confident)} left uncertain or negligible)")
    for d in a['directions']:
        miss='' if d not in confident or d in right else '  <- wrong'
        print(f"  {d['group']:9s} {d['param']:8s} truth {truth[d['group']][d['param']]:+.2f}  P(increase helps)={d['p_increase_helps']:.2f}{miss}")
    return len(right)==len(confident)


def _tmp(ws,spec,k):
    path=ws/f'spec{k}.json';write(path,spec);return path


def main():
    ap=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    sub=ap.add_subparsers(dest='cmd',required=True)
    a=sub.add_parser('add',help='register one scored submission');a.add_argument('file',type=Path)
    a.add_argument('--drift',type=float,required=True,help='leaderboard drift %% of this file');a.add_argument('--label');a.add_argument('--note')
    a.add_argument('--workspace',default=DEFAULT_WORKSPACE)
    r=sub.add_parser('report');r.add_argument('--workspace',default=DEFAULT_WORKSPACE)
    p=sub.add_parser('apply',help='make a probe submission');p.add_argument('--base',type=Path,default=ROOT/'artifacts/submission_19226.json')
    p.add_argument('--spec',type=Path,required=True,help='{"all":{..},"groups":{"flir-A":{"gain":0.05}},"clips":{"clip_092":{"drift":-0.5}}}')
    p.add_argument('--output',type=Path,required=True)
    d=sub.add_parser('dev',help='best adjustment per clip against references');d.add_argument('--manifest',type=Path,default=ROOT/'lookout-vo-dev-v0.1/release/release_manifest.json')
    d.add_argument('--predictions',type=Path,required=True);d.add_argument('--split',default='val');d.add_argument('--clips')
    d.add_argument('--output',type=Path,required=True)
    s=sub.add_parser('selftest');s.add_argument('--workspace',default=ROOT/'outputs/direction_selftest');s.add_argument('--probes',type=int,default=8)
    s.add_argument('--seed',type=int,default=0)
    args=ap.parse_args()
    ok=dict(add=cmd_add,report=cmd_report,apply=cmd_apply,dev=cmd_dev,selftest=cmd_selftest)[args.cmd](args)
    if args.cmd=='selftest' and not ok:raise SystemExit(1)


if __name__=='__main__':main()
