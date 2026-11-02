import json
import numpy as np
from .context import *
from . import speed as p, consensus as d, old_speed
from types import SimpleNamespace
OUT=ROOT/'runs/domain_v76_direction_guarded_speed'
BANKS=['base']
p.BANKS=['base','fresh','even_rows','odd_rows']
p.old=SimpleNamespace(speed=SimpleNamespace(envelope=envelope))


def direction_support(key,split,bank):
    raw=np.load(p.OUT/f'{split}/raw/{key}.npz');t,y,s=raw['times'],raw['yaw'],raw['speed'];_,_,_,r3,_,_=v.core.g.provider(key,split)
    mats=[np.load(d.pose_path(key,split,seed))['camera_to_world'] for seed in d.SEEDS]+[r3]
    intervals=p.FRESH if bank=='fresh' else d.INTERVALS
    if bank.startswith('omit_interval'):intervals=[a for i,a in enumerate(intervals) if i!=int(bank.removeprefix('omit_interval'))]
    omit=d.SEEDS.index(int(bank.removeprefix('omit_seed'))) if bank.startswith('omit_seed') else None
    thin=1 if bank=='even_rows' else 2 if bank=='odd_rows' else 0
    pos=v.core.integrate_velocity(t,y,s);knots=np.linspace(t[0],t[-1],int(np.ceil((t[-1]-t[0])/30))+1);all_values=[];offsets=[];details=[]
    for model,m in enumerate(mats):
        f,right=v.core.bg.frame_axes(m);mid=[];err=[];ident=[]
        for phase,length in intervals:
            rows,_=p.pose_rows(m,t,phase,length,thin)
            for row in rows:
                i,j=row['i'],row['j'];dp=m[j,:3,3]-m[i,:3,3];bp=pos[j]-pos[i]
                a=np.arctan2(dp@right,dp@f);b=np.arctan2(bp[0],bp[2])
                mid.append(.5*(t[i]+t[j]));err.append(v.core.bg.wrap(a-b));ident.append((i,j))
        mid=np.array(mid);err=np.array(err)
        # Circular median centered on circular mean, then residual wrap.
        center=float(np.angle(np.mean(np.exp(1j*err)))) if len(err) else 0.
        offset=float(center+np.median(v.core.bg.wrap(err-center))) if len(err) else 0.
        residual=v.core.bg.wrap(err-offset);offsets.append(float(np.rad2deg(offset)));values=[]
        for knot in knots:
            ix=np.flatnonzero(abs(mid-knot)<30)
            enough=len({ident[i] for i in ix})>=3 and model!=omit
            values.append(float(np.rad2deg(np.median(residual[ix]))) if enough else None)
        all_values.append(values)
    limit=2. if bank=='direction_tight2' else 4. if bank=='direction_loose4' else 3.;confidence=[]
    for n,knot in enumerate(knots):
        sl=[q[n] for q in all_values[:3] if q[n] is not None];rr=all_values[3][n];weight=0.
        if len(sl)>=2 and rr is not None:
            vals=np.array(sl+[rr]);disagreement=float(max(abs(vals).max(),np.ptp(vals)));weight=float(np.clip(1-disagreement/limit,0,1))
        confidence.append(weight);details.append(dict(time=float(knot),slam_median_residual_deg=sl,r3_median_residual_deg=rr,weight=weight))
    return np.interp(t,knots,confidence),dict(global_input_offsets_deg=offsets,knots=details,limit_deg=limit)

def infer(split):
    v.core.bg.guard()
    for key in (v.TRAIN if split=='train' else d.test.KEYS):
        case,_,_,_,_,_=v.core.g.provider(key,split);raw=np.load(p.OUT/f'{split}/raw/{key}.npz');z=np.load(p.OUT/f'{split}/base/signals/{key}.npz');t,y,s=z['times'],z['yaw'],z['speed'];old=z['v73_control'];neweligible=(abs(old)<=1e-12)&(not np.all(raw['backend'][:3]))
        anchor=[p.signals(raw,b)['pose_consensus'] for b in ['base','fresh','even_rows','odd_rows']]
        stable=p.old.speed.envelope(np.asarray(anchor))
        for bank in BANKS:
            variant=p.signals(raw,bank if bank in p.BANKS else 'base')['pose_consensus']
            joint=p.old.speed.envelope(np.array([stable,variant]));confidence,diag=direction_support(key,split,bank)
            residual=np.where(neweligible,joint,0.);unfiltered=np.where(neweligible,variant,0.)
            logs=dict(v65_control=z['v65_control'],v73_control=old,stable_residual_quarter=old+.25*residual,direction_guarded_quarter=old+.25*residual*confidence,direction_guarded_half=old+.5*residual*confidence,unfiltered_residual_quarter=old+.25*unfiltered)
            path=OUT/f'{split}/{bank}/signals/{key}.npz';path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,times=t,yaw=y,speed=s,direction_confidence=confidence,new_residual=residual,**logs)
            save_json(OUT/f'{split}/{bank}/diagnostics/{key}.json',dict(**diag,new_active_fraction=float(np.mean(abs(logs['direction_guarded_quarter']-old)>1e-12)),eligible_backend=bool(not np.all(raw['backend'][:3]))))
            for name,log in logs.items():save_json(OUT/f'{split}/{bank}/predictions/{name}/{key}.json',dict(case['pred'],positions=v.core.integrate_velocity(t,y,s*np.exp(log)).tolist()))
        print('V76_INFER',split,key,flush=True)
