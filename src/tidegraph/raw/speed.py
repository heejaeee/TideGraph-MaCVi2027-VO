import json
import numpy as np
from collections import Counter
from . import consensus as d, motion
from .context import *
OUT=ROOT/'runs/domain_v74_observable_speed'


FRESH = [(4,20), (4,28), (4,36)]

def pose_rows(m,t,phase,length,thin=0):
    f,right=v.core.bg.frame_axes(m);pos=m[:,:3,3];rows=[];reject=Counter()
    for n,i in enumerate(range(phase,len(t)-length,16)):
        if thin and n%2 != thin-1: continue
        j=i+length;dp=pos[j]-pos[i];norm=float(np.linalg.norm(dp));hor=float(np.linalg.norm([dp@f,dp@right]))
        if not np.isfinite(norm+hor): reject['nonfinite']+=1
        elif norm<=1e-8: reject['zero_displacement']+=1
        elif hor/norm<.7: reject['low_horizontal_fraction']+=1
        else: rows.append(dict(i=i,j=j,weight=1.,log_distance=float(np.log(hor))))
    return rows,dict(reject)

def extract(split):
    v.core.bg.guard()
    for key in (v.TRAIN if split=='train' else d.test.KEYS):
        dest=OUT/f'{split}/raw/{key}.npz'
        if dest.exists():continue
        case,_,_,r3,_,_=v.core.g.provider(key,split);z=np.load(d.OUT/f'{split}/signals/{key}.npz');t,y,s=z['times'],z['yaw'],z['speed']
        mats=[];backend=[];healthy=[];stats=[]
        for seed in d.SEEDS:
            p=d.pose_path(key,split,seed);q=np.load(p);assert np.array_equal(q['times'],t)
            st=json.loads(p.with_suffix('.json').read_text());mats.append(q['camera_to_world'])
            backend.append(st['loop_patch_edges']>0 and st['global_ba_successes']>0 and not st['global_ba_errors'])
            healthy.append(not st['global_ba_errors'] and bool(np.isfinite(mats[-1]).all()))
            stats.append(dict(seed=seed,loop_edges=st['loop_patch_edges'],ba_successes=st['global_ba_successes'],ba_errors=st['global_ba_errors']))
        mats.append(r3);backend.append(True);healthy.append(bool(np.isfinite(r3).all()))
        arrays={};diags={}
        for bank,intervals,thin in [('base',d.INTERVALS,0),('fresh',FRESH,0),('even_rows',d.INTERVALS,1),('odd_rows',d.INTERVALS,2)]:
            ds=[];valid=[];details=[]
            for model,m in enumerate(mats):
                for phase,length in intervals:
                    rows,reject=pose_rows(m,t,phase,length,thin)
                    if not healthy[model]: rows=[]
                    log,diag=motion.fit_motion(t,y,s,rows,'speed')
                    support=motion.basis(t,rows)[2] if not diag['fallback'] and diag.get('optimizer_success',False) else np.zeros(len(t))
                    ds.append(log);valid.append(support>0)
                    details.append(dict(model=model,phase=phase,length=length,backend=bool(backend[model]),healthy=healthy[model],fit=diag,rejected=reject,rows=rows))
            arrays[bank+'_delta']=np.array(ds).reshape(4,len(intervals),len(t))
            arrays[bank+'_valid']=np.array(valid).reshape(4,len(intervals),len(t))
            diags[bank]=details
        dest.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(dest,times=t,yaw=y,speed=s,backend=backend,**arrays)
        save_json(dest.with_suffix('.json'),dict(backend=stats,banks=diags))
        print('V74_EXTRACT',split,key,'BA',[int(x) for x in backend[:3]],flush=True)

def masked_envelope(ds,valid,minimum,median=False):
    count=valid.sum(0);lo=np.min(np.where(valid,ds,np.inf),axis=0);hi=np.max(np.where(valid,ds,-np.inf),axis=0)
    good=count>=minimum
    if median:
        # Diagnostic majority-style aggregation; missing is never encoded as zero.
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',RuntimeWarning);res=np.nanmedian(np.where(valid,ds,np.nan),axis=0)
        return np.where(good,np.nan_to_num(res),0.),good
    return np.where(good & (lo>0),lo,np.where(good & (hi<0),hi,0.)),good

def combine(ds,valid,backend,require_backend=False,median=False,omit_seed=None):
    logs=[];masks=[]
    for model in range(4):
        mask=valid[model].copy()
        if (require_backend and not backend[model]) or model==omit_seed: mask[:]=False
        log,ok=masked_envelope(ds[model],mask,int(np.ceil(.75*len(ds[model]))),median)
        logs.append(log);masks.append(ok)
    slam,sm=masked_envelope(np.array(logs[:3]),np.array(masks[:3]),2,median)
    result,_=masked_envelope(np.array([slam,logs[3]]),np.array([sm,masks[3]]),2)
    return result,slam,logs[3]

def signals(z,bank):
    source=bank if bank in ['fresh','even_rows','odd_rows'] else 'base'
    ds=z[source+'_delta'].copy();valid=z[source+'_valid'].copy();omit=None
    if bank.startswith('omit_interval'):
        keep=np.arange(ds.shape[1])!=int(bank.removeprefix('omit_interval'));ds=ds[:,keep];valid=valid[:,keep]
    if bank.startswith('omit_seed'):omit=d.SEEDS.index(int(bank.removeprefix('omit_seed')))
    missing,_,_=combine(ds,valid,z['backend'],True,omit_seed=omit)
    pose,slam,r3=combine(ds,valid,z['backend'],omit_seed=omit)
    med,_,_=combine(ds,valid,z['backend'],median=True,omit_seed=omit)
    return dict(missing_aware=missing,pose_consensus=pose,pose_consensus_half=.5*pose,family_median=med,r3_only=r3,slam_only=slam)
