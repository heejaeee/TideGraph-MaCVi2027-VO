import json
import numpy as np
import cv2
from functools import lru_cache
from .context import *
OUT=ROOT/'runs/domain_v65_submission'
INTERVALS=[(0,24),(8,24),(0,16),(0,32)]
SEEDS=[0,104729,209759]


def inputs(key,split):
    case,_,_,rp,k,meta=v.core.g.provider(key,split)
    z=np.load(ROOT/f'runs/domain_v24_submission/{split}/signals/{key}.npz')
    assert np.array_equal(case['t'],z['times'])
    f,right=v.core.bg.frame_axes(rp);up=np.cross(f,right)
    r=v.core.Rotation.from_rotvec((z['yaw']-v.core.local.motion.planar_yaw(rp))[:,None]*up).as_matrix()@rp[:,:3,:3]
    return case,z,r,k,meta,(f,right)

def prepare(key,split):
    path=OUT/f'{split}/tracks/{key}.npz'
    if path.exists():return dict(np.load(path))
    _,_,_,_,meta,_=inputs(key,split)
    static=json.loads((ROOT/'runs/domain_v5/segmentation/model.json').read_text())['static_ids']
    @lru_cache(maxsize=100)
    def mask(i):
        p=mask_path(key,i,split);lab=cv2.imread(str(p),0)
        if lab is None:raise FileNotFoundError(p)
        return cv2.erode(np.isin(lab,static).astype(np.uint8),np.ones((3,3),np.uint8))
    rows,chunks,offsets=[],[],[0]
    for p in sorted((frame_root(split)/f'cotracker_windows/{key}').glob('*.npz')):
        z=np.load(p);i=int(z['start'])
        if i%16 or len(z['tracks'])<=24:continue
        good=z['visible'][[0,16,24]].all(0)
        tr=z['tracks'][[0,16,24]][:,good].transpose(1,0,2)
        ok=np.ones(len(tr),bool)
        for a,off in enumerate([0,16,24]):
            mm=mask(i+off);xy=np.floor(tr[:,a]/[meta['width'],meta['height']]*[mm.shape[1],mm.shape[0]]).astype(int)
            inside=(xy[:,0]>=0)&(xy[:,1]>=0)&(xy[:,0]<mm.shape[1])&(xy[:,1]<mm.shape[0]);ids=np.flatnonzero(inside)
            inside[ids]&=mm[xy[ids,1],xy[ids,0]]>0;ok&=inside
        tr=tr[ok]
        if len(tr)<40:continue
        ids=np.lexsort((tr[:,0,1],tr[:,0,0]));ids=ids[np.linspace(0,len(ids)-1,min(240,len(ids))).astype(int)]
        rows.append([1,i,i+16,i+24]);chunks.append(tr[ids]);offsets.append(offsets[-1]+len(ids))
    data=dict(rows=np.asarray(rows,int).reshape(-1,4),tracks=np.concatenate(chunks) if chunks else np.empty((0,3,2)),offsets=np.array(offsets))
    path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,**data)
    return data

def infer(split):
    v.core.bg.guard();parity={}
    for key in (v.TRAIN if split=='train' else test.KEYS):
        case,z,_,_,_,_=inputs(key,split);t,y,s=z['times'],z['yaw'],z['speed'];ds=[];diagnostics=[]
        # Same ordering as V64: normal3,phase3,short3,long3.
        for phase,length in INTERVALS:
            for seed in SEEDS:
                path=pose_path(key,split,seed);zz=np.load(path);mat=zz['camera_to_world'];d=json.loads(path.with_suffix('.json').read_text())
                assert np.array_equal(t,zz['times']) and np.isfinite(mat).all()
                usable=d['loop_patch_edges']>0 and d['global_ba_successes']>0 and not d['global_ba_errors']
                f,right=v.core.bg.frame_axes(mat);pos=mat[:,:3,3];rows=[]
                for i in range(phase,len(t)-length,16):
                    j=i+length;dp=pos[j]-pos[i];norm=np.linalg.norm(dp);hor=np.linalg.norm([dp@f,dp@right])
                    if usable and norm>1e-8 and hor/norm>=.7:rows.append(dict(i=i,j=j,course=float(v.core.bg.wrap(np.arctan2(dp@right,dp@f)+z['initial_yaw'])),weight=1.))
                dy,diag=v.core.local.motion.fit_motion(t,y,s,rows,'course');ds.append(dy)
                diagnostics.append(dict(seed=seed,phase=phase,length=length,usable_backend=usable,fit=diag))
        ds=np.asarray(ds);lo=ds.min(0);hi=ds.max(0);env=np.where(lo>0,lo,np.where(hi<0,hi,0.))
        folds=[];fold_diags=[]
        for fold in range(5):
            rows=json.loads((OUT/f'{split}/course/fold{fold}/observations/{key}.json').read_text())['rows']['cotracker_multiview']
            dy,diag=v.core.local.motion.fit_motion(t,y,s,rows,'course');folds.append(dy);fold_diags.append(diag)
        med=np.median(folds,axis=0);yaw=y+np.clip(env+(y+med)-y,-np.deg2rad(3),np.deg2rad(3))
        pred=dict(case['pred'],positions=v.core.integrate_velocity(t,yaw,s).tolist())
        save_json(OUT/f'{split}/predictions/{key}.json',pred)
        path=OUT/f'{split}/signals/{key}.npz';path.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(path,times=t,speed=s,yaw=yaw,v24_control=y,variant_deltas=ds,fold_delta=folds,envelope=env)
        diag=dict(slam=diagnostics,course=fold_diags,envelope_fraction=float(np.mean(abs(env)>1e-12)),course_fraction=float(np.mean(abs(med)>1e-12)),max_correction_deg=float(np.rad2deg(abs(yaw-y).max())))
        save_json(OUT/f'{split}/diagnostics/{key}.json',diag)
        if split=='train':
            old=np.load(ROOT/f'runs/domain_v64_course_support_envelope/signals/{key}.npz')
            diff=float(abs(yaw-old['envelope_plus_course']).max())
            oldp=json.loads((ROOT/f'runs/domain_v64_course_support_envelope/predictions/envelope_plus_course/{key}.json').read_text())
            posdiff=float(abs(np.array(pred['positions'])-oldp['positions']).max())
            parity[key]=dict(max_yaw_difference_rad=diff,max_position_difference=posdiff)
            assert diff<1e-7 and posdiff<1e-5,(key,parity[key])
        print('V65_INFER',split,key,diag['envelope_fraction'],diag['course_fraction'],diag['max_correction_deg'],flush=True)
    if split=='train':save_json(REP/'train_parity.json',dict(passed=True,clips=parity))
