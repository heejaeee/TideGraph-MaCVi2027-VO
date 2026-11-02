import json
import numpy as np
from .context import *
from . import consensus as d
OUT=ROOT/'runs/domain_v73_submission'
from types import SimpleNamespace
speed=SimpleNamespace(envelope=envelope)


def infer(split):
    v.core.bg.guard();parity={}
    for key in (v.TRAIN if split=='train' else d.test.KEYS):
        case,_,_,r3,_,_=v.core.g.provider(key,split);base=np.load(d.OUT/f'{split}/signals/{key}.npz');t,y,s=base['times'],base['yaw'],base['speed'];assert np.array_equal(t,case['t']);ds=[];diags=[]
        for phase,length in d.INTERVALS:
            for seed in d.SEEDS:
                p=d.pose_path(key,split,seed);zz=np.load(p);m=zz['camera_to_world'];stat=json.loads(p.with_suffix('.json').read_text());assert np.array_equal(zz['times'],t)
                usable=stat['loop_patch_edges']>0 and stat['global_ba_successes']>0 and not stat['global_ba_errors'];f,right=v.core.bg.frame_axes(m);pos=m[:,:3,3];rows=[]
                for i in range(phase,len(t)-length,16):
                    j=i+length;dp=pos[j]-pos[i];norm=np.linalg.norm(dp);hor=np.linalg.norm([dp@f,dp@right])
                    if usable and norm>1e-8 and hor/norm>=.7:rows.append(dict(i=i,j=j,weight=1.,log_distance=float(np.log(hor))))
                log,diag=v.core.local.motion.fit_motion(t,y,s,rows,'speed');ds.append(log);diags.append(dict(source=f'SLAM{seed}',phase=phase,length=length,fit=diag))
        f,right=v.core.bg.frame_axes(r3);pos=r3[:,:3,3]
        for phase,length in d.INTERVALS:
            rows=[]
            for i in range(phase,len(t)-length,16):
                j=i+length;dp=pos[j]-pos[i];norm=np.linalg.norm(dp);hor=np.linalg.norm([dp@f,dp@right])
                if norm>1e-8 and hor/norm>=.7:rows.append(dict(i=i,j=j,weight=1.,log_distance=float(np.log(hor))))
            log,diag=v.core.local.motion.fit_motion(t,y,s,rows,'speed');ds.append(log);diags.append(dict(source='R3',phase=phase,length=length,fit=diag))
        ds=np.asarray(ds);assert ds.shape==(16,len(t));env=speed.envelope(ds);final_speed=s*np.exp(env)
        pred=dict(case['pred'],positions=v.core.integrate_velocity(t,y,final_speed).tolist());save_json(OUT/f'{split}/predictions/{key}.json',pred)
        path=OUT/f'{split}/signals/{key}.npz';path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,times=t,yaw=y,speed=final_speed,base_speed=s,log_speed_delta=env,variant_deltas=ds)
        save_json(OUT/f'{split}/diagnostics/{key}.json',dict(fits=diags,active_fraction=float(np.mean(abs(env)>1e-12)),max_speed_change_pct=float(100*abs(np.exp(env)-1).max())))
        if split=='train':
            q=np.load(ROOT/f'runs/domain_v71_independent_speed/signals/{key}.npz');old=json.loads((ROOT/f'runs/domain_v71_independent_speed/predictions/r3_slam_envelope/{key}.json').read_text())
            logerr=float(abs(env-q['r3_slam_envelope']).max());poserr=float(abs(np.array(pred['positions'])-old['positions']).max())
            assert logerr<1e-7 and poserr<1e-5,(key,logerr,poserr);parity[key]=dict(max_log_speed_difference=logerr,max_position_difference=poserr)
        print('V73_INFER',split,key,float(np.mean(abs(env)>1e-12)),float(100*abs(np.exp(env)-1).max()),flush=True)
    if split=='train':save_json(REP/'train_parity.json',dict(passed=True,clips=parity))
