"""Wide46 LOCO + wide VAL21 gate validation; FLIR8 transfer is separate.

TRAIN46 has no deployed R3/current-water cache. Generate the same local water
plane observer on TRAIN/VAL for signal-level comparisons, then separately test
actual deployed VAL/FLIR water with frozen V87 yaw. Do not conflate these arms.
"""
import argparse
import concurrent.futures
import pickle
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter1d,median_filter
from scipy.spatial.transform import Rotation,Slerp
import v101_protocol as p
import v103_speed_audit as audit
import train_speed_regressor_v3 as old

ROOT=p.ROOT;OUT=ROOT/'runs/domain_v110_speed_gate';REP=ROOT/'reports/domain_v110_speed_gate';MOD=ROOT/'models/v110_speed_gate'
TRAIN=[k for k,c in p.CLIPS.items() if c['split']=='train'];WTRAIN=[k for k in TRAIN if p.CLIPS[k]['camera_mode']=='wide'];WVAL=p.GROUPS['wide'];FLIR=p.GROUPS['flir-VIS']
METHODS=['constant','corr07','delta_only','corr_then_delta']

def freeze():
    p.guard();assert not (REP/'plan.json').exists();assert len(WTRAIN)==46 and len(WVAL)==21
    files=[Path(__file__),ROOT/'scripts/v110_water_worker.py',ROOT/'scripts/submission_water_infer_clip.py',ROOT/'scripts/train_speed_regressor_v3.py',ROOT/'models/flow_speed_v3/model.pkl']
    files += [ROOT/f'data/features/flow_speed_v2/{k}.npz' for k in TRAIN+WVAL]
    p.save(REP/'plan.json',dict(created_utc=p.now(),wide_train=WTRAIN,wide_val=WVAL,flir=FLIR,
        priors='Same fixed HGB hyperparameters, log target, feature preparation as original. Train46 evaluation:46 leave-one-clip-out fits on remaining67 TRAIN clips. VAL: original TRAIN68 model. FLIR: V103 LOCO frozen priors. No VAL or TEST training, no new parameter selection.',
        common_observer='TRAIN wide46 lacks deployed full R3/current-water states. Generate original960px2Hz short-baseline plane-flow observations, then signed forward speed in local camera/water plane; median6s + Gaussian5s,95percent vector magnitude clipping. VAL21 uses same raw plane-flow cache and processing. Shared observer arm evaluates speed profiles, not a claimed V87 TRAIN reproduction.',
        deployment='Separately reconstruct exact original water-current speed for VAL21 and read exact FLIR V103 LOCO water. Retain frozen V87 yaw. Original wide branch has a plane-residual fallback and is NOT the FLIR correlation-only gate. Report V87 control separately.',
        methods=METHODS,consensus='Every10seconds actual timestamps, positive normalized signals only. If both log-speed increments have same sign and exceed log1.05, retain the smaller magnitude. Otherwise incrementzero. Cumulative sum in log space, linear interpolation, Gaussian5s and global mean1 normalization. No signal amplitude estimated from GT.',
        corr07='If full-clip corr>=0.7 use water, otherwise constant1.',
        corr_then_delta='Keep water if original0.7gate passes; otherwise use delta consensus.',
        GT_diagnostics='Speed from Gaussian2s smoothed reference derivative, exclude2s ends. CV bands: <.1 flat, [.1,.25)moderate, >=.25dynamic; near-stationary mean<.2m/s separately uncertain. These are descriptive fixed bands. MAE of profiles normalized by clip valid mean, active outputCV>=.1; report misses and false activations.',
        limitations=['Wide-to-FLIR transfer is not guaranteed by camera-independent algebra. FOV, water, exposure and learned domain differ.',
             'Clip LOCO is not verified boat-disjoint; historic choices were already developed on this dataset.',
             'Two predictions may share errors; agreement is not independence or accuracy.',
             'No candidate selected by TEST feedback.'],
        dependencies={str(f.relative_to(ROOT)):p.sha(f) for f in files},preserved_submissions=p.read(p.REP/'plan.json')['preserved_submissions']))

def train():
    plan=p.read(REP/'plan.json');MOD.mkdir(parents=True,exist_ok=True)
    for f,h in plan['dependencies'].items():assert p.sha(ROOT/f)==h,f
    xs={};ys={}
    for k in TRAIN:
        with np.load(ROOT/f'data/features/flow_speed_v2/{k}.npz') as z:xs[k]=old.inputs(z);t=z['times']
        rt,rp=p.load_reference(p.DATA,p.CLIPS[k]);rp=gaussian_filter1d(rp,2/np.median(np.diff(rt)),axis=0)
        target=np.interp(t,rt,np.linalg.norm(np.gradient(rp,rt,axis=0),axis=1));target/=max(target.mean(),.01);ys[k]=np.log(target+.05)
    for k in WTRAIN:
        dest=MOD/f'omit_{k}.pkl'
        if dest.exists():continue
        keys=[x for x in TRAIN if x!=k];beg=time.perf_counter();model=old.model();model.fit(np.concatenate([xs[x] for x in keys]),np.concatenate([ys[x] for x in keys]))
        with dest.open('wb') as f:pickle.dump(model,f)
        p.save(dest.with_suffix('.json'),dict(train_keys=keys,excluded_keys=[k],sha256=p.sha(dest),seconds=time.perf_counter()-beg,parameters=model.get_params()))
        print('V110_TRAIN',k,round(time.perf_counter()-beg,1),flush=True)

def water():
    p.guard();logdir=OUT/'water_logs';logdir.mkdir(parents=True,exist_ok=True)
    def one(k):
        dest=OUT/f'water/diagnostics/{k}.water.npz'
        if dest.exists():return dict(key=k,returncode=0,reused=True)
        args=[sys.executable,str(ROOT/'scripts/v110_water_worker.py'),'--dataset',str(p.DATA/'release_manifest.json'),'--key',k,'--output',str(OUT/'water'),'--fps','2','--size','960','--translation-model','full','--residual-units','pixel']
        beg=time.perf_counter()
        with (logdir/f'{k}.log').open('w') as f:r=subprocess.run(args,stdout=f,stderr=subprocess.STDOUT,cwd=ROOT)
        out=dict(key=k,returncode=r.returncode,seconds=time.perf_counter()-beg)
        print('V110_WATER',k,r.returncode,round(out['seconds'],1),flush=True);return out
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:rows=list(ex.map(one,WTRAIN))
    p.save(REP/'water_extraction.json',dict(rows=rows,reference_access_blocked=True))
    assert all(r['returncode']==0 for r in rows),rows

def local_water(key):
    folder=OUT/'water' if key in WTRAIN else ROOT/'runs/water_dense_val';z=np.load(folder/f'diagnostics/{key}.water.npz');raw=z['raw'];t=np.array(p.read(folder/f'predictions/{key}.json')['times']);cal=p.CLIPS[key]['calibration']
    v=raw[:,4:7].copy();valid=np.isfinite(v).all(1)&(z['pair_dt']<=.125+1e-6)
    for d in range(3):v[:,d]=np.interp(t,t[valid],v[valid,d])
    norm=np.linalg.norm(v,axis=1);v*=np.minimum(1,np.quantile(norm,.95)/np.maximum(norm,1e-8))[:,None]
    cy=cal['cy']*960/cal['width'];fy=cal['fy']*960/cal['width'];nz=(cy-raw[:,3])/fy
    axis=np.c_[np.zeros(len(t)),-nz,np.ones(len(t))];axis/=np.linalg.norm(axis,axis=1,keepdims=True)
    speed=np.sum(v*axis,axis=1);dt=(t[-1]-t[0])/(len(t)-1);size=max(3,int(6/dt)//2*2+1)
    speed=gaussian_filter1d(median_filter(speed,size=size),5/dt);speed/=max(np.mean(abs(speed)),1e-8)
    return t,speed,dict(valid_water_fraction=float(valid.mean()),negative_fraction=float(np.mean(speed<0)),plane_residual_q90=float(np.nanquantile(raw[:,2],.9)))

def deployed_val_water(key):
    folder=ROOT/'runs/submission_fusion_validation_check';manifest=p.read(folder/'inference_manifest.json');rec=next(x for x in manifest['clips'] if x['key']==key);settings=manifest['settings'];path=lambda n:ROOT/settings[n]
    with np.load(folder/f'diagnostics/{key}.fusion.npz') as z:t=z['times'];current=z['current'];old_speed=z['speed']
    z=np.load(path('water')/f'diagnostics/{key}.water.npz');vt=np.array(p.read(path('water')/f'predictions/{key}.json')['times']);v=z['raw'][:,4:7].copy();dt=z['pair_dt'];valid=np.isfinite(v).all(1)&(dt<=.125+1e-6)
    for d in range(3):v[:,d]=np.interp(vt,vt[valid],v[valid,d])
    norm=np.linalg.norm(v,axis=1);v*=np.minimum(1,np.quantile(norm,.95)/np.maximum(norm,1e-8))[:,None]
    poses=np.load(path(rec['selected_orientation'])/f'diagnostics/{key}.poses.npz')['camera_to_world'];rot=Slerp(t,Rotation.from_matrix(poses[:,:3,:3]))(np.clip(vt-dt,t[0],t[-1])).as_matrix()
    down=np.median(poses[:,:3,1],axis=0);down/=np.linalg.norm(down);vel=np.einsum('nij,nj->ni',rot,v);vel-=np.outer(vel@down,down);step=np.median(np.diff(vt));window=max(3,int(6/step)//2*2+1)
    vel=gaussian_filter1d(median_filter(vel,size=(window,1)),5/step,axis=0);heading=rot[:,:,2].copy();heading-=np.outer(heading@down,down);heading/=np.maximum(np.linalg.norm(heading,axis=1,keepdims=True),1e-8)
    water=np.interp(t,vt,np.sum((vel+current)*heading,axis=1));water/=max(np.mean(abs(water)),1e-8)
    return t,water,rec,old_speed

def prior(key,t):
    path=MOD/f'omit_{key}.pkl' if key in WTRAIN else ROOT/'models/flow_speed_v3/model.pkl'
    with path.open('rb') as f:model=pickle.load(f)
    if key in WTRAIN:assert key not in p.read(path.with_suffix('.json'))['train_keys']
    with np.load(ROOT/f'data/features/flow_speed_v2/{key}.npz') as z:s=old.normalize_speed(model.predict(old.inputs(z)),'log');out=np.interp(t,z['times'],s)
    return out/max(out.mean(),1e-8)

def proposals(t,w,q):
    corr=float(np.corrcoef(w,q)[0,1]) if min(np.std(w),np.std(q))>1e-6 else 0.
    knots=np.r_[np.arange(t[0],t[-1],10),t[-1]];ww=np.interp(knots,t,w);pp=np.interp(knots,t,q)
    dw=np.diff(np.log(np.maximum(ww,.05)));dq=np.diff(np.log(np.maximum(pp,.05)))
    valid=(ww[:-1]>.05)&(ww[1:]>.05)&(pp[:-1]>.05)&(pp[1:]>.05)
    agree=valid&(dw*dq>0)&(abs(dw)>=np.log(1.05))&(abs(dq)>=np.log(1.05))
    delta=np.where(agree,np.sign(dw)*np.minimum(abs(dw),abs(dq)),0.)
    log=np.interp(t,knots,np.r_[0.,np.cumsum(delta)]);s=np.exp(gaussian_filter1d(log,5/((t[-1]-t[0])/(len(t)-1))));s/=s.mean()
    return dict(constant=np.ones(len(t)),corr07=w.copy() if corr>=.7 else np.ones(len(t)),delta_only=s,corr_then_delta=w.copy() if corr>=.7 else s.copy()),dict(correlation=corr,correlation_pass=corr>=.7,agreed_intervals=int(agree.sum()),intervals=len(agree),agreement_fraction=float(agree.mean()),delta_profile_cv=float(np.std(s)),water_cv_mean_abs=float(np.std(w)/max(np.mean(abs(w)),1e-8)),prior_cv=float(np.std(q)/q.mean()))

def predict(only_deployed=False):
    p.guard();files={};routing=[]
    for key in ([] if only_deployed else WTRAIN+WVAL):
        t,w,diag=local_water(key);q=prior(key,t);methods,route=proposals(t,w,q)
        dest=OUT/f'signals/common/{key}.npz';p.npzsave(dest,times=t,water=w,prior=q,**methods);files[str(dest.relative_to(ROOT))]=p.sha(dest);routing.append(dict(key=key,arm='common',split=p.CLIPS[key]['split'],**diag,**route))
    partial=REP/'predictions_frozen_deployed.json';reuse=partial.exists() and not only_deployed
    if reuse:
        cached=p.read(partial)
        for f,h in cached['files'].items():assert p.sha(ROOT/f)==h,f
        files.update(cached['files']);routing.extend(cached['routing'])
    for key in ([] if reuse else WVAL+FLIR):
        c=p.case(key)
        if key in WVAL:
            t,w,rec,old_speed=deployed_val_water(key);q=prior(key,t)
            assert abs(np.corrcoef(w,q)[0,1]-rec['speed_correlation'])<1e-8
            np.testing.assert_allclose(old_speed,q if rec['learned_speed_weight'] else w,atol=1e-9,rtol=0)
            before=c['speed']
        else:
            z=np.load(ROOT/f'runs/domain_v103_speed_audit/signals/loco/{key}.npz');t,w,q=z['times'],z['water'],z['prior'];before=z['speed']
        methods,route=proposals(t,w,q);methods['deployed_control']=before
        for name,speed in methods.items():
            if key in FLIR and name!='deployed_control':
                previous=audit.OUT
                try:audit.OUT=OUT/f'replay/{name}';speed,_=audit.replay(key,speed)
                finally:audit.OUT=previous
                if name=='corr07':np.testing.assert_allclose(speed,before,atol=1e-10,rtol=0)
                methods[name]=speed
            dest=OUT/f'predictions/{name}/{key}.json';pred=dict(c['pred'],positions=p.integrate(t,c['yaw'],speed).tolist());p.validate_prediction(pred,key,p.load_timestamps(p.DATA,c['clip']));p.save(dest,pred);files[str(dest.relative_to(ROOT))]=p.sha(dest)
        dest=OUT/f'signals/deployed/{key}.npz';p.npzsave(dest,times=t,water=w,prior=q,**methods);files[str(dest.relative_to(ROOT))]=p.sha(dest);routing.append(dict(key=key,arm='deployed',split=p.CLIPS[key]['split'],**route))
        print('V110_PREDICT',key,flush=True)
    p.save(REP/('predictions_frozen_deployed.json' if only_deployed else 'predictions_frozen.json'),dict(created_utc=p.now(),files=files,routing=routing,reference_access_blocked=True))

def gt_speed(key,t):
    rt,rp=p.load_reference(p.DATA,p.CLIPS[key]);smooth=gaussian_filter1d(rp,2/np.median(np.diff(rt)),axis=0,mode='nearest');vel=np.gradient(smooth,rt,axis=0);s=np.interp(t,rt,np.linalg.norm(vel[:,:2],axis=1));valid=(t>=rt[0]+2)&(t<=rt[-1]-2)
    return s,valid,rt,rp

def evaluate(only_deployed=False):
    frozen=p.read(REP/('predictions_frozen_deployed.json' if only_deployed else 'predictions_frozen.json'))
    for f,h in frozen['files'].items():assert p.sha(ROOT/f)==h,f
    profiles=[];drifts=[]
    for arm,keys in ([('deployed',WVAL+FLIR)] if only_deployed else [('common',WTRAIN+WVAL),('deployed',WVAL+FLIR)]):
        for key in keys:
            z=np.load(OUT/f'signals/{arm}/{key}.npz');t=z['times'];gt,valid,rt,rp=gt_speed(key,t);mean=float(np.mean(gt[valid]));cv=float(np.std(gt[valid])/max(mean,1e-8));kind='near_stationary' if mean<.2 else 'flat' if cv<.1 else 'moderate' if cv<.25 else 'dynamic'
            for name in METHODS+(['deployed_control'] if arm=='deployed' else []):
                s=z[name];sn=s/max(np.mean(abs(s[valid])),1e-8);gn=gt/max(mean,1e-8);scv=float(np.std(s[valid])/max(np.mean(abs(s[valid])),1e-8))
                profiles.append(dict(key=key,arm=arm,split=p.CLIPS[key]['split'],camera=p.CLIPS[key]['camera_mode'],method=name,GT_speed_cv=cv,GT_speed_mean_mps=mean,GT_band=kind,output_cv=scv,active=scv>=.1,relative_speed_mae=float(np.mean(abs(sn[valid]-gn[valid]))),relative_speed_rmse=float(np.sqrt(np.mean((sn[valid]-gn[valid])**2)))))
                if arm=='deployed':
                    pred=p.read(OUT/f'predictions/{name}/{key}.json');drifts.append(p.score(key,name,t,np.array(pred['positions']),rt,rp))
            p.npzsave(OUT/f'evaluation_only/{arm}/{key}.npz',times=t,gt_speed=gt,valid=valid)
    summary={}
    for arm,split,camera in [('common','train','wide'),('common','val','wide'),('deployed','val','wide'),('deployed','train','flir-VIS')]:
        if only_deployed and arm=='common':continue
        group=arm+'__'+split+'__'+camera;summary[group]={}
        for name in METHODS+(['deployed_control'] if arm=='deployed' else []):
            rr=[r for r in profiles if r['arm']==arm and r['split']==split and r['camera']==camera and r['method']==name]
            q=dict(clips=len(rr),mean_relative_speed_mae=float(np.mean([r['relative_speed_mae'] for r in rr])),dynamic_total=sum(r['GT_band']=='dynamic' for r in rr),dynamic_missed=sum(r['GT_band']=='dynamic' and not r['active'] for r in rr),flat_total=sum(r['GT_band']=='flat' for r in rr),flat_activated=sum(r['GT_band']=='flat' and r['active'] for r in rr))
            if arm=='deployed':
                base={r['key']:r['drift_pct'] for r in drifts if r['group']==camera and r['method']=='deployed_control'};dd=[r for r in drifts if r['group']==camera and r['method']==name];d=[r['drift_pct']-base[r['key']] for r in dd];q.update(mean_drift_pct=float(np.mean([r['drift_pct'] for r in dd])),wins=sum(v<-1e-8 for v in d),losses=sum(v>1e-8 for v in d),worst_delta_pp=max(d))
            summary[group][name]=q
    p.save(REP/('results_deployed.json' if only_deployed else 'results.json'),dict(created_utc=p.now(),summary=summary,profile_rows=profiles,drift_rows=drifts,official_test_scored=False,new_submission=False))
    print('V110_RESULTS',summary,flush=True)

def selfcheck():
    t=np.linspace(0,300,601);flat=np.ones(len(t));m,d=proposals(t,flat,flat);np.testing.assert_allclose(m['delta_only'],flat)
    wave=np.exp(.25*np.sin(t/30));wave/=wave.mean();m,d=proposals(t,wave,wave);assert d['agreed_intervals']>0 and np.corrcoef(m['delta_only'],wave)[0,1]>.98
    m,d=proposals(t,wave,1/wave);np.testing.assert_allclose(m['delta_only'],flat)
    p.save(REP/'consensus_checks.json',dict(passed=True,scope='Flat input unchanged, coherent changes retained, opposite changes rejected.'))

def predict_deployed():predict(True)
def evaluate_deployed():evaluate(True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['freeze','train','water','predict','evaluate','selfcheck','predict_deployed','evaluate_deployed']);args=ap.parse_args();globals()[args.action]()
