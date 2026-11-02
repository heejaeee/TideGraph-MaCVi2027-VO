"""Exclude evaluation clips from the frozen HGB speed-prior training recipe.

This audits a component of a historically tuned system, not pristine nested CV.
All FLIR VIS share one calibration signature; configuration holdout removes all8,
and is not claimed to be a boat-disjoint fold. No VAL/TEST labels are used to fit.
"""
import argparse
import collections
import gc
import pickle
import sys
import time
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter1d,median_filter
from scipy.spatial.transform import Rotation,Slerp
import v101_protocol as p
import train_speed_regressor_v3 as old

ROOT=p.ROOT;OUT=ROOT/'runs/domain_v103_speed_audit';REP=ROOT/'reports/domain_v103_speed_audit';MOD=ROOT/'models/v103_speed_audit'
FLIR=p.GROUPS['flir-VIS'];TRAIN=[k for k,c in p.CLIPS.items() if c['split']=='train']
TEST={c['key']:c for c in p.read(ROOT/'data/lookout-vo-test-v0.1/release_manifest.json')['clips']}
TEST_FLIR=[k for k,c in TEST.items() if c['camera_mode']=='flir-VIS']


def source(key,split='train'):
    folder=ROOT/('runs/domain_probe_v2/baseline' if split=='train' else 'runs/submission_test_v1/fused')
    manifest=p.read(folder/'inference_manifest.json');rec=next(x for x in manifest['clips'] if x['key']==key)
    settings=manifest['settings']
    def path(n):return ROOT/settings[n]
    sp=folder/f'diagnostics/{key}.fusion.npz'
    with np.load(sp) as z:t=z['times'];fullspeed=z['speed'];current=z['current']
    wp=path('water')/f'diagnostics/{key}.water.npz';z=np.load(wp)
    vt=np.array(p.read(path('water')/f'predictions/{key}.json')['times']);v=z['raw'][:,4:7].copy();dt=z['pair_dt']
    valid=np.all(np.isfinite(v),axis=1)&(dt<=.125+1e-6)
    for d in range(3):v[:,d]=np.interp(vt,vt[valid],v[valid,d])
    norm=np.linalg.norm(v,axis=1);v*=np.minimum(1,np.quantile(norm,.95)/np.maximum(norm,1e-8))[:,None]
    poses=np.load(path(rec['selected_orientation'])/f'diagnostics/{key}.poses.npz')['camera_to_world']
    rot=Slerp(t,Rotation.from_matrix(poses[:,:3,:3]))(np.clip(vt-dt,t[0],t[-1])).as_matrix()
    down=np.median(poses[:,:3,1],axis=0);down/=np.linalg.norm(down)
    vel=np.einsum('nij,nj->ni',rot,v);vel-=np.outer(vel@down,down)
    step=np.median(np.diff(vt));window=max(3,int(6/step)//2*2+1)
    vel=gaussian_filter1d(median_filter(vel,size=(window,1)),5/step,axis=0)
    heading=rot[:,:,2].copy();heading-=np.outer(heading@down,down);heading/=np.maximum(np.linalg.norm(heading,axis=1,keepdims=True),1e-8)
    water=np.interp(t,vt,np.sum((vel+current)*heading,axis=1));water/=max(np.mean(abs(water)),1e-8)
    fp=path('features')/f'{key}.npz'
    return dict(t=t,water=water,features=fp,record=rec,old_base=fullspeed,source_folder=folder,
                q90=rec['q90_plane_residual'],dependency=[sp,wp,fp,path(rec['selected_orientation'])/f'diagnostics/{key}.poses.npz'])


def prior_and_gate(c,model):
    with np.load(c['features']) as z:
        v=old.normalize_speed(model.predict(old.inputs(z)),'log');prior=np.interp(c['t'],z['times'],v)
    prior/=max(prior.mean(),1e-8)
    corr=float(np.corrcoef(c['water'],prior)[0,1]) if min(np.std(c['water']),np.std(prior))>1e-6 else 0.
    passed=corr>=.7
    # The inherited FLIR outer gate overwrites the inner fallback with constant1.
    speed=c['water'].copy() if passed else np.ones(len(prior))
    return prior,speed,dict(correlation=corr,gate_pass=bool(passed),inner_learned_weight=float(c['q90']>.1 and not passed))


def freeze():
    p.guard();assert not (REP/'plan.json').exists()
    record=p.read(ROOT/'models/flow_speed_v3/training_record.json');assert set(record['train_cache_sha256'])==set(TRAIN)
    assert record['selected_target']=='log' and record['selected_blend']==1
    signatures=collections.defaultdict(list)
    for key in TRAIN:
        c=p.CLIPS[key];sig=p.json.dumps(dict(camera_mode=c['camera_mode'],calibration=c['calibration']),sort_keys=True)
        signatures[sig].append(key)
    sig=next(x for x,keys in signatures.items() if FLIR[0] in keys);assert set(signatures[sig])==set(FLIR)
    folds={'refit68':[],**{'omit_'+k:[k] for k in FLIR},'omit_flir_configuration':FLIR}
    paths=[Path(__file__),ROOT/'scripts/train_speed_regressor_v3.py',ROOT/'models/flow_speed_v3/training_record.json',ROOT/'models/flow_speed_v3/model.pkl']
    for key in TRAIN:
        f=ROOT/f'data/features/flow_speed_v2/{key}.npz';assert p.sha(f)==record['train_cache_sha256'][key];paths.append(f)
    with (ROOT/'models/flow_speed_v3/model.pkl').open('rb') as f:model=pickle.load(f)
    audit=[]
    for split,keys in [('train',FLIR),('test',TEST_FLIR)]:
        for key in keys:
            c=source(key,split);prior,s,d=prior_and_gate(c,model);paths+=c['dependency']
            assert abs(d['correlation']-c['record']['speed_correlation'])<1e-9,(key,d,c['record'])
            inner=c['water'] if c['record']['learned_speed_weight']==0 else prior
            np.testing.assert_allclose(inner,c['old_base'],rtol=0,atol=1e-10)
            state=ROOT/f'runs/domain_v76_direction_guarded_speed/{split}/base/signals/{key}.npz'
            z=np.load(state);np.testing.assert_allclose(s,z['speed'],rtol=0,atol=1e-10)
            audit.append(dict(key=key,split=split,**d,prior_cv=float(np.std(prior)/np.mean(prior)),
                base_speed_cv=float(np.std(s)/np.mean(s)),correction_active_fraction=float(np.mean(abs(z['direction_guarded_quarter'])>1e-12)),
                correction_max_pct=float(100*np.max(abs(np.exp(z['direction_guarded_quarter'])-1)))))
            paths.append(state)
    p.save(REP/'routing_audit.json',dict(rows=audit,train_pass=sum(r['gate_pass'] for r in audit if r['split']=='train'),
        test_pass=sum(r['gate_pass'] for r in audit if r['split']=='test'),original_model_train_members=FLIR,
        note='After the outer FLIR correlation gate, the HGB prior is only a gate input: accepted speed is water; rejected speed is constant1.'))
    p.save(REP/'plan.json',dict(created_utc=p.now(),folds={f:dict(train_keys=[k for k in TRAIN if k not in out],heldout_keys=out) for f,out in folds.items()},
        algorithm=record['parameters'],target='same original log(normalized smoothed GT speed + .05)',
        inference='Same original feature preprocessing, normalization, correlation >=.7, and frozen downstream image-only rules. No refit on VAL or TEST.',
        group=dict(signature=p.json.loads(sig),heldout=FLIR,meaning='Configuration/modality holdout, NOT a verified boat ID. Cross-camera same-boat leakage cannot be excluded.'),
        frozen_choices='Existing selected target/log, blend1 and HGB hyperparameters held fixed. This is a component-exclusion audit, not nested CV of historical choices.',
        downstream='If the gate yields identical base speed, original correction path is exact and unchanged. If different, recompute V73/V74/V76 speed fits and direction support from cached image-only observations, holding the legacy geometry yaw fixed to isolate the speed branch.',
        scope='No TEST trajectory scoring or submission. TEST routing is a read-only audit of actual deployed states.',
        dependencies={str(f.relative_to(ROOT)):p.sha(f) for f in paths},preserved_submissions=p.read(p.REP/'plan.json')['preserved_submissions']))
    print('V103_FROZEN',len(folds),'models; train gate passes',sum(r['gate_pass'] for r in audit if r['split']=='train'),flush=True)


def train():
    plan=p.read(REP/'plan.json');MOD.mkdir(parents=True,exist_ok=True)
    for name,fold in plan['folds'].items():
        dest=MOD/f'{name}.pkl'
        if dest.exists():continue
        xx=[];yy=[];label_keys=[];beg=time.perf_counter()
        for key in fold['train_keys']:
            assert key not in fold['heldout_keys'] and p.CLIPS[key]['split']=='train'
            with np.load(ROOT/f'data/features/flow_speed_v2/{key}.npz') as z:xx.append(old.inputs(z));t=z['times']
            rt,rp=p.load_reference(p.DATA,p.CLIPS[key]);rp=gaussian_filter1d(rp,2/np.median(np.diff(rt)),axis=0)
            target=np.interp(t,rt,np.linalg.norm(np.gradient(rp,rt,axis=0),axis=1));target/=max(target.mean(),.01)
            yy.append(np.log(target+.05));label_keys.append(key)
        model=old.model();model.fit(np.concatenate(xx),np.concatenate(yy))
        with dest.open('wb') as f:pickle.dump(model,f)
        p.save(MOD/f'{name}.json',dict(created_utc=p.now(),train_keys=label_keys,excluded_keys=fold['heldout_keys'],
            samples=sum(len(a) for a in yy),features=xx[0].shape[1],seconds=time.perf_counter()-beg,
            evaluation_labels_used=False,sha256=p.sha(dest),parameters=model.get_params()))
        print('V103_TRAINED',name,len(label_keys),round(time.perf_counter()-beg,1),flush=True)
        del xx,yy,model;gc.collect()


def replay(key,speed):
    """Recompute downstream speed fitting when the outer gate changes."""
    from v19_motion import fit_motion,basis
    from v74_observable_speed import signals
    from v76_direction_guarded_speed import direction_support
    import v76_direction_guarded_speed as v76
    import v74_observable_speed as v74
    from v68_consensus_speed import envelope
    raw=dict(np.load(ROOT/f'runs/domain_v74_observable_speed/train/raw/{key}.npz'))
    oldstate=np.load(ROOT/f'runs/domain_v76_direction_guarded_speed/train/base/signals/{key}.npz')
    if np.array_equal(speed,oldstate['speed']):
        return speed*np.exp(oldstate['direction_guarded_quarter']),dict(replayed=False,identical_base=True,
            correction_active_fraction=float(np.mean(abs(oldstate['direction_guarded_quarter'])>1e-12)))
    t,y=raw['times'],raw['yaw'];oldrows=p.read(ROOT/f'runs/domain_v74_observable_speed/train/raw/{key}.json')['banks']
    raw['speed']=speed.copy()
    for bank,rows in oldrows.items():
        ds=[];valid=[]
        for item in rows:
            log,d=fit_motion(t,y,speed,item['rows'],'speed');ds.append(log)
            support=basis(t,item['rows'])[2] if not d['fallback'] and d.get('optimizer_success',False) else np.zeros(len(t))
            valid.append(support>0)
        raw[bank+'_delta']=np.array(ds).reshape(raw[bank+'_delta'].shape)
        raw[bank+'_valid']=np.array(valid).reshape(raw[bank+'_valid'].shape)
    gated=raw['base_delta'].copy()
    for m in range(3):
        if not raw['backend'][m]:gated[m]=0
    oldlog=envelope(gated.reshape(-1,len(t)))
    stable=envelope(np.array([signals(raw,b)['pose_consensus'] for b in ['base','fresh','even_rows','odd_rows']]))
    joint=envelope(np.array([stable,signals(raw,'base')['pose_consensus']]))
    neweligible=(abs(oldlog)<=1e-12)&(not np.all(raw['backend'][:3]))
    # Reuse the exact existing function with an isolated raw state path. It uses
    # cached SLAM/R3 poses and the provided base speed; no labels are read.
    temp=OUT/'replay_state';p.npzsave(temp/f'train/raw/{key}.npz',**raw)
    previous=v74.OUT
    try:
        v74.OUT=temp;confidence,diag=direction_support(key,'train','base')
    finally:v74.OUT=previous
    correction=oldlog+.25*np.where(neweligible,joint,0)*confidence
    p.npzsave(OUT/f'replay_signals/{key}.npz',times=t,base_speed=speed,correction=correction,confidence=confidence)
    return speed*np.exp(correction),dict(replayed=True,identical_base=False,
        correction_active_fraction=float(np.mean(abs(correction)>1e-12)),max_correction_pct=float(100*np.max(abs(np.exp(correction)-1))))


def predict():
    p.guard();plan=p.read(REP/'plan.json')
    for path,h in plan['dependencies'].items():assert p.sha(ROOT/path)==h,path
    files={};rows=[];models={}
    for name in plan['folds']:
        with (MOD/f'{name}.pkl').open('rb') as f:models[name]=pickle.load(f)
    for key in FLIR:
        c=source(key);base=p.case(key);t=c['t']
        with (ROOT/'models/flow_speed_v3/model.pkl').open('rb') as f:original=pickle.load(f)
        op,os,od=prior_and_gate(c,original)
        for method,model in [('original68',original),('refit68',models['refit68']),('loco',models['omit_'+key]),('configuration_holdout',models['omit_flir_configuration'])]:
            prior,s,d=prior_and_gate(c,model)
            if method=='refit68':np.testing.assert_allclose(prior,op,rtol=0,atol=1e-10)
            fs,rd=replay(key,s);pred=dict(base['pred'],positions=p.integrate(t,base['yaw'],fs).tolist())
            if method=='original68':np.testing.assert_allclose(pred['positions'],base['pred']['positions'],rtol=0,atol=1e-8)
            dest=OUT/f'predictions/{method}/{key}.json';p.save(dest,pred);files[str(dest.relative_to(ROOT))]=p.sha(dest)
            zp=OUT/f'signals/{method}/{key}.npz';p.npzsave(zp,times=t,yaw=base['yaw'],speed=fs,base_speed=s,prior=prior,water=c['water']);files[str(zp.relative_to(ROOT))]=p.sha(zp)
            rows.append(dict(key=key,method=method,**d,**rd,gate_changed=d['gate_pass']!=od['gate_pass'],
                 max_base_speed_change=float(np.max(abs(s-os))),max_final_speed_change=float(np.max(abs(fs-base['speed'])))))
    p.save(REP/'predictions_frozen.json',dict(created_utc=p.now(),files=files,reference_access_blocked=True))
    p.save(REP/'routing_retrained.json',dict(rows=rows,full_refit_parity=True))
    print('V103_PREDICTED',len(rows),flush=True)


def evaluate():
    frozen=p.read(REP/'predictions_frozen.json')
    for path,h in frozen['files'].items():assert p.sha(ROOT/path)==h,path
    rows=[]
    for key in FLIR:
        base=p.case(key);rt,rp=p.load_reference(p.DATA,p.CLIPS[key])
        for method in ['original68','refit68','loco','configuration_holdout']:
            pred=p.read(OUT/f'predictions/{method}/{key}.json');s=np.load(OUT/f'signals/{method}/{key}.npz')['speed']
            rows.append(p.score(key,method,base['t'],np.array(pred['positions']),rt,rp))
            if method in ['loco','configuration_holdout']:
                c=dict(base,speed=s,pred=pred)
                previous=p.OUT
                try:
                    p.OUT=OUT/'oracles'/method;paths,_=p.oracle_paths(c,rt,rp)
                finally:p.OUT=previous
                for name in ['GT_course_V87_speed','V87_yaw_GT_speed']:
                    rows.append(p.score(key,method+'__'+name,base['t'],paths[name],rt,rp))
        rows.append(p.score(key,'constant_speed',base['t'],p.integrate(base['t'],base['yaw'],np.ones(len(base['t']))),rt,rp))
        previous=p.OUT
        try:
            p.OUT=OUT/'oracles'/'constant';paths,_=p.oracle_paths(dict(base,speed=np.ones(len(base['t']))),rt,rp)
        finally:p.OUT=previous
        rows.append(p.score(key,'constant_speed__GT_course',base['t'],paths['GT_course_V87_speed'],rt,rp))
    methods=list(dict.fromkeys(r['method'] for r in rows));summary={}
    base={r['key']:r['drift_pct'] for r in rows if r['method']=='original68'}
    ref=p.read(p.REP/'results.json')['groups'];w=ref['wide']['V87']['mean_drift_pct'];ir=ref['flir-IR']['V87']['mean_drift_pct']
    for name in methods:
        rr=[r for r in rows if r['method']==name];dd=[r['drift_pct']-base[r['key']] for r in rr]
        mean=float(np.mean([r['drift_pct'] for r in rr]));summary[name]=dict(mean_drift_pct=mean,improved=sum(d<-1e-8 for d in dd),
            worsened=sum(d>1e-8 for d in dd),scored=len(rr),weighted_development_drift=(8*w+11*mean+ir)/20)
    p.save(REP/'results.json',dict(created_utc=p.now(),rows=rows,summary=summary,scope='Component exclusion diagnostic, repeated TRAIN8; not boat-disjoint or nested validation. Constant-speed weighted score is a counterfactual, not a TEST estimate.'))
    print('V103_RESULTS',summary,flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['freeze','train','predict','evaluate']);a=ap.parse_args();globals()[a.action]()
