"""Separate missing speed evidence from contradictory observed motion.

V65 headings are immutable. Pose observations are extracted without references;
evaluation runs in a separate process. No existing submission is overwritten.
"""
import argparse, datetime
from collections import Counter
import v73_pack_independent_speed as old
d, v, np, json, ROOT, save_json = old.d, old.v, old.np, old.json, old.ROOT, old.save_json
motion = v.core.local.motion
OUT = ROOT/'runs/domain_v74_observable_speed'
REP = ROOT/'reports/domain_v74_observable_speed'
METHODS = ['v65_control', 'v73_control', 'missing_aware', 'pose_consensus',
           'pose_consensus_half', 'family_median', 'r3_only', 'slam_only']
BANKS = ['base'] + [f'omit_seed{s}' for s in d.SEEDS] + [f'omit_interval{i}' for i in range(4)] + ['fresh', 'even_rows', 'odd_rows']
FRESH = [(4,20), (4,28), (4,36)]

def freeze():
    path=REP/'plan.json'
    if path.exists(): return
    files=[ROOT/'scripts/v74_observable_speed.py', ROOT/'scripts/v19_motion.py', ROOT/'scripts/v65_submission_inputs.py']
    for split, keys in [('train',v.TRAIN),('test',d.test.KEYS)]:
        for key in keys:
            files += [d.OUT/f'{split}/signals/{key}.npz', old.OUT/f'{split}/signals/{key}.npz']
            rroot='runs/domain_probe_v2/r3' if split=='train' else 'runs/submission_test_v1/r3'
            files += [ROOT/f'{rroot}/diagnostics/{key}.poses.npz']
            for seed in d.SEEDS:
                q=d.pose_path(key,split,seed); files += [q,q.with_suffix('.json')]
    save_json(path,dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), methods=METHODS,banks=BANKS,
        primary='missing_aware', secondary='pose_consensus',
        rules='V65 yaw and base speed fixed. Each estimator requires >=ceil(.75*Nintervals) locally supported successful fits. All valid observed signs must agree; observed zero is NOT missing. SLAM requires >=2 seed streams; these are one model family, not independent model votes. R3 is mandatory as a separate family. Smallest magnitude agreement, log1.1 cap. missing_aware retains BA gate; pose_consensus replaces BA-has-run gate with finite poses/no BA errors plus original horizontal and temporal support. Half and median are predeclared ablations only.',
        audit='All inactivity reasons explicitly counted before scoring. Base plus leave-one-seed/interval, fresh phase4 lengths20/28/36, alternate observation rows. Thresholds fixed before scores; TEST support not used to select thresholds. Geometric confirmation is a separate subsequent stage.',
        promotion='Compare against both V65 and V73. Screen: gain>=.05pp vs V65, maxharm<=.2pp vs both, mean not worse than V73, improvement remains excluding best vs V65. Deployment recommendation additionally requires all sensitivity means improve V65, maxharm<=.2pp and improvement excluding best. Same reused TRAIN8; no independent validation claim.',
        dependencies={str(p.relative_to(ROOT)):v.audit.sha256(p) for p in files},
        preserved={str(p.relative_to(ROOT)):v.audit.sha256(p) for p in (ROOT/'submissions').glob('*/submission.json')}))

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

def infer(split):
    v.core.bg.guard()
    for key in (v.TRAIN if split=='train' else d.test.KEYS):
        z=np.load(OUT/f'{split}/raw/{key}.npz');case,_,_,_,_,_=v.core.g.provider(key,split)
        for bank in BANKS:
            logs=signals(z,bank);logs=dict(v65_control=np.zeros(len(z['times'])),v73_control=np.load(old.OUT/f'{split}/signals/{key}.npz')['log_speed_delta'],**logs)
            p=OUT/f'{split}/{bank}/signals/{key}.npz';p.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(p,times=z['times'],yaw=z['yaw'],speed=z['speed'],**logs)
            for name,log in logs.items():
                save_json(OUT/f'{split}/{bank}/predictions/{name}/{key}.json',dict(case['pred'],positions=v.core.integrate_velocity(z['times'],z['yaw'],z['speed']*np.exp(log)).tolist()))
        print('V74_INFER',split,key,flush=True)

def audit():
    rows=[]
    for split,keys in [('train',v.TRAIN),('test',d.test.KEYS)]:
        for key in keys:
            q=json.loads((OUT/f'{split}/raw/{key}.json').read_text());oldq=json.loads((old.OUT/f'{split}/diagnostics/{key}.json').read_text())
            counts=Counter();rescue=0
            for new,prior in zip(sorted(q['banks']['base'][:12],key=lambda x:(d.INTERVALS.index((x['phase'],x['length'])),x['model'])),oldq['fits'][:12]):
                if prior['fit']['fallback']:
                    reason='backend_never_ran' if not new['backend'] and new['healthy'] else 'backend_error' if not new['healthy'] else 'temporal_support'
                    counts[reason]+=1
                    if not new['fit']['fallback']:rescue+=1
            z=np.load(OUT/f'{split}/base/signals/{key}.npz')
            rows.append(dict(split=split,key=key,old_inactive_reasons=dict(counts),potentially_usable_pose_fits=rescue,backend=q['backend'],active_fraction={n:float(np.mean(abs(z[n])>1e-12)) for n in METHODS}))
    save_json(REP/'support_audit.json',dict(rows=rows,meaning='Pose availability is not proof of accuracy; geometry must independently verify use.'))

def evaluate():
    from public_dataset import load_reference,load_timestamps,validate_prediction
    from metrics import score_trajectory
    refs={k:load_reference(ROOT/'data/release',v.CLIPS[k]) for k in v.TRAIN};rows=[]
    for bank in BANKS:
        controls={}
        for name in METHODS:
            per={}
            for key in v.TRAIN:
                p=json.loads((OUT/f'train/{bank}/predictions/{name}/{key}.json').read_text());validate_prediction(p,key,load_timestamps(ROOT/'data/release',v.CLIPS[key]))
                per[key]=float(score_trajectory(np.array(p['times']),np.array(p['positions']),*refs[key])['drift_pct'])
            if name.endswith('_control'):controls[name]=per
            a=v.audit.compare(per,controls['v65_control']);b=v.audit.compare(per,controls.get('v73_control',per))
            screen=a['mean_delta']<=-.05 and a['worst_regression']<=.2 and a['delta_excluding_best']<0 and b['mean_delta']<=1e-9 and b['worst_regression']<=.2
            rows.append(dict(bank=bank,name=name,mean=float(np.mean(list(per.values()))),per_clip=per,vs_v65=a,vs_v73=b,screen=screen))
            if bank=='base':print('V74_SCORE',name,rows[-1]['mean'],'worst',a['worst_regression'],'vs73',b['mean_delta'],flush=True)
    promotion={n:all(r['screen'] for r in rows if r['name']==n) for n in ['missing_aware','pose_consensus']}
    save_json(REP/'results.json',dict(rows=rows,promotion=promotion,reused_train_clips=8))
    print('V74_PROMOTION',promotion,flush=True)

def validate():
    plan=json.loads((REP/'plan.json').read_text())
    for rel,h in {**plan['dependencies'],**plan['preserved']}.items(): assert v.audit.sha256(ROOT/rel)==h,rel
    # Missing data must differ from a measured zero, and genuine conflict must veto.
    a=np.array([[.03,.03,.03],[.02,0.,-.02]])
    x,_=masked_envelope(a,np.ones_like(a,bool),2);assert np.allclose(x,[.02,0,0])
    mask=np.ones_like(a,bool);mask[1,1]=False;x,_=masked_envelope(a,mask,1);assert x[1]==.03
    total=0
    for split,keys in [('train',v.TRAIN),('test',d.test.KEYS)]:
        for key in keys:
            base=np.load(d.OUT/f'{split}/signals/{key}.npz')
            for bank in BANKS:
                z=np.load(OUT/f'{split}/{bank}/signals/{key}.npz');assert np.array_equal(z['yaw'],base['yaw']) and np.array_equal(z['speed'],base['speed'])
                for name in METHODS:
                    assert np.isfinite(z[name]).all() and abs(z[name]).max()<=np.log(1.1)+1e-9
                    q=json.loads((OUT/f'{split}/{bank}/predictions/{name}/{key}.json').read_text())
                    assert np.allclose(q['positions'],v.core.integrate_velocity(z['times'],z['yaw'],z['speed']*np.exp(z[name])),rtol=0,atol=1e-9);total+=1
    save_json(REP/'validation.json',dict(passed=True,predictions=total,train_evaluations=8*len(BANKS)*len(METHODS),preserved_submissions=len(plan['preserved']),direction_unchanged=True))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['freeze','extract','infer','audit','evaluate','validate']);ap.add_argument('--split',default='train',choices=['train','test']);a=ap.parse_args()
    if a.action in ['extract','infer']:globals()[a.action](a.split)
    else:globals()[a.action]()
