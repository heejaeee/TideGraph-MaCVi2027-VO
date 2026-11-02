"""Separate relative-speed sensitivity from V64 travel-direction correction.

Global monocular scale is eliminated in the fit. Three seeds and four interval
definitions must agree on the sign; no reference speed or per-clip labels.
"""
import argparse,datetime
import v65_submission_inputs as d
v,ROOT,np,json,save_json=d.v,d.ROOT,d.np,d.json,d.save_json
OUT=ROOT/'runs/domain_v68_consensus_speed';REP=ROOT/'reports/domain_v68_consensus_speed'
METHODS=['v24_control','v64_control','speed_only_envelope','combined_envelope','combined_half','combined_median12','without_seed0','without_seed104729','without_seed209759']

def freeze():
    if (REP/'plan.json').exists():return
    files=[ROOT/'scripts/v68_consensus_speed.py',ROOT/'scripts/v19_motion.py',ROOT/'scripts/v65_submission_inputs.py']
    for key in v.TRAIN:
        files += [ROOT/f'runs/domain_v64_course_support_envelope/signals/{key}.npz',ROOT/f'runs/domain_v24_submission/train/signals/{key}.npz']
        for seed in d.SEEDS:
            p=d.pose_path(key,'train',seed);files += [p,p.with_suffix('.json')]
    save_json(REP/'plan.json',dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),primary='combined_envelope',methods=METHODS,
        hypothesis='Remaining trajectory shape error may come from variable travel speed, independently ofheading. Evaluate frozenV19 log-speed fit on staticSLAM horizontal displacement; eliminate pertrajectory global scale.',
        rule='Same3SLAMseeds x4intervals/backendgate asV64; horizontalfraction>=.7. log_distance=log(horizontal displacement),weight1. V19 speedfit aroundfixedV64heading orV24heading,min8intervals/50percentcoverage,30secknots,log1.1cap. Applysmallestsame-signofall12,otherwisezero. Combinedhalf andmedian12ablations. No speed normalization usingreference. SpeedonlycontrolusesV24direction.',
        audits='Leaveeachseedout8conditionenvelope fixedbefore scores; audits arecorrelatedrepeatedinputs,notindependentvalidation. Primarygain>=.05pp,maxharm<=.2pp,excludingbest<0; all3leaveoneout meanimprove,maxharm<=.2pp versusV64. If fails no automaticreplacement.',
        dependencies={str(p.relative_to(ROOT)):v.audit.sha256(p) for p in files},preserved={str(p.relative_to(ROOT)):v.audit.sha256(p) for p in (ROOT/'submissions').glob('*/submission.json')}))

def envelope(ds):
    lo=ds.min(0);hi=ds.max(0);return np.where(lo>0,lo,np.where(hi<0,hi,0.))

def infer():
    v.core.bg.guard()
    for key in v.TRAIN:
        case,z,_,_,_,_=v.inputs(key);t,y,s=z['times'],z['yaw'],z['speed'];candidate=np.load(ROOT/f'runs/domain_v64_course_support_envelope/signals/{key}.npz')['envelope_plus_course']
        ds=[];base_ds=[];diags=[];seed_ids=[]
        for phase,length in d.INTERVALS:
            for seed in d.SEEDS:
                p=d.pose_path(key,'train',seed);zz=np.load(p);m=zz['camera_to_world'];stats=json.loads(p.with_suffix('.json').read_text())
                assert np.array_equal(zz['times'],t)
                usable=stats['loop_patch_edges']>0 and stats['global_ba_successes']>0 and not stats['global_ba_errors']
                f,right=v.core.bg.frame_axes(m);pos=m[:,:3,3];rows=[]
                for i in range(phase,len(t)-length,16):
                    j=i+length;dp=pos[j]-pos[i];norm=np.linalg.norm(dp);hor=np.linalg.norm([dp@f,dp@right])
                    if usable and norm>1e-8 and hor/norm>=.7:rows.append(dict(i=i,j=j,log_distance=float(np.log(hor)),weight=1.))
                log,diag=v.core.local.motion.fit_motion(t,candidate,s,rows,'speed');ds.append(log);seed_ids.append(seed)
                log,base_diag=v.core.local.motion.fit_motion(t,y,s,rows,'speed');base_ds.append(log)
                diags.append(dict(seed=seed,phase=phase,length=length,fit=diag,baseline_fit=base_diag))
        ds=np.asarray(ds);base_ds=np.asarray(base_ds);env=envelope(ds);zero=np.zeros(len(t))
        logs=dict(v24_control=zero,v64_control=zero,speed_only_envelope=envelope(base_ds),combined_envelope=env,combined_half=.5*env,combined_median12=np.median(ds,axis=0))
        for seed in d.SEEDS:logs[f'without_seed{seed}']=envelope(ds[np.array(seed_ids)!=seed])
        headings={name:y if name in ['v24_control','speed_only_envelope'] else candidate for name in METHODS}
        for name in METHODS:
            speed=s*np.exp(logs[name]);save_json(OUT/f'predictions/{name}/{key}.json',dict(case['pred'],positions=v.core.integrate_velocity(t,headings[name],speed).tolist()))
        p=OUT/f'signals/{key}.npz';p.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(p,times=t,speed=s,v24_yaw=y,v64_yaw=candidate,variant_log_deltas=ds,**logs)
        save_json(OUT/f'diagnostics/{key}.json',dict(variants=diags,active_fraction=float(np.mean(abs(env)>1e-12)),max_speed_change_pct=float(100*abs(np.exp(env)-1).max())))
        print('SPEED_INFER',key,float(np.mean(abs(env)>1e-12)),float(100*abs(np.exp(env)-1).max()),flush=True)

def evaluate():
    from public_dataset import load_reference,validate_prediction,load_timestamps
    from metrics import score_trajectory
    refs={k:load_reference(ROOT/'data/release',v.CLIPS[k]) for k in v.TRAIN};rows=[];base=None;v64=None
    for name in METHODS:
        per={}
        for key in v.TRAIN:
            p=json.loads((OUT/f'predictions/{name}/{key}.json').read_text());validate_prediction(p,key,load_timestamps(ROOT/'data/release',v.CLIPS[key]));per[key]=float(score_trajectory(np.array(p['times']),np.array(p['positions']),*refs[key])['drift_pct'])
        if base is None:base=per
        if name=='v64_control':v64=per
        comp=v.audit.compare(per,v64 if v64 is not None else base)
        rows.append(dict(name=name,mean=float(np.mean(list(per.values()))),per_clip=per,vs_v64=comp,vs_v24=v.audit.compare(per,base),screen=comp['mean_delta']<=-.05 and comp['worst_regression']<=.2 and comp['delta_excluding_best']<0))
        print('SPEED_SCORE',name,rows[-1]['mean'],comp['mean_delta'],comp['worst_regression'],flush=True)
    lookup={r['name']:r for r in rows};passed=lookup['combined_envelope']['screen'] and all(lookup[f'without_seed{s}']['vs_v64']['mean_delta']<0 and lookup[f'without_seed{s}']['vs_v64']['worst_regression']<=.2 for s in d.SEEDS)
    save_json(REP/'results.json',dict(rows=rows,primary='combined_envelope',primary_pass=passed))

def validate():
    plan=json.loads((REP/'plan.json').read_text())
    for rel,h in {**plan['dependencies'],**plan['preserved']}.items():assert v.audit.sha256(ROOT/rel)==h,rel
    for key in v.TRAIN:
        z=np.load(OUT/f'signals/{key}.npz');base=np.load(ROOT/f'runs/domain_v24_submission/train/signals/{key}.npz')
        assert np.array_equal(z['times'],base['times']) and np.array_equal(z['speed'],base['speed'])
        assert np.array_equal(z['combined_envelope'],envelope(z['variant_log_deltas']))
        for name in METHODS:
            assert np.isfinite(z[name]).all() and abs(z[name]).max()<=np.log(1.1)+1e-9
            yaw=z['v24_yaw'] if name in ['v24_control','speed_only_envelope'] else z['v64_yaw'];p=json.loads((OUT/f'predictions/{name}/{key}.json').read_text())
            assert np.allclose(p['positions'],v.core.integrate_velocity(z['times'],yaw,z['speed']*np.exp(z[name])),rtol=0,atol=1e-9)
    save_json(REP/'validation.json',dict(passed=True,predictions=8*len(METHODS),preserved_submissions=len(plan['preserved']),no_test_inference=True))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['freeze','infer','evaluate','validate']);a=ap.parse_args();globals()[a.action]()
