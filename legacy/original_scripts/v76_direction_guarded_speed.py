"""Follow-up after V74: preserve old consensus, shrink only new observations.

This is an adaptive development experiment prompted by V74's regressions,
not independent validation. Speed and direction support are checked separately.
"""
import argparse,datetime
import v74_observable_speed as p
d,v,np,json,ROOT,save_json=p.d,p.v,p.np,p.json,p.ROOT,p.save_json
OUT=ROOT/'runs/domain_v76_direction_guarded_speed';REP=ROOT/'reports/domain_v76_direction_guarded_speed'
METHODS=['v65_control','v73_control','stable_residual_quarter','direction_guarded_quarter','direction_guarded_half','unfiltered_residual_quarter']
BANKS=p.BANKS+['direction_tight2','direction_loose4']

def freeze():
    if (REP/'plan.json').exists():return
    files=[ROOT/'scripts/v76_direction_guarded_speed.py',ROOT/'scripts/v74_observable_speed.py',p.REP/'results.json',p.REP/'support_audit.json']
    for split,keys in [('train',v.TRAIN),('test',d.test.KEYS)]:
        for key in keys:
            files += [p.OUT/f'{split}/raw/{key}.npz',p.OUT/f'{split}/raw/{key}.json',d.OUT/f'{split}/signals/{key}.npz']
            files += list((p.OUT/f'{split}').glob(f'*/signals/{key}.npz'))
    save_json(REP/'plan.json',dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),methods=METHODS,banks=BANKS,primary='direction_guarded_quarter',secondary='stable_residual_quarter',
        reason='V74 removing BA eligibility worsens075/084, globalhalf also shrinks already-supported001gain. Preserve V73 entirely and treat newly admitted no-BA poses as a low-confidence residual. This design follows observed V74 TRAIN results; exploratory, not a new holdout.',
        rule='Only clips with at leastone missingBA seed can receive new residual; existingV73 nonzero corrections preserved. New signal must agree same-sign and minimum amplitude across original intervals,fresh intervals,and even/odd observation rows (all two-family consensuses). Newly admitted log-speed uses fixedquarter strength (max2.41percentincrease), not perclipalpha. Half diagnostic and withoutstability diagnostic frozen now. Truefamilyconflict remains zero.',
        direction='For each estimator, interval horizontal displacement course compared with frozenV65 integrated direction. Subtract one global median angle offset per estimator, which uses input trajectories only. At30secknots need>=3distinct intervals/model within30sec and>=2SLAM seeds plusR3. All observed model median residuals must be within3deg ofV65 and mutual range<=3deg. Confidence=clip(1-max(maxabs,range)/3deg,0,1). Interpolateconfidence; cap affects only new speed residual, not heading. 2/4deg sensitivity only, never select a better audit.',
        promotion='Primary base must improveV73>=.02pp, maxharm<=.2pp vsV73 andV65, improvement excludingbest versusV73. Every13bank must improveV73 and maxharm<=.2pp. Secondary same criteria if primary fails; fixed order, notlowestmean. ReusedTRAIN8; TESTnonzerochangeisdeploymentchecknotaccuracy.',
        dependencies={str(q.relative_to(ROOT)):v.audit.sha256(q) for q in files},preserved={str(q.relative_to(ROOT)):v.audit.sha256(q) for q in (ROOT/'submissions').glob('*/submission.json')}))

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

def evaluate():
    from public_dataset import load_reference,load_timestamps,validate_prediction
    from metrics import score_trajectory
    refs={k:load_reference(ROOT/'data/release',v.CLIPS[k]) for k in v.TRAIN};rows=[]
    for bank in BANKS:
        controls={}
        for name in METHODS:
            per={}
            for key in v.TRAIN:
                q=json.loads((OUT/f'train/{bank}/predictions/{name}/{key}.json').read_text());validate_prediction(q,key,load_timestamps(ROOT/'data/release',v.CLIPS[key]))
                per[key]=float(score_trajectory(np.array(q['times']),np.array(q['positions']),*refs[key])['drift_pct'])
            if name.endswith('_control'):controls[name]=per
            a=v.audit.compare(per,controls['v65_control']);b=v.audit.compare(per,controls.get('v73_control',per))
            gain=.02 if bank=='base' else 1e-7
            screen=b['mean_delta']<=-gain and b['worst_regression']<=.2 and a['worst_regression']<=.2 and b['delta_excluding_best']<0
            rows.append(dict(bank=bank,name=name,mean=float(np.mean(list(per.values()))),per_clip=per,vs_v65=a,vs_v73=b,screen=screen))
            if bank=='base':print('V76_SCORE',name,rows[-1]['mean'],'harm',b['worst_regression'],'exclude',b['delta_excluding_best'],flush=True)
    promotion={n:all(r['screen'] for r in rows if r['name']==n) for n in ['direction_guarded_quarter','stable_residual_quarter']}
    chosen=next((n for n,ok in promotion.items() if ok),None)
    save_json(REP/'results.json',dict(rows=rows,promotion=promotion,selected=chosen,adaptive_followup=True,reused_train_clips=8))
    print('V76_PROMOTION',promotion,'SELECTED',chosen,flush=True)

def validate():
    plan=json.loads((REP/'plan.json').read_text())
    for rel,h in {**plan['dependencies'],**plan['preserved']}.items():assert v.audit.sha256(ROOT/rel)==h,rel
    total=0
    for split,keys in [('train',v.TRAIN),('test',d.test.KEYS)]:
        for key in keys:
            base=np.load(d.OUT/f'{split}/signals/{key}.npz');old=np.load(p.old.OUT/f'{split}/signals/{key}.npz')['log_speed_delta']
            for bank in BANKS:
                z=np.load(OUT/f'{split}/{bank}/signals/{key}.npz');assert np.array_equal(z['yaw'],base['yaw']) and np.array_equal(z['speed'],base['speed'])
                assert np.all((z['direction_confidence']>=0)&(z['direction_confidence']<=1))
                for name in METHODS:
                    log=z[name];assert np.isfinite(log).all() and abs(log).max()<=np.log(1.1)+1e-9
                    if name not in ['v65_control','v73_control']:
                        assert np.array_equal(log[abs(old)>1e-12],old[abs(old)>1e-12])
                        cap=.5 if name=='direction_guarded_half' else .25;assert abs(log-old).max()<=cap*np.log(1.1)+1e-9
                    q=json.loads((OUT/f'{split}/{bank}/predictions/{name}/{key}.json').read_text());assert np.allclose(q['positions'],v.core.integrate_velocity(z['times'],z['yaw'],z['speed']*np.exp(log)),rtol=0,atol=1e-9);total+=1
    save_json(REP/'validation.json',dict(passed=True,predictions=total,train_evaluations=8*len(BANKS)*len(METHODS),preserved_submissions=len(plan['preserved']),old_supported_corrections_preserved=True,direction_unchanged=True))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['freeze','infer','evaluate','validate']);ap.add_argument('--split',default='train',choices=['train','test']);a=ap.parse_args()
    if a.action=='infer':infer(a.split)
    else:globals()[a.action]()
