"""Full dependency-removal audits and deployment of frozen V76 primary.

Earlier V76 omit variants intersected a fixed stability anchor. This audit
also removes the omitted evidence from every anchor before evaluating.
"""
import argparse,copy,datetime,shutil
import v76_direction_guarded_speed as q
p,d,v,np,json,ROOT,save_json=q.p,q.d,q.v,q.np,q.json,q.ROOT,q.save_json
OUT=ROOT/'runs/domain_v77_speed_deployment';REP=ROOT/'reports/domain_v77_speed_deployment'
DEST=ROOT/'submissions/macvi27_vo_v77_direction_guarded_speed'
BASE=p.old.BASE
METHOD='direction_guarded_quarter'
AUDITS=['base']+[f'full_omit_seed{s}' for s in d.SEEDS]+[f'full_omit_interval{i}' for i in range(4)]+[f'omit_anchor_{a}' for a in ['base','fresh','even_rows','odd_rows']]+['new_intervals']
NEW_INTERVALS=[(12,12),(12,28),(12,40)]

def freeze():
    if (REP/'plan.json').exists():return
    assert json.loads((q.REP/'results.json').read_text())['selected']==METHOD
    assert json.loads((q.REP/'validation.json').read_text())['passed']
    files=[ROOT/'scripts/v77_speed_deployment.py',ROOT/'scripts/v76_direction_guarded_speed.py',q.REP/'results.json',q.REP/'validation.json',BASE]
    files += list((q.OUT).glob('*/base/signals/*.npz'))
    save_json(REP/'plan.json',dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),selected=METHOD,audits=AUDITS,new_intervals=NEW_INTERVALS,
        reason='Audit full dependency removal: V76 sensitivity intersected a stability anchor that retained allseeds; do not mislabel it fullleaveoneout. Now remove eachseed from allfour anchorbanks, eachoriginal interval frombase/even/odd anchors,eachanchorbank. New12framephase6/14/20sec speedfits. No pickingbestaudit; deploy unchangedV76base onlyif all pass.',
        criteria='Everyaudit improvesV73 mean, maxclipharm<=.2pp, excludingbestimprovesV73. Base requires>=.02ppgain. SameTRAIN8 not independent validation. Speedonlydeployment,V65headingsunchanged,otherTESTclipsidentical. Officialscoreunknown.',
        dependencies={str(x.relative_to(ROOT)):v.audit.sha256(x) for x in files},preserved={str(x.relative_to(ROOT)):v.audit.sha256(x) for x in (ROOT/'submissions').glob('*/submission.json')}))

def new_signals(key,raw):
    t,y,s=raw['times'],raw['yaw'],raw['speed'];_,_,_,r3,_,_=v.core.g.provider(key,'train');mats=[np.load(d.pose_path(key,'train',seed))['camera_to_world'] for seed in d.SEEDS]+[r3];logs=[];valid=[]
    for m in mats:
        for phase,length in NEW_INTERVALS:
            rows,_=p.pose_rows(m,t,phase,length);log,diag=p.motion.fit_motion(t,y,s,rows,'speed')
            support=p.motion.basis(t,rows)[2] if not diag['fallback'] and diag.get('optimizer_success',False) else np.zeros(len(t))
            logs.append(log);valid.append(support>0)
    logs=np.array(logs).reshape(4,3,-1);valid=np.array(valid).reshape(4,3,-1)
    return p.combine(logs,valid,raw['backend'])[0]

def infer():
    v.core.bg.guard()
    for key in v.TRAIN:
        case,_,_,_,_,_=v.core.g.provider(key,'train');raw=np.load(p.OUT/f'train/raw/{key}.npz');z=np.load(p.OUT/f'train/base/signals/{key}.npz');old=z['v73_control'];eligible=(abs(old)<=1e-12)&(not np.all(raw['backend'][:3]));original=np.load(q.OUT/f'train/base/signals/{key}.npz');fresh=new_signals(key,raw)
        logs={}
        for audit in AUDITS:
            if audit=='base':logs[audit]=original[METHOD];continue
            anchors=[]
            for bank in ['base','fresh','even_rows','odd_rows']:
                if audit==f'omit_anchor_{bank}':continue
                ds=raw[bank+'_delta'].copy();valid=raw[bank+'_valid'].copy();omit=None
                if audit.startswith('full_omit_seed'):omit=d.SEEDS.index(int(audit.removeprefix('full_omit_seed')))
                if audit.startswith('full_omit_interval') and bank!='fresh':
                    ix=int(audit.removeprefix('full_omit_interval'));keep=np.arange(ds.shape[1])!=ix;ds=ds[:,keep];valid=valid[:,keep]
                anchors.append(p.combine(ds,valid,raw['backend'],omit_seed=omit)[0])
            if audit=='new_intervals':anchors.append(fresh)
            stable=p.old.speed.envelope(np.array(anchors));bank=audit.removeprefix('full_') if audit.startswith('full_') else 'base'
            confidence,_=q.direction_support(key,'train',bank)
            logs[audit]=old+.25*np.where(eligible,stable,0.)*confidence
        path=OUT/f'audit/signals/{key}.npz';path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,times=z['times'],yaw=z['yaw'],speed=z['speed'],**logs)
        for name,log in logs.items():save_json(OUT/f'audit/predictions/{name}/{key}.json',dict(case['pred'],positions=v.core.integrate_velocity(z['times'],z['yaw'],z['speed']*np.exp(log)).tolist()))
        print('V77_AUDIT',key,flush=True)

def evaluate():
    from public_dataset import load_reference
    from metrics import score_trajectory
    refs={k:load_reference(ROOT/'data/release',v.CLIPS[k]) for k in v.TRAIN};rows=[]
    base=next(r for r in json.loads((q.REP/'results.json').read_text())['rows'] if r['bank']=='base' and r['name']=='v73_control')['per_clip']
    for name in AUDITS:
        per={}
        for key in v.TRAIN:
            pred=json.loads((OUT/f'audit/predictions/{name}/{key}.json').read_text());per[key]=float(score_trajectory(np.array(pred['times']),np.array(pred['positions']),*refs[key])['drift_pct'])
        comp=v.audit.compare(per,base);screen=comp['mean_delta']<=(-.02 if name=='base' else -1e-7) and comp['worst_regression']<=.2 and comp['delta_excluding_best']<0
        rows.append(dict(name=name,mean=float(np.mean(list(per.values()))),per_clip=per,vs_v73=comp,screen=screen));print('V77_SCORE',name,rows[-1]['mean'],comp['worst_regression'],comp['delta_excluding_best'],screen,flush=True)
    save_json(REP/'results.json',dict(rows=rows,all_audits_pass=all(r['screen'] for r in rows),selected=METHOD,base_unchanged=True))

def pack():
    from public_dataset import validate_prediction
    from pack_submission import pack as official_pack
    v.core.bg.guard();plan=json.loads((REP/'plan.json').read_text());results=json.loads((REP/'results.json').read_text())
    assert results['all_audits_pass'],'Failed promotion; retain V65'
    base=json.loads(BASE.read_text());payload=copy.deepcopy(base);checks={}
    for key in d.test.KEYS:payload[key]=json.loads((q.OUT/f'test/base/predictions/{METHOD}/{key}.json').read_text())
    assert set(payload)==set(d.test.CLIPS) and len(payload)==20
    for key,pred in payload.items():
        ts=d.test.load_timestamps(d.test.DATA,d.test.CLIPS[key]);validate_prediction(pred,key,ts);t=np.array(pred['times']);pos=np.array(pred['positions'])
        assert pred['times']==base[key]['times'] and pred['frame_indices']==base[key]['frame_indices'];assert abs(t[0]-ts[0])<1e-6 and abs(t[-1]-ts[-1])<1e-6 and np.diff(t).max()<=1.000001
        extra={}
        if key in d.test.KEYS:
            z=np.load(q.OUT/f'test/base/signals/{key}.npz');old=np.load(d.OUT/f'test/signals/{key}.npz');assert np.array_equal(z['yaw'],old['yaw']) and np.array_equal(z['speed'],old['speed'])
            log=z[METHOD];assert abs(log).max()<=np.log(1.1)+1e-9;assert np.allclose(pos,v.core.integrate_velocity(t,z['yaw'],z['speed']*np.exp(log)),rtol=0,atol=1e-9)
            extra=dict(speed_active_fraction=float(np.mean(abs(log)>1e-12)),max_speed_change_percent=float(100*abs(np.expm1(log)).max()))
        else:assert pred==base[key]
        checks[key]=dict(poses=len(t),changed=pred!=base[key],max_position_difference=float(abs(pos-np.array(base[key]['positions'])).max()),**extra)
        save_json(OUT/f'final/predictions/{key}.json',pred)
    DEST.mkdir(parents=True,exist_ok=True);bundle=DEST/'submission.json'
    if bundle.exists():assert json.loads(bundle.read_text())==payload
    else:official_pack(d.test.MANIFEST,OUT/'final',bundle,'test')
    assert json.loads(bundle.read_text())==payload
    for rel,h in {**plan['dependencies'],**plan['preserved']}.items():assert v.audit.sha256(ROOT/rel)==h,rel
    # Audit predictions independently re-integrated, references only in evaluate.
    for key in v.TRAIN:
        z=np.load(OUT/f'audit/signals/{key}.npz')
        for name in AUDITS:
            pp=json.loads((OUT/f'audit/predictions/{name}/{key}.json').read_text());assert np.allclose(pp['positions'],v.core.integrate_velocity(z['times'],z['yaw'],z['speed']*np.exp(z[name])),rtol=0,atol=1e-9)
    validation=dict(passed=True,clips=20,official_packer_equal=True,all_audits_pass=True,headings_unchanged=True,old_submissions_preserved=len(plan['preserved']),checks=checks,train_evaluations=8*len(AUDITS),official_coverage_unknown=True)
    provenance=dict(version='V77',method=METHOD,train_mean=results['rows'][0]['mean'],official_score=None,sha256=v.audit.sha256(bundle),bytes=bundle.stat().st_size,base=str(BASE),base_sha256=v.audit.sha256(BASE),changed_from_v65=[k for k,z in checks.items() if z['changed']],new_speed_confirmation='Temporal and pose-direction agreement only. V75 third-image reprojection did not confirm new speed, not claimed as supporting evidence.',adaptive_development_on_reused_train8=True)
    save_json(DEST/'validation.json',validation);save_json(REP/'validation.json',validation);save_json(DEST/'provenance.json',provenance);(DEST/'SHA256SUMS').write_text(provenance['sha256']+'  submission.json\n')
    description='Frozen pretrained DPVO, VGGT 1B and Pi3X headings are fused by median. Static DPV-SLAM and CoTracker geometry provide a support gated course correction. R3 and three static DPV-SLAM runs estimate relative speed across multiple temporal intervals. Existing consensus is retained. Additional speed changes are reduced and applied only with temporal and movement direction agreement. AnyCalib estimates intrinsics and Mask2Former selects static features. Released calibration and timestamps and external pretrained weights are used. Settings are developed on public train data.'
    allowed=set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 -:,.()[]_/\\+*%$§!?');assert set(description)<=allowed
    (DEST/'submission_notes.txt').write_text('Upload only submission.json.\nModel name: TideGraph-DS\nHardware: NVIDIA RTX 5080\nMeta data: Yes. Additional datasets: Yes, external pretrained weights.\nFPS: End-to-end measurement unavailable. Do not invent a number.\nDescription: '+description+'\nOfficial score unknown. Reused TRAIN FLIR8 development mean '+str(provenance['train_mean'])+' percent. V65 direction retained. New speed affects only input-supported clips; no TEST labels used.\n')
    code=DEST/'provenance_code';code.mkdir(exist_ok=True)
    for name in ['v74_observable_speed.py','v75_threeview_speed.py','v76_direction_guarded_speed.py','v77_speed_deployment.py']:shutil.copy2(ROOT/'scripts'/name,code/name)
    shutil.copy2(REP/'plan.json',code/'plan.json')
    print('V77_READY',str(bundle),provenance['sha256'],'changed',provenance['changed_from_v65'],flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['freeze','infer','evaluate','pack']);a=ap.parse_args();globals()[a.action]()
