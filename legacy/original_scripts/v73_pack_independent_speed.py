"""Deploy the fixed V71 R3/SLAM speed rule after V72 audits passed.

V65/V64 remains immutable. This creates a separate optional TEST submission.
"""
import argparse,copy,datetime,shutil
import v65_submission_inputs as d
import v65_pack_submission as original
import v68_consensus_speed as speed
v,ROOT,np,json,save_json=d.v,d.ROOT,d.np,d.json,d.save_json
OUT=ROOT/'runs/domain_v73_submission';REP=ROOT/'reports/domain_v73_submission'
DEST=ROOT/'submissions/macvi27_vo_v73_independent_speed'
BASE=original.DEST/'submission.json'

def freeze():
    if (REP/'plan.json').exists():return
    assert json.loads((ROOT/'reports/domain_v72_speed_robustness/results.json').read_text())['all_audits_pass']
    assert json.loads((ROOT/'reports/domain_v72_speed_robustness/validation.json').read_text())['passed']
    files=[BASE,ROOT/'scripts/v73_pack_independent_speed.py',ROOT/'scripts/v19_motion.py',ROOT/'scripts/v71_independent_speed.py',ROOT/'scripts/v68_consensus_speed.py',ROOT/'scripts/v65_submission_inputs.py']
    for split,keys in [('train',v.TRAIN),('test',d.test.KEYS)]:
        for key in keys:
            files += [d.OUT/f'{split}/signals/{key}.npz']
            r3root=ROOT/('runs/domain_probe_v2/r3' if split=='train' else 'runs/submission_test_v1/r3')
            files += [r3root/f'diagnostics/{key}.poses.npz']
            for seed in d.SEEDS:
                p=d.pose_path(key,split,seed);files += [p,p.with_suffix('.json')]
    save_json(REP/'plan.json',dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),candidate='V71 R3/SLAM same-sign relative-speed envelope, deployedV73',train_mean=8.66615112040479,
        rule='KeepV65/V64headingunchanged. Fitrelative log-speed from3staticSLAMseeds x4intervals plusR3 x4intervals. Globalmonocularscaleeliminated; sameV19coverage/gates/log1.1cap. Ifall16changes same-sign takeweakest,otherwisezero. Other9clips exactV65copy. NoTESTlabelaccessorpercliprouting.',
        selection='V71primary nominated afterV68/V69failures. V72all8sensitivityaudits pass, same repeatedlyused8TRAINclips. Gains on2clips,notindependentdata. Do not select bestaudit configuration. Officialscoreunknown.',
        dependencies={str(p.relative_to(ROOT)):v.audit.sha256(p) for p in files},preserved={str(p.relative_to(ROOT)):v.audit.sha256(p) for p in (ROOT/'submissions').glob('*/submission.json')}))

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

def pack():
    from public_dataset import validate_prediction
    from pack_submission import pack as official_pack
    v.core.bg.guard();plan=json.loads((REP/'plan.json').read_text());assert json.loads((REP/'train_parity.json').read_text())['passed']
    base=json.loads(BASE.read_text());payload=copy.deepcopy(base);checks={}
    for key in d.test.KEYS:payload[key]=json.loads((OUT/f'test/predictions/{key}.json').read_text())
    assert set(payload)==set(d.test.CLIPS) and len(payload)==20
    for key,q in payload.items():
        ts=d.test.load_timestamps(d.test.DATA,d.test.CLIPS[key]);validate_prediction(q,key,ts);t=np.array(q['times']);pos=np.array(q['positions'])
        assert q['times']==base[key]['times'] and q['frame_indices']==base[key]['frame_indices']
        assert abs(t[0]-ts[0])<1e-6 and abs(t[-1]-ts[-1])<1e-6 and np.diff(t).max()<=1.000001
        if key not in d.test.KEYS:assert q==base[key]
        else:
            z=np.load(OUT/f'test/signals/{key}.npz');old=np.load(d.OUT/f'test/signals/{key}.npz')
            assert np.array_equal(z['yaw'],old['yaw']) and np.array_equal(z['base_speed'],old['speed'])
            assert np.array_equal(z['log_speed_delta'],speed.envelope(z['variant_deltas'])) and abs(z['log_speed_delta']).max()<=np.log(1.1)+1e-9
            assert np.allclose(pos,v.core.integrate_velocity(t,z['yaw'],z['speed']),atol=1e-9,rtol=0)
        checks[key]=dict(poses=len(t),changed=q!=base[key],full_released_timestamp_extent=True,max_gap_seconds=float(np.diff(t).max()))
        save_json(OUT/f'final/predictions/{key}.json',q)
    DEST.mkdir(parents=True,exist_ok=True);bundle=DEST/'submission.json'
    if bundle.exists():assert json.loads(bundle.read_text())==payload
    else:official_pack(d.test.MANIFEST,OUT/'final',bundle,'test')
    assert json.loads(bundle.read_text())==payload
    for rel,h in {**plan['dependencies'],**plan['preserved']}.items():assert v.audit.sha256(ROOT/rel)==h,rel
    validation=dict(passed=True,official_packer_equal=True,reference_loading_blocked=True,scoring_blocked=True,train_parity=True,headings_unchanged=True,preserved_submissions=len(plan['preserved']),checks=checks,scope='Server determines official coverage and score.')
    provenance=dict(candidate=plan['candidate'],sha256=v.audit.sha256(bundle),bytes=bundle.stat().st_size,clips=20,train_mean=plan['train_mean'],official_score=None,base=str(BASE),base_sha256=v.audit.sha256(BASE),changed_from_v65=[k for k,z in checks.items() if z['changed']],uploaded_to_drive=False,
        evidence_hashes={str(p.relative_to(ROOT)):v.audit.sha256(p) for p in OUT.rglob('*.npz')})
    save_json(DEST/'validation.json',validation);save_json(REP/'validation.json',validation);save_json(DEST/'provenance.json',provenance)
    (DEST/'SHA256SUMS').write_text(provenance['sha256']+'  submission.json\n')
    description='Frozen pretrained DPVO, VGGT 1B and Pi3X headings are fused by median. Static background DPV-SLAM and CoTracker three view geometry provide a support gated course correction capped at 3 degrees. Relative speed is adjusted only when R3 and three DPV-SLAM passes agree across four temporal intervals, with a maximum factor of 1.1. AnyCalib estimates intrinsics and Mask2Former selects static features. Released calibration and timestamps and external pretrained weights are used. All settings are fixed on public train data.'
    allowed=set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 -:,.()[]_/\\+*%$§!?');assert set(description)<=allowed
    (DEST/'submission_notes.txt').write_text('Upload only submission.json.\nModel name: TideGraph-IS\nHardware: NVIDIA RTX 5080\nMetadata: Yes. External pretrained data: Yes.\nFPS: End-to-end measurement unavailable; do not invent a value.\nDescription: '+description+'\nRepeated public TRAIN FLIR8 mean8.6661511204%. Two clips improve versusV64,other6unchanged. All8input sensitivityauditspass. OfficialTESTscoreunknown.\n')
    code=DEST/'provenance_code';code.mkdir(exist_ok=True);shutil.copy2(ROOT/'scripts/v73_pack_independent_speed.py',code/'v73_pack_independent_speed.py');shutil.copy2(REP/'plan.json',code/'plan.json');shutil.copy2(REP/'train_parity.json',DEST/'train_parity.json')
    print('V73_READY',bundle,provenance['sha256'],provenance['bytes'],flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['freeze','infer','pack']);ap.add_argument('--split',default='train',choices=['train','test']);a=ap.parse_args()
    if a.action=='infer':infer(a.split)
    else:globals()[a.action]()
