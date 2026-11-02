"""User-requested submission: fixed raw DPVO3 family median, no CLAHE or cap."""
import argparse,copy,datetime,shutil,sys
from v13_common import ROOT,TRAIN,base_case,save_json,np,json,gaussian_filter1d,integrate_velocity
from v13_infer import sources
from v5_common import planar_yaw
from public_dataset import sha256,validate_prediction
import v6_test_common as test

REP=ROOT/'reports/domain_v24_submission'
OUT=ROOT/'runs/domain_v24_submission'
DEST=ROOT/'submissions/macvi27_vo_v24_raw3_uncapped'
BASE=ROOT/'submissions/macvi27_vo_v18_long_graph_cap3/submission.json'
BASE_SHA='5c94a488c02f18407a2dccba76e26956d903e48f2642b9af4fd87f6106f06faf'
SEEDS=[0,104729,209759]

def input_files(key,split):
    frame_root=ROOT/('runs/domain_v5' if split=='train' else 'runs/domain_v6_test')
    pi=ROOT/('runs/domain_v3/pi3x_rgb' if split=='train' else 'runs/domain_v13/pi3x_test')
    vg=ROOT/('runs/domain_probe_v2/vggt' if split=='train' else 'runs/submission_test_v1/vggt')
    files=[frame_root/f'dpvo/random_s{s}/{key}.npz' for s in SEEDS]
    files += [frame_root/f'dpvo/random_s{s}/{key}.json' for s in SEEDS]
    files += [frame_root/f'frames/{key}/index.json',ROOT/f'runs/domain_v4/calibrated_rotation_{split}/{key}.npz',ROOT/f'runs/domain_v18/{split}/signals/{key}.npz']
    for folder in [pi/'windows'/key,vg/'windows'/key]:
        windows=sorted(folder.glob('*.npz'));assert windows,folder;files+=windows
    return files

def freeze():
    settings=dict(method='raw3_hierarchical',seeds=SEEDS,clahe=False,heading_cap_deg=None,
        dpvo='Existing frozen RANDOM96 DPVO,2Hz640wide, fixed image-estimated learned K. Reuse complete three-pass caches.',
        fusion='Planar yaw, Gaussian2s per DPVO seed, median across3 DPVO passes, then initial-angle-aligned median across DPVO/VGGT/Pi3X. No extra correction, heading gain or clipping.',
        speed='Exact TideGraph V18 speed. Same times and frame indices.',
        routing='Apply identical frozen rule to all11 FLIR-VIS TEST clips. Other9 predictions exact V18 copy. No per-clip ID overrides or TEST tuning.',
        failure='Stop on missing/incomplete/nonfinite inputs; no silent fallback.',
        train_mean=9.532022663626405,official_baseline=14.07,new_official_score=None,
        scope='Experimental comparison explicitly requested by user. No reference access, no upload, no final selection.')
    assert sha256(BASE)==BASE_SHA
    if (REP/'plan.json').exists():assert json.loads((REP/'plan.json').read_text())['settings']==settings;return
    files=[BASE,ROOT/'scripts/v24_pack_uncapped.py',ROOT/'scripts/v13_infer.py',ROOT/'scripts/v6_test_dpvo.py',ROOT/'scripts/v5_dpvo_masked.py',ROOT/'scripts/domain_fusion_v2.py',ROOT/'scripts/fuse_headings.py',ROOT/'repos/DPVO/dpvo.pth',ROOT/'repos/DPVO/config/default.yaml',ROOT/'repos/lookout-vo-challenge/starter/pack_submission.py',test.MANIFEST,
        ROOT/'reports/domain_v23_clahe/results.json',ROOT/'reports/domain_v23_clahe/validation.json']
    for split,keys in [('train',TRAIN),('test',test.KEYS)]:
        for key in keys:files+=input_files(key,split)
    previous=[p for p in (ROOT/'submissions').glob('*/submission.json') if p!=DEST/'submission.json']
    save_json(REP/'plan.json',dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),settings=settings,
        dependencies={str(p.relative_to(ROOT)):sha256(p) for p in sorted(set(files))},
        preserved_submissions={str(p.relative_to(ROOT)):sha256(p) for p in previous}))

def guard():
    import public_dataset,metrics
    blocked=[public_dataset.load_reference,metrics.score_trajectory]
    def forbidden(*args,**kwargs):raise RuntimeError('Reference loading/scoring forbidden for submission creation')
    for module in list(sys.modules.values()):
        if module is None:continue
        for name,value in list(vars(module).items()):
            if any(value is fn for fn in blocked):setattr(module,name,forbidden)

def infer(key,split):
    case,_,_=base_case(key,split);z=np.load(ROOT/f'runs/domain_v18/{split}/signals/{key}.npz')
    t,y,s=z['times'],z['joint_depth_graph'],z['speed'];assert np.array_equal(t,case['t'])
    src=sources(key,case,split);raw=[]
    frame_root=ROOT/('runs/domain_v5' if split=='train' else 'runs/domain_v6_test')
    meta=json.loads((frame_root/f'frames/{key}/index.json').read_text());assert np.array_equal(t,np.array(meta['times']))
    for seed in SEEDS:
        p=frame_root/f'dpvo/random_s{seed}/{key}.npz';zz=np.load(p);m=zz['camera_to_world']
        assert np.array_equal(t,zz['times']) and m.shape==(len(t),4,4) and np.isfinite(m).all()
        rotation=m[:,:3,:3];assert np.allclose(rotation.transpose(0,2,1)@rotation,np.eye(3),atol=1e-5)
        yy=np.interp(t,zz['times'],planar_yaw(m));assert np.isfinite(yy).all()
        raw.append(gaussian_filter1d(yy,2/np.median(np.diff(t))))
    dp=np.median(raw,axis=0);family=np.array([a-a[0]+y[0] for a in [dp,src['vggt'],src['pi3x']]])
    yaw=np.median(family,axis=0);assert np.isfinite(yaw).all()
    pred=dict(case['pred'],positions=integrate_velocity(t,yaw,s).tolist())
    save_json(OUT/split/'predictions'/f'{key}.json',pred)
    path=OUT/split/'signals'/f'{key}.npz';path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path,times=t,speed=s,yaw=yaw,baseline_yaw=y,family_yaws=family,dpvo_yaws=raw)
    diagnostic=dict(max_change_from_tidegraph_deg=float(np.rad2deg(abs(yaw-y).max())),
        fraction_exceeding_old3deg=float(np.mean(abs(yaw-y)>np.deg2rad(3))),family_span_p95_deg=float(np.percentile(np.rad2deg(np.ptp(family,axis=0)),95)),
        frames=len(t),speed_exact_tidegraph=True,cap_applied=False)
    save_json(OUT/split/'diagnostics'/f'{key}.json',diagnostic)
    return pred,diagnostic

def main():
    freeze();guard();plan=json.loads((REP/'plan.json').read_text())
    assert json.loads((ROOT/'reports/domain_v23_clahe/validation.json').read_text())['passed']
    train_check={}
    for key in TRAIN:
        pred,_=infer(key,'train');old=json.loads((ROOT/f'runs/domain_v20_five_pass/predictions/raw3_hierarchical/{key}.json').read_text())
        error=float(np.max(abs(np.array(pred['positions'])-np.array(old['positions']))));assert error<1e-9
        train_check[key]=dict(max_position_difference=error,times_identical=pred['times']==old['times'])
    print('TRAIN_RULE_REPRODUCED',len(train_check),flush=True)
    original=json.loads(BASE.read_text());payload=copy.deepcopy(original);diags={}
    for key in test.KEYS:
        pred,diag=infer(key,'test');payload[key]=pred;diags[key]=diag
        print('TEST',key,'max_heading_change_deg',round(diag['max_change_from_tidegraph_deg'],3),flush=True)
    assert set(payload)==set(test.CLIPS) and len(payload)==20
    checks={}
    for key,pred in payload.items():
        ts=test.load_timestamps(test.DATA,test.CLIPS[key]);validate_prediction(pred,key,ts)
        t=np.array(pred['times']);positions=np.array(pred['positions'])
        assert set(pred)=={'key','frame_indices','times','positions'}
        assert pred['times']==original[key]['times'] and pred['frame_indices']==original[key]['frame_indices']
        assert abs(t[0]-ts[0])<1e-6 and abs(t[-1]-ts[-1])<1e-6 and np.diff(t).max()<=1.000001
        assert np.isfinite(positions).all() and np.ptp(positions,axis=0).max()>1e-6
        if key not in test.KEYS:assert pred==original[key]
        else:
            z=np.load(OUT/f'test/signals/{key}.npz');old=np.load(ROOT/f'runs/domain_v18/test/signals/{key}.npz')
            assert np.array_equal(z['speed'],old['speed'])
            assert np.max(abs(positions-integrate_velocity(t,z['yaw'],z['speed'])))<1e-9
            assert np.array_equal(z['yaw'],np.median(z['family_yaws'],axis=0))
        save_json(OUT/'final/predictions'/f'{key}.json',pred)
        checks[key]=dict(poses=len(t),max_gap_seconds=float(np.diff(t).max()),released_timestamp_extent_complete=True,
            changed_from_tidegraph=pred!=original[key])
    # Use the official reference-free packer in the same guarded process.
    from pack_submission import pack
    DEST.mkdir(parents=True,exist_ok=True);bundle=DEST/'submission.json'
    if bundle.exists():assert json.loads(bundle.read_text())==payload
    else:pack(test.MANIFEST,OUT/'final',bundle,'test')
    assert json.loads(bundle.read_text())==payload and bundle.stat().st_size<64*1024*1024
    for p,h in plan['dependencies'].items():assert sha256(ROOT/p)==h,p
    for p,h in plan['preserved_submissions'].items():assert sha256(ROOT/p)==h,p
    changed=[k for k in payload if payload[k]!=original[k]]
    info=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),method='TideGraph-U',method_id='raw3_hierarchical',
        submission=str(bundle),sha256=sha256(bundle),bytes=bundle.stat().st_size,clips=20,
        seeds=SEEDS,clahe=False,heading_cap_deg=None,train_mean=9.532022663626405,official_baseline=14.07,new_official_score=None,
        changed_flir=changed,preserved_other=[k for k in payload if k not in test.KEYS],baseline_sha256=BASE_SHA,
        speed_exact_tidegraph=True,uploaded=False,notes='Experimental comparison. Six TRAIN clips improve and one worsens vs cap3; gain depends materially on clip090. No TEST labels or per-clip tuning.',
        diagnostics=diags)
    validation=dict(passed=True,official_packer_equal=True,reference_loading_blocked=True,scoring_blocked=True,
        train_rule_reproduced=train_check,preserved_submissions=len(plan['preserved_submissions']),dependency_hashes_checked=len(plan['dependencies']),checks=checks,
        scope='Local schema, full released timestamp extent and max gaps verified. Server determines official coverage/score.')
    save_json(DEST/'provenance.json',info);save_json(REP/'submission_candidate.json',info)
    save_json(DEST/'validation.json',validation);save_json(REP/'validation.json',validation)
    (DEST/'SHA256SUMS').write_text(info['sha256']+'  submission.json\n')
    (DEST/'submission_notes.txt').write_text('Upload only submission.json.\nModel name suggestion: TideGraph-U\n'
        'Configuration: Raw DPVO3 median, then median with VGGT and Pi3X headings. No CLAHE, no heading cap. Same existing speed.\n'
        'All20 TEST clips;11 FLIR predictions use the frozen new rule,9 other predictions preserved exactly from TideGraph14.07.\n'
        'TRAIN8 mean9.5320226636%. Official TEST score unknown. Keep the existing official-best submission.\n'
        'Description: Frozen pretrained DPVO orientations from three raw video passes are fused by median, then combined by median with VGGT 1B and Pi3X headings without an angular correction cap on FLIR clips. Existing camera specific motion and speed estimates are retained for other clips. Released camera calibration and timestamps and external pretrained weights are used. Settings are fixed using public train data only.\n'
        'Metadata and external pretrained data used. Hardware: NVIDIA RTX 5080. End-to-end FPS not measured.\n')
    code=DEST/'provenance_code';code.mkdir(exist_ok=True)
    for name in ['v24_pack_uncapped.py','v13_infer.py','v6_test_dpvo.py','v5_dpvo_masked.py','domain_fusion_v2.py','fuse_headings.py']:
        shutil.copy2(ROOT/'scripts'/name,code/name)
    shutil.copy2(REP/'plan.json',code/'plan.json')
    print('SUBMISSION_READY',bundle,'bytes',info['bytes'],'sha256',info['sha256'],flush=True)

if __name__=='__main__':main()
