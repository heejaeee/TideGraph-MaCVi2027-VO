"""Deploy the V82 wide-only rule on TEST; preserve every other V77 prediction."""
import argparse,concurrent.futures,gc,json,shutil,subprocess,sys,time
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
import v81_omega_pipeline as omega
import v82_omega_val as val
from domain_fusion_v2 import relative_window_yaw
from v78_external_data import sha,save

ROOT=omega.ROOT
DATA=ROOT/'data/lookout-vo-test-v0.1'
MANIFEST=DATA/'release_manifest.json'
OUT=ROOT/'runs/domain_v83_omega_wide_test'
REP=ROOT/'reports/domain_v83_omega_wide_submission'
DEST=ROOT/'submissions/macvi27_vo_v83_omega_wide'
BASE=ROOT/'submissions/macvi27_vo_v77_direction_guarded_speed/submission.json'
CLIPS={c['key']:c for c in omega.read(MANIFEST)['clips']}
WIDE={k:c for k,c in CLIPS.items() if c['camera_mode']=='wide'}
VAL_REPORT=val.REP
VAL_RUN=val.OUT
read=omega.read
now=val.now
guard=val.guard

def test_case(key):
    assert key in WIDE
    pred=read(OUT/f'baseline/predictions/{key}.json')
    with np.load(ROOT/f'runs/submission_test_v1/fused/diagnostics/{key}.fusion.npz') as z:
        t,y,s=z['times'],z['yaw'],z['speed']
    assert np.array_equal(t,pred['times'])
    assert np.array_equal(omega.integrate(t,y,s),pred['positions']),key
    return CLIPS[key],pred,t,y,s,None

def direction_prediction(pred,t,base_yaw,speed,source,key):
    yaw=gaussian_filter1d(relative_window_yaw(source,key,len(t)),2/np.median(np.diff(t)))
    yaw=yaw-yaw[0]+base_yaw[0]
    return dict(pred,positions=omega.integrate(t,yaw,speed).tolist()),yaw

def freeze():
    guard()
    if (REP/'plan.json').exists():return
    assert len(CLIPS)==20 and len(WIDE)==8
    base=read(BASE);assert set(base)==set(CLIPS)
    assert all(c['reference'] is None for c in CLIPS.values())
    for k,pred in base.items():save(OUT/f'baseline/predictions/{k}.json',pred)
    parity=[]
    for key in WIDE:
        _,pred,t,_,_,_=test_case(key)
        parity.append(dict(key=key,poses=len(t),baseline_position_error_m=0.0))
    # Reuse the exact prediction adapter on the already frozen VAL wide cases.
    val_parity=[]
    for key,c in val.CLIPS.items():
        if c['camera_mode']!='wide':continue
        _,pred,t,y,s,_=val.case(key)
        actual,_=direction_prediction(pred,t,y,s,VAL_RUN/'omega',key)
        expected=read(VAL_RUN/f'wide_only_posthoc/predictions/{key}.json')
        assert actual==expected,key
        val_parity.append(dict(key=key,position_error_m=0.0))
    files=[BASE,MANIFEST,ROOT/'scripts/v83_omega_wide_submission.py',ROOT/'scripts/v82_omega_val.py',
           ROOT/'scripts/v81_omega_pipeline.py',ROOT/'scripts/domain_fusion_v2.py',ROOT/'scripts/fuse_headings.py',
           ROOT/'repos/lookout-vo-challenge/starter/pack_submission.py',VAL_REPORT/'wide_only_posthoc/results.json',
           omega.REP/'checkpoint_access.json']
    files += [ROOT/f'runs/submission_test_v1/fused/diagnostics/{k}.fusion.npz' for k in WIDE]
    prior=read(VAL_REPORT/'plan.json')
    save(REP/'plan.json',dict(created_utc=now(),base_submission=str(BASE.relative_to(ROOT)),base_sha256=sha(BASE),
        rule='Only camera_mode wide: Omega16 relative-window heading, Gaussian2seconds, exact original speed. All other TEST predictions copied exactly from V77.',
        wide_keys=list(WIDE),preserved_keys=[k for k in CLIPS if k not in WIDE],model=prior['model'],
        selection='User-requested deployment of wide-only rule after inspecting publicVAL24. Posthoc VAL8.935048468614589%, not independent validation or predicted TEST score.',
        frozen_before_test_inference=True,reference_access_blocked=True,no_test_labels_available=True,
        dependencies={str(p.relative_to(ROOT)):sha(p) for p in files},
        preserved_submissions={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'submissions').glob('*/submission.json')},
        preserved_v82_reports={str(p.relative_to(ROOT)):sha(p) for p in VAL_REPORT.rglob('*.json')}))
    save(REP/'adapter_parity.json',dict(passed=True,test_baseline=parity,val_wide_only=val_parity,
         exact_test_baseline_clips=len(parity),exact_val_candidate_clips=len(val_parity),references_used=False))
    print('FROZEN',len(WIDE),'wide',20-len(WIDE),'preserved; VALadapter exact',len(val_parity),flush=True)

def prepare(key):
    guard();assert key in WIDE;cv2.setNumThreads(1)
    # Unmodified V82 frame extractor: active rectangle, width960, INTER_AREA,
    # PNG cache and the baseline original frame indices and timestamps.
    val.DATA=DATA;val.OUT=OUT;val.BASE=OUT/'baseline';val.CLIPS=WIDE;val.case=test_case
    folder,meta=val.frames(key)
    files=[folder/f'{i:04d}.png' for i in range(len(meta['times']))]
    assert all(p.is_file() for p in files)
    save(REP/f'inputs/{key}.json',dict(key=key,frames=len(files),index=str((folder/'index.json').relative_to(ROOT)),
         index_sha256=sha(folder/'index.json'),frame_sha256={p.name:sha(p) for p in files}))
    print('FRAMES_READY',key,len(files),flush=True)

def infer():
    guard();plan=read(REP/'plan.json')
    for p,h in plan['dependencies'].items():assert sha(ROOT/p)==h,p
    torch,model,load,decode=omega.model_and_io()
    for number,key in enumerate(WIDE,1):
        done=OUT/f'omega/diagnostics/{key}.json'
        if done.exists():continue
        inp=read(REP/f'inputs/{key}.json');index=ROOT/inp['index'];assert sha(index)==inp['index_sha256']
        meta=read(index);folder=index.parent;n=len(meta['times']);beg=time.perf_counter()
        paths=[str(folder/f'{i:04d}.png') for i in range(n)]
        for path in paths:
            p=ROOT/path;assert sha(p)==inp['frame_sha256'][p.name],p
        images=load(paths,image_resolution=416,mode='balanced');assert len(images)==n
        torch.cuda.reset_peak_memory_stats();dest=OUT/f'omega/windows/{key}';dest.mkdir(parents=True,exist_ok=True)
        for start in range(0,n,8):
            end=min(start+16,n);p=dest/f'{start:06d}.npz'
            if not p.exists():
                with torch.inference_mode():
                    out=model(images[start:end].cuda());ex,ki=decode(out['pose_enc'].float(),images.shape[-2:])
                ex=ex[0].cpu().numpy();c2w=np.tile(np.eye(4),(len(ex),1,1))
                c2w[:,:3,:3]=ex[:,:,:3].transpose(0,2,1)
                c2w[:,:3,3]=-np.einsum('nij,nj->ni',c2w[:,:3,:3],ex[:,:,3])
                assert np.isfinite(c2w).all()
                np.savez_compressed(p,start=np.array(start),c2w=c2w,intrinsics=ki[0].cpu().numpy())
                del out,ex,ki
            if start%160==0:print('OMEGA_WINDOW',number,len(WIDE),key,start,end,n,flush=True)
            if end==n:break
        assert np.isfinite(relative_window_yaw(OUT/'omega',key,n)).all()
        save(done,dict(key=key,frames=n,window=16,overlap=8,shape=list(images.shape),seconds=time.perf_counter()-beg,
            peak_cuda_gib=torch.cuda.max_memory_allocated()/2**30,checkpoint_sha256=plan['model']['sha256'],reference_access_blocked=True))
        del images;gc.collect();torch.cuda.empty_cache()
        print('OMEGA_CLIP_DONE',number,len(WIDE),key,flush=True)

def pack():
    from public_dataset import validate_prediction,load_timestamps
    from pack_submission import pack as official_pack
    guard();plan=read(REP/'plan.json');base=read(BASE);payload={};checks=[]
    for key,clip in CLIPS.items():
        pred=base[key]
        if key in WIDE:
            assert (OUT/f'omega/diagnostics/{key}.json').exists()
            _,old,t,base_yaw,speed,_=test_case(key)
            pred,yaw=direction_prediction(old,t,base_yaw,speed,OUT/'omega',key)
            p=OUT/f'signals/{key}.npz';p.parent.mkdir(exist_ok=True)
            np.savez_compressed(p,times=t,yaw=yaw,speed=speed,baseline_yaw=base_yaw)
            assert np.array_equal(speed,np.load(ROOT/f'runs/submission_test_v1/fused/diagnostics/{key}.fusion.npz')['speed'])
        else:assert pred==base[key]
        ts=load_timestamps(DATA,clip);t,positions=validate_prediction(pred,key,ts)
        assert pred['times']==base[key]['times'] and pred['frame_indices']==base[key]['frame_indices']
        assert t[0]==ts[0] and t[-1]==ts[-1] and np.diff(t).max()<=1.000001
        assert np.isfinite(positions).all() and np.ptp(positions,axis=0).max()>1e-6
        hi=np.searchsorted(t,ts);ix=np.clip(hi,0,len(t)-1);lo=np.clip(hi-1,0,len(t)-1)
        support=np.isclose(ts,t[ix],rtol=0,atol=1e-8)|((hi>0)&(hi<len(t))&(t[ix]-t[lo]<=1.+1e-8))
        assert support.all(),key
        payload[key]=pred;save(OUT/f'final/predictions/{key}.json',pred)
        checks.append(dict(key=key,camera_mode=clip['camera_mode'],poses=len(t),changed=pred!=base[key],
            speed_preserved=True,max_gap_seconds=float(np.diff(t).max()),released_timestamp_coverage=float(support.mean()),
            max_position_change_from_v77=float(np.max(abs(positions-np.array(base[key]['positions']))))))
    assert {c['key'] for c in checks if c['changed']}==set(WIDE)
    DEST.mkdir(parents=True,exist_ok=True);bundle=DEST/'submission.json'
    if bundle.exists():assert read(bundle)==payload
    else:official_pack(MANIFEST,OUT/'final',bundle,'test')
    assert read(bundle)==payload and set(payload)==set(CLIPS)
    for group in ['dependencies','preserved_submissions','preserved_v82_reports']:
        for path,h in plan[group].items():assert sha(ROOT/path)==h,path
    validation=dict(passed=True,clips=20,changed_wide=8,preserved_nonwide=12,checks=checks,
        official_packer_equal=True,all_speeds_preserved=True,all_timestamps_preserved=True,
        released_timestamp_coverage=1.,hidden_reference_coverage_unknown=True,
        previous_submissions_preserved=len(plan['preserved_submissions']))
    save(DEST/'validation.json',validation);save(REP/'validation.json',validation)
    provenance=dict(created_utc=now(),version='V83',method='Omega16 wide-only heading replacement',
        base=str(BASE.relative_to(ROOT)),base_sha256=sha(BASE),sha256=sha(bundle),bytes=bundle.stat().st_size,
        wide_keys=list(WIDE),preserved_keys=plan['preserved_keys'],checkpoint=plan['model'],
        reused_val_mean=8.935048468614589,selection_after_val=True,official_test_score=None,
        test_ground_truth_used=False,challenge_uploaded=False,
        source_predictions={k:sha(OUT/f'final/predictions/{k}.json') for k in CLIPS})
    save(DEST/'provenance.json',provenance);save(REP/'completion.json',provenance)
    (DEST/'SHA256SUMS').write_text(provenance['sha256']+'  submission.json\n')
    description='Camera-mode routed maritime visual odometry using frozen pretrained models. Wide clips use VGGT-Omega 416 rotations in 16-frame windows with 8-frame overlap, integrated with unchanged water-flow and train-fitted relative speed. FLIR visible clips retain DPVO, VGGT and Pi3X heading consensus with static geometry course correction and R3 and static DPV-SLAM based speed refinement. Infrared clips retain baseline heading and constant speed. Released calibration and timestamps and external pretrained weights are used. Routing is selected using public train and validation data.'
    allowed=set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 -:,.()[]_/\\+*%$§!?')
    assert set(description)<=allowed and '\n' not in description
    (DEST/'description.txt').write_text(description+'\n')
    notes='Upload only submission.json.\nModel name: TideGraph-OmegaW\nHardware: NVIDIA RTX 5080\nMeta data used: Yes.\nAdditional datasets used: Yes - external pretrained weights.\nFPS: Full end-to-end runtime has not been measured; no invented FPS.\n\n8 wide TEST clips use Omega16 direction and unchanged speed. Other12 clips are exact V77 copies.\nLocal public VAL24: 8.935048468614589 percent; posthoc routing selected after inspecting VAL. Not an official TEST score.\nHidden-reference coverage and score are determined by the challenge server.\n\nDescription:\n'+description+'\n'
    (DEST/'submission_notes.txt').write_text(notes)
    code=DEST/'provenance_code';code.mkdir(exist_ok=True)
    for name in ['v83_omega_wide_submission.py','v82_omega_val.py','v81_omega_pipeline.py']:
        shutil.copy2(ROOT/'scripts'/name,code/name)
    shutil.copy2(REP/'plan.json',code/'plan.json');shutil.copy2(REP/'adapter_parity.json',code/'adapter_parity.json')
    print('SUBMISSION_READY',str(bundle),bundle.stat().st_size,sha(bundle),flush=True)

def call(action,key=None):
    name=action if key is None else action+'_'+key
    folder=REP/'logs';folder.mkdir(exist_ok=True)
    args=[sys.executable,__file__,action]+(['--key',key] if key else [])
    print('START',name,flush=True)
    with (folder/f'{name}.log').open('a') as f:subprocess.run(args,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
    print('DONE',name,flush=True)

def run():
    freeze()
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        jobs=[pool.submit(call,'prepare',key) for key in WIDE if not (REP/f'inputs/{key}.json').exists()]
        for future in concurrent.futures.as_completed(jobs):future.result()
    call('infer');call('pack')

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['freeze','prepare','infer','pack','run']);ap.add_argument('--key');a=ap.parse_args()
    if a.action=='prepare':prepare(a.key)
    else:globals()[a.action]()
