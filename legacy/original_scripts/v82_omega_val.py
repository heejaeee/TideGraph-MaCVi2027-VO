"""Frozen Omega16 direction transfer on all 24 public VAL clips; no GT access."""
import argparse, datetime, gc, json, sys, time
from pathlib import Path
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
import v81_omega_pipeline as omega
from domain_fusion_v2 import relative_window_yaw
from v78_external_data import sha, save

ROOT = omega.ROOT
OUT = ROOT / 'runs/domain_v82_omega_val'
REP = ROOT / 'reports/domain_v82_omega_val'
DATA = ROOT / 'data/release'
BASE = ROOT / 'runs/domain_v2_gated_val'
STATE = ROOT / 'runs/heading_current_dense_filtered_gated_v3_klt25_val'
VGGT = ROOT / 'runs/vggt_val_w16_o8'
CLIPS = {c['key']: c for c in omega.read(DATA/'release_manifest.json')['clips'] if c['split']=='val'}
METHODS = ['baseline', 'omega_klt25_visible', 'omega_only_visible', 'omega_only_all', 'vggt_only_all']

def read(p): return omega.read(p)
def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def guard():
    from v24_pack_uncapped import guard as block
    block()

def case(key):
    clip = CLIPS[key]
    pred = read(BASE/f'predictions/{key}.json')
    with np.load(STATE/f'diagnostics/{key}.refined.npz') as z:
        t, yaw, speed = z['times'], z['yaw'], z['speed'].copy()
    assert np.array_equal(t, pred['times'])
    if clip['camera_mode']=='flir-IR': speed = np.ones(len(t))
    with np.load(ROOT/f'runs/heading_guarded_water_val/diagnostics/{key}.headings.npz') as z:
        klt = gaussian_filter1d(z['source_yaws'][2], 2/np.median(np.diff(t)))
    assert np.array_equal(omega.integrate(t,yaw,speed), pred['positions']), key
    return clip, pred, t, yaw, speed, klt

def frames(key):
    folder = ROOT/f'runs/domain_v5/frames/{key}'
    if not (folder/'index.json').exists(): folder = OUT/f'frames/{key}'
    if (folder/'index.json').exists():
        meta = read(folder/'index.json')
        pred = read(BASE/f'predictions/{key}.json')
        assert meta['frame_indices']==pred['frame_indices'] and meta['times']==pred['times']
        return folder, meta
    clip,pred,*_ = case(key)
    folder.mkdir(parents=True,exist_ok=True)
    ids=pred['frame_indices']; rect=clip['calibration']['active_rect'];x,y,w,h=rect
    height=round(h*960/w); cap=cv2.VideoCapture(str(DATA/clip['video']))
    assert cap.isOpened()
    ix=0; j=0; beg=time.perf_counter()
    while j<len(ids):
        ok=cap.grab()
        if not ok: break
        if ix==ids[j]:
            ok,im=cap.retrieve(); assert ok
            im=cv2.resize(im[y:y+h,x:x+w],(960,height),interpolation=cv2.INTER_AREA)
            assert cv2.imwrite(str(folder/f'{j:04d}.png'),im)
            j+=1
        ix+=1
    cap.release(); assert j==len(ids),(key,j,len(ids))
    meta=dict(key=key,frame_indices=ids,times=pred['times'],width=960,height=height,
              source_active_rect=rect,elapsed_seconds=time.perf_counter()-beg)
    save(folder/'index.json',meta)
    return folder,meta

def freeze():
    guard(); assert len(CLIPS)==24
    if (REP/'plan.json').exists(): return
    deps={}; parity=[]
    for key in CLIPS:
        clip,pred,t,y,s,k=case(key)
        for p in [BASE/f'predictions/{key}.json',STATE/f'diagnostics/{key}.refined.npz',
                  ROOT/f'runs/heading_guarded_water_val/diagnostics/{key}.headings.npz',
                  VGGT/f'diagnostics/{key}.json']:
            deps[str(p.relative_to(ROOT))]=sha(p)
        parity.append(dict(key=key,mode=clip['camera_mode'],poses=len(t),max_position_error_m=0.0))
    sources=[ROOT/'scripts/v82_omega_val.py',ROOT/'scripts/v81_omega_pipeline.py',
             ROOT/'scripts/domain_fusion_v2.py',ROOT/'scripts/fuse_headings.py',
             DATA/'release_manifest.json']
    for p in sources: deps[str(p.relative_to(ROOT))]=sha(p)
    save(REP/'plan.json',dict(created_utc=now(),keys=list(CLIPS),methods=METHODS,
        primary='omega_klt25_visible: 75% Omega16 relative-window yaw plus unchanged 25% KLT yaw, original speed; original baseline for infrared.',
        secondary='omega_only_visible: Omega16 alone for visible wide/zoom, original baseline for infrared.',
        controls='omega_only_all and vggt_only_all: single-model relative-window direction on all24 including infrared, same fixed baseline speed.',
        baseline='Exact V2 non-FLIR-VIS branch retained by V18/V24/V77. Reconstructed from existing refined yaw/speed, infrared constant-speed override; exact numeric parity on24.',
        model=dict(checkpoint=omega.WEIGHT,sha256=read(omega.REP/'checkpoint_access.json')['sha256'],revision=omega.REV,
                   resolution=416,preprocess='official balanced',window=16,overlap=8,seed=0),
        inference='2Hz original frame selections/timestamps, active-image cache width960, Gaussian2seconds; no new training, no speed retuning, no BA. All settings fixed before new VAL scores.',
        modalities={m:sum(c['camera_mode']==m for c in CLIPS.values()) for m in sorted({c['camera_mode'] for c in CLIPS.values()})},
        evaluation='One globalSim3 per complete trajectory; official100-800m drift, perclip and permodality, paired differences, failures and coverage. All predictions frozen before separate evaluator reads GT.',
        limitations=['Previously reused publicVAL, not fresh blind validation.','No FLIR-VIS inVAL: cannot directly validate the V81 FLIR-VIS family-median benefit.','Non-FLIR deployment uses old guardedR3/VGGT plusKLT, not DPVO/VGGT/Pi3X median.','Omega416 and originalVGGT518 use model-specific preprocessing.'],
        dependencies=deps,preserved_submissions={str(p.relative_to(ROOT)):sha(p) for p in sorted((ROOT/'submissions').glob('*/submission.json'))},
        preserved_v81={str(p.relative_to(ROOT)):sha(p) for p in sorted(omega.REP.glob('*.json'))},
        official_test_evaluated=False,new_submission=False))
    save(REP/'baseline_parity.json',dict(passed=True,rows=parity,exact=24,gt_used=False))
    print('V82_FROZEN',len(CLIPS),METHODS,flush=True)

def prepare():
    guard(); cv2.setNumThreads(1)
    inputs={}
    for key in CLIPS:
        folder,meta=frames(key)
        paths=[folder/f'{i:04d}.png' for i in range(len(meta['times']))]
        assert all(p.exists() for p in paths)
        inputs[key]=dict(index=str((folder/'index.json').relative_to(ROOT)),index_sha256=sha(folder/'index.json'),
                         frames=len(paths),frame_sha256={p.name:sha(p) for p in paths})
        print('FRAMES_READY',key,len(paths),flush=True)
    save(REP/'frame_inputs.json',inputs)

def infer():
    guard()
    plan=read(REP/'plan.json')
    for p,h in plan['dependencies'].items(): assert sha(ROOT/p)==h,p
    torch,model,load,decode=omega.model_and_io()
    for number,key in enumerate(CLIPS,1):
        done=OUT/f'omega/diagnostics/{key}.json'
        if done.exists(): continue
        folder,meta=frames(key); n=len(meta['times']);beg=time.perf_counter()
        paths=[str(folder/f'{i:04d}.png') for i in range(n)]
        images=load(paths,image_resolution=416,mode='balanced');assert len(images)==n
        torch.cuda.reset_peak_memory_stats()
        dest=OUT/f'omega/windows/{key}';dest.mkdir(parents=True,exist_ok=True)
        for start in range(0,n,8):
            end=min(start+16,n);p=dest/f'{start:06d}.npz'
            if not p.exists():
                with torch.inference_mode():
                    out=model(images[start:end].cuda())
                    ex,ki=decode(out['pose_enc'].float(),images.shape[-2:])
                ex=ex[0].cpu().numpy();c2w=np.tile(np.eye(4),(len(ex),1,1))
                c2w[:,:3,:3]=ex[:,:,:3].transpose(0,2,1)
                c2w[:,:3,3]=-np.einsum('nij,nj->ni',c2w[:,:3,:3],ex[:,:,3])
                assert np.isfinite(c2w).all()
                np.savez_compressed(p,start=np.array(start),c2w=c2w,intrinsics=ki[0].cpu().numpy())
                del out,ex,ki
            if start%160==0:print('OMEGA_WINDOW',number,24,key,start,end,n,flush=True)
            if end==n:break
        yaw=relative_window_yaw(OUT/'omega',key,n);assert np.isfinite(yaw).all()
        save(done,dict(key=key,frames=n,window=16,overlap=8,shape=list(images.shape),
                       seconds=time.perf_counter()-beg,peak_cuda_gib=torch.cuda.max_memory_allocated()/2**30,
                       checkpoint_sha256=plan['model']['sha256'],reference_access_blocked=True))
        del images;gc.collect();torch.cuda.empty_cache()
        print('OMEGA_CLIP_DONE',number,24,key,flush=True)

def predict():
    guard(); hashes={};checks=[]
    for key in CLIPS:
        clip,pred,t,base_yaw,speed,klt=case(key); n=len(t);sigma=2/np.median(np.diff(t))
        assert (OUT/f'omega/diagnostics/{key}.json').exists()
        oy=gaussian_filter1d(relative_window_yaw(OUT/'omega',key,n),sigma)
        vy=gaussian_filter1d(relative_window_yaw(VGGT,key,n),sigma)
        # Common initial heading is an image-only global gauge, independent of GT.
        oy=oy-oy[0]+base_yaw[0];vy=vy-vy[0]+base_yaw[0]
        kk=klt-klt[0]+base_yaw[0]
        visible=clip['camera_mode']!='flir-IR'
        yaws=dict(baseline=base_yaw,omega_klt25_visible=.75*oy+.25*kk if visible else base_yaw,
                  omega_only_visible=oy if visible else base_yaw,omega_only_all=oy,vggt_only_all=vy)
        for method,yaw in yaws.items():
            p=OUT/f'predictions/{method}/{key}.json'
            out=dict(pred,positions=omega.integrate(t,yaw,speed).tolist())
            if method=='baseline' or (not visible and method.endswith('_visible')): assert out==pred
            save(p,out);hashes[str(p.relative_to(ROOT))]=sha(p)
            f=OUT/f'signals/{method}/{key}.npz';f.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(f,times=t,yaw=yaw,speed=speed)
            checks.append(dict(key=key,method=method,poses=n,speed_exact=True))
    save(REP/'predictions_frozen.json',dict(created_utc=now(),files=hashes,all_predictions_completed_before_scoring=True))
    save(REP/'inference_checks.json',dict(passed=True,rows=checks,reference_access_blocked=True))
    print('PREDICTIONS_FROZEN',len(hashes),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['freeze','prepare','infer','predict']);a=ap.parse_args()
    globals()[a.action]()
