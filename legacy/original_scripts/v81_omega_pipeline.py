"""Official VGGT-Omega adapter and isolated TideGraph heading replacement.

Only the separate evaluator reads reference labels. Random-weight smoke runs
never produce predictions or accuracy claims.
"""
import argparse,datetime,gc,json,subprocess,sys,time
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter1d

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/domain_v81_vggt_ohm';REP=ROOT/'reports/domain_v81_vggt_ohm'
REPO=ROOT/'repos/vggt-omega';MODEL=ROOT/'models/v81_vggt_omega'
REV='1041e80fc0e911235d3426b0a3d9a81075111a53'
WEIGHT='vggt_omega_1b_416_reproduce.pt'
TRAIN=['clip_000','clip_001','clip_075','clip_076','clip_082','clip_084','clip_088','clip_090']
EXTERNAL=['clip_1000','clip_1001','clip_1002','clip_1003']
sys.path.insert(0,str(ROOT/'repos/lookout-vo-challenge/starter'))
from v78_external_data import sha,save
from domain_fusion_v2 import relative_window_yaw

def read(p):return json.loads(p.read_text())
def case(group,key):
    assert group in ['train','external'] and key in (TRAIN if group=='train' else EXTERNAL)
    if group=='train':
        frames=ROOT/f'runs/domain_v5/frames/{key}'
        v24=ROOT/f'runs/domain_v24_submission/train/signals/{key}.npz'
        state=ROOT/f'runs/domain_v76_direction_guarded_speed/train/base/signals/{key}.npz'
        prediction=ROOT/f'runs/domain_v76_direction_guarded_speed/train/base/predictions/direction_guarded_quarter/{key}.json'
        vggt=ROOT/'runs/domain_probe_v2/vggt'
    else:
        root=ROOT/'runs/domain_v78_external';frames=root/f'frames/{key}'
        v24=root/f'v24/external/signals/{key}.npz';state=root/f'v76/external/base/signals/{key}.npz'
        prediction=root/f'final/V77/{key}.json';vggt=root/'vggt'
    meta=read(frames/'index.json');base=read(prediction)
    assert meta['times']==base['times'] and meta['frame_indices']==base['frame_indices']
    return dict(frames=frames,meta=meta,v24=v24,state=state,prediction=prediction,base=base,vggt=vggt)

def integrate(t,yaw,speed):
    vel=np.column_stack([np.sin(yaw),np.zeros(len(t)),np.cos(yaw)])*speed[:,None]
    return np.vstack([np.zeros(3),np.cumsum((vel[:-1]+vel[1:])*.5*np.diff(t)[:,None],axis=0)])

def freeze():
    p=REP/'plan.json'
    if p.exists():return
    cases=[(g,k) for g,keys in [('train',TRAIN),('external',EXTERNAL)] for k in keys]
    dependencies={}
    for group,key in cases:
        c=case(group,key)
        for f in [c['frames']/'index.json',c['v24'],c['state'],c['prediction']]:dependencies[str(f.relative_to(ROOT))]=sha(f)
    save(p,dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),paper='https://arxiv.org/abs/2605.15195',
        repo_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),model_repo='facebook/VGGT-Omega',model_revision=REV,checkpoint=WEIGHT,
        reason='Official416-reproduction checkpoint is upstream benchmark reference; original512 checkpoint not used for reported performance comparison.',
        primary='Replace VGGT heading family with Omega416 balanced16frames overlap8, keeping deployedV77 speed and existing static-course residual fixed.',
        secondary='Omega416 balanced64frames overlap32, fixed before any Omega predictions/scores; temporal-context ablation, not a pretrained-weight change.',
        inference='Same existing2Hz active-image cache, original frameindices/timestamps, seed0, noCLAHE, no labels. Official preprocessing and camera/depth heads. Same relative-rotation fusion and Gaussian2sec as VGGT baseline.',
        fusion='Median(DPVO3 family,Omega,Pi3X) plus unchanged V77-minus-V24 course residual. Speed exactly V77. This isolates heading replacement and does not recompute staticBA or speed support.',
        cases=[dict(group=g,key=k) for g,k in cases],
        evaluation='All TRAIN8 reused development clips and4 external fixed5minute windows. Separate CANOE RGB and Pohang thermal/proxy results; one globalSim3 per trajectory,100-800m drift,coverage,worstcase. No officialTEST reads or new submission.',
        limitations=['TRAIN8 and external4 have already been inspected in earlier studies, not fresh blind validation.','CANOE external cases exercise constant-base-speed branch transfer, not complete wide-camera deployment.','Pohang reference is a sensor-origin SLAM proxy.','Fixed existing course residual may interact with new heading; this is an ablation, not a full geometric reoptimization.'],
        subsequent_validation='If primary warrants continuation, freeze its settings and validate all24officialVAL with the required baseline pipeline; pilot mean alone is insufficient for a new submission.',
        dependencies=dependencies,preserved_submissions={str(f.relative_to(ROOT)):sha(f) for f in sorted((ROOT/'submissions').glob('*/submission.json'))}))

def fetch():
    from huggingface_hub import hf_hub_download,HfApi,get_token
    token=get_token()
    # Project-specific HF_HOME changes the cache location; retain the user's
    # standard active login if the project cache has no active token.
    default_login=Path.home()/'.cache/huggingface/token'
    if token is None and default_login.is_file():token=default_login.read_text().strip()
    info=HfApi(token=token).model_info('facebook/VGGT-Omega',revision=REV,files_metadata=True)
    entry=next(x for x in info.siblings if x.rfilename==WEIGHT)
    lfs=entry.lfs;expected=getattr(lfs,'sha256',None) if lfs else None
    try:
        p=Path(hf_hub_download('facebook/VGGT-Omega',WEIGHT,revision=REV,local_dir=MODEL,token=token))
    except Exception as e:
        save(REP/'checkpoint_access.json',dict(allowed=False,error_type=type(e).__name__,http_status=getattr(getattr(e,'response',None),'status_code',None),
            checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),user_action='Request access at https://huggingface.co/facebook/VGGT-Omega using server authenticated account. Never send a token in chat.'))
        raise RuntimeError('Official checkpoint access unavailable; no Omega accuracy experiment was run.') from None
    h=sha(p);assert p.stat().st_size==entry.size
    if expected:assert h==expected
    save(REP/'checkpoint_access.json',dict(allowed=True,path=str(p),sha256=h,bytes=p.stat().st_size,revision=REV,official_lfs_sha256=expected))
    print('OFFICIAL_WEIGHTS_VERIFIED',h,flush=True);return p

def imports():
    sys.path.insert(0,str(REPO))
    import torch
    from vggt_omega.models import VGGTOmega
    from vggt_omega.utils.load_fn import load_and_preprocess_images
    from vggt_omega.utils.pose_enc import encoding_to_camera
    torch.set_num_threads(2)
    return torch,VGGTOmega,load_and_preprocess_images,encoding_to_camera

def model_and_io():
    torch,Model,load,decode=imports();p=MODEL/WEIGHT
    access=read(REP/'checkpoint_access.json');assert access['allowed'] and sha(p)==access['sha256']
    torch.manual_seed(0);np.random.seed(0);model=Model()
    weights=torch.load(p,map_location='cpu',weights_only=True);model.load_state_dict(weights,strict=True);del weights;gc.collect()
    while torch.cuda.mem_get_info()[0]<11.5*2**30:
        print('WAIT_GPU_MEMORY need11.5GiB; inference configuration unchanged',flush=True);time.sleep(10)
    return torch,model.eval().cuda(),load,decode

def infer(group,key,window):
    import v24_pack_uncapped
    v24_pack_uncapped.guard()
    torch,model,load,decode=model_and_io();c=case(group,key)
    dest=OUT/f'omega_w{window}/{group}';dest.mkdir(parents=True,exist_ok=True)
    if (dest/f'predictions/{key}.json').exists():return
    n=len(c['meta']['times']);paths=[str(c['frames']/f'{i:04d}.png') for i in range(n)]
    images=load(paths,image_resolution=416,mode='balanced');assert len(images)==n
    torch.cuda.reset_peak_memory_stats();beg=time.perf_counter();records=[]
    for start in range(0,n,window//2):
        end=min(n,start+window);file=dest/f'windows/{key}/{start:06d}.npz'
        if not file.exists():
            with torch.inference_mode():
                out=model(images[start:end].cuda());ex,ki=decode(out['pose_enc'].float(),images.shape[-2:])
            ex=ex[0].cpu().numpy();c2w=np.tile(np.eye(4),(len(ex),1,1));c2w[:,:3,:3]=ex[:,:,:3].transpose(0,2,1);c2w[:,:3,3]=-np.einsum('nij,nj->ni',c2w[:,:3,:3],ex[:,:,3])
            depth=out['depth'][0];conf=out['depth_conf'][0]
            if depth.ndim==4:assert depth.shape[-1]==1;depth=depth[...,0]
            if conf.ndim==4:assert conf.shape[-1]==1;conf=conf[...,0]
            assert torch.isfinite(depth).all() and torch.isfinite(conf).all() and np.isfinite(c2w).all()
            file.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(file,start=np.array(start),c2w=c2w,intrinsics=ki[0].cpu().numpy(),depth=depth[:,::4,::4].float().cpu().numpy(),conf=conf[:,::4,::4].float().cpu().numpy())
            del out,ex,ki,depth,conf
        with np.load(file) as z:records.append({k:z[k] for k in z.files})
        print('OMEGA_WINDOW',group,key,window,start,end,n,flush=True)
        if end==n:break
    from window_geometry import stitch_windows
    world,alignment=stitch_windows(records,'depth');assert len(world)==n and np.isfinite(world).all()
    save(dest/f'predictions/{key}.json',dict(c['base'],positions=world[:,:3,3].tolist()))
    np.savez_compressed(dest/f'diagnostics_{key}.npz',camera_to_world=world)
    save(dest/f'diagnostics/{key}.json',dict(arguments=dict(output=str(dest)),model='VGGT-Omega416Reproduction',checkpoint_sha256=read(REP/'checkpoint_access.json')['sha256'],
        frames=n,window=window,overlap=window//2,resolution=416,preprocess='official balanced',shape=list(images.shape),alignment=alignment,
        seconds=time.perf_counter()-beg,peak_cuda_gib=torch.cuda.max_memory_allocated()/2**30,reference_access_blocked=True))

def fused(c,source,key):
    with np.load(c['v24']) as z:b={k:z[k] for k in z.files}
    with np.load(c['state']) as z:s={k:z[k] for k in z.files}
    t=b['times'];assert np.array_equal(t,s['times'])
    yaw=gaussian_filter1d(relative_window_yaw(source,key,len(t)),2/np.median(np.diff(t)))
    family=b['family_yaws'].copy();family[1]=yaw-yaw[0]+b['baseline_yaw'][0]
    course_residual=s['yaw']-b['yaw'];new_yaw=np.median(family,axis=0)+course_residual
    speed=s['speed']*np.exp(s['direction_guarded_quarter'])
    pred=dict(c['base'],positions=integrate(t,new_yaw,speed).tolist())
    return pred,dict(times=t,yaw=new_yaw,speed=speed,course_residual=course_residual,family_yaws=family)

def fuse(group,key,window):
    import v24_pack_uncapped
    v24_pack_uncapped.guard();c=case(group,key);source=OUT/f'omega_w{window}/{group}'
    p=read(source/f'predictions/{key}.json');assert p['times']==c['base']['times'] and p['frame_indices']==c['base']['frame_indices']
    pred,diag=fused(c,source,key);dest=OUT/f'fused_w{window}/{group}'
    save(dest/f'predictions/{key}.json',pred);(dest/'signals').mkdir(exist_ok=True)
    np.savez_compressed(dest/f'signals/{key}.npz',**diag)

def parity():
    import v24_pack_uncapped
    v24_pack_uncapped.guard();rows=[]
    for group,keys in [('train',TRAIN),('external',EXTERNAL)]:
        for key in keys:
            c=case(group,key);pred,diag=fused(c,c['vggt'],key)
            error=float(abs(np.array(pred['positions'])-np.array(c['base']['positions'])).max())
            assert error<1e-5,(group,key,error)
            rows.append(dict(group=group,key=key,max_position_error_m=error,poses=len(pred['times'])))
    save(REP/'integration_parity.json',dict(passed=True,rows=rows,reference_access_blocked=True,
        scope='Injecting original cached VGGT into new replacement adapter reproduces existing V77 on TRAIN8 and external4. This verifies plumbing, NOT Omega performance.'))
    print('INTEGRATION_PARITY_PASS',len(rows),max(x['max_position_error_m'] for x in rows),flush=True)

def smoke():
    torch,Model,load,decode=imports();torch.manual_seed(0);model=Model().eval().cuda();c=case('train',TRAIN[0]);rows=[]
    for n in [16,64]:
        images=load([str(c['frames']/f'{i:04d}.png') for i in range(n)],image_resolution=416,mode='balanced').cuda()
        torch.cuda.reset_peak_memory_stats();beg=time.perf_counter()
        try:
            with torch.inference_mode():out=model(images);ex,ki=decode(out['pose_enc'].float(),images.shape[-2:])
            torch.cuda.synchronize();assert ex.shape==(1,n,3,4) and ki.shape==(1,n,3,3) and torch.isfinite(ex).all() and torch.isfinite(ki).all()
            rows.append(dict(frames=n,shape=list(images.shape),passed=True,seconds=time.perf_counter()-beg,peak_cuda_gib=torch.cuda.max_memory_allocated()/2**30,depth_shape=list(out['depth'].shape)))
            del out,ex,ki
        except torch.OutOfMemoryError:
            rows.append(dict(frames=n,passed=False,error='GPU OOM'))
        del images;gc.collect();torch.cuda.empty_cache()
    save(REP/'architecture_smoke.json',dict(random_initialization=True,performance_evaluated=False,predictions_saved=False,model_parameters=sum(p.numel() for p in model.parameters()),rows=rows,
        scope='Actual official1B architecture forward on real publicTRAIN images, RANDOM weights only. Tests import,preprocessing,shape,finite outputs and memory. No accuracy claim.'))
    print('ARCHITECTURE_SMOKE',rows,flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['freeze','fetch','parity','smoke','infer','fuse']);ap.add_argument('--group',choices=['train','external']);ap.add_argument('--key');ap.add_argument('--window',type=int,choices=[16,64],default=16);a=ap.parse_args()
    if a.action in ['infer','fuse']:globals()[a.action](a.group,a.key,a.window)
    else:globals()[a.action]()
