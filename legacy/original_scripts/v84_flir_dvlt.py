"""Frozen DVLT pilot on reused FLIR TRAIN8 and CANOE RGB2. No TEST inference."""
import argparse
import datetime
import gc
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'runs/domain_v84_flir_dvlt'
REP = ROOT / 'reports/domain_v84_flir_dvlt'
REPO = ROOT / 'repos/dvlt'
MODEL = ROOT / 'models/v84_dvlt'
TRAIN = ['clip_000','clip_001','clip_075','clip_076','clip_082','clip_084','clip_088','clip_090']
RGB = ['clip_1002','clip_1003']
KEYS = TRAIN + RGB
METHODS = ['V83_FLIR_baseline', 'Omega16_only', 'Omega64_only', 'DVLT12_only', 'DVLT16_only',
           'DVLT12_family', 'DVLT12_Omega16_equal', 'DVLT16_Omega16_equal', 'DVLT12_Omega16_agreement',
           'baseline_constant_speed', 'Omega16_constant_speed', 'DVLT12_constant_speed']
sys.path.insert(0, str(ROOT / 'repos/lookout-vo-challenge/starter'))
from domain_fusion_v2 import relative_window_yaw


def read(p): return json.loads(Path(p).read_text())
def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def save(p, obj):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')
def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()
def integrate(t, yaw, speed):
    vel = np.column_stack([np.sin(yaw), np.zeros(len(t)), np.cos(yaw)]) * speed[:,None]
    return np.vstack([np.zeros(3), np.cumsum((vel[:-1]+vel[1:])*.5*np.diff(t)[:,None], axis=0)])


def guard():
    # All labels are opened only by the separate evaluation action/process.
    def audit(event, args):
        if event == 'open' and args and isinstance(args[0], (str, bytes)):
            path = str(args[0])
            if '/references/' in path or '/evaluation_only/' in path:
                raise RuntimeError('Reference access blocked in V84 prediction process')
    sys.addaudithook(audit)


def case(key):
    assert key in KEYS
    group = 'train' if key in TRAIN else 'external'
    if group == 'train':
        frames = ROOT / f'runs/domain_v5/frames/{key}'
        state = ROOT / f'runs/domain_v76_direction_guarded_speed/train/base/signals/{key}.npz'
        base_path = ROOT / f'runs/domain_v76_direction_guarded_speed/train/base/predictions/direction_guarded_quarter/{key}.json'
        family = ROOT / f'runs/domain_v24_submission/train/signals/{key}.npz'
    else:
        frames = ROOT / f'runs/domain_v78_external/frames/{key}'
        state = ROOT / f'runs/domain_v78_external/v76/external/base/signals/{key}.npz'
        base_path = ROOT / f'runs/domain_v78_external/final/V77/{key}.json'
        family = ROOT / f'runs/domain_v78_external/v24/external/signals/{key}.npz'
    base = read(base_path); index = read(frames/'index.json')
    with np.load(state) as z:
        t, yaw = z['times'], z['yaw']
        speed = z['speed'] * np.exp(z['direction_guarded_quarter'])
    with np.load(family) as z:
        family_yaws = z['family_yaws']; course_residual = yaw - z['yaw']
    assert np.array_equal(t, base['times']) and index['frame_indices'] == base['frame_indices']
    assert np.max(abs(integrate(t,yaw,speed)-base['positions'])) < 1e-5
    return dict(key=key,group=group,frames=frames,index=index,base=base,base_path=base_path,state_path=state,
                family_path=family,t=t,yaw=yaw,speed=speed,family_yaws=family_yaws,course_residual=course_residual)


def freeze():
    guard()
    if (REP/'plan.json').exists(): return
    sources=[Path(__file__),MODEL/'config.json',MODEL/'model.safetensors',REPO/'src/dvlt/model/dvlt/model.py',
             ROOT/'scripts/domain_fusion_v2.py',ROOT/'scripts/fuse_headings.py']
    parity=[];frames={}
    for key in KEYS:
        c=case(key);sources += [c['base_path'],c['state_path'],c['family_path'],c['frames']/'index.json']
        paths=[c['frames']/f'{i:04d}.png' for i in range(len(c['t']))]
        frames[key]={p.name:sha(p) for p in paths}
        parity.append(dict(key=key,max_position_error=float(np.max(abs(integrate(c['t'],c['yaw'],c['speed'])-c['base']['positions'])))))
    save(REP/'frame_hashes.json',frames)
    save(REP/'plan.json',dict(created_utc=now(),keys=KEYS,train=TRAIN,external_rgb=RGB,methods=METHODS,
         primary='DVLT12_only: official default12 refinement steps, direction only, exact existing V77/V83 FLIR speed.',
         secondary='DVLT12_Omega16_equal: equal initial-heading-aligned average; no perclip choice or fitted weights.',
         sensitivity='DVLT16 is a predetermined compute sensitivity, not selected perclip. K12 and K16 are same-model variants, not independent votes.',
         agreement='Smooth10s binary agreement of DVLT12 and Omega16 within3deg; confidence blends their equal mean into V77 heading. This is exploratory input-only gating, not calibrated probability.',
         speed_controls='Existing speed versus constant1; same direction. No scale fitting to GT or learned changes.',
         inference=dict(model='nvidia/dvlt',checkpoint_revision='20f919875a00ee3ab2e840f2f31580f4652b7c5e',
             checkpoint_sha256=sha(MODEL/'model.safetensors'),repository_commit=subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip(),
             image_size=504,window=16,overlap=8,refinement_steps=[12,16],precision='bf16',seed=0,
             preprocessing='Official preprocess_images on same 960-wide active-image PNG cache and original2Hz frame/time selections as V77.',
             pose='Official rays-to-pose uniform weighting, OpenCV c2w; original relative-window-rotation integration and Gaussian2s heading.',
             labels_read=False,test_inference=False),
         dependencies={str(p.relative_to(ROOT)):sha(p) for p in sources},
         preserved_submissions={str(p.relative_to(ROOT)):sha(p) for p in sorted((ROOT/'submissions').glob('*/submission.json'))},
         limitations=['TRAIN8 and CANOE RGB2 are already reused development samples. No independent validation claim.',
                      'External RGB views are not FLIR telephoto-equivalent.','No FLIR-VIS in officialVAL24; no local officialTEST score.',
                      'Uniform-weight small-window pose benchmarks do not establish long maritime trajectory accuracy.'],
         selection='Report all predefined methods and clip deltas. No automatic submission or promotion from minimum mean.'))
    save(REP/'baseline_parity.json',dict(passed=True,rows=parity))
    print('V84_FROZEN',len(KEYS),len(METHODS),flush=True)


def load_model():
    sys.path.insert(0,str(REPO/'src'))
    import torch
    from accelerate import Accelerator
    from dvlt.model.dvlt.model import DVLT
    from dvlt.util.preprocess import preprocess_images
    config=read(MODEL/'config.json');config['load_patch_embed_weights']=False
    torch.set_num_threads(2);torch.manual_seed(0);np.random.seed(0)
    accelerator=Accelerator(mixed_precision='bf16')
    model=DVLT(**config);missing,extra=model.load_pretrained(str(MODEL/'model.safetensors'),strict=True)
    assert not missing and not extra
    model.setup_test(accelerator)
    return torch,accelerator,model,preprocess_images


def infer(keys=None,smoke=False):
    guard();plan=read(REP/'plan.json'); hashes=read(REP/'frame_hashes.json')
    for path,h in plan['dependencies'].items(): assert sha(ROOT/path)==h,path
    torch,accelerator,model,preprocess=load_model()
    from PIL import Image
    keys=keys or KEYS
    for key in keys:
        c=case(key);paths=[c['frames']/f'{i:04d}.png' for i in range(len(c['t']))]
        for p in paths:assert sha(p)==hashes[key][p.name]
        # CPU input cache; transfer one window at a time.
        images=[Image.open(p).convert('RGB') for p in paths]
        batch=preprocess(images,img_size=504,patch_size=14,device='cpu')
        for im in images:im.close()
        del images
        n=len(paths)
        for steps in [12,16]:
            dest=OUT/f'dvlt_k{steps}/{c["group"]}';done=dest/f'diagnostics/{key}.json'
            if done.exists() and not smoke: continue
            model.model.inference_steps=steps;beg=time.perf_counter();torch.cuda.reset_peak_memory_stats()
            for start in range(0,n,8):
                end=min(n,start+16);p=dest/f'windows/{key}/{start:06d}.npz'
                if not p.exists() or smoke:
                    torch.manual_seed(0)
                    small={k:v[:,start:end].to(accelerator.device) for k,v in batch.items()}
                    with torch.inference_mode(), accelerator.autocast():pred=model.predict(small,accelerator)
                    camera=pred['cameras'][0];ex=camera.camera_to_worlds.float().cpu().numpy()
                    poses=np.tile(np.eye(4),(len(ex),1,1));poses[:,:3,:4]=ex
                    intrinsics=camera.get_intrinsics_matrices().float().cpu().numpy()
                    assert np.isfinite(poses).all() and np.isfinite(intrinsics).all()
                    assert np.max(abs(np.linalg.det(poses[:,:3,:3])-1))<1e-3
                    if not smoke:
                        p.parent.mkdir(parents=True,exist_ok=True)
                        np.savez_compressed(p,start=np.array(start),c2w=poses,intrinsics=intrinsics)
                    else:
                        save(REP/f'smoke_k{steps}.json',dict(passed=True,key=key,frames=end-start,shape=list(small['images'].shape),
                            seconds=time.perf_counter()-beg,peak_cuda_gib=torch.cuda.max_memory_allocated()/2**30,
                            checkpoint_loaded_strictly=True,parameter_count=sum(p.numel() for p in model.model.parameters()),
                            finite_pose=True,poses_used_for_accuracy=False))
                    del pred,small,camera,ex,poses,intrinsics
                if smoke or end==n: break
                if start%160==0:print('V84_WINDOW',key,'K',steps,start,n,flush=True)
            if not smoke:
                assert np.isfinite(relative_window_yaw(dest,key,n)).all()
                save(done,dict(key=key,frames=n,refinement_steps=steps,seconds=time.perf_counter()-beg,
                     peak_cuda_gib=torch.cuda.max_memory_allocated()/2**30,reference_access_blocked=True))
                print('V84_CLIP_DONE',key,steps,flush=True)
            gc.collect();torch.cuda.empty_cache()
        del batch;gc.collect()


def predict():
    guard();ledger={};checks=[]
    for key in KEYS:
        c=case(key);t=c['t'];speed=c['speed'];sigma=2/np.median(np.diff(t))
        directions={}
        for name,source in [('Omega16',ROOT/f'runs/domain_v81_vggt_ohm/omega_w16/{c["group"]}'),
                            ('Omega64',ROOT/f'runs/domain_v81_vggt_ohm/omega_w64/{c["group"]}'),
                            ('DVLT12',OUT/f'dvlt_k12/{c["group"]}'),('DVLT16',OUT/f'dvlt_k16/{c["group"]}')]:
            y=gaussian_filter1d(relative_window_yaw(source,key,len(t)),sigma);directions[name]=y-y[0]+c['yaw'][0]
        y=directions; family=c['family_yaws'].copy(); family[1]=y['DVLT12']
        agreement=gaussian_filter1d((abs(y['DVLT12']-y['Omega16'])<=np.deg2rad(3)).astype(float),10/np.median(np.diff(t)))
        mean=(y['DVLT12']+y['Omega16'])*.5
        methods={'V83_FLIR_baseline':c['yaw'],'Omega16_only':y['Omega16'],'Omega64_only':y['Omega64'],
                 'DVLT12_only':y['DVLT12'],'DVLT16_only':y['DVLT16'],
                 'DVLT12_family':np.median(family,axis=0)+c['course_residual'],
                 'DVLT12_Omega16_equal':mean,'DVLT16_Omega16_equal':(y['DVLT16']+y['Omega16'])*.5,
                 'DVLT12_Omega16_agreement':c['yaw']+agreement*(mean-c['yaw']),
                 'baseline_constant_speed':c['yaw'],'Omega16_constant_speed':y['Omega16'],'DVLT12_constant_speed':y['DVLT12']}
        assert list(methods)==METHODS
        for method,yaw in methods.items():
            chosen=np.ones_like(speed) if method.endswith('constant_speed') else speed
            pred=dict(c['base'],positions=integrate(t,yaw,chosen).tolist())
            if method=='V83_FLIR_baseline':pred=c['base']
            p=OUT/f'predictions/{method}/{key}.json';save(p,pred);ledger[str(p.relative_to(ROOT))]=sha(p)
            sp=OUT/f'signals/{method}/{key}.npz';sp.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(sp,times=t,yaw=yaw,speed=chosen)
        checks.append(dict(key=key,poses=len(t),frame_indices_unchanged=True,timestamps_unchanged=True,
                           agreement_fraction=float(agreement.mean())))
    save(REP/'predictions_frozen.json',dict(created_utc=now(),files=ledger,reference_access_blocked=True))
    save(REP/'prediction_checks.json',dict(passed=True,rows=checks))
    print('V84_PREDICTIONS_FROZEN',len(ledger),flush=True)


def evaluate():
    import metrics
    from public_dataset import load_reference
    frozen=read(REP/'predictions_frozen.json');plan=read(REP/'plan.json')
    for path,h in frozen['files'].items(): assert sha(ROOT/path)==h,path
    clips={c['key']:c for c in read(ROOT/'data/release/release_manifest.json')['clips']};rows=[]
    for key in KEYS:
        if key in TRAIN:rt,rp=load_reference(ROOT/'data/release',clips[key]);group='reused_FLIR_TRAIN8'
        else:
            with np.load(ROOT/f'data/external_v78/evaluation_only/{key}.npz') as z:rt,rp=z['times'],z['positions']
            group='reused_CANOE_RGB2'
        for method in METHODS:
            p=read(OUT/f'predictions/{method}/{key}.json');raw=metrics.score_trajectory(np.array(p['times']),np.array(p['positions']),rt,rp)
            assert raw is not None and np.isfinite(raw['drift_pct']),(key,method)
            t=np.array(p['times']);hi=np.searchsorted(t,rt);ix=np.clip(hi,0,len(t)-1);lo=np.clip(hi-1,0,len(t)-1)
            cover=np.isclose(rt,t[ix],atol=1e-8,rtol=0)|((hi>0)&(hi<len(t))&(t[ix]-t[lo]<=1.+1e-8))
            rows.append(dict(key=key,group=group,method=method,drift_pct=float(raw['drift_pct']),coverage=float(cover.mean())))
    groups={}
    for group in ['reused_FLIR_TRAIN8','reused_CANOE_RGB2']:
        rr=[r for r in rows if r['group']==group];base={r['key']:r['drift_pct'] for r in rr if r['method']=='V83_FLIR_baseline'}
        groups[group]={}
        for method in METHODS:
            selected=[r for r in rr if r['method']==method]
            for r in selected:r['delta_pp']=r['drift_pct']-base[r['key']]
            delta=[r['delta_pp'] for r in selected];loo=[float(np.mean(delta[:i]+delta[i+1:])) for i in range(len(delta))]
            groups[group][method]=dict(mean_drift_pct=float(np.mean([r['drift_pct'] for r in selected])),
                mean_delta_pp=float(np.mean(delta)),improved=sum(d<-1e-8 for d in delta),worsened=sum(d>1e-8 for d in delta),
                worst_delta_pp=max(delta),leave_one_out_delta_range=[min(loo),max(loo)],scored=len(selected),min_coverage=min(r['coverage'] for r in selected))
    assert abs(groups['reused_FLIR_TRAIN8']['V83_FLIR_baseline']['mean_drift_pct']-8.643620098453221)<1e-9
    assert abs(groups['reused_FLIR_TRAIN8']['Omega16_only']['mean_drift_pct']-5.930328637395857)<1e-8
    for path,h in plan['preserved_submissions'].items(): assert sha(ROOT/path)==h,path
    save(REP/'results.json',dict(created_utc=now(),rows=rows,groups=groups,official_test_evaluated=False,
         scope='Reused FLIR8 and reused externalRGB2; not blind validation; no TEST inference or submission modification.',
         metric='Official score_trajectory, one global Sim3 per full trajectory;100-800m drift.'))
    print('V84_RESULTS',json.dumps(groups),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['freeze','smoke','infer','predict','evaluate']);ap.add_argument('--key');a=ap.parse_args()
    if a.action=='smoke':infer([TRAIN[0]],smoke=True)
    elif a.action=='infer':infer([a.key] if a.key else None)
    else:globals()[a.action]()
