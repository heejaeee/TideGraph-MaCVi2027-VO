"""Regenerate per-window camera rotations (W16/stride8) from raw video.

Reproduces the TideGraph Omega inputs (v81/v83/v150 scripts):
2Hz frames at released timestamps, active_rect crop, resize to width 960
(wide INTER_AREA, FLIR INTER_LINEAR), official 'balanced' preprocessing at
416 (wide) / 512 (flir-VIS), 16-frame windows every 8 frames, fp32.
Frames stay in memory (lossless PNG round trip is pixel-identical).
Output: <out>/windows/<key>/<start:06d>.npz with start, c2w (N,4,4).

usage: .venv-gpu/bin/python tools/omega_windows.py --data test --clips clip_092 --out outputs/omega_check
"""
import argparse,gc,json,sys,time
from pathlib import Path
import cv2,numpy as np

ROOT=Path(__file__).resolve().parents[1]
EXT=ROOT.parents[1]/'external_weights'
DATA={'test':ROOT/'lookout-vo-test-v0.1','dev':ROOT/'lookout-vo-dev-v0.1/release'}
MANIFEST={'test':DATA['test']/'release_manifest.json','dev':DATA['dev']/'release_manifest.json'}
RES={'wide':416,'flir-VIS':512}
INTERP={'wide':cv2.INTER_AREA,'flir-VIS':cv2.INTER_LINEAR}

def frame_ids(times):
    return np.unique(np.r_[np.searchsorted(times,np.arange(times[0],times[-1],.5)),len(times)-1])

def frames(root,clip,ids):
    x,y,w,h=clip['calibration']['active_rect'];height=round(h*960/w);wanted=set(int(i) for i in ids)
    cap=cv2.VideoCapture(str(root/clip['video']));assert cap.isOpened();out=[]
    for i in range(int(ids[-1])+1):
        if not cap.grab():raise RuntimeError((clip['key'],i))
        if i not in wanted:continue
        ok,bgr=cap.retrieve();assert ok
        out.append(cv2.resize(bgr[y:y+h,x:x+w],(960,height),interpolation=INTERP[clip['camera_mode']])[:,:,::-1].copy())
    cap.release();assert len(out)==len(ids);return out

def omega_backend(weights):
    sys.path.insert(0,str(EXT/'vggt-omega-src'))
    import torch
    from PIL import Image
    from torchvision import transforms as TF
    from vggt_omega.models import VGGTOmega
    from vggt_omega.utils import load_fn
    from vggt_omega.utils.pose_enc import encoding_to_camera
    torch.manual_seed(0);model=VGGTOmega()
    sd=torch.load(weights,map_location='cpu',weights_only=True);model.load_state_dict(sd,strict=True);del sd;gc.collect()
    model=model.eval().cuda();to_tensor=TF.ToTensor()
    def preprocess(rgb,res):
        # identical to load_fn.load_and_preprocess_images for one image, minus file IO
        im=load_fn._crop_to_supported_aspect_ratio(Image.fromarray(rgb));w,h=im.size
        th,tw=load_fn._balanced_target_shape(h/max(w,1),res,16)
        return to_tensor(im.resize((tw,th),Image.Resampling.BICUBIC))
    def run(batch):
        with torch.inference_mode():
            out=model(batch.cuda());ex,_=encoding_to_camera(out['pose_enc'].float(),batch.shape[-2:])
        ex=ex[0].cpu().numpy();c2w=np.tile(np.eye(4),(len(ex),1,1))
        c2w[:,:3,:3]=ex[:,:,:3].transpose(0,2,1);c2w[:,:3,3]=-np.einsum('nij,nj->ni',c2w[:,:3,:3],ex[:,:,3])
        return c2w
    run.model=model
    return torch,preprocess,run

def pi3x_backend(weights):
    """Pi3X (yyfz/Pi3 @9fa3ddb): official PIXEL_LIMIT=255000 / patch-14 size rule, LANCZOS, bf16 autocast."""
    sys.path.insert(0,str(EXT/'pi3-src'))
    import math,torch
    from PIL import Image
    from torchvision import transforms as TF
    from safetensors.torch import load_file
    from pi3.models.pi3x import Pi3X
    torch.manual_seed(0);model=Pi3X();sd=load_file(str(weights));model.load_state_dict(sd,strict=True);del sd;gc.collect()
    model=model.eval().cuda();to_tensor=TF.ToTensor()
    def preprocess(rgb,_res,limit=255000):
        H,W=rgb.shape[:2];s=math.sqrt(limit/(W*H));wt,ht=W*s,H*s;k,m=round(wt/14),round(ht/14)
        while (k*14)*(m*14)>limit:
            if k/m>wt/ht:k-=1
            else:m-=1
        return to_tensor(Image.fromarray(rgb).resize((k*14,m*14),Image.Resampling.LANCZOS))
    def run(batch):
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):out=model(batch[None].cuda())
        return out['camera_poses'][0].float().cpu().numpy().astype(np.float64)
    return torch,preprocess,run

BACKENDS=dict(omega=(omega_backend,EXT/'vggt_omega/vggt_omega_1b_416_reproduce.pt'),pi3x=(pi3x_backend,EXT/'pi3x/model.safetensors'))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',choices=['test','dev'],required=True)
    ap.add_argument('--clips',required=True,help='comma list, or a dev_split.json key list like flir_vis_val,wide_val')
    ap.add_argument('--out',type=Path,required=True);ap.add_argument('--model',default='omega',choices=list(BACKENDS))
    ap.add_argument('--weights',type=Path,help='default depends on --model')
    ap.add_argument('--window',type=int,default=16)
    a=ap.parse_args();sys.path.insert(0,str(DATA[a.data]/('starter')))
    from public_dataset import load_manifest,load_timestamps
    clips={c['key']:c for c in load_manifest(MANIFEST[a.data])['clips']}
    split=json.loads((ROOT/'configs/dev_split.json').read_text())
    keys=[k for part in a.clips.split(',') for k in (split[part] if part in split else [part])]
    backend,default=BACKENDS[a.model];a.weights=a.weights or default
    torch,preprocess,run=backend(a.weights)
    for key in keys:
        clip=clips[key];mode=clip['camera_mode']
        if mode not in RES:print('SKIP',key,mode,flush=True);continue
        dest=a.out/'windows'/key;info=a.out/'clips'/f'{key}.json'
        if info.exists():print('CACHED',key,flush=True);continue
        dest.mkdir(parents=True,exist_ok=True);beg=time.perf_counter()
        ts=load_timestamps(DATA[a.data],clip);ids=frame_ids(ts);n=len(ids)
        images=torch.stack([preprocess(f,RES[mode]) for f in frames(DATA[a.data],clip,ids)])
        torch.cuda.reset_peak_memory_stats()
        for start in range(0,n,a.window//2):
            end=min(n,start+a.window)
            np.savez_compressed(dest/f'{start:06d}.npz',start=np.array(start),c2w=run(images[start:end]))
            if end==n:break
        rec=dict(key=key,mode=mode,frames=n,frame_indices=ids.tolist(),times=ts[ids].tolist(),resolution=RES[mode],
                 input_hw=list(images.shape[-2:]),window=a.window,seconds=time.perf_counter()-beg,
                 peak_cuda_gib=torch.cuda.max_memory_allocated()/2**30,model=a.model,weights=str(a.weights))
        info.parent.mkdir(parents=True,exist_ok=True);info.write_text(json.dumps(rec,indent=1))
        print('DONE',key,mode,n,'frames',round(rec['seconds'],1),'s',flush=True);del images;gc.collect()

if __name__=='__main__':main()
