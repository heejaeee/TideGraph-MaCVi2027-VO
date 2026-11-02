"""VGGT-1B overlapping-window maritime VO; no reference labels are read."""
import argparse, gc, json, platform, random, sys, time
from pathlib import Path
import cv2
import numpy as np
import torch
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'repos/vggt'))
sys.path.insert(0,str(ROOT/'repos/lookout-vo-challenge/starter'))
from public_dataset import load_manifest,load_timestamps,select_clips,validate_prediction,write_json
from vggt.models.vggt import VGGT
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from window_geometry import stitch_windows


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--key',required=True); ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--fps',type=float,default=2); ap.add_argument('--size',type=int,default=518)
    ap.add_argument('--window',type=int,default=16); ap.add_argument('--overlap',type=int,default=8)
    ap.add_argument('--max-frames',type=int,default=0)
    ap.add_argument('--dataset',required=True)
    args=ap.parse_args()
    assert 2<=args.overlap<args.window and args.size%14==0
    random.seed(0); np.random.seed(0); torch.manual_seed(0); cv2.setNumThreads(2)
    run=args.output; run.mkdir(exist_ok=True,parents=True)
    pred=run/'predictions'/f'{args.key}.json'
    if pred.exists(): raise FileExistsError(pred)
    diag=run/'diagnostics'; diag.mkdir(exist_ok=True)
    raw=run/'windows'/args.key; raw.mkdir(parents=True,exist_ok=False)
    data=Path(args.dataset).resolve().parent; clip=select_clips(load_manifest(Path(args.dataset)),clips=args.key)[0]
    times=load_timestamps(data,clip)
    ids=np.unique(np.r_[np.searchsorted(times,np.arange(times[0],times[-1],1/args.fps)),len(times)-1])
    if args.max_frames: ids=ids[:args.max_frames]
    began=time.perf_counter(); wanted=set(ids.tolist()); frames=[]
    cap=cv2.VideoCapture(str(data/clip['video'])); x,y,w,h=clip['calibration']['active_rect']
    # Matches official crop preprocessing at size=518, performed in memory.
    new_h=round(h*(args.size/w)/14)*14
    for i in range(int(ids[-1])+1):
        if not cap.grab(): raise RuntimeError(f'Decode {i}')
        if i in wanted:
            ok,bgr=cap.retrieve()
            if not ok: raise RuntimeError(f'Retrieve {i}')
            im=Image.fromarray(cv2.cvtColor(bgr[y:y+h,x:x+w],cv2.COLOR_BGR2RGB)).resize((args.size,new_h),Image.Resampling.BICUBIC)
            ar=np.asarray(im).copy()
            if new_h>args.size: ar=ar[(new_h-args.size)//2:(new_h-args.size)//2+args.size]
            frames.append(ar)
    cap.release(); frames=np.stack(frames)
    print('INPUT',args.key,frames.shape,flush=True)
    model=VGGT(enable_point=False,enable_track=False)
    checkpoint=torch.load(ROOT/'repos/vggt/ckpt/model.pt',map_location='cpu',weights_only=True)
    expected=model.state_dict(); selected={k:v for k,v in checkpoint.items() if k in expected}
    ignored=set(checkpoint)-set(selected)
    assert all(k.startswith(('point_head.','track_head.')) for k in ignored),ignored
    model.load_state_dict(selected,strict=True)
    del checkpoint,expected,selected; gc.collect(); model.eval().cuda()
    torch.cuda.reset_peak_memory_stats(); infer=time.perf_counter(); windows=[]
    for start in range(0,len(ids),args.window-args.overlap):
        end=min(start+args.window,len(ids))
        images=torch.from_numpy(frames[start:end]).permute(0,3,1,2).float().cuda()/255.
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            out=model(images)
        ex,ki=pose_encoding_to_extri_intri(out['pose_enc'].float(),images.shape[-2:])
        ex=ex.cpu().numpy()[0]; c2w=np.tile(np.eye(4),(len(ex),1,1))
        c2w[:,:3,:3]=ex[:,:3,:3].transpose(0,2,1)
        c2w[:,:3,3]=-np.einsum('nij,nj->ni',c2w[:,:3,:3],ex[:,:3,3])
        item=dict(start=np.array(start),c2w=c2w,intrinsics=ki.cpu().numpy()[0],
                  depth=out['depth'][0,:,::4,::4,0].float().cpu().numpy(),
                  conf=out['depth_conf'][0,:,::4,::4].float().cpu().numpy())
        np.savez_compressed(raw/f'{start:06d}.npz',**item); windows.append(item)
        del out,images,ex,ki
        print(f'WINDOW {start}:{end}/{len(ids)} peak={torch.cuda.max_memory_allocated()/2**30:.2f}GiB',flush=True)
        if end==len(ids): break
    torch.cuda.synchronize(); seconds=time.perf_counter()-infer
    world,alignment=stitch_windows(windows,'depth')
    prediction=dict(key=args.key,frame_indices=ids.tolist(),times=times[ids].tolist(),positions=world[:,:3,3].tolist())
    validate_prediction(prediction,args.key,times); write_json(pred,prediction)
    np.savez_compressed(diag/f'{args.key}.poses.npz',camera_to_world=world,input_frame_indices=ids)
    write_json(diag/f'{args.key}.json',dict(arguments=vars(args)|{'output':str(run)},input_frames=len(ids),output_frames=len(world),alignment=alignment,
        inference_seconds=seconds,elapsed_seconds=time.perf_counter()-began,
        peak_cuda_allocated_gib=torch.cuda.max_memory_allocated()/2**30,peak_cuda_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
        environment=dict(python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0)),
        note='Pretrained VGGT-1B. Camera and depth heads. Overlap rotations + median confident depth ratio; no training or reference labels.'))
    print('CLIP_COMPLETE',args.key,len(ids),seconds,flush=True)
if __name__=='__main__': main()
