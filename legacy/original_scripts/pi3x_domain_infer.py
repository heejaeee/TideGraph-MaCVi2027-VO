"""Frozen Pi3X inference using released images/calibration only, no references."""
import argparse,gc,json,sys,time,shutil,hashlib
from pathlib import Path
import cv2,numpy as np,torch
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'repos/Pi3'));sys.path.insert(0,str(ROOT/'repos/lookout-vo-challenge/starter'))
from pi3.models.pi3x import Pi3X
from safetensors.torch import load_file
from public_dataset import load_manifest,select_clips,load_timestamps,validate_prediction,write_json
from window_geometry import stitch_windows

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--keys',required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--dataset',type=Path,default=ROOT/'data/release/release_manifest.json');ap.add_argument('--intrinsics',action='store_true');ap.add_argument('--size',type=int,default=448);ap.add_argument('--window',type=int,default=12);ap.add_argument('--overlap',type=int,default=6);ap.add_argument('--fps',type=float,default=2)
    a=ap.parse_args();assert a.size%14==0 and 2<=a.overlap<a.window
    cv2.setNumThreads(2);torch.manual_seed(0);np.random.seed(0)
    a.output.mkdir(parents=True,exist_ok=True);(a.output/'provenance').mkdir(exist_ok=True)
    shutil.copy2(__file__,a.output/'provenance'/Path(__file__).name)
    provenance=json.loads((ROOT/'reports/domain_v3/pi3x_source.json').read_text());write_json(a.output/'provenance/model.json',provenance)
    model=Pi3X();weights=load_file(provenance['checkpoint']);model.load_state_dict(weights,strict=True);del weights
    if not a.intrinsics:model.disable_multimodal()
    gc.collect();model.eval().cuda()
    for c in select_clips(load_manifest(a.dataset),clips=a.keys):
        key=c['key'];pp=a.output/'predictions'/f'{key}.json'
        if pp.exists():continue
        began=time.perf_counter();data=a.dataset.parent;ts=load_timestamps(data,c)
        ids=np.unique(np.r_[np.searchsorted(ts,np.arange(ts[0],ts[-1],1/a.fps)),len(ts)-1]);wanted=set(ids)
        x,y,w,h=c['calibration']['active_rect'];height=round(h*a.size/w/14)*14
        cap=cv2.VideoCapture(str(data/c['video']));frames=[]
        for i in range(int(ids[-1])+1):
            if not cap.grab():raise RuntimeError(f'Decode {key} {i}')
            if i not in wanted:continue
            ok,bgr=cap.retrieve()
            if not ok:raise RuntimeError(f'Retrieve {key} {i}')
            im=Image.fromarray(cv2.cvtColor(bgr[y:y+h,x:x+w],cv2.COLOR_BGR2RGB)).resize((a.size,height),Image.Resampling.LANCZOS)
            frames.append(np.asarray(im).copy())
        cap.release();frames=np.stack(frames);cal=c['calibration']
        k=np.array([[cal['fx']*a.size/w,0,cal['cx']*a.size/w],[0,cal['fy']*height/h,cal['cy']*height/h],[0,0,1]],dtype=np.float32)
        raw=a.output/'windows'/key;raw.mkdir(parents=True,exist_ok=True);windows=[];torch.cuda.reset_peak_memory_stats();infer=time.perf_counter()
        for start in range(0,len(ids),a.window-a.overlap):
            end=min(start+a.window,len(ids));dest=raw/f'{start:06d}.npz'
            if dest.exists():
                with np.load(dest) as z:item={n:z[n].copy() for n in z.files}
            else:
                im=torch.from_numpy(frames[start:end]).permute(0,3,1,2).float().cuda()[None]/255.
                kwargs={'intrinsics':torch.from_numpy(k).cuda()[None,None].expand(1,end-start,3,3)} if a.intrinsics else {}
                with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):out=model(im,**kwargs)
                item=dict(start=np.array(start),c2w=out['camera_poses'][0].float().cpu().numpy(),depth=out['local_points'][0,:,::4,::4,2].float().cpu().numpy(),conf=out['conf'][0,:,::4,::4,0].sigmoid().float().cpu().numpy())
                np.savez_compressed(dest,**item);del im,out,kwargs
            windows.append(item)
            if start%60==0:print('WINDOW',key,start,end,len(ids),'peak_GiB',round(torch.cuda.max_memory_allocated()/2**30,2),flush=True)
            if end==len(ids):break
        world,alignment=stitch_windows(windows,'depth');pred=dict(key=key,frame_indices=ids.tolist(),times=ts[ids].tolist(),positions=world[:,:3,3].tolist())
        validate_prediction(pred,key,ts);write_json(pp,pred);(a.output/'diagnostics').mkdir(exist_ok=True)
        np.savez_compressed(a.output/'diagnostics'/f'{key}.poses.npz',camera_to_world=world,input_frame_indices=ids)
        write_json(a.output/'diagnostics'/f'{key}.json',dict(arguments={n:str(v) if isinstance(v,Path) else v for n,v in vars(a).items()},model=provenance,alignment=alignment,frames=len(ids),inference_seconds=time.perf_counter()-infer,elapsed_seconds=time.perf_counter()-began,peak_cuda_allocated_gib=torch.cuda.max_memory_allocated()/2**30))
        print('COMPLETE',key,time.perf_counter()-began,flush=True)
if __name__=='__main__':main()
