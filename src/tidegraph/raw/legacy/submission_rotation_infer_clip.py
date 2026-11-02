"""Calibrated robust bearing rotation from KLT tracks; image-only motion prior."""
import argparse,json,sys,time,platform
from pathlib import Path
import cv2,numpy as np
from scipy.ndimage import gaussian_filter1d
ROOT=Path(__import__('os').environ['TIDEGRAPH_RAW_WORK']);sys.path.insert(0,str(ROOT/'repos/lookout-vo-challenge/starter'))
from public_dataset import load_manifest,load_timestamps,select_clips,validate_prediction,write_json

def fit_rotation(a,b,rng,threshold):
    # Row-vector convention b = a @ R.T. RANSAC handles moving water/boats.
    count=len(a)
    if count<8:return np.eye(3),np.zeros(count,dtype=bool)
    samples=np.stack([rng.choice(count,3,replace=False) for _ in range(128)])
    h=np.einsum('nki,nkj->nij',b[samples],a[samples])
    u,_,vt=np.linalg.svd(h);d=np.tile(np.eye(3),(len(samples),1,1));d[:,2,2]=np.linalg.det(u@vt);rs=u@d@vt
    errors=np.linalg.norm(np.einsum('rij,nj->rni',rs,a)-b[None],axis=-1)
    inliers=errors<threshold; idx=np.argmax(inliers.sum(1)); mask=inliers[idx]
    if mask.sum()<6:return rs[idx],mask
    for _ in range(2):
        u,_,vt=np.linalg.svd(b[mask].T@a[mask]);d=np.diag([1,1,np.linalg.det(u@vt)]);r=u@d@vt
        mask=np.linalg.norm(a@r.T-b,axis=1)<threshold
        if mask.sum()<6:break
    return r,mask

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--key',required=True);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--fps',type=float,default=2);ap.add_argument('--size',type=int,default=960);ap.add_argument('--max-frames',type=int,default=0)
    ap.add_argument('--dataset',required=True)
    args=ap.parse_args();cv2.setNumThreads(2);rng=np.random.default_rng(0);began=time.perf_counter()
    run=args.output;run.mkdir(exist_ok=True,parents=True);(run/'diagnostics').mkdir(exist_ok=True)
    data=Path(args.dataset).resolve().parent;clip=select_clips(load_manifest(Path(args.dataset)),clips=args.key)[0];ts=load_timestamps(data,clip)
    ids=np.unique(np.r_[np.searchsorted(ts,np.arange(ts[0],ts[-1],1/args.fps)),len(ts)-1]);wanted=set(ids)
    if args.max_frames:ids=ids[:args.max_frames];wanted=set(ids)
    cal=clip['calibration'];x,y,w,h=cal['active_rect'];scale=args.size/w;hh=round(h*scale)
    k=np.array([[cal['fx']*scale,0,cal['cx']*scale],[0,cal['fy']*scale,cal['cy']*scale],[0,0,1.]])
    inv=np.linalg.inv(k);cap=cv2.VideoCapture(str(data/clip['video']));last=None;poses=[];stats=[];world=np.eye(4);pairs=[]
    for i in range(int(ids[-1])+1):
        if not cap.grab():raise RuntimeError(f'Decode {i}')
        if i not in wanted:continue
        ok,frame=cap.retrieve()
        if not ok:raise RuntimeError(f'Retrieve {i}')
        gray=cv2.resize(cv2.cvtColor(frame[y:y+h,x:x+w],cv2.COLOR_BGR2GRAY),(args.size,hh))
        if last is not None:
            mask=np.zeros_like(gray);mask[int(.08*hh):int(.8*hh),int(.05*args.size):int(.95*args.size)]=255
            pts=cv2.goodFeaturesToTrack(last,maxCorners=1000,qualityLevel=.005,minDistance=8,mask=mask)
            r=np.eye(3);nf=ni=0
            if pts is not None:
                nxt,ok,_=cv2.calcOpticalFlowPyrLK(last,gray,pts,None,winSize=(25,25),maxLevel=4)
                back,bok,_=cv2.calcOpticalFlowPyrLK(gray,last,nxt,None,winSize=(25,25),maxLevel=4)
                good=(ok[:,0]>0)&(bok[:,0]>0)&(np.linalg.norm(back[:,0]-pts[:,0],axis=1)<1.)
                p=pts[good,0];q=nxt[good,0];nf=len(p)
                a=np.c_[p,np.ones(nf)]@inv.T;b=np.c_[q,np.ones(nf)]@inv.T
                a/=np.linalg.norm(a,axis=1,keepdims=True);b/=np.linalg.norm(b,axis=1,keepdims=True)
                r,inside=fit_rotation(a,b,rng,2./k[0,0]);ni=int(inside.sum())
            world[:3,:3]=world[:3,:3]@r.T
            stats.append([nf,ni]);pairs.append(r)
        poses.append(world.copy());last=gray
    cap.release();poses=np.array(poses);t=ts[ids]
    vertical=np.median(poses[:,:3,1],axis=0);vertical/=np.linalg.norm(vertical)
    heading=poses[:,:3,2].copy();heading-=np.outer(heading@vertical,vertical)
    heading=gaussian_filter1d(heading,2/np.median(np.diff(t)),axis=0);heading/=np.maximum(np.linalg.norm(heading,axis=1,keepdims=True),1e-8)
    positions=np.vstack([np.zeros(3),np.cumsum((heading[:-1]+heading[1:])*.5*np.diff(t)[:,None],axis=0)])
    poses[:,:3,3]=positions
    pred=dict(key=args.key,frame_indices=ids.tolist(),times=t.tolist(),positions=positions.tolist());validate_prediction(pred,args.key,ts)
    write_json(run/'predictions'/f'{args.key}.json',pred)
    np.savez_compressed(run/'diagnostics'/f'{args.key}.poses.npz',camera_to_world=poses,input_frame_indices=ids,rotation_pairs=pairs,track_counts=stats)
    write_json(run/'diagnostics'/f'{args.key}.json',dict(output_frames=len(ids),inference_seconds=time.perf_counter()-began,peak_cuda_allocated_gib=0,environment={'python':platform.python_version()},arguments=vars(args)|{'output':str(run)}))
    print('COMPLETE',args.key,time.perf_counter()-began,flush=True)
if __name__=='__main__':main()
