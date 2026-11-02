"""Compare nominal versus image-estimated intrinsics on identical KLT matches."""
import argparse, json, sys, time
from pathlib import Path
import cv2
import numpy as np

ROOT=Path(__import__('os').environ['TIDEGRAPH_RAW_WORK'])
sys.path.insert(0,str(ROOT/'repos/lookout-vo-challenge/starter'))
from public_dataset import load_manifest,select_clips,load_timestamps,write_json
from rotation_flow_infer_clip import fit_rotation


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--keys',required=True)
    ap.add_argument('--calibration',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--dataset',type=Path,default=ROOT/'data/release/release_manifest.json')
    a=ap.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    cv2.setNumThreads(2)
    cv2.setRNGSeed(0)
    data=a.dataset.parent
    for c in select_clips(load_manifest(a.dataset),clips=a.keys):
        key=c['key'];dest=a.output/f'{key}.npz'
        if dest.exists():continue
        started=time.perf_counter()
        estimate=json.loads((a.calibration/f'{key}.json').read_text())
        x,y,w,h=c['calibration']['active_rect']
        width=960;height=round(h*width/w)
        cal=c['calibration']
        nominal=np.array([cal['fx'],cal['fy'],cal['cx']-x,cal['cy']-y])
        learned=np.array(estimate['median'])
        ks={}
        for name,vals in [('nominal',nominal),('learned',learned)]:
            fx,fy,cx,cy=vals
            ks[name]=np.array([[fx*width/w,0,cx*width/w],[0,fy*height/h,cy*height/h],[0,0,1.]])
        ts=load_timestamps(data,c)
        ids=np.unique(np.r_[np.searchsorted(ts,np.arange(ts[0],ts[-1],.5)),len(ts)-1]);wanted=set(ids)
        cap=cv2.VideoCapture(str(data/c['video']));last=None
        rotations={name:[] for name in ks};stats=[];pairs=[];offsets=[0]
        for i in range(int(ids[-1])+1):
            if not cap.grab():raise RuntimeError((key,i))
            if i not in wanted:continue
            ok,bgr=cap.retrieve()
            if not ok:raise RuntimeError((key,i))
            gray=cv2.resize(cv2.cvtColor(bgr[y:y+h,x:x+w],cv2.COLOR_BGR2GRAY),(width,height))
            if last is not None:
                mask=np.zeros_like(last);mask[int(.12*height):int(.62*height),int(.05*width):int(.95*width)]=255
                pts=cv2.goodFeaturesToTrack(last,1200,.003,6,mask=mask)
                p=np.empty((0,2));q=p.copy()
                if pts is not None:
                    nxt,ok,_=cv2.calcOpticalFlowPyrLK(last,gray,pts,None,winSize=(25,25),maxLevel=4)
                    back,bok,_=cv2.calcOpticalFlowPyrLK(gray,last,nxt,None,winSize=(25,25),maxLevel=4)
                    good=(ok[:,0]>0)&(bok[:,0]>0)&(np.linalg.norm(back[:,0]-pts[:,0],axis=1)<.75)
                    p=pts[good,0];q=nxt[good,0]
                pairs.append(np.c_[p,q]);offsets.append(offsets[-1]+len(p))
                counts=[len(p)]
                for name,k in ks.items():
                    aa=np.c_[p,np.ones(len(p))]@np.linalg.inv(k).T
                    bb=np.c_[q,np.ones(len(q))]@np.linalg.inv(k).T
                    aa/=np.linalg.norm(aa,axis=1,keepdims=True)
                    bb/=np.linalg.norm(bb,axis=1,keepdims=True)
                    # Same RANSAC draws for each calibration and frame pair.
                    r,inside=fit_rotation(aa,bb,np.random.default_rng(len(stats)),1.5/k[0,0])
                    rotations[name].append(r);counts.append(int(inside.sum()))
                stats.append(counts)
            last=gray
        cap.release()
        np.savez_compressed(dest,times=ts[ids],frame_indices=ids,counts=stats,
            matches=np.concatenate(pairs).astype(np.float32),offsets=offsets,
            nominal_k=ks['nominal'],learned_k=ks['learned'],
            nominal_rotation=rotations['nominal'],learned_rotation=rotations['learned'])
        print('COMPLETE',key,'seconds',round(time.perf_counter()-started,1),flush=True)
    write_json(a.output/'settings.json',dict(method='Identical upper-background candidate KLT tracks, two independent calibration sources, no references',keys=a.keys,calibration=str(a.calibration)))


if __name__=='__main__':main()
