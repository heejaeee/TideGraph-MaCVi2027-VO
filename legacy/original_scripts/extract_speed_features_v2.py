"""Image-only speed features. This program never opens trajectory references."""
import argparse,json,sys,time,shutil
from pathlib import Path
import cv2,numpy as np
from scipy.ndimage import gaussian_filter1d
from probe_pairs import probe_pairs
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'repos/lookout-vo-challenge/starter'))
from public_dataset import load_manifest,load_timestamps

def features(prev,rgb,dt,cal):
    gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
    flow=cv2.calcOpticalFlowFarneback(prev,gray,None,.5,3,15,3,5,1.2,0)/(max(dt,1e-3))
    # Rates in calibrated normalized image coordinates (approximately rad/s).
    flow[:,:,0]/=cal['fx']*160/cal['width'];flow[:,:,1]/=cal['fy']*90/cal['height']
    row=np.median(gray[:,8:-8],axis=1);edge=np.abs(np.gradient(gaussian_filter1d(row.astype(float),1)))
    hy=36+np.argmax(edge[36:77]);values=[hy/90,cal['fx']/cal['width'],cal['fy']/cal['height']]
    for iy in range(3):
        for ix in range(4):
            ys=slice(iy*30,(iy+1)*30);xs=slice(ix*40,(ix+1)*40)
            colors=rgb[ys,xs].reshape(-1,3).astype(float)/255
            f=flow[ys,xs].reshape(-1,2)
            values.extend(colors.mean(0));values.extend(colors.std(0))
            values.extend(np.quantile(f,[.1,.5,.9],axis=0).ravel())
            values.extend(np.std(f,axis=0))
            values.append(float(np.mean(np.abs(cv2.Laplacian(gray[ys,xs],cv2.CV_32F))))/255)
    return np.array(values,dtype=np.float32)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--splits',default='train,val');ap.add_argument('--clips',default=None);a=ap.parse_args()
    cv2.setNumThreads(2);data=ROOT/'data/release';cache=ROOT/'data/features/flow_speed_v2';cache.mkdir(parents=True,exist_ok=True)
    manifest=load_manifest(data/'release_manifest.json');keys=set(a.clips.split(',')) if a.clips else None
    for c in manifest['clips']:
        if c['split'] not in a.splits.split(',') or (keys is not None and c['key'] not in keys):continue
        dest=cache/(c['key']+'.npz')
        if dest.exists():continue
        begin=time.perf_counter();ts=load_timestamps(data,c);ids=np.unique(np.searchsorted(ts,np.arange(ts[0]+.25,ts[-1],2.)))
        old_before=np.maximum(0,np.searchsorted(ts,ts[ids]-.125))
        ids,before=probe_pairs(ts)
        original=ROOT/'data/features/flow_speed_v1'/dest.name
        if np.array_equal(old_before,before) and original.exists():
            shutil.copy2(original,dest);print(c['key'],'reused identical input pairs',flush=True);continue
        pairs=dict(zip(ids,before));wanted=set(ids)|set(before)
        cap=cv2.VideoCapture(str(data/c['video']));x,y,w,h=c['calibration']['active_rect'];recent={};xx=[]
        for i in range(int(ids[-1])+1):
            if not cap.grab():raise RuntimeError(f'{c["key"]} decode {i}')
            if i not in wanted:continue
            ok,bgr=cap.retrieve()
            if not ok:raise RuntimeError(i)
            rgb=cv2.resize(cv2.cvtColor(bgr[y:y+h,x:x+w],cv2.COLOR_BGR2RGB),(160,90),interpolation=cv2.INTER_AREA)
            recent[i]=rgb
            if i in pairs:
                j=pairs[i];xx.append(features(cv2.cvtColor(recent[j],cv2.COLOR_RGB2GRAY),rgb,ts[i]-ts[j],c['calibration']))
                recent={k:v for k,v in recent.items() if k>=i}
        cap.release();np.savez_compressed(dest,features=np.array(xx),times=ts[ids],indices=ids,split=np.array(c['split']),camera_mode=np.array(c['camera_mode']))
        print(c['key'],c['split'],len(xx),f'{time.perf_counter()-begin:.1f}s',flush=True)
if __name__=='__main__':main()
