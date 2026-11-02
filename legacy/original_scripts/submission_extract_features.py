"""Extract the frozen v3 regressor's visual features from a supplied manifest."""
import argparse,sys,time
from pathlib import Path
import cv2,numpy as np
from extract_speed_features_v2 import features
from probe_pairs import probe_pairs
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'repos/lookout-vo-challenge/starter'))
from public_dataset import load_manifest,load_timestamps,select_clips
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--dataset',type=Path,required=True);ap.add_argument('--key',required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    cv2.setNumThreads(2);a.output.mkdir(parents=True,exist_ok=True);dest=a.output/f'{a.key}.npz'
    if dest.exists():raise FileExistsError(dest)
    data=a.dataset.parent;c=select_clips(load_manifest(a.dataset),clips=a.key)[0];ts=load_timestamps(data,c);ids,before=probe_pairs(ts)
    pairs=dict(zip(ids,before));wanted=set(ids)|set(before);recent={};xx=[];x,y,w,h=c['calibration']['active_rect'];cap=cv2.VideoCapture(str(data/c['video']))
    for i in range(int(ids[-1])+1):
        if not cap.grab():raise RuntimeError(f'Decode {a.key}: {i}')
        if i not in wanted:continue
        ok,bgr=cap.retrieve()
        if not ok:raise RuntimeError(i)
        rgb=cv2.resize(cv2.cvtColor(bgr[y:y+h,x:x+w],cv2.COLOR_BGR2RGB),(160,90),interpolation=cv2.INTER_AREA);recent[i]=rgb
        if i in pairs:
            j=pairs[i];xx.append(features(cv2.cvtColor(recent[j],cv2.COLOR_RGB2GRAY),rgb,ts[i]-ts[j],c['calibration']));recent={k:v for k,v in recent.items() if k>=i}
    cap.release();xx=np.asarray(xx);assert np.all(np.isfinite(xx))
    np.savez_compressed(dest,features=xx,times=ts[ids],indices=ids,split=np.array(c['split']),camera_mode=np.array(c['camera_mode']))
    print('COMPLETE',a.key,len(xx),flush=True)
if __name__=='__main__':main()
