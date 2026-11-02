"""Experimental water-plane speed from images/calibration, with no label access.
Assumes a locally planar water surface and constant camera height; the unknown
height is a single global scale. Wave motion violates this model and is logged.
"""
import argparse,sys,time,platform
from pathlib import Path
import cv2,numpy as np
from scipy.ndimage import gaussian_filter1d,median_filter
from rotation_flow_infer_clip import fit_rotation
from probe_pairs import probe_pairs
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'repos/lookout-vo-challenge/starter'))
from public_dataset import load_manifest,load_timestamps,select_clips,validate_prediction,write_json

def track(a,b,mask,n):
    p=cv2.goodFeaturesToTrack(a,n,.005,8,mask=mask)
    if p is None:return np.empty((0,2)),np.empty((0,2))
    q,s,_=cv2.calcOpticalFlowPyrLK(a,b,p,None,winSize=(25,25),maxLevel=4)
    back,bs,_=cv2.calcOpticalFlowPyrLK(b,a,q,None,winSize=(25,25),maxLevel=4)
    good=(s[:,0]>0)&(bs[:,0]>0)&(np.linalg.norm(back[:,0]-p[:,0],axis=1)<1.)
    return p[good,0],q[good,0]

def horizon(gray):
    row=np.median(gray[:,int(.05*gray.shape[1]):int(.95*gray.shape[1])],axis=1)
    edge=np.abs(np.gradient(gaussian_filter1d(row.astype(float),2)))
    lo,hi=int(.4*len(row)),int(.85*len(row))
    return lo+np.argmax(edge[lo:hi])

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--key',required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--fps',type=float,default=2);ap.add_argument('--size',type=int,default=960);ap.add_argument('--max-frames',type=int,default=0);ap.add_argument('--translation-model',choices=['full','forward'],default='full');ap.add_argument('--residual-units',choices=['pixel','world'],default='pixel')
    ap.add_argument('--dataset',required=True)
    a=ap.parse_args();cv2.setNumThreads(2);rng=np.random.default_rng(0);begin=time.perf_counter();run=a.output;run.mkdir(exist_ok=True,parents=True);(run/'diagnostics').mkdir(exist_ok=True)
    data=Path(a.dataset).resolve().parent;c=select_clips(load_manifest(Path(a.dataset)),clips=a.key)[0];ts=load_timestamps(data,c)
    ids,before=probe_pairs(ts,a.fps,.125,a.max_frames)
    pairs=dict(zip(ids,before));wanted=set(ids)|set(before);recent={}
    cal=c['calibration'];x,y,w,h=cal['active_rect'];scale=a.size/w;hh=round(h*scale)
    k=np.array([[cal['fx']*scale,0,cal['cx']*scale],[0,cal['fy']*scale,cal['cy']*scale],[0,0,1.]])
    inv=np.linalg.inv(k);cap=cv2.VideoCapture(str(data/c['video']));last=None;out=[];hy=[]
    for i in range(int(ids[-1])+1):
        if not cap.grab():raise RuntimeError(i)
        if i not in wanted:continue
        ok,im=cap.retrieve()
        if not ok:raise RuntimeError(i)
        gray=cv2.resize(cv2.cvtColor(im[y:y+h,x:x+w],cv2.COLOR_BGR2GRAY),(a.size,hh));hy.append(horizon(gray))
        recent[i]=gray
        if i not in pairs:continue
        j=pairs[i];last=recent[j];last_t=ts[j]
        if last is not None:
            mask=np.zeros_like(gray);mask[int(.08*hh):int(.8*hh),int(.05*a.size):int(.95*a.size)]=255
            p,q=track(last,gray,mask,1000);pa=np.c_[p,np.ones(len(p))]@inv.T;qb=np.c_[q,np.ones(len(q))]@inv.T
            pa/=np.maximum(np.linalg.norm(pa,axis=1,keepdims=True),1e-8);qb/=np.maximum(np.linalg.norm(qb,axis=1,keepdims=True),1e-8)
            r,inside=fit_rotation(pa,qb,rng,2./k[0,0])
            horizon_y=float(np.median(hy[-5:-1])) if len(hy)>1 else hy[-1]
            mask[:]=0; mask[min(hh-20,int(horizon_y+20)):int(.95*hh),int(.05*a.size):int(.95*a.size)]=255
            mask[int(.8*hh):,int(.25*a.size):int(.75*a.size)]=0
            p,q=track(last,gray,mask,600);pa=np.c_[p,np.ones(len(p))]@inv.T;qb=np.c_[q,np.ones(len(q))]@inv.T
            n=np.array([0.,1.,(k[1,2]-horizon_y)/k[1,1]]);n/=np.linalg.norm(n)
            valid=(pa@n)>.01;pa=pa[valid];qb=qb[valid]
            if len(pa)>=10:
                xyz=pa/(pa@n)[:,None];xyz=xyz@r.T
                A=np.zeros((len(pa),2,3));A[:,0,0]=1;A[:,1,1]=1;A[:,0,2]=-qb[:,0];A[:,1,2]=-qb[:,1]
                target=xyz[:,:2]-qb[:,:2]*xyz[:,2,None]
                if a.residual_units=='pixel':
                    # Optical-flow errors are image-coordinate errors. Weight
                    # by inverse depth so near-horizon world residuals do not
                    # dominate just because their assumed depth is large.
                    iz=1/np.maximum(np.abs(xyz[:,2]),.1)
                    A*=iz[:,None,None];target*=iz[:,None]
                axis=np.array([0.,-n[2],n[1]]);axis/=np.linalg.norm(axis)
                if a.translation_model=='forward':A=np.einsum('nij,j->ni',A,r@axis)[:,:,None]
                weights=np.ones(len(pa))
                for _ in range(5):
                    sw=np.sqrt(weights)[:,None,None];v=np.linalg.lstsq((A*sw).reshape(-1,A.shape[-1]),(target*np.sqrt(weights)[:,None]).ravel(),rcond=None)[0]
                    err=np.linalg.norm(np.einsum('nij,j->ni',A,v)-target,axis=1)
                    weights=1/np.maximum(1,err/max(np.median(err),1e-6))
                if a.translation_model=='forward':v=axis*float(v[0])
                else:v=v@r;v-=np.dot(v,n)*n
                out.append([np.linalg.norm(v)/(ts[i]-last_t),len(pa),float(np.median(err)),horizon_y,*(v/(ts[i]-last_t)).tolist()])
            else:out.append([float('nan'),len(pa),float('nan'),horizon_y,float('nan'),float('nan'),float('nan')])
        recent={k:v for k,v in recent.items() if k>=i}
    cap.release();out=np.asarray(out);speed=out[:,0];valid=np.isfinite(speed)
    if valid.sum()<10:raise RuntimeError('No usable plane-flow speeds')
    speed=np.interp(np.arange(len(speed)),np.flatnonzero(valid),speed[valid]);speed=np.minimum(speed,np.quantile(speed,.95));speed=gaussian_filter1d(median_filter(speed,size=3),2.5)
    pos=np.zeros((len(ids),3));pos[1:,0]=np.cumsum(.5*(speed[:-1]+speed[1:])*np.diff(ts[ids]))
    pred=dict(key=a.key,frame_indices=ids.tolist(),times=ts[ids].tolist(),positions=pos.tolist());validate_prediction(pred,a.key,ts);write_json(run/'predictions'/f'{a.key}.json',pred)
    np.savez_compressed(run/'diagnostics'/f'{a.key}.water.npz',raw=out,speed=speed,horizon=hy,frame_indices=ids,pair_frame_indices=before,probe_times=ts[ids],pair_dt=ts[ids]-ts[before])
    write_json(run/'diagnostics'/f'{a.key}.json',dict(output_frames=len(ids),inference_seconds=time.perf_counter()-begin,peak_cuda_allocated_gib=0,environment={'python':platform.python_version()},arguments=vars(a)|{'output':str(run)}))
    print('COMPLETE',a.key,time.perf_counter()-begin,flush=True)
if __name__=='__main__':main()
