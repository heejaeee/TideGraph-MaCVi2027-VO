"""Robust image-only heading consensus; a single rule applied to every clip."""
import argparse,json,shutil,subprocess,sys
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter1d,median_filter
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'repos/lookout-vo-challenge/starter'))
from public_dataset import load_manifest,select_clips,load_timestamps,validate_prediction,write_json,sha256

def planar_yaw(poses):
    vertical=np.median(poses[:,:3,1],axis=0);vertical/=np.linalg.norm(vertical)
    heading=poses[:,:3,2].copy();heading-=np.outer(heading@vertical,vertical)
    forward=heading[0]/np.linalg.norm(heading[0]);right=np.cross(vertical,forward);right/=np.linalg.norm(right)
    return np.unwrap(np.arctan2(heading@right,heading@forward))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--sources',nargs='+',required=True);ap.add_argument('--name',required=True);ap.add_argument('--clips',default=None);ap.add_argument('--speed-source',default=None);ap.add_argument('--fusion',choices=['median','guarded'],default='median');a=ap.parse_args()
    assert len(a.sources)>=3
    data=ROOT/'data/release';selected=select_clips(load_manifest(data/'release_manifest.json'),'val',a.clips)
    run=ROOT/'runs'/a.name;run.mkdir(exist_ok=False);(run/'provenance').mkdir();(run/'diagnostics').mkdir();shutil.copy2(__file__,run/'provenance')
    manifest=dict(method=a.fusion+' planar heading from R3, VGGT and calibrated KLT',seed=0,target_fps=2,environment={},settings=vars(a),expected_keys=[c['key'] for c in selected],clips=[])
    for c in selected:
        key=c['key'];angles=[];checks={};times=None
        for source in a.sources:
            src=ROOT/'runs'/source;predpath=src/'predictions'/f'{key}.json';pred=json.loads(predpath.read_text());pt=np.array(pred['times'])
            posespath=src/'diagnostics'/f'{key}.poses.npz';poses=np.load(posespath)['camera_to_world']
            yaw=planar_yaw(poses)
            if times is None:times=pt;ids=pred['frame_indices']
            if pt[0]>times[0]+1e-6 or pt[-1]<times[-1]-1e-6:raise ValueError('Source has incomplete time coverage')
            angles.append(np.interp(times,pt,yaw));checks[source]=dict(prediction_sha256=sha256(predpath),poses_sha256=sha256(posespath))
        angles=np.asarray(angles)
        selected_model='median'
        disagreement_fraction=None
        if a.fusion=='guarded':
            assert len(angles)==3, 'Order: R3 primary, VGGT fallback, KLT independent check'
            dv=angles[1]-angles[0];dk=angles[2]-angles[0]
            flag=(np.abs(dv)>np.deg2rad(20))&(np.abs(dk)>np.deg2rad(20))&(dv*dk>0)&(np.abs(angles[1]-angles[2])<np.deg2rad(30))
            disagreement_fraction=float(flag.mean())
            source_index=1 if disagreement_fraction>=.2 else 0
            angle=angles[source_index];selected_model=a.sources[source_index]
        else:angle=np.median(angles,axis=0)
        angle=gaussian_filter1d(angle,2/np.median(np.diff(times)))
        speed=np.ones(len(times))
        if a.speed_source:
            spath=ROOT/'runs'/a.speed_source/'predictions'/f'{key}.json'
            sp=json.loads(spath.read_text());st=np.array(sp['times']);sx=np.array(sp['positions']);ds=np.median(np.diff(st))
            sx=gaussian_filter1d(sx,1./ds,axis=0)
            sv=np.linalg.norm(np.gradient(sx,st,axis=0),axis=1)
            sv=median_filter(sv,size=max(3,int(1./ds)//2*2+1))
            sv=gaussian_filter1d(sv,5./ds);sv/=max(np.mean(sv),1e-8)
            speed=np.interp(times,st,sv)
            checks['speed_carrier']={'path':str(spath),'sha256':sha256(spath)}
        vel=np.stack([np.sin(angle),np.zeros(len(angle)),np.cos(angle)],axis=1)*speed[:,None]
        positions=np.vstack([np.zeros(3),np.cumsum(.5*(vel[:-1]+vel[1:])*np.diff(times)[:,None],axis=0)])
        pred=dict(key=key,frame_indices=ids,times=times.tolist(),positions=positions.tolist());validate_prediction(pred,key,load_timestamps(data,c));write_json(run/'predictions'/f'{key}.json',pred)
        np.savez_compressed(run/'diagnostics'/f'{key}.headings.npz',source_yaws=angles,consensus_yaw=angle,speed=speed)
        manifest['clips'].append(dict(key=key,status='predicted',n_poses=len(times),source_hashes=checks,selected_orientation=selected_model,disagreement_fraction=disagreement_fraction))
    write_json(run/'inference_manifest.json',manifest)
    subprocess.run([sys.executable,str(ROOT/'repos/lookout-vo-challenge/starter/evaluate.py'),'--dataset',str(data/'release_manifest.json'),'--predictions',str(run),'--clips',','.join(c['key'] for c in selected)],check=True)
    subprocess.run([sys.executable,str(ROOT/'scripts/summarize_run.py'),'--run',str(run)],check=True)
if __name__=='__main__':main()
