"""Apply the frozen 9.7548757% validation recipe to arbitrary released clips.

No reference trajectories or navigation signals are opened. The optional regression
check compares against saved validation predictions, not ground truth.
"""
import argparse,json,pickle,shutil,sys
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter1d,median_filter
from scipy.spatial.transform import Rotation,Slerp
from fuse_headings import planar_yaw
from horizon_geometry import fit_horizon
from probe_pairs import probe_pairs
from water_current import estimate_current
from train_speed_regressor_v3 import inputs,normalize_speed
ROOT=Path(__import__('os').environ['TIDEGRAPH_RAW_WORK']);sys.path.insert(0,str(ROOT/'repos/lookout-vo-challenge/starter'))
from public_dataset import load_manifest,select_clips,load_timestamps,validate_prediction,write_json,sha256

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--dataset',type=Path,required=True);ap.add_argument('--split',choices=['val','test'],required=True)
    for key in ['r3','vggt','klt','water','features','output']:ap.add_argument('--'+key,type=Path,required=True)
    ap.add_argument('--keys');ap.add_argument('--model',type=Path,default=ROOT/'models/flow_speed_v3');ap.add_argument('--regression',type=Path);a=ap.parse_args()
    run=a.output;run.mkdir(parents=True,exist_ok=True);(run/'diagnostics').mkdir(exist_ok=True);(run/'provenance').mkdir(exist_ok=True)
    for n in ['submission_fuse.py','fuse_headings.py','horizon_geometry.py','probe_pairs.py','water_current.py','train_speed_regressor_v3.py']:
        shutil.copy2(Path(__file__).parent/n,run/'provenance'/n)
    record=json.loads((a.model/'training_record.json').read_text());assert record['training_split']=='train' and record['train_clip_count']==68
    with (a.model/'model.pkl').open('rb') as f:model=pickle.load(f)
    shutil.copy2(a.model/'training_record.json',run/'provenance')
    clips=select_clips(load_manifest(a.dataset),a.split,a.keys);data=a.dataset.parent
    manifest=dict(method='Frozen R3/VGGT/KLT + dense signed water/current + TRAIN v3 speed fallback',dataset_sha256=sha256(a.dataset),model_sha256=sha256(a.model/'model.pkl'),settings={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},clips=[],expected_keys=[c['key'] for c in clips])
    for c in clips:
        key=c['key'];ts=load_timestamps(data,c);angles=[];poses_all=[];checks={};t=None
        for src in [a.r3,a.vggt,a.klt]:
            pp=src/'predictions'/f'{key}.json';po=src/'diagnostics'/f'{key}.poses.npz';pred=json.loads(pp.read_text());pt=np.array(pred['times']);poses=np.load(po)['camera_to_world']
            if t is None:t=pt;ids=pred['frame_indices']
            assert np.array_equal(t,pt),f'{key}: orientation timestamps differ'
            angles.append(np.interp(t,pt,planar_yaw(poses)));poses_all.append(poses);checks[str(src)]={'prediction':sha256(pp),'poses':sha256(po)}
        angles=np.asarray(angles);dv=angles[1]-angles[0];dk=angles[2]-angles[0]
        flag=(np.abs(dv)>np.deg2rad(20))&(np.abs(dk)>np.deg2rad(20))&(dv*dk>0)&(np.abs(angles[1]-angles[2])<np.deg2rad(30))
        selected=1 if float(flag.mean())>=.2 else 0
        yaw=gaussian_filter1d(angles[selected],2/np.median(np.diff(t)));poses=poses_all[selected]
        wp=a.water/'diagnostics'/f'{key}.water.npz';z=np.load(wp);vt=np.array(json.loads((a.water/'predictions'/f'{key}.json').read_text())['times'])
        v=z['raw'][:,4:7].copy();dt=z['pair_dt'];valid=np.all(np.isfinite(v),axis=1)&(dt<=.125+1e-6)
        if valid.sum()<10:raise ValueError(f'{key}: fewer than 10 valid water observations')
        for d in range(3):v[:,d]=np.interp(vt,vt[valid],v[valid,d])
        norm=np.linalg.norm(v,axis=1);v*=np.minimum(1,np.quantile(norm,.95)/np.maximum(norm,1e-8))[:,None]
        rot=Slerp(t,Rotation.from_matrix(poses[:,:3,:3]))(np.clip(vt-dt,t[0],t[-1])).as_matrix()
        down=np.median(poses[:,:3,1],axis=0);down/=np.linalg.norm(down)
        vel=np.einsum('nij,nj->ni',rot,v);vel-=np.outer(vel@down,down)
        sample_dt=np.median(np.diff(vt));window=max(3,int(6/sample_dt)//2*2+1)
        vel=gaussian_filter1d(median_filter(vel,size=(window,1)),5/sample_dt,axis=0)
        heading=rot[:,:,2].copy();heading-=np.outer(heading@down,down);heading/=np.maximum(np.linalg.norm(heading,axis=1,keepdims=True),1e-8)
        current,residual=estimate_current(vel,heading,down,.2,True)
        speed=np.sum((vel+current)*heading,axis=1);water=np.interp(t,vt,speed)
        # Horizon observations at the original 0.5 FPS audit cadence are a
        # subset of the 2 FPS water extraction; per-frame measurements are identical.
        sparse_ids,_=probe_pairs(ts);wanted=np.unique(np.r_[z['frame_indices'],z['pair_frame_indices']]);where=np.searchsorted(wanted,sparse_ids)
        assert np.array_equal(wanted[where],sparse_ids)
        observed=z['horizon'][where]
        fitted,normal,plane_residual=fit_horizon(poses_all[0],t,ts[sparse_ids],observed,c['calibration'])
        q90=float(np.quantile(plane_residual,.9))
        fp=a.features/f'{key}.npz'
        with np.load(fp) as f:
            alpha=record['selected_blend'];ps=(1-alpha)+alpha*normalize_speed(model.predict(inputs(f)),record['selected_target']);prior=np.interp(t,f['times'],ps)
        water/=max(np.mean(np.abs(water)),1e-8);prior/=max(prior.mean(),1e-8)
        agreement=float(np.corrcoef(water,prior)[0,1]) if min(np.std(water),np.std(prior))>1e-6 else 0.
        weight=float(q90>.1)
        if agreement>=.7:weight=0.
        speed=water*(1-weight)+prior*weight
        klt=gaussian_filter1d(angles[2],2/np.median(np.diff(t)));yaw=.75*yaw+.25*klt
        speed=np.sign(speed)*np.abs(speed)**1.
        velocity=np.stack([np.sin(yaw),np.zeros(len(t)),np.cos(yaw)],axis=1)*speed[:,None]
        positions=np.vstack([np.zeros(3),np.cumsum(.5*(velocity[:-1]+velocity[1:])*np.diff(t)[:,None],axis=0)])
        pred=dict(key=key,frame_indices=ids,times=t.tolist(),positions=positions.tolist());validate_prediction(pred,key,ts)
        assert np.max(np.diff(t))<=1 and t[0]==ts[0] and t[-1]==ts[-1]
        write_json(run/'predictions'/f'{key}.json',pred)
        np.savez_compressed(run/'diagnostics'/f'{key}.fusion.npz',times=t,yaw=yaw,speed=speed,current=current,plane_residual=plane_residual)
        entry=dict(key=key,status='predicted',selected_orientation=['r3','vggt'][selected],disagreement_fraction=float(flag.mean()),q90_plane_residual=q90,speed_correlation=agreement,learned_speed_weight=weight,n_poses=len(t),valid_water_observations=int(valid.sum()),max_prediction_gap_s=float(np.max(np.diff(t))),source_sha256=checks,water_sha256=sha256(wp),features_sha256=sha256(fp))
        if a.regression:
            expected=json.loads((a.regression/'predictions'/f'{key}.json').read_text());delta=float(np.max(np.abs(positions-np.array(expected['positions']))));entry['regression_max_abs_position_difference']=delta
            np.testing.assert_allclose(positions,expected['positions'],rtol=0,atol=1e-8)
        manifest['clips'].append(entry);write_json(run/'inference_manifest.json',manifest);print('FUSED',key,entry['selected_orientation'],'prior',weight,flush=True)
    print('COMPLETE',len(clips),flush=True)
if __name__=='__main__':main()
