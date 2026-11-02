"""Shared image-measurement extraction and frozen V11 inference on TEST.
First checks extraction parity against TRAIN caches. No trajectory labels.
"""
import argparse,datetime,time
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from v5_common import *
from v10_evidence_graph import score_rotation
from v10_bias import static_edges
from v11_temporal import OUT,REP,integrated_basis
from public_dataset import validate_prediction,sha256
import v6_test_common as test

def provider(key,split):
    if split=='test':
        case=test.get_case(key);case['headings']={'r3':case['r3_yaw']};track_dir=test.RUN/'cotracker_windows'/key;vggt_dir=ROOT/'runs/submission_test_v1/vggt/windows'/key
        poses=np.load(test.R3/'diagnostics'/f'{key}.poses.npz')['camera_to_world'];z=np.load(ROOT/'runs/domain_v4/calibrated_rotation_test'/f'{key}.npz');folder,meta=test.frames(key)
    else:
        case=get_case(key);track_dir=RUN/'cotracker_windows'/key;vggt_dir=ROOT/'runs/domain_probe_v2/vggt/windows'/key
        poses=np.load(ROOT/'runs/domain_probe_v2/r3/diagnostics'/f'{key}.poses.npz')['camera_to_world'];z=np.load(ROOT/'runs/domain_v4/calibrated_rotation_train'/f'{key}.npz');folder,meta=frames(key)
    assert np.array_equal(case['t'],z['times']) and len(case['t'])==len(poses)
    return case,track_dir,vggt_dir,poses,z['learned_k'],meta

def extract(key,split):
    dest=OUT/'transfer_features'/split/f'{key}.npz'
    if dest.exists():
        z=np.load(dest)
        if 'version' in z.files and int(z['version'])==2:return
    started=time.perf_counter();case,track_dir,vggt_dir,rp,k,meta=provider(key,split);windows=[dict(np.load(p)) for p in sorted(track_dir.glob('*.npz'))]
    vertical=np.median(rp[:,:3,1],axis=0);vertical/=np.linalg.norm(vertical);forward=rp[0,:3,2].copy();forward-=vertical*np.dot(forward,vertical);forward/=np.linalg.norm(forward);right=np.cross(vertical,forward);raw=planar_yaw(rp);static=[];static_rotations=[];epi=[]
    for win in windows:
        start=int(win['start']);tr=win['tracks'];vis=win['visible']
        for local_i in range(0,len(tr)-1,4):
            for gap in [4,8,16]:
                local_j=local_i+gap
                if local_j>=len(tr):continue
                good=vis[local_i]&vis[local_j];p=tr[local_i,good];q=tr[local_j,good]
                if len(p)<12:continue
                rot,inside=fit_rotation(bearings(p,k),bearings(q,k),np.random.default_rng(start*100+local_i+gap),1.5/k[0,0])
                if inside.sum()<12 or inside.mean()<.25:continue
                i=start+local_i;j=start+local_j;f=rp[i,:3,:3]@rot.T@np.array([0.,0.,1.]);yj=np.arctan2(f@right,f@forward);yi=np.arctan2(rp[i,:3,2]@right,rp[i,:3,2]@forward);dy=(yj-yi+np.pi)%(2*np.pi)-np.pi
                static.append([i,j,dy,min(1,int(inside.sum())/80)])
                static_rotations.append(rot)
    if static:
        # Keep the original batched float arithmetic, including R3 float32 yaw.
        si=np.array([v[0] for v in static],dtype=int);rot=np.array(static_rotations)
        f=np.einsum('nij,njk,k->ni',rp[si,:3,:3],rot.transpose(0,2,1),[0.,0.,1.])
        yj=np.arctan2(f@right,f@forward);yi=np.arctan2(rp[si,:3,2]@right,rp[si,:3,2]@forward)
        dy=(yj-yi+np.pi)%(2*np.pi)-np.pi
        for row,yaw_delta in zip(static,dy):row[2]=yaw_delta
    for path in sorted(vggt_dir.glob('*.npz')):
        z=np.load(path);start=int(z['start']);length=len(z['c2w'])
        for local_i in [0,4,8]:
            for gap in [4,8,15]:
                local_j=local_i+gap
                if local_j>=length:continue
                i=start+local_i;j=start+local_j
                if j>=len(rp):continue
                possible=[w for w in windows if int(w['start'])<=i and j<int(w['start'])+len(w['tracks'])]
                if not possible:continue
                win=max(possible,key=lambda w:int((w['visible'][i-int(w['start'])]&w['visible'][j-int(w['start'])]).sum()))
                st=int(win['start']);good=win['visible'][i-st]&win['visible'][j-st];p=win['tracks'][i-st,good];q=win['tracks'][j-st,good]
                if len(p)<24:continue
                order=np.lexsort((p[:,1],p[:,0]));order=order[np.linspace(0,len(order)-1,min(128,len(order))).astype(int)];p=p[order];q=q[order]
                a=bearings(p,k);b=bearings(q,k);rr=rp[j,:3,:3].T@rp[i,:3,:3];_,before=score_rotation(a,b,rr);axis=rp[j,:3,:3].T@vertical;grid=np.deg2rad(np.linspace(-3,3,25));fits=[];vals=[]
                for delta in grid:
                    f,v=score_rotation(a,b,Rotation.from_rotvec(-delta*axis).as_matrix()@rr);fits.append(f);vals.append(v)
                best=int(np.argmin(fits));width,height=meta['width'],meta['height'];spread=np.ptp(p,axis=0)/[width,height];cells=np.unique(np.floor(p/[width,height]*[8,4]).astype(int),axis=0)
                support=min(1,len(p)/80)*min(1,len(cells)/8)*min(1,spread[0]/.4);contrast=(float(np.quantile(fits,.75))-min(fits))/(min(fits)+.0005);gain=max(0.,(before-vals[best])/(before+.0005));trust=support*min(1,contrast)*min(1,gain/.25)
                if best in [0,len(grid)-1]:trust=0.
                epi.append([i,j,raw[j]-raw[i]+grid[best],trust])
    static=np.array(static).reshape(-1,4);epi=np.array(epi).reshape(-1,4);dest.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(dest,static=static,epipolar=epi,version=2)
    save_json(dest.with_suffix('.json'),dict(seconds=time.perf_counter()-started,static_edges=len(static),epipolar_edges=len(epi),split=split))
    print('FEATURES',split,key,len(static),len(epi),round(time.perf_counter()-started,1),flush=True)

def predict(key,split):
    case,*_=provider(key,split);t=case['t'];r=case['headings']['r3'];z=np.load(OUT/'transfer_features'/split/f'{key}.npz');e=np.concatenate([z['static'],z['epipolar']]);i=e[:,0].astype(int);j=e[:,1].astype(int);target=e[:,2];w=e[:,3];b=integrated_basis(t,r,3);a=b[j]-b[i];rhs=target-(r[j]-r[i]);weight=np.sqrt(w/np.maximum(t[j]-t[i],1));reg=10*np.sqrt(len(w)*.1)*np.maximum(1,np.arange(3)**2)
    sol=least_squares(lambda x:np.r_[(a@x-rhs)*weight,x*reg],np.zeros(3),bounds=(-np.deg2rad(.2),np.deg2rad(.2)),loss='soft_l1',f_scale=np.deg2rad(.3),max_nfev=100)
    correction=8*b@sol.x;bounded=np.clip(correction,-np.deg2rad(10),np.deg2rad(10));yaw=.65*r+bounded;speed=case['speed'] if case['speed_correlation']>=.7 else np.ones(len(t));speed=.5+.5*speed
    prediction={**case['pred'],'positions':integrate_velocity(t,yaw,speed).tolist()}
    return prediction,dict(coefficients_deg_s=np.rad2deg(sol.x).tolist(),uncapped_max_correction_deg=float(np.rad2deg(abs(correction).max())),capped_fraction=float(np.mean(abs(correction)>np.deg2rad(10))),maximum_correction_deg=float(np.rad2deg(abs(bounded).max())))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--parity-only',action='store_true');args=ap.parse_args();cv2.setNumThreads(1)
    save_json(REP/'test_plan.json',dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),method='trust_region10',chosen_from='TRAIN only; before TEST inference',scope='No TEST labels; compare input-derived predictions only. Existing non-FLIR9 preserved from official-best v2. No online upload.',preprocessing='Same static RANSAC and pairwise yaw evidence extraction as TRAIN, checked on all TRAIN clips.'))
    parity={}
    for key in TRAIN:
        extract(key,'train');pred,diag=predict(key,'train');old=json.loads((OUT/'predictions/trust_region10'/f'{key}.json').read_text());diff=float(np.max(abs(np.array(pred['positions'])-old['positions'])));assert diff<1e-5,(key,diff);parity[key]=diff
    save_json(REP/'transfer_parity.json',dict(max_coordinate_differences=parity,passed=True))
    if args.parity_only:return
    original=json.loads((ROOT/'submissions/macvi27_vo_domain_v2_gated/submission.json').read_text());submission=dict(original);details={}
    for key in test.KEYS:
        extract(key,'test');pred,diag=predict(key,'test');validate_prediction(pred,key,test.load_timestamps(test.DATA,test.CLIPS[key]));save_json(OUT/'test/predictions'/f'{key}.json',pred)
        # Preserve the same submission encoding as the previously scored file.
        submission[key]=dict(original[key]);submission[key]['positions']=pred['positions'];submission[key]['times']=pred['times']
        if 'frame_indices' in original[key]:submission[key]['frame_indices']=pred['frame_indices']
        delta=np.array(pred['positions'])-np.array(original[key]['positions']);details[key]={**diag,'max_coordinate_change_v2':float(np.max(abs(delta)))}
    assert len(submission)==20
    unchanged=[k for k in submission if submission[k]==original[k]]
    assert set(unchanged)==set(submission)-set(test.KEYS)
    dest=ROOT/'submissions/macvi27_vo_domain_v11_3246';dest.mkdir(parents=True,exist_ok=True);save_json(dest/'submission.json',submission)
    save_json(dest/'provenance.json',dict(method='trust_region10',train_mean=3.246459,official_score=None,unchanged_non_flir=unchanged,changed_flir=details,submission_sha256=sha256(dest/'submission.json'),scope='TRAIN-selected candidate. No official score or online upload.'))
    save_json(REP/'test_comparison.json',dict(clips=details,unchanged_non_flir=unchanged,submission=str((dest/'submission.json').relative_to(ROOT)),sha256=sha256(dest/'submission.json'),official_score=None))
    print('TEST_CANDIDATE',dest/'submission.json',flush=True)

if __name__=='__main__':main()
