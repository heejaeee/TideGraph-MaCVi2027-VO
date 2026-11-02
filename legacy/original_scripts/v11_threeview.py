"""Three-view static-ray consistency with nuisance structure/translation fitting.

Rotations supplied by R3 + yaw correction; three camera centers and shared 3D
points are solved jointly, with scale and translation gauge removed. This is a
restricted geometry probe, not full nonlinear bundle adjustment.
"""
import datetime,time
from scipy.spatial.transform import Rotation
from v5_common import *
from v11_temporal import OUT,REP,emit

GAINS=[.5,.65,.8,1.];BIASES=np.deg2rad(np.array([-.05,-.02,0.,.02,.05]))
CONFIGS=[dict(name=f'threeview_{kind}_{policy}',kind=kind,policy=policy) for kind in ['bias','joint'] for policy in ['direct','gated','blend']]

def structure_matrix(d):
    # d: N x 3 views x xyz in a common world frame. C0 is fixed to zero.
    proj=np.eye(3)[None,None]-d[:,:,:,None]*d[:,:,None,:]
    inv=np.linalg.pinv(proj.sum(1),rcond=1e-10)
    xmat=inv@np.concatenate([proj[:,1],proj[:,2]],axis=2)
    l=proj@xmat[:,None]
    l[:,1,:,:3]-=proj[:,1];l[:,2,:,3:]-=proj[:,2]
    return xmat,l.reshape(len(d),9,6)

def rays_error(d,xmat,centers):
    x=np.einsum('nij,j->ni',xmat,centers)
    c=np.vstack([np.zeros(3),centers.reshape(2,3)])
    vectors=x[:,None,:]-c[None];depth=np.sum(vectors*d,axis=-1)
    norm=np.maximum(np.linalg.norm(vectors,axis=2),1e-12)
    cos=np.sum(vectors*d,axis=2)/norm
    return np.arccos(np.clip(cos,-1,1)),depth

def threeview_score(d):
    xmat,l=structure_matrix(d);fit=np.arange(len(d))%2==0;w=np.ones(fit.sum())
    for _ in range(3):
        lf=l[fit];normal=np.einsum('nki,nkj,n->ij',lf,lf,w)
        eig,vec=np.linalg.eigh(normal);c=vec[:,0]
        error,depth=rays_error(d[fit],xmat[fit],c)
        if np.mean(depth>0)<.5:c=-c;error,depth=rays_error(d[fit],xmat[fit],c)
        per=np.mean(error,axis=1);w=1/np.maximum(1,per/.003)
    error,depth=rays_error(d,xmat,c);per=np.mean(error,axis=1)
    clipped=np.minimum(per,.03)
    return float(np.mean(clipped[fit])),float(np.mean(clipped[~fit])),float(np.mean(depth>0)),c,eig

def synthetic_check():
    rng=np.random.default_rng(9);points=rng.uniform([-5,-3,10],[5,3,40],(128,3));centers=np.array([[0.,0,0],[.4,0,1],[.8,.03,2.]])
    d=points[:,None]-centers[None];d/=np.linalg.norm(d,axis=2,keepdims=True)
    correct=threeview_score(d);bad=d.copy();bad[:,2]=bad[:,2]@Rotation.from_euler('y',2,degrees=True).as_matrix().T;wrong=threeview_score(bad)
    assert correct[1]<1e-6 and wrong[1]>correct[1]+1e-4,(correct,wrong)
    expected=centers[1:].ravel();expected/=np.linalg.norm(expected)
    assert np.dot(correct[3],expected)>.9999
    save_json(REP/'threeview_geometry_check.json',dict(correct_residual=correct[1],wrong_yaw_residual=wrong[1],translation_direction_cosine=float(np.dot(correct[3],expected)),passed=True))

def costs(key):
    dest=OUT/'threeview'/f'{key}.npz'
    if dest.exists():return
    began=time.perf_counter();case=get_case(key);t=case['t'];rp=np.load(ROOT/'runs/domain_probe_v2/r3/diagnostics'/f'{key}.poses.npz')['camera_to_world'];y=planar_yaw(rp)
    vertical=np.median(rp[:,:3,1],axis=0);vertical/=np.linalg.norm(vertical)
    k=np.load(ROOT/'runs/domain_v4/calibrated_rotation_train'/f'{key}.npz')['learned_k']
    b0=4*np.deg2rad(json.loads((ROOT/'runs/domain_v10/bias_diagnostics'/f'{key}.json').read_text())['bias_hybrid_bias']['bias_deg_s'])
    records=[];indices=[]
    for path in sorted((RUN/'cotracker_windows'/key).glob('*.npz')):
        z=np.load(path);st=int(z['start']);loc=np.array([0,12,24]);ids=st+loc
        if loc[-1]>=len(z['tracks']):continue
        valid=z['visible'][loc].all(0);pts=z['tracks'][loc][:,valid]
        if pts.shape[1]<32:continue
        order=np.lexsort((pts[0,:,1],pts[0,:,0]));order=order[np.linspace(0,len(order)-1,min(96,len(order))).astype(int)];pts=pts[:,order]
        camera=np.array([bearings(p,k) for p in pts]);grid=[]
        for g in GAINS:
            row=[]
            for b in BIASES:
                delta=(g-1)*(y[ids]-y[ids[0]])+(b0+b)*(t[ids]-t[ids[0]])
                rot=Rotation.from_rotvec(delta[:,None]*vertical).as_matrix()@rp[ids,:3,:3]
                world=np.einsum('vij,vnj->nvi',rot,camera)
                f,v,positive,c,eig=threeview_score(world);row.append([f,v,positive])
            grid.append(row)
        records.append(grid);indices.append(ids)
    dest.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(dest,costs=np.array(records),indices=indices,gains=GAINS,biases=BIASES,base_bias=b0)
    print('THREEVIEW_COST',key,'triplets',len(records),'seconds',round(time.perf_counter()-began,1),flush=True)

def main():
    save_json(REP/'threeview_plan.json',dict(configs=CONFIGS,gain_grid=GAINS,additional_bias_grid_deg_s=np.rad2deg(BIASES).tolist(),created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),hypothesis='Shared structure and camera translations across 3 views may be less ambiguous than independent pairwise epipolar residuals.',method='Linear structure elimination, robust camera-center eigensolve, held-out track angular reprojection. Corrections chosen by image fit residual; held-out points determine gate.',scope='No trajectory labels. Adaptive repeated TRAIN development. Structure unknown; translation scale not estimated.'))
    synthetic_check()
    for key in TRAIN:
        costs(key);z=np.load(OUT/'threeview'/f'{key}.npz');arr=z['costs'];case=get_case(key);t=case['t'];r=case['headings']['r3'];speed=case['speed'] if case['speed_correlation']>=.7 else np.ones(len(t));speed=.5+.5*speed
        # Normalize windows by their baseline fit residual, preserving equal clip regions.
        denom=np.maximum(arr[:,1,2,0],.0005);f=np.mean(arr[:,:,:,0]/denom[:,None,None],axis=0);v=np.mean(arr[:,:,:,1]/denom[:,None,None],axis=0)
        positive=arr[:,:,:,2].mean(0);base_yaw=.65*r+float(z['base_bias'])*(t-t[0]);diag={}
        for cfg in CONFIGS:
            if cfg['kind']=='bias':gi=1;bi=int(np.argmin(f[gi]))
            else:gi,bi=np.unravel_index(np.argmin(f),f.shape)
            gain=GAINS[gi];bias=float(z['base_bias'])+BIASES[bi];selected=gain*r+bias*(t-t[0])
            accepted=bool(v[gi,bi]<v[1,2]*.98 and positive[gi,bi]>.7 and bi not in [0,len(BIASES)-1])
            if cfg['kind']=='joint':accepted=accepted and gi not in [0,len(GAINS)-1]
            if cfg['policy']=='gated':yaw=selected if accepted else base_yaw
            elif cfg['policy']=='blend':yaw=.5*(selected+base_yaw)
            else:yaw=selected
            emit(key,cfg['name'],yaw,speed,case);diag[cfg['name']]=dict(gain=gain,bias_deg_s=float(np.rad2deg(bias)),accepted=accepted,fit_ratio=float(f[gi,bi]/f[1,2]),validation_ratio=float(v[gi,bi]/v[1,2]),positive_depth_fraction=float(positive[gi,bi]))
        save_json(OUT/'threeview'/f'{key}.json',diag);print('THREEVIEW_PREDICT',key,flush=True)

if __name__=='__main__':main()
