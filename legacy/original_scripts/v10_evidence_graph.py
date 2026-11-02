"""Static image evidence + multi-interval yaw graph. Never loads trajectory labels.

Not a reproduction of full F2SLAM/VGGT-SLAM: this is our restricted yaw backend.
Translation is a nuisance variable in epipolar scoring, not a speed measurement.
"""
import argparse, datetime, time
from scipy.spatial.transform import Rotation
from scipy.sparse import coo_matrix, vstack
from scipy.sparse.linalg import lsqr
from v5_common import *

OUT=ROOT/'runs/domain_v10'; REP=ROOT/'reports/domain_v10'
CONFIGS=[
    dict(name='v2',kind='baseline'),
    dict(name='graph_uniform',kind='uniform'),
    dict(name='graph_image_weight',kind='evidence'),
    dict(name='graph_image_observable',kind='observable'),
    dict(name='graph_image_refine',kind='refine'),
    dict(name='graph_image_refine_strong',kind='refine',strength=4.),
    dict(name='graph_image_refine_conservative',kind='refine',strength=.25),
    dict(name='graph_image_refine_no_robust',kind='refine',robust=False),
]

def epipolar_fit(a,b,r):
    q=a@r.T; n=np.cross(q,b); w=np.ones(len(n))
    for _ in range(3):
        _,v=np.linalg.eigh((n*w[:,None]).T@n); t=v[:,0]
        den=np.sqrt(np.sum(np.cross(t,q)**2,axis=1)+np.sum(np.cross(b,t)**2,axis=1))
        e=np.abs(n@t)/np.maximum(den,1e-8)
        w=1/np.maximum(1,e/.0015)
    return t

def epipolar_error(a,b,r,t):
    q=a@r.T
    den=np.sqrt(np.sum(np.cross(t,q)**2,axis=1)+np.sum(np.cross(b,t)**2,axis=1))
    return np.abs(np.cross(q,b)@t)/np.maximum(den,1e-8)

def score_rotation(a,b,r):
    # Alternating point IDs: validation tracks never fit translation or yaw.
    trans=epipolar_fit(a[::2],b[::2],r)
    fit=epipolar_error(a[::2],b[::2],r,trans)
    val=epipolar_error(a[1::2],b[1::2],r,trans)
    return float(np.mean(np.minimum(fit,.01))),float(np.mean(np.minimum(val,.01)))

def build_edges(key):
    dest=OUT/'edges'/f'{key}.npz'
    if dest.exists():return
    began=time.perf_counter(); case=get_case(key); n=len(case['t'])
    rp=np.load(ROOT/'runs/domain_probe_v2/r3/diagnostics'/f'{key}.poses.npz')['camera_to_world']
    raw_yaw=planar_yaw(rp)
    k=np.load(ROOT/'runs/domain_v4/calibrated_rotation_train'/f'{key}.npz')['learned_k']
    folder,meta=frames(key); width=meta['width'];height=meta['height']
    windows=[dict(np.load(p)) for p in sorted((RUN/'cotracker_windows'/key).glob('*.npz'))]
    vertical=np.median(rp[:,:3,1],axis=0);vertical/=np.linalg.norm(vertical)
    rows=[]
    for p in sorted((ROOT/'runs/domain_probe_v2/vggt/windows'/key).glob('*.npz')):
        z=np.load(p); start=int(z['start']); poses=z['c2w']; vy=planar_yaw(poses)
        for local_i in [0,4,8]:
            for gap in [4,8,15]:
                local_j=local_i+gap
                if local_j>=len(poses):continue
                i=start+local_i;j=start+local_j
                if j>=n:continue
                possible=[w for w in windows if int(w['start'])<=i and j<int(w['start'])+len(w['tracks'])]
                if not possible:continue
                # One window avoids treating duplicated tracked points as new evidence.
                w=max(possible,key=lambda w:int((w['visible'][i-int(w['start'])]&w['visible'][j-int(w['start'])]).sum()))
                st=int(w['start']);good=w['visible'][i-st]&w['visible'][j-st]
                p0=w['tracks'][i-st,good];p1=w['tracks'][j-st,good]
                if len(p0)<24:continue
                # Deterministic spatial ordering spreads both split halves across image.
                order=np.lexsort((p0[:,1],p0[:,0]));order=order[np.linspace(0,len(order)-1,min(128,len(order))).astype(int)]
                p0=p0[order];p1=p1[order];a=bearings(p0,k);b=bearings(p1,k)
                rr=rp[j,:3,:3].T@rp[i,:3,:3]
                vr=poses[local_j,:3,:3].T@poses[local_i,:3,:3]
                _,er=score_rotation(a,b,rr);_,ev=score_rotation(a,b,vr)
                # Search a physically consistent yaw correction around the R3 pose.
                # Positive world yaw gives negative relative camera rotation.
                axis=rp[j,:3,:3].T@vertical
                grid=np.deg2rad(np.linspace(-3,3,25));fits=[];vals=[]
                for d in grid:
                    r=Rotation.from_rotvec(-d*axis).as_matrix()@rr
                    f,v=score_rotation(a,b,r);fits.append(f);vals.append(v)
                best=int(np.argmin(fits));corr=grid[best]
                spread=np.ptp(p0,axis=0)/[width,height]
                cells=np.unique(np.floor(p0/[width,height]*[8,4]).astype(int),axis=0)
                support=min(1,len(p0)/80)*min(1,len(cells)/8)*min(1,spread[0]/.4)
                # Reject an unresolved yaw search, boundary optimum, or no held-out gain.
                contrast=(float(np.quantile(fits,.75))-min(fits))/(min(fits)+.0005)
                gain=max(0.,(er-vals[best])/(er+.0005))
                trust=support*min(1,contrast)*min(1,gain/.25)
                if best in [0,len(grid)-1]:trust=0.
                rows.append([i,j,raw_yaw[j]-raw_yaw[i],vy[local_j]-vy[local_i],raw_yaw[j]-raw_yaw[i]+corr,er,ev,vals[best],support,trust,len(p0),corr])
    arr=np.array(rows,dtype=float).reshape(-1,12)
    dest.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(dest,edges=arr,times=case['t'])
    save_json(dest.with_suffix('.json'),dict(key=key,seconds=time.perf_counter()-began,edges=len(rows),trusted_refinement_edges=int((arr[:,9]>.1).sum()),mean_support=float(np.mean(arr[:,8])),mean_refinement_deg=float(np.rad2deg(np.mean(arr[:,11]))),columns=['i','j','r3_yaw','vggt_yaw','image_yaw','r3_validation_error','vggt_validation_error','image_validation_error','support','refine_trust','points','yaw_correction']))
    print('EDGES',key,len(rows),'seconds',round(time.perf_counter()-began,1),flush=True)

def graph_yaw(base,edges,c):
    n=len(base); m=len(edges)
    if not m:return base.copy()
    ii=edges[:,0].astype(int);jj=edges[:,1].astype(int);gap=jj-ii
    kind=c['kind'];strength=c.get('strength',1.)
    if kind=='uniform':wv=np.full(m,.5);support=np.ones(m)
    else:
        # Relative image evidence; floor prevents tiny numerical residuals dominating.
        r=edges[:,5]+.0005;v=edges[:,6]+.0005
        wv=r*r/(r*r+v*v)
        support=np.ones(m) if kind=='evidence' else edges[:,8]*np.clip(.003/np.minimum(r,v),0,1)
    target=(1-wv)*edges[:,2]+wv*edges[:,3]
    if kind=='refine':
        trust=edges[:,9];target=(1-trust)*target+trust*edges[:,4]
    rhs=target-(base[jj]-base[ii])
    rows=np.repeat(np.arange(m),2);cols=np.c_[ii,jj].ravel();vals=np.tile([-1.,1.],m)
    a=coo_matrix((vals,(rows,cols)),shape=(m,n)).tocsr()[:,1:]
    # Adjacent R3 prior keeps disconnected/weakly observed parts well-defined.
    adj=coo_matrix((np.tile([-1.,1.],n-1),(np.repeat(np.arange(n-1),2),np.c_[np.arange(n-1),np.arange(1,n)].ravel())),shape=(n-1,n)).tocsr()[:,1:]
    prior_weight=.25
    weights=strength*support/np.maximum(gap,1)
    x=np.zeros(n-1)
    for _ in range(5 if c.get('robust',True) else 1):
        residual=a@x-rhs
        robust=1/np.maximum(1,np.abs(residual)/np.deg2rad(.5)) if c.get('robust',True) else np.ones(m)
        w=np.sqrt(weights*robust)
        mat=vstack([a.multiply(w[:,None]),adj*np.sqrt(prior_weight)],format='csr')
        x=lsqr(mat,np.r_[rhs*w,np.zeros(n-1)],atol=1e-9,btol=1e-9,iter_lim=3000)[0]
    return base+np.r_[0,x]

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--keys');ap.add_argument('--predict-only',action='store_true');args=ap.parse_args()
    keys=args.keys.split(',') if args.keys else TRAIN
    plan=REP/'plan.json'
    if not plan.exists():save_json(plan,dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),configs=CONFIGS,hypothesis='Held-out static track epipolar evidence selects and refines complementary multiview rotations better than uniform fusion.',stages=['static semantic CoTracker correspondences and AnyCalib intrinsics (cached)','R3 continuous / VGGT local-window multi-interval rotations','track-split epipolar likelihood, yaw refinement and observability gate','robust relative-yaw graph and fixed v2 speed integration'],scope='Reused TRAIN8 development; no independent generalization claim. No pose labels in inference. Hyperparameters fixed before this run.',limitation='Restricted yaw graph, not full pose/depth bundle adjustment; tracks within one clip are correlated; no metric speed estimation.'))
    cv2.setNumThreads(1)
    for key in keys:
        if not args.predict_only:build_edges(key)
        case=get_case(key);t=case['t'];base=case['headings']['r3'];speed=case['speed'] if case['speed_correlation']>=.7 else np.ones(len(t))
        edges=np.load(OUT/'edges'/f'{key}.npz')['edges']
        for c in CONFIGS:
            yaw=base.copy() if c['kind']=='baseline' else graph_yaw(base,edges,c)
            if c['kind']!='baseline':yaw=gaussian_filter1d(yaw,2/np.median(np.diff(t)))
            save_json(OUT/'predictions'/c['name']/f'{key}.json',{**case['pred'],'positions':integrate_velocity(t,yaw,speed).tolist()})
            dest=OUT/'signals'/c['name']/f'{key}.npz';dest.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(dest,times=t,yaw=yaw,speed=speed)
        print('GRAPH',key,'complete',flush=True)

if __name__=='__main__':main()
