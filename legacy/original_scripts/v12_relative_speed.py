"""Static three-view relative-speed graph. Image-only inference; no references.

Shared structure cancels monocular scale. Geometry estimates the ratio of two
adjacent baseline lengths. A smooth log-speed graph accumulates these ratios.
"""
import argparse, datetime, time
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from v12_common import *
from v11_test_candidate import provider, predict
from v11_threeview import threeview_score

CONFIGS=[dict(name=f'ratio_{rot}_reg{reg:g}_blend{blend:g}',rotation=rot,reg=reg,blend=blend)
         for rot in ['raw','v11'] for reg in [.1,1.,10.] for blend in [.25,.5,1.]]

def synthetic_check():
    rng=np.random.default_rng(12);p=rng.uniform([-8,-3,15],[8,3,50],(128,3));rows=[]
    for ratio in [.5,1.,2.]:
        c=np.array([[0.,0,0],[.2,0,1],[.2*(1+ratio),0,1+ratio]])
        d=p[:,None]-c[None];d/=np.linalg.norm(d,axis=2,keepdims=True)
        _,err,positive,fit,eig=threeview_score(d);cc=fit.reshape(2,3)
        recovered=np.linalg.norm(cc[1]-cc[0])/np.linalg.norm(cc[0])
        assert abs(recovered-ratio)<1e-5 and positive>.99 and err<1e-6
        rows.append(dict(expected=ratio,recovered=float(recovered),heldout_error=err))
    save_json(REP/'relative_speed_synthetic.json',dict(passed=True,cases=rows))

def extract(key,split='train'):
    dest=OUT/'relative_speed'/split/f'{key}.npz'
    if dest.exists():return
    tic=time.perf_counter();case,tracks,_,rp,k,_=provider(key,split);t=case['t'];raw=planar_yaw(rp)
    if split=='train':_,yaw,_=baseline(key)
    else:
        # Reuse frozen V11 coefficients; positions are not used here.
        _,diag=predict(key,split)
        from v11_temporal import integrated_basis
        b=integrated_basis(t,case['headings']['r3'],3)
        yaw=.65*case['headings']['r3']+np.clip(8*b@np.deg2rad(diag['coefficients_deg_s']),-np.deg2rad(10),np.deg2rad(10))
    vertical=np.median(rp[:,:3,1],axis=0);vertical/=np.linalg.norm(vertical)
    all_rot={'raw':rp[:,:3,:3], 'v11':Rotation.from_rotvec((yaw-raw)[:,None]*vertical).as_matrix()@rp[:,:3,:3]}
    rows={rot:[] for rot in all_rot}
    for path in sorted(tracks.glob('*.npz')):
        z=np.load(path);st=int(z['start'])
        for gap in [4,8,12]:
            for offset in [0,4]:
                loc=offset+np.array([0,gap,2*gap]);ids=st+loc
                if loc[-1]>=len(z['tracks']):continue
                valid=z['visible'][loc].all(0);pts=z['tracks'][loc][:,valid]
                if pts.shape[1]<32:continue
                order=np.lexsort((pts[0,:,1],pts[0,:,0]));order=order[np.linspace(0,len(order)-1,min(128,len(order))).astype(int)];pts=pts[:,order]
                camera=np.array([bearings(p,k) for p in pts])
                for name,rot in all_rot.items():
                    world=np.einsum('vij,vnj->nvi',rot[ids],camera)
                    fit,val,positive,cc,eig=threeview_score(world);c1,c2=cc.reshape(2,3)
                    v1=np.linalg.norm(c1)/(t[ids[1]]-t[ids[0]]);v2=np.linalg.norm(c2-c1)/(t[ids[2]]-t[ids[1]])
                    ratio=v2/max(v1,1e-10);parallax=float(np.median(np.linalg.norm(world[:,2]-world[:,0],axis=1)))
                    gap_ratio=max(float(eig[0]),0)/max(float(eig[1]),1e-12)
                    quality=min(1,len(pts[0])/80)*np.clip((positive-.6)/.3,0,1)*min(1,parallax/.005)*np.exp(-val/.002)*np.clip(1-gap_ratio/.2,0,1)
                    if ratio<.25 or ratio>4 or val>.005 or positive<.75:quality=0.
                    rows[name].append([*(t[ids]),np.log(max(ratio,1e-8)),quality,fit,val,positive,parallax,gap_ratio])
    dest.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(dest,**{k:np.array(v).reshape(-1,10) for k,v in rows.items()})
    print('RATIO_FEATURES',split,key,{k:sum(r[4]>.05 for r in v) for k,v in rows.items()},round(time.perf_counter()-tic,1),flush=True)

def infer(t,prior,rows,reg,blend):
    knots=np.linspace(t[0],t[-1],int(np.ceil((t[-1]-t[0])/10))+1);eye=np.eye(len(knots))
    def basis(x):return np.array([np.interp(x,knots,e) for e in eye]).T
    m1=(rows[:,0]+rows[:,1])/2;m2=(rows[:,1]+rows[:,2])/2
    a=basis(m2)-basis(m1);w=np.sqrt(rows[:,4]);rhs=rows[:,3]
    p=np.log(np.maximum(prior,1e-3));pk=np.interp(knots,t,p);pk-=pk.mean()
    d2=np.diff(eye,2,axis=0)
    def residual(x):return np.r_[(a@x-rhs)*w,reg*.1*(x-pk),reg*d2@x,10*x.mean()]
    sol=least_squares(residual,pk,bounds=(-np.log(4),np.log(4)),loss='soft_l1',f_scale=.15,max_nfev=80)
    delta=np.interp(t,knots,sol.x)-p;delta-=delta.mean();out=prior*np.exp(blend*np.clip(delta,-np.log(2),np.log(2)))
    return out,dict(valid_edges=int(np.sum(rows[:,4]>.05)),speed_min=float(out.min()),speed_max=float(out.max()),relative_change_rms=float(np.sqrt(np.mean((out/prior-1)**2))))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--extract-only',action='store_true');args=ap.parse_args();cv2.setNumThreads(1)
    save_json(REP/'relative_speed_plan.json',dict(configs=CONFIGS,created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),hypothesis='Shared static structure across three views estimates adjacent speed ratios independently of absolute scale. Robust log-speed graph plus smoothness and prior prevents accumulation of weak geometry.',scope='Image-only prediction. Repeated TRAIN8 development. No reference motion in extraction or inference.'))
    synthetic_check()
    for key in TRAIN:
        extract(key)
        if args.extract_only:continue
        case,y,s=baseline(key);z=np.load(OUT/'relative_speed/train'/f'{key}.npz');diag={}
        for cfg in CONFIGS:
            speed,d=infer(case['t'],s,z[cfg['rotation']],cfg['reg'],cfg['blend']);emit(key,cfg['name'],y,speed,case);diag[cfg['name']]=d
        save_json(OUT/'relative_speed/train'/f'{key}.json',diag)

if __name__=='__main__':main()
