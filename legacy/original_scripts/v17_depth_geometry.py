"""RePoseD/PoseLib geometry on cached Pi3X depth and static CoTracker matches.

This is an integration experiment, not reproduction of full MADPose or DGSfM.
All prediction paths are independent of trajectory references.
"""
import sys,argparse,datetime,time
from functools import lru_cache
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'.envs/poselib_v17_latest'))
import poselib
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from v13_common import TRAIN,base_case,save_json,np,json,integrate_velocity
from v5_common import cv2,bearings
from v11_test_candidate import provider
from v10_evidence_graph import epipolar_fit,epipolar_error
from public_dataset import sha256
OUT=ROOT/'runs/domain_v17';REP=ROOT/'reports/domain_v17'
DEPTH_FILTER='image_median'
FRONTEND='cotracker'
SOLVERS=['essential','depth_scale','depth_affine']
CONFIGS=[dict(name='v2',source=None,backend=None,cap=0)]
CONFIGS += [dict(name=f'{k}_{s}_{b}_cap3',source=f'{k}_{s}',backend=b,cap=3) for k in ['nominal','learned'] for s in SOLVERS for b in ['bias','graph']]
CONFIGS += [dict(name='learned_depth_affine_graph_cap10',source='learned_depth_affine',backend='graph',cap=10),
            dict(name='depth_consensus_graph_cap3',source='consensus',backend='graph',cap=3)]


def camera(k):return dict(model='PINHOLE',width=960,height=540,params=[float(k[0,0]),float(k[1,1]),float(k[0,2]),float(k[1,2])])


def solve(method,p,q,d1,d2,k,seed):
    opt=dict(ransac=dict(min_iterations=100,max_iterations=1000,success_prob=.999,seed=int(seed)),bundle=dict(max_iterations=50,loss_type='TRUNCATED_CAUCHY'))
    if method=='essential':
        opt['max_error']=1.5;pose,stat=poselib.estimate_relative_pose(p,q,camera(k),camera(k),opt)
        return pose,stat,None
    opt.update(max_errors=[3.,1.5],estimate_shift=method=='depth_affine',weight_sampson=1.)
    geom,stat=poselib.estimate_monodepth_relative_pose(p,q,d1,d2,camera(k),camera(k),opt)
    return geom.pose,stat,geom


def prepare_pairs(key,split):
    dest=OUT/'pairs'/split/f'{key}.npz'
    if dest.exists():return dict(np.load(dest))
    case,trackdir,_,rp,_,meta=provider(key,split)
    root=ROOT/('runs/domain_v3/pi3x_rgb' if split=='train' else 'runs/domain_v13/pi3x_test')
    files=sorted((root/'windows'/key).glob('*.npz'));starts=np.array([int(p.stem) for p in files])
    @lru_cache(maxsize=8)
    def window(idx):return dict(np.load(files[idx]))
    def depth(index,xy):
        candidates=np.flatnonzero((starts<=index)&(starts+12>index));assert len(candidates)
        ix=min(candidates,key=lambda v:abs(index-(starts[v]+5.5)));z=window(int(ix));loc=index-int(z['start']);dep=z['depth'][loc];conf=z['conf'][loc]
        # Pi3 input: resized 448 wide, 14-aligned height; maps store [::4,::4].
        ph=round(meta['height']*448/meta['width']/14)*14
        uv=((xy+.5)*[448/meta['width'],ph/meta['height']]-.5)/4
        d=cv2.remap(dep.astype(np.float32),uv[:,0].astype(np.float32)[None],uv[:,1].astype(np.float32)[None],cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=float('nan'))[0]
        c=cv2.remap(conf.astype(np.float32),uv[:,0].astype(np.float32)[None],uv[:,1].astype(np.float32)[None],cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=0)[0]
        return d,c,np.median(conf)
    segroot=ROOT/('runs/domain_v5/segmentation' if split=='train' else 'runs/domain_v6_test/segmentation')
    static=json.loads((ROOT/'runs/domain_v5/segmentation/model.json').read_text())['static_ids']
    @lru_cache(maxsize=12)
    def mask(i):
        p=segroot/key/f'{i:04d}.png'
        if not p.exists():p=ROOT/'runs/domain_v16'/split/'extra_masks'/key/f'{i:04d}.png'
        lab=cv2.imread(str(p),0)
        if lab is None:raise FileNotFoundError(p)
        return cv2.erode(np.isin(lab,static).astype(np.uint8),np.ones((3,3),np.uint8))
    def valid(i,p):
        m=mask(i);xy=np.floor(p/[meta['width'],meta['height']]*[m.shape[1],m.shape[0]]).astype(int)
        ok=(xy[:,0]>=0)&(xy[:,0]<m.shape[1])&(xy[:,1]>=0)&(xy[:,1]<m.shape[0]);ids=np.flatnonzero(ok);ok[ids]&=m[xy[ids,1],xy[ids,0]]>0;return ok
    records=[];chunks=[];offset=[0];seen=set()
    def raw_pairs():
        if FRONTEND=='roma':
            for path in sorted((OUT/'matches'/split/key).glob('*.npz')):
                z=np.load(path);yield int(z['i']),int(z['j']),z['p'],z['q']
        else:
            for path in sorted(trackdir.glob('*.npz')):
                z=np.load(path);st=int(z['start']);tr=z['tracks'];vis=z['visible']
                for a,b in [(0,8),(8,16),(16,24)]:
                    if b>=len(tr):continue
                    ok=vis[a]&vis[b];yield st+a,st+b,tr[a,ok],tr[b,ok]
    for i,j,p,q in raw_pairs():
            if (i,j) in seen:continue
            seen.add((i,j));ok=valid(i,p)&valid(j,q);p=p[ok];q=q[ok]
            if len(p)<30:continue
            d1,c1,th1=depth(i,p);d2,c2,th2=depth(j,q)
            ok=np.isfinite(d1)&np.isfinite(d2)&(d1>1e-6)&(d2>1e-6)
            if DEPTH_FILTER=='image_median':ok&=(c1>=th1)&(c2>=th2)
            p,q,d1,d2=p[ok],q[ok],d1[ok],d2[ok]
            if len(p)<30:continue
            order=np.lexsort((p[:,1],p[:,0]));order=order[np.linspace(0,len(order)-1,min(240,len(order))).astype(int)]
            scale1=float(np.median(d1));scale2=float(np.median(d2))
            v=np.c_[p[order],q[order],d1[order]/scale1,d2[order]/scale2]
            chunks.append(v);offset.append(offset[-1]+len(v));records.append([i,j,scale1,scale2])
    data=dict(pairs=np.array(records).reshape(-1,4),points=np.concatenate(chunks) if chunks else np.empty((0,6)),offsets=np.array(offset))
    dest.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(dest,**data);return data


def extract(key,split):
    dest=OUT/'edges'/split/f'{key}.npz'
    if dest.exists():return dict(np.load(dest))
    tic=time.perf_counter();data=prepare_pairs(key,split);case,_,_,rp,_,_=provider(key,split)
    z=np.load(ROOT/f'runs/domain_v4/calibrated_rotation_{split}'/f'{key}.npz');ks={n:z[n+'_k'] for n in ['nominal','learned']}
    up=np.median(rp[:,:3,1],axis=0);up/=np.linalg.norm(up);f=rp[0,:3,2].copy();f-=up*(f@up);f/=np.linalg.norm(f);right=np.cross(up,f)
    rows={f'{k}_{m}':[] for k in ks for m in SOLVERS};details=[]
    for idx,pair in enumerate(data['pairs']):
        i,j=pair[:2].astype(int);points=data['points'][data['offsets'][idx]:data['offsets'][idx+1]]
        p,q,d1,d2=points[:,:2],points[:,2:4],points[:,4],points[:,5];held=np.arange(len(p))%3==0;fit=~held
        rr=rp[j,:3,:3].T@rp[i,:3,:3];local={}
        for kn,k in ks.items():
            a,b=bearings(p,k),bearings(q,k);base_t=epipolar_fit(a[fit],b[fit],rr)
            before=float(np.median(epipolar_error(a[held],b[held],rr,base_t))*k[0,0])
            for method in SOLVERS:
                pose,stat,geom=solve(method,p[fit],q[fit],d1[fit],d2[fit],k,i*1000+j)
                r=np.asarray(pose.R);trans=np.asarray(pose.t);count=int(stat['num_inliers'])
                sane=np.isfinite(r).all() and np.isfinite(trans).all() and np.linalg.norm(trans)>1e-8
                angle=180.;after=1e6;dy=0.;fraction=count/fit.sum();cheirality=0.;depth_error=None
                if sane:
                    angle=float(np.rad2deg(Rotation.from_matrix(r@rr.T).magnitude()))
                    after=float(np.median(epipolar_error(a[held],b[held],r,trans/np.linalg.norm(trans)))*k[0,0])
                    target=rp[i,:3,:3]@r.T@np.array([0.,0.,1.]);yi=np.arctan2(rp[i,:3,2]@right,rp[i,:3,2]@f)
                    dy=np.arctan2(target@right,target@f)-yi;dy=float(np.arctan2(np.sin(dy),np.cos(dy)))
                    pn=(np.c_[p[held],np.ones(held.sum())]@np.linalg.inv(k).T);qn=(np.c_[q[held],np.ones(held.sum())]@np.linalg.inv(k).T)
                    xyz=cv2.triangulatePoints(np.c_[np.eye(3),np.zeros(3)],np.c_[r,trans],pn[:,:2].T,qn[:,:2].T)
                    xyz=xyz[:3]/np.where(abs(xyz[3])>1e-12,xyz[3],1e-12);other=r@xyz+trans[:,None]
                    cheirality=float(np.mean((xyz[2]>0)&(other[2]>0)))
                    if geom is not None:
                        x=pn*(d1[held]+geom.shift1)[:,None];z2=(x@r.T+trans)[:,2];expected=geom.scale*(d2[held]+geom.shift2)
                        positive=(z2>1e-8)&(expected>1e-8)
                        depth_error=float(np.median(abs(np.log(z2[positive]/expected[positive])))) if positive.any() else 100.
                accepted=bool(sane and count>=12 and fraction>=.4 and after<=1.5 and angle<=10 and cheirality>=.6)
                if geom is not None:accepted &= bool(depth_error is not None and depth_error<=.3)
                # Do not require relative gain for the raw geometry ablation.
                weight=min(1,count/60)*fraction/(1+after) if accepted else 0.
                rows[f'{kn}_{method}'].append([i,j,dy,weight])
                local[f'{kn}_{method}']=dict(inliers=count,fraction=float(fraction),before_px=before,after_px=after,rotation_delta_deg=angle,cheirality=cheirality,depth_log_error=depth_error,accepted=accepted,r=r.tolist() if sane else None)
        details.append(dict(i=int(i),j=int(j),points=len(p),methods=local))
    arr={k:np.array(v).reshape(-1,4) for k,v in rows.items()}
    arr['consensus']=arr['learned_depth_affine'].copy()
    for n,record in enumerate(details):
        e=record['methods']['learned_essential'];d=record['methods']['learned_depth_affine']
        agree=float(np.rad2deg(Rotation.from_matrix(np.asarray(e['r'])@np.asarray(d['r']).T).magnitude())) if e['r'] is not None and d['r'] is not None else 180.
        if not (e['accepted'] and d['accepted'] and agree<=1 and d['after_px']<.9*d['before_px']):arr['consensus'][n,3]=0.
        record['consensus_rotation_disagreement_deg']=agree
    dest.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(dest,**arr)
    save_json(dest.with_suffix('.json'),dict(key=key,split=split,seconds=time.perf_counter()-tic,pairs=len(details),accepted={k:int((v[:,3]>.01).sum()) for k,v in arr.items()},details=details))
    print('GEOMETRY',split,key,len(details),{k:int((v[:,3]>.01).sum()) for k,v in arr.items()},flush=True)
    return arr


def correction(t,y,edges,backend,cap):
    e=edges[edges[:,3]>.01]
    if len(e)<8:return np.zeros(len(t))
    i=e[:,0].astype(int);j=e[:,1].astype(int);rhs=e[:,2]-(y[j]-y[i]);w=np.sqrt(e[:,3]/np.maximum(t[j]-t[i],1))
    if backend=='bias':
        b=(t-t[0])[:,None];reg=np.array([[10*np.sqrt(len(e)*.1)]]);bounds=(-np.deg2rad(.2),np.deg2rad(.2))
    else:
        knots=np.linspace(t[0],t[-1],int(np.ceil((t[-1]-t[0])/30))+1);eye=np.eye(len(knots));b=np.array([np.interp(t,knots,v) for v in eye[1:]]).T
        reg=np.r_[.1*np.eye(len(knots)-1),np.diff(eye,2,axis=0)[:,1:]];bounds=(-np.deg2rad(cap),np.deg2rad(cap))
    a=b[j]-b[i]
    def loss(z):
        out=np.vstack([z,np.ones(len(z)),np.zeros(len(z))]);zz=z[:len(e)]
        out[:,:len(e)]=[2*(np.sqrt(1+zz)-1),1/np.sqrt(1+zz),-.5/(1+zz)**1.5];return out
    sol=least_squares(lambda x:np.r_[(a@x-rhs)*w,reg@x],np.zeros(b.shape[1]),bounds=bounds,loss=loss,f_scale=np.deg2rad(.3),max_nfev=100)
    return np.clip(b@sol.x,-np.deg2rad(cap),np.deg2rad(cap))


def synthetic():
    rng=np.random.default_rng(19);x=rng.uniform([-3,-2,5],[3,2,12],(100,3));r=Rotation.from_rotvec([.02,.08,-.01]).as_matrix();tt=np.array([.4,.1,.2]);xx=x@r.T+tt
    k=np.array([[800.,0,480],[0,800,270],[0,0,1.]])
    p=(x@k.T);p=p[:,:2]/p[:,2,None];q=xx@k.T;q=q[:,:2]/q[:,2,None]
    rows=[]
    for method in SOLVERS:
        shift=.4 if method=='depth_affine' else 0.
        pose,stat,geom=solve(method,p,q,x[:,2]-shift,xx[:,2]/1.7+shift,k,0)
        error=float(np.rad2deg(Rotation.from_matrix(pose.R@r.T).magnitude()))
        assert error<.02,(method,error)
        direction=np.asarray(pose.t)/np.linalg.norm(pose.t);assert direction@tt/np.linalg.norm(tt)>.999
        rows.append(dict(method=method,rotation_error_deg=error,inliers=int(stat['num_inliers'])))
    save_json(REP/'synthetic.json',dict(passed=True,cases=rows,poselib_version=poselib.__version__))


def main():
    global OUT,REP,DEPTH_FILTER
    ap=argparse.ArgumentParser();ap.add_argument('--split',choices=['train','test'],default='train');ap.add_argument('--keys');ap.add_argument('--synthetic-only',action='store_true');ap.add_argument('--depth-filter',choices=['image_median','geometry_only'],default='image_median');args=ap.parse_args();cv2.setNumThreads(1)
    DEPTH_FILTER=args.depth_filter
    if DEPTH_FILTER=='geometry_only':OUT=ROOT/'runs/domain_v17_geometry_only';REP=ROOT/'reports/domain_v17_geometry_only'
    if not (REP/'plan.json').exists():
        save_json(REP/'plan.json',dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),configs=CONFIGS,
          hypothesis='Depth-affine pose estimation can separate translation from rotation where pure-rotation fits failed. Compare essential, scale-only depth, scale+shift depth on identical image matches.',depth_filter=DEPTH_FILTER,
          adaptive_followup=DEPTH_FILTER=='geometry_only',followup_reason='Image-median depth confidence discarded all pairs on 3/8 TRAIN clips. Ablate this filter uniformly on all clips; retain static masks and held-out geometry checks. Original results remain separately stored.' if DEPTH_FILTER=='geometry_only' else None,
          stage_design=['Static-mask CoTracker matches and cached Pi3X depth','PoseLib RePoseD/essential solvers with held-out image points','Bounded R3-relative yaw correction; v2 speed unchanged'],
          fixed_gates='>=12 fit inliers; fraction>=.4; heldout Sampson<=1.5px; rotation change<=10deg; positive-depth fraction>=.6; depth solvers heldout log-depth inconsistency<=.3. Consensus additionally essential agreement<=1deg and >=10% heldout gain.',
          selection_rule='>=5/8 improved, max regression<=.5pp, mean delta excluding largest gain<0; no official score claims.',
          scope='Repeated TRAIN8 development. RePoseD integration, not full MADPose, DGSfM or TALO reproduction. No test labels or manual clip rules.',
          frozen={str(p.relative_to(ROOT)):sha256(p) for p in (ROOT/'submissions').glob('*/submission.json')}))
    synthetic()
    if args.synthetic_only:return
    if args.keys:keys=args.keys.split(',')
    elif args.split=='train':keys=TRAIN
    else:
        from v6_test_common import KEYS
        keys=KEYS
    for key in keys:
        case,y,s=base_case(key,args.split);arr=extract(key,args.split);signals={}
        for cfg in CONFIGS:
            delta=np.zeros(len(y)) if cfg['source'] is None else correction(case['t'],y,arr[cfg['source']],cfg['backend'],cfg['cap'])
            signals[cfg['name']]=y+delta
            save_json(OUT/args.split/'predictions'/cfg['name']/f'{key}.json',{**case['pred'],'positions':integrate_velocity(case['t'],y+delta,s).tolist()})
        path=OUT/args.split/'signals'/f'{key}.npz';path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,times=case['t'],speed=s,**signals)
        print('INFER',args.split,key,len(CONFIGS),flush=True)
    for p,h in json.loads((REP/'plan.json').read_text())['frozen'].items():assert sha256(ROOT/p)==h


if __name__=='__main__':main()
