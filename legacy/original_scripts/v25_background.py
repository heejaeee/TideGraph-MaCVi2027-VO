"""Static-background geometry around the fixed raw3 uncapped submission.

Inference uses only image correspondences, depth, poses and calibration. Reference
trajectories are blocked during inference and loaded by a separate evaluation run.
"""
import argparse, datetime, sys, time
import v17_depth_geometry as g
import v18_long_graph as lg
from v13_common import ROOT, TRAIN, CLIPS, np, json, save_json, integrate_velocity
from public_dataset import sha256

OUT=ROOT/'runs/domain_v25_background'
REP=ROOT/'reports/domain_v25_background'
METHODS=['baseline','depth_graph3','depth_graph6','depth_graph20','essential_graph6',
         'consensus_graph6','cycle_graph6','homography_graph6','focal_depth_graph6',
         'focal_essential_graph6','arbitrate_half','arbitrate_full','consensus_arbitrate_half']

def freeze():
    settings=dict(methods=METHODS,baseline='V24 raw3 uncapped; user screenshot #18940 13.19%, server JSON not yet verified.',
        speed='Exactly V24 speed, unchanged across all candidates.',
        graph='Recenter V17/V18 static RoMa relative rotations on raw3 uncapped. Existing cached edges inherit R3 quality gates. 30s knots; inherited robust graph and zero-correction prior. Local midpoint support envelope 30s, >=8 edges required.',
        homography='RANSAC fit on 2/3 static points, nearest rotation of K^-1 H K; heldout rotation-only reprojection median <=1.5px, >=50% <=3px, singular-value ratio <=1.03, >=4 image cells, xspan>=0.15. No planar-translation homography treated as rotation without audit.',
        focal='PoseLib shared focal on aspect-normalized static points. Fit even pair indices, audit odd pairs. Pair candidates require >=40% inliers, heldout Sampson<=1.5px, focal ratio 0.7..1.4. Median candidate, >=12 candidates and >=50% time bins, log MAD<=0.1. Audit essential-pose heldout pixel residual must improve >=5% and >=0.02px on >=12 pairs. Otherwise exact learned K fallback.',
        arbitrate='At every time, static depth edges within 45s score each of three family yaw increments using weighted clipped absolute angular residual. >=8 edges; softmax temperature1deg. Blend measured weighted mean with baseline by 0.5 or1. Consensus variant uses essential/depth agreement only. No reference-dependent routing.',
        selection='TRAIN8 repeated development only. Prefer >=5 wins, worst regression<=0.5pp, mean improvement after dropping best gain. Report all fixed variants including failures and LOCO method-selection audit; not an independent holdout.',
        scope='No official score prediction, no online submission. New combined pipeline ablation, not a full reproduction of another paper.')
    dest=REP/'plan.json'
    if dest.exists():assert json.loads(dest.read_text())['settings']==settings;return
    deps=[ROOT/'scripts/v25_background.py',ROOT/'scripts/v17_depth_geometry.py',ROOT/'scripts/v18_long_graph.py']
    for key in TRAIN:
        deps += [ROOT/f'runs/domain_v24_submission/train/signals/{key}.npz']
        for run in ['domain_v17_roma','domain_v18']:
            deps += [ROOT/f'runs/{run}/{kind}/train/{key}.{ext}' for kind,ext in [('edges','npz'),('edges','json'),('pairs','npz')]]
    save_json(dest,dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),settings=settings,
        dependencies={str(p.relative_to(ROOT)):sha256(p) for p in deps},
        preserved={str(p.relative_to(ROOT)):sha256(p) for p in (ROOT/'submissions').glob('*/submission.json')}))

def guard():
    import public_dataset, metrics
    blocked=[public_dataset.load_reference,metrics.score_trajectory]
    def forbidden(*a,**kw):raise RuntimeError('Ground truth access during V25 inference')
    for module in list(sys.modules.values()):
        if module is None:continue
        for name,value in list(vars(module).items()):
            if any(value is fn for fn in blocked):setattr(module,name,forbidden)

def wrap(a):return np.arctan2(np.sin(a),np.cos(a))

def pair_data(key,split):
    rows=[]
    for run in ['domain_v17_roma','domain_v18']:
        z=np.load(ROOT/f'runs/{run}/pairs/{split}/{key}.npz')
        for n,pair in enumerate(z['pairs']):
            pts=z['points'][z['offsets'][n]:z['offsets'][n+1]]
            rows.append((int(pair[0]),int(pair[1]),pts))
    return rows

def shared_focal(p,q,k,seed):
    f=np.sqrt(k[0,0]*k[1,1]);aspect=np.array([f/k[0,0],f/k[1,1]])
    pp=(p-k[:2,2])*aspect;qq=(q-k[:2,2])*aspect
    pair,stat=g.poselib.estimate_shared_focal_relative_pose(pp,qq,np.zeros(2),dict(
        max_error=1.5,ransac=dict(min_iterations=100,max_iterations=1000,seed=int(seed))))
    focal=float(pair.camera1.params[0]);kk=k.copy();kk[0,0]*=focal/f;kk[1,1]*=focal/f
    return kk,pair.pose,stat

def pixel_error(p,q,k,r,trans):
    trans=np.asarray(trans)
    if not np.isfinite(r).all() or not np.isfinite(trans).all() or np.linalg.norm(trans)<1e-9:return 1e6
    return float(np.median(g.epipolar_error(g.bearings(p,k),g.bearings(q,k),r,trans/np.linalg.norm(trans)))*np.sqrt(k[0,0]*k[1,1]))

def calibrate(rows,k,t):
    proposals=[];details=[]
    for n,(i,j,pts) in enumerate(rows):
        if n%2:continue
        p,q=pts[:,:2],pts[:,2:4];held=np.arange(len(p))%3==0
        try:
            kk,pose,stat=shared_focal(p[~held],q[~held],k,i*1000+j)
            ratio=float(kk[0,0]/k[0,0]);err=pixel_error(p[held],q[held],kk,np.asarray(pose.R),pose.t)
            frac=int(stat['num_inliers'])/(~held).sum()
            ok=bool(np.isfinite(ratio) and .7<=ratio<=1.4 and frac>=.4 and err<=1.5)
            if ok:proposals.append((ratio,(t[i]+t[j])/2))
            details.append(dict(i=i,j=j,ratio=ratio if np.isfinite(ratio) else None,heldout_px=err,inlier_fraction=float(frac),accepted=ok))
        except (RuntimeError,ValueError) as exc:details.append(dict(i=i,j=j,error=str(exc),accepted=False))
    ratio=1.;mad=None;coverage=0.;stable=False
    if proposals:
        v=np.array(proposals);ratio=float(np.exp(np.median(np.log(v[:,0]))));mad=float(np.median(abs(np.log(v[:,0]/ratio))))
        coverage=len(np.unique(((v[:,1]-t[0])/30).astype(int)))/int(np.ceil((t[-1]-t[0])/30))
        stable=bool(len(v)>=12 and coverage>=.5 and mad<=.1)
    kk=k.copy();kk[0,0]*=ratio;kk[1,1]*=ratio;audit=[]
    if stable:
        for n,(i,j,pts) in enumerate(rows):
            if not n%2:continue
            p,q=pts[:,:2],pts[:,2:4];held=np.arange(len(p))%3==0;errors=[]
            for intr in [k,kk]:
                pose,stat,_=g.solve('essential',p[~held],q[~held],pts[~held,4],pts[~held,5],intr,i*1000+j)
                errors.append(pixel_error(p[held],q[held],intr,np.asarray(pose.R),pose.t))
            if np.isfinite(errors).all() and max(errors)<10:audit.append(errors)
    before=float(np.median(np.array(audit)[:,0])) if audit else None
    after=float(np.median(np.array(audit)[:,1])) if audit else None
    accepted=bool(stable and len(audit)>=12 and after<.95*before and before-after>=.02)
    return (kk if accepted else k.copy()),dict(proposals=details,candidate_ratio=ratio,log_mad=mad,coverage=coverage,
        stable=stable,audit_pairs=len(audit),audit_before_px=before,audit_after_px=after,accepted=accepted,applied_ratio=ratio if accepted else 1.)

def frame_axes(rp):
    up=np.median(rp[:,:3,1],axis=0);up/=np.linalg.norm(up)
    f=rp[0,:3,2].copy();f-=up*(f@up);f/=np.linalg.norm(f)
    return f,np.cross(up,f)

def relative_yaw(r,rp,i,axes):
    f,right=axes;target=rp[i,:3,:3]@r.T@np.array([0.,0.,1.])
    return float(wrap(np.arctan2(target@right,target@f)-np.arctan2(rp[i,:3,2]@right,rp[i,:3,2]@f)))

def homography_rotation(p,q,k,seed):
    held=np.arange(len(p))%3==0;g.cv2.setRNGSeed(int(seed))
    h,mask=g.cv2.findHomography(p[~held],q[~held],g.cv2.USAC_MAGSAC,2.,maxIters=2000,confidence=.999)
    if h is None:return None,dict(accepted=False)
    a=np.linalg.inv(k)@h@k
    if np.linalg.det(a)<0:a=-a
    u,s,v=np.linalg.svd(a);r=u@v
    if np.linalg.det(r)<0:return None,dict(accepted=False)
    b=np.c_[p[held],np.ones(held.sum())]@np.linalg.inv(k).T@r.T@k.T
    projected=b[:,:2]/b[:,2,None];errors=np.linalg.norm(projected-q[held],axis=1)
    span=float(np.ptp(p[:,0])/960);cells=len(np.unique(np.floor(p/[960,540]*[8,4]),axis=0))
    err=float(np.median(errors));fraction=float(np.mean(errors<=3));shape=float(s[0]/s[-1])
    accepted=bool(err<=1.5 and fraction>=.5 and shape<=1.03 and cells>=4 and span>=.15)
    return r,dict(accepted=accepted,heldout_px=err,heldout_fraction=fraction,singular_ratio=shape,cells=cells,xspan=span)

def fresh_edges(key,split):
    dest=OUT/'edges'/split/f'{key}.npz'
    if dest.exists():return dict(np.load(dest))
    tic=time.perf_counter();case,_,_,rp,k,meta=g.provider(key,split);rows=pair_data(key,split)
    kk,cal=calibrate(rows,k,case['t']);axes=frame_axes(rp)
    edges={name:[] for name in ['homography','focal_depth','focal_essential']};details=[]
    for i,j,pts in rows:
        p,q=pts[:,:2],pts[:,2:4];held=np.arange(len(p))%3==0;local={}
        r,d=homography_rotation(p,q,k,i*1000+j);w=d.get('heldout_fraction',0)/(1+d.get('heldout_px',1e6)) if d['accepted'] else 0.
        edges['homography'].append([i,j,relative_yaw(r,rp,i,axes) if r is not None else 0.,w]);local['homography']=d
        for name,method in [('focal_depth','depth_scale'),('focal_essential','essential')]:
            pose,stat,geom=g.solve(method,p[~held],q[~held],pts[~held,4],pts[~held,5],kk,i*1000+j)
            r=np.asarray(pose.R);tt=np.asarray(pose.t);err=pixel_error(p[held],q[held],kk,r,tt)
            frac=int(stat['num_inliers'])/(~held).sum();sane=np.isfinite(r).all() and np.isfinite(tt).all() and np.linalg.norm(tt)>1e-8
            angle=float(np.rad2deg(g.Rotation.from_matrix(r@(rp[j,:3,:3].T@rp[i,:3,:3]).T).magnitude())) if sane else 180.
            pn=np.c_[p[held],np.ones(held.sum())]@np.linalg.inv(kk).T;qn=np.c_[q[held],np.ones(held.sum())]@np.linalg.inv(kk).T
            positive=0.;derr=None
            if sane:
                xyz=g.cv2.triangulatePoints(np.c_[np.eye(3),np.zeros(3)],np.c_[r,tt],pn[:,:2].T,qn[:,:2].T)
                xyz=xyz[:3]/np.where(abs(xyz[3])>1e-12,xyz[3],1e-12);positive=float(np.mean((xyz[2]>0)&((r@xyz+tt[:,None])[2]>0)))
                if geom is not None:
                    estimated=(pn*(pts[held,4]+geom.shift1)[:,None]@r.T+tt)[:,2];expected=geom.scale*(pts[held,5]+geom.shift2)
                    valid=(estimated>1e-8)&(expected>1e-8);derr=float(np.median(abs(np.log(estimated[valid]/expected[valid])))) if valid.any() else 100.
            accepted=bool(sane and stat['num_inliers']>=12 and frac>=.4 and err<=1.5 and angle<=10 and positive>=.6 and (derr is None or derr<=.3))
            w=min(1,int(stat['num_inliers'])/60)*frac/(1+err) if accepted else 0.
            edges[name].append([i,j,relative_yaw(r,rp,i,axes) if sane else 0.,w])
            local[name]=dict(accepted=accepted,heldout_px=err,cheirality=positive,depth_error=derr)
        details.append(dict(i=i,j=j,methods=local))
    arr={name:np.array(v).reshape(-1,4) for name,v in edges.items()};dest.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(dest,**arr)
    save_json(dest.with_suffix('.json'),dict(calibration=cal,details=details,seconds=time.perf_counter()-tic,accepted={n:int((v[:,3]>.01).sum()) for n,v in arr.items()}))
    print('EXTRACT',key,'focal',cal['applied_ratio'],{n:int((v[:,3]>.01).sum()) for n,v in arr.items()},round(time.perf_counter()-tic,1),flush=True)
    return arr

def supported_graph(t,y,edges,cap):
    e=edges[edges[:,3]>.01]
    if len(e)<8:return np.zeros(len(t))
    mid=(t[e[:,0].astype(int)]+t[e[:,1].astype(int)])/2
    support=np.clip(1-np.min(abs(t[:,None]-mid),axis=1)/30,0,1)
    return g.correction(t,y,lg.unwrap_edges(e,y),'graph',cap)*support

def arbitrate(t,y,families,edges,blend):
    e=edges[edges[:,3]>.01];out=y.copy();enabled=0
    if len(e)<8:return out,dict(enabled_fraction=0.)
    i=e[:,0].astype(int);j=e[:,1].astype(int);mid=(t[i]+t[j])/2
    residual=np.minimum(abs(np.rad2deg(wrap(families[:,j]-families[:,i]-e[:,2]))),5.)
    weights=[]
    for n,tt in enumerate(t):
        local=abs(mid-tt)<=45
        if local.sum()<8:weights.append([0.,0.,0.]);continue
        w=e[local,3]*(1-abs(mid[local]-tt)/60)
        score=np.average(residual[:,local],axis=1,weights=w);a=np.exp(-(score-score.min()));a/=a.sum()
        target=float(a@families[:,n]);out[n]=y[n]+blend*(target-y[n]);enabled+=1;weights.append(a.tolist())
    # All families and baseline share the same first-frame anchor.
    out-=out[0]-y[0]
    return out,dict(enabled_fraction=enabled/len(t),mean_weights=np.mean(weights,axis=0).tolist())

def infer(keys,split):
    guard();g.cv2.setNumThreads(1)
    for key in keys:
        case,*_=g.provider(key,split);z=np.load(ROOT/f'runs/domain_v24_submission/{split}/signals/{key}.npz')
        t,y,s,families=z['times'],z['yaw'],z['speed'],z['family_yaws'];assert np.array_equal(t,case['t'])
        cached=[np.load(ROOT/f'runs/{run}/edges/{split}/{key}.npz') for run in ['domain_v17_roma','domain_v18']]
        arrays={source:np.concatenate([a[source] for a in cached]) for source in ['learned_depth_scale','learned_essential','consensus']}
        cyc=[]
        for run,a in zip(['domain_v17_roma','domain_v18'],cached):
            details=json.loads((ROOT/f'runs/{run}/edges/{split}/{key}.json').read_text())['details'];cyc.append(lg.cycles(a['learned_depth_scale'],details)[0])
        arrays['cycle']=np.concatenate(cyc);arrays.update(fresh_edges(key,split))
        signals={'baseline':y};diagnostics={}
        for name,source,cap in [('depth_graph3','learned_depth_scale',3),('depth_graph6','learned_depth_scale',6),('depth_graph20','learned_depth_scale',20),
             ('essential_graph6','learned_essential',6),('consensus_graph6','consensus',6),('cycle_graph6','cycle',6),
             ('homography_graph6','homography',6),('focal_depth_graph6','focal_depth',6),('focal_essential_graph6','focal_essential',6)]:
            signals[name]=y+supported_graph(t,y,arrays[source],cap)
            diagnostics[name]=dict(edges=int((arrays[source][:,3]>.01).sum()),max_correction_deg=float(np.rad2deg(abs(signals[name]-y).max())))
        for name,source,blend in [('arbitrate_half','learned_depth_scale',.5),('arbitrate_full','learned_depth_scale',1.),('consensus_arbitrate_half','consensus',.5)]:
            signals[name],diagnostics[name]=arbitrate(t,y,families,arrays[source],blend)
        for name,yaw in signals.items():
            save_json(OUT/split/'predictions'/name/f'{key}.json',dict(case['pred'],positions=integrate_velocity(t,yaw,s).tolist()))
        dest=OUT/split/'signals'/f'{key}.npz';dest.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(dest,times=t,speed=s,**signals)
        save_json(OUT/split/'diagnostics'/f'{key}.json',diagnostics);print('INFER',split,key,flush=True)
    save_json(REP/f'{split}_inference_guard.json',dict(passed=True,reference_loading_blocked=True,scoring_blocked=True,keys=keys))

def synthetic():
    rng=np.random.default_rng(43);x=rng.uniform([-4,-3,5],[4,3,15],(150,3));r=g.Rotation.from_rotvec([.07,.1,.03]).as_matrix();xx=x@r.T+[.6,.1,.2]
    true=np.array([[800.,0,480],[0,880,270],[0,0,1.]])
    def project(x):
        p=x@true.T;return p[:,:2]/p[:,2,None]
    start=true.copy();start[0,0]/=1.2;start[1,1]/=1.2
    kk,pose,stat=shared_focal(project(x),project(xx),start,7)
    assert np.max(abs(kk-true))<.01
    recovered,diag=homography_rotation(project(x),project(x@r.T),true,7)
    assert diag['accepted'] and np.rad2deg(g.Rotation.from_matrix(recovered@r.T).magnitude())<.01
    t=np.arange(601)*.5;y=np.deg2rad(2)*t/t[-1];e=np.array([[i,i+128,0.,1.] for i in range(0,473,16)])
    corr=supported_graph(t,y,e,6);assert np.mean((y+corr)**2)<np.mean(y*y)
    assert np.array_equal(supported_graph(t,y,np.empty((0,4)),6),np.zeros(len(t)))
    fallback,_=arbitrate(t,y,np.array([y,y,y]),np.empty((0,4)),1.);assert np.array_equal(fallback,y)
    rp=np.tile(np.eye(4),(2,1,1));camera_r=g.Rotation.from_euler('y',-.04).as_matrix();assert abs(relative_yaw(camera_r,rp,0,frame_axes(rp))-.04)<1e-9
    save_json(REP/'synthetic.json',dict(passed=True,checks=['Aspect-preserving shared focal recovery','Pure-rotation homography recovery','Camera-to-world yaw sign','Known accumulating error reduction','Empty geometry exact fallback','Empty arbitration exact fallback']))

def evaluate():
    from public_dataset import load_reference,load_timestamps,validate_prediction
    from metrics import score_trajectory
    from v20_five_pass import compare
    refs={k:load_reference(ROOT/'data/release',CLIPS[k]) for k in TRAIN};rows=[]
    for name in METHODS:
        per={}
        for key in TRAIN:
            p=json.loads((OUT/'train/predictions'/name/f'{key}.json').read_text());validate_prediction(p,key,load_timestamps(ROOT/'data/release',CLIPS[key]))
            per[key]=float(score_trajectory(np.array(p['times']),np.array(p['positions']),*refs[key])['drift_pct'])
        rows.append(dict(name=name,mean=float(np.mean(list(per.values()))),per_clip=per));print('SCORE',name,rows[-1]['mean'],flush=True)
    baseline=rows[0];assert abs(baseline['mean']-9.532022663626405)<1e-8
    for row in rows:
        row['comparison']=compare(row,baseline);delta=np.array([row['per_clip'][k]-baseline['per_clip'][k] for k in TRAIN])
        row['screen_pass']=bool((delta< -1e-8).sum()>=5 and delta.max()<=.5 and np.delete(delta,np.argmin(delta)).mean()<0)
    loco=[]
    for key in TRAIN:
        chosen=min(rows,key=lambda r:np.mean([r['per_clip'][k] for k in TRAIN if k!=key]));loco.append(dict(held_clip=key,chosen=chosen['name'],drift=chosen['per_clip'][key]))
    save_json(REP/'results.json',dict(rows=rows,loco=loco,loco_mean=float(np.mean([r['drift'] for r in loco])),scope='Repeated public TRAIN8 development. LOCO selects among already-developed methods; not independent validation.'))

def validate():
    plan=json.loads((REP/'plan.json').read_text())
    for p,h in {**plan['preserved'],**plan['dependencies']}.items():assert sha256(ROOT/p)==h,p
    for key in TRAIN:
        z=np.load(OUT/'train/signals'/f'{key}.npz');old=np.load(ROOT/f'runs/domain_v24_submission/train/signals/{key}.npz')
        assert np.array_equal(z['speed'],old['speed']) and np.array_equal(z['times'],old['times'])
        for name in METHODS:
            p=json.loads((OUT/'train/predictions'/name/f'{key}.json').read_text())
            assert np.isfinite(p['positions']).all();assert np.max(abs(np.array(p['positions'])-integrate_velocity(z['times'],z[name],z['speed'])))<1e-9
    save_json(REP/'validation.json',dict(passed=True,preserved_submissions=len(plan['preserved']),dependency_files=len(plan['dependencies']),predictions=len(TRAIN)*len(METHODS),speed_unchanged=True))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['freeze','synthetic','infer','evaluate','validate']);ap.add_argument('--split',default='train');ap.add_argument('--keys');args=ap.parse_args()
    freeze()
    if args.action=='synthetic':synthetic()
    if args.action=='infer':infer(args.keys.split(',') if args.keys else TRAIN,args.split)
    if args.action=='evaluate':evaluate()
    if args.action=='validate':validate()
