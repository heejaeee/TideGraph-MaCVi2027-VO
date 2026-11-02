"""Two-view image-only translation, independently of BA centers/depth/scale.

Fixed-rotation and free-essential solvers share image features/calibration;
independence means no trajectory-position initialization, not independent data.
"""
import argparse,datetime,time
from collections import Counter
import v43_ba_ablation as audit
core=audit.core
ROOT,np,json,save_json=core.ROOT,core.np,core.json,core.save_json
cv2=core.cv2
OUT=ROOT/'runs/domain_v45_independent_course';REP=ROOT/'reports/domain_v45_independent_course'
TRAIN,CLIPS=core.TRAIN,core.CLIPS
CANDIDATES=['fixed_course','free_course','solver_agreement','ba_confirm_fixed','ba_confirm_free','ba_confirm_both','independent_confirm_ba']
CONTROLS=['v24_control','ba_course_control','ba_both_control','v42_control']
METHODS=CONTROLS+CANDIDATES
CFG=dict(ransac_iterations=512,pixel_threshold=1.5,min_matches=40,max_matches=240,
    fit_inliers=24,fit_fraction=.4,held_min=8,held_fraction=.5,positive_fraction=.7,
    parallax_deg=.5,horizontal_fraction=.7,agreement_deg=3.,min_cells=4,min_x_fraction=.1)


def freeze():
    path=REP/'plan.json'
    if path.exists():return
    files=list((ROOT/'scripts').glob('*.py'))
    for key in TRAIN:
        files+=list((ROOT/f'runs/domain_v27_sift/matches/train/{key}').glob('*.npz'))
        files += [ROOT/f'runs/domain_v24_submission/train/signals/{key}.npz',ROOT/f'runs/domain_v43_ba_ablation/signals/{key}.npz',
            ROOT/f'runs/domain_v43_ba_ablation/geometry/mixed_fixed/{key}.json',ROOT/f'runs/domain_v42_global_ba/train/signals/{key}.npz',
            ROOT/f'runs/domain_v4/calibrated_rotation_train/{key}.npz',ROOT/f'runs/domain_probe_v2/r3/diagnostics/{key}.poses.npz']
    save_json(path,dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),settings=CFG,methods=METHODS,
        extraction='Raw static RootSIFT mutual ratio matches; same cached images/calibration as BA, no BA-track retention or geometry filter. i%16==0,gap16..128 (8..64sec),spatially ordered max240; every5th correspondence withheld. No triangulated depth or camera centers read.',
        fixed='V24 camera rotation and R3 tilt compensate rotation. Two-normal RANSAC fits translation null direction; pixel Sampson scoring and SVD refit. Sign selected by fit cheirality, no trajectory displacement used.',
        free='Five-point essential matrix OpenCV RANSAC on fit points. Relative rotation and unit translation from fit-only recoverPose. No initialized camera positions, no learned depth, no BA. Global direction anchored by V24 orientation at first endpoint.',
        validation='Disjoint held-out correspondence Sampson residual, positive depth and compensated parallax; image spread and horizontal displacement. Unit-baseline triangulation solely for cheirality/parallax; no metric scale estimated.',
        fusion='Independent course or original V43 BA course when circular angle agrees<=3deg. Both-solvers consensus uses circular weighted mean. All targets retain V19 cap3deg/min8 observations/50% time coverage; V24 speed unchanged. No baseline-course agreement gate.',
        selection='Same V43 screen >=6/8 improved,mean delta<=-.1pp,worst regression<=.2pp,excluding best<0. LOCO and turn-group exclusion choose only among7 new methods with same proportional screen.',
        scope='Reused public TRAIN8; algorithmically independent centers, shared observations/calibration/orientation anchor. No independent dataset, TEST inference or upload.',
        dependencies={str(p.relative_to(ROOT)):audit.sha256(p) for p in files},
        preserved={str(p.relative_to(ROOT)):audit.sha256(p) for p in (ROOT/'submissions').glob('*/submission.json')}))


def inputs(key):
    case,td,vg,rp,k,meta=core.g.provider(key,'train')
    z=np.load(ROOT/f'runs/domain_v24_submission/train/signals/{key}.npz')
    f,right=core.bg.frame_axes(rp);up=np.cross(f,right)
    r=core.Rotation.from_rotvec((z['yaw']-core.local.motion.planar_yaw(rp))[:,None]*up).as_matrix()@rp[:,:3,:3]
    return case,z,r,k,meta,(f,right)


def normalized(p,k):return np.c_[p,np.ones(len(p))]@np.linalg.inv(k).T


def sampson(a,b,r,t,k):
    ra=a@r.T;ea=np.cross(t,ra);eb=np.cross(b,t)@r
    return np.abs((b*ea).sum(1))/np.maximum(np.sqrt((ea[:,:2]**2).sum(1)+(eb[:,:2]**2).sum(1)),1e-12)*np.sqrt(k[0,0]*k[1,1])


def positive(a,b,r,t):
    x=cv2.triangulatePoints(np.c_[np.eye(3),np.zeros(3)],np.c_[r,t],a[:,:2].T,b[:,:2].T)
    good=abs(x[3])>1e-12;xyz=np.zeros((len(a),3));xyz[good]=x[:3,good].T/x[3,good,None]
    return good&np.isfinite(xyz).all(1)&(xyz[:,2]>0)&((xyz@r.T+t)[:,2]>0)


def fixed_solver(a,b,r,k,seed):
    rng=np.random.default_rng(seed);ra=a@r.T;n=np.cross(ra,b)
    ids=rng.integers(0,len(a),(CFG['ransac_iterations'],2));ts=np.cross(n[ids[:,0]],n[ids[:,1]])
    norm=np.linalg.norm(ts,axis=1);ts=ts[norm>1e-12]/norm[norm>1e-12,None]
    if not len(ts):return None
    ea=np.cross(ts[:,None,:],ra[None]);eb=np.cross(b[None],ts[:,None,:])@r
    err=np.abs(np.einsum('ni,mi->mn',n,ts))/np.maximum(np.sqrt((ea[:,:,:2]**2).sum(2)+(eb[:,:,:2]**2).sum(2)),1e-12)*np.sqrt(k[0,0]*k[1,1])
    good=err<=CFG['pixel_threshold'];counts=good.sum(1)
    order=np.lexsort((np.minimum(err,CFG['pixel_threshold']).sum(1),-counts));inliers=good[order[0]]
    if inliers.sum()<CFG['fit_inliers']:return None
    t=ts[order[0]]
    for _ in range(3):
        if inliers.sum()<CFG['fit_inliers']:return None
        t=np.linalg.svd(n[inliers],full_matrices=False)[2][-1];inliers=sampson(a,b,r,t,k)<=CFG['pixel_threshold']
    if positive(a[inliers],b[inliers],r,-t).sum()>positive(a[inliers],b[inliers],r,t).sum():t=-t
    return r,t,inliers


def free_solver(p,q,k,seed):
    cv2.setRNGSeed(seed)
    e,mask=cv2.findEssentialMat(p,q,k,method=cv2.RANSAC,prob=.999,threshold=CFG['pixel_threshold'],maxIters=2000)
    if e is None or mask is None:return None
    choices=[]
    for ee in np.asarray(e).reshape(-1,3,3):
        count,r,t,m=cv2.recoverPose(ee,p,q,k,mask=mask.copy());choices.append((int(count),r,t.ravel(),m.ravel()>0))
    _,r,t,inliers=max(choices,key=lambda x:x[0]);return r,t,inliers


def inspect_solution(solution,p,q,fit,held,k,ri,axes,meta):
    if solution is None:return None,'no_fit',{}
    r,t,inliers=solution;a=normalized(p,k);b=normalized(q,k)
    detail=dict(fit_inliers=int(inliers.sum()),fit_fraction=float(inliers.mean()))
    if inliers.sum()<CFG['fit_inliers'] or inliers.mean()<CFG['fit_fraction']:return None,'fit_support',detail
    a_fit,b_fit=a[fit][inliers],b[fit][inliers]
    fit_positive=float(positive(a_fit,b_fit,r,t).mean())
    err=sampson(a[held],b[held],r,t,k);valid=err<=CFG['pixel_threshold']
    detail.update(held_median_px=float(np.median(err)),held_inliers=int(valid.sum()),held_fraction=float(valid.mean()),fit_positive=fit_positive)
    if valid.sum()<CFG['held_min'] or valid.mean()<CFG['held_fraction'] or np.median(err)>CFG['pixel_threshold']:return None,'held_residual',detail
    aa,bb=a[held][valid],b[held][valid];pos=positive(aa,bb,r,t)
    detail['held_positive']=float(pos.mean())
    if min(fit_positive,pos.mean())<CFG['positive_fraction']:return None,'cheirality',detail
    aa=aa[pos]@r.T;bb=bb[pos];aa/=np.linalg.norm(aa,axis=1,keepdims=True);bb/=np.linalg.norm(bb,axis=1,keepdims=True)
    parallax=float(np.median(np.rad2deg(np.arctan2(np.linalg.norm(np.cross(aa,bb),axis=1),(aa*bb).sum(1)))))
    detail['parallax_deg']=parallax
    if parallax<CFG['parallax_deg']:return None,'parallax',detail
    fitp=p[fit][inliers];spread=np.ptp(fitp,axis=0)/[meta['width'],meta['height']];cells=len(np.unique(np.floor(fitp/[meta['width'],meta['height']]*[8,4]).astype(int),axis=0))
    detail.update(cells=cells,x_fraction=float(spread[0]))
    if cells<CFG['min_cells'] or spread[0]<CFG['min_x_fraction']:return None,'image_spread',detail
    direction=ri@(-r.T@t);direction/=np.linalg.norm(direction);f,right=axes
    horizontal=float(np.linalg.norm([direction@f,direction@right]));detail['horizontal']=horizontal
    if horizontal<CFG['horizontal_fraction']:return None,'vertical_translation',detail
    weight=min(1,int(inliers.sum())/80)*float(valid.mean())/(1+float(np.median(err)))
    return dict(course=float(np.arctan2(direction@right,direction@f)),weight=weight,direction=direction.tolist()),'accepted',detail


def extract(keys):
    core.bg.guard();cv2.setNumThreads(1)
    # Guard against accidentally reintroducing BA-center initialization.
    def forbidden(*args,**kw):raise RuntimeError('Trajectory integration prohibited in independent extraction')
    core.integrate_velocity=forbidden;audit.initial=forbidden
    for key in keys:
        dest=OUT/f'observations/{key}.json'
        if dest.exists():continue
        case,z,r,k,meta,axes=inputs(key);rows={'fixed':[],'free':[]};details=[];counts=Counter();started=time.perf_counter()
        for path in sorted((ROOT/f'runs/domain_v27_sift/matches/train/{key}').glob('*.npz')):
            d=np.load(path);i,j=int(d['i']),int(d['j']);p,q=d['p'].astype(float),d['q'].astype(float)
            if j-i<16:continue
            if len(p)<CFG['min_matches']:counts['insufficient_raw_matches']+=1;continue
            ids=np.lexsort((p[:,1],p[:,0]));ids=ids[np.linspace(0,len(ids)-1,min(CFG['max_matches'],len(ids))).astype(int)];p,q=p[ids],q[ids]
            held=np.arange(len(p))%5==0;fit=~held;seed=i*1000+j
            for name in ['fixed','free']:
                try:
                    if name=='fixed':solution=fixed_solver(normalized(p[fit],k),normalized(q[fit],k),r[j].T@r[i],k,seed)
                    else:solution=free_solver(p[fit],q[fit],k,seed)
                    row,reason,diag=inspect_solution(solution,p,q,fit,held,k,r[i],axes,meta)
                except cv2.error as exc:row,reason,diag=None,'opencv_degenerate',dict(error=str(exc))
                counts[name+'_'+reason]+=1;details.append(dict(i=i,j=j,method=name,reason=reason,**diag))
                if row is not None:rows[name].append(dict(i=i,j=j,**row))
        save_json(dest,dict(rows=rows,details=details,counts=dict(counts),seconds=time.perf_counter()-started,reference_access_blocked=True,trajectory_integration_blocked=True))
        print('EXTRACT',key,dict(counts),flush=True)


def agreement(first,second,retain_first):
    lookup={(r['i'],r['j']):r for r in second};out=[]
    for r in first:
        other=lookup.get((r['i'],r['j']))
        if other is None:continue
        delta=float(core.bg.wrap(other['course']-r['course']))
        if abs(delta)>np.deg2rad(CFG['agreement_deg']):continue
        course=r['course'] if retain_first else float(core.bg.wrap(r['course']+delta*other['weight']/(r['weight']+other['weight'])))
        out.append(dict(i=r['i'],j=r['j'],course=course,weight=min(r['weight'],other['weight'])))
    return out


def infer(keys):
    core.bg.guard()
    for key in keys:
        case,z,r,k,meta,axes=inputs(key);t,y,s=z['times'],z['yaw'],z['speed']
        obs=json.loads((OUT/f'observations/{key}.json').read_text())['rows'];fixed,free=obs['fixed'],obs['free']
        ba=json.loads((ROOT/f'runs/domain_v43_ba_ablation/geometry/mixed_fixed/{key}.json').read_text())['courses']
        both=agreement(fixed,free,False)
        groups=dict(fixed_course=fixed,free_course=free,solver_agreement=both,
            ba_confirm_fixed=agreement(ba,fixed,True),ba_confirm_free=agreement(ba,free,True),
            ba_confirm_both=agreement(ba,both,True),independent_confirm_ba=agreement(both,ba,True))
        old=np.load(ROOT/f'runs/domain_v43_ba_ablation/signals/{key}.npz')
        signals=dict(v24_control=y,ba_course_control=old['ungated_course'],ba_both_control=old['ungated_both'],v42_control=old['v42_control']);diagnostics={}
        for name,rows in groups.items():
            dy,diag=core.local.motion.fit_motion(t,y,s,rows,'course');signals[name]=y+dy;diagnostics[name]=diag
        for name,yaw in signals.items():save_json(OUT/f'predictions/{name}/{key}.json',dict(case['pred'],positions=core.integrate_velocity(t,yaw,s).tolist()))
        dest=OUT/f'signals/{key}.npz';dest.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(dest,times=t,speed=s,**signals)
        save_json(OUT/f'diagnostics/{key}.json',dict(methods=diagnostics,groups=groups,total_turn_deg=float(np.rad2deg(abs(np.diff(y)).sum()))))
        print('INFER',key,{n:(len(rows),diagnostics[n]['fallback']) for n,rows in groups.items()},flush=True)


def evaluate():
    from public_dataset import load_reference,load_timestamps,validate_prediction
    from metrics import score_trajectory
    refs={k:load_reference(ROOT/'data/release',CLIPS[k]) for k in TRAIN};rows=[]
    for name in METHODS:
        per={}
        for key in TRAIN:
            pred=json.loads((OUT/f'predictions/{name}/{key}.json').read_text());validate_prediction(pred,key,load_timestamps(ROOT/'data/release',CLIPS[key]))
            per[key]=float(score_trajectory(np.array(pred['times']),np.array(pred['positions']),*refs[key])['drift_pct'])
        row=dict(name=name,mean=float(np.mean(list(per.values()))),per_clip=per);rows.append(row)
        row['vs_v24']=audit.compare(per,rows[0]['per_clip']);row['screen_pass']=name in CANDIDATES and audit.screen(row['vs_v24'],8)
        print('SCORE',name,row['mean'],row['screen_pass'],flush=True)
    for row,expected in zip(rows[:4],[9.532022663626405,9.286980562737416,9.246608608173823,8.907951350628013]):assert abs(row['mean']-expected)<1e-8
    def select(excluded):
        included=[k for k in TRAIN if k not in excluded];base={k:rows[0]['per_clip'][k] for k in included};candidates=[]
        for row in rows[4:]:
            per={k:row['per_clip'][k] for k in included};comp=audit.compare(per,base)
            if audit.screen(comp,len(included)):candidates.append((float(np.mean(list(per.values()))),row['name'],row))
        win=min(candidates,key=lambda x:(x[0],x[1]))[2] if candidates else rows[0]
        return dict(excluded=excluded,selected=win['name'],heldout={k:win['per_clip'][k] for k in excluded},delta={k:win['per_clip'][k]-rows[0]['per_clip'][k] for k in excluded})
    loco=[select([k]) for k in TRAIN];groups=json.loads((ROOT/'reports/domain_v43_ba_ablation/results.json').read_text())['turn_groups']
    group=[dict(group=g,**select(keys)) for g,keys in groups.items()];passed=[r for r in rows if r['screen_pass']]
    save_json(REP/'results.json',dict(rows=rows,selected=min(passed,key=lambda r:(r['mean'],r['name']))['name'] if passed else 'v24_control',
        loco=loco,loco_mean=float(np.mean([list(f['heldout'].values())[0] for f in loco])),group=group,group_mean=float(np.mean([v for f in group for v in f['heldout'].values()])),
        scope='Repeated public TRAIN8, independent of initialized centers but not independent data or all calibration/orientation inputs. No official TEST inference.'))


def synthetic():
    rng=np.random.default_rng(45);x=rng.uniform([-4.,-2.,8.],[4.,2.,20.],(150,3));r=core.Rotation.from_euler('y',4,degrees=True).as_matrix();t=np.array([-1.,.01,.1]);t/=np.linalg.norm(t)
    k=np.array([[900.,0,480],[0,900.,270],[0,0,1.]])
    def project(x):p=x@k.T;return p[:,:2]/p[:,2,None]
    p,q=project(x),project(x@r.T+t);a,b=normalized(p,k),normalized(q,k)
    rr,tt,inside=fixed_solver(a,b,r,k,45);angle=core.local.motion.angle(tt,t);assert angle<.001 and inside.all()
    assert np.median(sampson(a,b,r,t,k))<1e-8 and positive(a,b,r,t).all()
    rr,tt,inside=free_solver(p,q,k,45);assert core.local.motion.angle(tt,t)<.1
    # Corrupt 20% of correspondences; fixed-R RANSAC must retain true direction.
    noisy=q.copy();noisy[:30]=rng.uniform([0,0],[960,540],(30,2));rr,tt,inside=fixed_solver(a,normalized(noisy,k),r,k,45)
    assert core.local.motion.angle(tt,t)<.1 and inside.sum()>=115
    assert core.local.motion.angle(-r.T@t,-r.T@tt)<.1
    first=[dict(i=0,j=16,course=np.pi-.01,weight=1.)];second=[dict(i=0,j=16,course=-np.pi+.01,weight=1.)]
    assert len(agreement(first,second,False))==1
    save_json(REP/'synthetic.json',dict(passed=True,checks=['Fixed-R and free essential recover known translation','Pixel Sampson formula','Cheirality selects correct direction sign','20percent outlier robustness','Circular agreement wraps +/-pi']))
    print('SYNTHETIC PASSED',flush=True)


def validate():
    plan=json.loads((REP/'plan.json').read_text())
    for p,h in {**plan['dependencies'],**plan['preserved']}.items():assert audit.sha256(ROOT/p)==h,p
    assert json.loads((REP/'synthetic.json').read_text())['passed']
    for key in TRAIN:
        z=np.load(OUT/f'signals/{key}.npz');base=np.load(ROOT/f'runs/domain_v24_submission/train/signals/{key}.npz');diag=json.loads((OUT/f'diagnostics/{key}.json').read_text())
        assert np.array_equal(z['times'],base['times']) and np.array_equal(z['speed'],base['speed'])
        for name in METHODS:
            pred=json.loads((OUT/f'predictions/{name}/{key}.json').read_text());assert np.allclose(pred['positions'],core.integrate_velocity(z['times'],z[name],z['speed']),rtol=0,atol=1e-9)
            if name in CANDIDATES:
                assert abs(z[name]-base['yaw']).max()<=np.deg2rad(3)+1e-9
                if diag['methods'][name]['fallback']:assert np.array_equal(z[name],base['yaw'])
    save_json(REP/'validation.json',dict(passed=True,predictions=len(METHODS)*8,speed_unchanged=True,exact_fallback=True,
        preserved_submissions=len(plan['preserved']),input_hashes=len(plan['dependencies']),test_inference=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','synthetic','extract','infer','evaluate','validate']);p.add_argument('--keys');a=p.parse_args()
    if a.action=='freeze':freeze()
    else:
        assert (REP/'plan.json').exists()
        if a.action in ['extract','infer']:globals()[a.action](a.keys.split(',') if a.keys else TRAIN)
        else:globals()[a.action]()
