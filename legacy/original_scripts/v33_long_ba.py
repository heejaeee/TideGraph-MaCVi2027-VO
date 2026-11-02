"""Overlapping VGGT submap initialization and eight-view static-background BA.

An adaptation of multiview/shared-calibration principles, not a reproduction of
GLUEMAP, VGGT-SLAM 2.0, or an unpublished leaderboard entry.
"""
import argparse, datetime, time
from functools import lru_cache
import v30_horizon_ba as old
import v19_motion as motion
from v33_multiview_solver import bundle
from v13_common import ROOT,TRAIN,CLIPS,np,json,save_json,integrate_velocity
from public_dataset import sha256
bg=old.bg;g=bg.g;cv2=old.cv2;Rotation=old.Rotation
OUT=ROOT/'runs/domain_v33_long_ba';REP=ROOT/'reports/domain_v33_long_ba'
LOCAL=np.arange(0,32,4)
MODES=['shared','focal']
METHODS=['v29_control','initial_rotation','initial_course']+[m+'_'+f for m in MODES for f in ['rotation','residual','course','rotation_course']]

def freeze():
    settings=dict(methods=METHODS,views=LOCAL.tolist(),stride=32,max_tracks=72,max_nfev=120,
        initialization='Stitch three 16-frame VGGT submaps at offsets0,8,16 using eight identical overlap cameras. SO3 mean rotation, positive least-squares center scale and translation. Median intrinsic matrix shared across all8 selected views.',
        observations='Static eroded segmentation and CoTracker tracks visible in all8 views. >=24 tracks, >=4 of8x4 spatial cells. Initial triangulation positive in all views and median reprojection<20px. Spatially interleaved one-third withheld from BA.',
        acceptance='Heldout middle-view median<=1.5px, positive fraction>=.8, improvement>=10% or initial<=.5px and regression<=.05px. Shared focal +/-10%, reject within .001 of bounds. Same rules for every clip.',
        geometry='Joint8-view poses and landmarks. First camera fixed, translation-scale gauge, weak priors, softL1. Shared fixed K and shared focal refinement ablation. Save full optimized poses and K.',
        course='Only intervals>=8frames, median rotationally compensated bearing parallax>=.5deg, horizontal displacement>=.7, optimized displacement direction within10deg of initialization. Rebase to current yaw; same V19 temporal support and3deg cap. Speed fixed.',
        fusion='Rotation graph cap3deg, BA-change-only residual graph cap3deg, translation course cap3deg, rotation then course. Raw stitched initialization controls included.',
        promotion='Against V29: >=6/8 improved, mean delta<=-.1pp, worst regression<=.2pp, delta excluding best<0. No per-clip GT routing. Repeated TRAIN8 development, no independent validation or official-score prediction.')
    dest=REP/'plan.json'
    if dest.exists():assert json.loads(dest.read_text())['settings']==settings;return
    deps=[ROOT/'scripts'/p for p in ['v33_long_ba.py','v33_multiview_solver.py','v30_horizon_ba.py','v19_motion.py','v25_background.py']]
    for key in TRAIN:
        deps += [ROOT/f'runs/domain_v29_submission/train/signals/{key}.npz']
        deps += list((ROOT/f'runs/domain_probe_v2/vggt/windows/{key}').glob('*.npz'))+list((ROOT/f'runs/domain_v5/cotracker_windows/{key}').glob('*.npz'))
    save_json(dest,dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),settings=settings,
        dependencies={str(p.relative_to(ROOT)):sha256(p) for p in deps},preserved={str(p.relative_to(ROOT)):sha256(p) for p in (ROOT/'submissions').glob('*/submission.json')}))

def alignment(ar,ac,br,bc):
    rot=Rotation.from_matrix(ar@np.transpose(br,(0,2,1))).mean().as_matrix()
    b=bc@rot.T;a=ac-ac.mean(0);bb=b-b.mean(0)
    scale=float(np.sum(a*bb)/max(np.sum(bb*bb),1e-12))
    shift=ac.mean(0)-scale*b.mean(0)
    error=float(np.median(np.linalg.norm(scale*b+shift-ac,axis=1))/max(np.linalg.norm(ac[-1]-ac[0]),1e-6))
    return rot,scale,shift,error

def stitch(vgdir,st,meta):
    zs=[np.load(vgdir/f'{st+j:06d}.npz') for j in [0,8,16]]
    poses=zs[0]['c2w'].copy().astype(float);ks=zs[0]['intrinsics'].copy().astype(float);diag=[]
    for j,z in zip([8,16],zs[1:]):
        p=z['c2w'].astype(float);rot,sc,shift,err=alignment(poses[j:j+8,:3,:3],poses[j:j+8,:3,3],p[:8,:3,:3],p[:8,:3,3])
        if not (.2<sc<5 and err<.5):raise ValueError('unstable_overlap_alignment')
        p[:,:3,:3]=rot@p[:,:3,:3];p[:,:3,3]=sc*(p[:,:3,3]@rot.T)+shift
        poses=np.concatenate([poses,p[8:]]);ks=np.concatenate([ks,z['intrinsics'][8:]])
        diag.append(dict(scale=sc,normalized_center_error=err))
    k=np.median(ks,axis=0);ph=round(meta['height']*518/meta['width']/14)*14
    k[0]*=meta['width']/518;k[1]*=meta['height']/ph
    return poses[LOCAL,:3,:3],poses[LOCAL,:3,3],np.tile(k,(len(LOCAL),1,1)),diag

def extract(key,split='train'):
    dest=OUT/split/'observations'/f'{key}.json'
    if dest.exists():return json.loads(dest.read_text())
    case,trackdir,vgdir,rp,_,meta=g.provider(key,split);axes=bg.frame_axes(rp);raw=motion.planar_yaw(rp)
    static=json.loads((ROOT/'runs/domain_v5/segmentation/model.json').read_text())['static_ids']
    @lru_cache(maxsize=100)
    def mask(i):
        paths=[ROOT/f'runs/{"domain_v5" if split=="train" else "domain_v6_test"}/segmentation/{key}/{i:04d}.png',ROOT/f'runs/domain_v16/test/extra_masks/{key}/{i:04d}.png']
        p=next((p for p in paths if p.exists()),None);assert p is not None
        lab=cv2.imread(str(p),0);return cv2.erode(np.isin(lab,static).astype(np.uint8),np.ones((3,3),np.uint8))
    observations={m:dict(edges=[],residual_edges=[],courses=[]) for m in MODES+['initial']};details=[];tic=time.perf_counter()
    for path in sorted(trackdir.glob('*.npz')):
        st=int(path.stem)
        if st%32:continue
        rec=dict(start=st,methods={});details.append(rec)
        try:r,c,k,align=stitch(vgdir,st,meta)
        except (FileNotFoundError,IndexError,ValueError) as e:rec['reason']=str(e);continue
        tr=np.load(path)
        if len(tr['tracks'])<=LOCAL[-1]:rec['reason']='short_track_window';continue
        p=tr['tracks'][LOCAL].astype(float);valid=tr['visible'][LOCAL].all(axis=0)
        for j,local in enumerate(LOCAL):
            lab=mask(st+local);xy=np.floor(p[j]/[meta['width'],meta['height']]*[lab.shape[1],lab.shape[0]]).astype(int)
            inside=(xy[:,0]>=0)&(xy[:,0]<lab.shape[1])&(xy[:,1]>=0)&(xy[:,1]<lab.shape[0]);ix=np.flatnonzero(inside)
            inside[ix]&=lab[xy[ix,1],xy[ix,0]]>0;valid&=inside
        p=p[:,valid];rec.update(static_tracks=p.shape[1],alignment=align)
        if p.shape[1]<24:rec['reason']='few_static_tracks';continue
        order=np.lexsort((p[0,:,1],p[0,:,0]));p=p[:,order[np.linspace(0,len(order)-1,min(72,len(order))).astype(int)]]
        if len(np.unique(np.floor(p[0]/[meta['width'],meta['height']]*[8,4]),axis=0))<4:rec['reason']='narrow_support';continue
        x=old.triangulate(p,r,c,k);uv,z=old.project(x,r,c,k)
        p=p[:,(z>0).all(axis=0)&(np.median(np.linalg.norm(uv-p,axis=2),axis=0)<20)]
        if p.shape[1]<24:rec['reason']='poor_initial_geometry';continue
        held=np.arange(p.shape[1])%3==0;before,positive=old.audit(p[:,held],r,c,k);rec['before_px']=before
        def append(name,rr,cc,kk,weight):
            target=observations[name]
            for a,b in [(i,i+1) for i in range(7)]+[(0,4),(3,7),(0,7)]:
                i,j=int(st+LOCAL[a]),int(st+LOCAL[b]);dy=bg.relative_yaw(rr[b].T@rr[a],rp,i,axes);init=bg.relative_yaw(r[b].T@r[a],rp,i,axes)
                target['edges'].append([i,j,dy,weight]);target['residual_edges'].append([i,j,float(bg.wrap(dy-init)),weight])
                if LOCAL[b]-LOCAL[a]<8:continue
                d=rp[i,:3,:3]@rr[a].T@(cc[b]-cc[a]);d0=rp[i,:3,:3]@r[a].T@(c[b]-c[a])
                fraction=float(np.linalg.norm([d@axes[0],d@axes[1]])/max(np.linalg.norm(d),1e-10))
                angle=motion.angle(d,d0)
                ba=motion.bearings(p[a],kk[a]);bb=motion.bearings(p[b],kk[b]);rel=rr[b].T@rr[a]
                parallax=float(np.rad2deg(np.median(np.arccos(np.clip(np.sum((ba@rel.T)*bb,axis=1),-1,1)))))
                if fraction<.7 or angle>10 or parallax<.5:continue
                target['courses'].append(dict(i=i,j=j,weight=weight,course=float(bg.wrap(np.arctan2(d@axes[1],d@axes[0])-raw[i])),parallax_deg=parallax,horizontal_fraction=fraction,initial_angle_deg=angle))
        if before<=1.5 and positive>=.8:append('initial',r,c,k,1/(1+before))
        for mode in MODES:
            rr,cc,kk,diag=bundle(p[:,~held],r,c,k,focal=mode=='focal')
            after,positive=old.audit(p[:,held],rr,cc,kk)
            accepted=bool(np.isfinite(after) and positive>=.8 and after<=1.5 and (after<=.9*before or (before<=.5 and after<=before+.05)) and .901<diag['focal_ratio']<1.099)
            diag.update(accepted=accepted,before_px=before,after_px=after,positive_fraction=positive);rec['methods'][mode]=diag
            path=OUT/split/'geometry'/key/f'{st:04d}_{mode}.npz';path.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(path,r=rr,c=cc,k=kk,initial_r=r,initial_c=c,initial_k=k,tracks=p,heldout=held,frames=st+LOCAL)
            if accepted:append(mode,rr,cc,kk,min(1,p.shape[1]/60)/(1+after))
        print('BA',key,st,{m:round(d['after_px'],3) for m,d in rec['methods'].items()},flush=True)
    out=dict(key=key,split=split,seconds=time.perf_counter()-tic,observations=observations,details=details);save_json(dest,out);return out

def infer(keys,split='train'):
    bg.guard();cv2.setNumThreads(1)
    for key in keys:
        obs=extract(key,split)['observations'];base=np.load(ROOT/f'runs/domain_v29_submission/{split}/signals/{key}.npz');t,y,s=base['times'],base['yaw'],base['speed'];case,*_=g.provider(key,split)
        signals=dict(v29_control=y);diags={}
        for mode,data in obs.items():
            edge=np.array(data['edges']).reshape(-1,4);rot=y+bg.supported_graph(t,y,edge,3)
            er=np.array(data['residual_edges']).reshape(-1,4)
            if len(er):er[:,2]+=y[er[:,1].astype(int)]-y[er[:,0].astype(int)]
            res=y+bg.supported_graph(t,y,er,3)
            def course(yaw):
                rows=[dict(row,course=float(bg.wrap(row['course']+yaw[row['i']]))) for row in data['courses']]
                return motion.fit_motion(t,yaw,s,rows,'course')
            dc,d=course(y);signals[mode+'_rotation']=rot;signals[mode+'_course']=y+dc;diags[mode]=dict(course=d,edges=len(edge),course_edges=len(data['courses']))
            if mode!='initial':
                signals[mode+'_residual']=res;dc,d=course(rot);signals[mode+'_rotation_course']=rot+dc;diags[mode]['rotation_course']=d
        for name,yaw in signals.items():save_json(OUT/split/'predictions'/name/f'{key}.json',dict(case['pred'],positions=integrate_velocity(t,yaw,s).tolist()))
        path=OUT/split/'signals'/f'{key}.npz';path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,times=t,speed=s,**signals)
        save_json(OUT/split/'diagnostics'/f'{key}.json',dict(methods=diags,reference_loading_blocked=True,scoring_blocked=True));print('INFER',key,flush=True)

def synthetic():
    rng=np.random.default_rng(33);v=8;x=rng.uniform([-2,-1,5],[2,1,12],(60,3));r=Rotation.from_euler('y',np.linspace(0,2,v),degrees=True).as_matrix();c=np.c_[np.linspace(0,1.4,v),np.zeros((v,2))];k=np.tile(np.array([[800.,0,480],[0,800,270],[0,0,1.]]),(v,1,1))
    p,_=old.project(x,r,c,k);nr=r.copy();nr[1:]=Rotation.from_euler('y',rng.uniform(-.2,.2,v-1),degrees=True).as_matrix()@r[1:]
    before,_=old.audit(p[:,::3],nr,c,k);rr,cc,kk,_=bundle(p[:,np.arange(60)%3!=0],nr,c,k)
    after,_=old.audit(p[:,::3],rr,cc,kk);assert after<before*.1
    rot=Rotation.from_rotvec([.1,.2,.03]).as_matrix();br=rot.T@r;bc=(c-[1,2,3])@rot/2.5
    ar,sc,sh,err=alignment(r,c,br,bc);assert np.allclose(ar,rot) and abs(sc-2.5)<1e-8 and np.allclose(sh,[1,2,3]) and err<1e-8
    assert np.allclose(old.triangulate(p,r,c,k),x,atol=1e-7)
    save_json(REP/'synthetic.json',dict(passed=True,views=v,before_px=before,after_px=after,checks=['Sim3 submap gauge recovery','8-view BA improves heldout geometry','Projection and triangulation']))

def evaluate():
    old.OUT=OUT/'train';old.REP=REP;old.METHODS=METHODS
    # V30 evaluator expects its second row to be the V29 control.
    old.METHODS=[METHODS[1],METHODS[0]]+METHODS[2:];old.evaluate()

def validate():
    plan=json.loads((REP/'plan.json').read_text())
    for p,h in {**plan['dependencies'],**plan['preserved']}.items():assert sha256(ROOT/p)==h,p
    for key in TRAIN:
        z=np.load(OUT/'train/signals'/f'{key}.npz');base=np.load(ROOT/f'runs/domain_v29_submission/train/signals/{key}.npz');assert np.array_equal(z['speed'],base['speed'])
        for name in METHODS:
            p=json.loads((OUT/'train/predictions'/name/f'{key}.json').read_text());assert np.allclose(p['positions'],integrate_velocity(z['times'],z[name],z['speed']),rtol=0,atol=1e-9)
            assert abs(z[name]-base['yaw']).max()<=np.deg2rad(6)+1e-9
    save_json(REP/'validation.json',dict(passed=True,predictions=len(TRAIN)*len(METHODS),preserved_submissions=len(plan['preserved']),speed_unchanged=True))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['freeze','synthetic','infer','evaluate','validate']);ap.add_argument('--keys');ap.add_argument('--split',default='train');a=ap.parse_args();freeze()
    if a.action=='infer':infer(a.keys.split(',') if a.keys else TRAIN,a.split)
    elif a.action!='freeze':globals()[a.action]()
