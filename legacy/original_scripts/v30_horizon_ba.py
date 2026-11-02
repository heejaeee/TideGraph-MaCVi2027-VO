"""Four-view VGGT-initialized local bundle adjustment with held-out tracks.

Camera rotations, translations and 3D landmarks are optimized jointly. Optional
shared focal and direct sky-water horizon constraints are controlled ablations.
This is not a reproduction of the undocumented competing submission.
"""
import argparse,datetime,time
from functools import lru_cache
from scipy.sparse import lil_matrix
from scipy.optimize import least_squares
import v25_background as bg
from v13_common import ROOT,TRAIN,CLIPS,np,json,save_json,integrate_velocity
from public_dataset import sha256
g=bg.g;cv2=g.cv2;Rotation=g.Rotation
OUT=ROOT/'runs/domain_v30_horizon_ba';REP=ROOT/'reports/domain_v30_horizon_ba'
MODES=['ba','ba_focal','ba_horizon','ba_focal_horizon']
METHODS=['v24_control','v29_control','vggt_initial_graph']+[base+'_'+mode for base in ['v24','v29'] for mode in MODES]
LOCAL=[0,5,10,15]

def freeze():
    settings=dict(methods=METHODS,windows='VGGT16-frame windows, every32 frames. Four views at offsets0,5,10,15. CoTracker correspondence identities visible and static in all4; maximum96 spatially distributed tracks.',
        ba='First camera fixed. Joint optimize remaining3 rotations/translations and fit-track3D landmarks. Rotation increments bounded3deg per axis, center increments bounded0.5 initial baseline. Translation scale anchor; weak pose/point priors. scipy sparse least_squares soft_l1, max50 evaluations.',
        focal='Optional one common multiplicative focal factor for4 cameras, +/-10%, log prior0.05. Principal points and aspect ratios fixed. VGGT intrinsics rescaled exactly from518 x14-aligned input.',
        horizon='Only direct sky-to-water class transitions. >=40 columns, xspan>=35% frame width, RANSAC line residual<=2px at480wide. Require>=2 valid views. Fixed initial world normal from first available line; optimize camera tilt consistency. No roof/shoreline substitution.',
        audit='Spatially interleaved1/3 tracks held out. Triangulate heldout endpoints and score middle-view reprojection. >=12 fit,>=6 audit tracks. Positive depth>=80%, audit median<=1.5px and >=10% improvement over initialization, or initial already<=0.5px and no >0.05px regression. Reject focal at bound. No GT in inference.',
        fusion='Relative rotation edges projected in existing R3 gravity plane and robust support-gated graph, cap3deg. Apply separately to V24 and V29 yaw; exact existing speed preserved. Initial VGGT graph control uses unchanged poses with <=1.5px audit.',
        promotion='Against V29: >=6/8 improved, worst regression<=0.2pp, mean improvement>=0.1pp, improvement excluding best clip. Otherwise retain V29 for scarce submission slots.',
        scope='Repeated TRAIN8 development, no independent holdout. New local BA integration inspired by public description; not exact competitor reproduction. No online upload.')
    p=REP/'plan.json'
    if p.exists():assert json.loads(p.read_text())['settings']==settings;return
    files=[ROOT/'scripts/v30_horizon_ba.py',ROOT/'scripts/v25_background.py',ROOT/'scripts/v17_depth_geometry.py']
    for key in TRAIN:
        files += [ROOT/f'runs/domain_v24_submission/train/signals/{key}.npz',ROOT/f'runs/domain_v29_submission/train/signals/{key}.npz']
        files += list((ROOT/f'runs/domain_probe_v2/vggt/windows/{key}').glob('*.npz'))+list((ROOT/f'runs/domain_v5/cotracker_windows/{key}').glob('*.npz'))
    save_json(p,dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),settings=settings,
        dependencies={str(p.relative_to(ROOT)):sha256(p) for p in files},preserved={str(p.relative_to(ROOT)):sha256(p) for p in (ROOT/'submissions').glob('*/submission.json')}))

def project(x,r,c,k):
    q=np.einsum('vni,vij->vnj',x[None]-c[:,None],r)
    z=q[:,:,2];u=np.einsum('vni,vji->vnj',q,k)
    return u[:,:,:2]/np.maximum(z[:,:,None],1e-6),z

def triangulate(p,r,c,k):
    ex=[np.c_[r[j].T,-r[j].T@c[j]] for j in [0,len(r)-1]]
    uv=[(np.c_[p[j],np.ones(p.shape[1])]@np.linalg.inv(k[j]).T)[:,:2].T for j in [0,len(r)-1]]
    xyz=cv2.triangulatePoints(ex[0],ex[1],uv[0],uv[1]);return (xyz[:3]/np.where(abs(xyz[3])>1e-10,xyz[3],1e-10)).T

def audit(p,r,c,k):
    x=triangulate(p,r,c,k);pred,z=project(x,r,c,k);err=np.linalg.norm(pred[1:-1]-p[1:-1],axis=2)
    return float(np.median(err)),float(np.mean(z>0))

def line_from_labels(lab):
    # ADE20K sky=2, water/sea/river etc. Only immediately adjacent classes.
    sky=lab==2;water=np.isin(lab,[21,26,60,109,113,128]);ys,xs=np.where(sky[:-1]&water[1:])
    keep=(xs>.05*lab.shape[1])&(xs<.95*lab.shape[1]);p=np.c_[xs[keep],ys[keep]+.5].astype(float)
    if len(p)<40:return None,dict(points=len(p),accepted=False)
    rng=np.random.default_rng(27);best=np.zeros(len(p),bool)
    for _ in range(128):
        a,b=p[rng.choice(len(p),2,replace=False)];dx=b[0]-a[0]
        if abs(dx)<.2*lab.shape[1]:continue
        slope=(b[1]-a[1])/dx
        if abs(slope)>.3:continue
        residual=abs(p[:,1]-(slope*p[:,0]+a[1]-slope*a[0]));inside=residual<=2
        if inside.sum()>best.sum():best=inside
    if best.sum()<40:return None,dict(points=len(p),accepted=False)
    pp=p[best];span=float(np.ptp(pp[:,0])/lab.shape[1]);slope,bias=np.linalg.lstsq(np.c_[pp[:,0],np.ones(len(pp))],pp[:,1],rcond=None)[0]
    accepted=bool(span>=.35);line=np.array([-slope,1.,-bias])
    # Labels are480wide; raw image is960wide, including y scale2.
    line[2]*=2
    return (line if accepted else None),dict(points=len(p),inliers=int(best.sum()),xspan=span,accepted=accepted)

def bundle(p,r0,c0,k0,focal=False,lines=None,max_nfev=50):
    v,n=p.shape[:2];assert v==4
    x0=triangulate(p,r0,c0,k0);scale=max(float(np.linalg.norm(c0[-1]-c0[0])),1e-4)
    depth=max(float(np.median(np.linalg.norm(x0-c0[0],axis=1))),scale)
    nc=6*(v-1);npnt=3*n;total=nc+npnt+int(focal);initial=np.r_[np.zeros(nc),x0.ravel(),np.zeros(int(focal))]
    lo=np.full(total,-np.inf);hi=np.full(total,np.inf)
    for j in range(v-1):lo[j*6:j*6+3]=-np.deg2rad(3);hi[j*6:j*6+3]=np.deg2rad(3);lo[j*6+3:j*6+6]=-.5;hi[j*6+3:j*6+6]=.5
    if focal:lo[-1]=np.log(.9);hi[-1]=np.log(1.1)
    lines=lines if lines is not None else [None]*v;valid=[j for j,l in enumerate(lines) if l is not None]
    if len(valid)<2:valid=[]
    def unpack(z):
        delta=z[:nc].reshape(v-1,6);r=r0.copy();c=c0.copy();r[1:]=Rotation.from_rotvec(delta[:,:3]).as_matrix()@r0[1:];c[1:]+=delta[:,3:]*scale
        k=k0.copy();f=float(np.exp(z[-1])) if focal else 1.;k[:,0,0]*=f;k[:,1,1]*=f
        return r,c,k,z[nc:nc+npnt].reshape(n,3),delta,f
    def residual(z):
        r,c,k,x,d,f=unpack(z);uv,depths=project(x,r,c,k)
        # Pixel residuals plus weak pose and landmark priors; retain gauge scale.
        result=[(uv-p).ravel(),(d[:,:3]/np.deg2rad(3)).ravel(),(d[:,3:]/.3).ravel(),((x-x0)/(depth*.5)).ravel(),np.array([(np.linalg.norm(c[-1]-c[0])/scale-1)/.01])]
        if focal:result.append(np.array([np.log(f)/.05]))
        if valid:
            normals=[]
            for j in valid:
                local=k[j].T@lines[j];local/=np.linalg.norm(local);normals.append(r[j]@local)
            reference=r0[valid[0]]@(k[valid[0]].T@lines[valid[0]]);reference/=np.linalg.norm(reference)
            result.append((np.cross(np.array(normals),reference)/np.deg2rad(.5)).ravel())
        return np.concatenate(result)
    count=len(residual(initial));sparsity=lil_matrix((count,total),dtype=int)
    for j in range(v):
        for a in range(n):
            rr=(j*n+a)*2;sparsity[rr:rr+2,nc+a*3:nc+a*3+3]=1
            if j:sparsity[rr:rr+2,(j-1)*6:j*6]=1
            if focal:sparsity[rr:rr+2,-1]=1
    off=2*v*n
    for j in range(v-1):
        sparsity[off+j*3:off+j*3+3,j*6:j*6+3]=np.eye(3,dtype=int)
        sparsity[off+3*(v-1)+j*3:off+3*(v-1)+j*3+3,j*6+3:j*6+6]=np.eye(3,dtype=int)
    off+=nc;sparsity[off:off+npnt,nc:nc+npnt]=np.eye(npnt,dtype=int);off+=npnt
    sparsity[off,(v-2)*6+3:(v-1)*6]=1;off+=1
    if focal:sparsity[off,-1]=1;off+=1
    for j in valid:
        if j:sparsity[off:off+3,(j-1)*6:(j-1)*6+3]=1
        if focal:sparsity[off:off+3,-1]=1
        off+=3
    assert off==count
    sol=least_squares(residual,initial,jac_sparsity=sparsity.tocsr(),bounds=(lo,hi),loss='soft_l1',f_scale=1.,max_nfev=max_nfev,x_scale='jac',ftol=1e-5)
    r,c,k,x,d,f=unpack(sol.x)
    return r,c,k,dict(nfev=sol.nfev,success=bool(sol.success),initial_cost=float(np.mean(residual(initial)[:2*v*n]**2)),final_cost=float(np.mean(residual(sol.x)[:2*v*n]**2)),focal_ratio=f,horizon_views=len(valid))

def extract(key):
    dest=OUT/'edges'/f'{key}.npz'
    if dest.exists():return dict(np.load(dest))
    case,trackdir,vgdir,rp,k,meta=g.provider(key,'train');axes=bg.frame_axes(rp);static=json.loads((ROOT/'runs/domain_v5/segmentation/model.json').read_text())['static_ids']
    @lru_cache(maxsize=100)
    def labels(i):
        p=ROOT/f'runs/domain_v5/segmentation/{key}/{i:04d}.png';lab=cv2.imread(str(p),0);assert lab is not None;return lab
    @lru_cache(maxsize=100)
    def horizon(i):return line_from_labels(labels(i))
    edges={mode:[] for mode in MODES+['initial']};details=[];tic=time.perf_counter()
    for path in sorted(vgdir.glob('*.npz')):
        st=int(path.stem)
        if st%32:continue
        vg=np.load(path)
        if len(vg['c2w'])<16:continue
        tracksfile=trackdir/f'{st:04d}.npz'
        if not tracksfile.exists():continue
        tr=np.load(tracksfile);p=tr['tracks'][LOCAL].astype(float);valid=tr['visible'][LOCAL].all(axis=0)
        if len(p)<4:continue
        for j,local in enumerate(LOCAL):
            lab=labels(st+local);xy=np.floor(p[j]/[meta['width'],meta['height']]*[lab.shape[1],lab.shape[0]]).astype(int)
            inside=(xy[:,0]>=0)&(xy[:,0]<lab.shape[1])&(xy[:,1]>=0)&(xy[:,1]<lab.shape[0]);ids=np.flatnonzero(inside)
            mask=cv2.erode(np.isin(lab,static).astype(np.uint8),np.ones((3,3),np.uint8));inside[ids]&=mask[xy[ids,1],xy[ids,0]]>0;valid&=inside
        p=p[:,valid];rec=dict(start=st,static_tracks=p.shape[1],methods={})
        if p.shape[1]<24:rec['reason']='few_static_tracks';details.append(rec);continue
        order=np.lexsort((p[0,:,1],p[0,:,0]));order=order[np.linspace(0,len(order)-1,min(96,len(order))).astype(int)];p=p[:,order]
        cells=len(np.unique(np.floor(p[0]/[meta['width'],meta['height']]*[8,4]),axis=0))
        if cells<4:rec['reason']='narrow_spatial_support';details.append(rec);continue
        r=vg['c2w'][LOCAL,:3,:3].copy();c=vg['c2w'][LOCAL,:3,3].copy();k=vg['intrinsics'][LOCAL].astype(float)
        new_h=round(meta['height']*518/meta['width']/14)*14;k[:,0,:]*=meta['width']/518;k[:,1,:]*=meta['height']/new_h
        x=triangulate(p,r,c,k);uv,z=project(x,r,c,k);okay=(z>0).all(axis=0)&(np.median(np.linalg.norm(uv-p,axis=2),axis=0)<20)
        p=p[:,okay];rec['triangulated_tracks']=p.shape[1]
        if p.shape[1]<24:rec['reason']='poor_initial_geometry';details.append(rec);continue
        held=np.arange(p.shape[1])%3==0;before,positive=audit(p[:,held],r,c,k);rec['before_px']=before
        h=[horizon(st+i)[0] for i in LOCAL];rec['horizons']=[horizon(st+i)[1] for i in LOCAL]
        def append(name,rr,weight):
            for a,b in [(0,1),(1,2),(2,3),(0,3)]:
                rot=rr[b].T@rr[a];edges[name].append([st+LOCAL[a],st+LOCAL[b],bg.relative_yaw(rot,rp,st+LOCAL[a],axes),weight])
        if before<=1.5 and positive>=.8:append('initial',r,1/(1+before))
        for mode in MODES:
            rr,cc,kk,diag=bundle(p[:,~held],r,c,k,focal='focal' in mode,lines=h if 'horizon' in mode else None)
            after,positive=audit(p[:,held],rr,cc,kk)
            accepted=bool(np.isfinite(after) and positive>=.8 and after<=1.5 and (after<=.9*before or (before<=.5 and after<=before+.05)) and .901<diag['focal_ratio']<1.099)
            diag.update(accepted=accepted,before_px=before,after_px=after,positive_fraction=positive);rec['methods'][mode]=diag
            if accepted:append(mode,rr,min(1,p.shape[1]/60)/(1+after))
        details.append(rec);print('WINDOW',key,st,'tracks',p.shape[1],{m:round(d['after_px'],3) for m,d in rec['methods'].items()},flush=True)
    arr={m:np.array(e).reshape(-1,4) for m,e in edges.items()};dest.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(dest,**arr)
    save_json(dest.with_suffix('.json'),dict(key=key,seconds=time.perf_counter()-tic,details=details,edges={m:len(e) for m,e in edges.items()}));return arr

def infer(keys):
    bg.guard();cv2.setNumThreads(1)
    for key in keys:
        edges=extract(key);case,*_=g.provider(key,'train');a=np.load(ROOT/f'runs/domain_v24_submission/train/signals/{key}.npz');b=np.load(ROOT/f'runs/domain_v29_submission/train/signals/{key}.npz')
        t,s=a['times'],a['speed'];assert np.array_equal(s,b['speed']);signals=dict(v24_control=a['yaw'],v29_control=b['yaw'])
        signals['vggt_initial_graph']=b['yaw']+bg.supported_graph(t,b['yaw'],edges['initial'],3)
        for base,z in [('v24',a),('v29',b)]:
            for mode in MODES:signals[base+'_'+mode]=z['yaw']+bg.supported_graph(t,z['yaw'],edges[mode],3)
        for name,y in signals.items():save_json(OUT/'predictions'/name/f'{key}.json',dict(case['pred'],positions=integrate_velocity(t,y,s).tolist()))
        dest=OUT/'signals'/f'{key}.npz';dest.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(dest,times=t,speed=s,**signals)
        save_json(OUT/'guards'/f'{key}.json',dict(reference_loading_blocked=True,scoring_blocked=True));print('INFER',key,flush=True)

def synthetic():
    rng=np.random.default_rng(30);x=rng.uniform([-2,-1,5],[2,1,12],(60,3));r=Rotation.from_euler('y',[0,.5,1,1.5],degrees=True).as_matrix();c=np.c_[np.linspace(0,.8,4),np.zeros((4,2))]
    k=np.tile(np.array([[800.,0,480],[0,800,270],[0,0,1.]]),(4,1,1));p,_=project(x,r,c,k);noisy=r.copy();noisy[1:]=Rotation.from_euler('y',[.2,-.2,.1],degrees=True).as_matrix()@noisy[1:]
    before,_=audit(p[:,::3],noisy,c,k);rr,cc,kk,d=bundle(p[:,np.arange(60)%3!=0],noisy,c,k,max_nfev=80);after,_=audit(p[:,::3],rr,cc,kk)
    assert after<before*.5,(before,after)
    lab=np.full((270,480),2,np.uint8);lab[140:]=26;line,diag=line_from_labels(lab);assert diag['accepted'] and abs(line[2]+279)<1e-5
    lab[130:145]=1;line,diag=line_from_labels(lab);assert line is None
    assert np.allclose(triangulate(p,r,c,k),x,atol=1e-7)
    save_json(REP/'synthetic.json',dict(passed=True,heldout_before_px=before,heldout_after_px=after,checks=['Projection and triangulation convention','Joint BA improves withheld-track reprojection','Direct sky-water line accepted','Building skyline rejected']))

def evaluate():
    from public_dataset import load_reference,load_timestamps,validate_prediction
    from metrics import score_trajectory
    from v20_five_pass import compare
    refs={k:load_reference(ROOT/'data/release',CLIPS[k]) for k in TRAIN};rows=[]
    for name in METHODS:
        per={}
        for key in TRAIN:
            p=json.loads((OUT/'predictions'/name/f'{key}.json').read_text());validate_prediction(p,key,load_timestamps(ROOT/'data/release',CLIPS[key]));per[key]=float(score_trajectory(np.array(p['times']),np.array(p['positions']),*refs[key])['drift_pct'])
        rows.append(dict(name=name,mean=float(np.mean(list(per.values()))),per_clip=per));print('SCORE',name,rows[-1]['mean'],flush=True)
    base=rows[1];assert abs(base['mean']-9.195614709280228)<1e-6
    for row in rows:
        row['comparison']=compare(row,base);d=row['comparison'];row['promote']=bool(d['improved']>=6 and d['worst_regression']<=.2 and d['mean_delta']<=-.1 and d['delta_excluding_best']<0)
    save_json(REP/'results.json',dict(rows=rows,scope='Repeated TRAIN8 development; no TEST evaluation or new submission.'))

def validate():
    plan=json.loads((REP/'plan.json').read_text())
    for p,h in {**plan['dependencies'],**plan['preserved']}.items():assert sha256(ROOT/p)==h,p
    for key in TRAIN:
        z=np.load(OUT/'signals'/f'{key}.npz');base=np.load(ROOT/f'runs/domain_v29_submission/train/signals/{key}.npz');assert np.array_equal(z['speed'],base['speed'])
        for name in METHODS:
            pred=json.loads((OUT/'predictions'/name/f'{key}.json').read_text());assert np.isfinite(pred['positions']).all();assert np.max(abs(np.array(pred['positions'])-integrate_velocity(z['times'],z[name],z['speed'])))<1e-9
    save_json(REP/'validation.json',dict(passed=True,preserved_submissions=len(plan['preserved']),predictions=len(TRAIN)*len(METHODS),speed_unchanged=True))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['freeze','synthetic','infer','evaluate','validate']);ap.add_argument('--keys');a=ap.parse_args();freeze()
    if a.action=='synthetic':synthetic()
    if a.action=='infer':infer(a.keys.split(',') if a.keys else TRAIN)
    if a.action=='evaluate':evaluate()
    if a.action=='validate':validate()
