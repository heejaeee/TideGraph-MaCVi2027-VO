# V38 algorithm preserved; only split routing, output paths and selected mode differ.
"""Connected-map global BA of long static SIFT tracks, optionally deep tracks.

All registered cameras and shared landmarks in a connected component are solved
jointly. Local feed-forward camera priors are not mistaken for measured points.
"""
import argparse,datetime,sys,time,types,importlib.util
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'.envs/global_ba_packages'))
import pycolmap,pyceres
import v33_long_ba as local
from v13_common import TRAIN,CLIPS,np,json,save_json,integrate_velocity
from public_dataset import sha256
bg=local.bg;g=bg.g;cv2=local.cv2;Rotation=local.Rotation
SPLIT='train'
OUT=ROOT/'runs/domain_v42_global_ba'/SPLIT;REP=ROOT/'reports/domain_v42_submission'
MODES=['mixed_fixed']
METHODS=['v29_control','v37_control']+[m+'_'+kind for m in MODES for kind in ['rotation','course','rotation_course']]

def initial(key):
    case,td,vg,rp,k,meta=g.provider(key,SPLIT);z=np.load(ROOT/f'runs/domain_v29_submission/{SPLIT}/signals/{key}.npz');t=z['times'];raw=local.motion.planar_yaw(rp);f,right=bg.frame_axes(rp);up=np.cross(f,right)
    r=Rotation.from_rotvec((z['graph_yaw']-raw)[:,None]*up).as_matrix()@rp[:,:3,:3]
    p=integrate_velocity(t,z['yaw'],z['speed']);c=p[:,0,None]*right+p[:,2,None]*f
    return case,t,r,c,k,meta,td,z,(f,right)

class Tracks:
    def __init__(self):self.parent={};self.members={};self.conflicts=0
    def find(self,a):
        if a not in self.parent:self.parent[a]=a;self.members[a]={a[0]:a}
        if self.parent[a]!=a:self.parent[a]=self.find(self.parent[a])
        return self.parent[a]
    def merge(self,a,b):
        a=self.find(a);b=self.find(b)
        if a==b:return
        aa,bb=self.members[a],self.members[b]
        if any(aa[k]!=bb[k] for k in aa.keys()&bb.keys()):self.conflicts+=1;return
        if len(aa)<len(bb):a,b=b,a;aa,bb=bb,aa
        self.parent[b]=a;aa.update(bb);del self.members[b]

def prepare(key):
    dest=OUT/'tracks'/f'{key}.json'
    if dest.exists():return json.loads(dest.read_text())
    case,t,r,c,k,meta,td,z,axes=initial(key);union=Tracks();pairs=[];failed_pairs=[];tic=time.perf_counter();cv2.setNumThreads(1)
    for path in sorted((ROOT/f'runs/domain_v27_sift/matches/{SPLIT}/{key}').glob('*.npz')):
        d=np.load(path);i,j=int(d['i']),int(d['j']);p,q=d['p'],d['q']
        if len(p)<20:continue
        cv2.setRNGSeed(i*1000+j)
        try:fund,mask=cv2.findFundamentalMat(p,q,cv2.USAC_MAGSAC,1.5,.999,10000)
        except cv2.error as exc:
            failed_pairs.append(dict(i=i,j=j,reason='degenerate_fundamental_fit',error=str(exc)));continue
        if fund is None or mask is None:continue
        good=mask.ravel()>0
        if good.sum()<16 or good.mean()<.4:continue
        for a,b in zip(p[good],q[good]):union.merge((i,float(a[0]),float(a[1])),(j,float(b[0]),float(b[1])))
        pairs.append(dict(i=i,j=j,inliers=int(good.sum())))
    long=[]
    for member in union.members.values():
        obs=sorted(member.values());idx=[a[0] for a in obs]
        if len(obs)<3 or t[idx[-1]]-t[idx[0]]<16:continue
        if len(obs)>16:obs=[obs[i] for i in np.linspace(0,len(obs)-1,16).astype(int)]
        long.append(obs)
    long.sort(key=lambda obs:(-len(obs),-(obs[-1][0]-obs[0][0]),obs[0]));long=long[:2000]
    static=json.loads((ROOT/'runs/domain_v5/segmentation/model.json').read_text())['static_ids'];deep=[]
    @lru_cache(maxsize=100)
    def mask(i):
        folder='domain_v5' if SPLIT=='train' else 'domain_v6_test'
        path=ROOT/f'runs/{folder}/segmentation/{key}/{i:04d}.png'
        if SPLIT=='test' and not path.exists():path=ROOT/f'runs/domain_v16/test/extra_masks/{key}/{i:04d}.png'
        lab=cv2.imread(str(path),0);assert lab is not None,str(path)
        return cv2.erode(np.isin(lab,static).astype(np.uint8),np.ones((3,3),np.uint8))
    for path in sorted(td.glob('*.npz')):
        st=int(path.stem)
        if st%16:continue
        tr=np.load(path);local_ids=[0,8,16,24]
        if len(tr['tracks'])<=24:continue
        p=tr['tracks'][local_ids];valid=tr['visible'][local_ids].all(0)
        for j,off in enumerate(local_ids):
            m=mask(st+off);xy=np.floor(p[j]/[meta['width'],meta['height']]*[m.shape[1],m.shape[0]]).astype(int);inside=(xy[:,0]>=0)&(xy[:,0]<m.shape[1])&(xy[:,1]>=0)&(xy[:,1]<m.shape[0]);ix=np.flatnonzero(inside);inside[ix]&=m[xy[ix,1],xy[ix,0]]>0;valid&=inside
        p=p[:,valid]
        if p.shape[1]==0:continue
        ids=np.lexsort((p[0,:,1],p[0,:,0]));ids=ids[np.linspace(0,len(ids)-1,min(48,len(ids))).astype(int)]
        for a in ids:deep.append([[st+off,float(p[j,a,0]),float(p[j,a,1])] for j,off in enumerate(local_ids)])
    result=[];stats={}
    for family,source in [('sift',long),('deep',deep)]:
        kept=[]
        for obs in source:
            obs=np.array(obs);idx=obs[:,0].astype(int);pix=obs[:,1:];rr=r[idx];cc=c[idx];kk=np.tile(k,(len(idx),1,1))
            xyz=local.old.triangulate(pix[:,None],rr,cc,kk)[0];uv,depth=local.old.project(xyz[None],rr,cc,kk);err=np.linalg.norm(uv[:,0]-pix,axis=1)
            if (depth>0).mean()<.8 or np.median(err)>=12 or not np.isfinite(xyz).all():continue
            okay=(depth[:,0]>0)&(err<=20);obs=obs[okay]
            if len(obs)<3:continue
            if family=='sift' and t[int(obs[-1,0])]-t[int(obs[0,0])]<16:continue
            kept.append(dict(family=family,obs=obs.tolist(),xyz=xyz.tolist(),initial_median_px=float(np.median(err))))
        kept.sort(key=lambda d:(d['obs'][0][0],d['obs'][0][1],d['obs'][0][2]))
        for i,d in enumerate(kept):d['heldout']=i%5==0
        result+=kept;stats[family]=dict(raw=len(source),retained=len(kept),heldout=sum(d['heldout'] for d in kept))
    out=dict(key=key,tracks=result,stats=stats,verified_pairs=pairs,failed_pairs=failed_pairs,merge_conflicts=union.conflicts,seconds=time.perf_counter()-tic);save_json(dest,out);print('TRACKS',key,stats,flush=True);return out

def components(tracks):
    graph=defaultdict(set)
    for d in tracks:
        ids=[int(o[0]) for o in d['obs']]
        for i in ids:graph[i].update(ids)
    unseen=set(graph);out=[]
    while unseen:
        stack=[min(unseen)];seen=set()
        while stack:
            i=stack.pop()
            if i in seen:continue
            seen.add(i);unseen.discard(i);stack.extend(graph[i]-seen)
        out.append(sorted(seen))
    return sorted(out,key=len,reverse=True)

def reconstruction(tracks,ids,r,c,k,meta):
    rec=pycolmap.Reconstruction();rec.add_camera_with_trivial_rig(pycolmap.Camera(camera_id=1,model='PINHOLE',width=meta['width'],height=meta['height'],params=[k[0,0],k[1,1],k[0,2],k[1,2]]))
    obs=defaultdict(list);links=[]
    for d in tracks:
        tr=pycolmap.Track()
        for i,x,y in d['obs']:
            i=int(i);tr.add_element(i+1,len(obs[i]));obs[i].append([x,y])
        links.append((d['xyz'],tr))
    for i in ids:
        im=pycolmap.Image(image_id=i+1,camera_id=1,name=f'{i:04d}.png')
        im.points2D=pycolmap.Point2DList([pycolmap.Point2D(np.array(p)) for p in obs[i]])
        rec.add_image_with_trivial_frame(im,pycolmap.Rigid3d(pycolmap.Rotation3d(r[i].T),-r[i].T@c[i]))
    for xyz,tr in links:rec.add_point3D(xyz,tr)
    return rec

def solve(rec,mode):
    if mode=='gluemap_real_cpu':
        # Official no-virtual path only. The upstream C++ CPU branch is exactly
        # ceres::Solve; this explicit adapter delegates to pyceres.solve.
        adapter=types.ModuleType('pygluemap');adapter.solve_cuda=pyceres.solve
        sys.modules['pygluemap']=adapter
        spec=importlib.util.spec_from_file_location('gluemap_official_real_ba',ROOT/'repos/gluemap/gluemap/estimators/augmented_bundle_adjustment.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        rec,_,s=module.bundle_adjustment(rec,None,{},max_num_iterations=200)
        return rec,dict(usable=bool(s.IsSolutionUsable()),report=s.BriefReport(),iterations=int(s.num_successful_steps+s.num_unsuccessful_steps),official_real_track_function=True,cpu_solver_adapter=True)
    opts=pycolmap.BundleAdjustmentOptions();opts.refine_focal_length=mode.endswith('focal');opts.refine_principal_point=False;opts.refine_extra_params=False;opts.print_summary=False;opts.ceres.loss_function_type=pycolmap.LossFunctionType.HUBER;opts.ceres.loss_function_scale=1.;opts.ceres.solver_options.max_num_iterations=200;opts.ceres.solver_options.num_threads=1
    cfg=pycolmap.BundleAdjustmentConfig()
    for i in rec.images:cfg.add_image(i)
    cfg.fix_gauge(pycolmap.BundleAdjustmentGauge.TWO_CAMS_FROM_WORLD)
    ba=pycolmap.create_default_ceres_bundle_adjuster(opts,cfg,rec)
    if opts.refine_focal_length:
        params=rec.cameras[1].params
        for j in [0,1]:ba.problem.set_parameter_lower_bound(params,j,float(params[j]*.8));ba.problem.set_parameter_upper_bound(params,j,float(params[j]*1.2))
    s=ba.solve();cs=s.ceres_summary
    return rec,dict(usable=bool(s.is_solution_usable()),report=s.brief_report(),iterations=int(cs.num_successful_steps+cs.num_unsuccessful_steps),official_real_track_function=False,cpu_solver_adapter=False)

def poses(rec,ids,r,c,k):
    rr=r.copy();cc=c.copy();kk=k.copy();param=rec.cameras[1].params;kk[0,0],kk[1,1]=param[:2]
    for i in ids:
        pose=rec.images[i+1].cam_from_world();rr[i]=pose.rotation.matrix().T;cc[i]=-rr[i]@pose.translation
    return rr,cc,kk

def audit(tracks,r,c,k):
    errors=[];positive=[];per=defaultdict(list)
    for d in tracks:
        p=np.array(d['obs']);idx=p[:,0].astype(int);pix=p[:,1:];rr,cc=r[idx],c[idx];kk=np.tile(k,(len(idx),1,1))
        x=local.old.triangulate(pix[:,None],rr,cc,kk);uv,z=local.old.project(x,rr,cc,kk);err=np.linalg.norm(uv[1:-1,0]-pix[1:-1],axis=1)
        if np.isfinite(err).all():
            errors.extend(err.tolist());positive.extend((z>0).ravel().tolist())
            for i,e in zip(idx[1:-1],err):per[int(i)].append(float(e))
    return dict(median_px=float(np.median(errors)) if errors else 1e9,positive_fraction=float(np.mean(positive)) if positive else 0.,tracks=len(tracks),per_camera={str(i):dict(count=len(e),median=float(np.median(e))) for i,e in per.items()})

def geometry(key,mode):
    dest=OUT/'geometry'/mode/f'{key}.json'
    if dest.exists():return json.loads(dest.read_text())
    data=prepare(key);case,t,r,c,k,meta,td,z,axes=initial(key);tracks=[d for d in data['tracks'] if mode!='sift_fixed' or d['family']=='sift'];fit=[d for d in tracks if not d['heldout']];held=[d for d in tracks if d['heldout']];edges=[];courses=[];details=[];tic=time.perf_counter()
    for n,ids in enumerate(components(fit)):
        selected=[d for d in fit if int(d['obs'][0][0]) in ids];hold=[d for d in held if all(int(o[0]) in ids for o in d['obs'])];d=dict(component=n,cameras=len(ids),frames=ids,span_seconds=float(t[ids[-1]]-t[ids[0]]),fit_tracks=len(selected),heldout_tracks=len(hold));details.append(d)
        if len(ids)<8 or len(selected)<40 or d['span_seconds']<30 or len(hold)<8:d['reason']='insufficient_connected_map';continue
        before=audit(hold,r,c,k);rec=reconstruction(selected,ids,r,c,k,meta);rec,solver=solve(rec,mode);rr,cc,kk=poses(rec,ids,r,c,k);after=audit(hold,rr,cc,kk);ratio=kk[[0,1],[0,1]]/k[[0,1],[0,1]]
        accepted=bool(solver['usable'] and np.isfinite(rr).all() and np.isfinite(cc).all() and min(ratio)>.8 and max(ratio)<1.2 and after['median_px']<=2 and after['positive_fraction']>=.8 and(after['median_px']<=.9*before['median_px'] or(before['median_px']<=.5 and after['median_px']<=before['median_px']+.05)))
        d.update(before=before,after=after,solver=solver,focal_ratio=ratio.tolist(),accepted=accepted)
        folder=OUT/'maps'/mode/key/f'{n:02d}';folder.mkdir(parents=True,exist_ok=True);rec.write(folder)
        np.savez_compressed(folder/'poses.npz',frames=ids,r=rr[ids],c=cc[ids],k=kk,initial_r=r[ids],initial_c=c[ids],initial_k=k)
        if not accepted:continue
        pointsets={i:set() for i in ids}
        for ix,tr in enumerate(selected):
            for o in tr['obs']:pointsets[int(o[0])].add(ix)
        supported={int(i) for i,a in after['per_camera'].items() if a['count']>=3 and a['median']<=3}
        f,right=axes
        yaw0=np.arctan2(r[:,:,2]@right,r[:,:,2]@f);yaw1=np.arctan2(rr[:,:,2]@right,rr[:,:,2]@f)
        for i in ids:
            if i%16 or i not in supported:continue
            for gap in [16,32,64,128]:
                j=i+gap
                if j not in supported:continue
                common=len(pointsets[i]&pointsets[j])
                if common<8:continue
                weight=min(1,common/40)/(1+max(after['per_camera'][str(i)]['median'],after['per_camera'][str(j)]['median']))
                change=float(bg.wrap((yaw1[j]-yaw1[i])-(yaw0[j]-yaw0[i])));edges.append([i,j,change,weight])
                v=cc[j]-cc[i];v0=c[j]-c[i];fraction=float(np.linalg.norm([v@right,v@f])/max(np.linalg.norm(v),1e-9));angle=local.motion.angle(v,v0)
                if fraction<.7 or angle>10:continue
                courses.append(dict(i=i,j=j,weight=weight,course=float(np.arctan2(v@right,v@f)),common_points=common,horizontal_fraction=fraction,angle_from_initial=angle))
        print('GLOBAL',key,mode,'cams',len(ids),'tracks',len(selected),'audit',before['median_px'],after['median_px'],'accepted',accepted,flush=True)
    out=dict(key=key,mode=mode,edges=edges,courses=courses,details=details,seconds=time.perf_counter()-tic);save_json(dest,out);return out

def infer(keys):
    bg.guard();cv2.setNumThreads(1)
    for key in keys:
        case,t,r,c,k,meta,td,z,axes=initial(key);y,s=z['yaw'],z['speed'];v37=np.load(ROOT/f'runs/domain_v33_long_ba/{SPLIT}/signals/{key}.npz')['focal_course'];signals=dict(v29_control=y,v37_control=v37);diags={}
        for mode in MODES:
            obs=geometry(key,mode);e=np.array(obs['edges']).reshape(-1,4)
            if len(e):e[:,2]+=y[e[:,1].astype(int)]-y[e[:,0].astype(int)]
            rot=y+bg.supported_graph(t,y,e,3);dc,cd=local.motion.fit_motion(t,y,s,obs['courses'],'course');both,bd=local.motion.fit_motion(t,rot,s,obs['courses'],'course')
            signals[mode+'_rotation']=rot;signals[mode+'_course']=y+dc;signals[mode+'_rotation_course']=rot+both;diags[mode]=dict(course=cd,combined=bd,rotation_edges=len(e))
        for name,yaw in signals.items():save_json(OUT/'predictions'/name/f'{key}.json',dict(case['pred'],positions=integrate_velocity(t,yaw,s).tolist()))
        path=OUT/'signals'/f'{key}.npz';path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,times=t,speed=s,**signals);save_json(OUT/'diagnostics'/f'{key}.json',dict(methods=diags,gt_loading_blocked=True,scoring_blocked=True));print('INFER',key,flush=True)

