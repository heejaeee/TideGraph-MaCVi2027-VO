"""Long-baseline static-image rotation constraints around frozen v2 speed.

Extends V17's 4s pairs with 8/16/32/64s pairs. This is not a full TALO or
VGGT-SLAM reproduction. No trajectory references enter this module.
"""
import sys,argparse,time,datetime
from pathlib import Path
import v17_depth_geometry as g
from v13_common import np,json,save_json,base_case,integrate_velocity,TRAIN
from public_dataset import sha256

ROOT=g.ROOT;OUT=ROOT/'runs/domain_v18';REP=ROOT/'reports/domain_v18'
OLD=ROOT/'runs/domain_v17_roma'
GAPS=[16,32,64,128]
CONFIGS=[
 dict(name='v2',kind='control',backend=None,cap=0),
 dict(name='v17_bias',kind='control',backend='bias',cap=3),
 dict(name='v17_graph',kind='control',backend='graph',cap=3),
 dict(name='long_depth_bias',kind='long',backend='bias',cap=3),
 dict(name='long_depth_graph',kind='long',backend='graph',cap=3),
 dict(name='joint_depth_bias',kind='joint',backend='bias',cap=3),
 dict(name='joint_depth_graph',kind='joint',backend='graph',cap=3),
 dict(name='joint_depth_graph_cap6',kind='joint',backend='graph',cap=6),
 dict(name='far_joint_depth_graph',kind='far',backend='graph',cap=3),
 dict(name='joint_essential_graph',kind='essential',backend='graph',cap=3),
 dict(name='cycle_joint_depth_bias',kind='cycle',backend='bias',cap=3),
 dict(name='cycle_joint_depth_graph',kind='cycle',backend='graph',cap=3),
]

def setup():
    g.OUT=OUT;g.REP=REP;g.FRONTEND='roma';g.DEPTH_FILTER='geometry_only'

def freeze():
    if (REP/'plan.json').exists():return
    save_json(REP/'plan.json',dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),configs=CONFIGS,
        hypothesis='Long image pair constraints reduce accumulated heading error beyond V17 short-pair correction; v2 speed fixed.',
        pair_design=dict(stride_frames=16,gaps_frames=GAPS,approx_seconds=[8,16,32,64],sampling_fps=2),
        matching='Official frozen RoMa v2 fast; 5000 samples, overlap>=0.5; same masks/depth/geometric gates as V17.',
        cycle_gate='Accepted SO(3) triangle closure median<=1 degree; require at least one measured triangle per edge. To enable cycle variant require >=8 long edges and >=50% occupied 30s midpoint bins; otherwise exact V17 same-backend fallback.',
        selection='Compare to V17 bias, not only v2: >=5/8 wins, worst regression<=0.5pp, mean delta excluding best gain<0.',
        scope='Repeated TRAIN8 development, not independent holdout. Parameters fixed before V18 scoring. Full TALO/VGGT-SLAM not reproduced. No test GT.',
        frozen={str(p.relative_to(ROOT)):sha256(p) for p in (ROOT/'submissions').glob('*/submission.json')},
        prior_code={str(p.relative_to(ROOT)):sha256(p) for p in (ROOT/'scripts').glob('v17*.py')}))

def cycles(arr,details,source='learned_depth_scale'):
    """SO(3) closure for i->j->k versus i->k; no base-pose labels."""
    valid={tuple(e[:2].astype(int)):n for n,e in enumerate(arr) if e[3]>.01}
    mats={tuple([d['i'],d['j']]):np.asarray(d['methods'][source]['r']) for d in details if tuple([d['i'],d['j']]) in valid}
    residuals=[[] for _ in arr];triangles=[]
    for (i,k),idx in valid.items():
        for j in range(i+16,k,16):
            if (i,j) not in valid or (j,k) not in valid:continue
            residual=float(np.rad2deg(g.Rotation.from_matrix(mats[(i,k)]@(mats[(j,k)]@mats[(i,j)]).T).magnitude()))
            for pair in [(i,j),(j,k),(i,k)]:residuals[valid[pair]].append(residual)
            triangles.append(dict(i=int(i),j=int(j),k=int(k),error_deg=residual))
    filtered=arr.copy();rows=[]
    for n,res in enumerate(residuals):
        median=float(np.median(res)) if res else None
        keep=bool(res and median<=1.)
        if not keep:filtered[n,3]=0
        rows.append(dict(i=int(arr[n,0]),j=int(arr[n,1]),triangles=len(res),median_deg=median,accepted=keep))
    return filtered,dict(triangles=triangles,edges=rows)

def unwrap_edges(arr,y):
    e=arr.copy();i=e[:,0].astype(int);j=e[:,1].astype(int);base=y[j]-y[i]
    e[:,2]=base+np.arctan2(np.sin(e[:,2]-base),np.cos(e[:,2]-base))
    return e

def coverage(t,arr):
    e=arr[arr[:,3]>.01]
    if not len(e):return dict(edges=0,occupied_fraction=0.)
    mid=(t[e[:,0].astype(int)]+t[e[:,1].astype(int)])/2
    bins=len(np.unique(np.floor((mid-t[0])/30).astype(int)))
    return dict(edges=len(e),occupied_fraction=bins/max(1,int(np.ceil((t[-1]-t[0])/30))))

def infer(key,split):
    setup();arr=g.extract(key,split);details=json.loads((OUT/'edges'/split/f'{key}.json').read_text())['details']
    case,y,s=base_case(key,split);old=np.load(OLD/split/'signals'/f'{key}.npz');short=np.load(OLD/'edges'/split/f'{key}.npz')
    cycle,cycle_diag=cycles(arr['learned_depth_scale'],details)
    obs=coverage(case['t'],cycle);cycle_on=obs['edges']>=8 and obs['occupied_fraction']>=.5
    signals={};diagnostics={}
    for cfg in CONFIGS:
        name,kind,backend=cfg['name'],cfg['kind'],cfg['backend']
        fallback=False;support=None
        if kind=='control':
            pred_y=y.copy() if name=='v2' else old[f'learned_depth_scale_{backend}_cap3'].copy()
        else:
            source='learned_essential' if kind=='essential' else 'learned_depth_scale'
            longs=cycle.copy() if kind=='cycle' else arr[source].copy()
            if kind=='far':longs=longs[longs[:,1]-longs[:,0]>=64]
            support=coverage(case['t'],longs)
            if kind=='cycle' and not cycle_on:
                pred_y=old[f'learned_depth_scale_{backend}_cap3'].copy();fallback=True
            else:
                edges=longs if kind=='long' else np.r_[short[source],longs]
                edges=unwrap_edges(edges,y)
                pred_y=y+g.correction(case['t'],y,edges,backend,cfg['cap'])
        signals[name]=pred_y
        diagnostics[name]=dict(fallback=fallback,long_support=support,max_correction_deg=float(np.max(abs(np.rad2deg(pred_y-y)))))
        save_json(OUT/split/'predictions'/name/f'{key}.json',{**case['pred'],'positions':integrate_velocity(case['t'],pred_y,s).tolist()})
    dest=OUT/split/'signals'/f'{key}.npz';dest.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(dest,times=case['t'],speed=s,**signals)
    save_json(OUT/split/'diagnostics'/f'{key}.json',dict(key=key,split=split,cycle_enabled=cycle_on,cycle_support=obs,methods=diagnostics,cycle=cycle_diag,
        gap_support={str(gap):coverage(case['t'],arr['learned_depth_scale'][arr['learned_depth_scale'][:,1]-arr['learned_depth_scale'][:,0]==gap]) for gap in GAPS}))
    print('INFER',split,key,'cycle',cycle_on,obs,flush=True)

def synthetic():
    # Correct rotation composition must close, a perturbed direct edge must not.
    r1=g.Rotation.from_euler('xyz',[.1,.2,.3]).as_matrix();r2=g.Rotation.from_euler('xyz',[-.2,.1,.05]).as_matrix()
    pairs=[(0,16),(16,32),(0,32)];rs=[r1,r2,r2@r1];arr=np.array([[i,j,0.,1.] for i,j in pairs])
    def records(mats):return [dict(i=i,j=j,methods={'learned_depth_scale':dict(r=r.tolist())}) for (i,j),r in zip(pairs,mats)]
    out,d=cycles(arr,records(rs));assert (out[:,3]>0).all() and d['triangles'][0]['error_deg']<1e-8
    rs[2]=g.Rotation.from_euler('y',4,degrees=True).as_matrix()@rs[2]
    out,d=cycles(arr,records(rs));assert (out[:,3]==0).all()
    # Measurement wrapping must not introduce a full-turn correction.
    y=np.array([0.,3.2]);e=unwrap_edges(np.array([[0,1,3.2-2*np.pi,1.]]),y);assert abs(e[0,2]-3.2)<1e-10
    # Known 2-degree accumulating error: long constraints reduce heading RMSE.
    t=np.arange(601)*.5;y=np.deg2rad(2)*t/t[-1]
    e=np.array([[i,i+128,0.,1.] for i in range(0,473,16)])
    delta=g.correction(t,y,e,'graph',3)
    before=float(np.rad2deg(np.sqrt(np.mean(y*y))))
    after=float(np.rad2deg(np.sqrt(np.mean((y+delta)**2))))
    # The inherited graph has a quadratic zero-correction prior; it should
    # improve the trajectory, not recover the unregularized solution exactly.
    assert after<.8*before
    assert np.array_equal(g.correction(t,y,np.empty((0,4)),'graph',3),np.zeros(len(t)))
    save_json(REP/'synthetic.json',dict(passed=True,heading_rmse_before_deg=before,heading_rmse_after_deg=after,checks=['SO3 composition and inconsistent triangle rejection','Angle wrapping','Known accumulated heading error reduction','Empty-observation fallback']))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--split',default='train',choices=['train','test']);ap.add_argument('--keys');ap.add_argument('--infer-only',action='store_true');ap.add_argument('--synthetic-only',action='store_true');args=ap.parse_args()
    setup();freeze();synthetic();g.cv2.setNumThreads(1)
    if args.synthetic_only:return
    if args.keys:keys=args.keys.split(',')
    elif args.split=='train':keys=TRAIN
    else:
        from v6_test_common import KEYS
        keys=KEYS
    if not args.infer_only:
        sys.path.insert(0,str(ROOT/'repos/RoMaV2-v17/src'))
        import torch
        from romav2 import RoMaV2
        torch.set_num_threads(2);torch.set_float32_matmul_precision('highest')
        model=RoMaV2(RoMaV2.Cfg(setting='fast'))
    for key in keys:
        if not args.infer_only:
            case,_,_,_,_,meta=g.provider(key,args.split)
            frame_root=ROOT/('runs/domain_v5/frames' if args.split=='train' else 'runs/domain_v6_test/frames')/key
            pairs=[(i,i+gap) for i in range(0,len(case['t']),16) for gap in GAPS if i+gap<len(case['t'])]
            tic=time.perf_counter()
            for n,(i,j) in enumerate(pairs):
                dest=OUT/'matches'/args.split/key/f'{i:04d}_{j:04d}.npz'
                if dest.exists():continue
                torch.manual_seed(i*1000+j)
                with torch.inference_mode():
                    pred=model.match(frame_root/f'{i:04d}.png',frame_root/f'{j:04d}.png')
                    matches,overlap,*_=model.sample(pred,5000)
                    p,q=model.to_pixel_coordinates(matches,meta['height'],meta['width'],meta['height'],meta['width'])
                    ok=overlap.reshape(-1)>=.5;p=p[ok].cpu().numpy();q=q[ok].cpu().numpy()
                dest.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(dest,i=i,j=j,p=p,q=q)
                if n%40==0:print('MATCH',key,n+1,len(pairs),'points',len(p),'seconds',round(time.perf_counter()-tic,1),flush=True)
            save_json(OUT/'match_timing'/args.split/f'{key}.json',dict(seconds=time.perf_counter()-tic,pairs=len(pairs),note='Includes cache skips when resumed; not end-to-end FPS.'))
        infer(key,args.split)
    plan=json.loads((REP/'plan.json').read_text())
    for p,h in {**plan['frozen'],**plan['prior_code']}.items():assert sha256(ROOT/p)==h,p

if __name__=='__main__':main()
