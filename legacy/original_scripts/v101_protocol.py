"""Frozen TEST-composition development protocol. Oracles are evaluation-only.

Stages: freeze -> IR acquisition (separate process) / predict -> evaluate.
No model training, label-based gates or per-clip method selection.
"""
import argparse
import csv
import datetime
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'repos/lookout-vo-challenge/starter'))
import metrics
from public_dataset import load_reference, load_timestamps, validate_prediction
from fuse_headings import planar_yaw
from domain_fusion_v2 import relative_window_yaw

OUT = ROOT/'runs/domain_v101_protocol'
REP = ROOT/'reports/domain_v101_protocol'
DATA = ROOT/'data/release'
CLIPS = {c['key']: c for c in json.loads((DATA/'release_manifest.json').read_text())['clips']}
GROUPS = {
    'wide': [k for k,c in CLIPS.items() if c['split']=='val' and c['camera_mode']=='wide'],
    'flir-VIS': [k for k,c in CLIPS.items() if c['split']=='train' and c['camera_mode']=='flir-VIS'],
    'flir-IR': [k for k,c in CLIPS.items() if c['camera_mode']=='flir-IR'],
}
KEYS = sum(GROUPS.values(), [])
VISIBLE = GROUPS['wide'] + GROUPS['flir-VIS']
WEIGHTS = {'wide':8, 'flir-VIS':11, 'flir-IR':1}
METHODS = ['V87', 'constant_speed', 'remove_small_speed_correction', 'A1_adjacent_graph', 'A1_all_pairs_graph']
ORACLES = ['GT_course_V87_speed', 'V87_yaw_GT_speed', 'GT_course_GT_speed',
           'raw_GT_direction_V87_distance', 'V87_direction_raw_GT_distance', 'GT_resampled']


def read(p): return json.loads(Path(p).read_text())
def save(p, o):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(o,indent=2,allow_nan=False)+'\n')
def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def guard():
    def audit(event,args):
        if event=='open' and args and isinstance(args[0],(str,bytes)):
            s=str(args[0])
            if any(x in s for x in ['/references/','/evaluation_only/','v98_flir_training_targets']):
                raise RuntimeError('GT forbidden in V101 prediction: '+s)
    sys.addaudithook(audit)
def integrate(t,y,s):
    v=np.c_[np.sin(y),np.zeros(len(y)),np.cos(y)]*s[:,None]
    return np.vstack([np.zeros(3),np.cumsum(.5*(v[:-1]+v[1:])*np.diff(t)[:,None],axis=0)])
def npzsave(p,**kw):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(p,**kw)


def case(key):
    c=CLIPS[key]
    if key in GROUPS['flir-VIS']:
        pp=ROOT/f'runs/domain_v84_flir_dvlt/predictions/Omega16_only/{key}.json'
        zp=ROOT/f'runs/domain_v84_flir_dvlt/signals/Omega16_only/{key}.npz'
        with np.load(zp) as z: t,y,s=z['times'],z['yaw'],z['speed']
        sp=ROOT/f'runs/domain_v76_direction_guarded_speed/train/base/signals/{key}.npz'
        with np.load(sp) as z:
            uncorrected=z['speed'];expected=uncorrected*np.exp(z['direction_guarded_quarter'])
        np.testing.assert_allclose(s,expected,rtol=0,atol=1e-12)
        source=ROOT/'runs/domain_v81_vggt_ohm/omega_w16/train'
        dependencies=[pp,zp,sp]
    elif c['split']=='val':
        pp=ROOT/f'runs/domain_v82_omega_val/predictions/omega_only_visible/{key}.json'
        zp=ROOT/f'runs/domain_v82_omega_val/signals/omega_only_visible/{key}.npz'
        with np.load(zp) as z: t,y,s=z['times'],z['yaw'],z['speed']
        uncorrected=s.copy();source=ROOT/'runs/domain_v82_omega_val/omega'
        dependencies=[pp,zp]
    else:
        pp=OUT/f'ir_baseline/predictions/{key}.json';zp=OUT/f'ir_baseline/signals/{key}.npz'
        with np.load(zp) as z:t,y,s=z['times'],z['yaw'],z['speed']
        source=None;uncorrected=s.copy();dependencies=[pp,zp]
    pred=read(pp)
    assert np.array_equal(t,pred['times']),key
    err=float(np.max(abs(integrate(t,y,s)-np.array(pred['positions']))))
    assert err<1e-8,(key,err)
    return dict(key=key,clip=c,t=t,yaw=y,speed=s,uncorrected_speed=uncorrected,pred=pred,
                source=source,dependencies=dependencies,baseline_parity_error=err)


def freeze():
    guard()
    assert [len(GROUPS[x]) for x in WEIGHTS]==[21,8,5]
    assert not (REP/'plan.json').exists(),'Protocol already frozen; do not silently refreeze.'
    tm=read(ROOT/'data/lookout-vo-test-v0.1/release_manifest.json')
    assert {m:sum(c['camera_mode']==m for c in tm['clips']) for m in WEIGHTS}==WEIGHTS
    assert all(c['reference'] is None for c in tm['clips'])
    inv=[];deps={}
    for key in KEYS:
        try:c=case(key)
        except FileNotFoundError:
            assert key in GROUPS['flir-IR'] and CLIPS[key]['split']=='train'
            inv.append(dict(key=key,baseline_cached=False));continue
        for p in c['dependencies']:deps[str(p.relative_to(ROOT))]=sha(p)
        wins=list((c['source']/f'windows/{key}').glob('*.npz')) if key in VISIBLE else []
        for p in wins:deps[str(p.relative_to(ROOT))]=sha(p)
        inv.append(dict(key=key,baseline_cached=True,windows=len(wins),parity_error=c['baseline_parity_error']))
    fovs={}
    for group,kk in GROUPS.items():
        fovs[group]=[]
        for key in kk:
            c=CLIPS[key]['calibration'];fx,fy=c['fx'],c['fy'];w,h=c['width'],c['height']
            fovs[group].append(dict(key=key,hfov_deg=float(np.rad2deg(2*np.arctan(w/(2*fx)))),
                vfov_deg=float(np.rad2deg(2*np.arctan(h/(2*fy)))),provenance=c['provenance']))
    wc=CLIPS[GROUPS['wide'][0]]['calibration'];fc=CLIPS[GROUPS['flir-VIS'][0]]['calibration']
    ratio_x=(fc['width']/fc['fx'])/(wc['width']/wc['fx'])
    ratio_y=(fc['height']/fc['fy'])/(wc['height']/wc['fy'])
    for p in [Path(__file__),DATA/'release_manifest.json',ROOT/'repos/lookout-vo-challenge/starter/metrics.py',
              ROOT/'scripts/fuse_headings.py',ROOT/'scripts/domain_fusion_v2.py']:
        deps[str(p.relative_to(ROOT))]=sha(p)
    local=ROOT/'repos/lookout-vo-challenge/starter/metrics.py';remote=Path('/tmp/v101_official_metrics.py')
    assert local.read_bytes()==remote.read_bytes()
    save(REP/'plan.json',dict(created_utc=now(),groups=GROUPS,test_weights=WEIGHTS,methods=METHODS,oracles=ORACLES,
        primary='A1_all_pairs_graph, FLIR-only first. Adjacent graph is a projection/optimization control.',
        graph=dict(initial_edge_weight=1,huber_delta_degrees=3,max_irls_iterations=30,angle_tolerance_rad=1e-9,
                   smoothing_seconds=2,anchor='first frame zero; original initial heading restored',
                   projection='Each window aligned to baseline 3D chain at its first frame, then projected onto the fixed baseline vertical. All pairs of projected headings.',
                   correlation='Within-window edges are correlated and transitively redundant; they do not constitute independent observations.'),
        oracle=dict(smoothing_seconds=2,minimum_course_speed_mps=.2,low_speed='interpolate unwrapped valid course for diagnostic only; excluded from offset statistics',
                    speed='norm of differentiated smoothed reference position',coordinate='course atan2(east velocity,north velocity); integrate into XZ, official Sim3 handles gauge',
                    label_access='Evaluation only, not inference. Not an additive error budget or guaranteed achievable bound.',
                    comparison='Also report raw-step oracles compatible with the V98 diagnostic and GT reconstruction floor.'),
        adoption='FLIR wins >=6/8, FLIR mean improves, wide mean does not worsen, full coverage. No threshold search or per-clip best choice.',
        boats=dict(manifest_fields=sorted(next(iter(CLIPS.values())).keys()),boat_count=None,boat_folds_available=False,
                   reason='Public anonymous manifest has no boat/session/device identifier. Calibration and adjacent IDs cannot establish boat identity.'),
        crop_proxy=dict(horizontal_ratio=ratio_x,vertical_ratio=ratio_y,feasible=ratio_x<=1 and ratio_y<=1,
                        decision='Do not create a synthetic FLIR-FOV crop: released nominal FLIR VIS FOV is wider than wide. Calibration is nominal, not measured zoom.'),
        fovs=fovs,inventory=inv,dependencies=deps,official_metric=dict(sha256=sha(local),matches_current_upstream=True,
              source='https://raw.githubusercontent.com/LOOKOUT-AI/lookout-vo-challenge/main/starter/metrics.py',
              global_sim3=True,segment_m=list(metrics.SEGMENT_LENGTHS_M),start_stride=10,min_intervals=3),
        ir_acquisition='Four TRAIN IR lack V87 legacy caches. Reconstruct frozen R3/VGGT/KLT heading and constant speed with original inference adapters; no new method tuning.',
        limitations=['Reused development clips, not a fresh independent benchmark.',
                     'Baseline speed prior was previously trained using official TRAIN; this is not unbiased TRAIN generalization.',
                     'TEST composition weighting is a development proxy, not an official TEST score prediction.',
                     'Course-minus-Omega differences also include camera yaw error, pan and mount variation; cannot establish sideslip/current alone.'],
        preserved_submissions={str(p.relative_to(ROOT)):sha(p) for p in sorted((ROOT/'submissions').glob('*/submission.json'))}))
    print('V101_FROZEN',GROUPS,flush=True)


def chain_and_edges(source,key,n):
    windows=[];increments=[[] for _ in range(n-1)]
    for p in sorted((source/f'windows/{key}').glob('*.npz')):
        with np.load(p) as z:start=int(z['start']);r=z['c2w'][:,:3,:3]
        windows.append((start,r))
        dr=Rotation.from_matrix(r[:-1].transpose(0,2,1)@r[1:]).as_rotvec()
        for j,v in enumerate(dr):increments[start+j].append(v)
    assert all(len(v)>0 for v in increments),key
    R=np.tile(np.eye(3),(n,1,1))
    for i,rr in enumerate(increments):R[i+1]=R[i]@Rotation.from_rotvec(np.median(rr,axis=0)).as_matrix()
    vertical=np.median(R[:,:,1],axis=0);vertical/=np.linalg.norm(vertical)
    f=R[0,:,2]-vertical*(R[0,:,2]@vertical);f/=np.linalg.norm(f)
    right=np.cross(vertical,f);right/=np.linalg.norm(right)
    base=np.unwrap(np.arctan2(R[:,:,2]@right,R[:,:,2]@f))
    ii=[];jj=[];zz=[];window_ids=[]
    for wid,(start,local) in enumerate(windows):
        aligned=R[start]@local[0].T@local
        y=np.unwrap(np.arctan2(aligned[:,:,2]@right,aligned[:,:,2]@f))
        a,b=np.triu_indices(len(y),1);z=y[b]-y[a];i,j=a+start,b+start
        z+=2*np.pi*np.round(((base[j]-base[i])-z)/(2*np.pi))
        ii.extend(i);jj.extend(j);zz.extend(z);window_ids.extend([wid]*len(i))
    return base,np.asarray(ii),np.asarray(jj),np.asarray(zz),np.asarray(window_ids)


def solve_yaw(n,i,j,z,initial,delta_deg=3):
    """Huber IRLS of theta[j]-theta[i]=z; theta[0] fixed to zero."""
    ne=len(i);idx=np.arange(ne)
    A=coo_matrix((np.r_[-np.ones(ne),np.ones(ne)],(np.r_[idx,idx],np.r_[i,j])),shape=(ne,n)).tocsr()[:,1:]
    x=initial[1:]-initial[0];w=np.ones(ne);delta=np.deg2rad(delta_deg)
    for it in range(30):
        old=x.copy();Aw=A.multiply(w[:,None]);x=spsolve((A.T@Aw).tocsc(),A.T@(w*z))
        residual=A@x-z;w=np.minimum(1.,delta/np.maximum(abs(residual),1e-12))
        if np.max(abs(x-old))<1e-9:break
    assert np.isfinite(x).all()
    return np.r_[0.,x],dict(edges=ne,iterations=it+1,downweighted_fraction=float(np.mean(w<1-1e-8)),
        residual_median_deg=float(np.rad2deg(np.median(abs(residual)))),residual_p95_deg=float(np.rad2deg(np.quantile(abs(residual),.95))),
        final_step_rad=float(np.max(abs(x-old))),converged=bool(np.max(abs(x-old))<1e-9))


def selfcheck():
    # Independent known-angle constraints, recovery across wrap and one gross outlier.
    n=70;truth=np.linspace(0,4.5,n);i,j=np.triu_indices(n,1);z=truth[j]-truth[i]
    clean,d=solve_yaw(n,i,j,z,np.zeros(n));assert np.max(abs(clean-truth))<1e-9
    bad=z.copy();bad[40]+=.8
    fit,d=solve_yaw(n,i,j,bad,np.zeros(n));assert np.max(abs(fit-truth))<.003
    assert np.max(abs(fit-truth))<np.max(abs(solve_yaw(n,i,j,bad,np.zeros(n),delta_deg=180)[0]-truth))
    # Camera-to-world convention: positive y rotation projects to positive yaw.
    rs=Rotation.from_euler('y',truth).as_matrix();poses=np.tile(np.eye(4),(n,1,1));poses[:,:3,:3]=rs
    np.testing.assert_allclose(planar_yaw(poses),truth,atol=1e-10)
    save(REP/'solver_checks.json',dict(passed=True,exact_angle_error=float(np.max(abs(clean-truth))),
         outlier_max_error_deg=float(np.rad2deg(np.max(abs(fit-truth)))),checked='anchoring, angle wrap, rotation sign, robust outlier attenuation'))


def predict():
    guard();plan=read(REP/'plan.json')
    for p,h in plan['dependencies'].items():assert sha(ROOT/p)==h,p
    selfcheck();files={};rows=[]
    for key in VISIBLE:
        c=case(key);t=c['t'];n=len(t);base,i,j,z,wid=chain_and_edges(c['source'],key,n)
        np.testing.assert_allclose(base,relative_window_yaw(c['source'],key,n),atol=1e-12)
        by=gaussian_filter1d(base,2/np.median(np.diff(t)));by+=c['yaw'][0]-by[0]
        np.testing.assert_allclose(by,c['yaw'],rtol=0,atol=1e-10)
        yy={'V87':c['yaw'],'constant_speed':c['yaw'],'remove_small_speed_correction':c['yaw']};diags={}
        for name,mask in [('A1_adjacent_graph',j==i+1),('A1_all_pairs_graph',np.ones(len(i),bool))]:
            y,d=solve_yaw(n,i[mask],j[mask],z[mask],base)
            y=gaussian_filter1d(y,2/np.median(np.diff(t)));y+=c['yaw'][0]-y[0];yy[name]=y
            d['unique_windows']=len(set(wid[mask].tolist()));d['yaw_change_p95_deg']=float(np.rad2deg(np.quantile(abs(y-by),.95)))
            diags[name]=d
        for name,y in yy.items():
            speed=np.ones(n) if name=='constant_speed' else c['uncorrected_speed'] if name=='remove_small_speed_correction' else c['speed']
            pred=dict(c['pred'],positions=integrate(t,y,speed).tolist())
            if name=='V87':pred=c['pred']
            validate_prediction(pred,key,load_timestamps(DATA,c['clip']))
            p=OUT/f'predictions/{name}/{key}.json';save(p,pred);files[str(p.relative_to(ROOT))]=sha(p)
        npzsave(OUT/f'signals/{key}.npz',times=t,speed=c['speed'],**yy)
        save(REP/f'graph_diagnostics/{key}.json',diags)
        rows.append(dict(key=key,baseline_parity_error=c['baseline_parity_error'],max_gap_s=float(np.diff(t).max())))
        print('V101_PREDICTED',key,{k:round(v['yaw_change_p95_deg'],3) for k,v in diags.items()},flush=True)
    save(REP/'predictions_frozen.json',dict(created_utc=now(),files=files,reference_access_blocked=True,rows=rows))


def coverage(t,rt):
    hi=np.searchsorted(t,rt);ix=np.clip(hi,0,len(t)-1);lo=np.clip(hi-1,0,len(t)-1)
    exact=np.isclose(rt,t[ix],atol=1e-8,rtol=0)
    ok=exact|((hi>0)&(hi<len(t))&(t[ix]-t[lo]<=1.+1e-8))
    return float(ok.mean())


def score(key,name,t,p,rt,rp):
    raw=metrics.score_trajectory(t,p,rt,rp);good=raw is not None and np.isfinite(raw['drift_pct'])
    return dict(key=key,group=CLIPS[key]['camera_mode'],split=CLIPS[key]['split'],method=name,
        drift_pct=float(raw['drift_pct']) if good else None,scored=bool(good),coverage=coverage(t,rt),
        max_gap_s=float(np.diff(t).max()),path_length_m=float(raw['path_length_m']) if raw is not None else None,
        ate_m=float(raw['ate_m']) if good else None)


def oracle_paths(c,rt,rp):
    t,y,s=c['t'],c['yaw'],c['speed'];sigma=2/np.median(np.diff(rt))
    smooth=gaussian_filter1d(rp,sigma,axis=0,mode='nearest');vel=np.gradient(smooth,rt,axis=0)
    vt=metrics.resample_to(rt,vel,t);speed=np.linalg.norm(vt[:,:2],axis=1)
    valid=(speed>=.2)&(t>=rt[0]+2)&(t<=rt[-1]-2)
    assert valid.sum()>2,c['key']
    course=np.interp(t,t[valid],np.unwrap(np.arctan2(vt[valid,0],vt[valid,1])))
    paths={'GT_course_V87_speed':integrate(t,course,s),'V87_yaw_GT_speed':integrate(t,y,speed),
           'GT_course_GT_speed':integrate(t,course,speed)}
    C=metrics.resample_to(rt,rp,t);dp=np.diff(C,axis=0);lengths=np.linalg.norm(dp,axis=1);ids=np.flatnonzero(lengths>1e-6)
    d=np.column_stack([np.interp(np.arange(len(dp)),ids,dp[ids,j]/lengths[ids]) for j in range(3)])
    d/=np.maximum(np.linalg.norm(d,axis=1,keepdims=True),1e-12)
    v=np.c_[np.sin(y),np.zeros(len(y)),np.cos(y)];v=v[:-1]+v[1:];v/=np.maximum(np.linalg.norm(v,axis=1,keepdims=True),1e-12)
    paths['raw_GT_direction_V87_distance']=np.vstack([np.zeros(3),np.cumsum(d*(.5*(s[:-1]+s[1:])*np.diff(t))[:,None],axis=0)])
    paths['V87_direction_raw_GT_distance']=np.vstack([np.zeros(3),np.cumsum(v*lengths[:,None],axis=0)])
    paths['GT_resampled']=C
    # Evaluation-only global Sim3, never a correction to inference outputs.
    mask=(rt>=t[0])&(rt<=t[-1]);est=metrics.resample_to(t,np.array(c['pred']['positions']),rt[mask])
    _,R,_=metrics.umeyama(est,rp[mask],True)
    headings=np.c_[np.sin(y),np.zeros(len(y)),np.cos(y)]@R.T
    mapped=np.unwrap(np.arctan2(headings[:,0],headings[:,1]))
    diff=np.angle(np.exp(1j*(course-mapped)));offset=float(np.angle(np.mean(np.exp(1j*diff[valid]))))
    residual=np.angle(np.exp(1j*(diff-offset)));degrees=np.rad2deg(residual[valid])
    diag=dict(key=c['key'],group=c['clip']['camera_mode'],valid_course_fraction=float(valid.mean()),
        single_global_course_offset_deg=float(np.rad2deg(offset)),centered_p90_minus_p10_deg=float(np.ptp(np.quantile(degrees,[.1,.9]))),
        centered_abs_p95_deg=float(np.quantile(abs(degrees),.95)),gt_speed_cv=float(np.std(speed[valid])/np.mean(speed[valid])),
        v87_speed_cv=float(np.std(s)/max(abs(np.mean(s)),1e-9)),negative_v87_speed_fraction=float(np.mean(s<0)),
        interpretation='Diagnostic only: relative camera direction error, pan, current and sideslip are not identifiable separately.')
    npzsave(OUT/f'evaluation_only/course_signals/{c["key"]}.npz',times=t,gt_course=course,gt_speed=speed,
            v87_yaw=y,aligned_v87_yaw=mapped,course_minus_yaw=residual,valid_course=valid,v87_speed=s)
    return paths,diag


def summarize(rows,methods):
    summary={}
    for group,keys in GROUPS.items():
        baseline={r['key']:r['drift_pct'] for r in rows if r['group']==group and r['method']=='V87'}
        summary[group]={}
        for name in methods:
            rr=[r for r in rows if r['group']==group and r['method']==name];sc=[r for r in rr if r['scored']]
            if not rr:continue
            deltas=[r['drift_pct']-baseline[r['key']] for r in sc if baseline.get(r['key']) is not None]
            summary[group][name]=dict(mean_drift_pct=float(np.mean([r['drift_pct'] for r in sc])) if sc else None,
                mean_delta_pp=float(np.mean(deltas)) if deltas else None,improved=sum(d<-1e-8 for d in deltas),
                worsened=sum(d>1e-8 for d in deltas),ties=sum(abs(d)<=1e-8 for d in deltas),scored=len(sc),expected=len(keys),
                worst_delta_pp=max(deltas) if deltas else None,min_coverage=min(r['coverage'] for r in rr),
                leave_one_clip_out_delta_range=[float(min(np.mean(deltas[:i]+deltas[i+1:]) for i in range(len(deltas)))),
                                               float(max(np.mean(deltas[:i]+deltas[i+1:]) for i in range(len(deltas))))] if len(deltas)>1 else None)
    return summary


def evaluate():
    frozen=read(REP/'predictions_frozen.json')
    for p,h in frozen['files'].items():assert sha(ROOT/p)==h,p
    rows=[];offsets=[]
    for key in KEYS:
        c=case(key);rt,rp=load_reference(DATA,c['clip']);t=c['t']
        for name in METHODS:
            p=read(OUT/f'predictions/{name}/{key}.json') if key in VISIBLE else c['pred']
            rows.append(score(key,name,t,np.array(p['positions']),rt,rp))
        if key in VISIBLE:
            paths,d=oracle_paths(c,rt,rp);offsets.append(d)
            for name,p in paths.items():rows.append(score(key,name,t,p,rt,rp))
    summary=summarize(rows,METHODS+ORACLES)
    weighted={}
    for name in METHODS:
        for scope in ['flir_only','wide_only','both_visible'] if name.startswith('A1') else ['both_visible']:
            choice={g:name if g!='flir-IR' and (scope=='both_visible' or scope=='flir_only' and g=='flir-VIS' or scope=='wide_only' and g=='wide') else 'V87' for g in GROUPS}
            ss={g:summary[g][m] for g,m in choice.items()}
            complete=all(s['scored']==s['expected'] for s in ss.values())
            mean=sum(WEIGHTS[g]*s['mean_drift_pct'] for g,s in ss.items())/20 if complete else None
            weighted[name+'__'+scope]=dict(test_like_drift_pct=mean,methods=choice,all_expected_scored=complete,
                eligible_coverage=all(s['min_coverage']>=.95 for s in ss.values()))
    fl=summary['flir-VIS']['A1_all_pairs_graph'];wide=summary['wide']['A1_all_pairs_graph']
    adopt=fl['improved']>=6 and fl['mean_delta_pp']<0 and wide['mean_delta_pp']<=0 and all(r['scored'] and r['coverage']>=.95 for r in rows if r['method']=='A1_all_pairs_graph')
    save(REP/'results.json',dict(created_utc=now(),groups=summary,weighted=weighted,rows=rows,course_diagnostics=offsets,
        A1_adopt=adopt,official_test_score_available=False,new_submission_created=False,training_performed=False,
        baseline_ir='Original frozen V87 legacy rule applied to all five IR; no IR direction changes in A1 comparisons.'))
    with (REP/'per_clip.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0])+['delta_from_V87_pp']);writer.writeheader()
        base={r['key']:r['drift_pct'] for r in rows if r['method']=='V87'}
        for r in rows:writer.writerow(dict(r,delta_from_V87_pp=None if r['drift_pct'] is None else r['drift_pct']-base[r['key']]))
    for p,h in read(REP/'plan.json')['preserved_submissions'].items():assert sha(ROOT/p)==h,p
    save(REP/'integrity.json',dict(passed=True,preserved_submissions=len(read(REP/'plan.json')['preserved_submissions']),predictions_verified=len(frozen['files'])))
    print(json.dumps(dict(groups=summary,weighted=weighted,A1_adopt=adopt),indent=2),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['freeze','predict','evaluate','selfcheck']);a=ap.parse_args();globals()[a.action]()
