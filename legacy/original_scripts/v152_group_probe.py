"""Frozen image-only FLIR grouping; single A gain120 probe from entry19182."""
import asyncio, copy, json, re
from pathlib import Path
import cv2, numpy as np
import v101_protocol as p
import v150_flir512_submission as v
from pack_submission import pack
ROOT=p.ROOT
REP=ROOT/'reports/domain_v152_group_probe'
OUT=ROOT/'runs/domain_v152_group_probe'
DEST=ROOT/'submissions/macvi27_vo_v152_FLIR512_groupA_gain120'
BASE=v.DEST/'submission.json'
STATIC=[0,1,4,9,13,16,17,25,29,34,46,48,61,68,84,94]

def main():
    p.guard()
    assert p.sha(BASE)=='89d22b3ffcbe1686582b44549b1c2c0dcd4b8cc73818b43cca4e8db20e4a4364'
    plan=REP/'plan.json'
    if not plan.exists():
        p.save(plan,dict(created_utc=p.now(),baseline_entry=19182,baseline_score=11.33,
            score_source='User-reported official result; result JSON not received or independently read.',
            baseline_sha256=p.sha(BASE),group_rule='Rank descending median land/building semantic pixel fraction in upper70percent, central80percent of image, uniform frame indices0:16:600. A=top6, B=remaining5; ties by key. Existing frozen Mask2Former labels only. No GT, predictions or leaderboard perclip feedback used to group.',
            static_ids=STATIC,change='A gain1.20, B gain1.10; input512 A1 sigma2 and exact speeds retained. Wide8 IR1 unchanged.',
            supersedes='Cancel global512 gain1.20 followup in V150 plan; original historical plan preserved.',
            next={'improved_beyond_0.01':'Adopt A120 and next lower B to1.00 on adopted baseline','worse_beyond_0.01':'Revert19182 and next A1.00','within_0.01':'Retain19182 and next B1.00'},
            preserved={str(f.relative_to(ROOT)):p.sha(f) for f in (ROOT/'submissions').glob('*/submission.json')}))
    base=p.read(BASE);rows=[];mask_hashes={}
    for k in sorted(v.KEYS):
        oldmeta=p.read(ROOT/f'runs/domain_v6_test/frames/{k}/index.json')
        _,meta,_=v.inputs(k)
        assert oldmeta['frame_indices']==meta['frame_indices'], k
        fractions=[]
        for i in range(0,600,16):
            candidates=[ROOT/f'runs/domain_v6_test/segmentation/{k}/{i:04d}.png',ROOT/f'runs/domain_v16/test/extra_masks/{k}/{i:04d}.png',ROOT/f'runs/domain_v65_submission/test/masks/{k}/{i:04d}.png']
            path=next((f for f in candidates if f.exists()),None)
            assert path is not None,(k,i)
            lab=cv2.imread(str(path),0);h,w=lab.shape
            fractions.append(float(np.isin(lab[:int(.7*h),int(.1*w):int(.9*w)],STATIC).mean()))
            mask_hashes[str(path.relative_to(ROOT))]=p.sha(path)
        rows.append(dict(key=k,median_land_fraction=float(np.median(fractions)),frame_fractions=fractions,duration=float(meta['times'][-1]-meta['times'][0])))
    rows.sort(key=lambda r:(-r['median_land_fraction'],r['key']))
    a=[r['key'] for r in rows[:6]];b=[r['key'] for r in rows[6:]]
    grouping=dict(A=a,B=b,rows=rows,masks=mask_hashes,plan_sha256=p.sha(plan))
    gp=REP/'grouping.json'
    if gp.exists():assert p.read(gp)==grouping
    else:p.save(gp,grouping)
    bundle=copy.deepcopy(base);checks=[]
    for k in a:
        z=np.load(v.OUT/f'signals/{k}.npz');t=z['times'];s=z['speed']
        y=v.yaw(t,z['raw_A1'],z['yaw'][0],1.1)
        np.testing.assert_allclose(y,z['yaw'],atol=1e-10,rtol=0)
        np.testing.assert_allclose(p.integrate(t,y,s),base[k]['positions'],atol=1e-8,rtol=0)
        y=v.yaw(t,z['raw_A1'],z['yaw'][0],1.2)
        bundle[k]['positions']=p.integrate(t,y,s).tolist()
        p.npzsave(OUT/f'signals/{k}.npz',times=t,yaw=y,speed=s)
    assert sorted(k for k in bundle if bundle[k]!=base[k])==sorted(a)
    for k,pred in bundle.items():
        ts=p.load_timestamps(v.DATA,v.CLIPS[k]);p.validate_prediction(pred,k,ts)
        t=np.asarray(pred['times']);cov=p.coverage(t,ts);gap=float(np.diff(t).max())
        assert cov>=.95 and gap<=1
        for f in ['key','frame_indices','times']:assert pred[f]==base[k][f]
        if k not in a:assert pred==base[k]
        p.save(OUT/f'predictions/{k}.json',pred)
        checks.append(dict(key=k,changed=k in a,coverage=cov,max_gap_s=gap))
    for f,h in p.read(plan)['preserved'].items():assert p.sha(ROOT/f)==h
    DEST.mkdir(parents=True,exist_ok=True);dest=DEST/'submission.json'
    if not dest.exists():pack(v.DATA/'release_manifest.json',OUT/'predictions',dest)
    assert p.read(dest)==bundle
    desc='Monocular maritime visual odometry using video, timestamps and calibration. Frozen VGGT Omega with 16 frame windows and 8 frame overlap uses input resolution 512 for FLIR visible and 416 for wide clips. FLIR uses an all pairs robust yaw graph, 2 second Gaussian smoothing and rotation gains 1.20 for six clips with higher land visibility and 1.10 for the remaining five. Groups are fixed by pretrained semantic masks without test ground truth. Wide clips retain TV10 water flow speed. Infrared retains the previous pretrained rotation and KLT fusion. External pretrained weights are used. Global settings were selected using development and leaderboard feedback.'
    assert re.fullmatch(r'[A-Za-z0-9 \-:,.()\[\]_/\\+*%$§!?]+',desc)
    (DEST/'description.txt').write_text(desc+'\n')
    p.save(DEST/'validation.json',dict(passed=True,checks=checks,changed=6,unchanged_exact=14,server_scoreability_not_locally_observable=True))
    p.save(DEST/'provenance.json',dict(baseline_entry=19182,baseline_score=11.33,score_source='user report',baseline_sha256=p.sha(BASE),sha256=p.sha(dest),A=a,B=b,gains={'A':1.2,'B':1.1},grouping_sha256=p.sha(gp),plan_sha256=p.sha(plan),candidate_test_score=None,model_name='TideGraph-512',hardware='NVIDIA RTX 5080',metadata=True,additional_datasets=True,group_delta_formula='(overall score -11.33)*20/6',challenge_uploaded=False))
    p.save(REP/'current_best.json',dict(entry=19182,score=11.33,source='user reported; official JSON not locally verified',sha256=p.sha(BASE),path=str(BASE),global_gain120_cancelled=True))
    print(json.dumps({'A':a,'B':b,'rows':[(r['key'],r['median_land_fraction']) for r in rows],'sha256':p.sha(dest),'submission':str(dest)},indent=2),flush=True)

if __name__=='__main__':main()
