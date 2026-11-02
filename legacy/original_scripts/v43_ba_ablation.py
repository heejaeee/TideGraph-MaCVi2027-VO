"""V24-only global BA ablations with predeclared image-only observability gates.

Public TRAIN8 repeated development. No official test inference in this experiment.
"""
import argparse, datetime, html
from collections import Counter
import v42_global_core as core
from public_dataset import sha256
ROOT,np,json,save_json=core.ROOT,core.np,core.json,core.save_json
TRAIN,CLIPS=core.TRAIN,core.CLIPS
OUT=ROOT/'runs/domain_v43_ba_ablation'
REP=ROOT/'reports/domain_v43_ba_ablation'
GATES={
    'ungated':None,
    'observable':dict(points=16,spread_deg=.5,parallax_deg=.5,rank_ratio=.02,null_ratio=.3,converged=False),
    'strict':dict(points=24,spread_deg=1.,parallax_deg=1.,rank_ratio=.05,null_ratio=.2,converged=False),
    'converged':dict(points=16,spread_deg=.5,parallax_deg=.5,rank_ratio=.02,null_ratio=.3,converged=True),
}
KINDS=['rotation','course','both']
CANDIDATES=[f'{g}_{k}' for g in GATES for k in KINDS]
METHODS=['v24_control','v29_control','v42_control']+CANDIDATES


def initial(key):
    case,td,vg,rp,k,meta=core.g.provider(key,'train')
    z=np.load(ROOT/f'runs/domain_v24_submission/train/signals/{key}.npz')
    t=z['times'];f,right=core.bg.frame_axes(rp);up=np.cross(f,right)
    r=core.Rotation.from_rotvec((z['yaw']-core.local.motion.planar_yaw(rp))[:,None]*up).as_matrix()@rp[:,:3,:3]
    p=core.integrate_velocity(t,z['yaw'],z['speed']);c=p[:,0,None]*right+p[:,2,None]*f
    return case,t,r,c,k,meta,td,z,(f,right)


def configure():
    core.OUT=OUT;core.SPLIT='train';core.initial=initial;core.cv2.setNumThreads(1)


def freeze():
    path=REP/'plan.json'
    if path.exists():return
    deps={}
    for p in list((ROOT/'scripts').glob('*.py'))+[ROOT/'data/release/release_manifest.json']:
        deps[str(p.relative_to(ROOT))]=sha256(p)
    old=json.loads((ROOT/'reports/domain_v42_submission/plan.json').read_text())['dependencies']
    for rel,h in old.items():
        if rel.startswith('runs/') and '/test/' not in rel and 'domain_v6_test' not in rel:
            assert sha256(ROOT/rel)==h,rel;deps[rel]=h
    for key in TRAIN:
        for rel in [f'runs/domain_v24_submission/train/signals/{key}.npz',f'runs/domain_v29_submission/train/signals/{key}.npz',f'runs/domain_v42_global_ba/train/signals/{key}.npz']:
            deps[rel]=sha256(ROOT/rel)
    save_json(path,dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        gates=GATES,methods=METHODS,
        initialization='Recompute tracks and 200-iteration fixed-K global BA with V24 yaw and centers. No V29 background rotation/course in initialization.',
        rotation='BA relative rotation change, existing supported graph cap3deg.',
        course='BA camera displacement bearing, existing independent course fit cap3deg and >=50% temporal coverage. Same fixed V24 speed.',
        gate='Shared real fitted tracks in each interval. Rotation: count and minimum endpoint normalized bearing minor-axis spread. Course: additionally min(initial,optimized) compensated parallax and optimized epipolar-normal SVD rank/null ratios. No GT.',
        observability_details='Bearing angular spread is degrees(arctan(smallest sample standard deviation of normalized image xy)). Normals are unit cross products of world rays with >0.05deg separation. SVD s1/s0 must exceed rank_ratio, s2/s1 <=null_ratio. At least required count finite normals.',
        converged='Additional diagnostic requiring solver CONVERGENCE, excluding NO_CONVERGENCE.',
        scope='Repeated public TRAIN8. No test predictions, online upload or new submission. #18940 presumed V24, uploaded hash not available.',
        selection='All12 frozen candidates compared with V24. Screen >=6 improvements of >1e-8pp, mean delta<=-.1pp, worst regression<=.2pp, excluding best delta<0. No per-clip candidate routing.',
        exclusion_audit='Select on other7 with same proportional screen (>=ceil(.75*n),mean<=-.1,worst<=.2,excludingbest<0), or V24 fallback. Also twofold split by median input-only V24 total absolute yaw change. Not independent generalization: all clips previously exposed. Groups are turn conditions, not known recording sessions.',
        dependencies=deps,preserved={str(p.relative_to(ROOT)):sha256(p) for p in (ROOT/'submissions').glob('*/submission.json')}))


def ray_stats(p,q,ri,rj,k):
    a=np.c_[p,np.ones(len(p))]@np.linalg.inv(k).T
    b=np.c_[q,np.ones(len(q))]@np.linalg.inv(k).T
    def spread(x):
        xy=x[:,:2]/x[:,2,None]
        return float(np.rad2deg(np.arctan(np.linalg.svd(xy-xy.mean(0),compute_uv=False)[-1]/np.sqrt(max(1,len(xy)-1)))))
    spatial=min(spread(a),spread(b))
    a=a@ri.T;b=b@rj.T
    a/=np.linalg.norm(a,axis=1,keepdims=True);b/=np.linalg.norm(b,axis=1,keepdims=True)
    cross=np.cross(a,b);norm=np.linalg.norm(cross,axis=1)
    angle=np.rad2deg(np.arctan2(norm,(a*b).sum(1)))
    good=norm>np.sin(np.deg2rad(.05));sv=np.zeros(3)
    if good.sum()>=3:sv=np.linalg.svd(cross[good]/norm[good,None],compute_uv=False)
    return dict(spread_deg=spatial,parallax_deg=float(np.median(angle)),
        normals=int(good.sum()),rank_ratio=float(sv[1]/max(sv[0],1e-12)),null_ratio=float(sv[2]/max(sv[1],1e-12)))


def observations(key,geom):
    dest=OUT/'observability'/f'{key}.json'
    if dest.exists():return json.loads(dest.read_text())
    _,t,r,c,k,meta,td,z,axes=initial(key)
    tracks=core.prepare(key)['tracks'];fit=[d for d in tracks if not d['heldout']]
    result={}
    for component in geom['details']:
        if not component.get('accepted'):continue
        n=component['component'];ids=component['frames'];idset=set(ids)
        selected=[d for d in fit if int(d['obs'][0][0]) in idset]
        per={i:{} for i in ids}
        for ix,d in enumerate(selected):
            for frame,x,y in d['obs']:per[int(frame)][ix]=[x,y]
        poses=np.load(OUT/f'maps/mixed_fixed/{key}/{n:02d}/poses.npz')
        rr={int(i):v for i,v in zip(poses['frames'],poses['r'])}
        solver=component['solver']['report'];converged='NO_CONVERGENCE' not in solver and 'CONVERGENCE' in solver
        for i,j,_,_ in geom['edges']:
            i,j=int(i),int(j)
            if i not in idset or j not in idset:continue
            shared=sorted(per[i].keys()&per[j].keys())
            p=np.array([per[i][x] for x in shared]);q=np.array([per[j][x] for x in shared])
            before=ray_stats(p,q,r[i],r[j],k);after=ray_stats(p,q,rr[i],rr[j],poses['k'])
            result[f'{i}_{j}']=dict(points=len(shared),spread_deg=after['spread_deg'],
                parallax_deg=min(before['parallax_deg'],after['parallax_deg']),rank_ratio=after['rank_ratio'],
                null_ratio=after['null_ratio'],normals=after['normals'],converged=converged,
                initial_parallax_deg=before['parallax_deg'],optimized_parallax_deg=after['parallax_deg'],component=n)
    save_json(dest,result);return result


def accept(row,cfg,course):
    if cfg is None:return True
    if cfg['converged'] and not row['converged']:return False
    if row['points']<cfg['points'] or row['spread_deg']<cfg['spread_deg']:return False
    return not course or (row['parallax_deg']>=cfg['parallax_deg'] and row['normals']>=cfg['points']
        and row['rank_ratio']>=cfg['rank_ratio'] and row['null_ratio']<=cfg['null_ratio'])


def infer(keys):
    core.bg.guard();configure()
    for key in keys:
        case,t,r,c,k,meta,td,z,axes=initial(key);y,s=z['yaw'],z['speed']
        geom=core.geometry(key,'mixed_fixed');quality=observations(key,geom)
        signals={'v24_control':y,'v29_control':np.load(ROOT/f'runs/domain_v29_submission/train/signals/{key}.npz')['yaw'],
            'v42_control':np.load(ROOT/f'runs/domain_v42_global_ba/train/signals/{key}.npz')['mixed_fixed_rotation_course']}
        diag={}
        for gate,cfg in GATES.items():
            edges=[e for e in geom['edges'] if accept(quality[f'{int(e[0])}_{int(e[1])}'],cfg,False)]
            courses=[d for d in geom['courses'] if accept(quality[f"{d['i']}_{d['j']}"],cfg,True)]
            e=np.array(edges).reshape(-1,4)
            if len(e):e[:,2]+=y[e[:,1].astype(int)]-y[e[:,0].astype(int)]
            rot=y+core.bg.supported_graph(t,y,e,3)
            dc,cd=core.local.motion.fit_motion(t,y,s,courses,'course')
            both,bd=core.local.motion.fit_motion(t,rot,s,courses,'course')
            signals[gate+'_rotation']=rot;signals[gate+'_course']=y+dc;signals[gate+'_both']=rot+both
            diag[gate]=dict(rotation_edges=len(e),course_edges=len(courses),course=cd,both=bd)
        for name,yaw in signals.items():
            assert np.isfinite(yaw).all()
            if name in CANDIDATES:assert np.max(abs(yaw-y))<=np.deg2rad(6)+1e-9
            save_json(OUT/f'predictions/{name}/{key}.json',dict(case['pred'],positions=core.integrate_velocity(t,yaw,s).tolist()))
        path=OUT/f'signals/{key}.npz';path.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(path,times=t,speed=s,**signals)
        save_json(OUT/f'diagnostics/{key}.json',dict(gates=diag,total_turn_deg=float(np.rad2deg(abs(np.diff(y)).sum())),
            gt_loading_blocked=True,scoring_blocked=True))
        print('INFER',key,{g:(v['rotation_edges'],v['course_edges']) for g,v in diag.items()},flush=True)


def compare(per,base):
    d=np.array([per[k]-base[k] for k in per]);order=np.sort(d)
    return dict(mean_delta=float(d.mean()),improved=int((d<-1e-8).sum()),worsened=int((d>1e-8).sum()),
        unchanged=int((abs(d)<=1e-8).sum()),worst_regression=max(0.,float(d.max())),
        delta_excluding_best=float(order[1:].mean()),per_clip_delta={k:float(per[k]-base[k]) for k in per})


def screen(comp,n):
    return bool(comp['improved']>=int(np.ceil(.75*n)) and comp['mean_delta']<=-.1 and comp['worst_regression']<=.2 and comp['delta_excluding_best']<0)


def evaluate():
    from public_dataset import load_reference,load_timestamps,validate_prediction
    from metrics import score_trajectory
    refs={key:load_reference(ROOT/'data/release',CLIPS[key]) for key in TRAIN};rows=[]
    for name in METHODS:
        per={}
        for key in TRAIN:
            p=json.loads((OUT/f'predictions/{name}/{key}.json').read_text())
            validate_prediction(p,key,load_timestamps(ROOT/'data/release',CLIPS[key]))
            per[key]=float(score_trajectory(np.array(p['times']),np.array(p['positions']),*refs[key])['drift_pct'])
        rows.append(dict(name=name,mean=float(np.mean(list(per.values()))),per_clip=per))
        print('SCORE',name,rows[-1]['mean'],flush=True)
    assert abs(rows[0]['mean']-9.532022663626405)<1e-8
    assert abs(rows[1]['mean']-9.195614707272306)<1e-8
    assert abs(rows[2]['mean']-8.907951350628013)<1e-8
    for row in rows:
        row['vs_v24']=compare(row['per_clip'],rows[0]['per_clip']);row['screen_pass']=row['name'] in CANDIDATES and screen(row['vs_v24'],len(TRAIN))
    def select(excluded):
        included=[k for k in TRAIN if k not in excluded]
        base={k:rows[0]['per_clip'][k] for k in included}
        candidates=[]
        for row in rows[3:]:
            per={k:row['per_clip'][k] for k in included};comp=compare(per,base)
            if screen(comp,len(included)):candidates.append((float(np.mean(list(per.values()))),row['name'],row))
        winner=min(candidates,key=lambda x:(x[0],x[1]))[2] if candidates else rows[0]
        return dict(excluded=excluded,selected=winner['name'],heldout={k:winner['per_clip'][k] for k in excluded},
            delta={k:winner['per_clip'][k]-rows[0]['per_clip'][k] for k in excluded},eligible=len(candidates))
    loco=[select([k]) for k in TRAIN]
    turns={k:json.loads((OUT/f'diagnostics/{k}.json').read_text())['total_turn_deg'] for k in TRAIN}
    cut=float(np.median(list(turns.values())))
    groups={'low_turn':[k for k in TRAIN if turns[k]<=cut],'high_turn':[k for k in TRAIN if turns[k]>cut]}
    group=[dict(group=g,**select(keys)) for g,keys in groups.items()]
    passed=[r for r in rows if r['screen_pass']]
    selected=min(passed,key=lambda r:(r['mean'],r['name']))['name'] if passed else 'v24_control'
    result=dict(rows=rows,selected_by_frozen_screen=selected,loco=loco,
        loco_mean=float(np.mean([list(f['heldout'].values())[0] for f in loco])),
        group_audit=group,group_mean=float(np.mean([v for f in group for v in f['heldout'].values()])),
        turn_groups=groups,turn_degrees=turns,
        limitations='15 configs x 8 reused public TRAIN clips. Exclusion audits only assess this fixed selection procedure, not independent generalization. Same session groups unknown. No TEST labels or new inference; no claim to beat13.19.')
    save_json(REP/'results.json',result)


def validate():
    plan=json.loads((REP/'plan.json').read_text())
    for rel,h in {**plan['dependencies'],**plan['preserved']}.items():assert sha256(ROOT/rel)==h,rel
    for key in TRAIN:
        z=np.load(OUT/f'signals/{key}.npz');base=np.load(ROOT/f'runs/domain_v24_submission/train/signals/{key}.npz')
        assert np.array_equal(z['speed'],base['speed']) and np.array_equal(z['times'],base['times'])
        for name in METHODS:
            p=json.loads((OUT/f'predictions/{name}/{key}.json').read_text())
            assert np.max(abs(np.array(p['positions'])-core.integrate_velocity(z['times'],z[name],z['speed'])))<1e-9
        d=json.loads((OUT/f'diagnostics/{key}.json').read_text())
        for gate in GATES:
            if d['gates'][gate]['course']['fallback']:assert np.array_equal(z[gate+'_course'],z['v24_control'])
    save_json(REP/'validation.json',dict(passed=True,predictions=len(METHODS)*len(TRAIN),
        speed_unchanged=True,exact_course_fallback=True,preserved_submissions=len(plan['preserved']),
        frozen_hashes=len(plan['dependencies']),no_test_inference=True))


def report():
    d=json.loads((REP/'results.json').read_text());val=json.loads((REP/'validation.json').read_text())
    labels={'v24_control':'V24 기준','v29_control':'V29 배경 보정','v42_control':'V42 기존 전역 BA'}
    table=''
    for row in d['rows']:
        c=row['vs_v24'];table+=f"<tr><td>{html.escape(labels.get(row['name'],row['name']))}</td><td>{row['mean']:.4f}%</td><td>{c['mean_delta']:+.4f}</td><td>{c['improved']} / {c['unchanged']} / {c['worsened']}</td><td>{c['worst_regression']:.4f}</td><td>{c['delta_excluding_best']:+.4f}</td><td>{'통과' if row['screen_pass'] else '-'}</td></tr>"
    per=''
    for row in d['rows']:
        per+='<tr><td>'+row['name']+'</td>'+''.join(f'<td>{row["per_clip"][k]:.3f}</td>' for k in TRAIN)+'</tr>'
    diag=''
    for k in TRAIN:
        a=json.loads((OUT/f'diagnostics/{k}.json').read_text())
        diag+='<tr><td>'+k+'</td>'+''.join(f'<td>{a["gates"][g]["rotation_edges"]} / {a["gates"][g]["course_edges"]}</td>' for g in GATES)+'</tr>'
    folds=''.join(f'<tr><td>{",".join(f["excluded"])}</td><td>{f["selected"]}</td><td>{np.mean(list(f["delta"].values())):+.4f}</td></tr>' for f in d['loco']+d['group_audit'])
    doc='''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>TideGraph BA 분리 실험</title><style>body{font:16px/1.7 system-ui;margin:36px auto;max-width:1250px;padding:0 20px;background:#f4f7fb;color:#203047}h1,h2{color:#12345b}section{background:white;border-radius:12px;padding:24px;margin:20px 0;overflow:auto}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:8px 12px;text-align:right;border-bottom:1px solid #dde4ed;white-space:nowrap}td:first-child,th:first-child{text-align:left}.note{background:#fff3d6;padding:16px;border-radius:8px}</style><h1>TideGraph: BA 효과 분리와 관측 조건 검증</h1>'''
    doc+=f'<p>공식 최고 표시값 13.19%, V42 13.20%. 아래 수치는 <b>반복 사용한 공개 TRAIN8</b> 결과이며 공식 TEST 성능이 아닙니다.</p><section><h2>설계</h2><p>V24 자세·속도에서 지도와 전역 BA를 다시 계산했습니다. V29 배경 그래프 보정을 제외하고, 회전만 / 이동방향만 / 둘 다를 분리했습니다. 속도는 전부 동일합니다. 공통 held-out 재투영 검증 위에 아래 관측 조건을 적용했습니다.</p><ul><li>ungated: 기존 BA 관측 및 시간 지지 조건만 적용.</li><li>observable: 공통점 16개, 화면 각도 분산 0.5도. 이동방향에는 시차 0.5도, normal SVD rank 0.02 이상/null 0.3 이하.</li><li>strict: 공통점 24개, 분산 1도, 시차 1도, rank 0.05 이상/null 0.2 이하.</li><li>converged: observable 조건에 BA 수렴 판정을 추가.</li></ul><p>조건은 결과 계산 전에 동결했습니다. 현재 보정량 상한은 회전/이동방향 각각 3도입니다.</p></section>'
    doc+=f'<section><h2>전체 비교</h2><table><tr><th>구성</th><th>평균 drift</th><th>V24 대비 %p</th><th>개선 / 유지 / 악화</th><th>최대 악화 %p</th><th>최대 이득 영상 제외 Δ</th><th>사전 screen</th></tr>{table}</table><p>선택: <b>{d["selected_by_frozen_screen"]}</b>. Screen: 6/8 이상 개선, 평균 -0.1%p 이하, 최대 악화 0.2%p 이하, 최대 이득 영상 제외해도 개선.</p></section>'
    doc+=f'<section><h2>영상 제외 선택 감사</h2><p>한 영상 제외 선택 평균 {d["loco_mean"]:.4f}%, 회전량 그룹 제외 선택 평균 {d["group_mean"]:.4f}%. 제외 영상의 점수로 해당 fold의 구성을 선택하지 않았습니다. 이 수치는 서로 다른 fold 선택 모델의 결합 평가이며 단일 제출 후보 점수가 아닙니다.</p><p class="note">모든 영상은 이전 실험에서 이미 사용됐습니다. 이 감사는 독립 검증이나 최종 테스트 성능 추정이 아닙니다. 공개 FLIR-VIS val이 없고 실제 촬영 세션도 알 수 없어, 입력 방향의 총 변화량으로 그룹을 나눴습니다.</p><table><tr><th>제외 영상</th><th>나머지에서 선택</th><th>제외 영상 평균 Δ %p</th></tr>{folds}</table></section>'
    doc+='<section><h2>영상별 drift</h2><table><tr><th>구성</th>'+''.join(f'<th>{k}</th>' for k in TRAIN)+f'</tr>{per}</table></section>'
    doc+='<section><h2>관측 수: 회전 / 이동방향</h2><table><tr><th>영상</th>'+''.join(f'<th>{g}</th>' for g in GATES)+f'</tr>{diag}</table><p>관측이 남아도 최소 8개·시간 범위 등 후속 조건을 통과하지 못하면 보정은 꺼집니다.</p></section>'
    doc+=f'<section><h2>검증과 범위</h2><p>{val["predictions"]}개 예측 검증, 기존 제출 {val["preserved_submissions"]}개 해시 보존. TEST 추론/업로드/최종 선택 없음. 공식 JSON에는 영상별 drift가 없으므로 어느 TEST 영상의 악화인지 알 수 없습니다.</p><p>설정: reports/domain_v43_ba_ablation/plan.json<br>수치: reports/domain_v43_ba_ablation/results.json<br>코드: scripts/v43_ba_ablation.py</p></section></html>'
    (ROOT/'TideGraph_BA_ablation_report.html').write_text(doc)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['freeze','infer','evaluate','validate','report']);ap.add_argument('--keys');a=ap.parse_args()
    if a.action=='freeze':freeze()
    else:
        assert (REP/'plan.json').exists()
        if a.action=='infer':infer(a.keys.split(',') if a.keys else TRAIN)
        else:globals()[a.action]()
