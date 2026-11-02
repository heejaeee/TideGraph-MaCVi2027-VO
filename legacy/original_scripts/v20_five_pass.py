"""Controlled DPVO3 versus DPVO5 experiment, retaining all original three seeds."""
import argparse, datetime, itertools, shutil, time
from v13_common import ROOT, TRAIN, base_case, save_json, np, json, gaussian_filter1d, integrate_velocity
from v13_infer import sources
from v5_common import planar_yaw
from public_dataset import sha256

REP = ROOT/'reports/domain_v20_five_pass'
OUT = ROOT/'runs/domain_v20_five_pass'
ORIGINAL = [0, 104729, 209759]
ADDED = [314159, 419431]
ALL = ORIGINAL+ADDED
PRIMARY = ['tidegraph', 'v19_course', 'raw3_hierarchical', 'raw3_cap3', 'raw5_hierarchical', 'raw5_cap3']


def freeze():
    p = REP/'plan.json'
    settings = dict(original_seeds=ORIGINAL, additional_seeds=ADDED, primary=PRIMARY,
        inference='Reuse v5_dpvo_masked.work(key, random, seed) unchanged: 2 Hz cached frames, width640, same learned intrinsics, random patches, same frozen checkpoint; no CLAHE.',
        aggregation='Smooth each planar yaw 2s; median within DPVO seeds; initial-angle anchor; median across DPVO/VGGT/Pi3X; optionally cap +/-3 degrees relative to TideGraph.',
        speed='Exactly TideGraph speed for every new candidate.',
        sensitivity='All 10 three-seed subsets of the five, bounded branch only. Diagnostic overlapping subsets, not independent repeats; never select seeds by drift.',
        decision='Primary paired change is raw5_cap3 minus fixed raw3_cap3. Report mean, per-clip effects, worst regression and excluding-best-clip mean. Also retain TideGraph comparison and LOCO method-selection audit.',
        transfer_screen='>=5/8 improved versus TideGraph; max regression <=.5pp; gain after removing largest improvement; LOCO <=TideGraph. This does not override the primary 3-versus-5 comparison.',
        scope='Repeated TRAIN8 development, no TEST labels or inference, no new submission. Single fixed nested 3-to-5 ensemble comparison, not an independent multi-ensemble stability study.')
    if p.exists():
        assert json.loads(p.read_text())['settings'] == settings
        return
    old_files = [ROOT/f'runs/domain_v5/dpvo/random_s{s}/{k}.npz' for s in ORIGINAL for k in TRAIN]
    for x in old_files: assert x.exists(), x
    deps = [ROOT/'scripts/v5_dpvo_masked.py', ROOT/'repos/DPVO/dpvo.pth',
            ROOT/'repos/DPVO/config/default.yaml', ROOT/'repos/DPVO/dpvo/dpvo.py', ROOT/'repos/DPVO/dpvo/net.py']
    save_json(p, dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), settings=settings,
        previous_three_pass_mean=9.935760637284028,
        frozen={str(x.relative_to(ROOT)):sha256(x) for x in old_files+list((ROOT/'submissions').glob('*/submission.json'))},
        dependencies={str(x.relative_to(ROOT)):sha256(x) for x in deps}))


def run_seed(seed):
    assert seed in ADDED
    import v5_dpvo_masked as d
    d.cv2.setNumThreads(1);d.torch.set_num_threads(2)
    frozen = json.loads((REP/'plan.json').read_text())
    for p, h in frozen['dependencies'].items(): assert sha256(ROOT/p) == h
    tic = time.perf_counter()
    for key in TRAIN:
        print('START', key, 'seed', seed, flush=True)
        before = time.perf_counter()
        d.work(key, 'random', seed)
        p = ROOT/f'runs/domain_v5/dpvo/random_s{seed}/{key}.npz'
        z = np.load(p);case, _, _ = base_case(key)
        assert np.isfinite(z['camera_to_world']).all()
        assert np.array_equal(z['times'], case['t'])
        save_json(REP/'runs'/f'{seed}_{key}.json',dict(seed=seed, key=key,
            seconds_this_call=time.perf_counter()-before, cache=str(p.relative_to(ROOT)), sha256=sha256(p),
            original_runner_diagnostic=json.loads(p.with_suffix('.json').read_text())))
        print('DONE', key, 'seed', seed, flush=True)
    save_json(REP/f'seed_{seed}_complete.json',dict(seed=seed, clips=TRAIN, seconds=time.perf_counter()-tic))


def infer():
    for key in TRAIN:
        case, _, _ = base_case(key)
        z = np.load(ROOT/f'runs/domain_v18/train/signals/{key}.npz')
        t, y, speed = z['times'], z['joint_depth_graph'], z['speed']
        models = sources(key, case);raw = []; hashes = {}
        for seed in ALL:
            p = ROOT/f'runs/domain_v5/dpvo/random_s{seed}/{key}.npz'
            zz = np.load(p);assert np.array_equal(t, zz['times'])
            yy = np.interp(t, zz['times'], planar_yaw(zz['camera_to_world']))
            raw.append(gaussian_filter1d(yy, 2/np.median(np.diff(t))))
            hashes[str(p.relative_to(ROOT))] = sha256(p)
        raw = np.array(raw)
        groups = {'raw3':(0,1,2), 'raw5':(0,1,2,3,4)}
        groups.update({'subset_'+'_'.join(map(str,ids)):ids for ids in itertools.combinations(range(5),3)})
        signals = {}; diagnostics = {}
        for group, ids in groups.items():
            dpvo = np.median(raw[list(ids)], axis=0)
            family = np.array([a-a[0]+y[0] for a in [dpvo,models['vggt'],models['pi3x']]])
            target = np.median(family, axis=0)
            variants = [('cap3', y+np.clip(target-y,-np.deg2rad(3),np.deg2rad(3)))]
            if group in ['raw3','raw5']: variants.append(('hierarchical',target))
            for suffix, yaw in variants:
                name = group+'_'+suffix
                pred = dict(case['pred'],positions=integrate_velocity(t,yaw,speed).tolist())
                save_json(OUT/'predictions'/name/f'{key}.json',pred);signals[name]=yaw
                diagnostics[name]=dict(seeds=[ALL[i] for i in ids], max_correction_deg=float(np.rad2deg(abs(yaw-y).max())),
                    saturated_fraction=float(np.mean(abs(target-y)>=np.deg2rad(3))))
                if group=='raw3':
                    old=json.loads((ROOT/f'runs/domain_v20_screen/predictions/{name}/{key}.json').read_text())
                    assert np.allclose(old['positions'],pred['positions'],rtol=0,atol=1e-9)
        for name,path in [('tidegraph',ROOT/f'runs/domain_v18/train/predictions/joint_depth_graph/{key}.json'),
                          ('v19_course',ROOT/f'runs/domain_v19/train/predictions/course/{key}.json')]:
            save_json(OUT/'predictions'/name/f'{key}.json',json.loads(path.read_text()))
        path=OUT/'signals'/f'{key}.npz';path.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(path,times=t,speed=speed,baseline_yaw=y,dpvo_yaws=raw,**signals)
        save_json(OUT/'diagnostics'/f'{key}.json',dict(input_hashes=hashes,methods=diagnostics))
        print('INFER',key,flush=True)


def compare(row, baseline):
    delta={k:row['per_clip'][k]-baseline['per_clip'][k] for k in TRAIN}
    return dict(mean_delta=float(np.mean(list(delta.values()))),per_clip_delta=delta,
        improved=sum(v < -1e-6 for v in delta.values()),worsened=sum(v > 1e-6 for v in delta.values()),
        worst_regression=max(delta.values()),delta_excluding_best=float(np.mean(sorted(delta.values())[1:])))


def evaluate():
    from v13_common import CLIPS
    from public_dataset import load_reference,load_timestamps,validate_prediction
    from metrics import score_trajectory
    refs={k:load_reference(ROOT/'data/release',CLIPS[k]) for k in TRAIN}
    names=PRIMARY+['subset_'+'_'.join(map(str,ids))+'_cap3' for ids in itertools.combinations(range(5),3)]
    rows=[]
    for name in names:
        per={}
        for key in TRAIN:
            p=json.loads((OUT/'predictions'/name/f'{key}.json').read_text())
            validate_prediction(p,key,load_timestamps(ROOT/'data/release',CLIPS[key]))
            per[key]=float(score_trajectory(np.array(p['times']),np.array(p['positions']),*refs[key])['drift_pct'])
        rows.append(dict(name=name,mean=float(np.mean(list(per.values()))),per_clip=per))
    by={r['name']:r for r in rows}
    assert abs(by['raw3_cap3']['mean']-9.935760637284028)<1e-8
    assert abs(by['tidegraph']['mean']-10.493886655215604)<1e-8
    for r in rows:
        r['versus_tidegraph']=compare(r,by['tidegraph'])
        r['versus_fixed_three']=compare(r,by['raw3_cap3'])
        if r['name'] in PRIMARY:print('SCORE',r['name'],r['mean'],r['versus_fixed_three'],flush=True)
    # Only the bounded 3/5 candidates participate in method selection. Subsets
    # are never available to the selector; uncapped runs are diagnostics only.
    folds=[]
    for held in TRAIN:
        rest=[k for k in TRAIN if k!=held];pool=[by['tidegraph']]
        for name in ['raw3_cap3','raw5_cap3']:
            r=by[name];delta=[r['versus_tidegraph']['per_clip_delta'][k] for k in rest]
            if sum(v < -1e-6 for v in delta)>=4 and max(delta)<=.5 and np.mean(sorted(delta)[1:])<0:pool.append(r)
        chosen=min(pool,key=lambda r:np.mean([r['per_clip'][k] for k in rest]))
        folds.append(dict(held=held,selected=chosen['name'],drift=chosen['per_clip'][held]))
    audit=float(np.mean([f['drift'] for f in folds]))
    for r in rows:
        c=r['versus_tidegraph'];r['eligible']=bool(r['name'] in ['raw3_cap3','raw5_cap3'] and c['improved']>=5 and c['worst_regression']<=.5 and c['delta_excluding_best']<0 and audit<=by['tidegraph']['mean'])
    subset_rows=[r for r in rows if r['name'].startswith('subset_')]
    save_json(REP/'results.json',dict(rows=rows,primary_comparison=compare(by['raw5_cap3'],by['raw3_cap3']),
        leave_one_clip_out=dict(mean=audit,folds=folds),
        subset_sensitivity=dict(mean_range=[min(r['mean'] for r in subset_rows),max(r['mean'] for r in subset_rows)],
            median_mean=float(np.median([r['mean'] for r in subset_rows])),
            five_pass_better_than_subsets=sum(by['raw5_cap3']['mean']<r['mean'] for r in subset_rows),
            caution='Ten overlapping three-seed subsets, not independent runs or a seed-selection pool.'),
        evaluations=len(rows)*len(TRAIN),scope='Repeated TRAIN8 development. No TEST score or new submission.'))


def validate():
    checks={}
    for key in TRAIN:
        z=np.load(OUT/'signals'/f'{key}.npz');t,y,s=z['times'],z['baseline_yaw'],z['speed']
        old=np.load(ROOT/f'runs/domain_v18/train/signals/{key}.npz');assert np.array_equal(s,old['speed'])
        checks[key]={}
        change=z['raw5_cap3']-z['raw3_cap3']
        checks[key]['five_vs_three']=dict(max_heading_change_deg=float(np.rad2deg(abs(change).max())),
            rms_heading_change_deg=float(np.rad2deg(np.sqrt(np.mean(change**2)))),
            changed_frame_fraction=float(np.mean(abs(change)>1e-8)))
        for name in ['raw3_cap3','raw5_cap3']:
            assert abs(z[name]-y).max()<=np.deg2rad(3)+1e-9
            p=json.loads((OUT/'predictions'/name/f'{key}.json').read_text())
            diff=float(np.max(abs(np.array(p['positions'])-integrate_velocity(t,z[name],s))))
            assert diff<1e-9
            checks[key][name]=dict(max_reintegration_error=diff,cap_deg=float(np.rad2deg(abs(z[name]-y).max())),speed_preserved=True)
        assert np.array_equal(z['subset_0_1_2_cap3'],z['raw3_cap3'])
    for p,h in json.loads((REP/'plan.json').read_text())['frozen'].items():assert sha256(ROOT/p)==h,p
    save_json(REP/'validation.json',dict(passed=True,checks=checks,old_three_pass_reproduced=True,original_caches_and_submissions_unchanged=True))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int);p.add_argument('--infer',action='store_true');p.add_argument('--evaluate',action='store_true');p.add_argument('--validate',action='store_true');a=p.parse_args();freeze()
    if a.seed is not None:run_seed(a.seed)
    if a.infer:infer()
    if a.evaluate:evaluate()
    if a.validate:validate()
