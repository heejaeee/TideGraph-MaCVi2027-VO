"""Reference-free three-view translation from two different image frontends.

Frozen primary experiment and input perturbations. No model/seed selection on
the repeated TRAIN scores. Existing submission files are never changed.
"""
import argparse
import datetime
import time
from collections import Counter
from functools import lru_cache
from scipy.spatial import cKDTree
import v45_independent_course as v

ROOT, np, json, save_json, cv2 = v.ROOT, v.np, v.json, v.save_json, v.cv2
OUT = ROOT / 'runs/domain_v50_multiview_course'
REP = ROOT / 'reports/domain_v50_multiview_course'
SCENARIOS = {
    'base': dict(fold=0, thin=1, focal=1., seed=0),
    'held_cells': dict(fold=1, thin=1, focal=1., seed=104729),
    'half_points': dict(fold=0, thin=2, focal=1., seed=209759),
    'focal_minus2': dict(fold=0, thin=1, focal=.98, seed=0),
    'focal_plus2': dict(fold=0, thin=1, focal=1.02, seed=0),
}
METHODS = ['v24_control', 'roma_pair', 'cotracker_pair', 'roma_multiview',
           'cotracker_multiview', 'cross_multiview', 'cross_conservative']
PRIMARY = 'cross_conservative'


def freeze():
    path = REP / 'plan.json'
    if path.exists():
        assert json.loads(path.read_text())['scenarios'] == SCENARIOS
        return
    files = list((ROOT/'scripts').glob('*.py'))
    for key in v.TRAIN:
        for folder, pattern in [
            (f'runs/domain_v17_roma/matches/train/{key}', '*.npz'),
            (f'runs/domain_v5/cotracker_windows/{key}', '*.npz'),
            (f'runs/domain_v5/segmentation/{key}', '*.png')]:
            files += list((ROOT/folder).glob(pattern))
        files += [ROOT/f'runs/domain_v24_submission/train/signals/{key}.npz',
                  ROOT/f'runs/domain_v4/calibrated_rotation_train/{key}.npz',
                  ROOT/f'runs/domain_probe_v2/r3/diagnostics/{key}.poses.npz']
    files += [ROOT/'runs/domain_v5/segmentation/model.json']
    save_json(path, dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        methods=METHODS, primary=PRIMARY, scenarios=SCENARIOS,
        observations='Triplets i,i+8,i+16 at i%16=0 (0,4,8 sec). CoTracker visible common tracks, RoMa pair joins by mutual nearest middle coordinates within1.5px. Static masks at all3 views, eroded3px. Max240 spatially ordered tracks; 48px image blocks partition FIT/heldout by (cellx+2*celly)%5.',
        estimation='First0-8 five-point essential plus cheirality; triangulate FIT inliers. PnP-RANSAC and iterative FIT-only refinement at16, then heldout third-view reprojection and endpoint epipolar/parallax checks. Independent0-16 essential direction must agree<=3deg with PnP translation. No depth network, BA centers or integrated trajectory in extraction. OpenCV recoverPose default far-depth filtering retained.',
        checks='Inherited V45 endpoint min24FIT/.4fraction,held8/.5fraction,1.5pxSampson,.7positive,.5degparallax,.7horizontal,4cells/.1xspread. Seed triangulation >=24 points,>=.2degparallax,<=2px reprojection. PnP >=24 and .4 inliers,3pxRANSAC. Heldout3D>=8 and>=.5held,third reprojection median<=2px,>=.5within3px,>=.7positive.',
        fusion='RoMa/CoTracker same-interval directions agree<=3deg; circular confidence mean. One vote per interval. V19 course fit cap3deg,min8 intervals,50percent time-bin support. Primary scales fitted correction by.5, so cap1.5deg. Speed and V24 rotations remain unchanged; only integrated travel direction changes.',
        selection='Primary cross_conservative fixed before any scoring; others ablations only. No per-clip rule or best perturbation choice. Promotion requires base gain>=.1pp,all stress means improve,each scenario max harm<=.2pp and removing best gain still improves. Exact fallback is allowed but zero change is not improvement.',
        limitations='All8TRAIN clips reused repeatedly. Spatial split/stresses are evidence and numerical audits, not independent generalization. Shared K/images/V24 initial orientation can cause common bias. No official TEST inference or score predicted.',
        dependencies={str(p.relative_to(ROOT)):v.audit.sha256(p) for p in files},
        preserved={str(p.relative_to(ROOT)):v.audit.sha256(p) for p in (ROOT/'submissions').glob('*/submission.json')}))


def prepare(key):
    path = OUT/f'tracks/{key}.npz'
    if path.exists():
        return dict(np.load(path))
    _, _, _, _, meta, _ = v.inputs(key)
    static = json.loads((ROOT/'runs/domain_v5/segmentation/model.json').read_text())['static_ids']

    @lru_cache(maxsize=100)
    def mask(i):
        p = ROOT/f'runs/domain_v5/segmentation/{key}/{i:04d}.png'
        lab = cv2.imread(str(p), 0)
        if lab is None:
            raise FileNotFoundError(p)
        return cv2.erode(np.isin(lab, static).astype(np.uint8), np.ones((3, 3), np.uint8))

    def valid(i, p):
        m = mask(i)
        xy = np.floor(p/[meta['width'], meta['height']]*[m.shape[1], m.shape[0]]).astype(int)
        ok = (xy[:, 0] >= 0) & (xy[:, 1] >= 0) & (xy[:, 0] < m.shape[1]) & (xy[:, 1] < m.shape[0])
        ids = np.flatnonzero(ok)
        ok[ids] &= m[xy[ids, 1], xy[ids, 0]] > 0
        return ok

    rows, chunks, offsets, counts = [], [], [0], Counter()
    def add(family, i, tracks):
        if len(tracks) == 0:
            counts[family+'_empty'] += 1
            return
        ok = np.ones(len(tracks), bool)
        for j in range(3):
            ok &= valid(i+8*j, tracks[:, j])
        tracks = tracks[ok]
        if len(tracks) < 40:
            counts[family+'_few_static'] += 1
            return
        order = np.lexsort((tracks[:, 0, 1], tracks[:, 0, 0]))
        order = order[np.linspace(0, len(order)-1, min(240, len(order))).astype(int)]
        tracks = tracks[order]
        rows.append([0 if family == 'roma' else 1, i, i+8, i+16])
        chunks.append(tracks); offsets.append(offsets[-1]+len(tracks))
        counts[family+'_retained'] += 1

    matchroot = ROOT/f'runs/domain_v17_roma/matches/train/{key}'
    for path0 in sorted(matchroot.glob('*.npz')):
        z0 = np.load(path0); i, j = int(z0['i']), int(z0['j'])
        if i % 16 or j-i != 8:
            continue
        path1 = matchroot/f'{j:04d}_{j+8:04d}.npz'
        if not path1.exists():
            continue
        z1 = np.load(path1); p, q, u, w = z0['p'], z0['q'], z1['p'], z1['q']
        if not len(q) or not len(u):
            continue
        dist, idx = cKDTree(u).query(q)
        _, back = cKDTree(q).query(u)
        take = np.flatnonzero((dist <= 1.5) & (back[idx] == np.arange(len(q))))
        add('roma', i, np.stack([p[take], .5*(q[take]+u[idx[take]]), w[idx[take]]], axis=1))
    for path0 in sorted((ROOT/f'runs/domain_v5/cotracker_windows/{key}').glob('*.npz')):
        z = np.load(path0); i = int(z['start'])
        if i % 16 or len(z['tracks']) <= 16:
            continue
        good = z['visible'][[0, 8, 16]].all(0)
        add('cotracker', i, z['tracks'][[0, 8, 16]][:, good].transpose(1, 0, 2))
    data = dict(rows=np.asarray(rows, int).reshape(-1, 4), tracks=np.concatenate(chunks) if chunks else np.empty((0, 3, 2)), offsets=np.array(offsets))
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **data)
    save_json(path.with_suffix('.json'), dict(counts=dict(counts), reference_access_blocked=True))
    print('PREPARE', key, dict(counts), flush=True)
    return data


def triangulate(p, q, k, r, t):
    a, b = v.normalized(p, k), v.normalized(q, k)
    x = cv2.triangulatePoints(np.c_[np.eye(3), np.zeros(3)], np.c_[r, t], a[:, :2].T, b[:, :2].T)
    finite = abs(x[3]) > 1e-12
    xyz = np.zeros((len(p), 3))
    xyz[finite] = x[:3, finite].T/x[3, finite, None]
    x1 = xyz@r.T+t
    uv0, uv1 = project(xyz, k), project(x1, k)
    aa, bb = a@r.T, b.copy()
    aa /= np.linalg.norm(aa, axis=1, keepdims=True); bb /= np.linalg.norm(bb, axis=1, keepdims=True)
    par = np.rad2deg(np.arctan2(np.linalg.norm(np.cross(aa, bb), axis=1), (aa*bb).sum(1)))
    ok = finite & np.isfinite(xyz).all(1) & (xyz[:, 2] > 0) & (x1[:, 2] > 0) & (par >= .2)
    ok &= np.maximum(np.linalg.norm(uv0-p, axis=1), np.linalg.norm(uv1-q, axis=1)) <= 2.
    return xyz, ok


def project(x, k):
    q = x@k.T
    den = q[:, 2].copy(); den[abs(den) < 1e-12] = 1e-12
    return q[:, :2]/den[:, None]


def multiview(points, fit, held, k, ri, axes, meta, seed, endpoint):
    p, q, w = points[:, 0], points[:, 1], points[:, 2]
    first = v.free_solver(p[fit], q[fit], k, seed+17)
    if first is None:
        return None, 'seed_fit', {}
    r1, t1, inside = first
    xyz, good = triangulate(p[fit], q[fit], k, r1, t1)
    good &= inside
    if good.sum() < 24:
        return None, 'seed_triangulation', dict(seed_points=int(good.sum()))
    ids = np.flatnonzero(good)
    cv2.setRNGSeed(seed+29)
    ok, rv, tv, inliers = cv2.solvePnPRansac(np.ascontiguousarray(xyz[good]), np.ascontiguousarray(w[fit][good]), k, None,
        iterationsCount=500, reprojectionError=3., confidence=.999, flags=cv2.SOLVEPNP_EPNP)
    if not ok or inliers is None or len(inliers) < 24 or len(inliers)/fit.sum() < .4:
        return None, 'pnp_support', dict(seed_points=int(good.sum()), pnp_points=0 if inliers is None else len(inliers))
    ix = ids[inliers.ravel()]
    ok, rv, tv = cv2.solvePnP(np.ascontiguousarray(xyz[ix]), np.ascontiguousarray(w[fit][ix]), k, None,
        rvec=rv, tvec=tv, useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok or not np.isfinite(rv).all() or not np.isfinite(tv).all() or np.linalg.norm(tv) < 1e-9:
        return None, 'pnp_invalid', {}
    r2 = cv2.Rodrigues(rv)[0]; t2 = tv.ravel()
    mask = np.zeros(fit.sum(), bool); mask[ix] = True
    row, reason, detail = v.inspect_solution((r2, t2/np.linalg.norm(t2), mask), p, w, fit, held, k, ri, axes, meta)
    if row is None:
        return None, 'endpoint_'+reason, detail
    if endpoint is None:
        return None, 'no_independent_endpoint', detail
    disagree = abs(float(np.rad2deg(v.core.bg.wrap(row['course']-endpoint['course']))))
    detail['endpoint_disagreement_deg'] = disagree
    if disagree > 3.:
        return None, 'endpoint_disagreement', detail
    xh, valid = triangulate(p[held], q[held], k, r1, t1)
    detail['held_seed_points'] = int(valid.sum())
    if valid.sum() < 8 or valid.mean() < .5:
        return None, 'held_triangulation', detail
    x2 = xh[valid]@r2.T+t2
    err = np.linalg.norm(project(x2, k)-w[held][valid], axis=1)
    med, fraction, positive = float(np.median(err)), float(np.mean(err <= 3.)), float(np.mean(x2[:, 2] > 0))
    detail.update(third_median_px=med, third_fraction=fraction, third_positive=positive)
    if med > 2. or fraction < .5 or positive < .7:
        return None, 'held_third_view', detail
    row['weight'] = min(row['weight'], endpoint['weight'])*fraction/(1+med)
    return row, 'accepted', detail


def extract(scenario, keys):
    v.core.bg.guard(); cv2.setNumThreads(1)
    def forbidden(*args, **kw):
        raise RuntimeError('Trajectory integration forbidden during measurement extraction')
    v.core.integrate_velocity = forbidden; v.audit.initial = forbidden
    cfg = SCENARIOS[scenario]
    for key in keys:
        dest = OUT/f'{scenario}/observations/{key}.json'
        if dest.exists():
            continue
        started = time.perf_counter()
        _, z, r, k, meta, axes = v.inputs(key); k = k.copy()
        k[0, 0] *= cfg['focal']; k[1, 1] *= cfg['focal']
        data = prepare(key); groups = {n: [] for n in METHODS[1:5]}; details, counts = [], Counter()
        for n, (fam, i, m, j) in enumerate(data['rows']):
            family = ['roma', 'cotracker'][fam]; i, j = int(i), int(j)
            pts = data['tracks'][data['offsets'][n]:data['offsets'][n+1]].astype(float)[::cfg['thin']]
            cell = np.floor(pts[:, 0]/48).astype(int)
            held = (cell[:, 0]+2*cell[:, 1]) % 5 == cfg['fold']; fit = ~held
            if held.sum() < 8 or fit.sum() < 30:
                counts[family+'_few_split'] += 1
                continue
            seed = i*1000+j+cfg['seed']
            order = np.random.default_rng(seed).permutation(fit.sum())
            a, b = pts[:, 0], pts[:, 2]
            try:
                sol = v.free_solver(a[fit][order], b[fit][order], k, seed)
                if sol is not None:
                    rr, tt, mask = sol; restored = np.zeros(fit.sum(), bool); restored[order] = mask
                    sol = rr, tt, restored
                endpoint, reason, diag = v.inspect_solution(sol, a, b, fit, held, k, r[i], axes, meta)
                counts[family+'_pair_'+reason] += 1
                details.append(dict(i=i, j=j, method=family+'_pair', reason=reason, **diag))
                if endpoint is not None:
                    groups[family+'_pair'].append(dict(i=i, j=j, **endpoint))
                row, reason, diag = multiview(pts, fit, held, k, r[i], axes, meta, seed, endpoint)
                counts[family+'_multiview_'+reason] += 1
                details.append(dict(i=i, j=j, method=family+'_multiview', reason=reason, **diag))
                if row is not None:
                    groups[family+'_multiview'].append(dict(i=i, j=j, **row))
            except cv2.error as exc:
                counts[family+'_degenerate'] += 1
                details.append(dict(i=i, j=j, method=family, reason='opencv_degenerate', error=str(exc)))
        save_json(dest, dict(rows=groups, counts=dict(counts), details=details, seconds=time.perf_counter()-started,
            reference_access_blocked=True, trajectory_integration_blocked=True))
        print('EXTRACT', scenario, key, {n: len(g) for n, g in groups.items()}, round(time.perf_counter()-started, 1), flush=True)


def infer(scenario):
    v.core.bg.guard()
    for key in v.TRAIN:
        case, z, _, _, _, _ = v.inputs(key); t, y, speed = z['times'], z['yaw'], z['speed']
        groups = json.loads((OUT/f'{scenario}/observations/{key}.json').read_text())['rows']
        groups['cross_multiview'] = v.agreement(groups['roma_multiview'], groups['cotracker_multiview'], False)
        signals, diagnostics = {'v24_control': y}, {}
        for name, rows in groups.items():
            assert len(rows) == len({(a['i'], a['j']) for a in rows})
            dy, diag = v.core.local.motion.fit_motion(t, y, speed, rows, 'course')
            signals[name] = y+dy; diagnostics[name] = diag
        signals['cross_conservative'] = y+.5*(signals['cross_multiview']-y)
        for name, yaw in signals.items():
            save_json(OUT/f'{scenario}/predictions/{name}/{key}.json', dict(case['pred'], positions=v.core.integrate_velocity(t, yaw, speed).tolist()))
        dest = OUT/f'{scenario}/signals/{key}.npz'; dest.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(dest, times=t, speed=speed, **signals)
        save_json(OUT/f'{scenario}/diagnostics/{key}.json', dict(methods=diagnostics, groups=groups, total_turn_deg=float(np.rad2deg(abs(np.diff(y)).sum()))))
        print('INFER', scenario, key, {n: (d['edges'], d['fallback']) for n, d in diagnostics.items()}, flush=True)


def evaluate(scenario):
    from public_dataset import load_reference, load_timestamps, validate_prediction
    from metrics import score_trajectory
    refs = {key: load_reference(ROOT/'data/release', v.CLIPS[key]) for key in v.TRAIN}
    rows = []
    for name in METHODS:
        per = {}
        for key in v.TRAIN:
            p = json.loads((OUT/f'{scenario}/predictions/{name}/{key}.json').read_text())
            validate_prediction(p, key, load_timestamps(ROOT/'data/release', v.CLIPS[key]))
            per[key] = float(score_trajectory(np.array(p['times']), np.array(p['positions']), *refs[key])['drift_pct'])
        comp = v.audit.compare(per, rows[0]['per_clip'] if rows else per)
        rows.append(dict(name=name, mean=float(np.mean(list(per.values()))), per_clip=per, vs_v24=comp))
        print('SCORE', scenario, name, rows[-1]['mean'], 'wins', comp['improved'], 'bad', comp['worsened'], flush=True)
    assert abs(rows[0]['mean']-9.532022663626405) < 1e-8
    save_json(REP/f'{scenario}_results.json', dict(rows=rows, primary=PRIMARY, scenario=scenario, scope='Repeated TRAIN8, frozen primary; ablations do not select submission.'))


def synthetic():
    rng = np.random.default_rng(50)
    xyz = rng.uniform([-4., -2., 8.], [4., 2., 20.], (240, 3))
    k = np.array([[900., 0, 480], [0, 900., 270], [0, 0, 1.]])
    r1 = v.core.Rotation.from_euler('y', 2., degrees=True).as_matrix()
    r2 = v.core.Rotation.from_euler('y', 4., degrees=True).as_matrix()
    c1, c2 = np.array([.5, 0, .7]), np.array([1.1, 0, 1.5])
    points = np.stack([project(xyz, k), project((xyz-c1)@r1.T, k), project((xyz-c2)@r2.T, k)], axis=1)
    held = np.arange(240) % 5 == 0; fit = ~held
    axes = (np.array([0, 0, 1.]), np.array([1., 0, 0]))
    meta = dict(width=960, height=540)
    ep, reason, _ = v.inspect_solution(v.free_solver(points[fit, 0], points[fit, 2], k, 50), points[:, 0], points[:, 2], fit, held, k, np.eye(3), axes, meta)
    assert ep is not None, reason
    row, reason, detail = multiview(points, fit, held, k, np.eye(3), axes, meta, 50, ep)
    assert row is not None, (reason, detail)
    assert abs(v.core.bg.wrap(row['course']-np.arctan2(c2[0], c2[2]))) < 1e-4
    corrupted = points.copy(); corrupted[held, 2] += [30., -20.]
    bad, reason, _ = multiview(corrupted, fit, held, k, np.eye(3), axes, meta, 50, ep)
    assert bad is None, 'Heldout wrong tracks must be rejected'
    save_json(REP/'synthetic.json', dict(passed=True, checks=['Three-view camera center convention', 'Known camera travel direction recovered', 'Heldout corruption rejected'], course_error_deg=float(np.rad2deg(abs(row['course']-np.arctan2(c2[0], c2[2]))))))
    print('SYNTHETIC PASS', flush=True)


def validate():
    plan = json.loads((REP/'plan.json').read_text())
    for rel, sha in {**plan['dependencies'], **plan['preserved']}.items():
        assert v.audit.sha256(ROOT/rel) == sha, rel
    assert json.loads((REP/'synthetic.json').read_text())['passed']
    for scenario in SCENARIOS:
        for key in v.TRAIN:
            z = np.load(OUT/f'{scenario}/signals/{key}.npz')
            base = np.load(ROOT/f'runs/domain_v24_submission/train/signals/{key}.npz')
            diag = json.loads((OUT/f'{scenario}/diagnostics/{key}.json').read_text())
            assert np.array_equal(z['times'], base['times']) and np.array_equal(z['speed'], base['speed'])
            for name in METHODS:
                p = json.loads((OUT/f'{scenario}/predictions/{name}/{key}.json').read_text())
                assert np.allclose(p['positions'], v.core.integrate_velocity(z['times'], z[name], z['speed']), rtol=0, atol=1e-9)
                assert abs(z[name]-z['v24_control']).max() <= np.deg2rad(1.5 if name == PRIMARY else 3.)+1e-9
                dn = 'cross_multiview' if name == PRIMARY else name
                if dn in diag['methods'] and diag['methods'][dn]['fallback']:
                    assert np.array_equal(z[name], z['v24_control'])
    save_json(REP/'validation.json', dict(passed=True, predictions=len(METHODS)*len(SCENARIOS)*8,
        preserved_submissions=len(plan['preserved']), unchanged_speed=True, exact_fallback=True, no_test_inference=True))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('action', choices=['freeze', 'prepare', 'extract', 'infer', 'evaluate', 'synthetic', 'validate'])
    ap.add_argument('--scenario', choices=list(SCENARIOS), default='base'); ap.add_argument('--keys'); args = ap.parse_args()
    keys = args.keys.split(',') if args.keys else v.TRAIN
    if args.action == 'freeze': freeze()
    else:
        assert (REP/'plan.json').exists()
        if args.action == 'prepare':
            v.core.bg.guard()
            for key in keys: prepare(key)
        elif args.action == 'extract': extract(args.scenario, keys)
        elif args.action in ['infer', 'evaluate']: globals()[args.action](args.scenario)
        else: globals()[args.action]()
