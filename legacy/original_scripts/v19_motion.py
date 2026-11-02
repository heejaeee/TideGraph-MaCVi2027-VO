"""V19: reference-free, observable interval translation on frozen TideGraph.

Uses cached RoMa static correspondences and Pi3X depth. RePoseD translation
is recovered explicitly; no reference trajectory is read on this path.
"""
import argparse
import datetime
import time
from collections import Counter
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from v13_common import ROOT, TRAIN, base_case, save_json, np, json, integrate_velocity
from v11_test_candidate import provider
from v5_common import bearings, planar_yaw, cv2
from public_dataset import sha256
import v17_depth_geometry as geometry

OUT = ROOT / 'runs/domain_v19'
REP = ROOT / 'reports/domain_v19'
METHODS = ['tidegraph', 'course', 'speed', 'combined']
CONFIG = dict(
    version=1, minimum_points=40, split_rotation_deg=1.,
    split_translation_deg=10., split_log_scale=.15, split_log_distance=.25,
    minimum_parallax_deg=.2, minimum_cells=4, minimum_x_fraction=.1,
    minimum_horizontal_fraction=.7, minimum_edges=8, minimum_coverage=.5,
    knot_seconds=30., course_cap_deg=3., speed_log_cap=float(np.log(1.1)),
    course_noise_deg=3., speed_noise_log=.1, scale_graph_noise=.1,
    maximum_scale_residual=.15, prior_weight=.5, smooth_weight=2.,
    local_support_seconds=30., maximum_evaluations=120,
    methods=METHODS,
)


def freeze():
    path = REP / 'implementation_config.json'
    if path.exists():
        assert json.loads(path.read_text())['config'] == CONFIG, 'Frozen config mismatch'
    else:
        save_json(path, dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            config=CONFIG, status='Frozen before candidate scoring',
            meaning='Independent course and relative-speed fits; combined applies both. '
            'A single input-only rule for every clip. Temporal support masks enforce local fallback.',
            inherited_gates='V17 learned_depth_scale accepted: heldout Sampson <=1.5px, '
            'cheirality >=.6, depth log error <=.3, rotation delta <=10deg, '
            'inliers >=12 and fraction >=.4.'))
    return sha256(path)


def wrap(x):
    return np.arctan2(np.sin(x), np.cos(x))


def angle(a, b):
    return float(np.rad2deg(np.arccos(np.clip(
        np.dot(a, b) / max(np.linalg.norm(a)*np.linalg.norm(b), 1e-12), -1, 1))))


def closest_window(index, starts):
    valid = np.flatnonzero((starts <= index) & (starts + 12 > index))
    assert len(valid), index
    return int(starts[min(valid, key=lambda v: abs(index-(starts[v]+5.5)))])


def extract(key, split):
    config_hash = freeze()
    path = OUT / 'observations' / split / f'{key}.json'
    if path.exists():
        cached = json.loads(path.read_text())
        assert cached['config_sha256'] == config_hash
        for p, h in cached['input_hashes'].items():
            assert sha256(ROOT/p) == h, p
        return cached
    start_time = time.perf_counter()
    case, _, _, poses, k, meta = provider(key, split)
    signals = np.load(ROOT / f'runs/domain_v18/{split}/signals/{key}.npz')
    yaw = signals['joint_depth_graph']
    raw = planar_yaw(poses)
    up = np.median(poses[:, :3, 1], axis=0); up /= np.linalg.norm(up)
    forward = poses[0, :3, 2].copy(); forward -= up * (forward@up)
    forward /= np.linalg.norm(forward)
    right = np.cross(up, forward); right /= np.linalg.norm(right)
    winroot = ROOT / ('runs/domain_v3/pi3x_rgb' if split == 'train'
                      else 'runs/domain_v13/pi3x_test')
    starts = np.array([int(p.stem) for p in sorted((winroot/'windows'/key).glob('*.npz'))])
    rows, rejected, input_hashes = [], Counter(), {}
    for source in ['domain_v17_roma', 'domain_v18']:
        pairpath = ROOT/f'runs/{source}/pairs/{split}/{key}.npz'
        edgepath = ROOT/f'runs/{source}/edges/{split}/{key}.npz'
        for p in [pairpath, edgepath]:
            input_hashes[str(p.relative_to(ROOT))] = sha256(p)
        data = np.load(pairpath)
        old = np.load(edgepath)['learned_depth_scale']
        assert len(old) == len(data['pairs'])
        for idx, pair in enumerate(data['pairs']):
            i, j = pair[:2].astype(int)
            assert np.array_equal(old[idx, :2], [i, j])
            pts = data['points'][data['offsets'][idx]:data['offsets'][idx+1]]
            if old[idx, 3] <= .01:
                rejected['inherited_geometry'] += 1; continue
            if len(pts) < CONFIG['minimum_points']:
                rejected['few_points'] += 1; continue
            p, q, d1, d2 = pts[:, :2], pts[:, 2:4], pts[:, 4], pts[:, 5]
            cells = len(np.unique(np.floor(p/[meta['width'], meta['height']]*[4, 3]), axis=0))
            if cells < CONFIG['minimum_cells'] or np.ptp(p[:, 0])/meta['width'] < CONFIG['minimum_x_fraction']:
                rejected['spatial_support'] += 1; continue
            # Spatially sorted interleaved subsets each cover the image extent.
            order = np.lexsort((p[:, 1], p[:, 0]))
            fits = []
            for take in [np.arange(len(p)), order[::2], order[1::2]]:
                pose, stat, geom = geometry.solve('depth_scale', p[take], q[take],
                    d1[take], d2[take], k, int(i*1000+j))
                r, trans = np.asarray(pose.R), np.asarray(pose.t)
                fits.append((r, trans, float(geom.scale)))
            if any(not (np.isfinite(r).all() and np.isfinite(tr).all() and
                         np.linalg.norm(tr) > 1e-8 and np.isfinite(sc) and sc > 0)
                   for r, tr, sc in fits):
                rejected['invalid_fit'] += 1; continue
            r, tr, scale = fits[0]
            r1, tr1, sc1 = fits[1]; r2, tr2, sc2 = fits[2]
            disagreement = dict(
                rotation_deg=float(np.rad2deg(Rotation.from_matrix(r1@r2.T).magnitude())),
                translation_deg=angle(-r1.T@tr1, -r2.T@tr2),
                log_scale=float(abs(np.log(sc1/sc2))),
                log_distance=float(abs(np.log(np.linalg.norm(tr1)/np.linalg.norm(tr2)))))
            limits = dict(rotation_deg=CONFIG['split_rotation_deg'],
                          translation_deg=CONFIG['split_translation_deg'],
                          log_scale=CONFIG['split_log_scale'],
                          log_distance=CONFIG['split_log_distance'])
            if any(disagreement[n] > limits[n] for n in limits):
                rejected['split_disagreement'] += 1; continue
            a, b = bearings(p, k), bearings(q, k)
            parallax = float(np.rad2deg(np.median(np.arccos(np.clip(np.sum((a@r.T)*b, axis=1), -1, 1)))))
            displacement = poses[i, :3, :3] @ (-r.T @ tr)
            horizontal = np.array([displacement@right, displacement@forward])
            fraction = float(np.linalg.norm(horizontal)/np.linalg.norm(displacement))
            if parallax < CONFIG['minimum_parallax_deg'] or fraction < CONFIG['minimum_horizontal_fraction']:
                rejected['weak_translation'] += 1; continue
            course = float(wrap(np.arctan2(horizontal[0], horizontal[1]) + yaw[i]-raw[i]))
            rows.append(dict(i=int(i), j=int(j), source=source, weight=float(old[idx, 3]),
                course=course, horizontal_distance=float(np.linalg.norm(horizontal)*pair[2]),
                wi=closest_window(i, starts), wj=closest_window(j, starts),
                scale_rhs=float(np.log(scale)+np.log(pair[2]/pair[3])),
                split=disagreement, parallax_deg=parallax, cells=cells, horizontal_fraction=fraction))
    result = dict(key=key, split=split, config_sha256=config_hash, input_hashes=input_hashes,
        seconds=time.perf_counter()-start_time, accepted=len(rows), rejected=dict(rejected), rows=rows)
    save_json(path, result)
    print('OBSERVATIONS', split, key, len(rows), dict(rejected), flush=True)
    return result


def coverage(t, rows):
    if not rows:
        return 0.
    bins = max(1, int(np.ceil((t[-1]-t[0])/CONFIG['knot_seconds'])))
    mid = np.array([.5*(t[r['i']]+t[r['j']]) for r in rows])
    occupied = np.unique(np.minimum(bins-1, ((mid-t[0])/CONFIG['knot_seconds']).astype(int)))
    return float(len(occupied)/bins)


def enough(t, rows):
    return len(rows) >= CONFIG['minimum_edges'] and coverage(t, rows) >= CONFIG['minimum_coverage']


def basis(t, rows):
    knots = np.linspace(t[0], t[-1], int(np.ceil((t[-1]-t[0])/CONFIG['knot_seconds']))+1)
    b = np.array([np.interp(t, knots, v) for v in np.eye(len(knots))]).T
    # Smooth local support envelope, exactly zero beyond 30 seconds from evidence.
    mid = np.array([.5*(t[r['i']]+t[r['j']]) for r in rows])
    distance = np.min(abs(t[:, None]-mid), axis=1)
    support = np.clip(1-distance/CONFIG['local_support_seconds'], 0, 1)
    return b*support[:, None], knots, support


def solve_scale_graph(rows):
    """Depth-scale ratios from shared static scene points in relative poses."""
    if not rows:
        return [], dict(reason='no_observations')
    nodes = sorted({r[n] for r in rows for n in ['wi', 'wj']})
    neighbors = {n: set() for n in nodes}
    for r in rows:
        neighbors[r['wi']].add(r['wj']); neighbors[r['wj']].add(r['wi'])
    components, unseen = [], set(nodes)
    while unseen:
        stack = [min(unseen)]; component = set()
        while stack:
            n = stack.pop()
            if n in component: continue
            component.add(n); unseen.discard(n); stack.extend(neighbors[n]-component)
        components.append(component)
    # Selection uses measurement support only, never drift or clip identity.
    component = max(components, key=lambda c: sum(r['wi'] in c for r in rows))
    selected = [r for r in rows if r['wi'] in component]
    nodes = sorted(component); index = {n: i for i, n in enumerate(nodes)}
    a = np.zeros((len(selected), len(nodes)))
    rhs = np.array([r['scale_rhs'] for r in selected])
    w = np.sqrt([r['weight'] for r in selected])
    for ix, r in enumerate(selected):
        a[ix, index[r['wi']]] -= 1; a[ix, index[r['wj']]] += 1
    if len(nodes) > 1:
        sol = least_squares(lambda x: (a[:, 1:]@x-rhs)*w,
            np.zeros(len(nodes)-1), loss='soft_l1', f_scale=CONFIG['scale_graph_noise'],
            max_nfev=CONFIG['maximum_evaluations'])
        x = np.r_[0., sol.x]
    else:
        x = np.zeros(1)
    residual = a@x-rhs
    output = [dict(r, log_distance=float(np.log(r['horizontal_distance'])+x[index[r['wi']]]),
                   scale_residual=float(err))
              for r, err in zip(selected, residual) if abs(err) <= CONFIG['maximum_scale_residual']]
    return output, dict(components=len(components), chosen_nodes=len(nodes),
        input_edges=len(rows), component_edges=len(selected), retained=len(output),
        median_abs_log_residual=float(np.median(abs(residual))),
        scale_log_range=float(np.ptp(x)), node_log_scales={str(n): float(x[index[n]]) for n in nodes})


def fit_motion(t, yaw, speed, rows, kind):
    zeros = np.zeros(len(t))
    if not enough(t, rows):
        return zeros, dict(fallback=True, reason='insufficient_temporal_support',
            edges=len(rows), coverage=coverage(t, rows))
    b, knots, support = basis(t, rows)
    i = np.array([r['i'] for r in rows]); j = np.array([r['j'] for r in rows])
    w = np.sqrt([r['weight'] for r in rows])
    cap = np.deg2rad(CONFIG['course_cap_deg']) if kind == 'course' else CONFIG['speed_log_cap']
    noise = np.deg2rad(CONFIG['course_noise_deg']) if kind == 'course' else CONFIG['speed_noise_log']
    target = np.array([r['course'] if kind == 'course' else r['log_distance'] for r in rows])
    def residual(x):
        delta = b@x
        pos = integrate_velocity(t, yaw+delta if kind == 'course' else yaw,
                                 speed*np.exp(delta) if kind == 'speed' else speed)
        dp = pos[j]-pos[i]
        if kind == 'course':
            err = wrap(np.arctan2(dp[:, 0], dp[:, 2])-target)
        else:
            err = np.log(np.maximum(np.linalg.norm(dp, axis=1), 1e-8))-target
            # Eliminate the arbitrary global monocular scale.
            err -= np.average(err, weights=w*w)
        return np.r_[w*err/noise, CONFIG['prior_weight']*x/cap,
            CONFIG['smooth_weight']*np.diff(x, 2)/cap, np.mean(delta)/cap]
    initial = np.zeros(len(knots))
    sol = least_squares(residual, initial, bounds=(-cap, cap), loss='soft_l1',
        f_scale=1., max_nfev=CONFIG['maximum_evaluations'])
    delta = b@sol.x
    assert np.max(abs(delta)) <= cap+1e-10 and np.all(delta[support == 0] == 0)
    return delta, dict(fallback=False, edges=len(rows), coverage=coverage(t, rows),
        max_abs_correction=float(np.max(abs(delta))), active_frame_fraction=float(np.mean(support > 0)),
        cost_before=float(np.sum(residual(initial)**2)), cost_after=float(np.sum(residual(sol.x)**2)),
        optimizer_success=bool(sol.success), optimizer_status=int(sol.status), nfev=int(sol.nfev))


def predict(key, split):
    observations = extract(key, split)
    case, _, _ = base_case(key, split)
    z = np.load(ROOT/f'runs/domain_v18/{split}/signals/{key}.npz')
    t, y, s = z['times'], z['joint_depth_graph'], z['speed']
    assert np.array_equal(t, case['t'])
    baseline = json.loads((ROOT/f'runs/domain_v18/{split}/predictions/joint_depth_graph/{key}.json').read_text())
    assert np.allclose(baseline['positions'], integrate_velocity(t, y, s), rtol=0, atol=1e-9)
    course, course_diag = fit_motion(t, y, s, observations['rows'], 'course')
    speed_rows, scale_diag = solve_scale_graph(observations['rows'])
    speed, speed_diag = fit_motion(t, y, s, speed_rows, 'speed')
    for name in METHODS:
        dy = course if name in ['course', 'combined'] else np.zeros(len(t))
        ds = speed if name in ['speed', 'combined'] else np.zeros(len(t))
        pred = dict(baseline, positions=integrate_velocity(t, y+dy, s*np.exp(ds)).tolist())
        if not np.any(dy) and not np.any(ds): pred = baseline
        save_json(OUT/split/'predictions'/name/f'{key}.json', pred)
    path = OUT/split/'signals'/f'{key}.npz'; path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, times=t, baseline_yaw=y, baseline_speed=s,
                        course_delta=course, speed_log_delta=speed)
    save_json(OUT/split/'diagnostics'/f'{key}.json',
        dict(course=course_diag, speed=speed_diag, scale_graph=scale_diag))
    print('INFER', split, key, 'course_deg', float(np.rad2deg(abs(course).max())),
          'speed_pct', float(100*np.max(abs(np.exp(speed)-1))), flush=True)


def synthetic():
    # Verify transform convention and normalized-depth scale graph algebra.
    rng = np.random.default_rng(19)
    x = rng.uniform([-3, -2, 5], [3, 2, 12], (120, 3))
    r = Rotation.from_rotvec([.02, .08, -.01]).as_matrix()
    tr = np.array([.4, .1, .2]); xx = x@r.T+tr
    k = np.array([[800., 0, 480], [0, 800, 270], [0, 0, 1.]])
    p = x@k.T; p = p[:, :2]/p[:, 2, None]
    q = xx@k.T; q = q[:, :2]/q[:, 2, None]
    pose, _, geom = geometry.solve('depth_scale', p, q, x[:, 2], xx[:, 2]/1.7, k, 0)
    assert angle(-np.asarray(pose.R).T@pose.t, -r.T@tr) < .02
    assert abs(np.log(geom.scale/1.7)) < .001
    logs = np.array([0., .2, -.1, .3])
    rows = [dict(wi=i, wj=j, weight=1., scale_rhs=float(logs[j]-logs[i]),
                 horizontal_distance=float(np.exp(-logs[i])*2), i=i, j=j)
            for i, j in [(0, 1), (1, 2), (2, 3), (0, 3), (0, 2)]]
    recovered, _ = solve_scale_graph(rows)
    assert max(abs(r['log_distance']-np.log(2)) for r in recovered) < 1e-6
    t = np.linspace(0, 300, 601); y = .1*np.sin(t/100); s = np.ones(len(t))
    for kind in ['course', 'speed']:
        delta, diag = fit_motion(t, y, s, [], kind)
        assert diag['fallback'] and np.array_equal(delta, np.zeros(len(t)))
    # Recover a smooth synthetic course/speed shape; retain bounded corrections.
    actual_y = y+np.deg2rad(2)*np.sin(t/70)
    actual_s = np.exp(.07*np.sin(t/70))
    yp = integrate_velocity(t, actual_y, s)
    sp = integrate_velocity(t, y, actual_s)
    rows = []
    for i in range(0, 560, 16):
        j = i+32; dp = yp[j]-yp[i]
        rows.append(dict(i=i, j=j, weight=1., course=float(np.arctan2(dp[0], dp[2])),
                         log_distance=float(np.log(np.linalg.norm(sp[j]-sp[i])))))
    dc, cd = fit_motion(t, y, s, rows, 'course')
    ds, sd = fit_motion(t, y, s, rows, 'speed')
    assert np.corrcoef(dc, actual_y-y)[0, 1] > .8
    assert np.corrcoef(ds, np.log(actual_s))[0, 1] > .8
    assert cd['cost_after'] < cd['cost_before'] and sd['cost_after'] < sd['cost_before']
    dc2, _ = fit_motion(t, y+.3, 2*s,
        [dict(r, course=r['course']+.3) for r in rows], 'course')
    ds2, _ = fit_motion(t, y+.3, 2*s,
        [dict(r, log_distance=r['log_distance']+np.log(7)) for r in rows], 'speed')
    assert np.max(abs(dc-dc2)) < 1e-6 and np.max(abs(ds-ds2)) < 1e-6
    save_json(REP/'synthetic.json', dict(passed=True, checks=[
        'camera-center displacement sign/frame', 'RePoseD depth scale convention',
        'static observation depth scale graph', 'insufficient-observation exact fallback',
        'bounded smooth synthetic course recovery', 'bounded synthetic relative-speed recovery',
        'global heading and scale gauge invariance'], course=cd, speed=sd))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', choices=['train', 'test'], default='train')
    parser.add_argument('--keys')
    parser.add_argument('--synthetic-only', action='store_true')
    args = parser.parse_args()
    cv2.setNumThreads(1)
    freeze(); synthetic()
    if args.synthetic_only: return
    if args.keys: keys = args.keys.split(',')
    elif args.split == 'train': keys = TRAIN
    else:
        from v6_test_common import KEYS
        keys = KEYS
    for key in keys: predict(key, args.split)
    for p, h in json.loads((REP/'plan.json').read_text())['frozen'].items():
        assert sha256(ROOT/p) == h, p


if __name__ == '__main__':
    main()
