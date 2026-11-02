import time
import numpy as np
import cv2
from collections import Counter
from . import course as v

METHODS = ['v24_control', 'roma_pair', 'cotracker_pair', 'roma_multiview',
           'cotracker_multiview', 'cross_multiview', 'cross_conservative']

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
