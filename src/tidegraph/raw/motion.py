"""Frozen motion fit, extracted from V19; inference only."""
import numpy as np
from scipy.optimize import least_squares
from ..core import integrate as integrate_velocity

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

def wrap(x):
    return np.arctan2(np.sin(x), np.cos(x))

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
