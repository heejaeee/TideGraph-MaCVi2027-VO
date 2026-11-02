"""Input-only conservative course envelope across five spatial track folds.

Not a probabilistic confidence interval: same scene and calibration are shared.
Adaptive after V52, with all thresholds fixed before these predictions scored.
"""
import argparse
import datetime
import v50_multiview_course as m
v, ROOT, np, json, save_json = m.v, m.ROOT, m.np, m.json, m.save_json
OUT = ROOT/'runs/domain_v53_course_uncertainty'
REP = ROOT/'reports/domain_v53_course_uncertainty'
BANKS = ['base', 'half_points', 'focal_minus2', 'focal_plus2', 'focal_minus4', 'focal_plus4']
OFFSETS = [0, 104729, 314159, 419431, 523717]
PRIMARY = 'conservative_envelope'
METHODS = ['v24_control', 'median_folds', PRIMARY]


def source(bank, fold, key):
    if fold == 0:
        experiment = 'domain_v52_course_transfer_audit' if bank.endswith('4') else 'domain_v51_longer_course'
        return ROOT/f'runs/{experiment}/{bank}/signals/{key}.npz'
    if bank == 'base':
        if fold == 1: return ROOT/f'runs/domain_v51_longer_course/held_cells/signals/{key}.npz'
        return ROOT/f'runs/domain_v52_course_transfer_audit/cells{fold}/signals/{key}.npz'
    return OUT/f'{bank}/fold{fold}/signals/{key}.npz'


def freeze():
    p = REP/'plan.json'
    if p.exists(): return
    deps = json.loads((ROOT/'reports/domain_v52_course_transfer_audit/plan.json').read_text())['dependencies'].copy()
    for bank in BANKS:
        for fold in range(5):
            for key in v.TRAIN:
                pp = source(bank, fold, key)
                if pp.exists(): deps[str(pp.relative_to(ROOT))] = v.audit.sha256(pp)
    deps['scripts/v53_course_uncertainty.py'] = v.audit.sha256(ROOT/'scripts/v53_course_uncertainty.py')
    save_json(p, dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), primary=PRIMARY, methods=METHODS, banks=BANKS,
        nomination='Adaptive after V52: all10 tested means improve but some clip harms>.2pp, especially calibration/feature-split sensitivity. Use observable disagreement for shrinkage rather than GT-based video selection.',
        inference='Five fixed48px block holdout folds, V51 CoTracker3view solver/coursecap3deg, inactive folds give exactly0 correction. At every time sort5 course corrections. Retain second-smallest ifpositive or second-largest ifnegative, otherwise0. Thus require4of5 same-sign correction, choose conservative magnitude. Median5 ablation only. No selected fold and no GT in rule.',
        limitations='Envelope is empirical sensitivity, not calibrated probability. Shared scene,K,orientation cause correlated errors. AllTRAIN8 previously reused. These6banks are perturbation tests, not unseen videos.',
        promotion='Primary fixed. Base gain>=.1pp,each of6means strictly improves,max perclip harm<=.2pp each,excluding-best improves each. No changing thresholds if failed. No automatic submission.',
        dependencies=deps, preserved={str(pp.relative_to(ROOT)):v.audit.sha256(pp) for pp in (ROOT/'submissions').glob('*/submission.json')}))


def extract_and_infer(bank, fold):
    assert bank != 'base' and fold in range(1, 5)
    v.core.bg.guard()
    m.OUT = OUT/bank; m.REP = REP
    focal = {'half_points': 1., 'focal_minus2': .98, 'focal_plus2': 1.02, 'focal_minus4': .96, 'focal_plus4': 1.04}[bank]
    scenario = f'fold{fold}'
    m.SCENARIOS = {scenario: dict(fold=fold, thin=2 if bank == 'half_points' else 1, focal=focal, seed=OFFSETS[fold]+(209759 if bank == 'half_points' else 0))}
    m.prepare = lambda key: dict(np.load(ROOT/f'runs/domain_v51_longer_course/tracks/{key}.npz'))
    # Extraction and motion integration must be separate processes: V50 actively
    # forbids integration during extraction. This action performs extraction only.
    m.extract(scenario, v.TRAIN)


def fold_infer(bank, fold):
    v.core.bg.guard()
    for key in v.TRAIN:
        _, z, _, _, _, _ = v.inputs(key)
        rows = json.loads((OUT/f'{bank}/fold{fold}/observations/{key}.json').read_text())['rows']['cotracker_multiview']
        dy, diag = v.core.local.motion.fit_motion(z['times'], z['yaw'], z['speed'], rows, 'course')
        p = source(bank, fold, key); p.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(p, times=z['times'], speed=z['speed'], v24_control=z['yaw'], cotracker_multiview=z['yaw']+dy)
        save_json(p.with_suffix('.json'), diag)


def combine(delta):
    ordered = np.sort(delta, axis=0)
    return np.where(ordered[1] > 0, ordered[1], np.where(ordered[3] < 0, ordered[3], 0.))


def infer(bank):
    v.core.bg.guard()
    for key in v.TRAIN:
        case, z, _, _, _, _ = v.inputs(key); y = z['yaw']; t = z['times']; s = z['speed']
        folds = []
        for fold in range(5):
            q = np.load(source(bank, fold, key))
            assert np.array_equal(q['times'], t) and np.array_equal(q['speed'], s) and np.array_equal(q['v24_control'], y)
            folds.append(q['cotracker_multiview']-y)
        dy = np.array(folds)
        signals = dict(v24_control=y, median_folds=y+np.median(dy, axis=0), conservative_envelope=y+combine(dy))
        for name, yaw in signals.items():
            save_json(OUT/f'{bank}/predictions/{name}/{key}.json', dict(case['pred'], positions=v.core.integrate_velocity(t, yaw, s).tolist()))
        p = OUT/f'{bank}/signals/{key}.npz'; p.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(p, times=t, speed=s, fold_delta=dy, **signals)
        save_json(OUT/f'{bank}/diagnostics/{key}.json', dict(active_fraction=float(np.mean(combine(dy) != 0)), max_correction_deg=float(np.rad2deg(abs(combine(dy)).max()))))


def evaluate(bank):
    m.OUT = OUT; m.REP = REP; m.METHODS = METHODS; m.PRIMARY = PRIMARY
    m.evaluate(bank)


def validate():
    plan = json.loads((REP/'plan.json').read_text())
    for rel, sha in {**plan['dependencies'], **plan['preserved']}.items(): assert v.audit.sha256(ROOT/rel) == sha, rel
    samples = np.array([[1, 2, 3, 4, -9], [-1, -2, -3, -4, 9], [1, 2, 3, -4, -5], [0, 1, 2, 3, 4], [0, 0, 1, 2, 3]], float).T
    assert np.array_equal(combine(samples), [1, -1, 0, 1, 0])
    for bank in BANKS:
        for key in v.TRAIN:
            z = np.load(OUT/f'{bank}/signals/{key}.npz'); base = np.load(ROOT/f'runs/domain_v24_submission/train/signals/{key}.npz')
            assert np.array_equal(z['times'], base['times']) and np.array_equal(z['speed'], base['speed'])
            assert np.allclose(z[PRIMARY]-z['v24_control'], combine(z['fold_delta']), rtol=0, atol=1e-14)
            for name in METHODS:
                p = json.loads((OUT/f'{bank}/predictions/{name}/{key}.json').read_text())
                assert np.allclose(p['positions'], v.core.integrate_velocity(z['times'], z[name], z['speed']), rtol=0, atol=1e-9)
                assert abs(z[name]-z['v24_control']).max() <= np.deg2rad(3)+1e-9
    save_json(REP/'validation.json', dict(passed=True, predictions=144, preserved_submissions=len(plan['preserved']), speed_unchanged=True, no_test_inference=True,
        synthetic=['4of5positive selects conservative magnitude', '4of5negative selects conservative magnitude', 'Mixed signs fallback', 'One inactive fold allowed', 'Two inactive folds fallback']))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('action', choices=['freeze', 'extract', 'fold_infer', 'infer', 'evaluate', 'validate'])
    ap.add_argument('--bank', choices=BANKS, default='base'); ap.add_argument('--fold', type=int); args = ap.parse_args()
    if args.action == 'freeze': freeze()
    else:
        assert (REP/'plan.json').exists()
        if args.action == 'extract': extract_and_infer(args.bank, args.fold)
        elif args.action == 'fold_infer': fold_infer(args.bank, args.fold)
        elif args.action in ['infer', 'evaluate']: globals()[args.action](args.bank)
        else: validate()
