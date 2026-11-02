"""Validate and pack V64 deployment; never reads TEST reference trajectories."""
import copy,shutil,datetime
from v65_submission_inputs import *
DEST=ROOT/'submissions/macvi27_vo_v65_support_envelope'

def main():
    from public_dataset import validate_prediction
    from pack_submission import pack
    v.core.bg.guard();plan=json.loads((REP/'plan.json').read_text())
    assert json.loads((REP/'train_parity.json').read_text())['passed']
    original=json.loads(BASE.read_text());payload=copy.deepcopy(original);checks={}
    for key in test.KEYS:
        payload[key]=json.loads((OUT/f'test/predictions/{key}.json').read_text())
    assert set(payload)==set(test.CLIPS) and len(payload)==20
    for key,p in payload.items():
        ts=test.load_timestamps(test.DATA,test.CLIPS[key]);validate_prediction(p,key,ts)
        t=np.array(p['times']);pos=np.array(p['positions']);assert np.isfinite(pos).all()
        assert p['times']==original[key]['times'] and p['frame_indices']==original[key]['frame_indices']
        assert abs(t[0]-ts[0])<1e-6 and abs(t[-1]-ts[-1])<1e-6 and np.diff(t).max()<=1.000001
        if key not in test.KEYS:assert p==original[key]
        else:
            z=np.load(OUT/f'test/signals/{key}.npz');base=np.load(ROOT/f'runs/domain_v24_submission/test/signals/{key}.npz')
            assert np.array_equal(z['times'],base['times']) and np.array_equal(z['speed'],base['speed'])
            assert abs(z['yaw']-base['yaw']).max()<=np.deg2rad(3)+1e-9
            ds=z['variant_deltas'];lo=ds.min(0);hi=ds.max(0);env=np.where(lo>0,lo,np.where(hi<0,hi,0.))
            assert np.array_equal(env,z['envelope'])
            expected=base['yaw']+np.clip(env+(base['yaw']+np.median(z['fold_delta'],axis=0))-base['yaw'],-np.deg2rad(3),np.deg2rad(3))
            assert np.array_equal(expected,z['yaw'])
            assert np.allclose(pos,v.core.integrate_velocity(t,z['yaw'],z['speed']),atol=1e-9,rtol=0)
        save_json(OUT/f'final/predictions/{key}.json',p)
        checks[key]=dict(poses=len(t),max_gap_seconds=float(np.diff(t).max()),full_released_timestamp_extent=True,changed=p!=original[key])
    DEST.mkdir(parents=True,exist_ok=True);bundle=DEST/'submission.json'
    if bundle.exists():assert json.loads(bundle.read_text())==payload
    else:pack(test.MANIFEST,OUT/'final',bundle,'test')
    assert json.loads(bundle.read_text())==payload
    for rel,h in {**plan['dependencies'],**plan['preserved']}.items():assert v.audit.sha256(ROOT/rel)==h,rel
    evidence={}
    for folder in ['test/poses','test/course','test/signals','test/masks']:
        for p in (OUT/folder).rglob('*'):
            if p.is_file():evidence[str(p.relative_to(ROOT))]=v.audit.sha256(p)
    for key in test.KEYS:
        for p in (frame_root('test')/f'frames/{key}').glob('*.png'):evidence[str(p.relative_to(ROOT))]=v.audit.sha256(p)
        n=len(json.loads((frame_root('test')/f'frames/{key}/index.json').read_text())['times'])
        for i in range(n):
            p=mask_path(key,i,'test');evidence[str(p.relative_to(ROOT))]=v.audit.sha256(p)
    validation=dict(passed=True,official_packer_equal=True,reference_loading_blocked=True,scoring_blocked=True,checks=checks,preserved_submissions=len(plan['preserved']),train_parity=True,speed_unchanged=True,scope='Server determines official coverage and score.')
    provenance=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),candidate='V64 envelope_plus_course deployed asV65',train_mean=plan['train_mean'],official_score=None,
        baseline=str(BASE),baseline_sha256=v.audit.sha256(BASE),sha256=v.audit.sha256(bundle),bytes=bundle.stat().st_size,clips=20,
        changed_clips=[k for k,d in checks.items() if d['changed']],evidence_hashes=evidence,uploaded_to_drive=False)
    save_json(DEST/'validation.json',validation);save_json(REP/'validation.json',validation)
    save_json(DEST/'provenance.json',provenance)
    (DEST/'SHA256SUMS').write_text(provenance['sha256']+'  submission.json\n')
    description='Frozen pretrained DPVO, VGGT 1B and Pi3X headings are fused by median. Static background DPV-SLAM motion from three seeds and four time intervals provides a conservative course correction, combined with CoTracker three view geometry across five spatial splits and capped at 3 degrees. Camera specific speed estimates are retained. R3 supplies orientation initialization, AnyCalib estimates intrinsics and Mask2Former selects static features. Released calibration and timestamps and external pretrained weights are used. All settings are fixed on public train data.'
    allowed=set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 -:,.()[]_/\\+*%$§!?')
    assert set(description)<=allowed
    (DEST/'submission_notes.txt').write_text('Upload only submission.json to MaCVi.\nModel name: TideGraph-SE\nHardware: NVIDIA RTX 5080\nProcessing speed: End-to-end FPS not measured; do not invent a value.\nMetadata used: Yes. External datasets/pretrained data: Yes.\nDescription: '+description+'\nLocal TRAIN FLIR8:9.0283777922%. Official TEST score unknown. Keep the existing official13.19 submission.\n')
    code=DEST/'provenance_code';code.mkdir(exist_ok=True)
    for p in sorted((ROOT/'scripts').glob('v65_*.py')):shutil.copy2(p,code/p.name)
    shutil.copy2(REP/'plan.json',code/'plan.json');shutil.copy2(REP/'train_parity.json',DEST/'train_parity.json')
    print('SUBMISSION_READY',str(bundle),provenance['sha256'],provenance['bytes'],flush=True)

if __name__=='__main__':main()
