"""Single-change TEST probe: #19163 FLIR input416->512, same weights/A1/gain."""
import argparse,asyncio,json,re,time,subprocess
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter1d
import v101_protocol as p
import v81_omega_pipeline as omega
from pack_submission import pack
ROOT=p.ROOT;REP=ROOT/'reports/domain_v150_flir512';OUT=ROOT/'runs/domain_v150_flir512';DEST=ROOT/'submissions/macvi27_vo_v150_FLIR512_A1_gain110'
BASE=ROOT/'submissions/macvi27_vo_v129_TV10_FLIR_A1_gain110/submission.json';DATA=ROOT/'data/lookout-vo-test-v0.1'
CLIPS={c['key']:c for c in p.read(DATA/'release_manifest.json')['clips']};KEYS=[k for k,c in CLIPS.items() if c['camera_mode']=='flir-VIS']
def inputs(k):
    record=p.read(ROOT/f'reports/domain_v87_omega_flir_submission/inputs/{k}.json');index=ROOT/record['index'];assert p.sha(index)==record['index_sha256'];meta=p.read(index);return index.parent,meta,record
def yaw(t,raw,y0,gain=1.1):
    y=gaussian_filter1d(raw,2/np.median(np.diff(t)));y+=y0-y[0];return y0+gain*(y-y0)
def freeze():
    p.guard();assert not (REP/'plan.json').exists();assert p.sha(BASE)=='47ddcee1940fbd8fc5003ecd43e2a8af20b437f3269a3d8084ef26a908864bd3';assert len(KEYS)==11;assert all(c['reference'] is None for c in CLIPS.values());base=p.read(BASE);deps=[Path(__file__),BASE,DATA/'release_manifest.json',ROOT/'scripts/v101_protocol.py',ROOT/'scripts/v81_omega_pipeline.py',ROOT/'scripts/v143_omega_input.py',ROOT/'reports/domain_v143_omega_input/results.json'];parity=[]
    for k in KEYS:
        folder,meta,rec=inputs(k);assert meta['times']==base[k]['times'] and meta['frame_indices']==base[k]['frame_indices'];z=np.load(ROOT/f'runs/domain_v132_rebase_A1/test_signals/{k}.npz');t=z['times'];np.testing.assert_array_equal(t,base[k]['times']);y=yaw(t,z['raw_A1'],z['base'][0]);np.testing.assert_allclose(y,z['base'],atol=1e-10,rtol=0);err=float(np.max(abs(p.integrate(t,y,z['speed'])-base[k]['positions'])));assert err<1e-8;parity.append(dict(key=k,max_position_error=err,frames=len(t)));deps += [ROOT/f'reports/domain_v87_omega_flir_submission/inputs/{k}.json',folder/'index.json',ROOT/f'runs/domain_v132_rebase_A1/test_signals/{k}.npz']
    # Same graph+smoothing adapter reproduces completed development512 outputs.
    dev=[]
    for k in p.GROUPS['flir-VIS']:
        source=ROOT/'runs/domain_v143_omega_input/size512';z=np.load(source/f'signals/{k}.npz');raw,i,j,e,w=p.chain_and_edges(source,k,len(z['times']));a,d=p.solve_yaw(len(z['times']),i,j,e,raw);assert d['converged'];actual=yaw(z['times'],a,z['yaw'][0]);expected=z['yaw'][0]+1.1*(z['yaw']-z['yaw'][0]);err=float(np.max(abs(actual-expected)));assert err<1e-10;dev.append(dict(key=k,max_yaw_error_rad=err));deps.append(source/f'signals/{k}.npz')
    p.save(REP/'plan.json',dict(created_utc=p.now(),parent_entry=19163,parent_score=11.35,parent_sha256=p.sha(BASE),changed_keys=KEYS,unchanged_keys=[k for k in CLIPS if k not in KEYS],
        change='Only FLIR Omega preprocessing image_resolution416->512, official balanced. Same416-reproduction checkpoint,2Hz same exact original frame bytes,W16/O8,A1 graph,Huber3deg,Gaussian sigma2seconds,gain1.10,exact original speed. Wide8/IR1 copied.',
        model=p.read(omega.REP/'checkpoint_access.json'),development=p.read(ROOT/'reports/domain_v143_omega_input/results.json')['summary']['size512_gain1.1'],baseline_parity=parity,development_adapter_parity=dev,
        score_rule=dict(accept_at_or_below=11.33,reject_tie_band=[11.34,11.36],reject_at_or_above=11.37,FLIR_mean_delta_formula='(displayed overall drift -11.35)*20/11',on_accept='Adopt512 then separately probe gain1.20; do not combine before result.',on_reject='Retain19163; no further512 tuning.'),
        exclusions='No4Hz, focal regression, perclip gain, yaw4s, speed retraining or pitch-water modification. No challenge portal upload.',
        dependencies={str(f.relative_to(ROOT)):p.sha(f) for f in deps},preserved={str(f.relative_to(ROOT)):p.sha(f) for f in (ROOT/'submissions').rglob('submission.json')}))
    print('FROZEN',len(KEYS),'FLIR; unchanged9; parity passed',flush=True)
def verify():
    plan=p.read(REP/'plan.json')
    for f,h in plan['dependencies'].items():assert p.sha(ROOT/f)==h,f
    return plan
def infer():
    p.guard();plan=verify();torch,model,load,decode=omega.model_and_io();p.save(REP/'environment.json',dict(torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),repo_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=omega.REPO,text=True).strip(),same_model_weights=plan['model']['sha256']))
    for number,k in enumerate(KEYS,1):
        final=OUT/f'signals/{k}.npz'
        if final.exists():continue
        folder,meta,record=inputs(k);n=len(meta['times']);beg=time.monotonic()
        for i in range(n):
            f=folder/f'{i:04d}.png';assert p.sha(f)==record['frame_sha256'][f.name],f
        torch.cuda.reset_peak_memory_stats();shape=None
        for start in range(0,n,8):
            end=min(n,start+16);dest=OUT/f'omega/windows/{k}/{start:06d}.npz'
            if not dest.exists():
                images=load([str(folder/f'{i:04d}.png') for i in range(start,end)],image_resolution=512,mode='balanced');shape=list(images.shape[-2:])
                with torch.inference_mode():out=model(images.cuda());ex,ki=decode(out['pose_enc'].float(),images.shape[-2:])
                ex=ex[0].cpu().numpy();K=ki[0].cpu().numpy();c2w=np.tile(np.eye(4),(len(ex),1,1));c2w[:,:3,:3]=ex[:,:,:3].transpose(0,2,1);c2w[:,:3,3]=-np.einsum('nij,nj->ni',c2w[:,:3,:3],ex[:,:,3]);assert np.isfinite(c2w).all() and np.isfinite(K).all();assert np.max(abs(np.linalg.det(c2w[:,:3,:3])-1))<1e-3;p.npzsave(dest,start=np.array(start),c2w=c2w,intrinsics=K,input_hw=np.array(shape));del out,ex,ki,images
            else:
                with np.load(dest) as z:shape=z['input_hw'].tolist();assert int(z['start'])==start
            if start%160==0:print('WINDOW',number,11,k,start,end,n,flush=True)
            if end==n:break
        z=np.load(ROOT/f'runs/domain_v132_rebase_A1/test_signals/{k}.npz');t=z['times'];raw,i,j,e,w=p.chain_and_edges(OUT/'omega',k,n);graph,d=p.solve_yaw(n,i,j,e,raw);assert d['converged'];yy=yaw(t,graph,z['base'][0]);positions=p.integrate(t,yy,z['speed']);assert np.isfinite(positions).all();p.npzsave(final,times=t,raw_A1=graph,yaw=yy,speed=z['speed']);delta=np.rad2deg(yy-z['base']);p.save(OUT/f'diagnostics/{k}.json',dict(key=k,frames=n,image_resolution=512,input_hw=shape,window=16,overlap=8,seconds=time.monotonic()-beg,peak_cuda_GiB=torch.cuda.max_memory_allocated()/2**30,graph=d,yaw_change_p95_deg=float(np.quantile(abs(delta),.95)),yaw_change_max_deg=float(np.max(abs(delta))),speed_exactly_retained=True,GT_access_blocked=True,frame_bytes_verified=True));print('COMPLETE_CLIP',number,11,k,round(time.monotonic()-beg,1),flush=True)
        p.save(REP/'status.json',dict(stage='inference',completed=number,total=11,last_key=k,updated_utc=p.now()))
    p.save(REP/'signals_frozen.json',dict(files={str((OUT/f'signals/{k}.npz').relative_to(ROOT)):p.sha(OUT/f'signals/{k}.npz') for k in KEYS},GT_access_blocked=True))
def build():
    p.guard();plan=verify();base=p.read(BASE);bundle={k:dict(v) for k,v in base.items()};checks=[]
    for f,h in p.read(REP/'signals_frozen.json')['files'].items():assert p.sha(ROOT/f)==h,f
    for k in KEYS:
        z=np.load(OUT/f'signals/{k}.npz');old=np.load(ROOT/f'runs/domain_v132_rebase_A1/test_signals/{k}.npz');assert np.array_equal(z['times'],old['times']) and np.array_equal(z['speed'],old['speed']);bundle[k]['positions']=p.integrate(z['times'],z['yaw'],z['speed']).tolist()
    changed=[k for k in bundle if bundle[k]!=base[k]];assert set(changed)==set(KEYS)
    for k,pred in bundle.items():
        ts=p.load_timestamps(DATA,CLIPS[k]);p.validate_prediction(pred,k,ts);t=np.asarray(pred['times']);coverage=p.coverage(t,ts);gap=float(np.diff(t).max());assert coverage>=.95 and gap<=1
        assert all(pred[f]==base[k][f] for f in ['key','frame_indices','times'])
        if k not in KEYS:assert pred==base[k]
        p.save(OUT/f'predictions/{k}.json',pred);checks.append(dict(key=k,changed=k in KEYS,camera_mode=CLIPS[k]['camera_mode'],frames=len(t),released_timestamp_coverage=coverage,max_gap_s=gap))
    for f,h in plan['preserved'].items():assert p.sha(ROOT/f)==h,f
    DEST.mkdir(parents=True,exist_ok=True);dest=DEST/'submission.json'
    if not dest.exists():pack(DATA/'release_manifest.json',OUT/'predictions',dest)
    packed=p.read(dest);assert packed==bundle
    description='Monocular maritime visual odometry using video, timestamps and calibration. Frozen VGGT Omega 416 reproduction weights use balanced input resolution 512 for FLIR visible clips and 416 for wide clips, with 16 frame windows and 8 frame overlap. FLIR rotations use robust all pairs within window yaw graph optimization, Gaussian smoothing with sigma 2 seconds and accumulated yaw gain 1.10. Wide speed uses total variation filtering with lambda 10. Existing FLIR speed and infrared processing are retained. External pretrained weights and a public TRAIN fitted speed prior are used. Settings are selected using public development evaluation and aggregate leaderboard feedback. No VAL or TEST video training.'
    assert re.fullmatch(r'[A-Za-z0-9 \-:,\.()\[\]_/\\+*%$§!?]+',description);(DEST/'description.txt').write_text(description+'\n')
    record=dict(version=150,parent_entry=19163,parent_score=11.35,parent_sha256=p.sha(BASE),sha256=p.sha(dest),bytes=dest.stat().st_size,changed_keys=KEYS,unchanged_keys=plan['unchanged_keys'],configuration=dict(input_resolution=512,FLIR_gain=1.1,sigma_seconds=2,graph='A1_all_pairs',sample_rate='unchanged2Hz'),metadata_used=True,external_pretrained_data_used=True,model_name='TideGraph-512',hardware='NVIDIA RTX 5080',development=plan['development'],score_rule=plan['score_rule'],TEST_score=None,challenge_uploaded=False)
    p.save(DEST/'provenance.json',record);p.save(DEST/'validation.json',dict(passed=True,clip_count=20,changed_count=11,unchanged_count=9,checks=checks,GT_access_blocked=True,note='Released-timestamp support only; server reference scoreability is not locally observable.'));p.save(REP/'delivery.json',record);p.save(REP/'status.json',dict(stage='validated',updated_utc=p.now()));print('SUBMISSION_READY',str(dest),record['sha256'],flush=True)
async def upload():
    from v73_upload_drive import upload as send
    assert p.read(DEST/'validation.json')['passed'];dest=DEST/'submission.json';rec=await send(dest,'TideGraph_V150_FLIR512_'+p.sha(dest)[:10]+'_submission.json');p.save(REP/'drive_submission.json',rec);p.save(DEST/'drive.json',rec);p.save(REP/'status.json',dict(stage='uploaded_to_drive',url=rec['url'],updated_utc=p.now()));print('DRIVE_READY',rec['url'],flush=True)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['freeze','infer','build','upload']);args=ap.parse_args();asyncio.run(upload()) if args.action=='upload' else globals()[args.action]()
